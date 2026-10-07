#!/usr/bin/env python3
"""RunPod Serverless worker for bounded LoRA training / evaluation jobs (queue endpoint).

Not an inference server. One job = one experiment step on one GPU:

  smoke_test  host checks only (GPU visible, free VRAM, CUDA alloc + tiny bf16 forward/backward, pinned imports,
              network-volume write/fsync/read-back, free disk). With a bundle it also verifies every bundle file by
              sha256 and runs the bundle's host preflight script, or the trainer with --check-only. Never trains.
  train       verifies the bundle, runs host checks, then RUNS THE BUNDLE'S OWN TRAINER (delivered by sha256 on the
              network volume; no training logic lives here) in a job directory ON the network volume, so checkpoints
              land on persistent storage as they are written.
  evaluate    verifies the bundle (+ optional checkpoints of an earlier train run, sha-checked against its manifest),
              runs host checks, then runs the bundle's own evaluator.

Results: every job ends with a manifest written atomically (tmp + fsync + rename + dir fsync) to
<volume>/<output_dir>/MANIFEST.json BEFORE the handler returns. The returned JSON is compact:
  status COMPLETED        the step ran and every supplied gate passed
  status SCIENTIFIC_FAIL  the step ran, outputs are persisted, but a gate failed or training stopped early
  status INFRA_FAILURE    the step could not run or could not be persisted; "stage" says where, "retryable" says
                          whether another worker/attempt could succeed (bad inputs are never retryable)
Jobs are idempotent by experiment_id: the same input returns the persisted result; an interrupted step is redone
(or resumed when the job names the trainer's resume flag); the same experiment_id with different input is refused.

Secrets come only from the environment (RunPod endpoint secrets); none are passed to the bundle's scripts except
HF_TOKEN (model download). Configuration: see README.md, section "RunPod Serverless worker".
"""
import hashlib
import json
import os
import pathlib
import re
import shutil
import signal
import socket
import subprocess
import time
import traceback
import urllib.request
import uuid

import runpod

HANDLER_VERSION = "1.0.0"
MANIFEST_SCHEMA = "serverless-run-manifest/1"
MODES = ("smoke_test", "train", "evaluate")
DEFAULT_PINS = {"torch": "2.10.0", "transformers": "5.3.0", "peft": "0.18.1", "accelerate": "1.12.0"}

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
RELPATH_RE = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+){0,5}$")
ARG_RE = re.compile(r"^[A-Za-z0-9_.=:/,+{}@-]{0,256}$")
GATE_OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
            "<": lambda a, b: a < b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}

ALLOWED_KEYS = {"mode", "experiment_id", "bundle", "recipe", "output_dir", "trainer", "trainer_args", "resume_arg",
                "evaluator", "evaluator_args", "preflight_script", "checkpoints_from", "gates", "min_free_vram_gb",
                "min_free_disk_gb", "max_runtime_s", "train_timeout_s", "eval_timeout_s", "note"}
BUNDLE_KEYS = {"dir", "files", "source"}
# Files the handler itself writes into a job directory (never reported as trainer/evaluator outputs).
HANDLER_FILES = {"BUNDLE.SHA256SUMS", ".host_preflight.bin"}
TRAIN_PARTIAL = ("adapter", "ckpt", "out", "RECEIPT.json", "TRAIN_LOG.jsonl")


def env(name, default=None):
    v = os.environ.get(name)
    return default if v in (None, "") else v


def volume_root():
    return pathlib.Path(env("PLLM_VOLUME_ROOT", "/runpod-volume"))


def train_python():
    return env("PLLM_TRAIN_PYTHON", "/opt/phone-llm/venv/bin/python")


def runtime_cap_s():
    return int(env("PLLM_MAX_RUNTIME_S", "13500"))


class Fail(Exception):
    """INFRA_FAILURE at a named stage."""

    def __init__(self, stage, reason, retryable=False, detail=None):
        super().__init__(f"{stage}: {reason}")
        self.stage, self.reason, self.retryable, self.detail = stage, reason, retryable, detail


def log(*a):
    print("[worker]", *a, flush=True)


# ----------------------------------------------------------------------------------------------- hashing / atomic IO
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _fsync_dir(d):
    try:
        fd = os.open(str(d), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def write_atomic(path, data):
    """tmp file in the same directory + fsync + rename + directory fsync."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def write_manifest_atomic(path, manifest):
    data = (json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n").encode()
    write_atomic(path, data)
    back = json.loads(pathlib.Path(path).read_text())
    if back.get("status") != manifest.get("status") or back.get("fingerprint") != manifest.get("fingerprint"):
        raise OSError("manifest read-back mismatch")
    return hashlib.sha256(data).hexdigest()


def copy_verified(src, dst, want_sha):
    dst = pathlib.Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp-{uuid.uuid4().hex[:8]}")
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
        fo.flush()
        os.fsync(fo.fileno())
    got = sha256_file(tmp)
    if got != want_sha:
        tmp.unlink(missing_ok=True)
        raise Fail("inputs", f"sha256 mismatch after copy: {dst.name}", False, {"expected": want_sha, "got": got})
    os.replace(tmp, dst)


def fsync_tree(root, rels):
    for rel in rels:
        try:
            with open(root / rel, "rb") as fh:
                os.fsync(fh.fileno())
        except OSError as e:
            raise Fail("persist", f"fsync failed for {rel}: {e!r}", True)
    _fsync_dir(root)


# ----------------------------------------------------------------------------------------------- validation
def _relpath(value, what, prefix=None):
    if not isinstance(value, str) or not RELPATH_RE.match(value) or any(part in (".", "..") for part in value.split("/")):
        raise Fail("validation", f"{what} must be a relative path without '..' (got {value!r})")
    if prefix and not value.startswith(prefix):
        raise Fail("validation", f"{what} must start with {prefix!r}")
    return value


def _number(v, what, lo, hi):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
        raise Fail("validation", f"{what} must be a number in [{lo}, {hi}]")
    return v


def _args(v, what):
    if not isinstance(v, list) or not all(isinstance(x, str) and ARG_RE.match(x) for x in v) or len(v) > 32:
        raise Fail("validation", f"{what} must be a list of at most 32 plain argument strings")
    return list(v)


def validate(inp):
    """Strict, fail-closed validation. Returns a normalised job spec."""
    if not isinstance(inp, dict):
        raise Fail("validation", "job input must be a JSON object")
    unknown = sorted(set(inp) - ALLOWED_KEYS)
    if unknown:
        raise Fail("validation", f"unknown field(s): {unknown}")
    mode = inp.get("mode")
    if mode not in MODES:
        raise Fail("validation", f"mode must be one of {list(MODES)}")
    eid = inp.get("experiment_id")
    if not isinstance(eid, str) or not ID_RE.match(eid):
        raise Fail("validation", "experiment_id must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    spec = {"mode": mode, "experiment_id": eid, "files": {}, "bundle_dir": None, "source": None}

    bundle = inp.get("bundle")
    if bundle is None and mode in ("train", "evaluate"):
        raise Fail("validation", f"bundle is required for mode {mode}")
    if bundle is not None:
        if not isinstance(bundle, dict) or set(bundle) - BUNDLE_KEYS:
            raise Fail("validation", f"bundle must be an object with keys {sorted(BUNDLE_KEYS)}")
        files = bundle.get("files")
        if not isinstance(files, dict) or not files or len(files) > 256:
            raise Fail("validation", "bundle.files must map 1..256 relative paths to sha256 digests")
        for rel, sha in files.items():
            _relpath(rel, f"bundle file {rel!r}")
            if rel in HANDLER_FILES and rel != "BUNDLE.SHA256SUMS":
                raise Fail("validation", f"bundle file name {rel!r} is reserved")
            if not isinstance(sha, str) or not SHA_RE.match(sha):
                raise Fail("validation", f"bundle file {rel!r} needs a lowercase hex sha256")
        spec["files"] = dict(files)
        spec["bundle_dir"] = _relpath(bundle.get("dir", f"bundles/{eid}"), "bundle.dir", "bundles/")
        src = bundle.get("source")
        if src is not None:
            if not isinstance(src, dict) or src.get("type") not in ("https", "s3"):
                raise Fail("validation", "bundle.source must be {type: https|s3, ...}")
            if src["type"] == "https" and not (isinstance(src.get("base_url"), str) and src["base_url"].startswith("https://")):
                raise Fail("validation", "bundle.source.base_url must be an https:// URL")
            if src["type"] == "s3" and not (isinstance(src.get("bucket"), str) and src["bucket"]):
                raise Fail("validation", "bundle.source.bucket is required for s3")
            spec["source"] = src

    out = inp.get("output_dir")
    if out is None and mode in ("train", "evaluate"):
        raise Fail("validation", f"output_dir is required for mode {mode}")
    spec["output_dir"] = _relpath(out if out is not None else f"runs/{eid}", "output_dir", "runs/")

    recipe = inp.get("recipe")
    if mode in ("train", "evaluate") and recipe is None:
        raise Fail("validation", "recipe {file, sha256} naming the approved config inside the bundle is required")
    if recipe is not None:
        if not isinstance(recipe, dict) or set(recipe) != {"file", "sha256"}:
            raise Fail("validation", "recipe must be {file, sha256}")
        if spec["files"].get(recipe["file"]) != recipe["sha256"]:
            raise Fail("validation", "recipe file/sha256 is not in bundle.files with the same digest")
    spec["recipe"] = recipe

    spec["trainer"] = inp.get("trainer", "train_lora_v2.py")
    if not isinstance(spec["trainer"], str) or "/" in spec["trainer"]:
        raise Fail("validation", "trainer must be a top-level bundle file name")
    _relpath(spec["trainer"], "trainer")
    if mode == "train" and spec["trainer"] not in spec["files"]:
        raise Fail("validation", f"trainer {spec['trainer']!r} must be listed in bundle.files")
    spec["trainer_args"] = _args(inp.get("trainer_args", []), "trainer_args")
    ra = inp.get("resume_arg")
    if ra is not None and not (isinstance(ra, str) and re.match(r"^--[a-z0-9][a-z0-9-]{0,63}$", ra)):
        raise Fail("validation", "resume_arg must look like --flag-name")
    spec["resume_arg"] = ra

    spec["evaluator"] = inp.get("evaluator", "pod_eval/pod_eval.py")
    _relpath(spec["evaluator"], "evaluator")
    if mode == "evaluate" and spec["evaluator"] not in spec["files"]:
        raise Fail("validation", f"evaluator {spec['evaluator']!r} must be listed in bundle.files")
    spec["evaluator_args"] = _args(inp.get("evaluator_args", ["--job", "{job}"]), "evaluator_args")

    spec["preflight_script"] = inp.get("preflight_script", "host_preflight.py")
    _relpath(spec["preflight_script"], "preflight_script")

    cf = inp.get("checkpoints_from")
    if cf is not None:
        if mode != "evaluate" or not isinstance(cf, dict) or set(cf) != {"experiment_id"} or not ID_RE.match(str(cf["experiment_id"])):
            raise Fail("validation", "checkpoints_from must be {experiment_id} and is only valid for mode evaluate")
    spec["checkpoints_from"] = cf

    gates = inp.get("gates", [])
    if not isinstance(gates, list) or len(gates) > 64:
        raise Fail("validation", "gates must be a list")
    for g in gates:
        if not isinstance(g, dict) or set(g) != {"metric", "op", "value"} or g["op"] not in GATE_OPS \
                or not isinstance(g["metric"], str) or isinstance(g["value"], bool) or not isinstance(g["value"], (int, float)):
            raise Fail("validation", "each gate must be {metric: str, op: one of >= <= > < == !=, value: number}")
    spec["gates"] = gates

    spec["min_free_vram_gb"] = _number(inp.get("min_free_vram_gb", 20), "min_free_vram_gb", 1, 200)
    spec["min_free_disk_gb"] = _number(inp.get("min_free_disk_gb", 10 if mode == "train" else 1), "min_free_disk_gb", 0, 10000)
    spec["max_runtime_s"] = min(_number(inp.get("max_runtime_s", runtime_cap_s()), "max_runtime_s", 60, 86400), runtime_cap_s())
    spec["train_timeout_s"] = _number(inp.get("train_timeout_s", 10800), "train_timeout_s", 60, 86400)
    spec["eval_timeout_s"] = _number(inp.get("eval_timeout_s", 5400), "eval_timeout_s", 60, 86400)
    if "note" in inp and not (isinstance(inp["note"], str) and len(inp["note"]) <= 500):
        raise Fail("validation", "note must be a string of at most 500 characters")
    spec["fingerprint"] = hashlib.sha256(json.dumps(inp, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return spec


# ----------------------------------------------------------------------------------------------- subprocesses
def child_env(extra=None):
    """Least-privilege environment for bundle scripts: no RunPod/S3/fetch credentials; HF_TOKEN only if set."""
    tp_bin = str(pathlib.Path(train_python()).parent)
    keep = ("HOME", "LANG", "LC_ALL", "TZ", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES", "CUDA_HOME", "TMPDIR")
    e = {k: os.environ[k] for k in keep if k in os.environ}
    e.update({"PATH": tp_bin + ":" + os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
              "HF_HOME": env("HF_HOME", str(volume_root() / "hf")), "HF_HUB_ENABLE_HF_TRANSFER": "0",
              "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "PIP_NO_INPUT": "1"})
    if env("HF_TOKEN"):
        e["HF_TOKEN"] = os.environ["HF_TOKEN"]
    if env("PLLM_HF_OFFLINE") == "1":
        e.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    e.update(extra or {})
    return e


def run_proc(cmd, cwd, log_path, timeout, grace_s=120):
    """Run cmd, append stdout+stderr to log_path. On timeout: SIGTERM to the process group (the trainer saves a
    partial adapter + receipt on SIGTERM), then SIGKILL after grace_s. Returns (rc, timed_out, tail)."""
    log_path = pathlib.Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    with open(log_path, "ab") as lf:
        lf.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} $ {' '.join(map(str, cmd))}\n".encode())
        lf.flush()
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), stdout=lf, stderr=subprocess.STDOUT,
                             env=child_env(), start_new_session=True)
        try:
            rc = p.wait(timeout=max(1, int(timeout)))
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                rc = p.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                rc = p.wait()
        lf.flush()
        os.fsync(lf.fileno())
    return rc, timed_out, tail(log_path)


def tail(path, n=1500):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - n))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


# ----------------------------------------------------------------------------------------------- host checks
TENSOR_CHECK = ("import torch;assert torch.cuda.is_available(),'no cuda';x=torch.empty(1<<28,dtype=torch.float32,device='cuda');x.fill_(1.0);"
                "torch.cuda.synchronize();assert float(x[:16].sum())==16.0;del x;"
                "m=torch.nn.Sequential(torch.nn.Linear(256,256),torch.nn.GELU(),torch.nn.Linear(256,8)).cuda().to(torch.bfloat16);"
                "i=torch.randn(32,256,device='cuda',dtype=torch.bfloat16);l=m(i).float().pow(2).mean();l.backward();"
                "g=sum(float(p.grad.abs().sum()) for p in m.parameters());assert torch.isfinite(l) and g>0;"
                "print('CUDA_OK',torch.cuda.get_device_name(0))")


def pins_check_code(names):
    return ("import json,importlib.metadata as m;print('PINS '+json.dumps({k:m.version(k).split('+')[0] for k in %r}))" % (sorted(names),))


def host_checks(spec, ctx, workdir, expected_pins):
    """Generic GPU/CUDA/VRAM/import/storage checks. Raises Fail('preflight', ...) on the first failure."""
    checks = []
    logp = ctx["run_dir"] / "logs" / f"host_checks-{ctx['attempt_tag']}.log"

    def record(name, ok, detail, retryable=True):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        ctx["preflight"] = checks
        if not ok:
            raise Fail("preflight", f"host check failed: {name}", retryable, {"check": name, "detail": detail})

    rc, to, out = run_proc(["nvidia-smi", "-L"], workdir, logp, 30)
    record("gpu_visible", rc == 0 and "GPU" in out, out.strip()[-200:])
    rc, to, out = run_proc(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"], workdir, logp, 30)
    try:
        line = [ln for ln in out.strip().splitlines() if ln.count(",") == 2][-1]
        name, tot, used = [x.strip() for x in line.split(",")]
        free = (float(tot) - float(used)) / 1024
        ctx["gpu"] = {"name": name, "total_gb": round(float(tot) / 1024, 1), "free_gb": round(free, 1)}
        ok_vram = free >= spec["min_free_vram_gb"]
        vram_detail = {"gpu": name, "free_gb": round(free, 1), "min_gb": spec["min_free_vram_gb"]}
    except (IndexError, ValueError) as e:
        ok_vram, vram_detail = False, f"unparseable nvidia-smi output: {out[-160:]!r} {e!r}"
    record("vram_free", ok_vram, vram_detail)
    try:
        p = ctx["run_dir"] / f".fscheck-{uuid.uuid4().hex[:8]}"
        blob = os.urandom(64 << 20)
        with open(p, "wb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        ok = sha256_file(p) == hashlib.sha256(blob).hexdigest()
        p.unlink()
        fs_detail = "64 MiB write/fsync/read-back on the network volume"
    except OSError as e:
        ok, fs_detail = False, repr(e)
    record("volume_writable", ok, fs_detail)
    free_disk = shutil.disk_usage(volume_root()).free / 2 ** 30
    record("volume_free_disk", free_disk >= spec["min_free_disk_gb"], {"free_gb": round(free_disk, 1), "min_gb": spec["min_free_disk_gb"]})
    rc, to, out = run_proc([train_python(), "-c", pins_check_code(expected_pins)], workdir, logp, 180)
    try:
        got = json.loads([ln for ln in out.splitlines() if ln.startswith("PINS ")][-1][5:])
    except (IndexError, ValueError):
        got = None
    record("pinned_imports", got == expected_pins, {"expected": expected_pins, "observed": got if got is not None else out[-200:]},
           retryable=False)
    rc, to, out = run_proc([train_python(), "-c", TENSOR_CHECK], workdir, logp, 240)
    record("cuda_alloc_tiny_model", rc == 0 and "CUDA_OK" in out, out.strip()[-200:])
    return checks


def bundle_preflight(spec, ctx, job_dir):
    """Bundle-supplied host preflight script (if listed), else the trainer's own --check-only (if listed)."""
    logp = ctx["run_dir"] / "logs" / f"bundle_preflight-{ctx['attempt_tag']}.log"
    ps = spec["preflight_script"]
    if ps in spec["files"]:
        sums = job_dir / "BUNDLE.SHA256SUMS"
        if "BUNDLE.SHA256SUMS" not in spec["files"]:
            top = {r: s for r, s in spec["files"].items() if "/" not in r and r != ps}
            write_atomic(sums, "".join(f"{s}  {r}\n" for r, s in sorted(top.items())).encode())
        rc, to, out = run_proc([train_python(), job_dir / ps, "--job", job_dir, "--min-free-vram-gb", str(spec["min_free_vram_gb"]),
                                "--pins", ""], job_dir, logp, 1500)
        line = next((ln for ln in out.splitlines() if ln.startswith("HOST_PREFLIGHT ")), None)
        ok = rc == 0 and bool(line) and '"result": "PASS"' in line
        ctx.setdefault("preflight", []).append({"check": "bundle_preflight_script", "pass": ok, "detail": (line or out[-200:])})
        if not ok:
            raise Fail("preflight", "bundle host preflight script failed", True, {"line": line, "tail": out[-400:]})
        return
    if spec["trainer"] in spec["files"]:
        rc, to, out = run_proc([train_python(), job_dir / spec["trainer"], "--check-only"], job_dir, logp, 900)
        ok = rc == 0 and "CHECK OK" in out
        ctx.setdefault("preflight", []).append({"check": "trainer_check_only", "pass": ok, "detail": out.strip()[-200:]})
        if not ok:
            # the check renders every data row; a deterministic data/template failure is not fixed by another worker
            raise Fail("preflight", "trainer --check-only failed", bool(to), {"rc": rc, "tail": out[-400:]})


# ----------------------------------------------------------------------------------------------- inputs
def _https_get(url, dst):
    req = urllib.request.Request(url)
    tok = env("PLLM_FETCH_BEARER_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 20)


def _s3_get(src, rel, dst):
    import boto3  # credentials only from the standard AWS_* environment variables / RunPod secrets
    key = "/".join(x for x in (src.get("prefix", "").strip("/"), rel) if x)
    boto3.client("s3", endpoint_url=env("AWS_ENDPOINT_URL")).download_file(src["bucket"], key, str(dst))


def fetch_missing(spec, bdir):
    src = spec["source"]
    for rel, want in spec["files"].items():
        p = bdir / rel
        if p.exists():
            continue
        if not src:
            raise Fail("inputs", f"bundle file missing on the volume: {rel}", False)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.fetch-{uuid.uuid4().hex[:8]}")
        try:
            if src["type"] == "https":
                _https_get(src["base_url"].rstrip("/") + "/" + rel, tmp)
            else:
                _s3_get(src, rel, tmp)
        except Exception as e:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            raise Fail("inputs", f"fetch failed for {rel}: {type(e).__name__}", True)
        got = sha256_file(tmp)
        if got != want:
            tmp.unlink(missing_ok=True)
            raise Fail("inputs", f"sha256 mismatch for fetched {rel}", False, {"expected": want, "got": got})
        os.replace(tmp, p)


def stage_inputs(spec, job_dir):
    """sha256-verify every bundle file on the volume, cross-check the recipe's own data digests, copy into job_dir."""
    bdir = volume_root() / spec["bundle_dir"]
    fetch_missing(spec, bdir)
    for rel, want in sorted(spec["files"].items()):
        got = sha256_file(bdir / rel)
        if got != want:
            raise Fail("inputs", f"sha256 mismatch: {rel}", False, {"expected": want, "got": got})
    if spec["recipe"]:
        try:
            card = json.loads((bdir / spec["recipe"]["file"]).read_text())
        except ValueError:
            raise Fail("inputs", "recipe file is not valid JSON", False)
        for rel, want in (card.get("data_sha256") or {}).items():
            if spec["files"].get(rel) != want:
                raise Fail("inputs", f"recipe data_sha256[{rel}] does not match bundle.files", False)
    copied = 0
    for rel, want in sorted(spec["files"].items()):
        dst = job_dir / rel
        if dst.exists() and sha256_file(dst) == want:
            continue
        copy_verified(bdir / rel, dst, want)
        copied += 1
    return copied


def expected_pins(spec, job_dir):
    if spec["recipe"]:
        try:
            pins = json.loads((job_dir / spec["recipe"]["file"]).read_text()).get("pins")
            if isinstance(pins, dict) and pins and all(isinstance(v, str) for v in pins.values()):
                return dict(pins)
        except ValueError:
            pass
    return dict(DEFAULT_PINS)


# ----------------------------------------------------------------------------------------------- outputs / metrics
def list_outputs(job_dir, spec):
    out = {}
    inputs = set(spec["files"])
    for p in sorted(job_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(job_dir).as_posix()
        if rel in inputs or rel in HANDLER_FILES or "__pycache__" in rel or p.name.startswith(".") \
                or rel.startswith(("init_adapter/", "screen_core/")):
            continue
        out[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def verify_outputs(job_dir, outputs):
    for rel, meta in outputs.items():
        p = job_dir / rel
        if not p.is_file() or p.stat().st_size != meta["bytes"] or sha256_file(p) != meta["sha256"]:
            return False
    return True


def train_metrics(job_dir):
    rec_p = job_dir / "RECEIPT.json"
    if not rec_p.exists():
        return None
    rec = json.loads(rec_p.read_text())
    m = {"steps_done": rec.get("steps_done"), "max_steps": rec.get("max_steps"), "train_wall_s": rec.get("train_wall_s")}
    if isinstance(m["steps_done"], int) and isinstance(m["max_steps"], int) and m["max_steps"] > 0:
        m["steps_done_ratio"] = round(m["steps_done"] / m["max_steps"], 6)
    tl = job_dir / "TRAIN_LOG.jsonl"
    losses, evals = [], []
    if tl.exists():
        for line in tl.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row.get("loss"), (int, float)):
                losses.append(row["loss"])
            if isinstance(row.get("eval_loss"), (int, float)):
                evals.append(row["eval_loss"])
    if losses:
        m["final_train_loss"] = losses[-1]
    if evals:
        m["final_eval_loss"], m["min_eval_loss"] = evals[-1], min(evals)
    m["partial"] = bool(rec.get("partial"))
    return {k: v for k, v in m.items() if v is not None}, rec


def extra_metrics(job_dir):
    m = {}
    for p in (job_dir / "METRICS.json", job_dir / "evals" / "METRICS.json"):
        if p.exists():
            try:
                for k, v in json.loads(p.read_text()).items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        m[str(k)] = v
            except (ValueError, AttributeError):
                raise Fail("evaluate", f"{p.name} is not a flat JSON object", False)
    return m


def apply_gates(gates, metrics):
    res = []
    for g in gates:
        if g["metric"] not in metrics:
            raise Fail("gates", f"gate metric not produced: {g['metric']}", False, {"available": sorted(metrics)})
        obs = metrics[g["metric"]]
        res.append({**g, "observed": obs, "pass": bool(GATE_OPS[g["op"]](obs, g["value"]))})
    return res


# ----------------------------------------------------------------------------------------------- lock
def worker_id():
    return env("RUNPOD_POD_ID", socket.gethostname())


def acquire_lock(run_dir, stale_s):
    lock = run_dir / ".lock"
    run_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w") as fh:
                json.dump({"worker": worker_id(), "t": time.time()}, fh)
                fh.flush()
                os.fsync(fh.fileno())
            return lock
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue
            if age < stale_s:
                raise Fail("lock", "another worker holds this experiment_id", True, {"lock_age_s": int(age)})
            lock.unlink(missing_ok=True)
    raise Fail("lock", "could not acquire experiment lock", True)


# ----------------------------------------------------------------------------------------------- steps
def remaining(ctx, reserve=600):
    return int(ctx["deadline"] - time.time() - reserve)


def archive_partial(job_dir, run_dir, attempt):
    dst = run_dir / "attempts" / f"{attempt:03d}"
    moved = []
    for name in TRAIN_PARTIAL:
        if (job_dir / name).exists():
            dst.mkdir(parents=True, exist_ok=True)
            os.replace(job_dir / name, dst / name)
            moved.append(name)
    return moved


def latest_ckpt(job_dir):
    ck = job_dir / "ckpt"
    steps = sorted(p for p in ck.iterdir() if p.is_dir() and p.name.startswith("step-")) if ck.exists() else []
    return steps[-1] if steps else None


def step_train(spec, ctx, man, job_dir):
    st = man["steps"].get("train", {})
    if st.get("status") == "DONE" and verify_outputs(job_dir, man.get("outputs", {})):
        ctx["resumed"] = "resumed_steps"
        return st["metrics"], st.get("scientific")
    attempt = int(st.get("attempt", 0)) + 1
    cmd = [train_python(), "-u", job_dir / spec["trainer"], *spec["trainer_args"]]
    ck = latest_ckpt(job_dir)
    resumed_from = None
    if st.get("status") in ("RUNNING", "FAILED", "DONE") and ck is not None and spec["resume_arg"]:
        cmd += [spec["resume_arg"], ck]
        resumed_from = ck.relative_to(job_dir).as_posix()
        ctx["resumed"] = "resumed_from_checkpoint"
    elif st.get("status") in ("RUNNING", "FAILED", "DONE"):
        moved = archive_partial(job_dir, ctx["run_dir"], attempt - 1)
        man.setdefault("restarts", []).append({"attempt": attempt, "archived": moved,
                                               "why": "no resume flag in this job (trainer has no resume support): restarted from step 0"})
        ctx["resumed"] = "restarted"
    timeout = min(spec["train_timeout_s"], remaining(ctx))
    if timeout < 300:
        raise Fail("deadline", "not enough runtime left to train", True, {"left_s": remaining(ctx)})
    man["steps"]["train"] = {"status": "RUNNING", "attempt": attempt, "t_start": time.time(), "worker": worker_id(),
                             "timeout_s": timeout, "resumed_from": resumed_from}
    persist(ctx, man)
    progress(ctx, f"train attempt {attempt} started")
    rc, to, out = run_proc(cmd, job_dir, ctx["run_dir"] / "logs" / f"train-{attempt:03d}.log", timeout)
    tr = man["steps"]["train"]
    tr.update(rc=rc, timed_out=to, t_end=time.time(), status="FAILED")
    # whatever the outcome, record what the trainer left on the volume before deciding
    man["outputs"] = list_outputs(job_dir, spec)
    fsync_tree(job_dir, man["outputs"])
    if to or rc == 124:
        raise Fail("train", "training hit its timeout (partial adapter/receipt persisted)", True, {"rc": rc})
    if rc in (5, 6):
        raise Fail("environment", "trainer refused the environment (pins / CUDA)", rc == 6, {"rc": rc, "tail": out[-400:]})
    if rc == 7:
        raise Fail("inputs", "trainer's own input digest check failed", False, {"rc": rc})
    if rc != 0:
        raise Fail("train", f"trainer exited {rc}", rc != 8, {"rc": rc, "tail": out[-600:]})
    tm = train_metrics(job_dir)
    if tm is None:
        raise Fail("train_outputs", "trainer exited 0 but wrote no RECEIPT.json", False)
    metrics, rec = tm
    ad = job_dir / "adapter" / "adapter_model.safetensors"
    if not ad.exists() or (rec.get("adapter_sha256") and sha256_file(ad) != rec["adapter_sha256"]):
        raise Fail("train_outputs", "adapter missing or its sha256 differs from the receipt", True)
    metrics["adapter_sha256"] = sha256_file(ad)
    scientific = None
    if metrics.get("steps_done") != metrics.get("max_steps"):
        scientific = f"training stopped early at {metrics.get('steps_done')}/{metrics.get('max_steps')} steps (divergence/early stop)"
    tr.update(status="DONE", metrics=metrics, scientific=scientific)
    return metrics, scientific


def stage_checkpoints(spec, job_dir):
    src_id = spec["checkpoints_from"]["experiment_id"]
    src_run = volume_root() / "runs" / src_id
    try:
        src_man = json.loads((src_run / "MANIFEST.json").read_text())
    except (OSError, ValueError):
        raise Fail("inputs", f"checkpoints_from: no readable manifest for {src_id}", False)
    if src_man.get("mode") != "train" or src_man.get("status") not in ("COMPLETED", "SCIENTIFIC_FAIL"):
        raise Fail("inputs", f"checkpoints_from: {src_id} is not a finished train run", False)
    src_job = volume_root() / src_man["output_dir"] / "job"
    n = 0
    for rel, meta in sorted(src_man.get("outputs", {}).items()):
        if rel.startswith(("ckpt/", "adapter/")):
            dst = job_dir / rel
            if dst.exists() and sha256_file(dst) == meta["sha256"]:
                continue
            if not (src_job / rel).exists() or sha256_file(src_job / rel) != meta["sha256"]:
                raise Fail("inputs", f"checkpoints_from: {rel} is missing or differs from its manifest digest", False)
            copy_verified(src_job / rel, dst, meta["sha256"])
            n += 1
    return n


def step_evaluate(spec, ctx, man, job_dir):
    st = man["steps"].get("evaluate", {})
    if st.get("status") == "DONE" and verify_outputs(job_dir, man.get("outputs", {})):
        ctx["resumed"] = "resumed_steps"
        return st["metrics"]
    attempt = int(st.get("attempt", 0)) + 1
    timeout = min(spec["eval_timeout_s"], remaining(ctx))
    if timeout < 120:
        raise Fail("deadline", "not enough runtime left to evaluate", True, {"left_s": remaining(ctx)})
    man["steps"]["evaluate"] = {"status": "RUNNING", "attempt": attempt, "t_start": time.time(), "worker": worker_id(), "timeout_s": timeout}
    persist(ctx, man)
    progress(ctx, f"evaluate attempt {attempt} started")
    args = [a.replace("{job}", str(job_dir)) for a in spec["evaluator_args"]]
    rc, to, out = run_proc([train_python(), "-u", job_dir / spec["evaluator"], *args], job_dir,
                           ctx["run_dir"] / "logs" / f"evaluate-{attempt:03d}.log", timeout)
    ev = man["steps"]["evaluate"]
    ev.update(rc=rc, timed_out=to, t_end=time.time(), status="FAILED")
    man["outputs"] = list_outputs(job_dir, spec)
    fsync_tree(job_dir, man["outputs"])
    if to or rc != 0:
        raise Fail("evaluate", "evaluator timed out" if to else f"evaluator exited {rc}", True, {"rc": rc, "tail": out[-600:]})
    metrics = {"output_files": sum(1 for r in man["outputs"] if r.startswith("evals/"))}
    if metrics["output_files"] == 0:
        raise Fail("evaluate", "evaluator exited 0 but wrote nothing under evals/", False)
    sj = job_dir / "evals" / "SCREEN.json"
    if sj.exists():
        try:
            s = json.loads(sj.read_text())
            metrics["eval_wall_s"] = s.get("wall_s")
            for k, v in (s.get("rows") or {}).items():
                metrics[f"rows/{k}"] = v
        except ValueError:
            pass
    metrics.update(extra_metrics(job_dir))
    metrics = {k: v for k, v in metrics.items() if v is not None}
    ev.update(status="DONE", metrics=metrics)
    return metrics


# ----------------------------------------------------------------------------------------------- manifest / result
def persist(ctx, man):
    man["updated_unix"] = time.time()
    return write_manifest_atomic(ctx["manifest_path"], man)


def progress(ctx, msg):
    try:
        runpod.serverless.progress_update(ctx["job"], msg)
    except Exception:  # noqa: BLE001  (progress is best-effort)
        pass


def compact(spec, ctx, status, stage, reason=None, metrics=None, gates=None, retryable=None, outputs=None):
    r = {"status": status, "experiment_id": spec.get("experiment_id"), "mode": spec.get("mode"), "stage": stage,
         "reason": reason, "metrics": metrics or {}, "gates": gates or [], "resumed": ctx.get("resumed", "fresh"),
         "manifest": str(ctx["manifest_path"]) if ctx.get("manifest_path") else None,
         "outputs": {"files": len(outputs or {}), "bytes": sum(m["bytes"] for m in (outputs or {}).values())},
         "worker": {"id": worker_id(), "gpu": (ctx.get("gpu") or {}).get("name")},
         "handler_version": HANDLER_VERSION, "wall_s": round(time.time() - ctx["t0"], 1)}
    if status == "INFRA_FAILURE":
        r["retryable"] = bool(retryable)
    return r


def handler(job):
    t0 = time.time()
    ctx = {"t0": t0, "job": job, "resumed": "fresh"}
    spec, man, lock = {}, None, None
    try:
        inp = job.get("input") if isinstance(job, dict) else None
        spec = validate(inp)
        root = volume_root()
        if env("PLLM_REQUIRE_MOUNT", "1") != "0" and not os.path.ismount(root):
            raise Fail("storage", f"no network volume mounted at {root}: refusing to run without persistent storage", False)
        run_dir = root / spec["output_dir"]
        ctx.update(run_dir=run_dir, deadline=t0 + spec["max_runtime_s"], attempt_tag=f"{int(t0)}-{uuid.uuid4().hex[:6]}")
        try:
            lock = acquire_lock(run_dir, spec["max_runtime_s"] + 900)
        except OSError as e:
            raise Fail("storage", f"network volume not writable: {e!r}", True)
        ctx["manifest_path"] = run_dir / "MANIFEST.json"

        if ctx["manifest_path"].exists():
            man = json.loads(ctx["manifest_path"].read_text())
            if man.get("fingerprint") != spec["fingerprint"]:
                man = None
                raise Fail("validation", "experiment_id/output_dir already used with different input (refused; never overwritten)", False)
            if spec["mode"] != "smoke_test" and man.get("status") in ("COMPLETED", "SCIENTIFIC_FAIL") \
                    and verify_outputs(run_dir / "job", man.get("outputs", {})):
                ctx["resumed"] = "cached"
                r = dict(man["result"], resumed="cached", wall_s=round(time.time() - t0, 1), persisted=True)
                r["manifest_sha256"] = sha256_file(ctx["manifest_path"])
                return r
        else:
            man = {"schema": MANIFEST_SCHEMA, "experiment_id": spec["experiment_id"], "mode": spec["mode"],
                   "fingerprint": spec["fingerprint"], "output_dir": spec["output_dir"], "inputs": spec["files"],
                   "recipe": spec["recipe"], "created_unix": t0, "steps": {}, "attempts": [], "outputs": {},
                   "promotion": "none (worker results are candidates only)", "handler_version": HANDLER_VERSION}
        man["status"] = "RUNNING"
        man["attempts"].append({"tag": ctx["attempt_tag"], "worker": worker_id(), "t_start": t0})
        persist(ctx, man)

        job_dir = run_dir / ("smoke" if spec["mode"] == "smoke_test" else "job")
        job_dir.mkdir(parents=True, exist_ok=True)
        if spec["files"]:
            progress(ctx, "verifying inputs")
            copied = stage_inputs(spec, job_dir)
            if spec["checkpoints_from"]:
                copied += stage_checkpoints(spec, job_dir)
            man["steps"]["inputs"] = {"status": "DONE", "files": len(spec["files"]), "copied": copied, "t": time.time()}
            persist(ctx, man)

        progress(ctx, "host checks")
        host_checks(spec, ctx, job_dir, expected_pins(spec, job_dir))
        if spec["mode"] in ("smoke_test", "train") and spec["files"]:
            bundle_preflight(spec, ctx, job_dir)
        man["steps"]["preflight"] = {"status": "DONE", "checks": ctx.get("preflight", []), "gpu": ctx.get("gpu"), "t": time.time()}
        man["attempts"][-1]["gpu"] = ctx.get("gpu")
        persist(ctx, man)

        scientific = None
        if spec["mode"] == "smoke_test":
            metrics = {"checks_passed": sum(1 for c in ctx.get("preflight", []) if c["pass"]),
                       "gpu_free_vram_gb": (ctx.get("gpu") or {}).get("free_gb")}
            metrics = {k: v for k, v in metrics.items() if v is not None}
            man["outputs"] = list_outputs(job_dir, spec)
        elif spec["mode"] == "train":
            metrics, scientific = step_train(spec, ctx, man, job_dir)
        else:
            metrics = step_evaluate(spec, ctx, man, job_dir)
        fsync_tree(job_dir, man["outputs"])
        gates = apply_gates(spec["gates"], metrics)
        failed = [g["metric"] for g in gates if not g["pass"]]
        if scientific or failed:
            status, reason = "SCIENTIFIC_FAIL", scientific or f"gate(s) failed: {failed}"
        else:
            status, reason = "COMPLETED", None
        result = compact(spec, ctx, status, "done", reason, metrics, gates, outputs=man["outputs"])
        man.update(status=status, result=result, metrics=metrics, gates=gates)
        man["attempts"][-1].update(t_end=time.time(), outcome=status)
        try:
            msha = persist(ctx, man)  # success is only ever returned after this durable write
        except OSError as e:
            raise Fail("persist", f"result manifest could not be written durably: {e!r}", True)
        return dict(result, manifest_sha256=msha, persisted=True)
    except Fail as f:
        return _infra(spec, ctx, man, f.stage, f.reason, f.retryable, f.detail)
    except Exception as e:  # noqa: BLE001  (never let an exception escape as an unstructured job error)
        return _infra(spec, ctx, man, "handler", f"{type(e).__name__}: {e}", True, {"trace": traceback.format_exc()[-800:]})
    finally:
        if lock is not None:
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                pass


def _infra(spec, ctx, man, stage, reason, retryable, detail):
    result = compact(spec, ctx, "INFRA_FAILURE", stage, reason, retryable=retryable, outputs=(man or {}).get("outputs"))
    if detail is not None:
        s = json.dumps(detail, default=str)
        result["detail"] = detail if len(s) <= 2000 else {"truncated": s[-1500:]}
    result["persisted"] = False
    if man is not None and ctx.get("manifest_path"):
        man.update(status="INFRA_FAILURE", result=result)
        if man.get("attempts"):
            man["attempts"][-1].update(t_end=time.time(), outcome=f"INFRA_FAILURE:{stage}")
        try:
            result["manifest_sha256"] = persist(ctx, man)
            result["persisted"] = True
        except Exception:  # noqa: BLE001  (storage may be the failing component)
            pass
    log("INFRA_FAILURE", stage, reason)
    return result


runpod.serverless.start({"handler": handler})
