"""Offline tests for handler.py. The RunPod SDK and every GPU/trainer/evaluator subprocess are faked; no network.
Run from the repository root:  python3 -m unittest discover -s tests -v
"""
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
STARTS = []
_fake = types.ModuleType("runpod")
_fake.serverless = types.SimpleNamespace(start=lambda cfg: STARTS.append(cfg), progress_update=lambda job, msg: None)
sys.modules["runpod"] = _fake
_spec = importlib.util.spec_from_file_location("handler", ROOT / "handler.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

PINS = dict(H.DEFAULT_PINS)


def sha(b):
    return hashlib.sha256(b).hexdigest()


class FakeHost:
    """Stands in for run_proc: answers host checks, and plays the bundle's trainer / evaluator / preflight script."""

    def __init__(self):
        self.calls = []
        self.vram_used_mib = 500
        self.pins = dict(PINS)
        self.train = self.train_ok
        self.evaluate = self.eval_ok

    def __call__(self, cmd, cwd, log_path, timeout, grace_s=120):
        cmd = [str(c) for c in cmd]
        cwd = pathlib.Path(cwd)
        self.calls.append(cmd)
        pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        if cmd[:2] == ["nvidia-smi", "-L"]:
            return 0, False, "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-x)"
        if cmd[0] == "nvidia-smi":
            return 0, False, f"NVIDIA GeForce RTX 4090, 24564, {self.vram_used_mib}"
        if len(cmd) > 2 and cmd[1] == "-c" and "PINS" in cmd[2]:
            return 0, False, "PINS " + json.dumps(self.pins)
        if len(cmd) > 2 and cmd[1] == "-c" and "CUDA_OK" in cmd[2]:
            return 0, False, "CUDA_OK NVIDIA GeForce RTX 4090"
        script = next((c for c in cmd if c.endswith(".py")), "")
        if script.endswith("host_preflight.py"):
            assert (cwd / "BUNDLE.SHA256SUMS").exists()
            return 0, False, 'HOST_PREFLIGHT {"result": "PASS", "failed": [], "wall_s": 1.0}'
        if script.endswith("train_lora_v2.py") and "--check-only" in cmd:
            return 0, False, "[train] CHECK OK"
        if script.endswith("train_lora_v2.py"):
            return self.train(cmd, cwd)
        if script.endswith("pod_eval.py"):
            return self.evaluate(cmd, cwd)
        raise AssertionError(f"unexpected command {cmd}")

    @staticmethod
    def write_train_outputs(cwd, steps_done=10, max_steps=10, partial=False):
        (cwd / "ckpt" / "step-00005").mkdir(parents=True, exist_ok=True)
        (cwd / "ckpt" / "step-00005" / "adapter_model.safetensors").write_bytes(b"ckpt5")
        (cwd / "adapter").mkdir(exist_ok=True)
        ad = cwd / "adapter" / "adapter_model.safetensors"
        ad.write_bytes(b"adapter-final-%d" % steps_done)
        (cwd / "adapter" / "adapter_config.json").write_text("{}")
        (cwd / "TRAIN_LOG.jsonl").write_text(json.dumps({"step": 5, "loss": 1.5}) + "\n" +
                                             json.dumps({"step": 5, "eval_loss": 1.2}) + "\n" +
                                             json.dumps({"step": 10, "loss": 0.9, "eval_loss": 1.1}) + "\n")
        (cwd / "RECEIPT.json").write_text(json.dumps({"steps_done": steps_done, "max_steps": max_steps, "train_wall_s": 12.5,
                                                      "adapter_sha256": sha(ad.read_bytes()), "partial": partial}))

    def train_ok(self, cmd, cwd):
        self.write_train_outputs(cwd)
        return 0, False, "DONE"

    def eval_ok(self, cmd, cwd):
        d = cwd / "evals" / "code-core@s100"
        d.mkdir(parents=True, exist_ok=True)
        (d / "outputs.jsonl").write_text('{"id": "a", "output": "x"}\n')
        (cwd / "evals" / "SCREEN.json").write_text(json.dumps({"wall_s": 3.0, "rows": {"code/core@s100": 1}}))
        (cwd / "evals" / "METRICS.json").write_text(json.dumps({"exact_match": 0.75}))
        return 0, False, "ALL DONE"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.vol = self.tmp / "vol"
        self.vol.mkdir()
        self.env = mock.patch.dict(os.environ, {"PLLM_VOLUME_ROOT": str(self.vol), "PLLM_REQUIRE_MOUNT": "0",
                                                "PLLM_TRAIN_PYTHON": "/opt/fake/venv/bin/python", "RUNPOD_POD_ID": "worker-a"})
        self.env.start()
        self.host = FakeHost()
        self.p1 = mock.patch.object(H, "run_proc", self.host)
        self.p1.start()

    def tearDown(self):
        self.p1.stop()
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def bundle(self, eid, extra=None, data_sha_override=None):
        b = self.vol / "bundles" / eid
        b.mkdir(parents=True, exist_ok=True)
        files = {"train.jsonl": b'{"messages": []}\n', "valid.jsonl": b'{"messages": []}\n', "train_lora_v2.py": b"# trainer stand-in\n"}
        files.update(extra or {})
        run = {"run_id": eid, "pins": PINS, "data_sha256": {"train.jsonl": sha(files["train.jsonl"]), "valid.jsonl": sha(files["valid.jsonl"])}}
        if data_sha_override:
            run["data_sha256"].update(data_sha_override)
        files["RUN.json"] = json.dumps(run).encode()
        for rel, data in files.items():
            (b / rel).parent.mkdir(parents=True, exist_ok=True)
            (b / rel).write_bytes(data)
        return {rel: sha(data) for rel, data in files.items()}

    def train_job(self, eid="exp-1", **kw):
        files = self.bundle(eid)
        inp = {"mode": "train", "experiment_id": eid, "bundle": {"files": files}, "recipe": {"file": "RUN.json", "sha256": files["RUN.json"]},
               "output_dir": f"runs/{eid}", "min_free_disk_gb": 0}
        inp.update(kw)
        return {"id": "job-1", "input": inp}

    def manifest(self, eid="exp-1"):
        return json.loads((self.vol / "runs" / eid / "MANIFEST.json").read_text())

    def trainer_calls(self):
        return [c for c in self.host.calls if any(x.endswith("train_lora_v2.py") for x in c) and "--check-only" not in c]


class TestValidation(Base):
    def assertRejected(self, inp, stage="validation"):
        r = H.handler({"id": "j", "input": inp})
        self.assertEqual(r["status"], "INFRA_FAILURE", r)
        self.assertEqual(r["stage"], stage, r)
        self.assertFalse(r["retryable"])
        self.assertEqual(self.host.calls, [])
        return r

    def test_missing_input_and_mode(self):
        self.assertRejected(None)
        self.assertRejected({"experiment_id": "x"})
        self.assertRejected({"mode": "inference", "experiment_id": "x"})

    def test_unknown_field_and_bad_id(self):
        self.assertRejected({"mode": "smoke_test", "experiment_id": "x", "surprise": 1})
        self.assertRejected({"mode": "smoke_test", "experiment_id": "../etc"})
        self.assertRejected({"mode": "smoke_test"})

    def test_train_requires_bundle_recipe_output(self):
        job = self.train_job()["input"]
        for k in ("bundle", "recipe", "output_dir"):
            self.assertRejected({x: v for x, v in job.items() if x != k})

    def test_bad_sha_and_traversal(self):
        job = self.train_job()["input"]
        bad = json.loads(json.dumps(job))
        bad["bundle"]["files"]["train.jsonl"] = "ABC"
        self.assertRejected(bad)
        bad = json.loads(json.dumps(job))
        bad["bundle"]["files"]["../secret"] = "0" * 64
        self.assertRejected(bad)
        bad = dict(job, output_dir="runs/../../x")
        self.assertRejected(bad)
        bad = dict(job, output_dir="/abs/runs")
        self.assertRejected(bad)
        bad = json.loads(json.dumps(job))
        bad["bundle"]["dir"] = "elsewhere/x"
        self.assertRejected(bad)

    def test_recipe_must_match_bundle(self):
        job = self.train_job()["input"]
        self.assertRejected(dict(job, recipe={"file": "RUN.json", "sha256": "0" * 64}))

    def test_bad_gate(self):
        job = self.train_job()["input"]
        self.assertRejected(dict(job, gates=[{"metric": "x", "op": "~", "value": 1}]))
        self.assertRejected(dict(job, gates=[{"metric": "x", "op": ">=", "value": True}]))

    def test_no_files_written_on_validation_failure(self):
        self.assertRejected({"mode": "train", "experiment_id": "exp-v"})
        self.assertFalse((self.vol / "runs").exists())


class TestInputs(Base):
    def test_sha_mismatch_is_infra_failure_at_inputs(self):
        job = self.train_job()
        (self.vol / "bundles" / "exp-1" / "train.jsonl").write_bytes(b"tampered\n")
        r = H.handler(job)
        self.assertEqual((r["status"], r["stage"], r["retryable"]), ("INFRA_FAILURE", "inputs", False))
        self.assertEqual(self.trainer_calls(), [])
        self.assertEqual(self.host.calls, [])  # not even host checks run on bad inputs
        self.assertTrue(r["persisted"])
        self.assertEqual(self.manifest()["status"], "INFRA_FAILURE")

    def test_missing_file_is_infra_failure_at_inputs(self):
        job = self.train_job()
        (self.vol / "bundles" / "exp-1" / "valid.jsonl").unlink()
        r = H.handler(job)
        self.assertEqual((r["status"], r["stage"]), ("INFRA_FAILURE", "inputs"))

    def test_recipe_data_digest_must_match_bundle(self):
        files = self.bundle("exp-d", data_sha_override={"train.jsonl": "1" * 64})
        job = {"input": {"mode": "train", "experiment_id": "exp-d", "bundle": {"files": files}, "output_dir": "runs/exp-d",
                         "recipe": {"file": "RUN.json", "sha256": files["RUN.json"]}}}
        r = H.handler(job)
        self.assertEqual((r["status"], r["stage"]), ("INFRA_FAILURE", "inputs"))
        self.assertEqual(self.trainer_calls(), [])

    def test_https_fetch_verifies_sha(self):
        job = self.train_job()
        good = (self.vol / "bundles" / "exp-1" / "valid.jsonl").read_bytes()
        (self.vol / "bundles" / "exp-1" / "valid.jsonl").unlink()
        job["input"]["bundle"]["source"] = {"type": "https", "base_url": "https://example.invalid/b"}
        with mock.patch.object(H, "_https_get", lambda url, dst: pathlib.Path(dst).write_bytes(b"wrong")):
            r = H.handler(job)
        self.assertEqual((r["status"], r["stage"], r["retryable"]), ("INFRA_FAILURE", "inputs", False))
        self.assertFalse((self.vol / "bundles" / "exp-1" / "valid.jsonl").exists())
        with mock.patch.object(H, "_https_get", lambda url, dst: pathlib.Path(dst).write_bytes(good)):
            r = H.handler(job)
        self.assertEqual(r["status"], "COMPLETED", r)

    def test_inputs_copied_into_job_dir_on_volume(self):
        r = H.handler(self.train_job())
        self.assertEqual(r["status"], "COMPLETED", r)
        job_dir = self.vol / "runs" / "exp-1" / "job"
        for rel, digest in self.manifest()["inputs"].items():
            self.assertEqual(H.sha256_file(job_dir / rel), digest)


class TestPersistence(Base):
    def test_success_only_after_durable_manifest(self):
        events = []
        real_write, real_fsync = H.write_manifest_atomic, H.fsync_tree

        def spy_write(path, man):
            events.append(("manifest", man["status"]))
            return real_write(path, man)

        def spy_fsync(root, rels):
            events.append(("fsync_outputs", len(rels)))
            return real_fsync(root, rels)

        with mock.patch.object(H, "write_manifest_atomic", spy_write), mock.patch.object(H, "fsync_tree", spy_fsync):
            r = H.handler(self.train_job())
        self.assertEqual(r["status"], "COMPLETED", r)
        self.assertEqual(events[-1], ("manifest", "COMPLETED"))
        last_fsync = max(i for i, e in enumerate(events) if e[0] == "fsync_outputs")
        self.assertLess(last_fsync, len(events) - 1)  # outputs fsynced before the final manifest
        mp = self.vol / "runs" / "exp-1" / "MANIFEST.json"
        self.assertEqual(r["manifest_sha256"], H.sha256_file(mp))
        man = self.manifest()
        self.assertEqual(man["status"], "COMPLETED")
        self.assertIn("adapter/adapter_model.safetensors", man["outputs"])
        self.assertIn("ckpt/step-00005/adapter_model.safetensors", man["outputs"])
        self.assertTrue(r["persisted"])
        self.assertFalse(list(mp.parent.glob(".MANIFEST.json.tmp-*")))  # atomic: no temp files left behind
        self.assertFalse((mp.parent / ".lock").exists())

    def test_manifest_write_failure_never_reports_success(self):
        real_write = H.write_manifest_atomic

        def failing(path, man):
            if man["status"] == "COMPLETED":
                raise OSError("volume went away")
            return real_write(path, man)

        with mock.patch.object(H, "write_manifest_atomic", failing):
            r = H.handler(self.train_job())
        self.assertEqual((r["status"], r["stage"], r["retryable"]), ("INFRA_FAILURE", "persist", True))
        self.assertNotEqual(self.manifest()["status"], "COMPLETED")

    def test_atomic_write_replaces_whole_file(self):
        d = self.tmp / "atomic"
        p = d / "m.json"
        H.write_manifest_atomic(p, {"status": "RUNNING", "fingerprint": "f"})
        H.write_manifest_atomic(p, {"status": "COMPLETED", "fingerprint": "f"})
        self.assertEqual(json.loads(p.read_text())["status"], "COMPLETED")
        self.assertEqual([x.name for x in d.iterdir()], ["m.json"])


class TestIdempotency(Base):
    def test_same_input_returns_cached_result(self):
        job = self.train_job()
        r1 = H.handler(job)
        r2 = H.handler(job)
        self.assertEqual(r1["status"], "COMPLETED")
        self.assertEqual(r2["status"], "COMPLETED")
        self.assertEqual(r2["resumed"], "cached")
        self.assertEqual(r2["metrics"], r1["metrics"])
        self.assertEqual(len(self.trainer_calls()), 1)

    def test_same_id_different_input_is_refused(self):
        H.handler(self.train_job())
        before = self.manifest()
        job = self.train_job()
        job["input"]["note"] = "changed"
        r = H.handler(job)
        self.assertEqual((r["status"], r["stage"], r["retryable"]), ("INFRA_FAILURE", "validation", False))
        self.assertEqual(self.manifest(), before)

    def test_tampered_outputs_force_rerun(self):
        job = self.train_job()
        H.handler(job)
        (self.vol / "runs" / "exp-1" / "job" / "adapter" / "adapter_model.safetensors").write_bytes(b"corrupt")
        r = H.handler(job)
        self.assertEqual(r["status"], "COMPLETED")
        self.assertEqual(r["resumed"], "restarted")
        self.assertEqual(len(self.trainer_calls()), 2)

    def test_interrupted_train_restarts_and_archives_partial(self):
        def timeout_train(cmd, cwd):
            FakeHost.write_train_outputs(cwd, steps_done=5, partial=True)
            return 124, True, "SIGTERM"

        self.host.train = timeout_train
        job = self.train_job()
        r1 = H.handler(job)
        self.assertEqual((r1["status"], r1["stage"], r1["retryable"]), ("INFRA_FAILURE", "train", True))
        self.assertTrue(r1["persisted"])
        man = self.manifest()
        self.assertEqual(man["steps"]["train"]["status"], "FAILED")
        self.assertIn("RECEIPT.json", man["outputs"])  # the partial receipt was persisted and recorded
        self.host.train = self.host.train_ok
        r2 = H.handler(job)
        self.assertEqual(r2["status"], "COMPLETED", r2)
        self.assertEqual(r2["resumed"], "restarted")
        self.assertTrue((self.vol / "runs" / "exp-1" / "attempts" / "001" / "RECEIPT.json").exists())
        self.assertEqual(self.manifest()["steps"]["inputs"]["copied"], 0)  # inputs already staged and verified

    def test_resume_from_last_checkpoint_when_trainer_supports_it(self):
        def timeout_train(cmd, cwd):
            FakeHost.write_train_outputs(cwd, steps_done=5, partial=True)
            return 124, True, "SIGTERM"

        self.host.train = timeout_train
        job = self.train_job(resume_arg="--resume-from")
        H.handler(job)
        self.host.train = self.host.train_ok
        r = H.handler(job)
        self.assertEqual(r["status"], "COMPLETED", r)
        self.assertEqual(r["resumed"], "resumed_from_checkpoint")
        last = self.trainer_calls()[-1]
        self.assertEqual(last[-2], "--resume-from")
        self.assertTrue(last[-1].endswith("ckpt/step-00005"))
        self.assertFalse((self.vol / "runs" / "exp-1" / "attempts").exists())

    def test_concurrent_worker_lock(self):
        run = self.vol / "runs" / "exp-1"
        run.mkdir(parents=True)
        (run / ".lock").write_text("{}")
        r = H.handler(self.train_job())
        self.assertEqual((r["status"], r["stage"], r["retryable"]), ("INFRA_FAILURE", "lock", True))
        self.assertEqual(self.trainer_calls(), [])
        old = time.time() - 10 ** 6
        os.utime(run / ".lock", (old, old))
        self.assertEqual(H.handler(self.train_job())["status"], "COMPLETED")


class TestClassification(Base):
    def run_train(self, train, **kw):
        self.host.train = train
        return H.handler(self.train_job(**kw))

    def test_divergence_is_scientific_fail_with_outputs_persisted(self):
        def diverged(cmd, cwd):
            FakeHost.write_train_outputs(cwd, steps_done=4, max_steps=10)
            return 0, False, "DIVERGENCE"

        r = self.run_train(diverged)
        self.assertEqual(r["status"], "SCIENTIFIC_FAIL", r)
        self.assertNotIn("retryable", r)
        self.assertEqual(r["metrics"]["steps_done"], 4)
        self.assertEqual(self.manifest()["status"], "SCIENTIFIC_FAIL")

    def test_gates(self):
        r = self.run_train(self.host.train_ok, gates=[{"metric": "steps_done_ratio", "op": ">=", "value": 1.0},
                                                      {"metric": "min_eval_loss", "op": "<=", "value": 1.15}])
        self.assertEqual(r["status"], "COMPLETED", r)
        self.assertTrue(all(g["pass"] for g in r["gates"]))

    def test_failed_gate_is_scientific_fail(self):
        r = self.run_train(self.host.train_ok, gates=[{"metric": "final_eval_loss", "op": "<", "value": 0.5}])
        self.assertEqual(r["status"], "SCIENTIFIC_FAIL")
        self.assertEqual(r["gates"][0]["observed"], 1.1)

    def test_missing_gate_metric_fails_closed(self):
        r = self.run_train(self.host.train_ok, gates=[{"metric": "nonexistent", "op": ">", "value": 0}])
        self.assertEqual((r["status"], r["stage"], r["retryable"]), ("INFRA_FAILURE", "gates", False))

    def test_trainer_exit_codes(self):
        cases = {6: ("environment", True), 5: ("environment", False), 8: ("train", False), 1: ("train", True), 7: ("inputs", False)}
        for rc, (stage, retry) in cases.items():
            with self.subTest(rc=rc):
                eid = f"exp-rc{rc}"
                self.host.train = lambda cmd, cwd, rc=rc: (rc, False, "boom")
                r = H.handler(self.train_job(eid=eid, output_dir=f"runs/{eid}"))
                self.assertEqual((r["status"], r["stage"], r["retryable"]), ("INFRA_FAILURE", stage, retry))

    def test_exit_zero_without_receipt(self):
        r = self.run_train(lambda cmd, cwd: (0, False, "nothing"))
        self.assertEqual((r["status"], r["stage"]), ("INFRA_FAILURE", "train_outputs"))

    def test_receipt_adapter_mismatch(self):
        def bad(cmd, cwd):
            FakeHost.write_train_outputs(cwd)
            (cwd / "adapter" / "adapter_model.safetensors").write_bytes(b"other")
            return 0, False, ""

        r = self.run_train(bad)
        self.assertEqual((r["status"], r["stage"]), ("INFRA_FAILURE", "train_outputs"))

    def test_low_vram_and_pin_mismatch_are_preflight_failures(self):
        self.host.vram_used_mib = 10000
        r = H.handler(self.train_job())
        self.assertEqual((r["status"], r["stage"], r["retryable"]), ("INFRA_FAILURE", "preflight", True))
        self.assertEqual(r["detail"]["check"], "vram_free")
        self.assertEqual(self.trainer_calls(), [])
        self.host.vram_used_mib = 500
        self.host.pins = dict(PINS, torch="2.9.0")
        r = H.handler(self.train_job(eid="exp-p", output_dir="runs/exp-p"))
        self.assertEqual((r["status"], r["stage"], r["retryable"]), ("INFRA_FAILURE", "preflight", False))
        self.assertEqual(r["detail"]["check"], "pinned_imports")

    def test_unexpected_exception_is_structured(self):
        with mock.patch.object(H, "host_checks", side_effect=RuntimeError("kaboom")):
            r = H.handler(self.train_job())
        self.assertEqual((r["status"], r["stage"], r["retryable"]), ("INFRA_FAILURE", "handler", True))

    def test_no_volume_mount_refused(self):
        with mock.patch.dict(os.environ, {"PLLM_REQUIRE_MOUNT": "1"}):
            r = H.handler(self.train_job())
        self.assertEqual((r["status"], r["stage"]), ("INFRA_FAILURE", "storage"))
        self.assertEqual(self.host.calls, [])


class TestSmokeAndEvaluate(Base):
    def test_smoke_payload_file_runs_host_checks_only(self):
        payload = json.loads((ROOT / "smoke_test.json").read_text())
        r = H.handler(payload)
        self.assertEqual(r["status"], "COMPLETED", r)
        self.assertEqual(r["metrics"]["checks_passed"], 6)
        self.assertEqual(self.trainer_calls(), [])
        man = json.loads((self.vol / "runs" / payload["input"]["experiment_id"] / "MANIFEST.json").read_text())
        self.assertEqual(man["status"], "COMPLETED")
        r2 = H.handler(payload)  # smoke is re-run on every call (each worker must prove itself), never served from cache
        self.assertEqual(r2["resumed"], "fresh")
        self.assertEqual(len(self.manifest(payload["input"]["experiment_id"])["attempts"]), 2)

    def test_smoke_with_bundle_never_trains(self):
        files = self.bundle("smoke-b")
        r = H.handler({"input": {"mode": "smoke_test", "experiment_id": "smoke-b", "bundle": {"files": files},
                                 "recipe": {"file": "RUN.json", "sha256": files["RUN.json"]}}})
        self.assertEqual(r["status"], "COMPLETED", r)
        trainer = [c for c in self.host.calls if any(x.endswith("train_lora_v2.py") for x in c)]
        self.assertEqual(len(trainer), 1)
        self.assertIn("--check-only", trainer[0])

    def test_smoke_with_bundle_preflight_script(self):
        files = self.bundle("smoke-h", extra={"host_preflight.py": b"# preflight stand-in\n"})
        r = H.handler({"input": {"mode": "smoke_test", "experiment_id": "smoke-h", "bundle": {"files": files}}})
        self.assertEqual(r["status"], "COMPLETED", r)
        hp = [c for c in self.host.calls if any(x.endswith("host_preflight.py") for x in c)][0]
        self.assertEqual(hp[hp.index("--pins") + 1], "")  # no runtime pip installs: the pins are baked into the image
        self.assertEqual(self.trainer_calls(), [])

    def test_evaluate_checkpoints_from_train_run(self):
        self.assertEqual(H.handler(self.train_job(eid="t1", output_dir="runs/t1"))["status"], "COMPLETED")
        files = self.bundle("e1", extra={"pod_eval/pod_eval.py": b"# evaluator stand-in\n", "pod_eval/screen.json": b"{}"})
        job = {"input": {"mode": "evaluate", "experiment_id": "e1", "bundle": {"files": files}, "output_dir": "runs/e1",
                         "recipe": {"file": "RUN.json", "sha256": files["RUN.json"]}, "checkpoints_from": {"experiment_id": "t1"},
                         "gates": [{"metric": "exact_match", "op": ">=", "value": 0.7}]}}
        r = H.handler(job)
        self.assertEqual(r["status"], "COMPLETED", r)
        self.assertEqual(r["metrics"]["rows/code/core@s100"], 1)
        ev = [c for c in self.host.calls if any(x.endswith("pod_eval.py") for x in c)][0]
        self.assertEqual(ev[-2:], ["--job", str(self.vol / "runs" / "e1" / "job")])
        self.assertTrue((self.vol / "runs" / "e1" / "job" / "ckpt" / "step-00005" / "adapter_model.safetensors").exists())
        self.assertEqual(self.trainer_calls()[-1][2], str(self.vol / "runs" / "t1" / "job" / "train_lora_v2.py"))  # no extra training

    def test_evaluate_refuses_tampered_checkpoint(self):
        H.handler(self.train_job(eid="t2", output_dir="runs/t2"))
        (self.vol / "runs" / "t2" / "job" / "ckpt" / "step-00005" / "adapter_model.safetensors").write_bytes(b"tampered")
        files = self.bundle("e2", extra={"pod_eval/pod_eval.py": b"# evaluator stand-in\n"})
        r = H.handler({"input": {"mode": "evaluate", "experiment_id": "e2", "bundle": {"files": files}, "output_dir": "runs/e2",
                                 "recipe": {"file": "RUN.json", "sha256": files["RUN.json"]}, "checkpoints_from": {"experiment_id": "t2"}}})
        self.assertEqual((r["status"], r["stage"], r["retryable"]), ("INFRA_FAILURE", "inputs", False))


class TestProcessAndEnv(unittest.TestCase):
    def test_child_env_drops_credentials(self):
        secret_env = {"RUNPOD_API_KEY": "rk", "AWS_SECRET_ACCESS_KEY": "aws", "AWS_ACCESS_KEY_ID": "id",
                      "PLLM_FETCH_BEARER_TOKEN": "tok", "HF_TOKEN": "hf", "PLLM_TRAIN_PYTHON": "/opt/x/venv/bin/python"}
        with mock.patch.dict(os.environ, secret_env):
            e = H.child_env()
        for k in ("RUNPOD_API_KEY", "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "PLLM_FETCH_BEARER_TOKEN"):
            self.assertNotIn(k, e)
        self.assertEqual(e["HF_TOKEN"], "hf")
        self.assertTrue(e["PATH"].startswith("/opt/x/venv/bin:"))
        self.assertEqual(e["HF_HUB_ENABLE_HF_TRANSFER"], "0")

    def test_timeout_sends_sigterm_so_trainer_can_save(self):
        d = pathlib.Path(tempfile.mkdtemp())
        code = ("import signal,sys,time,pathlib\n"
                "def h(*a):\n pathlib.Path('saved_partial').write_text('x'); sys.exit(124)\n"
                "signal.signal(signal.SIGTERM,h)\nprint('started',flush=True)\ntime.sleep(60)\n")
        rc, timed_out, out = H.run_proc([sys.executable, "-c", code], d, d / "logs" / "t.log", timeout=2, grace_s=10)
        self.assertTrue(timed_out)
        self.assertEqual(rc, 124)
        self.assertTrue((d / "saved_partial").exists())
        self.assertIn("started", out)
        shutil.rmtree(d, ignore_errors=True)


class TestEntrypoint(unittest.TestCase):
    def test_serverless_start_registered_with_handler(self):
        self.assertEqual(len(STARTS), 1)
        self.assertIs(STARTS[0]["handler"], H.handler)
        last = [ln for ln in (ROOT / "handler.py").read_text().splitlines() if ln.strip()][-1]
        self.assertEqual(last, 'runpod.serverless.start({"handler": handler})')

    def test_dockerfile_pins_base_by_digest_and_copies_handler(self):
        df = (ROOT / "Dockerfile").read_text()
        self.assertIn("FROM ghcr.io/brohrt/phone-llm-environment@sha256:ca17c0a629c8767451c7cf0a212993f651197706c529ef6bcfc15f2240952bae", df)
        self.assertIn("--require-hashes", df)
        self.assertIn("COPY handler.py", df)
        lock = (ROOT / "serverless" / "requirements.lock").read_text()
        self.assertIn("runpod==1.12.0", lock)


if __name__ == "__main__":
    unittest.main()
