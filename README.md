# Pinned Linux amd64 environment

Private environment build repository. Contains only the immutable Runpod base reference, dependency wheel lock, package-install provenance, and environment build/check scripts. No model weights, data, evaluation cases, adapters, credentials, or training/scoring implementation.

Build via the manually dispatched GitHub Actions workflow. The job token publishes to GHCR with packages:write. Image verification checks Python3.12.3,55 locked distributions, imports of torch2.10.0/transformers5.3.0/peft0.18.1/accelerate1.12.0, and preinstalled SSH. It does not require or test a CUDA device. The published digest is pulled and checked again. Logs and receipts are retained as Actions artifacts.

Package visibility changes require the owner's explicit action. Publishing a private package does not prove anonymous pull access.

## RunPod Serverless worker

The repository root also builds a **RunPod Serverless queue worker** for bounded LoRA training, checkpoint evaluation and smoke tests. It is a fallback/overflow executor, not an inference server. The worker contains no training or scoring logic. Every job brings its own trainer, evaluator and data as a sha256-pinned bundle on the endpoint's network volume. The worker verifies the bundle, runs host checks, runs the bundle's scripts, persists everything to the volume and returns a compact JSON result.

| File | Purpose |
|---|---|
| `Dockerfile` | `FROM` this repository's environment image **by digest**; adds the RunPod SDK in a separate venv; `CMD` runs the handler |
| `handler.py` | job validation, sha256 verification, host checks, script execution, atomic manifest, result; ends with `runpod.serverless.start({"handler": handler})` |
| `serverless/requirements.lock` | `runpod==1.12.0` and its dependencies, wheels only, every hash pinned |
| `smoke_test.json` | smoke-test request payload |
| `tests/` | offline tests (`python3 -m unittest discover -s tests -v`); RunPod SDK and subprocesses are faked |
| `.dockerignore` | limits the root build context to the four files above (the `image/` build is unaffected) |

The base image already holds the locked training environment (`/opt/phone-llm/venv`: Python 3.12.3, torch 2.10.0+cu128, transformers 5.3.0, peft 0.18.1, accelerate 1.12.0), and the worker build leaves it unchanged. The build re-runs `verify_stack.py --runtime` to prove it.

### Deploying (nothing in this repository creates endpoints or spends money)

**Option A: GitHub integration.** In RunPod: Serverless → New Endpoint → GitHub repo → this repository, branch `main`, Dockerfile path `Dockerfile`, build context `.` (repository root). RunPod builds the image on its side. The base image is pulled anonymously from GHCR, so the package must stay publicly pullable.

**Option B: registry image.** `docker build --platform linux/amd64 -t <registry>/<name>:<tag> . && docker push <registry>/<name>:<tag>`. Then create the endpoint from that image, preferably referenced by digest. A private registry needs RunPod registry credentials.

### Endpoint settings (recommended)

| Setting | Value | Why |
|---|---|---|
| Endpoint type | **Queue** (not load balancer) | long-running jobs; async `/run` + `/status` |
| GPUs per worker | 1 | |
| GPU selection (priority order) | 24 GB PRO (RTX 4090) → 48 GB PRO (L40S) → 80 GB (A100 80 GB) → 80 GB PRO (H100); add A100 40 GB if offered | broad fallback, not H100-only; all need ≥ 24 GB and bf16 support. Check each UI tier's card list. Tiers containing slower 24 GB cards (L4, A5000, 3090) are optional |
| Active (min) workers | **0** | nothing bills while idle |
| Max workers | **2** | caps concurrent spend |
| Worker type | flex | |
| Idle timeout | 5–10 s | a finished worker stops billing almost immediately |
| Execution timeout | **14 400 s (4 h)** | ≥ 3 h for ~1–2 h jobs; above the handler's own cap (`PLLM_MAX_RUNTIME_S` = 13 500 s), so the handler always finishes persisting before RunPod kills the job |
| FlashBoot | on | faster cold starts, no extra cost |
| Allowed CUDA versions | 12.8 and newer | the base image declares `cuda>=12.8` |
| Network volume | attach one (mounted at `/runpod-volume`) | **required**: the worker refuses to run without it |
| Container disk | 20 GB | scratch only; all outputs go to the volume |

A network volume ties the endpoint to that volume's data centre. Pick a data centre that offers several of the GPU types above, or the GPU fallback list will shrink.

### Secrets and environment variables

No secret is required for public base models. Set credentials only as RunPod **Secrets**, referenced from endpoint env vars, e.g. `HF_TOKEN={{ RUNPOD_SECRET_hf_token }}`. Never put them in the repository or the image.

| Name | Kind | Needed when |
|---|---|---|
| `HF_TOKEN` | secret, optional | gated/private base model, or to avoid anonymous Hub rate limits |
| `PLLM_FETCH_BEARER_TOKEN` | secret, optional | jobs use `bundle.source` `https` with bearer auth |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | secrets, optional | jobs use `bundle.source` `s3` |
| `AWS_ENDPOINT_URL`, `AWS_DEFAULT_REGION` | env, optional | S3-compatible storage for `bundle.source` `s3` |
| `PLLM_VOLUME_ROOT` | env, default `/runpod-volume` | leave as is |
| `HF_HOME` | env, default `/runpod-volume/hf` | base-model cache on the volume (downloaded once, shared by all workers) |
| `PLLM_HF_OFFLINE` | env, default unset | set `1` once the base model is cached on the volume |
| `PLLM_MAX_RUNTIME_S` | env, default `13500` | hard cap on any job's runtime (cost control) |

The worker needs **no RunPod API key**. Bundle scripts run with a minimal environment: `HF_TOKEN` is passed through, but RunPod, S3 and fetch credentials are not.

### GPU memory

**Minimum 24 GB VRAM.** A ~2B-parameter base model in bf16 (e.g. Qwen3.5-2B) with LoRA r16 on all blocks, micro-batch 2 × gradient accumulation 2, sequence cap 1024 and gradient checkpointing has trained to completion on an RTX 4090 24 GB (all steps, ~3.6 s/step). Jobs default to `min_free_vram_gb: 20`. The host check refuses a worker with less free VRAM, and the job comes back as `INFRA_FAILURE` at stage `preflight` with `retryable: true`.

### Storage layout and size

```
/runpod-volume/
  hf/                          base-model cache (HF_HOME)                         ≈ 4.5–5 GB for a ~2B bf16 model
  bundles/<experiment_id>/     inputs staged before the job (sha256-pinned)       ≈ 0.01–0.1 GB each
  runs/<experiment_id>/
    MANIFEST.json              atomic manifest: inputs, steps, outputs (sha256 + bytes), metrics, gates, result
    job/                       verified copy of the bundle + everything the trainer/evaluator writes
                               (adapter/, ckpt/step-N/, TRAIN_LOG.jsonl, RECEIPT.json, evals/)
    logs/                      per-stage subprocess logs
    attempts/NNN/              partial outputs of an interrupted attempt (kept, never deleted)
```

Per training run: adapter ≈ 67 MB (r16, all blocks) × (final + 3–8 checkpoints + trainer copy), plus a second copy of the bundle, ≈ 0.6–1 GB. Per evaluation run: copied checkpoints plus outputs, ≈ 0.3–0.7 GB. **Recommended volume: 50 GB** (≈ 5 GB model cache + ~40 runs), billed per GB-month whether or not workers run. Prune `runs/` after results have been pulled.

**Staging bundles:** copy files to `/runpod-volume/bundles/<experiment_id>/` with a short-lived pod that has the same network volume attached (or the volume's S3-compatible API where available). Then submit the job with every file's sha256. A job can instead set `bundle.source` (`https` or `s3`): files missing from the volume are downloaded, sha256-verified, then kept.

### Job input

| Field | Required | Meaning |
|---|---|---|
| `mode` | yes | `smoke_test` \| `train` \| `evaluate` |
| `experiment_id` | yes | `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`; idempotency key |
| `bundle.files` | train, evaluate | `{relative path: sha256}` for **every** input (scripts, configs, data, adapters) |
| `bundle.dir` | no | default `bundles/<experiment_id>` (must be under `bundles/`) |
| `bundle.source` | no | `{"type": "https", "base_url": "https://…"}` or `{"type": "s3", "bucket": "…", "prefix": "…"}` |
| `recipe` | train, evaluate | `{"file": "RUN.json", "sha256": "…"}`: the approved config, which must also appear in `bundle.files`. Its `data_sha256` and `pins`, if present, are cross-checked |
| `output_dir` | train, evaluate | under `runs/`, e.g. `runs/<experiment_id>` |
| `trainer`, `trainer_args` | no | default `train_lora_v2.py`, `[]`; run as `python <job>/<trainer> <args>` in the job directory |
| `resume_arg` | no | the trainer's resume flag, if it has one: an interrupted run then continues from the newest `ckpt/step-*`. Without it, an interrupted run restarts from step 0 and the partial outputs are archived |
| `evaluator`, `evaluator_args` | no | default `pod_eval/pod_eval.py`, `["--job", "{job}"]` |
| `preflight_script` | no | default `host_preflight.py`; runs if listed in the bundle (`--pins ""`: no runtime pip). Otherwise the trainer's `--check-only` runs |
| `checkpoints_from` | evaluate only | `{"experiment_id": "<finished train run>"}`: copies its `ckpt/` and `adapter/`, verified against that run's manifest |
| `gates` | no | `[{"metric": "min_eval_loss", "op": "<=", "value": 1.2}, …]`; ops `>= <= > < == !=` |
| `min_free_vram_gb`, `min_free_disk_gb` | no | defaults 20 and 10 (train) / 1 |
| `max_runtime_s`, `train_timeout_s`, `eval_timeout_s` | no | defaults 13 500 (capped by `PLLM_MAX_RUNTIME_S`), 10 800, 5 400 |

Trainer contract: on success, write `RECEIPT.json` (`steps_done`, `max_steps`, `adapter_sha256`) and `adapter/adapter_model.safetensors` in its own directory. On SIGTERM (timeout), save what it can and exit 124. Evaluator contract: write under `evals/`; an optional flat `evals/METRICS.json` supplies gate metrics.

### Result

```json
{"status": "COMPLETED", "experiment_id": "…", "mode": "train", "stage": "done", "reason": null,
 "metrics": {"steps_done": 648, "max_steps": 648, "steps_done_ratio": 1.0, "final_eval_loss": 0.41, "adapter_sha256": "…"},
 "gates": [], "resumed": "fresh", "manifest": "/runpod-volume/runs/…/MANIFEST.json", "manifest_sha256": "…",
 "outputs": {"files": 14, "bytes": 412000000}, "worker": {"id": "…", "gpu": "NVIDIA GeForce RTX 4090"},
 "persisted": true, "handler_version": "1.0.0", "wall_s": 3123.4}
```

- `COMPLETED`: the step ran, outputs are persisted, and every gate passed.
- `SCIENTIFIC_FAIL`: the step ran and outputs are persisted, but a gate failed or training stopped early (e.g. divergence).
- `INFRA_FAILURE`: the step could not run or could not be persisted. `stage` is one of `validation`, `storage`, `lock`, `inputs`, `preflight`, `environment`, `train`, `train_outputs`, `evaluate`, `gates`, `deadline`, `persist`, `handler`. `retryable: false` means resubmitting the same job cannot help (bad inputs, pin mismatch).

`COMPLETED` and `SCIENTIFIC_FAIL` are only returned after the manifest has been written atomically (tmp + fsync + rename + directory fsync) and read back. Resubmitting the same input returns the stored result (`resumed: "cached"`). Reusing an `experiment_id` with different input is refused. `smoke_test` always re-runs, because each worker must prove its own host. Results are candidates only: nothing is promoted by the worker. From RunPod's side every job is "COMPLETED" once the handler returns, so read `output.status`.

### Smoke test

From the endpoint's **Requests** tab, send `smoke_test.json`:

```json
{"input": {"mode": "smoke_test", "experiment_id": "smoke-test-001", "min_free_vram_gb": 20, "min_free_disk_gb": 1}}
```

It checks: GPU visible, free VRAM ≥ 20 GB, network-volume write/fsync/read-back and free space, pinned imports in the training venv, and a CUDA allocation plus a tiny bf16 forward/backward. It writes `runs/smoke-test-001/MANIFEST.json` and never trains. Adding a `bundle` also verifies every bundle file and runs the bundle's preflight script, or the trainer's `--check-only`.

### Cost controls

- 0 active workers (no idle billing) and max 2 workers.
- 5–10 s idle timeout.
- 4 h execution timeout, with the handler's 3 h 45 min cap inside it.
- Per-stage timeouts. On timeout the trainer gets SIGTERM, so a partial adapter is saved, not lost.
- Idempotent `experiment_id`, so a resubmitted job never trains twice.
- Host checks run before any training, so a bad worker fails in about a minute.

Worst-case spend for a burst is max workers × execution timeout × the highest per-second rate among the selected GPUs. Read the rates in the endpoint's GPU picker and keep that product inside the programme budget. The worker itself has no budget authority: whoever submits jobs owns the budget gate.
