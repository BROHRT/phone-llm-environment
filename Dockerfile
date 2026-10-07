# RunPod Serverless worker (queue endpoint) for bounded LoRA training / evaluation jobs.
# Built by RunPod's GitHub integration from the repository root (or locally: docker build --platform linux/amd64 -t <tag> .).
#
# Base = this repository's own published environment image, pinned BY DIGEST (built by .github/workflows/build-image.yml
# from image/Dockerfile + image/requirements.lock): runpod/base (CUDA 12.8.1, cuDNN 9.8, Ubuntu 24.04, Python 3.12.3) with
# torch 2.10.0+cu128 / transformers 5.3.0 / peft 0.18.1 / accelerate 1.12.0 locked in /opt/phone-llm/venv.
# That training environment is NOT modified here: the RunPod SDK goes into a separate venv (/opt/serverless/venv),
# installed wheels-only with every hash pinned (serverless/requirements.lock).
FROM ghcr.io/brohrt/phone-llm-environment@sha256:ca17c0a629c8767451c7cf0a212993f651197706c529ef6bcfc15f2240952bae

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

COPY serverless/requirements.lock /opt/serverless/requirements.lock
RUN python3 -m venv /opt/serverless/venv \
 && /opt/serverless/venv/bin/python -m pip install --no-cache-dir --only-binary=:all: --require-hashes \
      --index-url https://pypi.org/simple -r /opt/serverless/requirements.lock \
 && /opt/serverless/venv/bin/python -c "import importlib.metadata as m; assert m.version('runpod') == '1.12.0'" \
 && /opt/phone-llm/venv/bin/python -I /opt/phone-llm/verify_stack.py --runtime > /opt/serverless/base-environment-verification.json

COPY handler.py /opt/serverless/handler.py

# Persistent storage = the endpoint's network volume (RunPod mounts it at /runpod-volume on serverless workers).
# No secrets are baked in: HF_TOKEN and optional fetch credentials come from RunPod endpoint secrets / env vars.
ENV PLLM_VOLUME_ROOT=/runpod-volume \
    PLLM_TRAIN_PYTHON=/opt/phone-llm/venv/bin/python \
    PLLM_MAX_RUNTIME_S=13500 \
    HF_HOME=/runpod-volume/hf \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/serverless
# The inherited NVIDIA entrypoint execs this command; the base image's interactive /start.sh is not used on serverless.
CMD ["/opt/serverless/venv/bin/python", "-u", "/opt/serverless/handler.py"]
