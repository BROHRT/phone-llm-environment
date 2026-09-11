# Build on a CPU-only Linux amd64 Docker host. No GPU or model weights required.
FROM runpod/base@sha256:9e8910f574d37264ab4662616466d50c46f19f6b700389c35f00323a1d65e170
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
COPY requirements.lock source-install-report.json expected.json verify_stack.py /opt/phone-llm/
RUN test "$(python3 -c 'import platform;print(platform.python_version())')" = 3.12.3 \
 && python3 -m venv /opt/phone-llm/venv \
 && /opt/phone-llm/venv/bin/python -m pip install --no-cache-dir --only-binary=:all: --require-hashes \
      --index-url https://pypi.org/simple --report /opt/phone-llm/pip-install-report.json \
      -r /opt/phone-llm/requirements.lock \
 && /opt/phone-llm/venv/bin/python -I /opt/phone-llm/verify_stack.py > /opt/phone-llm/build-verification.json
# Preserve inherited ENTRYPOINT, CMD, PATH, WORKDIR and /start.sh exactly.
# Runtime explicitly invokes /opt/phone-llm/venv/bin/python; no shell activation.
LABEL org.phone-llm.stack="python3.12.3-torch2.10.0-transformers5.3.0-peft0.18.1-accelerate1.12.0"
