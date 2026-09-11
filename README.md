# Pinned Linux amd64 environment

Private environment build repository. Contains only the immutable Runpod base reference, dependency wheel lock, package-install provenance, and environment build/check scripts. No model weights, data, evaluation cases, adapters, credentials, or training/scoring implementation.

Build via the manually dispatched GitHub Actions workflow. The job token publishes to GHCR with packages:write. Image verification checks Python3.12.3,55 locked distributions, imports of torch2.10.0/transformers5.3.0/peft0.18.1/accelerate1.12.0, and preinstalled SSH. It does not require or test a CUDA device. The published digest is pulled and checked again. Logs and receipts are retained as Actions artifacts.

Package visibility changes require the owner's explicit action. Publishing a private package does not prove anonymous pull access.
