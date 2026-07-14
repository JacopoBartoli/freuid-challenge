# FREUID Challenge 2026 — DA-MIL Document Fraud Detection

**Public leaderboard: FREUID 0.068**

Domain-adversarial multiple instance learning on DINOv2 patch tokens with document tiling.
Each image is resized to 672×896 and split into a 3×4 grid of 224×224 tiles; each tile is
processed by a frozen DINOv2 ViT-B/14 backbone, yielding 3072 tokens per document.
A gradient reversal layer suppresses template-specific features during training.

## Requirements

- Python ≥ 3.10, CUDA-capable GPU
- `pip install -e .` (or `uv sync`)

## Quick start

```bash
git clone <repo-url>
cd freuid-challenge
pip install -e .

# 1. Generate metadata.csv
python scripts/prepare_metadata.py --data-dir /path/to/freuid_data

# 2. Train (online DINOv2 extraction, no pre-caching needed)
python scripts/train_online.py --data-dir /path/to/freuid_data --epochs 25

# Best checkpoint → results/checkpoints/online/<run>/best.pt
```

## Checkpoint

The trained MIL head (epoch-7, FREUID 0.068) is included in the repo at
`checkpoints/epoch-7.pt` (1.7 MB — only the MIL head is saved; the DINOv2
backbone weights are frozen and not stored).

To run inference standalone (outside Docker):
```bash
python scripts/infer.py  # inside the container, or adapt paths for local use
```

## Docker inference (offline)

The Docker image bundles DINOv2 weights and the trained checkpoint.
Runs with `--network none`; maps `/data/` → images, `/submissions/` → output CSV.

```bash
# Build (copies DINOv2 weights from local HuggingFace cache)
bash docker_build.sh

# Run
docker run --gpus all --network none \
    -v /path/to/test/images:/data:ro \
    -v /path/to/output:/submissions \
    freuid-inference:latest

# Output: /path/to/output/submission.csv  (columns: id, label)
```

## Architecture

```
Image (672×896) → tile 3×4 → 12 tiles (224×224)
                                    ↓
                          DINOv2 ViT-B/14 (frozen)
                                    ↓
                    3072 patch tokens [768-dim] per document
                                    ↓
                          Gated Attention MIL
                         /                  \
                   Fraud Head          Domain Head (train only)
                  sigmoid(fc)     ← GRL ← softmax(fc, 5-class)
                       ↓
                  fraud score ∈ [0,1]
```

## Training flags

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | required | Path to FREUID data (with `metadata.csv`) |
| `--output-dir` | `results/checkpoints/online` | Checkpoint directory |
| `--epochs` | 25 | Training epochs |
| `--batch-size` | 32 | Documents per step |
| `--lr` | 1e-4 | Learning rate |
| `--max-lambda` | 1.0 | GRL max weight |
| `--hidden-dim` | 256 | MIL hidden dimension |
| `--seed` | 42 | Random seed |

## License

MIT — see [LICENSE](LICENSE).
