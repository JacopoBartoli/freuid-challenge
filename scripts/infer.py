"""Inference script per il container Docker della reproducibility.

Legge immagini da /data/, scrive fraud scores in /submissions/submission.csv.
Funziona offline (--network none): pesi DINOv2 e checkpoint inclusi nell'immagine.

Architettura: DINOv2 ViT-B/14 (frozen) + tiling 672x896→12 tile 224x224 + DomainAdversarialMIL.

Uso (dentro il container):
    python scripts/infer.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from freuid.mil.domain_adversarial import DomainAdversarialMIL
from freuid.models.backbones import DINOv2Backbone

BACKBONE_NAME = "vit_base_patch14_dinov2.lvd142m"
VIT_IMG_SIZE  = 224
TARGET_IMG_H  = 672
TARGET_IMG_W  = 896
PATCH_DIM     = 768
HIDDEN_DIM    = 256
DROPOUT       = 0.25
# Ogni documento = 12 tile × 256 token = 3072 token; 8 doc/batch = 96 tile backbone forward
BATCH_SIZE    = 8

DATA_DIR    = Path("/data")
OUTPUT_PATH = Path("/submissions/submission.csv")
CHECKPOINT  = Path("/model/checkpoint.pt")
HF_CACHE    = Path("/model/dinov2")

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

IMG_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def apply_tiling(x: torch.Tensor) -> torch.Tensor:
    """[C, H, W] → [N_tiles, C, 224, 224]  —  N_tiles = 12 per 672×896."""
    tiles = x.unfold(1, VIT_IMG_SIZE, VIT_IMG_SIZE).unfold(2, VIT_IMG_SIZE, VIT_IMG_SIZE)
    tiles = tiles.permute(1, 2, 0, 3, 4)
    return tiles.reshape(-1, 3, VIT_IMG_SIZE, VIT_IMG_SIZE)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_tiles = (TARGET_IMG_H // VIT_IMG_SIZE) * (TARGET_IMG_W // VIT_IMG_SIZE)
    print(f"Tiling: {TARGET_IMG_H}x{TARGET_IMG_W} → {n_tiles} tile da {VIT_IMG_SIZE}x{VIT_IMG_SIZE}")
    print(f"Token per documento: {n_tiles} x 256 = {n_tiles * 256}")

    # ── Modello ──────────────────────────────────────────────────────────────
    model = DomainAdversarialMIL(in_dim=PATCH_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    ckpt  = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    print(f"Checkpoint caricato (ep={ckpt.get('epoch', '?')})")

    # ── Backbone ─────────────────────────────────────────────────────────────
    os.environ["HF_HOME"] = str(HF_CACHE)
    backbone = DINOv2Backbone(
        model_name=BACKBONE_NAME, img_size=VIT_IMG_SIZE, cache_dir=str(HF_CACHE),
    ).to(device)
    backbone.eval()

    transform = transforms.Compose([
        transforms.Resize(
            (TARGET_IMG_H, TARGET_IMG_W),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    # ── Immagini ──────────────────────────────────────────────────────────────
    image_paths = sorted([
        p for p in DATA_DIR.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS
    ])
    print(f"Immagini trovate: {len(image_paths)}")

    # Resume: salta ID già processati
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        done = set(pd.read_csv(OUTPUT_PATH)["id"].astype(str))
        image_paths = [p for p in image_paths if p.stem not in done]
        print(f"Resume: {len(done)} già processati, {len(image_paths)} rimanenti")

    results: list[dict] = []

    for batch_start in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="infer"):
        batch_paths = image_paths[batch_start : batch_start + BATCH_SIZE]

        tiles_list: list[torch.Tensor] = []
        valid_paths: list[Path] = []
        for p in batch_paths:
            try:
                tiles_list.append(apply_tiling(transform(Image.open(p).convert("RGB"))))
                valid_paths.append(p)
            except Exception as e:
                print(f"WARN: {p.name} — {e}")
                results.append({"id": p.stem, "label": 0.5})

        if not tiles_list:
            continue

        batch_tiles = torch.stack(tiles_list)          # [B, N_tiles, C, H, W]
        B, T, C, H, W = batch_tiles.shape

        with torch.no_grad():
            flat_tiles  = batch_tiles.view(B * T, C, H, W).to(device)
            flat_tokens = backbone.encode_patches(flat_tiles)  # [B*T, 256, 768]
            tokens = flat_tokens.reshape(B, T * flat_tokens.shape[1], PATCH_DIM)
            logits, _, _ = model(tokens, lambda_=0.0)
            scores = torch.sigmoid(logits).cpu().numpy()

        for p, score in zip(valid_paths, scores):
            results.append({"id": p.stem, "label": float(score)})

        # Flush parziale ogni 100 batch (resume-friendly)
        if (batch_start // BATCH_SIZE + 1) % 100 == 0:
            _flush(results, OUTPUT_PATH)
            results = []

    _flush(results, OUTPUT_PATH)
    print(f"\nScritto: {OUTPUT_PATH}")
    df = pd.read_csv(OUTPUT_PATH)
    print(f"  Righe: {len(df)} | score medio: {df['label'].mean():.3f}")


def _flush(results: list[dict], path: Path) -> None:
    if not results:
        return
    df_new = pd.DataFrame(results)
    if path.exists():
        df_new = pd.concat([pd.read_csv(path), df_new], ignore_index=True)
    df_new.to_csv(path, index=False)


if __name__ == "__main__":
    main()
