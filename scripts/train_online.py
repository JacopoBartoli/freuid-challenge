"""Training DA-MIL con estrazione feature DINOv2 online (no cache su disco).

Pensato per ambienti senza HDD esterno (es. Google Colab).
Il backbone DINOv2 è frozen e gira sulla GPU insieme al modello MIL.

Setup:
    uv sync                                    # oppure: pip install -e ".[dev]"
    cp .env.example .env                       # e compilare FREUID_DATA_LOCAL
    uv run python scripts/train_online.py --data-dir /path/to/freuid/data

Su Colab:
    !pip install -e ".[dev]" -q
    !python scripts/train_online.py --data-dir /content/drive/MyDrive/freuid_data

Flags principali:
    --data-dir      path alla cartella con metadata.csv e le immagini
    --output-dir    dove salvare i checkpoint (default: results/checkpoints/online)
    --epochs        numero di epoche (default 25)
    --batch-size    documenti per step (default 32)
    --whitening     sottrae la media spaziale delle patch (template-agnostic)
    --max-lambda    peso massimo del gradient reversal (default 1.0)
"""
from __future__ import annotations

import argparse
from datetime import datetime
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from freuid.eval.metrics import apcer_at_bpcer, audet, eer
from freuid.mil.domain_adversarial import (
    DomainAdversarialMIL,
    TEMPLATE_TO_IDX,
    lambda_schedule,
)
from freuid.models.backbones import DINOv2Backbone

BACKBONE_NAME = "vit_base_patch14_dinov2.lvd142m"
PATCH_DIM = 768
# TARGET_IMG_H = 896
# TARGET_IMG_W = 1344
TARGET_IMG_H = 672
TARGET_IMG_W = 896
IMG_SIZE = (TARGET_IMG_H, TARGET_IMG_W)
VIT_IMG_SIZE = 224

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class EmbeddingCache:
    def __init__(self, path_root: Path, cache_limit: int = 15000):
        self._path_cache = path_root

        self._path_cache.mkdir(parents=True, exist_ok=True)
        self._cache_limit = cache_limit
        self._n_files_in_cache = len(list(self._path_cache.glob("*.pt")))

    def get_sample_embs(self, sample_id: str) -> torch.Tensor | None:
        path_cached_embs = self._path_cache / f"{sample_id}.pt"
        if path_cached_embs.exists():
            return torch.load(path_cached_embs)
        
        return None
    
    def register_embs(self, embs: torch.Tensor, sample_id) -> None:
        if self._n_files_in_cache < self._cache_limit:
            torch.save(embs.cpu(), self._path_cache / f"{sample_id}.pt")
            self._n_files_in_cache += 1


# ── Dataset ──────────────────────────────────────────────────────────────────

class DocumentDataset(Dataset):
    """Legge immagini da disco e le restituisce come tensori normalizzati."""

    def __init__(self, df: pd.DataFrame, img_size: tuple[int, int] = IMG_SIZE) -> None:
        self.df = df.reset_index(drop=True)
        self._target_img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize(img_size,
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img = Image.open(str(row["image_path"])).convert("RGB")

        img_transformed = self.transform(img)
        img_transformed = self._apply_tiling(img_transformed)

        tmpl  = str(row.get("document_template", ""))
        return {
            "sample_id": Path(row["image_path"]).stem,
            "image":        img_transformed,
            "label":        int(row["label"]),
            "template_idx": TEMPLATE_TO_IDX.get(tmpl, -1),
        }
    
    def _apply_tiling(self, x: torch.Tensor) -> torch.Tensor:
        rgb_channels = 3

        # Define your fixed tile sizes
        tile_h, tile_w = VIT_IMG_SIZE, VIT_IMG_SIZE

        # 1. Unfold height (dim 1) and width (dim 2)
        # Output shape: (C, num_tiles_h, num_tiles_w, tile_h, tile_w)
        tiles = x.unfold(1, tile_h, tile_h).unfold(2, tile_w, tile_w)

        # 2. Permute to bring tile counts to the front
        # Output shape: (num_tiles_h, num_tiles_w, C, tile_h, tile_w)
        tiles = tiles.permute(1, 2, 0, 3, 4)

        # 3. Collapse the tile grid dimensions into a single patch dimension P
        # Output shape: (P, C, tile_h, tile_w)
        patches = tiles.reshape(-1, rgb_channels, tile_h, tile_w)

        return patches


# ── Metrica ───────────────────────────────────────────────────────────────────

def freuid_score(audet_val: float, apcer1_val: float) -> float:
    g_a, g_p = 1.0 - audet_val, 1.0 - apcer1_val
    d = g_a + g_p
    return 1.0 - 2.0 * g_a * g_p / d if d > 0 else 1.0


# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(backbone, model, loader, device, cache: EmbeddingCache, whitening: bool) -> dict:
    backbone.eval(); model.eval()
    all_scores, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)

        # b,t,c,h,w  = images.shape

        # with torch.no_grad():
        #     # Remove tile dimension and encode single tiles
        #     batch_tiles = torch.reshape(images, shape=(-1, c, h, w))
        #     sample_embs = backbone.encode_patches(batch_tiles)      # backbone frozen    

        # # Reshape back to dimension with tiles
        # sample_embs = torch.reshape(sample_embs, shape=(b, t, *sample_embs.shape[-2:]))
        # # Remove tile dimension and group together embeddings of the same image
        # sample_embs = torch.reshape(sample_embs, shape=(b, -1, sample_embs.shape[-1]))

        # cache.register_embs(sample_embs, sample_id)

        sample_ids = batch["sample_id"]
        list_batch_embs = []
        for index_in_batch in range(len(sample_ids)):
            sample_id = sample_ids[index_in_batch]
            sample_embs = cache.get_sample_embs(sample_id)
            if sample_embs is None:
                with torch.no_grad():
                    # Remove tile dimension and encode single tiles
                    # batch_tiles = torch.reshape(images, shape=(-1, c, h, w))
                    sample_image = images[index_in_batch]
                    sample_embs = backbone.encode_patches(sample_image)      # backbone frozen    

                # Remove tile dimension and group together embeddings of the same image
                sample_embs = torch.reshape(sample_embs, shape=(-1, sample_embs.shape[-1]))

                cache.register_embs(sample_embs, sample_id)
            else:
                sample_embs = sample_embs.to(device)
            
            list_batch_embs.append(sample_embs)

        batch_embs = torch.stack(list_batch_embs)
        
        if whitening:
            batch_embs = batch_embs - batch_embs.mean(dim=1, keepdim=True)
        fraud_logit, _, _ = model(batch_embs, lambda_=0.0)   # domain head off
        all_scores.extend(torch.sigmoid(fraud_logit).cpu().tolist())
        all_labels.extend(batch["label"].tolist())

    s, l = np.array(all_scores), np.array(all_labels)
    a = audet(s, l)
    p = apcer_at_bpcer(s, l, 0.01)
    return {"audet": a, "eer": eer(s, l), "apcer1": p, "freuid": freuid_score(a, p)}


# ── Training epoch ────────────────────────────────────────────────────────────

def train_epoch(
    backbone, model, loader, optimizer, device,
    pos_weight: float, lambda_: float,
    epoch: int, total_epochs: int,
    cache: EmbeddingCache,
    whitening: bool = False,
) -> tuple[float, float]:
    model.train()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    ce  = nn.CrossEntropyLoss(ignore_index=-1)

    total_fraud = total_domain = n_seen = n_batches = 0
    t0 = time.time()

    for batch in loader:
        images: torch.Tensor   = batch["image"].to(device)
        labels   = batch["label"].float().to(device)
        tmpl_idx = batch["template_idx"].to(device)

        # b,t,c,h,w  = images.shape

        # with torch.no_grad():
        #     # Remove tile dimension and encode single tiles
        #     batch_tiles = torch.reshape(images, shape=(-1, c, h, w))
        #     sample_embs = backbone.encode_patches(batch_tiles)      # backbone frozen    

        # # Reshape back to dimension with tiles
        # sample_embs = torch.reshape(sample_embs, shape=(b, t, *sample_embs.shape[-2:]))
        # # Remove tile dimension and group together embeddings of the same image
        # sample_embs = torch.reshape(sample_embs, shape=(b, -1, sample_embs.shape[-1]))

        # cache.register_embs(sample_embs, sample_id)

        sample_ids = batch["sample_id"]
        list_batch_embs = []
        for index_in_batch in range(len(sample_ids)):
            sample_id = sample_ids[index_in_batch]
            sample_embs = cache.get_sample_embs(sample_id)
            if sample_embs is None:
                with torch.no_grad():
                    # Remove tile dimension and encode single tiles
                    # batch_tiles = torch.reshape(images, shape=(-1, c, h, w))
                    sample_image = images[index_in_batch]
                    sample_embs = backbone.encode_patches(sample_image)      # backbone frozen    

                # Remove tile dimension and group together embeddings of the same image
                sample_embs = torch.reshape(sample_embs, shape=(-1, sample_embs.shape[-1]))

                cache.register_embs(sample_embs, sample_id)
            else:
                sample_embs = sample_embs.to(device)
            
            list_batch_embs.append(sample_embs)

        batch_embs = torch.stack(list_batch_embs)

        if whitening:
            batch_embs = batch_embs - batch_embs.mean(dim=1, keepdim=True)

        fraud_logit, template_logit, _ = model(batch_embs, lambda_=lambda_)

        loss_fraud  = bce(fraud_logit, labels)
        loss_domain = ce(template_logit, tmpl_idx)
        loss        = loss_fraud + loss_domain

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = len(labels)
        total_fraud  += loss_fraud.item()  * bs
        total_domain += loss_domain.item() * bs
        n_seen       += bs
        n_batches    += 1

        if n_batches % 20 == 0:
            elapsed = time.time() - t0
            it_s    = n_batches / elapsed
            eta_min = (len(loader) - n_batches) / it_s / 60
            print(
                f"    ep {epoch:02d}/{total_epochs} [{n_seen:6d}/{len(loader.dataset)}] "
                f"fraud={total_fraud/n_seen:.4f} domain={total_domain/n_seen:.4f} "
                f"λ={lambda_:.3f} | {it_s:.1f} batch/s | ETA: {eta_min:.1f} min",
                flush=True,
            )

    return total_fraud / max(n_seen, 1), total_domain / max(n_seen, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def load_df(data_dir: Path) -> pd.DataFrame:
    csv = data_dir / "metadata.csv"
    if not csv.exists():
        raise FileNotFoundError(f"metadata.csv non trovato in {data_dir}")
    df = pd.read_csv(csv)
    df = df[
        (df["split"] == "train")
        & (df["file_exists"] == True)          # noqa: E712
        & (df["attack_type"] != "print_and_capture")
    ].copy().reset_index(drop=True)
    df["global_idx"] = df.index
    df["label"] = df["label"].astype(int)
    # Risolve i path relativi rispetto a data_dir
    if not Path(str(df["image_path"].iloc[0])).is_absolute():
        df["image_path"] = df["image_path"].apply(lambda p: str(data_dir / p))
    return df


def make_protocol_a(df: pd.DataFrame, val_frac: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    val_indices: list[int] = []
    for _, group in df.groupby("label"):
        n_val = max(1, round(len(group) * val_frac))
        val_indices.extend(rng.choice(group["global_idx"].values, n_val, replace=False))
    val_set    = set(val_indices)
    train_mask = ~df["global_idx"].isin(val_set)
    val_mask   =  df["global_idx"].isin(val_set)
    return df[train_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)


def main() -> None:
    path_default_out_dir = Path("results") / datetime.strftime(datetime.now(), format="%Y%m%d-%H%M%S") / "online"

    parser = argparse.ArgumentParser(
        description="DA-MIL training con estrazione DINOv2 online (no cache)"
    )
    parser.add_argument("--data-dir",    required=True,
                        help="Cartella con metadata.csv e le immagini di training")
    parser.add_argument("--output-dir",  default=str(path_default_out_dir),
                        help="Dove salvare i checkpoint")
    parser.add_argument("--epochs",      type=int,   default=25)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--hidden-dim",  type=int,   default=256)
    parser.add_argument("--dropout",     type=float, default=0.25)
    parser.add_argument("--num-workers", type=int,   default=2)
    parser.add_argument("--whitening",   action="store_true")
    parser.add_argument("--max-lambda",  type=float, default=1.0)
    parser.add_argument("--gamma",       type=float, default=10.0)
    parser.add_argument("--val-frac",    type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--hf-cache",    default=None,
                        help="Cartella locale per i pesi DINOv2 (HuggingFace cache)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Dati ────────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    df       = load_df(data_dir)
    train_df, val_df = make_protocol_a(df, args.val_frac, args.seed)

    n_bf  = int((train_df["label"] == 0).sum())
    n_atk = int((train_df["label"] == 1).sum())
    pos_weight = n_bf / n_atk
    print(f"Dataset: {len(df)} immagini")
    print(f"Train: {n_bf} BF + {n_atk} ATK | val: {len(val_df)} | pos_weight={pos_weight:.2f}")
    print(f"Whitening: {args.whitening} | max_lambda: {args.max_lambda}")

    train_ds = DocumentDataset(train_df)
    val_ds   = DocumentDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    train_cache = EmbeddingCache(Path(f"data/cache/vit-{VIT_IMG_SIZE}x{VIT_IMG_SIZE}-img-{TARGET_IMG_H}x{TARGET_IMG_W}/train"))
    val_cache = EmbeddingCache(Path(f"data/cache/vit-{VIT_IMG_SIZE}x{VIT_IMG_SIZE}-img-{TARGET_IMG_H}x{TARGET_IMG_W}/val"), cache_limit=0)

    # ── Backbone (frozen) ────────────────────────────────────────────────────
    hf_cache = args.hf_cache
    if hf_cache is None and "HF_HOME" in os.environ:
        hf_cache = os.environ["HF_HOME"]
    backbone = DINOv2Backbone(
        model_name=BACKBONE_NAME, img_size=VIT_IMG_SIZE, cache_dir=hf_cache,
    ).to(device)
    print(f"Backbone: {BACKBONE_NAME} @ {VIT_IMG_SIZE}px (frozen)")

    # ── Modello ──────────────────────────────────────────────────────────────
    model = DomainAdversarialMIL(
        in_dim=PATCH_DIM, hidden_dim=args.hidden_dim, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DA-MIL: {n_params/1e3:.1f}k parametri")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint dir: {ckpt_dir}")

    # ── Training loop ────────────────────────────────────────────────────────
    best_freuid = float("inf")
    best_epoch  = 0

    for epoch in range(1, args.epochs + 1):
        lam = lambda_schedule(epoch, args.epochs, gamma=args.gamma, max_lambda=args.max_lambda)

        loss_f, loss_d = train_epoch(
            backbone, model, train_loader, optimizer, device,
            pos_weight, lam, epoch, args.epochs, cache=train_cache,
            whitening=args.whitening,
        )
        metrics = evaluate(backbone, model, val_loader, device, val_cache, args.whitening)
        scheduler.step()

        marker = ""
        if metrics["freuid"] < best_freuid:
            best_freuid = metrics["freuid"]
            best_epoch  = epoch
            torch.save({
                "model":   model.state_dict(),
                "epoch":   epoch,
                "metrics": metrics,
                "args":    vars(args),
            }, ckpt_dir / "best.pt")
            marker = " *"

        print(
            f"  ep {epoch:02d}/{args.epochs} | "
            f"loss_fraud={loss_f:.4f} loss_domain={loss_d:.4f} | "
            f"FREUID={metrics['freuid']:.4f} | "
            f"AuDET={metrics['audet']:.4f} | APCER@1%={metrics['apcer1']:.3f} | "
            f"λ={lam:.3f}{marker}"
        )

    print(f"\nBest: epoch {best_epoch}, FREUID={best_freuid:.4f}")
    print(f"Checkpoint salvato in: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
