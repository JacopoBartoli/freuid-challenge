"""Backbone congelati per l'estrazione di feature.

Questi modelli NON vengono mai aggiornati durante il training.
Le loro feature vengono estratte una volta e cachate su disco
(vedi src/freuid/utils/feature_extractor.py e freq_extractor.py).

Durante il training la GPU non carica il backbone se la cache è già completa.

Backbone disponibili
--------------------
CLIPBackbone   — CLIP ViT-L/14 (openai), output 768-dim CLS token
DINOv2Backbone — DINOv2 ViT-L/14 (facebook), output 1024-dim CLS token
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class CLIPBackbone(nn.Module):
    """CLIP ViT-L/14 frozen. Usato solo per estrarre il CLS token (768-dim).

    Args:
        model_name:  nome modello open_clip (default "ViT-L-14").
        pretrained:  checkpoint openai o altro (default "openai").
        cache_dir:   cartella locale per i pesi scaricati.

    Esempio::

        backbone = CLIPBackbone()
        feats = backbone.encode(images)  # [B, 768]
    """

    OUT_DIM = 768

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        import open_clip

        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            cache_dir=cache_dir,
            quick_gelu=True,
        )
        self.clip = clip_model
        self.model_name = model_name
        self.pretrained = pretrained

        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.clip.eval()

    def train(self, mode: bool = True) -> "CLIPBackbone":
        # Backbone sempre in eval, mai in training mode
        super().train(mode)
        self.clip.eval()
        return self

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Estrae il CLS token per un batch di immagini.

        Args:
            x: immagini normalizzate con mean/std CLIP, shape [B, 3, 224, 224].

        Returns:
            Feature tensor [B, 768].
        """
        return self.clip.encode_image(x, normalize=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class DINOv2Backbone(nn.Module):
    """DINOv2 ViT-L/14 frozen. Output: CLS token (1024-dim).

    Caricato tramite timm (pretrained=True scarica da HuggingFace Hub).
    Usa ImageNet normalization — diversa da CLIP: vedi dinov2_eval_transforms().

    Args:
        model_name: nome timm (default "vit_large_patch14_dinov2").
        img_size:   risoluzione di input. 518 = nativa DINOv2, 224 = veloce.
        cache_dir:  cartella HuggingFace Hub locale per i pesi.

    Esempio::

        backbone = DINOv2Backbone()
        feats = backbone.encode(images)  # [B, 1024]
    """

    OUT_DIM = 1024

    def __init__(
        self,
        model_name: str = "vit_large_patch14_dinov2",
        img_size: int = 518,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        import os
        import timm

        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(cache_dir))

        self.dino = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            img_size=img_size,
        )
        self.model_name = model_name
        self.img_size = img_size

        for p in self.dino.parameters():
            p.requires_grad_(False)
        self.dino.eval()

    def train(self, mode: bool = True) -> "DINOv2Backbone":
        super().train(mode)
        self.dino.eval()
        return self

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """CLS token DINOv2: [B, D]."""
        return self.dino(x)

    @torch.no_grad()
    def encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Patch tokens DINOv2 (no CLS): [B, N_patches, D].

        Per ViT-B/14 a 224px: [B, 256, 768].
        Usa dino.forward_features che ritorna [B, N+1, D] con CLS all'indice 0.
        """
        features = self.dino.forward_features(x)   # [B, N+1, D]
        return features[:, 1:, :]                   # [B, N, D] — salta CLS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)
