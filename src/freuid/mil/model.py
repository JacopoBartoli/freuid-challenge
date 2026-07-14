"""Attention-based MIL per fraud detection su doc_masked (Ilse et al., 2018)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    """Gated attention MIL.

    Tratta ogni documento come una bag di N patch token DINOv2.
    Almeno un patch del campo alterato dovrebbe ricevere alta attenzione
    nei documenti fraud.

    Input:  patch tokens  [B, N, D]
    Output: fraud logit   [B],  attention weights [B, N]

    Args:
        in_dim:     dim dei patch token in input (768 per ViT-B/14)
        hidden_dim: dim della proiezione interna
        dropout:    dropout applicato dopo la proiezione
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Gated attention: V e U sono gate indipendenti (tanh e sigmoid)
        self.attn_V = nn.Linear(hidden_dim, hidden_dim)
        self.attn_U = nn.Linear(hidden_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, D]  patch token per un batch di documenti

        Returns:
            logits: [B]   fraud logit (prima di sigmoid)
            attn:   [B, N] attention weights normalizzati (somma a 1 per documento)
        """
        h = self.proj(x)                            # [B, N, H]
        V = torch.tanh(self.attn_V(h))              # [B, N, H]
        U = torch.sigmoid(self.attn_U(h))           # [B, N, H]
        A = self.attn_w(V * U)                      # [B, N, 1]
        A = F.softmax(A, dim=1)                     # [B, N, 1]  normalizzati
        z = (A * h).sum(dim=1)                      # [B, H]     bag embedding
        logits = self.classifier(z).squeeze(-1)     # [B]
        return logits, A.squeeze(-1)                # [B], [B, N]
