"""Domain Adversarial MIL per fraud detection cross-template.

Architettura: AttentionMIL + GradientReversalLayer + template classifier head.
Il bag embedding z viene forzato ad essere template-agnostico tramite gradient reversal.

Riferimento: Ganin et al., "Domain-Adversarial Training of Neural Networks" (2016).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TEMPLATES = ["BENIN/DL", "EGYPT/DL", "GUINEA/DL", "MAURITIUS/ID", "MOZAMBIQUE/DL"]
TEMPLATE_TO_IDX = {t: i for i, t in enumerate(TEMPLATES)}
N_TEMPLATES = len(TEMPLATES)


# ── Gradient Reversal ─────────────────────────────────────────────────────────

class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Identità in forward, moltiplica il gradiente per -λ in backward."""

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_)


def lambda_schedule(epoch: int, total_epochs: int, gamma: float = 10.0, max_lambda: float = 1.0) -> float:
    """Schedule di Ganin et al. 2016: parte da 0, sale verso max_lambda."""
    p = epoch / max(total_epochs, 1)
    return max_lambda * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


# ── Modello ───────────────────────────────────────────────────────────────────

class DomainAdversarialMIL(nn.Module):
    """AttentionMIL con testa template adversariale.

    Input:  patch tokens  [B, N, D]
    Output: fraud logit   [B]
            template logit [B, N_TEMPLATES]  (usato solo in training)
            attention weights [B, N]

    Args:
        in_dim:      dim dei patch token (768 per ViT-B/14)
        hidden_dim:  dim della proiezione interna
        dropout:     dropout applicato dopo la proiezione
        n_templates: numero di template da classificare (5 per FREUID)
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.25,
        n_templates: int = N_TEMPLATES,
    ) -> None:
        super().__init__()

        # ── Shared: proiezione + gated attention ──────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn_V = nn.Linear(hidden_dim, hidden_dim)
        self.attn_U = nn.Linear(hidden_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1, bias=False)

        # ── Fraud head ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # ── Domain adversarial head ───────────────────────────────────────────
        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_templates),
        )

    def forward(
        self,
        x: torch.Tensor,
        lambda_: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:       [B, N, D]
            lambda_: peso del gradient reversal (aggiornato ogni epoca)

        Returns:
            fraud_logit:    [B]
            template_logit: [B, n_templates]  (CrossEntropy input)
            attn:           [B, N]
        """
        h = self.proj(x)                             # [B, N, H]
        V = torch.tanh(self.attn_V(h))               # [B, N, H]
        U = torch.sigmoid(self.attn_U(h))            # [B, N, H]
        A = F.softmax(self.attn_w(V * U), dim=1)     # [B, N, 1]
        z = (A * h).sum(dim=1)                       # [B, H]  bag embedding

        fraud_logit = self.classifier(z).squeeze(-1) # [B]

        self.grl.lambda_ = lambda_
        z_rev = self.grl(z)                          # [B, H]  gradiente invertito
        template_logit = self.domain_classifier(z_rev)  # [B, n_templates]

        return fraud_logit, template_logit, A.squeeze(-1)
