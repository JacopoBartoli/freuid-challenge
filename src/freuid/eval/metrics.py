"""Metriche ufficiali FREUID Challenge 2026.

Convenzione universale in tutto il progetto:
  label 0 = bona-fide, label 1 = attacco
  score alto  => alta probabilità di frode (classificato come attacco se score >= τ)

Metriche primarie:
  AuDET  — area sotto curva DET (più bassa = meglio)
  APCER @ 1% BPCER — tasso di errore operativo

Metrica secondaria:
  EER — Equal Error Rate
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ── Core ──────────────────────────────────────────────────────────────────────

def bpcer_apcer_at_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcola BPCER e APCER a ogni soglia ricavata dagli score unici.

    Returns:
        thresholds: array di soglie in ordine crescente
        bpcer:      P(score >= τ | label=0)  — falsi positivi sui bona-fide
        apcer:      P(score <  τ | label=1)  — falsi negativi sugli attacchi
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    bona = scores[labels == 0]
    attack = scores[labels == 1]

    if len(bona) == 0:
        raise ValueError("Nessun sample bona-fide (label=0) nel set.")
    if len(attack) == 0:
        raise ValueError("Nessun sample attacco (label=1) nel set.")

    # Una soglia extra a +inf per avere il punto (BPCER=0, APCER=1)
    thresholds = np.concatenate([np.unique(scores), [np.inf]])

    bpcer = np.array([(bona >= t).mean() for t in thresholds])
    apcer = np.array([(attack < t).mean() for t in thresholds])

    return thresholds, bpcer, apcer


def apcer_at_bpcer(
    scores: np.ndarray,
    labels: np.ndarray,
    bpcer_target: float = 0.01,
) -> float:
    """APCER alla soglia che produce BPCER ≈ bpcer_target.

    Trova τ = quantile(bona, 1 - bpcer_target): la soglia che lascia fuori
    esattamente bpcer_target dei bona-fide. Poi misura quanti attacchi
    cadono sotto τ (non rilevati).

    Args:
        scores:       score di frode (alto = sospetto).
        labels:       0 = bona-fide, 1 = attacco.
        bpcer_target: tasso di falsi allarme target (default 1%).

    Returns:
        APCER in [0, 1]. Più basso = meglio.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    bona = scores[labels == 0]
    attack = scores[labels == 1]

    if len(bona) == 0:
        raise ValueError("Nessun sample bona-fide (label=0).")
    if len(attack) == 0:
        raise ValueError("Nessun sample attacco (label=1).")

    tau = np.quantile(bona, 1.0 - bpcer_target)
    return float((attack < tau).mean())


def find_threshold_at_bpcer(
    scores: np.ndarray,
    labels: np.ndarray,
    bpcer_target: float = 0.01,
) -> float:
    """Restituisce la soglia τ tale che BPCER ≤ bpcer_target."""
    scores = np.asarray(scores, dtype=float)
    bona = scores[np.asarray(labels, dtype=int) == 0]
    if len(bona) == 0:
        raise ValueError("Nessun sample bona-fide (label=0).")
    return float(np.quantile(bona, 1.0 - bpcer_target))


def audet(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area sotto la curva DET (scala lineare, integrazione trapezoidale).

    La curva DET traccia APCER (asse y) vs BPCER (asse x) al variare
    della soglia. L'area è calcolata ordinando per BPCER crescente.

    Returns:
        AuDET in [0, 1]. Più basso = meglio.
        0 = classificatore perfetto, ~0.5 = casuale, 1 = classificatore invertito.
    """
    _, bpcer, apcer = bpcer_apcer_at_thresholds(scores, labels)

    # Ordina per BPCER crescente, con APCER decrescente come tiebreak.
    # Il tiebreak è critico: a parità di BPCER, la curva DET deve scendere
    # lungo l'asse APCER (da alto a basso) prima di procedere su BPCER.
    # Senza di questo, i punti con BPCER=0 creano trapezi spuri con area non nulla.
    order = np.lexsort((-apcer, bpcer))
    bpcer_s = bpcer[order]
    apcer_s = apcer[order]

    return float(np.trapezoid(apcer_s, bpcer_s))


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Equal Error Rate: punto in cui BPCER ≈ APCER.

    Returns:
        EER in [0, 1]. Più basso = meglio.
    """
    _, bpcer, apcer = bpcer_apcer_at_thresholds(scores, labels)
    idx = np.argmin(np.abs(bpcer - apcer))
    return float((bpcer[idx] + apcer[idx]) / 2.0)


# ── Plot ──────────────────────────────────────────────────────────────────────

def det_curve_plot(
    scores: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path,
    title: str = "DET Curve",
) -> Path:
    """Salva la curva DET (BPCER vs APCER) come PNG.

    Args:
        scores:    score di frode.
        labels:    0 = bona-fide, 1 = attacco.
        save_path: path del file PNG di output.
        title:     titolo del grafico.

    Returns:
        Path del file salvato.
    """
    import matplotlib.pyplot as plt
    from scipy.special import ndtri  # probit transform

    _, bpcer, apcer = bpcer_apcer_at_thresholds(scores, labels)

    order = np.lexsort((-apcer, bpcer))
    bpcer_s = bpcer[order]
    apcer_s = apcer[order]

    _audet = float(np.trapezoid(apcer_s, bpcer_s))
    _eer = eer(scores, labels)
    _apcer1 = apcer_at_bpcer(scores, labels, bpcer_target=0.01)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scala lineare
    ax = axes[0]
    ax.plot(bpcer_s, apcer_s, lw=2, color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="EER line")
    ax.scatter([_eer], [_eer], color="red", zorder=5, label=f"EER={_eer:.3f}")
    ax.set_xlabel("BPCER (False Alarm Rate)")
    ax.set_ylabel("APCER (Miss Rate)")
    ax.set_title(f"{title} — linear\nAuDET={_audet:.4f}, APCER@1%={_apcer1:.3f}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Scala probit (standard DET)
    ax = axes[1]
    # Clip per evitare ndtri(0) e ndtri(1) = ±inf
    eps = 1e-4
    bpcer_clip = np.clip(bpcer_s, eps, 1 - eps)
    apcer_clip = np.clip(apcer_s, eps, 1 - eps)
    ax.plot(ndtri(bpcer_clip), ndtri(apcer_clip), lw=2, color="steelblue")
    ticks = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    tick_labels = ["0.1%", "1%", "5%", "10%", "20%", "50%"]
    tick_vals = ndtri(ticks)
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("BPCER")
    ax.set_ylabel("APCER")
    ax.set_title(f"{title} — probit scale")
    ax.grid(True, alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
