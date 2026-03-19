"""Continual learning evaluation metrics (paper Section 4.1).

Success matrix S ∈ ℝ^{(T+1) × T}:
  S[i, j] = success rate on task j after training through task i.
  S[0, j] = base model zero-shot success on task j.

Four metrics:
  AVG  = (1/T)   Σ_j  S[T, j]                     — final average
  NBT  = (1/(T-1)) Σ_{j<T} (S[j,j] - S[T,j])      — forgetting (lower=better)
  FWT  = (1/(T-1)) Σ_{j>0} (S[j-1,j] - S[0,j])    — forward transfer
  ZS   = (1/|H|)  Σ_{h∈H} S^held[T, h]             — held-out zero-shot
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class MetricsResult:
    """Container for all CRL evaluation metrics."""

    avg: float         # Final average success over training tasks
    nbt: float         # Negative backward transfer (forgetting)
    fwt: float         # Forward transfer
    zs: float          # Zero-shot on held-out tasks

    # Standard errors (over seeds)
    avg_se: float = 0.0
    nbt_se: float = 0.0
    fwt_se: float = 0.0
    zs_se: float = 0.0

    def __str__(self) -> str:
        return (
            f"AVG={self.avg:.3f}±{self.avg_se:.3f} | "
            f"NBT={self.nbt:.3f}±{self.nbt_se:.3f} | "
            f"FWT={self.fwt:.3f}±{self.fwt_se:.3f} | "
            f"ZS={self.zs:.3f}±{self.zs_se:.3f}"
        )

    def to_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


def compute_metrics(
    success_matrix: np.ndarray,
    held_out_success: Optional[np.ndarray] = None,
) -> MetricsResult:
    """Compute AVG, NBT, FWT, ZS from the success matrix.

    Args:
        success_matrix: shape (T+1, T) where row 0 = zero-shot baseline.
                        S[i, j] = success on task j after training tasks 0..i-1.
        held_out_success: shape (|H|,) — success rates on held-out tasks
                          evaluated after all training. None → ZS=0.

    Returns:
        MetricsResult with all four metrics.
    """
    S = np.array(success_matrix, dtype=np.float64)
    n_rows, T = S.shape

    if n_rows != T + 1:
        raise ValueError(
            f"success_matrix must have shape (T+1, T), got {S.shape}. "
            f"Row 0 = zero-shot, rows 1..T = after training task i."
        )

    S_final = S[T, :]   # final row: success after all training
    S_0 = S[0, :]       # row 0: zero-shot / base model

    # ── AVG: final average over all training tasks ─────────────────────────
    avg = float(S_final.mean())

    # ── NBT: forgetting (paper eq. 4) ─────────────────────────────────────
    # After training task j (row j+1 of S, since row 0 = zero-shot),
    # S[j+1, j] = "best achievable on task j" ≈ peak performance.
    # NBT = (1/(T-1)) Σ_{j=0}^{T-2} (S[j+1, j] - S[T, j])
    if T > 1:
        nbt_terms = np.array(
            [S[j + 1, j] - S[T, j] for j in range(T - 1)]
        )
        nbt = float(nbt_terms.mean())
    else:
        nbt = 0.0

    # ── FWT: forward transfer (paper eq. 5) ───────────────────────────────
    # FWT = (1/(T-1)) Σ_{j=1}^{T-1} (S[j, j] - S[0, j])
    # S[j, j] = zero-shot on task j right before we train it (after j-1 tasks)
    # We approximate with S[j, j] = success on task j at row j (pre-training)
    if T > 1:
        fwt_terms = np.array(
            [S[j, j] - S_0[j] for j in range(1, T)]
        )
        fwt = float(fwt_terms.mean())
    else:
        fwt = 0.0

    # ── ZS: held-out zero-shot ─────────────────────────────────────────────
    if held_out_success is not None and len(held_out_success) > 0:
        zs = float(np.mean(held_out_success))
    else:
        zs = 0.0

    return MetricsResult(avg=avg, nbt=nbt, fwt=fwt, zs=zs)


def aggregate_metrics_over_seeds(
    results: list[MetricsResult],
) -> MetricsResult:
    """Compute mean ± standard error over multiple seed runs."""
    avgs = np.array([r.avg for r in results])
    nbts = np.array([r.nbt for r in results])
    fwts = np.array([r.fwt for r in results])
    zss = np.array([r.zs for r in results])
    n = len(results)

    def se(x: np.ndarray) -> float:
        return float(x.std() / np.sqrt(n)) if n > 1 else 0.0

    return MetricsResult(
        avg=float(avgs.mean()),
        nbt=float(nbts.mean()),
        fwt=float(fwts.mean()),
        zs=float(zss.mean()),
        avg_se=se(avgs),
        nbt_se=se(nbts),
        fwt_se=se(fwts),
        zs_se=se(zss),
    )


def print_success_matrix(
    success_matrix: np.ndarray,
    task_ids: list[str],
    title: str = "Success Matrix",
) -> None:
    """Pretty-print the success matrix S."""
    S = np.array(success_matrix)
    T = len(task_ids)
    col_w = max(max(len(t) for t in task_ids), 8)
    header = f"{'After →':>12} | " + " | ".join(f"{t:>{col_w}}" for t in task_ids)
    sep = "-" * len(header)
    print(f"\n{title}\n{sep}")
    print(header)
    print(sep)
    row_labels = ["ZeroShot"] + [f"Task {i+1}" for i in range(T)]
    for i, row_lbl in enumerate(row_labels):
        row_str = " | ".join(f"{S[i,j]*100:>{col_w}.1f}%" for j in range(T))
        print(f"{row_lbl:>12} | {row_str}")
    print(sep)
