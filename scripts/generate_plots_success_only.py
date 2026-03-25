"""Generate comparison plots: R1 vs R2 vs Success-Only SFT + GRPO T1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path("docs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

TASKS = ["banana", "gelatin_box", "mug", "cup"]
HELD_OUT = ["bowl", "marker", "foam_brick", "tuna_can"]

# ── Load data from JSON files ─────────────────────────────────────────────────

def load_sm(path):
    with open(path) as f:
        d = json.load(f)
    return {
        "matrix": np.array([[x if x is not None and str(x) != "NaN" else np.nan for x in row] for row in d["matrix"]]),
        "held_out": np.array([x if x is not None and str(x) != "NaN" else np.nan for x in d["held_out"]]),
        "held_out_zero_shot": np.array([x if x is not None and str(x) != "NaN" else np.nan for x in d.get("held_out_zero_shot", [np.nan]*4)]),
    }

# Previous experiments
R1 = load_sm("results/continual_rl/seed_0/success_matrix_final.json")
R2 = load_sm("results_round2/continual_rl/seed_0/success_matrix_final.json")

# Success-only SFT
SO_SFT = load_sm("results_success_only/sft/seed_0/success_matrix_final.json")

# Success-only GRPO partial (zero-shot + after T1)
SO_GRPO_PARTIAL = load_sm("results_success_only/continual_rl/seed_0/success_matrix_partial.json")

# Success-only GRPO T1 full eval (training + held-out)
EVAL_PATH = Path("results_success_only/continual_rl/seed_0/eval_after_grpo_t1.json")
SO_GRPO_T1_EVAL = load_sm(EVAL_PATH) if EVAL_PATH.exists() else None

# GRPO training curves for success-only (from log)
SO_CURVES_BANANA = [0.479, 0.417, 0.458, 0.490, 0.417, 0.427, 0.385, 0.438, 0.417, 0.448]

R1_CURVES = {
    "banana":      [0.479, 0.417, 0.458, 0.490, 0.417, 0.427, 0.385, 0.438, 0.417, 0.448],
    "gelatin_box": [0.448, 0.479, 0.438, 0.406, 0.448, 0.448, 0.500, 0.500, 0.469, 0.417],
    "mug":         [0.281, 0.333, 0.365, 0.333, 0.365, 0.375, 0.396, 0.260, 0.365, 0.344],
    "cup":         [0.177, 0.188, 0.208, 0.177, 0.115, 0.177, 0.198, 0.125, 0.156, 0.188],
}

COLORS = {
    "R1": "#457b9d",
    "R2": "#e76f51",
    "SO_SFT": "#2a9d8f",
    "SO_GRPO": "#9b2226",
}

# ── Plot 1: SFT comparison — regular vs success-only ─────────────────────────

def plot_sft_comparison():
    """Compare regular SFT vs success-only SFT final performance."""
    # Regular SFT from R1 zero-shot row (they share same SFT)
    regular_sft_final = R1["matrix"][0]  # zero-shot = before GRPO = after SFT

    # Success-only SFT final row (after all 4 tasks)
    so_sft_final = SO_SFT["matrix"][4]
    so_sft_zs = SO_SFT["matrix"][0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: full SFT trajectory comparison
    ax = axes[0]
    stages = ["Zero-shot", "After T1", "After T2", "After T3", "After T4"]
    for j, task in enumerate(TASKS):
        so_traj = SO_SFT["matrix"][:, j] * 100
        ax.plot(range(5), so_traj, "o-", label=f"{task} (success-only)", linewidth=2, markersize=5)

    ax.set_xticks(range(5))
    ax.set_xticklabels(stages, rotation=20, ha="right")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 60)
    ax.set_title("Success-Only SFT — Task Trajectory", fontweight="bold")
    ax.legend(fontsize=8)

    # Right: final performance bar comparison
    ax = axes[1]
    x = np.arange(4)
    w = 0.35

    # Regular SFT final = R1 zero-shot (same starting checkpoint for GRPO R1)
    regular_sft_row = np.array([0.365, 0.471, 0.337, 0.212])  # from blog SFT table row 4
    ax.bar(x - w/2, regular_sft_row * 100, w, label="Regular SFT (all demos)", color="#a8dadc", edgecolor="white")
    ax.bar(x + w/2, so_sft_final * 100, w, label="Success-Only SFT", color=COLORS["SO_SFT"], edgecolor="white")

    for i in range(4):
        delta = (so_sft_final[i] - regular_sft_row[i]) * 100
        ax.text(i + w/2, so_sft_final[i] * 100 + 0.5, f"{delta:+.1f}pp",
                ha="center", fontsize=8, color="green" if delta >= 0 else "red")

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, rotation=15, ha="right")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 62)
    ax.set_title("Final SFT Performance: Regular vs Success-Only", fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("SFT Comparison — Regular Demos vs Success-Only Demos", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sft_comparison_success_only.png", dpi=150)
    plt.close(fig)
    print("  saved sft_comparison_success_only.png")


# ── Plot 2: GRPO T1 comparison across all methodologies ──────────────────────

def plot_grpo_t1_comparison():
    """After-task-1 GRPO performance: R1, R2, success-only."""
    fig, ax = plt.subplots(figsize=(10, 5))

    r1_after_t1 = R1["matrix"][1] * 100
    r2_after_t1 = R2["matrix"][1] * 100
    so_after_t1 = SO_GRPO_PARTIAL["matrix"][1] * 100

    x = np.arange(4)
    w = 0.25

    ax.bar(x - w, r1_after_t1, w, label="R1: regular SFT → GRPO T1", color=COLORS["R1"], edgecolor="white")
    ax.bar(x, r2_after_t1, w, label="R2: R1 checkpoint → GRPO T1", color=COLORS["R2"], edgecolor="white")
    ax.bar(x + w, so_after_t1, w, label="Success-Only SFT → GRPO T1", color=COLORS["SO_GRPO"], edgecolor="white")

    for i in range(4):
        for offset, vals in [(-w, r1_after_t1), (0, r2_after_t1), (w, so_after_t1)]:
            if not np.isnan(vals[i]):
                ax.text(i + offset, vals[i] + 0.5, f"{vals[i]:.1f}", ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, rotation=15, ha="right")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 62)
    ax.set_title("After GRPO Task 1 (put_banana_on_plate): All Methods", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "grpo_t1_comparison_success_only.png", dpi=150)
    plt.close(fig)
    print("  saved grpo_t1_comparison_success_only.png")


# ── Plot 3: Banana training curve comparison ──────────────────────────────────

def plot_banana_curve_comparison():
    """Banana GRPO training curve: R1 vs success-only (same 10 updates)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    updates = list(range(1, 11))

    ax.plot(updates, [v * 100 for v in R1_CURVES["banana"]], "o-",
            color=COLORS["R1"], linewidth=2, markersize=5, label="R1 (regular SFT start, rollout_epochs=4)")
    ax.plot(updates, [v * 100 for v in SO_CURVES_BANANA], "s--",
            color=COLORS["SO_GRPO"], linewidth=2, markersize=5, label="Success-Only (rollout_epochs=16)")

    # Zero-shot baselines
    ax.axhline(R1["matrix"][0, 0] * 100, color=COLORS["R1"], linestyle=":", alpha=0.5,
               label=f"R1 zero-shot ({R1['matrix'][0,0]*100:.1f}%)")
    ax.axhline(SO_GRPO_PARTIAL["matrix"][0, 0] * 100, color=COLORS["SO_GRPO"], linestyle=":", alpha=0.5,
               label=f"SO zero-shot ({SO_GRPO_PARTIAL['matrix'][0,0]*100:.1f}%)")

    ax.set_xlabel("GRPO Update")
    ax.set_ylabel("Rollout Success Rate (%)")
    ax.set_ylim(0, 65)
    ax.set_title("Banana GRPO Training Curve: R1 vs Success-Only", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "banana_curve_comparison_success_only.png", dpi=150)
    plt.close(fig)
    print("  saved banana_curve_comparison_success_only.png")


# ── Plot 4: Full eval comparison (if eval_after_grpo_t1.json exists) ─────────

def plot_full_eval_comparison():
    """Full training + held-out comparison after GRPO T1 eval."""
    if SO_GRPO_T1_EVAL is None:
        print("  skipping full_eval_comparison (eval JSON not yet available)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training tasks
    ax = axes[0]
    x = np.arange(4)
    w = 0.25

    r1_t1 = R1["matrix"][1] * 100
    r2_t1 = R2["matrix"][1] * 100
    so_t1 = SO_GRPO_T1_EVAL["matrix"][1] * 100  # row 1 = after task 1

    ax.bar(x - w, r1_t1, w, label="R1 after banana", color=COLORS["R1"], edgecolor="white")
    ax.bar(x, r2_t1, w, label="R2 after banana", color=COLORS["R2"], edgecolor="white")
    ax.bar(x + w, so_t1, w, label="Success-Only after banana", color=COLORS["SO_GRPO"], edgecolor="white")

    for i in range(4):
        for offset, vals in [(-w, r1_t1), (0, r2_t1), (w, so_t1)]:
            if not np.isnan(vals[i]):
                ax.text(i + offset, vals[i] + 0.5, f"{vals[i]:.1f}", ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, rotation=15, ha="right")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 65)
    ax.set_title("Training Tasks — After GRPO T1", fontweight="bold")
    ax.legend(fontsize=9)

    # Held-out tasks
    ax = axes[1]
    x = np.arange(4)

    r1_held_zs = R1["held_out_zero_shot"] * 100
    so_held_zs = SO_GRPO_PARTIAL["held_out_zero_shot"] * 100
    so_held_t1 = SO_GRPO_T1_EVAL["held_out"] * 100

    ax.bar(x - w, r1_held_zs, w, label="R1 zero-shot held-out", color="#a8dadc", edgecolor="white")
    ax.bar(x, so_held_zs, w, label="Success-Only zero-shot held-out", color="#84a98c", edgecolor="white")
    ax.bar(x + w, so_held_t1, w, label="Success-Only after GRPO T1", color=COLORS["SO_GRPO"], edgecolor="white")

    for i in range(4):
        if not np.isnan(so_held_t1[i]):
            delta = so_held_t1[i] - so_held_zs[i]
            ax.text(i + w, so_held_t1[i] + 0.5, f"{delta:+.1f}pp", ha="center", fontsize=8,
                    color="green" if delta >= 0 else "red")

    ax.set_xticks(x)
    ax.set_xticklabels(HELD_OUT, rotation=15, ha="right")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 65)
    ax.set_title("Held-Out Tasks — After GRPO T1", fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("Full Eval After GRPO Task 1 — Success-Only vs R1/R2", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "full_eval_comparison_success_only.png", dpi=150)
    plt.close(fig)
    print("  saved full_eval_comparison_success_only.png")


# ── Plot 5: Zero-shot comparison across all conditions ────────────────────────

def plot_zero_shot_comparison():
    """Zero-shot performance: regular SFT vs success-only SFT starting points."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(4)
    w = 0.25

    r1_zs = R1["matrix"][0] * 100
    r2_zs = R2["matrix"][0] * 100
    so_zs = SO_GRPO_PARTIAL["matrix"][0] * 100

    ax.bar(x - w, r1_zs, w, label="R1 SFT start (zero-shot for GRPO)", color=COLORS["R1"], edgecolor="white")
    ax.bar(x, r2_zs, w, label="R2 start (after R1 GRPO)", color=COLORS["R2"], edgecolor="white")
    ax.bar(x + w, so_zs, w, label="Success-Only SFT start", color=COLORS["SO_GRPO"], edgecolor="white")

    for i in range(4):
        for offset, vals in [(-w, r1_zs), (0, r2_zs), (w, so_zs)]:
            ax.text(i + offset, vals[i] + 0.3, f"{vals[i]:.1f}", ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, rotation=15, ha="right")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 62)
    ax.set_title("Zero-Shot Starting Performance — Before GRPO", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "zero_shot_comparison_success_only.png", dpi=150)
    plt.close(fig)
    print("  saved zero_shot_comparison_success_only.png")


if __name__ == "__main__":
    print("Generating success-only comparison figures...")
    plot_sft_comparison()
    plot_grpo_t1_comparison()
    plot_banana_curve_comparison()
    plot_zero_shot_comparison()
    plot_full_eval_comparison()
    print(f"\nDone! Figures saved to {FIGURES_DIR}/")
