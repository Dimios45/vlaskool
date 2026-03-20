"""Generate result visualizations for the blog/documentation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGURES_DIR = Path("docs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

COLORS = {
    "banana": "#f4a261",
    "gelatin_box": "#2a9d8f",
    "mug": "#e76f51",
    "cup": "#264653",
}
HELD_OUT_COLORS = {
    "bowl": "#a8dadc",
    "marker": "#457b9d",
    "foam_brick": "#e9c46a",
    "tuna_can": "#606c38",
}

# ── Data ──────────────────────────────────────────────────────────────────────

TASKS = ["banana", "gelatin_box", "mug", "cup"]
HELD_OUT = ["bowl", "marker", "foam_brick", "tuna_can"]

# Success matrix: rows = [zero-shot, after_task1, ..., after_task4]
SM = np.array([
    [0.394, 0.404, 0.365, 0.163],  # zero-shot
    [0.490, 0.433, 0.375, 0.163],  # after banana GRPO
    [0.471, 0.471, 0.356, 0.212],  # after gelatin GRPO
    [0.519, 0.481, 0.298, 0.212],  # after mug GRPO
    [0.404, 0.471, 0.356, 0.231],  # after cup GRPO
])

HELD_OUT_ZS = np.array([0.433, 0.327, 0.337, 0.423])
HELD_OUT_FINAL = np.array([0.404, 0.308, 0.279, 0.413])

# GRPO training curves (rollout success per update)
GRPO_CURVES = {
    "banana":      [0.479, 0.417, 0.458, 0.490, 0.417, 0.427, 0.385, 0.438, 0.417, 0.448],
    "gelatin_box": [0.448, 0.479, 0.438, 0.406, 0.448, 0.448, 0.500, 0.500, 0.469, 0.417],
    "mug":         [0.281, 0.333, 0.365, 0.333, 0.365, 0.375, 0.396, 0.260, 0.365, 0.344],
    "cup":         [0.177, 0.188, 0.208, 0.177, 0.115, 0.177, 0.198, 0.125, 0.156, 0.188],
}

# SFT success matrix
SFT_SM = np.array([
    [0.394, 0.404, 0.288, 0.183],  # zero-shot (pretrained)
    [0.471, 0.462, 0.327, 0.212],  # after SFT banana
    [0.385, 0.500, 0.288, 0.260],  # after SFT gelatin
    [0.481, 0.442, 0.337, 0.231],  # after SFT mug
    [0.365, 0.471, 0.337, 0.212],  # after SFT cup
])


# ── Plot 1: Success Matrix Heatmap ───────────────────────────────────────────

def plot_success_matrix():
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(SM * 100, cmap="YlGn", aspect="auto", vmin=0, vmax=60)

    ax.set_xticks(range(4))
    ax.set_xticklabels(TASKS, rotation=30, ha="right")
    ax.set_yticks(range(5))
    ax.set_yticklabels(["Zero-shot", "After banana", "After gelatin", "After mug", "After cup"])

    for i in range(5):
        for j in range(4):
            val = SM[i, j] * 100
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if val > 40 else "black")

    ax.set_title("GRPO Success Matrix — Eval After Each Task")
    fig.colorbar(im, ax=ax, label="Success Rate (%)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "success_matrix_heatmap.png", dpi=150)
    plt.close(fig)
    print("  saved success_matrix_heatmap.png")


# ── Plot 2: GRPO Training Curves ─────────────────────────────────────────────

def plot_grpo_curves():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)

    for ax, task in zip(axes.flat, TASKS):
        vals = GRPO_CURVES[task]
        baseline = SM[0, TASKS.index(task)]
        updates = list(range(1, len(vals) + 1))

        ax.plot(updates, [v * 100 for v in vals], "o-",
                color=COLORS[task], linewidth=2, markersize=5, label="Rollout success")
        ax.axhline(baseline * 100, color="gray", linestyle="--", alpha=0.7, label=f"Zero-shot ({baseline*100:.1f}%)")
        ax.set_title(task, fontweight="bold")
        ax.set_xlabel("GRPO Update")
        ax.set_ylabel("Success Rate (%)")
        ax.set_ylim(0, 65)
        ax.set_xticks(updates)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("GRPO Training Curves (Rollout Success per Update)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "grpo_training_curves.png", dpi=150)
    plt.close(fig)
    print("  saved grpo_training_curves.png")


# ── Plot 3: Before/After Bar Chart ───────────────────────────────────────────

def plot_before_after():
    fig, ax = plt.subplots(figsize=(10, 5))

    all_tasks = TASKS + HELD_OUT
    zs = list(SM[0] * 100) + list(HELD_OUT_ZS * 100)
    final = list(SM[-1] * 100) + list(HELD_OUT_FINAL * 100)

    x = np.arange(len(all_tasks))
    w = 0.35

    bars1 = ax.bar(x - w/2, zs, w, label="Zero-shot (after SFT)", color="#457b9d", alpha=0.85)
    bars2 = ax.bar(x + w/2, final, w, label="After GRPO (all 4 tasks)", color="#e76f51", alpha=0.85)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    # Separator between training and held-out
    ax.axvline(3.5, color="gray", linestyle=":", alpha=0.5)
    ax.text(1.5, 57, "Training Tasks", ha="center", fontsize=10, fontstyle="italic", color="gray")
    ax.text(5.5, 57, "Held-out Tasks", ha="center", fontsize=10, fontstyle="italic", color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(all_tasks, rotation=30, ha="right")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 62)
    ax.set_title("Zero-shot vs After GRPO — All 8 Tasks", fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "before_after_bar.png", dpi=150)
    plt.close(fig)
    print("  saved before_after_bar.png")


# ── Plot 4: Forgetting / Transfer Diagram ────────────────────────────────────

def plot_forgetting_transfer():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Left: task performance over time (forgetting view)
    stages = ["ZS", "T1", "T2", "T3", "T4"]
    for j, task in enumerate(TASKS):
        vals = SM[:, j] * 100
        ax1.plot(stages, vals, "o-", color=COLORS[task], linewidth=2, markersize=6, label=task)

    ax1.set_ylabel("Success Rate (%)")
    ax1.set_xlabel("After training task →")
    ax1.set_title("Per-Task Performance Over Training\n(forgetting = downward slope)", fontweight="bold")
    ax1.set_ylim(0, 60)
    ax1.legend(fontsize=9)

    # Right: held-out before/after
    x = np.arange(len(HELD_OUT))
    w = 0.35
    ax2.bar(x - w/2, HELD_OUT_ZS * 100, w, label="Zero-shot", color="#a8dadc")
    ax2.bar(x + w/2, HELD_OUT_FINAL * 100, w, label="After GRPO", color="#457b9d")

    for i, (zs, fn) in enumerate(zip(HELD_OUT_ZS, HELD_OUT_FINAL)):
        delta = (fn - zs) * 100
        ax2.text(i + w/2, fn * 100 + 0.8, f"{delta:+.1f}pp", ha="center", fontsize=8, color="red")

    ax2.set_xticks(x)
    ax2.set_xticklabels(HELD_OUT, rotation=30, ha="right")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title("Held-out Task Generalization\n(delta = change from zero-shot)", fontweight="bold")
    ax2.set_ylim(0, 55)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "forgetting_transfer.png", dpi=150)
    plt.close(fig)
    print("  saved forgetting_transfer.png")


# ── Plot 5: Metrics Summary ──────────────────────────────────────────────────

def plot_metrics_summary():
    fig, ax = plt.subplots(figsize=(6, 4))

    metrics = {"AVG": 0.365, "NBT": 0.010, "FWT": 0.022, "ZS": 0.351}
    names = list(metrics.keys())
    vals = list(metrics.values())
    colors = ["#2a9d8f", "#e76f51", "#f4a261", "#264653"]

    bars = ax.bar(names, vals, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Value")
    ax.set_title("Continual Learning Metrics", fontweight="bold")
    ax.set_ylim(0, 0.45)

    # Annotations
    ax.annotate("AVG: higher = better\nNBT: lower = less forgetting\nFWT: higher = better transfer\nZS: higher = better generalization",
                xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
                fontsize=8, color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "metrics_summary.png", dpi=150)
    plt.close(fig)
    print("  saved metrics_summary.png")


# ── Plot 6: SFT vs GRPO Comparison ───────────────────────────────────────────

def plot_sft_vs_grpo():
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(TASKS))
    w = 0.2

    pretrained = SFT_SM[0] * 100
    after_sft = SFT_SM[-1] * 100
    after_grpo = SM[-1] * 100
    peak_grpo = np.max(SM[1:], axis=0) * 100

    ax.bar(x - 1.5*w, pretrained, w, label="Pretrained (no training)", color="#ccc", edgecolor="white")
    ax.bar(x - 0.5*w, after_sft, w, label="After SFT (all 4 tasks)", color="#457b9d", edgecolor="white")
    ax.bar(x + 0.5*w, after_grpo, w, label="After GRPO (all 4 tasks)", color="#e76f51", edgecolor="white")
    ax.bar(x + 1.5*w, peak_grpo, w, label="Peak during GRPO", color="#2a9d8f", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, rotation=15, ha="right")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Performance Progression: Pretrained → SFT → GRPO", fontweight="bold")
    ax.set_ylim(0, 62)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sft_vs_grpo.png", dpi=150)
    plt.close(fig)
    print("  saved sft_vs_grpo.png")


# ── Generate all ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...")
    plot_success_matrix()
    plot_grpo_curves()
    plot_before_after()
    plot_forgetting_transfer()
    plot_metrics_summary()
    plot_sft_vs_grpo()
    print(f"\nDone! {len(list(FIGURES_DIR.glob('*.png')))} figures in {FIGURES_DIR}/")
