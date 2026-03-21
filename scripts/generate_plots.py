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

# ── Data ──────────────────────────────────────────────────────────────────────

TASKS = ["banana", "gelatin_box", "mug", "cup"]
HELD_OUT = ["bowl", "marker", "foam_brick", "tuna_can"]

# Round 1 success matrix
R1_SM = np.array([
    [0.394, 0.404, 0.365, 0.163],
    [0.490, 0.433, 0.375, 0.163],
    [0.471, 0.471, 0.356, 0.212],
    [0.519, 0.481, 0.298, 0.212],
    [0.404, 0.471, 0.356, 0.231],
])

R1_HELD_ZS = np.array([0.433, 0.327, 0.337, 0.423])
R1_HELD_FINAL = np.array([0.404, 0.308, 0.279, 0.413])

# Round 2 success matrix
R2_SM = np.array([
    [0.394, 0.404, 0.337, 0.115],
    [0.490, 0.433, 0.365, 0.144],
    [0.471, 0.471, 0.385, 0.212],
    [0.519, 0.481, 0.337, 0.212],
    [0.404, 0.471, 0.385, 0.221],
])

R2_HELD_ZS = np.array([0.433, 0.327, 0.337, 0.423])
R2_HELD_FINAL = np.array([0.433, 0.308, 0.279, 0.413])

# GRPO training curves (rollout success per update)
R1_CURVES = {
    "banana":      [0.479, 0.417, 0.458, 0.490, 0.417, 0.427, 0.385, 0.438, 0.417, 0.448],
    "gelatin_box": [0.448, 0.479, 0.438, 0.406, 0.448, 0.448, 0.500, 0.500, 0.469, 0.417],
    "mug":         [0.281, 0.333, 0.365, 0.333, 0.365, 0.375, 0.396, 0.260, 0.365, 0.344],
    "cup":         [0.177, 0.188, 0.208, 0.177, 0.115, 0.177, 0.198, 0.125, 0.156, 0.188],
}

R2_CURVES = {
    "banana":      [0.479, 0.417, 0.458, 0.490, 0.417, 0.427, 0.385, 0.438, 0.417, 0.448],
    "gelatin_box": [0.448, 0.479, 0.438, 0.406, 0.448, 0.448, 0.500, 0.500, 0.469, 0.417],
    "mug":         [0.312, 0.333, 0.333, 0.365, 0.417, 0.354, 0.323, 0.271, 0.354, 0.312],
    "cup":         [0.167, 0.146, 0.198, 0.167, 0.062, 0.177, 0.156, 0.156, 0.177, 0.229],
}

# SFT success matrix (for reference)
SFT_SM = np.array([
    [0.394, 0.404, 0.288, 0.183],
    [0.471, 0.462, 0.327, 0.212],
    [0.385, 0.500, 0.288, 0.260],
    [0.481, 0.442, 0.337, 0.231],
    [0.365, 0.471, 0.337, 0.212],
])

# Metrics
R1_METRICS = {"AVG": 0.365, "NBT": 0.010, "FWT": 0.022, "ZS": 0.351}
R2_METRICS = {"AVG": 0.370, "NBT": 0.013, "FWT": 0.058, "ZS": 0.358}


# ── Plot 1: Success Matrix Heatmaps (R1 vs R2 side by side) ──────────────────

def plot_success_matrix_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, sm, title in [(ax1, R1_SM, "Round 1 (updates 1-10)"),
                           (ax2, R2_SM, "Round 2 (updates 11-20)")]:
        im = ax.imshow(sm * 100, cmap="YlGn", aspect="auto", vmin=0, vmax=60)
        ax.set_xticks(range(4))
        ax.set_xticklabels(TASKS, rotation=30, ha="right")
        ax.set_yticks(range(5))
        ax.set_yticklabels(["Zero-shot", "After banana", "After gelatin", "After mug", "After cup"])
        for i in range(5):
            for j in range(4):
                val = sm[i, j] * 100
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if val > 40 else "black")
        ax.set_title(title, fontweight="bold")

    fig.colorbar(im, ax=[ax1, ax2], label="Success Rate (%)", shrink=0.8)
    fig.suptitle("GRPO Success Matrix — Round 1 vs Round 2", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "success_matrix_comparison.png", dpi=150)
    plt.close(fig)
    print("  saved success_matrix_comparison.png")


# ── Plot 2: Combined Training Curves (R1 + R2 = 20 updates) ──────────────────

def plot_combined_training_curves():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)

    for ax, task in zip(axes.flat, TASKS):
        r1 = R1_CURVES[task]
        r2 = R2_CURVES[task]
        combined = r1 + r2
        updates = list(range(1, 21))
        baseline_r1 = R1_SM[0, TASKS.index(task)]

        ax.plot(updates, [v * 100 for v in combined], "o-",
                color=COLORS[task], linewidth=2, markersize=4)
        ax.axhline(baseline_r1 * 100, color="gray", linestyle="--", alpha=0.7,
                   label=f"SFT baseline ({baseline_r1*100:.1f}%)")
        ax.axvline(10.5, color="red", linestyle=":", alpha=0.4, label="Round 1 → 2")
        ax.set_title(task, fontweight="bold")
        ax.set_xlabel("GRPO Update (cumulative)")
        ax.set_ylabel("Rollout Success (%)")
        ax.set_ylim(0, 65)
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("GRPO Training Curves — 20 Updates Total (Round 1 + Round 2)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "grpo_combined_curves.png", dpi=150)
    plt.close(fig)
    print("  saved grpo_combined_curves.png")


# ── Plot 3: Metrics Comparison (R1 vs R2) ─────────────────────────────────────

def plot_metrics_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(R1_METRICS.keys())
    r1_vals = list(R1_METRICS.values())
    r2_vals = list(R2_METRICS.values())

    x = np.arange(len(names))
    w = 0.3

    bars1 = ax.bar(x - w/2, r1_vals, w, label="Round 1 (10 updates)", color="#457b9d", edgecolor="white")
    bars2 = ax.bar(x + w/2, r2_vals, w, label="Round 2 (20 total)", color="#e76f51", edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Value")
    ax.set_ylim(0, 0.45)
    ax.set_title("Continual Learning Metrics — Round 1 vs Round 2", fontweight="bold")
    ax.legend()

    ax.annotate("AVG: higher = better\nNBT: lower = less forgetting\nFWT: higher = better transfer\nZS: higher = better generalization",
                xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
                fontsize=8, color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "metrics_comparison.png", dpi=150)
    plt.close(fig)
    print("  saved metrics_comparison.png")


# ── Plot 4: Final Performance Bar (R1 vs R2 all 8 tasks) ─────────────────────

def plot_final_comparison_bar():
    fig, ax = plt.subplots(figsize=(11, 5))

    all_tasks = TASKS + HELD_OUT

    r1_final = list(R1_SM[-1] * 100) + list(R1_HELD_FINAL * 100)
    r2_final = list(R2_SM[-1] * 100) + list(R2_HELD_FINAL * 100)
    sft_final = list(SFT_SM[-1] * 100) + [np.nan] * 4  # no SFT held-out at same stage

    x = np.arange(len(all_tasks))
    w = 0.25

    ax.bar(x - w, [SFT_SM[-1, i] * 100 if i < 4 else np.nan for i in range(8)], w,
           label="After SFT", color="#ccc", edgecolor="white")
    ax.bar(x, r1_final, w, label="After GRPO R1 (10 upd)", color="#457b9d", edgecolor="white")
    ax.bar(x + w, r2_final, w, label="After GRPO R2 (20 upd)", color="#e76f51", edgecolor="white")

    # Value labels on R2
    for i, v in enumerate(r2_final):
        if not np.isnan(v):
            ax.text(i + w, v + 0.5, f"{v:.1f}", ha="center", fontsize=7.5)

    # Separator
    ax.axvline(3.5, color="gray", linestyle=":", alpha=0.5)
    ax.text(1.5, 57, "Training Tasks", ha="center", fontsize=10, fontstyle="italic", color="gray")
    ax.text(5.5, 57, "Held-out Tasks", ha="center", fontsize=10, fontstyle="italic", color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(all_tasks, rotation=30, ha="right")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 62)
    ax.set_title("Final Performance: SFT → GRPO Round 1 → Round 2", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_comparison_bar.png", dpi=150)
    plt.close(fig)
    print("  saved final_comparison_bar.png")


# ── Plot 5: Forgetting Trajectories (R1 vs R2 overlay) ───────────────────────

def plot_forgetting_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    stages_r1 = ["SFT", "R1-T1", "R1-T2", "R1-T3", "R1-T4"]
    stages_r2 = ["R1end", "R2-T1", "R2-T2", "R2-T3", "R2-T4"]
    stages_all = stages_r1 + stages_r2

    for ax, j, task in zip(axes.flat, range(4), TASKS):
        r1_vals = R1_SM[:, j] * 100
        r2_vals = R2_SM[:, j] * 100
        combined = list(r1_vals) + list(r2_vals)

        ax.plot(range(10), combined, "o-", color=COLORS[task], linewidth=2, markersize=5)
        ax.axvline(4.5, color="red", linestyle=":", alpha=0.4)
        ax.text(2, 58, "Round 1", ha="center", fontsize=8, color="gray")
        ax.text(7, 58, "Round 2", ha="center", fontsize=8, color="gray")

        ax.set_xticks(range(10))
        ax.set_xticklabels(stages_all, rotation=45, ha="right", fontsize=7)
        ax.set_title(task, fontweight="bold")
        ax.set_ylabel("Eval Success (%)")
        ax.set_ylim(0, 62)

    fig.suptitle("Per-Task Eval Trajectory Across Both Rounds", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "forgetting_trajectory_both.png", dpi=150)
    plt.close(fig)
    print("  saved forgetting_trajectory_both.png")


# ── Plot 6: Held-out comparison ───────────────────────────────────────────────

def plot_held_out_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(HELD_OUT))
    w = 0.25

    ax.bar(x - w, R1_HELD_ZS * 100, w, label="Zero-shot (SFT)", color="#a8dadc", edgecolor="white")
    ax.bar(x, R1_HELD_FINAL * 100, w, label="After R1 (10 upd)", color="#457b9d", edgecolor="white")
    ax.bar(x + w, R2_HELD_FINAL * 100, w, label="After R2 (20 upd)", color="#e76f51", edgecolor="white")

    for i in range(len(HELD_OUT)):
        delta_r1 = (R1_HELD_FINAL[i] - R1_HELD_ZS[i]) * 100
        delta_r2 = (R2_HELD_FINAL[i] - R1_HELD_ZS[i]) * 100
        ax.text(i + w, R2_HELD_FINAL[i] * 100 + 0.8, f"{delta_r2:+.1f}pp",
                ha="center", fontsize=8, color="red" if delta_r2 < 0 else "green")

    ax.set_xticks(x)
    ax.set_xticklabels(HELD_OUT, rotation=15, ha="right")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Held-out Task Generalization — Zero-shot vs R1 vs R2", fontweight="bold")
    ax.set_ylim(0, 55)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "held_out_comparison.png", dpi=150)
    plt.close(fig)
    print("  saved held_out_comparison.png")


# ── Plot 7: Improvement delta (R2 - R1 final) ────────────────────────────────

def plot_improvement_delta():
    fig, ax = plt.subplots(figsize=(8, 5))

    all_tasks = TASKS + HELD_OUT
    r1_all = list(R1_SM[-1]) + list(R1_HELD_FINAL)
    r2_all = list(R2_SM[-1]) + list(R2_HELD_FINAL)
    deltas = [(r2 - r1) * 100 for r1, r2 in zip(r1_all, r2_all)]

    colors = ["#2a9d8f" if d >= 0 else "#e76f51" for d in deltas]
    bars = ax.bar(range(len(all_tasks)), deltas, color=colors, edgecolor="white", width=0.6)

    for i, (bar, d) in enumerate(zip(bars, deltas)):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.2 if d >= 0 else -0.8),
                f"{d:+.1f}pp", ha="center", va="bottom" if d >= 0 else "top",
                fontsize=9, fontweight="bold")

    ax.axvline(3.5, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(all_tasks)))
    ax.set_xticklabels(all_tasks, rotation=30, ha="right")
    ax.set_ylabel("Change in Success Rate (pp)")
    ax.set_title("Round 2 vs Round 1 — Per-Task Improvement (final eval)", fontweight="bold")
    ax.text(1.5, max(deltas) * 0.9, "Training", ha="center", fontsize=9, fontstyle="italic", color="gray")
    ax.text(5.5, max(deltas) * 0.9, "Held-out", ha="center", fontsize=9, fontstyle="italic", color="gray")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "improvement_delta.png", dpi=150)
    plt.close(fig)
    print("  saved improvement_delta.png")


# ── Generate all ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...")
    plot_success_matrix_comparison()
    plot_combined_training_curves()
    plot_metrics_comparison()
    plot_final_comparison_bar()
    plot_forgetting_comparison()
    plot_held_out_comparison()
    plot_improvement_delta()
    print(f"\nDone! {len(list(FIGURES_DIR.glob('*.png')))} figures in {FIGURES_DIR}/")
