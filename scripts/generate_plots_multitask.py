"""Generate figures for Experiment 4: Multitask SFT + GRPO."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

COLORS = {
    "banana": "#f4a261",
    "gelatin_box": "#2a9d8f",
    "mug": "#e76f51",
    "cup": "#264653",
}

# ── Data ──────────────────────────────────────────────────────────────────────

# R1: sequential SFT 100ep/task → GRPO 10 updates
R1_SM = np.array([
    [0.394, 0.404, 0.365, 0.163],
    [0.490, 0.433, 0.375, 0.163],
    [0.471, 0.471, 0.356, 0.212],
    [0.519, 0.481, 0.298, 0.212],
    [0.404, 0.471, 0.356, 0.231],
])
R1_HELD_ZS   = np.array([0.433, 0.327, 0.337, 0.423])
R1_HELD_FINAL = np.array([0.404, 0.308, 0.279, 0.413])
R1_METRICS = {"AVG": 0.365, "NBT": 0.010, "FWT": 0.022, "ZS": 0.351}
R1_CURVES = {
    "banana":      [0.479, 0.417, 0.458, 0.490, 0.417, 0.427, 0.385, 0.438, 0.417, 0.448],
    "gelatin_box": [0.448, 0.479, 0.438, 0.406, 0.448, 0.448, 0.500, 0.500, 0.469, 0.417],
    "mug":         [0.281, 0.333, 0.365, 0.333, 0.365, 0.375, 0.396, 0.260, 0.365, 0.344],
    "cup":         [0.177, 0.188, 0.208, 0.177, 0.115, 0.177, 0.198, 0.125, 0.156, 0.188],
}

# R2: R1 checkpoint → 10 more GRPO updates
R2_SM = np.array([
    [0.394, 0.404, 0.337, 0.115],
    [0.490, 0.433, 0.365, 0.144],
    [0.471, 0.471, 0.385, 0.212],
    [0.519, 0.481, 0.337, 0.212],
    [0.404, 0.471, 0.385, 0.221],
])
R2_HELD_ZS   = np.array([0.433, 0.327, 0.337, 0.423])
R2_HELD_FINAL = np.array([0.433, 0.308, 0.279, 0.413])
R2_METRICS = {"AVG": 0.370, "NBT": 0.013, "FWT": 0.058, "ZS": 0.358}

# E4: multitask SFT 200ep → GRPO 10 updates
E4_SM = np.array([
    [0.394, 0.404, 0.327, 0.087],
    [0.490, 0.433, 0.375, 0.125],
    [0.471, 0.471, 0.356, 0.202],
    [0.519, 0.481, 0.308, 0.231],
    [0.404, 0.471, 0.394, 0.221],
])
E4_HELD_ZS   = np.array([0.423, 0.327, 0.337, 0.423])
E4_HELD_FINAL = np.array([0.433, 0.308, 0.279, 0.413])
E4_METRICS = {"AVG": 0.373, "NBT": 0.000, "FWT": 0.067, "ZS": 0.358}
E4_CURVES = {
    "banana":      [0.396, 0.438, 0.510, 0.396, 0.458, 0.500, 0.458, 0.427, 0.375, 0.438],
    "gelatin_box": [0.510, 0.448, 0.469, 0.438, 0.427, 0.469, 0.427, 0.427, 0.458, 0.396],
    "mug":         [0.385, 0.365, 0.375, 0.344, 0.344, 0.333, 0.333, 0.271, 0.281, 0.281],
    "cup":         [0.135, 0.125, 0.208, 0.156, 0.146, 0.198, 0.188, 0.146, 0.146, 0.156],
}


# ── Plot 1: Metrics comparison — R1 vs R2 vs E4 ───────────────────────────────

def plot_metrics_all():
    fig, ax = plt.subplots(figsize=(9, 5))

    names = list(R1_METRICS.keys())
    r1_vals = list(R1_METRICS.values())
    r2_vals = list(R2_METRICS.values())
    e4_vals = list(E4_METRICS.values())

    x = np.arange(len(names))
    w = 0.22

    b1 = ax.bar(x - w,    r1_vals, w, label="R1: Seq SFT → GRPO (10 upd)",       color="#457b9d", edgecolor="white")
    b2 = ax.bar(x,        r2_vals, w, label="R2: R1 ckpt → GRPO (20 total upd)",  color="#e76f51", edgecolor="white")
    b3 = ax.bar(x + w,    e4_vals, w, label="E4: Multitask SFT → GRPO (10 upd)",  color="#2a9d8f", edgecolor="white")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Value")
    ax.set_ylim(0, 0.50)
    ax.set_title("Continual Learning Metrics — R1 vs R2 vs Multitask SFT", fontweight="bold")
    ax.legend(fontsize=9)
    ax.annotate(
        "AVG: higher = better\nNBT: lower = less forgetting\nFWT: higher = better transfer\nZS: higher = better generalization",
        xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
        fontsize=8, color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "metrics_all_experiments.png", dpi=150)
    plt.close(fig)
    print("  saved metrics_all_experiments.png")


# ── Plot 2: Success matrix heatmap for E4 ────────────────────────────────────

def plot_e4_success_matrix():
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(E4_SM * 100, cmap="YlGn", aspect="auto", vmin=0, vmax=60)
    ax.set_xticks(range(4))
    ax.set_xticklabels(TASKS, rotation=30, ha="right")
    ax.set_yticks(range(5))
    ax.set_yticklabels(["Zero-shot", "After banana", "After gelatin", "After mug", "After cup"])
    for i in range(5):
        for j in range(4):
            val = E4_SM[i, j] * 100
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if val > 40 else "black")
    fig.colorbar(im, ax=ax, label="Success Rate (%)", shrink=0.8)
    ax.set_title("Multitask SFT + GRPO — Success Matrix", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "success_matrix_multitask.png", dpi=150)
    plt.close(fig)
    print("  saved success_matrix_multitask.png")


# ── Plot 3: Final bar — R1 vs R2 vs E4 (all 8 tasks) ─────────────────────────

def plot_final_all():
    fig, ax = plt.subplots(figsize=(12, 5))

    all_tasks = TASKS + HELD_OUT
    r1_final = list(R1_SM[-1] * 100) + list(R1_HELD_FINAL * 100)
    r2_final = list(R2_SM[-1] * 100) + list(R2_HELD_FINAL * 100)
    e4_final = list(E4_SM[-1] * 100) + list(E4_HELD_FINAL * 100)

    x = np.arange(len(all_tasks))
    w = 0.22

    ax.bar(x - w,  r1_final, w, label="R1: Seq SFT → GRPO (10 upd)",      color="#457b9d", edgecolor="white")
    ax.bar(x,      r2_final, w, label="R2: R1 ckpt → GRPO (20 total upd)", color="#e76f51", edgecolor="white")
    ax.bar(x + w,  e4_final, w, label="E4: Multitask SFT → GRPO (10 upd)", color="#2a9d8f", edgecolor="white")

    for i, v in enumerate(e4_final):
        ax.text(i + w, v + 0.5, f"{v:.1f}", ha="center", fontsize=7.5, color="#2a9d8f", fontweight="bold")

    ax.axvline(3.5, color="gray", linestyle=":", alpha=0.5)
    ax.text(1.5, 58, "Training Tasks", ha="center", fontsize=10, fontstyle="italic", color="gray")
    ax.text(5.5, 58, "Held-out Tasks", ha="center", fontsize=10, fontstyle="italic", color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(all_tasks, rotation=30, ha="right")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 65)
    ax.set_title("Final Performance — R1 vs R2 vs Multitask SFT (E4)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_comparison_all.png", dpi=150)
    plt.close(fig)
    print("  saved final_comparison_all.png")


# ── Plot 4: Zero-shot starting points — R1 vs R2 vs E4 ───────────────────────

def plot_sft_init_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(4)
    w = 0.22

    r1_zs = R1_SM[0] * 100
    r2_zs = R2_SM[0] * 100
    e4_zs = E4_SM[0] * 100

    ax.bar(x - w, r1_zs, w, label="R1: Seq SFT (100ep/task)",     color="#457b9d", edgecolor="white")
    ax.bar(x,     r2_zs, w, label="R2: same init as R1",           color="#e76f51", edgecolor="white")
    ax.bar(x + w, e4_zs, w, label="E4: Multitask SFT (200ep all)", color="#2a9d8f", edgecolor="white")

    for i in range(4):
        ax.text(i - w, r1_zs[i] + 0.5, f"{r1_zs[i]:.1f}", ha="center", fontsize=8)
        ax.text(i,     r2_zs[i] + 0.5, f"{r2_zs[i]:.1f}", ha="center", fontsize=8)
        ax.text(i + w, e4_zs[i] + 0.5, f"{e4_zs[i]:.1f}", ha="center", fontsize=8,
                color="#2a9d8f", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, rotation=15, ha="right")
    ax.set_ylabel("Zero-shot Success Rate (%)")
    ax.set_ylim(0, 55)
    ax.set_title("GRPO Starting Points — Effect of SFT Strategy", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sft_init_comparison.png", dpi=150)
    plt.close(fig)
    print("  saved sft_init_comparison.png")


# ── Plot 5: GRPO training curves — R1 vs E4 side by side ─────────────────────

def plot_grpo_curves_e4():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)

    for ax, task in zip(axes.flat, TASKS):
        r1 = [v * 100 for v in R1_CURVES[task]]
        e4 = [v * 100 for v in E4_CURVES[task]]
        updates = list(range(1, 11))

        ax.plot(updates, r1, "o--", color="#457b9d", linewidth=1.8, markersize=4, label="R1 (seq SFT init)", alpha=0.8)
        ax.plot(updates, e4, "o-",  color="#2a9d8f", linewidth=2,   markersize=5, label="E4 (multitask SFT init)")

        sft_r1 = R1_SM[0, TASKS.index(task)] * 100
        sft_e4 = E4_SM[0, TASKS.index(task)] * 100
        ax.axhline(sft_r1, color="#457b9d", linestyle=":", alpha=0.5, linewidth=1)
        ax.axhline(sft_e4, color="#2a9d8f", linestyle=":", alpha=0.5, linewidth=1)

        ax.set_title(task, fontweight="bold")
        ax.set_xlabel("GRPO Update")
        ax.set_ylabel("Rollout Success (%)")
        ax.set_ylim(0, 65)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("GRPO Training Curves — R1 vs Multitask SFT (E4)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "grpo_curves_multitask.png", dpi=150)
    plt.close(fig)
    print("  saved grpo_curves_multitask.png")


# ── Plot 6: NBT / forgetting trajectory — E4 ─────────────────────────────────

def plot_forgetting_e4():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    stages = ["Zero-shot", "After T1", "After T2", "After T3", "After T4"]

    for ax, j, task in zip(axes.flat, range(4), TASKS):
        r1_vals = R1_SM[:, j] * 100
        e4_vals = E4_SM[:, j] * 100

        ax.plot(range(5), r1_vals, "o--", color="#457b9d", linewidth=1.8, markersize=5,
                label="R1 (seq SFT)", alpha=0.8)
        ax.plot(range(5), e4_vals, "o-",  color="#2a9d8f", linewidth=2,   markersize=6,
                label="E4 (multitask SFT)")

        ax.set_xticks(range(5))
        ax.set_xticklabels(stages, rotation=20, ha="right", fontsize=9)
        ax.set_title(task, fontweight="bold")
        ax.set_ylabel("Eval Success (%)")
        ax.set_ylim(0, 62)
        ax.legend(fontsize=8)

    fig.suptitle("Per-Task Eval Trajectory — R1 vs Multitask SFT (E4)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "forgetting_trajectory_multitask.png", dpi=150)
    plt.close(fig)
    print("  saved forgetting_trajectory_multitask.png")


# ── Generate all ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures for Experiment 4 (Multitask SFT)...")
    plot_metrics_all()
    plot_e4_success_matrix()
    plot_final_all()
    plot_sft_init_comparison()
    plot_grpo_curves_e4()
    plot_forgetting_e4()
    print(f"\nDone! Figures in {FIGURES_DIR}/")
