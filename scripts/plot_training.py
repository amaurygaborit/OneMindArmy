"""
plot_training.py — Visualise OneMindArmy training curves
=========================================================

Usage:
    python plot_training.py --log data/chess/training_log.jsonl
    python plot_training.py --log data/chess/training_log.jsonl --smooth 500
    python plot_training.py --log data/chess/training_log.jsonl --mode epoch
    python plot_training.py --log data/chess/training_log.jsonl --iter 5 10 15

Arguments:
    --log     Path to the training_log.jsonl file (required)
    --mode    "batch" (default) or "epoch"
    --smooth  EMA smoothing window for batch curves (default: 200)
    --iter    Show vertical markers at these iteration boundaries
    --save    If given, save the figure to this path instead of showing it
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ==============================================================================
# --- STYLE ---
# ==============================================================================

DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
GRID_COL  = "#2a2a4a"
TEXT_COL  = "#e0e0e0"
ACCENT    = "#0f3460"

V_COLOR   = "#e94560"   # red  — value loss
P_COLOR   = "#00d4ff"   # cyan — policy loss
T_COLOR   = "#ffa500"   # orange — total loss
LR_COLOR  = "#a8ff78"   # green — learning rate

ALPHA_RAW  = 0.20
ALPHA_SMOOTH = 0.90


# ==============================================================================
# --- HELPERS ---
# ==============================================================================

def load_jsonl(path: Path):
    batch_records = []
    epoch_records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "batch":
                batch_records.append(r)
            elif r.get("type") == "epoch":
                epoch_records.append(r)
    return batch_records, epoch_records


def ema_smooth(values, alpha=0.99):
    """Exponential moving average. alpha close to 1 = very smooth."""
    if not values:
        return []
    out   = [values[0]]
    for v in values[1:]:
        out.append(alpha * out[-1] + (1 - alpha) * v)
    return out


def alpha_from_window(window: int) -> float:
    """Convert a 'smoothing window' count to an EMA alpha."""
    # alpha such that the half-life ≈ window/2 steps
    return 1.0 - 2.0 / max(window + 1, 2)


def set_style():
    matplotlib.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    GRID_COL,
        "axes.labelcolor":   TEXT_COL,
        "axes.grid":         True,
        "grid.color":        GRID_COL,
        "grid.linewidth":    0.6,
        "xtick.color":       TEXT_COL,
        "ytick.color":       TEXT_COL,
        "text.color":        TEXT_COL,
        "legend.facecolor":  PANEL_BG,
        "legend.edgecolor":  GRID_COL,
        "legend.labelcolor": TEXT_COL,
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
    })


def add_iteration_markers(ax, batch_records, show_iters):
    """Draw vertical dashed lines at the first global_step of each requested iteration."""
    iter_steps = {}
    for r in batch_records:
        it = r["iteration"]
        if it not in iter_steps or r["global_step"] < iter_steps[it]:
            iter_steps[it] = r["global_step"]

    for it in show_iters:
        if it in iter_steps:
            ax.axvline(iter_steps[it], color="#ffffff", linewidth=0.8,
                       linestyle="--", alpha=0.4)
            ax.text(iter_steps[it], ax.get_ylim()[1] * 0.97,
                    f" iter {it}", color="#ffffff", alpha=0.5,
                    fontsize=7, va="top")


# ==============================================================================
# --- BATCH MODE ---
# ==============================================================================

def plot_batch(batch_records, smooth_alpha, show_iters, save_path):
    if not batch_records:
        print("[Error] No batch records found in log file.")
        return

    steps   = [r["global_step"] for r in batch_records]
    v_vals  = [r["v_loss"]      for r in batch_records]
    p_vals  = [r["p_loss"]      for r in batch_records]
    t_vals  = [r["total_loss"]  for r in batch_records]
    lr_vals = [r["lr"]          for r in batch_records]
    norms   = [r["grad_norm"]   for r in batch_records]

    v_smooth = ema_smooth(v_vals,  smooth_alpha)
    p_smooth = ema_smooth(p_vals,  smooth_alpha)
    t_smooth = ema_smooth(t_vals,  smooth_alpha)

    set_style()
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle("Training Curves — Batch Level", fontsize=14, y=0.98)
    fig.subplots_adjust(hspace=0.08, top=0.94, bottom=0.06, left=0.07, right=0.97)

    # --- Panel 1: V-Loss & P-Loss ---
    ax = axes[0]
    ax.plot(steps, v_vals,  color=V_COLOR, alpha=ALPHA_RAW,    linewidth=0.6)
    ax.plot(steps, v_smooth, color=V_COLOR, alpha=ALPHA_SMOOTH, linewidth=1.5, label="V-Loss (value MSE)")
    ax.plot(steps, p_vals,  color=P_COLOR, alpha=ALPHA_RAW,    linewidth=0.6)
    ax.plot(steps, p_smooth, color=P_COLOR, alpha=ALPHA_SMOOTH, linewidth=1.5, label="P-Loss (policy CE)")
    ax.set_ylabel("Loss")
    ax.set_title("Value Loss & Policy Loss")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

    # --- Panel 2: Total Loss & Grad Norm ---
    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(steps, t_vals,  color=T_COLOR, alpha=ALPHA_RAW,    linewidth=0.6)
    ax.plot(steps, t_smooth, color=T_COLOR, alpha=ALPHA_SMOOTH, linewidth=1.5, label="Total Loss")
    ax2.plot(steps, norms, color="#9966cc", alpha=0.35, linewidth=0.7, label="Grad Norm")
    ax.set_ylabel("Total Loss", color=T_COLOR)
    ax2.set_ylabel("Grad Norm",  color="#9966cc")
    ax2.tick_params(axis="y", labelcolor="#9966cc")
    ax.set_title("Total Loss & Gradient Norm")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # --- Panel 3: Learning Rate ---
    ax = axes[2]
    ax.plot(steps, lr_vals, color=LR_COLOR, linewidth=1.2, label="Learning Rate")
    ax.set_ylabel("LR")
    ax.set_xlabel("Global Step")
    ax.set_title("Learning Rate Schedule")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Iteration markers on all panels
    for ax in axes:
        add_iteration_markers(ax, batch_records, show_iters)

    _finish(fig, save_path)


# ==============================================================================
# --- EPOCH MODE ---
# ==============================================================================

def plot_epoch(epoch_records, show_iters, save_path):
    if not epoch_records:
        print("[Error] No epoch records found in log file.")
        return

    iters   = [r["iteration"]    for r in epoch_records]
    v_vals  = [r["avg_v_loss"]   for r in epoch_records]
    p_vals  = [r["avg_p_loss"]   for r in epoch_records]
    t_vals  = [r["avg_total_loss"] for r in epoch_records]
    times   = [r["time_s"] / 60  for r in epoch_records]   # → minutes

    set_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Training Curves — Iteration (Epoch) Level", fontsize=14, y=0.98)
    fig.subplots_adjust(hspace=0.1, top=0.93, bottom=0.08, left=0.08, right=0.97)

    # --- Panel 1: V-Loss & P-Loss per iteration ---
    ax = axes[0]
    ax.plot(iters, v_vals, color=V_COLOR, marker="o", markersize=4,
            linewidth=1.5, label="Avg V-Loss")
    ax.plot(iters, p_vals, color=P_COLOR, marker="s", markersize=4,
            linewidth=1.5, label="Avg P-Loss")
    ax.plot(iters, t_vals, color=T_COLOR, marker="^", markersize=3,
            linewidth=1.0, linestyle="--", label="Avg Total Loss", alpha=0.7)
    ax.set_ylabel("Loss")
    ax.set_title("Average Losses per Iteration")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # --- Panel 2: Training time per iteration ---
    ax = axes[1]
    ax.bar(iters, times, color=ACCENT, edgecolor=GRID_COL, linewidth=0.5,
           alpha=0.85, label="Train time (min)")
    ax.set_ylabel("Time (min)")
    ax.set_xlabel("Iteration")
    ax.set_title("Training Time per Iteration")
    ax.legend(loc="upper left")

    _finish(fig, save_path)


# ==============================================================================
# --- FINISH ---
# ==============================================================================

def _finish(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Plot] Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ==============================================================================
# --- ENTRY POINT ---
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot OneMindArmy training curves")
    parser.add_argument("--log",    type=str, required=True,
                        help="Path to training_log.jsonl")
    parser.add_argument("--mode",   type=str, default="batch",
                        choices=["batch", "epoch"],
                        help="'batch' = fine-grained (default) | 'epoch' = per-iteration summary")
    parser.add_argument("--smooth", type=int, default=200,
                        help="EMA smoothing window in batches (batch mode only, default: 200)")
    parser.add_argument("--iter",   type=int, nargs="*", default=[],
                        help="Iteration numbers to mark with vertical lines")
    parser.add_argument("--save",   type=str, default=None,
                        help="Save figure to this path (PNG/PDF) instead of displaying")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[Error] Log file not found: {log_path}")
        sys.exit(1)

    print(f"[Plot] Reading {log_path} ...")
    batch_records, epoch_records = load_jsonl(log_path)
    print(f"[Plot] {len(batch_records)} batch records, {len(epoch_records)} epoch records")

    alpha = alpha_from_window(args.smooth)

    if args.mode == "batch":
        plot_batch(batch_records, alpha, args.iter, args.save)
    else:
        plot_epoch(epoch_records, args.iter, args.save)


if __name__ == "__main__":
    main()