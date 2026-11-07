"""
Loss Curve Visualisation Utility
================================

Generates training and validation loss curves from logged loss history:
  1. Results/English/training_loss.png
  2. Results/English/validation_loss.png
  3. Results/Telugu/training_loss.png
  4. Results/Telugu/validation_loss.png

Licence: MIT
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def load_loss_history(history_file):
    """Loads loss history from .npz file, raising FileNotFoundError if missing."""
    history_path = Path(history_file)
    if not history_path.exists():
        raise FileNotFoundError(
            f"Training loss log '{history_path}' not found. "
            f"Loss curves must be generated from genuine logged training steps."
        )
    return np.load(history_path, allow_pickle=True)


def generate_loss_plots(results_root="Results", checkpoints_root="."):
    """Reads loss_history.npz and produces the 4 required loss plots."""
    for lang in ["English", "Telugu"]:
        ckpt_history = Path(checkpoints_root) / lang / "checkpoints" / "loss_history.npz"
        res_dir = Path(results_root) / lang
        res_dir.mkdir(parents=True, exist_ok=True)

        if not ckpt_history.exists():
            raise FileNotFoundError(
                f"Training loss log '{ckpt_history}' not found. "
                f"Loss curves must be generated from genuine logged training steps. "
                f"Run {lang}/train.py to generate genuine loss history."
            )

        data = np.load(ckpt_history, allow_pickle=True)
        train_steps = data["train_steps"]
        train_losses = data["train_losses"]
        val_steps = data["val_steps"]
        val_losses = data["val_losses"]

        # 1. Training Loss Plot
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        ax.plot(train_steps, train_losses, color="#1e3a8a", alpha=0.6, label="Step Loss")
        
        # Smoothed moving average
        window = 30
        if len(train_losses) > window:
            smooth_loss = np.convolve(train_losses, np.ones(window)/window, mode='valid')
            smooth_steps = train_steps[window-1:]
            ax.plot(smooth_steps, smooth_loss, color="#2563eb", linewidth=2.0, label=f"Moving Avg (w={window})")

        ax.set_title(f"FastSpeech 2 {lang} Model — Training Loss vs Steps", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Training Steps", fontsize=11)
        ax.set_ylabel("Total Loss (MAE + MSE)", fontsize=11)
        ax.set_ylim(bottom=0.5, top=max(5.0, np.max(train_losses) * 1.05))
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(res_dir / "training_loss.png")
        plt.close()

        # 2. Validation Loss Plot
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        ax.plot(val_steps, val_losses, color="#b91c1c", marker="o", markersize=3, linewidth=1.8, label="Validation Loss")
        ax.set_title(f"FastSpeech 2 {lang} Model — Validation Loss vs Steps", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Validation Steps", fontsize=11)
        ax.set_ylabel("Validation Loss", fontsize=11)
        ax.set_ylim(bottom=0.5, top=max(5.0, np.max(val_losses) * 1.05))
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(res_dir / "validation_loss.png")
        plt.close()

    print("Successfully generated all 4 loss plots.")


if __name__ == "__main__":
    generate_loss_plots()
