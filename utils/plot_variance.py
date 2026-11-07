"""
Variance Adaptor Visualisation Utility
======================================

Generates 30 high-resolution plots for the predicted variance adaptor features:
  - English: 5 Duration, 5 Pitch (F0), 5 Energy plots (15 total)
  - Telugu:  5 Duration, 5 Pitch (F0), 5 Energy plots (15 total)
Total: 30 required plots.

Usage:
    python utils/plot_variance.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Configure matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Gautami', 'Nirmala UI']
plt.rcParams['axes.unicode_minus'] = False


def load_predictions(npz_file):
    """Loads prediction dictionary from .npz file, raising FileNotFoundError if missing."""
    npz_path = Path(npz_file)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Model prediction file '{npz_path}' not found. "
            f"Variance adaptor plots must be generated from genuine model predictions."
        )
    return np.load(npz_path, allow_pickle=True)


def plot_duration(durations, tokens, title, output_path):
    """Plots predicted phoneme/grapheme durations in frames."""
    n = min(len(durations), len(tokens))
    durations = durations[:n]
    tokens = tokens[:n]

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    x = np.arange(n)

    ax.bar(x, durations, color="#2b5c8f", alpha=0.85, edgecolor="#1a365d", width=0.6)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Tokens / Phoneme Position", fontsize=11)
    ax.set_ylabel("Duration (Mel Frames, ~11.6ms each)", fontsize=11)

    if n <= 40:
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    else:
        step = max(1, n // 25)
        ticks = np.arange(0, n, step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([tokens[i] for i in ticks], rotation=45, ha="right", fontsize=8)

    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_pitch(pitch, title, output_path, hop_length=256, sr=22050):
    """Plots predicted fundamental frequency F0 over time with voiced/unvoiced distinction."""
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    time = np.arange(len(pitch)) * (hop_length / sr)

    # Voiced mask
    voiced = pitch > 0
    pitch_clean = np.where(voiced, pitch, np.nan)

    ax.plot(time, pitch_clean, color="#e53e3e", linewidth=2.0, label="Predicted F0 (Hz)")
    ax.scatter(time[~voiced], np.zeros(np.sum(~voiced)), color="#a0aec0", s=6, label="Unvoiced / Silence")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Fundamental Frequency (Hz)", fontsize=11)
    ax.set_ylim(bottom=0, top=max(400, np.nanmax(pitch_clean) * 1.15 if np.any(voiced) else 300))
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_energy(energy, title, output_path, hop_length=256, sr=22050):
    """Plots predicted frame-level acoustic energy over time."""
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    time = np.arange(len(energy)) * (hop_length / sr)

    ax.plot(time, energy, color="#38a169", linewidth=2.0, label="Predicted Energy (RMS)")
    ax.fill_between(time, energy, color="#38a169", alpha=0.25)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Energy (L2 Norm)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def generate_all_variance_plots(results_root="Results"):
    """Scans generated test audios and produces all 30 plots."""
    for lang in ["English", "Telugu"]:
        pred_dir = Path(results_root) / lang / "test_generated_audios"
        plot_base = Path(results_root) / lang / "variance_adaptor_plots"
        
        sent_ids = ["E1", "E2a", "E2b", "E3", "E4"] if lang == "English" else ["T1", "T2a", "T2b", "T3", "T4"]

        for sid in sent_ids:
            npz_file = pred_dir / f"{sid}_predictions.npz"
            if not npz_file.exists():
                raise FileNotFoundError(
                    f"Model prediction file '{npz_file}' not found. "
                    f"Variance adaptor plots must be generated from genuine model predictions. "
                    f"Run {lang}/inference.py using a trained checkpoint before plotting."
                )

            data = np.load(npz_file, allow_pickle=True)
            pitch = data["pitch"]
            energy = data["energy"]
            duration = data["duration"]
            tokens = list(data["tokens"])

            # Generate 3 plots for each sentence
            plot_duration(
                durations=duration,
                tokens=tokens,
                title=f"{lang} Sentence {sid} - Predicted Token Durations",
                output_path=plot_base / "duration" / f"{sid}_duration.png",
            )
            plot_pitch(
                pitch=pitch,
                title=f"{lang} Sentence {sid} - Predicted Pitch (F0) Contour",
                output_path=plot_base / "pitch" / f"{sid}_pitch.png",
            )
            plot_energy(
                energy=energy,
                title=f"{lang} Sentence {sid} - Predicted Energy Contour",
                output_path=plot_base / "energy" / f"{sid}_energy.png",
            )

    print("Successfully generated all 30 Variance Adaptor plots (15 English, 15 Telugu).")


if __name__ == "__main__":
    generate_all_variance_plots()
