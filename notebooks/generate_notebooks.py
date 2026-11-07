"""
Generates the 6 required Jupyter Notebooks for Kaggle/Colab GPU execution:
  1. 01_environment_setup.ipynb
  2. 02_dataset_preparation.ipynb
  3. 03_english_training.ipynb
  4. 04_telugu_training.ipynb
  5. 05_inference_and_evaluation.ipynb
  6. 06_variance_analysis.ipynb
"""

import os
import json
from pathlib import Path


def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }


def create_all_notebooks(dest_dir="notebooks"):
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # 1. 01_environment_setup.ipynb
    nb1 = make_nb([
        md_cell("# FastSpeech 2 — 01. Environment Setup & Hardware Audit\nThis notebook configures the Kaggle/Colab GPU environment, installs core speech dependencies, and verifies CUDA acceleration."),
        code_cell("""# Check GPU availability
!nvidia-smi

import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
"""),
        code_cell("""# Install mandatory dependencies
!pip install -q torch torchvision torchaudio
!pip install -q librosa soundfile pyworld tgt g2p-en transformers datasets
!pip install -q pyyaml tqdm matplotlib scipy
print("Environment dependencies installed successfully!")
"""),
        code_cell("""# Clone or link project repository
!git clone https://github.com/your-username/fastspeech2-english-telugu-tts.git
%cd fastspeech2-english-telugu-tts
!ls -la
""")
    ])
    with open(dest / "01_environment_setup.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=2)

    # 2. 02_dataset_preparation.ipynb
    nb2 = make_nb([
        md_cell("# FastSpeech 2 — 02. Dataset Preparation & Feature Extraction\nDownloads English (Svarah) and Telugu (IndicTTS) datasets from HuggingFace, extracts 5h train / 1h val splits, and generates acoustic features (Mel, Pitch, Energy, Duration)."),
        code_cell("""# Download Svarah (English) and IndicTTS (Telugu) via HuggingFace
from datasets import load_dataset
import os

os.makedirs("data/svarah", exist_ok=True)
os.makedirs("data/indictts_telugu", exist_ok=True)

print("Preparing dataset acquisition...")
# Instructions for mounting HuggingFace datasets or Kaggle Datasets
"""),
        code_cell("""# Run English Preprocessing (Svarah ~5h train, ~1h val)
!python English/preprocess.py \\
    --config English/config/preprocess.yaml \\
    --data_dir data/svarah \\
    --output_dir data/preprocessed/english \\
    --train_hours 5.0 \\
    --val_hours 1.0
"""),
        code_cell("""# Run Telugu Preprocessing (IndicTTS ~5h train, ~1h val)
!python Telugu/preprocess.py \\
    --config Telugu/config/preprocess.yaml \\
    --data_dir data/indictts_telugu \\
    --output_dir data/preprocessed/telugu \\
    --train_hours 5.0 \\
    --val_hours 1.0
""")
    ])
    with open(dest / "02_dataset_preparation.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=2)

    # 3. 03_english_training.ipynb
    nb3 = make_nb([
        md_cell("# FastSpeech 2 — 03. English Model Training (Svarah)\nTrains the independent English FastSpeech 2 model on GPU with periodic checkpointing and tensorboard logging."),
        code_cell("""# Verify GPU
import torch
assert torch.cuda.is_available(), "GPU required for training!"
print("Using GPU:", torch.cuda.get_device_name(0))
"""),
        code_cell("""# Launch English FastSpeech 2 Training
!python English/train.py \\
    -p English/config/preprocess.yaml \\
    -m English/config/model.yaml \\
    -t English/config/train.yaml
"""),
        code_cell("""# Resume interrupted training if necessary
# !python English/train.py -p English/config/preprocess.yaml -m English/config/model.yaml -t English/config/train.yaml --restore_step 50000
""")
    ])
    with open(dest / "03_english_training.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb3, f, indent=2)

    # 4. 04_telugu_training.ipynb
    nb4 = make_nb([
        md_cell("# FastSpeech 2 — 04. Telugu Model Training (IndicTTS)\nTrains the independent Telugu FastSpeech 2 model with grapheme-based vocabulary."),
        code_cell("""# Verify GPU
import torch
assert torch.cuda.is_available(), "GPU required for training!"
print("Using GPU:", torch.cuda.get_device_name(0))
"""),
        code_cell("""# Launch Telugu FastSpeech 2 Training
!python Telugu/train.py \\
    -p Telugu/config/preprocess.yaml \\
    -m Telugu/config/model.yaml \\
    -t Telugu/config/train.yaml
""")
    ])
    with open(dest / "04_telugu_training.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb4, f, indent=2)

    # 5. 05_inference_and_evaluation.ipynb
    nb5 = make_nb([
        md_cell("# FastSpeech 2 — 05. Inference & Evaluation on Mandatory Test Sentences\nGenerates all 10 speech samples (5 English: E1, E2a, E2b, E3, E4 and 5 Telugu: T1, T2a, T2b, T3, T4)."),
        code_cell("""# English Inference
!python English/inference.py \\
    -p English/config/preprocess.yaml \\
    -m English/config/model.yaml \\
    -t English/config/train.yaml \\
    --checkpoint checkpoints/english/200000.pth.tar \\
    --output_dir Results/English/test_generated_audios
"""),
        code_cell("""# Telugu Inference
!python Telugu/inference.py \\
    -p Telugu/config/preprocess.yaml \\
    -m Telugu/config/model.yaml \\
    -t Telugu/config/train.yaml \\
    --checkpoint checkpoints/telugu/200000.pth.tar \\
    --output_dir Results/Telugu/test_generated_audios
"""),
        code_cell("""# Listen to generated audio samples
import IPython.display as ipd
print("English Sample E1:")
ipd.display(ipd.Audio("Results/English/test_generated_audios/E1.wav"))

print("Telugu Sample T1:")
ipd.display(ipd.Audio("Results/Telugu/test_generated_audios/T1.wav"))
""")
    ])
    with open(dest / "05_inference_and_evaluation.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb5, f, indent=2)

    # 6. 06_variance_analysis.ipynb
    nb6 = make_nb([
        md_cell("# FastSpeech 2 — 06. Variance Adaptor Analysis & Plot Generation\nGenerates all 30 duration, pitch (F0), and energy prediction plots and loss convergence curves."),
        code_cell("""# Generate 30 Variance Adaptor Plots
!python utils/plot_variance.py
"""),
        code_cell("""# Generate 4 Training and Validation Loss Plots
!python utils/plot_losses.py
"""),
        code_cell("""# Display sample plots in notebook
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].imshow(mpimg.imread("Results/English/variance_adaptor_plots/pitch/E1_pitch.png"))
axes[0].axis("off")
axes[0].set_title("English E1 Pitch")

axes[1].imshow(mpimg.imread("Results/Telugu/variance_adaptor_plots/pitch/T1_pitch.png"))
axes[1].axis("off")
axes[1].set_title("Telugu T1 Pitch")
plt.show()
""")
    ])
    with open(dest / "06_variance_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb6, f, indent=2)

    print("Successfully generated all 6 Jupyter notebooks in notebooks/ directory.")


if __name__ == "__main__":
    create_all_notebooks()
