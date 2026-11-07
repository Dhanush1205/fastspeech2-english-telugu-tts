# Environment Setup and Hardware Audit

## FastSpeech 2: English and Telugu TTS System

---

## 1. Local Hardware & System Audit

An empirical audit of the local development environment was conducted:

| Parameter | Observed Local State | Implications for Training |
|:---|:---|:---|
| **Operating System** | Windows 11 Home (x86_64) | Fully compatible with PyTorch and Python audio libraries |
| **Python Version** | Python 3.11.3 | Compatible with PyTorch 2.x and modern dependencies |
| **Physical Memory (RAM)** | ~16.0 GB (15.4 GB available) | Sufficient for text processing, evaluation, and plotting |
| **Storage (C:)** | ~146 GB Free Space | Sufficient for datasets, intermediate mels, and checkpoints |
| **NVIDIA GPU Availability** | ❌ None Detected (`nvidia-smi` not found) | Local GPU training is not possible |
| **PyTorch Accelerator** | `torch.cuda.is_available() == False` (`cpu` only) | Training requires remote GPU compute |

---

## 2. Remote GPU Strategy (Kaggle & Google Colab)

Because training two 200,000-step FastSpeech 2 models requires significant GPU acceleration (~12–18 hours on an NVIDIA Tesla T4 or P100 per language), a dual-environment architecture is implemented:
- **Local Workspace:** Code orchestration, feature extraction scripts, text normalizers, evaluation, visualization utilities, unit testing, and technical report generation.
- **Remote GPU Environment:** Model training and batch loss optimization via the 6 prepared notebooks in [`notebooks/`](../notebooks/).

### Recommended Hardware Specifications for Training:
- **GPU:** NVIDIA Tesla T4 (16GB VRAM), P100 (16GB VRAM), or A100.
- **VRAM Required:** Minimum 12GB for batch size 16 at max sequence length 1000.
- **CUDA Version:** 11.8 or 12.1+.
- **Mixed Precision:** FP16 enabled if VRAM is constrained (`torch.cuda.amp.autocast`).

---

## 3. Installation Guide

### Step 1: Virtual Environment Creation
```bash
# Clone the repository
git clone https://github.com/saidh/fastspeech2-english-telugu-tts.git
cd fastspeech2-english-telugu-tts

# Create a clean virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Linux / macOS:
source venv/bin/activate
```

### Step 2: Install Core Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Verify Core Libraries
```python
import torch
import torchaudio
import librosa
import soundfile
import pyworld
import tgt
import transformers

print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("All audio libraries imported cleanly!")
```

---

## 4. Troubleshooting & Known Platform Gotchas

### Matplotlib Headless Execution on Windows
When running plotting scripts (e.g., `utils/plot_variance.py` or `utils/plot_losses.py`) in non-GUI terminal sessions, Matplotlib may raise `ValueError: PyCapsule_New called with null pointer` if attempting to attach to a Tkinter window.
**Resolution:** Explicitly set the headless backend before importing pyplot:
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

### PyTorch Serialization (`weights_only`)
In PyTorch 2.4+, `torch.load()` defaults to `weights_only=False` with a deprecation warning. Both English and Telugu inference pipelines handle checkpoint loading cleanly across CPU and CUDA devices.

### MFA / Alignment Tools on Windows
Montreal Forced Aligner (MFA) requires Kaldi binaries which are easiest to run under Conda or Linux. For Windows or Kaggle notebooks, pre-computed alignments or HuggingFace CTC-segmentation models (`ai4bharat/indicwav2vec_v1_telugu`) avoid native binary compilation issues.
