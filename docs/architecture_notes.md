# Architecture Notes
# FastSpeech 2: English and Telugu TTS

## Source: FastSpeech 2 Paper (Ren et al., ICLR 2021)
**ArXiv:** https://arxiv.org/abs/2006.04558

---

## 1. Motivation

FastSpeech (v1) uses a teacher-autoregressive model (Transformer TTS) to:
1. Provide duration targets via attention alignment extraction
2. Provide mel-spectrogram targets (soft labels via knowledge distillation)

**Problems with FastSpeech 1:**
- Two-stage training pipeline (teacher + student) is complex
- Teacher attention alignment is often imprecise/noisy
- One-to-many mapping problem: same text → many valid prosodies
- Soft distilled mel targets are blurry / over-smoothed

**FastSpeech 2 Improvements:**
- Uses **forced alignment** (MFA) for precise duration supervision
- Directly predicts from **ground-truth mel-spectrograms** (no distillation)
- Adds **pitch (F0)** and **energy** as variance targets
- Introduces **continuous pitch/energy** (not quantized like FastSpeech 2s)
- Much faster training convergence

---

## 2. Architecture Components

### 2.1 Text Encoder
- Input: Phoneme sequence (or grapheme sequence)
- Embedding: Learned phoneme/character embeddings
- Positional Encoding: Sinusoidal
- N = 4 Feed-Forward Transformer (FFT) Blocks
- Each FFT Block:
  - Multi-Head Self-Attention (h=2 heads, d_model=256)
  - Position-wise Feed-Forward Network (conv1d-based, kernel=9 or 1)
  - Layer Normalization
  - Dropout

### 2.2 Variance Adaptor
Processes encoded phoneme representations to add prosodic variation.

#### Duration Predictor
- 2-layer 1D Conv + ReLU + LayerNorm + Dropout
- Linear projection to scalar duration per phoneme
- Loss: MSE on log-domain durations

#### Length Regulator
- Repeats each encoded frame by its predicted duration
- Input: [B, T_phone, d_model], durations [B, T_phone]
- Output: [B, T_mel, d_model]

#### Pitch Predictor
- Same architecture as Duration Predictor
- Predicts continuous F0 per frame (after length regulation)
- F0 extraction: PyWorld (DIO + StoneMask)
- Normalization: per-speaker mean/std
- Loss: MSE

#### Energy Predictor
- Same architecture as Duration Predictor
- Predicts L2-norm of mel frame magnitudes
- Normalization: mean/std across dataset
- Loss: MSE

### 2.3 Mel-Spectrogram Decoder
- Same FFT Block structure as encoder (N=4 blocks)
- Input: Length-regulated + pitch + energy conditioned frames
- Output: mel-spectrogram frames [B, T_mel, n_mel]
- Loss: MAE on mel-spectrograms

### 2.4 Postnet (optional)
- 5-layer 1D Conv
- Residual correction on mel output
- Loss: MAE on postnet output

---

## 3. Training Objective

Total loss = w_mel * L_mel + w_mel_post * L_mel_postnet + w_dur * L_dur + w_pitch * L_pitch + w_energy * L_energy

Default weights: all 1.0

Loss details:
- L_mel: MAE between predicted and ground-truth mel
- L_mel_postnet: MAE between postnet output and ground-truth mel
- L_dur: MSE on log(duration + 1)
- L_pitch: MSE on normalized F0
- L_energy: MSE on normalized energy

---

## 4. Vocoder Integration

The reference implementation supports:
- HiFi-GAN (primary)
- MelGAN
- WaveGlow (older)

Preferred: **HiFi-GAN** (Universal V1 pretrained checkpoint)
- Mel channels: 80
- Sampling rate: 22050 Hz
- Language-agnostic (works for both English and Telugu)

---

## 5. Key Hyperparameters (from ming024 LJSpeech config)

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| Encoder layers | 4 |
| Encoder heads | 2 |
| Encoder FFN kernel | 9 |
| Decoder layers | 4 |
| Decoder heads | 2 |
| Variance predictor layers | 2 |
| Variance predictor kernel | 3 |
| Variance predictor filter | 256 |
| Max length (frames) | 1000 |
| Mel channels | 80 |
| Sampling rate | 22050 |
| FFT size | 1024 |
| Hop size | 256 |
| Win size | 1024 |
| fmin | 0 |
| fmax | 8000 |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Warmup steps | 4000 |
| Total steps | 900000 (LJSpeech) |
| Optimizer | Adam (β1=0.9, β2=0.98, ε=1e-9) |

---

## 6. English-Telugu Adaptation Notes

### English
- Phoneme vocabulary: ~70 symbols (CMU ARPABET via g2p_en)
- Alignment: MFA with English acoustic model + CMU dictionary
- Vocab embedding size: 256

### Telugu
- Grapheme vocabulary: ~80-100 Telugu Unicode characters
- Alignment: CTC-Segmentation with IndicWav2Vec
- Telugu Unicode range: U+0C00–U+0C7F
- Special handling needed for:
  - Halant (్) conjunct consonants
  - Dependent vowel signs (మాత్రలు)
  - Anusvara (ం), Visarga (ః)
  - Independent vowels
  - Nukta characters

