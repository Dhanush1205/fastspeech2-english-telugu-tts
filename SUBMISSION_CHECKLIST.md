# Final Submission & Technical Verification Checklist
## IIIT Hyderabad Voice AI Research Assignment: FastSpeech 2 (English & Telugu)

This checklist tracks the verification status of all components following the comprehensive technical audit.

---

## 1. Assignment Architecture & Implementation Status

- [x] **Repository Structure:** Clean, modular structure with independent English and Telugu directories.
- [x] **Primary Reference Paper:** Ren et al., *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech* (arXiv:2006.04558)
- [x] **Base Reference Code:** `ming024/FastSpeech2` (MIT License) with exact attribution in `ORIGINAL_VS_MODIFIED.md`.
- [x] **English Model Pipeline:** Configured for Svarah dataset (~5h training, ~1h validation), ARPABET phonemization, independent config, training, and inference.
- [x] **Telugu Model Pipeline:** Configured for IndicTTS Telugu dataset (~5h training, ~1h validation), Unicode NFC akshara tokenization, independent config, training, and inference.
- [x] **Telugu Duration Alignment:** Primary aligner configured as **AI4Bharat IndicMFA** (Grapheme-to-Grapheme Telugu acoustic model and dictionary) with CTC-Segmentation fallback.
- [x] **Audio Inversion Vocoder:** Pretrained HiFi-GAN Generator architecture (`hifigan/models.py`, `hifigan/config.json`, `hifigan/get_vocoder.py`) with Griffin-Lim algorithmic fallback. Strict error raising on inversion failure (sine-wave fallbacks removed).
- [x] **Unit & Integration Tests:** 10/10 automated tests passing (`tests/test_model.py`) covering checkpoint validation, vocoder failure handling, dataset strictness, plot utility error handling, text processing, and model forward/loss passes.
- [x] **Notebook Workflows:** Runnable Kaggle/Colab notebooks (`notebooks/01_` through `06_`) executing repository scripts with GPU detection and genuine logging.

---

## 2. Experimental Verification & GPU Training Status

| Component | Status | Verification Evidence |
|:---|:---:|:---|
| **Local Code & Unit Tests** | **VERIFIED** | 10 unit tests passing in 1.5s (`python -m unittest tests/test_model.py`) |
| **Fake Checkpoints & Fallbacks** | **CLEANED** | Removed dummy `200000.pth.tar` checkpoints, sine wave generators, and synthetic plots |
| **Remote GPU Training** | **PENDING EXECUTION** | Ready to run on Kaggle/Colab GPU (P100 / T4) via `notebooks/03_` and `04_` |
| **English Synthesis (E1–E4)** | **PENDING GPU RUN** | Will be populated upon checkpoint training (WAV + `.npz` predictions) |
| **Telugu Synthesis (T1–T4)** | **PENDING GPU RUN** | Will be populated upon checkpoint training (WAV + `.npz` predictions) |
| **Variance Adaptor Plots (30)** | **PENDING GPU RUN** | Will be generated via `utils/plot_variance.py` from genuine predictions |
| **Loss Curves (4)** | **PENDING GPU RUN** | Will be generated via `utils/plot_losses.py` from genuine `loss_history.npz` |

> [!NOTE]
> Following the technical audit, all placeholder audio (which previously produced ~0.3s artifacts from untrained weights and missing vocoder libraries) and artificial loss curves have been eliminated. Results will be populated strictly with genuine outputs from remote GPU training.

---

## 3. Mandatory Test Sentences Specification

### English Test Utterances
| ID | Sentence Text | Character Count | Acoustic Highlights |
|:---:|:---|:---:|:---|
| **E1** | *The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.* | 108 | Morphological complexity, polysyllabic stress |
| **E2a** | *The orchestra's conduct during the ceremony was praised by the audience.* | 71 | Noun heteronym: conduct (/ˈkɒndʌkt/) |
| **E2b** | *The researchers will conduct an experiment to evaluate the proposed method.* | 73 | Verb heteronym: conduct (/kənˈdʌkt/) |
| **E3** | *The rhyth  m of the orchestra changed unexpectedly.* | 50 | Typo / whitespace preservation ("rhyth  m") |
| **E4** | *The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.* | 90 | Numeric expansion + Telugu loanword ("andaroo") |

### Telugu Test Utterances
| ID | Sentence Text | Character Count | Linguistic Highlights |
|:---:|:---|:---:|:---|
| **T1** | *న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.* | 68 | Polysyllabic scientific loanword, conjuncts |
| **T2a** | *తాడుపై అతని పట్టు చాలా బలంగా ఉంది.* | 34 | Polysemous word: పట్టు (grip / hold) |
| **T2b** | *ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.* | 40 | Polysemous word: పట్టు (silk) |
| **T3** | *పులి అడ  విలో నిశ్శబ్దంగా నడిచింది.* | 34 | Whitespace preservation ("అడ  విలో") |
| **T4** | *ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.* | 92 | Date, time, Telugu numeral formatting |

---

## 4. Execution Commands Quick Reference

```bash
# 1. Run automated unit test suite (verifies model, tokenizer, strict error handling)
python -m unittest tests/test_model.py

# 2. Prepare datasets (accumulates 5.0h train / 1.0h val splits)
python data_prep/prepare_svarah.py --train_hours 5.0 --val_hours 1.0
python data_prep/prepare_indictts.py --train_hours 5.0 --val_hours 1.0

# 3. Download IndicMFA models for Telugu alignment
python Telugu/preprocess.py --download_indicmfa_models

# 4. Run preprocessing & feature extraction
python English/preprocess.py --config English/config.yaml
python Telugu/preprocess.py --config Telugu/config.yaml

# 5. Launch GPU training (remote instance)
python English/train.py -p English/config/preprocess.yaml -m English/config/model.yaml -t English/config/train.yaml
python Telugu/train.py -p Telugu/config/preprocess.yaml -m Telugu/config/model.yaml -t Telugu/config/train.yaml

# 6. Run inference on test sentences (requires trained checkpoint)
python English/inference.py --checkpoint English/checkpoints/checkpoint_final.pth.tar
python Telugu/inference.py --checkpoint Telugu/checkpoints/checkpoint_final.pth.tar

# 7. Generate variance plots and loss curves from genuine outputs
python utils/plot_variance.py
python utils/plot_losses.py
```
