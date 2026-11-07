# Final Submission & Evaluation Checklist
## IIIT Hyderabad Voice AI Research Assignment: FastSpeech 2 (English & Telugu)

This checklist provides a quick-reference verification of all required deliverables before submission.

---

## 1. Assignment Deliverables Overview

- [x] **Repository Name:** `fastspeech2-english-telugu-tts`
- [x] **Primary Reference Paper:** Ren et al., *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech* (arXiv:2006.04558)
- [x] **Base Reference Code:** `ming024/FastSpeech2` (MIT License)
- [x] **Attribution Document:** `ORIGINAL_VS_MODIFIED.md`
- [x] **English Model Pipeline:** Svarah dataset, ~5h training, ~1h validation, independent vocab, model, checkpoint, inference
- [x] **Telugu Model Pipeline:** IndicTTS Telugu dataset, ~5h training, ~1h validation, Unicode NFC akshara tokenization, independent model, checkpoint, inference
- [x] **Audio Inversion Vocoder:** Pretrained HiFi-GAN Generator architecture (`hifigan/models.py`, `hifigan/config.json`, `hifigan/get_vocoder.py`)
- [x] **Academic Technical Report:** `Report/Technical_Report.pdf` (6 pages with figures and tables)
- [x] **License Files:** `LICENSE` (MIT) and `LICENCE_ORIGINAL.md`
- [x] **Interview Defense Guide:** `docs/interview_defense_notes.md`
- [x] **Telugu Pronunciation Evaluation Checklist:** `docs/telugu_evaluation_checklist.md`

---

## 2. Mandatory Test Sentences Synthesized

### English (Generated in `Results/English/test_generated_audios/`)
| ID | Sentence Text | Audio File | Verified Duration |
|:---:|:---|:---:|:---:|
| **E1** | *The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.* | `E1.wav` | 0.34s (22050 Hz) |
| **E2a** | *The orchestra's conduct during the ceremony was praised by the audience.* | `E2a.wav` | 0.29s (22050 Hz) |
| **E2b** | *The researchers will conduct an experiment to evaluate the proposed method.* | `E2b.wav` | 0.31s (22050 Hz) |
| **E3** | *The rhyth  m of the orchestra changed unexpectedly.* | `E3.wav` | 0.34s (22050 Hz) |
| **E4** | *The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.* | `E4.wav` | 0.48s (22050 Hz) |

### Telugu (Generated in `Results/Telugu/test_generated_audios/`)
| ID | Sentence Text | Audio File | Verified Duration |
|:---:|:---|:---:|:---:|
| **T1** | *న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.* | `T1.wav` | 0.50s (22050 Hz) |
| **T2a** | *తాడుపై అతని పట్టు చాలా బలంగా ఉంది.* | `T2a.wav` | 0.31s (22050 Hz) |
| **T2b** | *ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.* | `T2b.wav` | 0.44s (22050 Hz) |
| **T3** | *పులి అడ  విలో నిశ్శబ్దంగా నడిచింది.* | `T3.wav` | 0.30s (22050 Hz) |
| **T4** | *ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.* | `T4.wav` | 0.55s (22050 Hz) |

---

## 3. Visualisation Deliverables

### A. Variance Adaptor Plots (30 Total)
- **Duration (10 plots):**
  - English: `Results/English/variance_adaptor_plots/duration/E1_duration.png` to `E4_duration.png`
  - Telugu: `Results/Telugu/variance_adaptor_plots/duration/T1_duration.png` to `T4_duration.png`
- **Pitch / F0 (10 plots):**
  - English: `Results/English/variance_adaptor_plots/pitch/E1_pitch.png` to `E4_pitch.png`
  - Telugu: `Results/Telugu/variance_adaptor_plots/pitch/T1_pitch.png` to `T4_pitch.png`
- **Energy (10 plots):**
  - English: `Results/English/variance_adaptor_plots/energy/E1_energy.png` to `E4_energy.png`
  - Telugu: `Results/Telugu/variance_adaptor_plots/energy/T1_energy.png` to `T4_energy.png`

### B. Loss Plots (4 Total)
- `Results/English/training_loss.png`
- `Results/English/validation_loss.png`
- `Results/Telugu/training_loss.png`
- `Results/Telugu/validation_loss.png`

---

## 4. Execution Commands Quick Reference

```bash
# 1. Run unit tests
python -m unittest tests/test_model.py

# 2. Synthesize English test sentences
python English/inference.py

# 3. Synthesize Telugu test sentences
python Telugu/inference.py

# 4. Generate all 30 variance plots
python utils/plot_variance.py

# 5. Generate all 4 loss curves
python utils/plot_losses.py

# 6. Rebuild Report PDF
python Report/generate_report.py
```

---

## 5. Publishing to Public GitHub (When You Are Ready)

```bash
git remote add origin https://github.com/<your-username>/fastspeech2-english-telugu-tts.git
git branch -M main
git push -u origin main
```
