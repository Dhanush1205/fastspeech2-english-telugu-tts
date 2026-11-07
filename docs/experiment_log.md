# Experiment Log

## FastSpeech 2 English & Telugu TTS

---

## Run 001 — Environment Setup
**Date:** 2026-11-07  
**Status:** ✅ Completed  

### Environment
- OS: Windows 11
- Python: 3.11.3
- PyTorch: 2.4.1+cpu (CPU ONLY — NO GPU)
- CUDA: Not available
- RAM: ~15.4 GB
- Disk: ~146 GB free

### Result
Local machine does not have GPU. Training will be done on Kaggle or Google Colab.

---

## Run 002 — Project Structure Creation
**Date:** 2026-11-07  
**Status:** ✅ Completed  

Created all required directories per assignment specification.

---

## Pending Runs (Requiring Kaggle GPU)

### Run 003 — English Preprocessing (Expected)
**Platform:** Kaggle
**Status:** ⏳ Not started — awaiting dataset download and Kaggle setup

### Run 004 — Telugu Preprocessing (Expected)  
**Platform:** Kaggle
**Status:** ⏳ Not started — awaiting dataset download and alignment tool decision

### Run 005 — English FastSpeech 2 Training (Expected)
**Platform:** Kaggle T4/P100 GPU
**Expected batch size:** 16
**Expected steps:** 200,000
**Expected time:** ~12-24 hours
**Status:** ⏳ Not started

### Run 006 — Telugu FastSpeech 2 Training (Expected)
**Platform:** Kaggle T4/P100 GPU
**Expected batch size:** 16
**Expected steps:** 200,000
**Expected time:** ~12-24 hours
**Status:** ⏳ Not started

### Run 007 — English Inference (Expected)
**Status:** ⏳ Not started — requires trained checkpoint

### Run 008 — Telugu Inference (Expected)
**Status:** ⏳ Not started — requires trained checkpoint

---

## Checkpoint Registry

| Model | Step | Val Loss | Location | Verified |
|-------|------|----------|----------|---------|
| English | - | - | English/checkpoints/ | ❌ |
| Telugu | - | - | Telugu/checkpoints/ | ❌ |

---

## Notes

Training logs will be added here as experiments execute.

