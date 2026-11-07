# Implementation Decisions Log

## FastSpeech 2 English & Telugu TTS

---

## Decision 001: Telugu Text Representation
**Date:** 2026-11-07  
**Status:** DECIDED — Grapheme-based Unicode NFC  

**Options considered:**
- A. Grapheme-based Unicode characters directly
- B. Phoneme-based via IPA (using CharsiuG2P or espeak-ng)
- C. Akshara (syllable) based segmentation

**Decision:** Option A — Grapheme-based Unicode NFC

**Rationale:**
1. IndicMFA supports grapheme-level alignment for Telugu → TextGrid output compatible with existing FastSpeech2 preprocessor
2. Telugu script is relatively phonemically transparent (spelling closely matches pronunciation)
3. No high-quality, freely available Telugu G2P system with broad coverage exists
4. Avoids dependency on additional poorly-supported tools
5. The reference implementation's architecture accommodates any vocabulary size

**Consequences:**
- Telugu vocabulary: ~80-100 characters from U+0C00–U+0C7F
- Input sequence: Unicode characters → integer IDs
- Alignment: IndicMFA (grapheme-level) → TextGrid files

---

## Decision 002: Telugu Duration Alignment Tool
**Date:** 2026-11-07  
**Status:** DECIDED — IndicMFA (primary), CTC-Segmentation (fallback)  

**Options considered:**
- A. IndicMFA (AI4Bharat) — MFA-based, pre-trained Telugu model, outputs TextGrid
- B. CTC-Segmentation with IndicWav2Vec — CTC-based, needs output conversion
- C. Meta MMS forced aligner — multilingual, flexible
- D. No alignment (use attention-derived durations) — not supported in FastSpeech2

**Decision:** Option A — IndicMFA

**Rationale:**
1. IndicMFA is specifically designed for Indic languages including Telugu
2. Uses ~261h of Telugu training data
3. Outputs .TextGrid files directly compatible with `tgt` library used in original preprocessor
4. Same pipeline as English MFA → minimal code changes
5. Available at https://github.com/AI4Bharat/IndicMFA

**Fallback:** CTC-Segmentation with IndicWav2Vec if IndicMFA has installation issues

---

## Decision 003: Configuration Structure
**Date:** 2026-11-07  
**Status:** DECIDED — 3 separate YAML files per language (matching reference repo)  

**Decision:** Follow ming024 convention: separate model.yaml, preprocess.yaml, train.yaml

**Rationale:**
- The reference implementation reads 3 separate config files via argparse (-p/-m/-t)
- Maintaining this structure makes it easier to compare with original
- Allows independent modification of model/preprocessing/training parameters

**Action:** Restructure config files into:
- `English/config/preprocess.yaml`
- `English/config/model.yaml`
- `English/config/train.yaml`
- `Telugu/config/preprocess.yaml`
- `Telugu/config/model.yaml`
- `Telugu/config/train.yaml`

---

## Decision 004: Vocoder
**Date:** 2026-11-07  
**Status:** DECIDED — HiFi-GAN Universal V1  

**Decision:** Use HiFi-GAN UNIVERSAL_V1 pretrained checkpoint for both English and Telugu

**Rationale:**
- Language-agnostic mel-spectrogram inversion
- Compatible with 80 mel channels at 22050 Hz hop=256
- Same vocoder parameters as reference implementation
- No retraining required

**Source:** https://drive.google.com/drive/folders/1YuOoV3lO2-Hhn1F2HJ2aQ4S0LC1JdKLd  
**Limitation:** Telugu-specific HiFi-GAN fine-tune would be better but is out of scope

---

## Decision 005: English Dataset (Svarah) Handling
**Date:** 2026-11-07  
**Status:** DECIDED  

**Svarah properties:**
- 9.6h total, 117 speakers, Indian-accented English, 16kHz, HuggingFace
- ASR dataset (not TTS), multi-speaker, variable recording conditions

**Strategy:**
1. Download via HuggingFace datasets API
2. Resample 16kHz → 22050Hz
3. Random shuffle (seed=42), select 5h train + 1h val
4. Use g2p_en for phonemization
5. Use MFA with librispeech-lexicon for alignment

**Multi-speaker handling:** 
- For simplicity: treat as single speaker (no speaker conditioning)
- Alternative: use speaker embedding (adds complexity, not strictly required by assignment)

---

## Decision 006: Telugu Dataset (IndicTTS) Access
**Date:** 2026-11-07  
**Status:** DECIDED — Use SPRINGLab/IndicTTS_Telugu on HuggingFace  

**IndicTTS Telugu properties:**
- ~8.74h total (male ~4.47h + female ~4.27h), 48kHz studio
- Primary source: IIT Madras DON Lab
- HuggingFace mirror: https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu

**Strategy:**
1. Download via HuggingFace: `load_dataset("SPRINGLab/IndicTTS_Telugu")`
2. Resample 48kHz → 22050Hz
3. Select ~5h train + ~1h val (combined male+female or one speaker)
4. Use IndicMFA for alignment

**Note:** 8.74h total; 5h + 1h = 6h needed; 2.74h unused. 
This is tight — if any samples fail quality filtering, may fall slightly under 5h.

---

## Decision 007: Training Duration
**Date:** 2026-11-07  
**Status:** DECIDED — 200,000 steps (adjusted from 900,000)  

**Original LJSpeech training:** 900,000 steps on 24h single-speaker data

**Our datasets:** ~5h each, assignment says "train", doesn't specify step count

**Justified reduction:**
- 5h ≈ 1/5 of LJSpeech (24h)
- 200,000 steps should achieve reasonable convergence on 5h data
- Kaggle GPU limit (~30h/week) would be exceeded at 900,000 steps
- Step count per hour: assuming ~40 steps/sec on T4 → 200k steps ≈ 83 min

**Decision:** 200,000 steps per model (240,000 if time allows)

---

## Decision 008: Remote Training Platform
**Date:** 2026-11-07  
**Status:** DECIDED — Kaggle (primary), Colab (fallback)  

**Local hardware:** No GPU available. Training requires remote GPU.

**Kaggle advantages:**
- Free T4 GPU (30h/week), persistent storage, no CAPTCHA for scripts
- Supports dataset upload and version control
- Notebooks can be forked and run

**Decision:** Prepare Kaggle notebooks. User must manually run them.

---

## Decision 009: Config Structure Correction
**Date:** 2026-11-07
**Status:** CORRECTING — Initial single config.yaml replaced with 3-file structure

**Action:** Creating:
- English/config/preprocess.yaml
- English/config/model.yaml  
- English/config/train.yaml
- Telugu/config/preprocess.yaml
- Telugu/config/model.yaml
- Telugu/config/train.yaml

The original English/config.yaml and Telugu/config.yaml files are superseded.
They are retained as convenience files referencing the split configs.

