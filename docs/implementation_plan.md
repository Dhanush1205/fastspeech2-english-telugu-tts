# Implementation Plan

## FastSpeech 2: English and Telugu TTS

**Reference Paper:** FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (Ren et al., 2020)  
**Reference Code:** https://github.com/ming024/FastSpeech2  

---

## Architecture Overview (from Paper)

FastSpeech 2 is a non-autoregressive TTS model that uses:

1. **Text Encoder** — Feed-forward Transformer (FFT) blocks
2. **Variance Adaptor** — Duration, Pitch, Energy predictors
3. **Length Regulator** — Expands encoded frames by predicted durations
4. **Mel-Spectrogram Decoder** — FFT blocks to predict mel frames
5. **Vocoder** — Converts mel-spectrogram to waveform (HiFi-GAN or similar)

Key improvement over FastSpeech 1:
- Uses real duration labels from forced alignment (not teacher model)
- Adds pitch and energy predictors for richer variance
- Directly predicts mel-spectrograms (not distilled from teacher)

---

## Implementation Phases

### Phase 0: Environment & Planning
- [x] Read assignment PDF
- [x] Audit local hardware (NO GPU — Kaggle required)
- [x] Create project skeleton
- [ ] Clone ming024/FastSpeech2
- [ ] Read reference implementation

### Phase 1: Reference Implementation Study
1. Clone https://github.com/ming024/FastSpeech2
2. Inspect: model/, utils/, config/, preprocessor/
3. Identify: which configs/preprocessors to extend for Svarah and Telugu

### Phase 2: English Pipeline (Svarah Dataset)

#### 2a. Dataset Download
- Svarah: https://huggingface.co/datasets/ai4bharat/Svarah
- Select ~5h train, ~1h validation
- Document speaker distribution, sampling rate, transcript format

#### 2b. English Preprocessing
- Phonemization: g2p_en or CMU pronunciation dictionary
- Alignment: Montreal Forced Aligner (MFA) with English acoustic model
- Mel-spectrogram: librosa or torchaudio
- Pitch: pyworld or torchcrepe
- Energy: RMS of mel-spectrogram

#### 2c. English Training
- Config: English/config.yaml (based on reference LJSpeech config)
- Adapt dataset loader for Svarah TSV/JSON format
- Run training on Kaggle GPU (T4 or P100)
- Expected: ~200k steps, ~12-24h on T4

### Phase 3: Telugu Pipeline (IndicTTS Telugu)

#### 3a. Dataset Download
- IndicTTS: https://www.iitm.ac.in/donlab/tts/database.php
- Alternative: HuggingFace ai4bharat/IndicTTS
- Select ~5h train, ~1h validation

#### 3b. Telugu Preprocessing — CRITICAL DECISIONS NEEDED

**Text Representation Options:**
- A. **Grapheme-based** — Use Unicode Telugu characters directly as tokens
  - Pros: No phoneme dictionary needed; works out of box
  - Cons: Grapheme ≠ phoneme; alignment may be harder
- B. **Phoneme-based via IPA/ARPABET** — Use phoneme converter
  - Pros: Linguistically accurate; better alignment
  - Cons: Requires a Telugu G2P system (limited availability)
- C. **Aksharas (syllable clusters)** — Segment by aksharas
  - Pros: Linguistically meaningful units for Dravidian languages
  - Cons: Requires aksharas segmentation library

**Recommendation: Start with grapheme-based (A) using Unicode normalization**
- Devanagari/Telugu Unicode normalization via `unicodedata`
- NFC normalization
- Character-level vocabulary
- Decision must be confirmed by user

**Duration Alignment for Telugu:**
- MFA does NOT support Telugu natively
- Options:
  1. Use CTC-Segmentation with a multilingual model (wav2vec2-based)
  2. Use forced alignment with IndicWav2Vec (AI4Bharat)
  3. Use grapheme-to-grapheme alignment with DTW
  4. Use energy-based VAD for coarse alignment
- **Awaiting user decision on alignment approach**

#### 3c. Telugu Training
- Config: Telugu/config.yaml
- Vocab: Telugu character set (~60-80 characters)
- Same FastSpeech 2 architecture, different vocab size

### Phase 4: Notebooks (Kaggle)
- 01_environment_setup.ipynb
- 02_dataset_preparation.ipynb
- 03_english_training.ipynb
- 04_telugu_training.ipynb
- 05_inference_and_evaluation.ipynb
- 06_variance_analysis.ipynb

### Phase 5: Inference and Evaluation
- Run inference on 10 test sentences
- Generate 30 variance adaptor plots
- Generate 4 loss plots
- Validate all audio files

### Phase 6: Technical Report
- Write Report/Technical_Report.pdf
- Use LaTeX or ReportLab

### Phase 7: GitHub Submission
- Audit for secrets and large files
- Prepare .gitignore
- Get user approval before push

---

## Configuration Strategy

| Parameter | English | Telugu |
|-----------|---------|--------|
| Input representation | Phonemes (g2p_en) | Graphemes (Unicode NFC) |
| Vocab size | ~70 phonemes | ~80-100 characters |
| Sampling rate | 22050 Hz | 22050 Hz |
| Mel channels | 80 | 80 |
| FFT size | 1024 | 1024 |
| Hop size | 256 | 256 |
| Win size | 1024 | 1024 |
| Aligner | MFA | CTC-Segmentation/IndicWav2Vec |
| Vocoder | HiFi-GAN | HiFi-GAN |

---

## Random Seeds for Reproducibility
- Dataset split seed: 42
- Training seed: 42

---

## Key Files to Modify in Reference Repo

| File | Modification |
|------|-------------|
| `config/LJSpeech/preprocess.yaml` | → `English/config.yaml` |
| `preprocessor/ljspeech.py` | → extend for Svarah |
| `text/__init__.py` | → extend for Telugu |
| `text/symbols.py` | → add Telugu character set |
| `dataset.py` | → adapt for both datasets |
| `train.py` | → parameterize for language |
| `synthesize.py` | → parameterize for language |

