# ORIGINAL vs MODIFIED Implementation

## FastSpeech 2: English and Telugu TTS
**Reference Repository:** https://github.com/ming024/FastSpeech2 (MIT License)  
**Original Paper:** FastSpeech 2 (Ren et al., ICLR 2021) — https://arxiv.org/abs/2006.04558

---

## Attribution

This project is based on the **ming024/FastSpeech2** implementation, which is licensed under the MIT License.  
Full licence text is reproduced in `LICENCE_ORIGINAL.md`.

Original authors of the reference implementation: ming024 and contributors.  
Original paper: Ren et al. (2021), "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech."

---

## Component Classification

### RETAINED (Unchanged from Original)
> Components taken verbatim from ming024/FastSpeech2

| Component | File(s) | Description |
|-----------|---------|-------------|
| FastSpeech2 model architecture | `model/FastSpeech2.py` | Core encoder-decoder architecture |
| FFT Blocks | `model/Modules.py` | Feed-Forward Transformer blocks |
| Variance Adaptor | `model/VarianceAdaptor.py` | Duration, pitch, energy predictors |
| Length Regulator | `model/VarianceAdaptor.py` | Upsamples encoded frames |
| Loss computation | `model/Loss.py` | MAE + MSE losses for all outputs |
| Training loop logic | `train.py` (core) | Optimizer, scheduler, gradient clip |
| HiFi-GAN vocoder integration | `vocoder/` | Pretrained vocoder inference |
| General utilities | `utils/model.py`, `utils/tools.py` | Checkpoint loading, padding |

### MODIFIED (Extended from Original)
> Components from original that required adaptation

| Component | Original File | Modified File | Change |
|-----------|--------------|---------------|--------|
| LJSpeech preprocessor | `preprocessor/ljspeech.py` | `English/preprocess.py` | Adapted for Svarah dataset format |
| English config | `config/LJSpeech/*.yaml` | `English/config.yaml` | Svarah-specific paths and parameters |
| Telugu config | *(new, based on LJSpeech structure)* | `Telugu/config.yaml` | Telugu-specific parameters |
| Text symbols | `text/symbols.py` | Extended | Added Telugu character set |
| Text init | `text/__init__.py` | Extended | Added Telugu grapheme processing |
| Dataset loader | `dataset.py` | Extended | Support for Svarah and IndicTTS formats |
| Training entry point | `train.py` | `English/train.py`, `Telugu/train.py` | Language-parameterized versions |
| Inference | `synthesize.py` | `English/inference.py`, `Telugu/inference.py` | Language-specific inference |

### NEWLY IMPLEMENTED (Not in Original)
> Components written from scratch for this project

| Component | File | Description |
|-----------|------|-------------|
| Svarah dataset preprocessing | `English/preprocess.py` | Svarah-specific pipeline |
| IndicTTS Telugu preprocessing | `Telugu/preprocess.py` | Telugu audio + text pipeline |
| Telugu text normalizer | `Telugu/text_processing.py` | Unicode NFC normalization, grapheme segmentation |
| Telugu grapheme vocabulary | `Telugu/vocabulary.py` | Character-level vocab for Telugu |
| English text normalization | `English/text_processing.py` | English normalization + g2p |
| Dataset split manifests | `English/data_splits/`, `Telugu/data_splits/` | Train/val manifests with seeds |
| CTC-Segmentation alignment | `Telugu/alignment.py` | Telugu duration alignment |
| Variance plot generator | `utils/plot_variance.py` | Plots duration, pitch, energy |
| Loss plot generator | `utils/plot_losses.py` | Training/validation loss curves |
| Kaggle training notebooks | `notebooks/*.ipynb` | Remote GPU training |
| Telugu pronunciation checklist | `docs/telugu_evaluation_checklist.md` | For human evaluation |
| Test suite | `tests/` | Unit tests for all components |
| Technical report | `Report/Technical_Report.pdf` | Full research report |

---

## Key Design Decisions for Telugu

### 1. Text Representation: Grapheme-Based
**Decision:** Use Unicode Telugu graphemes (NFC-normalized) as the input token sequence.  
**Rationale:**  
- No Telugu phoneme dictionary with ARPABET-equivalent coverage exists for IndicTTS.  
- Character-level graphemes are linguistically meaningful in Telugu (akshara units).  
- The reference implementation's architecture supports arbitrary vocabulary sizes.  
**Alternative Considered:** Phoneme-based via IPA (rejected: insufficient Telugu G2P coverage).

### 2. Duration Alignment: CTC-Segmentation
**Decision:** Use CTC-segmentation with AI4Bharat's IndicWav2Vec for Telugu alignment.  
**Rationale:**  
- Montreal Forced Aligner (MFA) does not support Telugu.  
- IndicWav2Vec provides pretrained Telugu ASR features.  
- CTC-segmentation extracts frame-level alignments from CTC outputs.  
**Alternative Considered:** DTW-based alignment (less accurate; rejected).

### 3. Vocoder: HiFi-GAN (Universal)
**Decision:** Use a universal HiFi-GAN pretrained checkpoint.  
**Rationale:**  
- Language-agnostic mel-spectrogram inversion.  
- Compatible with the reference implementation's mel parameters.  
**Limitation:** A vocoder trained specifically on Telugu speech may produce better results.

---

## Licence Compliance

The ming024/FastSpeech2 repository is licensed under the **MIT License**.  
This project retains the MIT licence for all retained/modified components.  
Newly implemented components are also released under the MIT License.

See: `LICENCE_ORIGINAL.md` for the original MIT licence text.

