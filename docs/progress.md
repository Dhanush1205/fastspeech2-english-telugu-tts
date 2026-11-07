# FastSpeech 2 Project Progress & Technical Audit Log

**Project:** FastSpeech 2-Based English and Telugu Text-to-Speech Synthesis  
**Institution:** IIIT Hyderabad — Voice AI Research Internship / RA  
**Status:** ✅ CODEBASE AUDITED, CORRECTED & UNIT-TESTED (Ready for Remote GPU Execution)

---

## 1. Compliance Audit & Verification Table

| Requirement / Component | Status | Evidence & File Location | Verification Method |
|:---|:---:|:---|:---|
| **Original Paper Studied** | ✅ Verified | `docs/architecture_notes.md` | arXiv:2006.04558 architecture, loss formulation, variance adaptor documented |
| **Reference Repo Studied** | ✅ Verified | `ORIGINAL_VS_MODIFIED.md` | Full directory & code inspection of `ming024/FastSpeech2` |
| **English Dataset Pipeline (Svarah)** | ✅ Verified | `data_prep/prepare_svarah.py`, `English/preprocess.py` | Hugging Face streaming loader, ~5h train, ~1h val split |
| **Telugu Dataset Pipeline (IndicTTS)** | ✅ Verified | `data_prep/prepare_indictts.py`, `Telugu/preprocess.py` | Hugging Face streaming loader, ~5h train, ~1h val split |
| **English Preprocessing** | ✅ Verified | `English/preprocess.py`, `English/text_processing.py` | 22050 Hz resampling, 80-channel mel, PyWorld F0, energy, ARPABET phonemization |
| **Telugu Preprocessing & Tokenization** | ✅ Verified | `Telugu/preprocess.py`, `Telugu/text_processing.py` | Unicode NFC normalization, Telugu akshara grapheme vocabulary (100 tokens) |
| **Telugu Alignment Pipeline** | ✅ Verified | `Telugu/preprocess.py`, `Telugu/config.yaml` | Primary aligner: AI4Bharat IndicMFA (G2G model + dictionary); Fallback: CTC-Segmentation |
| **Model Architecture** | ✅ Verified | `model/fastspeech2.py`, `model/modules.py` | Encoder, Variance Adaptor, Decoder, PostNet verified in forward pass |
| **Loss & Optimization** | ✅ Verified | `model/loss.py`, `model/optimizer.py` | 5-part loss (mel, postnet, duration, pitch, energy) & Noam ScheduledOptim |
| **Audio Inversion Strictness** | ✅ Verified | `English/inference.py`, `Telugu/inference.py` | HiFi-GAN with Griffin-Lim fallback; removed all synthetic sine-wave tone generators |
| **Checkpoint Integrity** | ✅ Verified | `English/inference.py`, `Telugu/inference.py` | Strict validation of state_dict; rejects missing or corrupted checkpoints |
| **Unit & Integration Tests** | ✅ Verified | `tests/test_model.py` | 10/10 automated tests passing (checkpoint validation, inversion error, dataset, plotting) |
| **Visualization Strictness** | ✅ Verified | `utils/plot_variance.py`, `utils/plot_losses.py` | Raises descriptive FileNotFoundError when predictions or logs are absent (no fake plots) |
| **Kaggle / Colab Notebooks** | ✅ Verified | `notebooks/` (01 to 06) | Configured to execute actual project scripts on remote GPU |
| **Attribution & Licenses** | ✅ Verified | `ORIGINAL_VS_MODIFIED.md`, `LICENSE`, `LICENCE_ORIGINAL.md` | Clear MIT licensing and explicit attribution to ming024 & Ren et al. |
| **Human Evaluation Guide** | ✅ Verified | `docs/telugu_evaluation_checklist.md` | Pronunciation evaluation criteria for Telugu native listeners |
| **GPU Model Checkpoints** | ⏳ Pending GPU | `English/checkpoints/`, `Telugu/checkpoints/` | Ready to train on Kaggle/Colab GPU |
| **Evaluation Synthesis & Plots** | ⏳ Pending GPU | `Results/` | Will be generated upon completing remote GPU training |

---

## 2. Key Corrections Completed in Technical Audit

1. **Eliminated Synthetic / Sine Wave Audio:**
   - Removed artificial 220Hz sine-wave generators from `English/inference.py` and `Telugu/inference.py`. Audio inversion now strictly requires a functional neural vocoder or librosa Griffin-Lim, raising a clear `RuntimeError` on failure.
2. **Eliminated Fabricated Variance & Loss Plots:**
   - Removed artificial random curve generators from `utils/plot_variance.py` and exponential decay generators from `utils/plot_losses.py`. Missing prediction `.npz` or training log `.npz` files now raise descriptive `FileNotFoundError` exceptions.
3. **Strict Checkpoint Validation:**
   - Removed fallback to random uninitialized weights in `English/inference.py` and `Telugu/inference.py`. Replaced dummy `200000.pth.tar` checkpoints with clean README documentation. Checkpoints must be genuinely trained before inference is executed.
4. **Resolved Short Audio Duration Artifacts:**
   - Identified root cause of ~0.3s generated audio: untrained FastSpeech 2 weights predict values near 0 ($e^0 - 1 = 0$), clamping to 0–1 frames per character (~30 frames $\approx 0.35$s), followed by fallback to a sine wave. Removed both the fake weights and the sine fallback.
5. **Harmonized Telugu Alignment (IndicMFA Primary):**
   - Configured official AI4Bharat IndicMFA as primary Telugu alignment approach (using `Telugu_Acoustic_Model.zip` and `Telugu_Dictionary_g2g.txt` from AI4Bharat IndicMFA releases).
   - Implemented `parse_indicmfa_textgrid()` with `tgt` for grapheme duration extraction.
   - Retained CTC-segmentation as a secondary fallback.
6. **Harmonized Feature Representation:**
   - Configured `pitch.feature: "frame_level"` and `energy.feature: "frame_level"` consistently across configs, preprocessing, dataset loading, and model architecture.
7. **Telugu EOS Alignment Consistency:**
   - Aligned `<eos>` token with duration length in both dataset loading and inference.
8. **Automated Test Suite Expansion:**
   - Expanded `tests/test_model.py` to 10 comprehensive tests covering checkpoint rejection, audio inversion error handling, prediction file validation, dataset strictness, and model execution. All 10 tests pass.
