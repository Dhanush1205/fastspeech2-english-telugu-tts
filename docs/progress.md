# FastSpeech 2 Project Progress and Verification Log

**Project:** English (Svarah) and Telugu (IndicTTS) FastSpeech 2 pipelines  
**Assignment:** IIIT Hyderabad Voice AI assignment  
**Status:** Code fixes and focused dataset tests are in progress. Full data preparation, training and final synthesis are not yet verified.

## 1. Verified evidence

| Item | Status | Evidence |
|---|---|---|
| Repository access | Verified | GitHub branch `fix/tts-dataset-pipeline` and draft PR #1 |
| IndicTTS dataset schema inspection | Verified in Colab | `audio` returned as Hugging Face `AudioDecoder`; `text` and `gender` fields observed; no stable speaker ID exposed |
| IndicTTS decoder unit tests | Passed | Five regression tests initially passed in GitHub Actions; expanded tests are being rerun |
| Dataset preparation script syntax | Passed in CI | `py_compile` for Svarah and IndicTTS scripts |
| Initial smoke data before revised patch | Observed in Colab | 10 train / 4 validation records per language; this predates the revised Telugu decoder/split implementation |
| Full 5h train / 1h validation data | Pending | Final metadata must show target durations and zero utterance overlap |
| Full preprocessing/alignment | Pending | Must be run on the final full manifests |
| English and Telugu model training | Pending | No verified final checkpoints or complete training logs yet |
| Required inference audio and variance plots | Pending | Must be generated from trained checkpoints |

## 2. Dataset limitations and split semantics

### Svarah

The inspected public schema did not expose a unique speaker-ID field. The script derives a demographic composite from available fields. This can create disjoint demographic groups, but it is not a ground-truth speaker identity. Reports must use the term *demographic-proxy grouping* and must not claim verified speaker-disjoint evaluation.

### IndicTTS Telugu

The observed `SPRINGLab/IndicTTS_Telugu` schema contained `audio`, `text`, and `gender`. Its audio values were `datasets.features._torchcodec.AudioDecoder` objects. The schema did not expose a stable speaker ID. The corrected acquisition script decodes the actual object, resamples to 22,050 Hz, validates saved WAV/transcript files, creates an utterance-disjoint partition, and records speaker identity as unavailable.

## 3. Corrections made on the working branch

1. Reworked IndicTTS audio decoding for the observed AudioDecoder API.
2. Added waveform finite-value checks, mono conversion, sample-rate handling, resampling and saved-file verification.
3. Added a hard failure when full training/validation duration targets are not reached.
4. Explicitly documented the lack of stable Telugu speaker IDs.
5. Corrected Svarah's split metadata/log wording to identify the demographic proxy.
6. Updated Colab smoke-test reporting so it does not treat unavailable speaker identities as speaker-leakage pass/fail evidence.
7. Replaced an unsubstantiated report generator with evidence-driven reporting and removed the stale PDF that claimed completed training and pronunciation observations.
8. Added dataset-preparation unit tests and GitHub Actions validation.

## 4. Remaining assignment work

- Re-run the revised smoke tests in Colab.
- Prepare the complete 5h/1h dataset for both languages and verify manifests/durations.
- Verify alignment and acoustic feature extraction for both languages.
- Train each model, save genuine checkpoints and loss histories to persistent storage.
- Synthesize the five required sentences per language.
- Generate all 30 duration, pitch and energy prediction plots and the four loss plots.
- Complete listening-based pronunciation observations and cross-language analysis.
- Generate the final report from the measured artifacts.

## 5. Reporting policy

No training convergence, audio quality, pronunciation accuracy, or native-speaker evaluation is claimed without corresponding checkpoints, logs, audio files and human observations. Smoke tests validate plumbing only; they do not satisfy the full training-data or model-training requirements.
