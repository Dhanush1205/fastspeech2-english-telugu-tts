# Known Limitations

## FastSpeech 2 English & Telugu TTS

---

## 1. Hardware Limitations

### L-001: No Local GPU
- **Description:** The local machine has no NVIDIA GPU (nvidia-smi not found, PyTorch CUDA=False).
- **Impact:** Training cannot be run locally. Requires Kaggle or Colab GPU.
- **Mitigation:** Kaggle notebooks prepared for remote training.
- **Status:** Active limitation.

---

## 2. Dataset Limitations

### L-002: Svarah is ASR-focused, not TTS-focused
- **Description:** Svarah was designed for ASR evaluation, not TTS training. It may have:
  - Multiple speakers with varied recording conditions
  - Some noise in recordings
  - Inconsistent loudness levels
- **Impact:** TTS models trained on multi-speaker data with varied conditions may produce less consistent output than single-speaker studio-recorded data.
- **Mitigation:** Normalize audio, filter low-quality utterances, possibly select single-speaker subset.

### L-003: Indian-accented English and "andaroo"
- **Description:** E4 contains "andaroo" — a Telugu word used in English context.
  g2p_en may not correctly phonemize this word.
- **Impact:** Incorrect pronunciation of "andaroo" in generated speech.
- **Mitigation:** Add custom pronunciation entry or accept approximation.

---

## 3. Telugu-Specific Limitations

### L-004: No Native Telugu Forced Aligner
- **Description:** Montreal Forced Aligner (MFA) does not support Telugu.
  CTC-Segmentation with IndicWav2Vec is used instead.
- **Impact:** Telugu duration alignments may be less precise than English MFA alignments.
  This can affect duration predictor quality and overall speech rhythm.
- **Mitigation:** Use CTC-Segmentation; validate alignment on small sample.

### L-005: Grapheme-Based Telugu (Not Phoneme-Based)
- **Description:** Telugu input uses Unicode graphemes rather than phonemes.
  Telugu graphemes can be ambiguous (same character, different pronunciation in context).
- **Impact:** Model must learn grapheme-to-acoustic mapping without explicit phoneme supervision.
  May reduce pronunciation accuracy for ambiguous grapheme clusters.
- **Mitigation:** NFC normalization; character-level vocabulary; monitor inference quality.

### L-006: Telugu Vocoder Quality
- **Description:** HiFi-GAN Universal checkpoint is trained on English data.
  It may produce suboptimal audio quality for Telugu mel-spectrograms.
- **Impact:** Generated Telugu speech may sound less natural compared to a Telugu-specific vocoder.
- **Mitigation:** Use Universal HiFi-GAN; note limitation in report; suggest Telugu vocoder as future work.

### L-007: Short Training Data May Limit Quality
- **Description:** 5 hours of training data is relatively limited for a TTS model.
  LJSpeech (the reference dataset) has ~24 hours from a single speaker.
- **Impact:** Model may not fully learn prosodic patterns. Voice quality may be average.
- **Mitigation:** Use data augmentation if needed; tune hyperparameters; document limitation.

---

## 4. Test Sentence Limitations

### L-008: Unusual Spacing in E3 and T3
- **Description:** E3 contains "rhyth  m" and T3 contains "అడ  విలో" — both with double spaces.
  This appears intentional (testing unusual tokenization).
- **Impact:** Phonemizer / tokenizer must handle this gracefully.
- **Mitigation:** Preserve text as-is; test tokenization behaviour; document result.

### L-009: Telugu T2b and T4 Line Breaks
- **Description:** T2b "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది." has a line break before "ది" in the PDF.
  T4 has two line breaks mid-sentence.
- **Impact:** May affect sentence boundary detection.
- **Mitigation:** Reconstructed as single sentences. User confirmation required.

---

## 5. Training Limitations

### L-010: Limited Training Budget on Kaggle
- **Description:** Kaggle provides ~30 GPU hours per week.
  Training two models for 200k steps each may exceed this.
- **Impact:** May need to reduce training steps or use multiple sessions.
- **Mitigation:** Checkpoint every 2000 steps; support training resumption; plan step count.

### L-011: 5-Hour Datasets vs. Full-Scale Production Models
- **Description:** FastSpeech 2 reference models typically use 24h (LJSpeech) or more.
  With 5 hours, model quality will be limited.
- **Impact:** Generated speech quality will be lower than state-of-the-art.
- **Mitigation:** Document this honestly in report; focus on correct implementation.

---

## 6. Evaluation Limitations

### L-012: No MOS Evaluation
- **Description:** Mean Opinion Score (MOS) requires human listening tests with multiple raters.
- **Impact:** Objective quality measurement will be limited to MCD, F0 RMSE.
- **Mitigation:** User (who speaks Telugu) can evaluate Telugu; document subjective observations.

---

## Proposed Improvements (Future Work)

1. Use a TTS-specific single-speaker dataset for more consistent English results
2. Train a Telugu-specific HiFi-GAN vocoder
3. Implement phoneme-based Telugu input using IndicPhones or Devanagari-to-IPA conversion
4. Use speaker embeddings for multi-speaker capability
5. Increase training data duration for both languages
6. Apply data augmentation (speed perturbation, pitch shifting)
7. Conduct formal MOS evaluation with native speakers
8. Investigate joint English-Telugu training with language conditioning

