# Submission Checklist and Verification Status

**Assignment:** FastSpeech 2 for English (Svarah) and Telugu (IndicTTS)  
**Status:** Implementation fixes under validation; full experiments are not complete.

This checklist separates code-level checks from dataset and GPU experiment evidence. Passing a unit test does not establish that either model has been trained.

## 1. Code and pipeline checks

- [x] Separate English and Telugu model/configuration directories are present.
- [x] Dataset acquisition scripts exist for AI4Bharat Svarah and SPRINGLab IndicTTS Telugu.
- [x] IndicTTS acquisition supports the observed Hugging Face AudioDecoder record type.
- [x] Dataset preparation checks waveform validity, sample rate, resampling and saved WAV readability.
- [x] Telugu preparation documents that the inspected source schema does not expose a stable speaker ID.
- [x] Svarah split reporting distinguishes demographic composite groups from verified speaker identities.
- [x] Dataset preparation unit tests are included in GitHub Actions.
- [x] Report generation is evidence-driven and no longer inserts synthetic loss plots or invented pronunciation observations.
- [ ] Re-run the revised pipeline smoke tests in the live Colab environment.
- [ ] Complete an end-to-end preprocessing/alignment test for each language.

## 2. Dataset requirements

The assignment target is approximately 5 hours of training audio and 1 hour of validation audio for each language. These targets are not complete until final manifests and metadata show the achieved durations and zero utterance-ID overlap.

| Requirement | English / Svarah | Telugu / IndicTTS |
|---|---|---|
| Upstream dataset | ai4bharat/Svarah | SPRINGLab/IndicTTS_Telugu |
| Upstream split inspected | test | train |
| Local partition strategy | Deterministic demographic-proxy grouping; not verified speaker IDs | Deterministic utterance-disjoint partition |
| Stable speaker ID in observed schema | No | No |
| Full 5h train / 1h validation verified | Pending | Pending |
| Final utterance overlap check | Pending on revised full manifests | Pending on revised full manifests |

Important: the Svarah demographic composite is not a unique speaker identifier. The inspected IndicTTS features are audio, text, and gender; they do not include a stable speaker identifier. Neither pipeline may claim true speaker-disjoint evaluation on the basis of these fields alone.

## 3. GPU experiments and deliverables

| Deliverable | Status |
|---|---|
| English preprocessing and duration alignment on full dataset | Pending |
| Telugu preprocessing and duration alignment on full dataset | Pending |
| English FastSpeech 2 training and genuine checkpoint | Pending |
| Telugu FastSpeech 2 training and genuine checkpoint | Pending |
| Training/validation loss histories for both models | Pending |
| Five English assignment WAVs (E1, E2a, E2b, E3, E4) | Pending |
| Five Telugu assignment WAVs (T1, T2a, T2b, T3, T4) | Pending |
| Genuine prediction arrays for all 10 sentences | Pending |
| 30 duration/pitch/energy variance-adaptor plots | Pending |
| Human listening review / pronunciation observations | Pending |
| Final report based on actual experiment evidence | Pending |

A smoke test is only a small technical check. It is not a substitute for the assignment's full-duration training and validation datasets.

## 4. Reproduction commands

Run from the repository root after the revised branch is checked out:

    python -m unittest tests.test_dataset_preparation -v

    python data_prep/prepare_svarah.py --output_dir data/svarah --splits_dir English/data_splits --train_hours 5.0 --val_hours 1.0

    python data_prep/prepare_indictts.py --output_dir data/indictts_telugu --splits_dir Telugu/data_splits --train_hours 5.0 --val_hours 1.0

Do not start full training until both datasets reach their requested durations, manifests are verified, and preprocessing/alignment completes without fabricated or fallback features.

## 5. Reporting policy

Report/generate_report.py reads local split metadata and artifact presence. It reports missing evidence as pending. The previous pre-generated PDF was removed because it contained unsupported claims about completed training, generated speech and pronunciation quality.

Final submission should include real checkpoints, audio files, training histories, plots and a report generated from those artifacts.
