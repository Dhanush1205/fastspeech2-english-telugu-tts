# Dataset Notes

## FastSpeech 2 English & Telugu TTS

---

## 1. English Dataset: Svarah

### Source
- **Name:** Svarah
- **Provider:** AI4Bharat
- **HuggingFace:** https://huggingface.co/datasets/ai4bharat/Svarah
- **Paper:** Svarah: Evaluating English ASR Systems on Indian-Accented English
- **Licence:** CC BY 4.0

### Description
Svarah is a dataset of Indian-accented English speech, covering speakers from across India.
It was designed for ASR evaluation but is used here for TTS training.

### Key Properties (Expected)
- Language: English (Indian-accented)
- Speakers: Multiple (diverse Indian accents)
- Sampling rate: 16000 Hz (needs resampling to 22050 Hz for TTS)
- Format: WAV files + transcripts
- Approximate total duration: TBD (pending download)

### Target Usage
- Training: ~5 hours
- Validation: ~1 hour
- Split strategy: Random speaker-independent split (seed=42)

### Selection Strategy
1. Download full dataset
2. Compute duration of all utterances
3. Shuffle with seed=42
4. Select utterances until 5h training + 1h validation is reached
5. Ensure no speaker overlap between train/val if sufficient speakers
6. Save manifests: English/data_splits/train.txt, English/data_splits/val.txt

### Notes
- Indian-accented English may require MFA acoustic model fine-tuning
- Phonemization via g2p_en may mis-handle Indian names/words
- E4 test sentence contains "andaroo" — a Telugu-influenced English word

---

## 2. Telugu Dataset: IndicTTS Telugu

### Source
- **Name:** IndicTTS Telugu
- **Provider:** IIT Madras Speech Lab / AI4Bharat
- **Primary URL:** https://www.iitm.ac.in/donlab/tts/database.php
- **Alternative (HuggingFace):** https://huggingface.co/datasets/ai4bharat/IndicTTS (if available)
- **Licence:** Creative Commons (exact version TBD — check on download)

### Description
IndicTTS is a multi-speaker TTS corpus for Indian languages developed by IIT Madras.
The Telugu subset contains read speech from professional speakers.

### Key Properties (Expected)
- Language: Telugu
- Speakers: 1-2 professional speakers (male + female)
- Sampling rate: 48000 Hz or 16000 Hz (needs verification)
- Format: WAV files + transcription text files
- Approximate total duration: ~15+ hours total (Telugu)

### Target Usage
- Training: ~5 hours
- Validation: ~1 hour
- Split strategy: Utterance-level random split (seed=42)
  (speaker-independent split not applicable if only 1-2 speakers)

### Selection Strategy
1. Download Telugu subset
2. Verify sampling rate and format
3. Compute utterance durations
4. Shuffle with seed=42
5. Select first N utterances for train (until ~5h), next M for val (~1h)
6. Save manifests: Telugu/data_splits/train.txt, Telugu/data_splits/val.txt

### Dataset Access
⚠️ **IIT Madras IndicTTS may require form submission or account registration.**
If access requires manual authentication, user must complete this step.

---

## 3. Dataset Statistics Template

(To be filled after download)

### English (Svarah)
| Metric | Value |
|--------|-------|
| Total available duration | TBD |
| Training duration | ~5h |
| Validation duration | ~1h |
| Training samples (count) | TBD |
| Validation samples (count) | TBD |
| Number of speakers | TBD |
| Sampling rate (original) | TBD |
| Sampling rate (after resampling) | 22050 Hz |
| Format | WAV |
| Random seed | 42 |
| Split method | Random utterance split |

### Telugu (IndicTTS)
| Metric | Value |
|--------|-------|
| Total available duration | TBD |
| Training duration | ~5h |
| Validation duration | ~1h |
| Training samples (count) | TBD |
| Validation samples (count) | TBD |
| Number of speakers | TBD |
| Sampling rate (original) | TBD |
| Sampling rate (after resampling) | 22050 Hz |
| Format | WAV |
| Random seed | 42 |
| Split method | Random utterance split |

---

## 4. Test Sentences (Exact from PDF)

### English
| ID | Sentence |
|----|----------|
| E1 | The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings. |
| E2a | The orchestra's conduct during the ceremony was praised by the audience. |
| E2b | The researchers will conduct an experiment to evaluate the proposed method. |
| E3 | The rhyth  m of the orchestra changed unexpectedly. |
| E4 | The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time. |

### Telugu
| ID | Sentence |
|----|----------|
| T1 | న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు. |
| T2a | తాడుపై అతని పట్టు చాలా బలంగా ఉంది. |
| T2b | ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది. |
| T3 | పులి అడ  విలో నిశ్శబ్దంగా నడిచింది. |
| T4 | ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి. |

**Notes on Telugu sentences:**
- T2b: PDF has a line break before "ది" — reconstructed as single sentence
- T3: Has unusual double space in "అడ  విలో" — preserved as-is (mirrors E3's "rhyth  m")
- T4: PDF has line breaks mid-sentence — reconstructed as single sentence
- Please confirm T2b and T4 reconstruction is correct

---

## 5. Data Licence Summary

| Dataset | Licence | Commercial Use | Attribution Required |
|---------|---------|----------------|---------------------|
| Svarah (AI4Bharat) | CC BY 4.0 | Yes | Yes |
| IndicTTS Telugu | CC (TBD) | TBD | Yes |

**IMPORTANT:** Dataset audio files must NOT be committed to the public GitHub repository.
Only manifests (file lists with paths) should be committed.

