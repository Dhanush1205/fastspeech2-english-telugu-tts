# Voice AI Interview Defense & Technical Concept Guide
## IIIT Hyderabad Research Internship / RA Technical Interview Preparation

This document breaks down the foundational speech processing and deep learning concepts of this project into clear, interview-ready answers. Study this guide to confidently explain, demonstrate, and defend the project during technical interviews.

---

## 1. Core Architecture Questions

### Q1: What is the fundamental difference between FastSpeech 1 and FastSpeech 2?
> **Answer:**
> - **Supervision & Distillation:** FastSpeech 1 relies on **knowledge distillation** from an autoregressive teacher model (like Transformer TTS or Tacotron 2) to obtain soft mel-spectrogram targets and attention-extracted duration labels. This causes information loss and blurry spectrograms. FastSpeech 2 directly trains on **ground-truth mel-spectrograms** and **hard duration targets** from an external forced aligner (like Montreal Forced Aligner).
> - **Variance Adaptor:** FastSpeech 1 only predicts duration. FastSpeech 2 introduces explicit predictors for **pitch (fundamental frequency F0)** and **energy**, solving the one-to-many mapping problem (the fact that a single text sentence can be spoken with many valid pitch and energy variations).
> - **Training Speed:** FastSpeech 2 trains in a single stage without needing to pre-train a separate teacher model first.

### Q2: What are Feed-Forward Transformer (FFT) blocks and how do they differ from standard Transformers?
> **Answer:**
> In standard NLP Transformers, the feed-forward network (FFN) consists of two dense linear layers applied point-wise (`Linear → ReLU → Linear`).  
> In FastSpeech 2's FFT blocks, the FFN is replaced with **1D Convolutions** (typically a kernel of size 9 followed by a kernel of size 1). This allows the model to capture local context across neighbouring phonemes/frames within each attention layer.

### Q3: What is the Length Regulator and how does it function during training vs inference?
> **Answer:**
> - **Purpose:** The text encoder outputs representations at the phoneme rate ($T_{\text{phone}}$), but the mel decoder must generate frames at the acoustic rate ($T_{\text{mel}}$). The Length Regulator bridges this temporal mismatch by repeating each hidden vector $h_i$ by its duration $d_i$.
> - **Training vs Inference:**
>   - *During Training:* It uses **ground-truth durations** obtained from forced alignment (MFA/IndicMFA), ensuring exact temporal alignment with the ground-truth audio target.
>   - *During Inference:* It uses **predicted durations** from the Duration Predictor (rounded to integers: $\text{round}(\exp(\hat{d}) - 1)$), allowing autonomous speech pacing without any reference audio.

### Q4: Why are Pitch and Energy quantized into 256 bins rather than added directly as continuous values?
> **Answer:**
> Adding raw continuous scalar values (e.g., F0 in Hz like 180.5) directly to high-dimensional hidden vectors ($d_{\text{model}} = 256$) has weak expressive capacity and can cause training instability. By quantizing values into 256 discrete bins and passing them through a learned **Embedding layer** ($\mathbb{R}^{256} \to \mathbb{R}^{d_{\text{model}}}$), the model projects pitch and energy into dense representation vectors that can interact non-linearly with the phonetic hidden states.

---

## 2. Telugu Speech & Language Adaptation

### Q5: Why did you choose a Grapheme-based representation for Telugu instead of Phonemes?
> **Answer:**
> 1. **Lack of Exhaustive Pronouncing Dictionaries:** Unlike English (which has the CMU Pronouncing Dictionary with over 130,000 words), Dravidian languages lack standardized, freely licensed open-source G2P dictionaries with complete vocabulary coverage.
> 2. **Phonemic Transparency:** Telugu orthography is inherently phonetic and syllabic (akshara-based). Unlike English (where "cough", "rough", "through" have identical letter sequences with completely different pronunciations), Telugu letters map almost 1:1 to their acoustic realization.
> 3. **Architectural Flexibility:** FastSpeech 2's text encoder is vocabulary-agnostic. A 100-token Unicode NFC character vocabulary allows the self-attention heads to learn acoustic-phonetic mappings directly from the 5 hours of studio recordings.

### Q6: How does the model handle Telugu conjunct consonants (సంయుక్తాక్షరాలు / Ottulu)?
> **Answer:**
> Telugu conjuncts are formed using the Halant/Virama sign (`్`, U+0C4D), e.g., `క` + `్` + `క` = `క్క`.  
> In our pipeline, Unicode NFC canonical normalization preserves virama markers as distinct tokens. During duration alignment and variance adaptation, geminates (ottulu) receive extended duration targets (often 2× normal frame length), which correctly simulates the lengthened oral closure characteristic of geminate stops in Telugu.

---

## 3. Acoustic Processing & Vocoding

### Q7: Explain the Mel-Spectrogram feature extraction parameters used in this project.
> **Answer:**
> - **Sampling Rate ($f_s$):** 22,050 Hz (standard for speech synthesis).
> - **FFT Size ($N_{\text{fft}}$):** 1024 points (~46.4 ms window).
> - **Hop Length ($H$):** 256 points (~11.6 ms frame shift).
> - **Window Length:** 1024 points (Hann window).
> - **Mel Channels:** 80 channels distributed across 0 Hz to 8000 Hz.
> - **Relationship:** At $f_s = 22050$ Hz and $H = 256$, 1 second of speech corresponds to approximately $22050 / 256 \approx 86.13$ mel frames.

### Q8: What vocoder was used and why?
> **Answer:**
> We used **HiFi-GAN (Universal V1)**. HiFi-GAN is a generative adversarial network (GAN) that inverts mel-spectrograms into high-fidelity time-domain waveforms. The Universal V1 checkpoint is pre-trained on multi-speaker, varied speech data and accepts standard 80-channel mel-spectrograms at 22,050 Hz, making it language-agnostic and capable of synthesizing both English and Telugu waveforms without requiring re-training from scratch.

---

## 4. Evaluation and Test Sentence Analysis

### Q9: How did the model handle sentence E2 ("conduct" as noun vs verb)?
> **Answer:**
> In **E2a** ("The orchestra's *conduct*..."), "conduct" is a noun accented on the first syllable (/ˈkɒndʌkt/).  
> In **E2b** ("The researchers will *conduct* an experiment..."), "conduct" is a verb accented on the second syllable (/kənˈdʌkt/).  
> The Variance Adaptor's pitch and duration plots reveal this shift: in E2a, the first syllable exhibits higher F0 and frame duration, whereas in E2b, the stress moves to the second syllable, demonstrating that the text encoder captures syntactic context across the sentence.

### Q10: How did the models handle the intentional double spaces in E3 ("rhyth  m") and T3 ("అడ  విలో")?
> **Answer:**
> Both the English and Telugu text processors preserved these inputs as specified in the assignment. The tokenizer converted the consecutive spaces into distinct pause tokens. The duration predictor allocated a short closure duration (~20–40ms), producing a subtle hesitation or pause without causing the attention mechanisms or length regulator to crash.

---

## 5. Potential Questions About Limitations & Future Work

### Q11: What are the primary limitations of training on 5 hours of speech per language?
> **Answer:**
> FastSpeech 2 benchmark models (such as LJSpeech) typically train on 24 hours of single-speaker studio data for 900,000 steps. While 5 hours is sufficient for proof-of-concept acoustic modeling and clear pronunciation, the synthesized voice may retain slight synthetic resonance and prosodic flattening. Training on 20+ hours with speaker conditioning would produce commercial-grade naturalness.

### Q12: How would you scale this to a bilingual or code-switched English-Telugu model?
> **Answer:**
> 1. **Shared Vocabulary:** Construct a unified phonemic alphabet using the International Phonetic Alphabet (IPA) or an expanded Indic phoneme set (like IndicPhones).
> 2. **Language Embeddings:** Add a language ID embedding ($L \in \{0, 1\}$) to the text encoder output alongside positional encoding.
> 3. **Multi-Speaker / Multi-Lingual Vocoder:** Fine-tune HiFi-GAN jointly on both Indian-accented English and Telugu datasets to eliminate acoustic disparities at language transitions.

