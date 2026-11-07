"""
FastSpeech 2 PyTorch Dataset & Collate Function
==============================================

Loads preprocessed mel spectrograms, pitch, energy, durations, and text sequences.
Supports both English (phoneme sequences) and Telugu (grapheme sequences).

Licence: MIT
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple


class FastSpeech2Dataset(Dataset):
    """Dataset for training/validation of FastSpeech 2."""

    def __init__(
        self,
        filename: str,
        preprocess_config: dict,
        train_config: dict,
        language: str = "en",
    ):
        self.dataset_name = preprocess_config.get("dataset", "TTS")
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.language = language

        # Load manifest
        # Can be inside preprocessed_path or configured split path
        manifest_path = os.path.join(self.preprocessed_path, filename)
        if not os.path.exists(manifest_path):
            # check splits directory in language root
            alt_path = os.path.join(
                os.path.dirname(preprocess_config["path"].get("raw_path", "")),
                "data_splits",
                filename,
            )
            if os.path.exists(alt_path):
                manifest_path = alt_path

        self.samples = []
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 3:
                        # id|audio_path|text or id|audio_path|text|speaker
                        sample_id = parts[0]
                        text = parts[2]
                        speaker = parts[3] if len(parts) > 3 else "default"
                        self.samples.append({
                            "id": sample_id,
                            "text": text,
                            "speaker": speaker,
                        })

        # Load pitch and energy statistics for normalization
        p_mean_path = os.path.join(self.preprocessed_path, "pitch_mean.npy")
        p_std_path = os.path.join(self.preprocessed_path, "pitch_std.npy")
        e_mean_path = os.path.join(self.preprocessed_path, "energy_mean.npy")
        e_std_path = os.path.join(self.preprocessed_path, "energy_std.npy")

        self.p_mean = float(np.load(p_mean_path)) if os.path.exists(p_mean_path) else 0.0
        self.p_std = float(np.load(p_std_path)) if os.path.exists(p_std_path) else 1.0
        self.e_mean = float(np.load(e_mean_path)) if os.path.exists(e_mean_path) else 0.0
        self.e_std = float(np.load(e_std_path)) if os.path.exists(e_std_path) else 1.0

        # Load language tokenizer/vocab
        if self.language == "te":
            vocab_path = os.path.join(self.preprocessed_path, "telugu_vocab.json")
            if not os.path.exists(vocab_path):
                vocab_path = os.path.join("Telugu", "telugu_vocab.json")
            from Telugu.text_processing import TeluguTokenizer, build_telugu_vocabulary
            if os.path.exists(vocab_path):
                self.tokenizer = TeluguTokenizer.from_vocab_file(vocab_path)
            else:
                self.tokenizer = TeluguTokenizer(build_telugu_vocabulary())
        else:
            from English.text_processing import EnglishPhonemizer, build_english_phoneme_vocabulary
            self.phonemizer = EnglishPhonemizer()
            self.vocab = build_english_phoneme_vocabulary()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        sample_id = sample["id"]
        raw_text = sample["text"]

        # Text to token sequence
        if self.language == "te":
            phone = np.array(self.tokenizer.text_to_sequence(raw_text))
        else:
            phone = np.array(self.phonemizer.text_to_sequence(raw_text, self.vocab))

        # Load preprocessed features
        mel_path = os.path.join(self.preprocessed_path, "mel", f"{sample_id}.npy")
        pitch_path = os.path.join(self.preprocessed_path, "pitch", f"{sample_id}.npy")
        energy_path = os.path.join(self.preprocessed_path, "energy", f"{sample_id}.npy")
        dur_path = os.path.join(self.preprocessed_path, "duration", f"{sample_id}.npy")

        if os.path.exists(mel_path):
            mel = np.load(mel_path).T  # [T_mel, n_mel]
        else:
            mel = np.zeros((len(phone) * 5, 80), dtype=np.float32)

        if os.path.exists(pitch_path):
            pitch = np.load(pitch_path)
            # Normalize
            nonzero_idx = pitch > 0
            pitch[nonzero_idx] = (pitch[nonzero_idx] - self.p_mean) / (self.p_std + 1e-8)
        else:
            pitch = np.zeros(mel.shape[0], dtype=np.float32)

        if os.path.exists(energy_path):
            energy = np.load(energy_path)
            energy = (energy - self.e_mean) / (self.e_std + 1e-8)
        else:
            energy = np.zeros(mel.shape[0], dtype=np.float32)

        if os.path.exists(dur_path):
            duration = np.load(dur_path)
        else:
            # Fallback uniform duration if alignment is missing for a sample
            t_mel = mel.shape[0]
            t_phone = len(phone)
            dur_val = max(1, t_mel // max(1, t_phone))
            duration = np.full(t_phone, dur_val, dtype=np.int64)
            # Adjust remainder to sum to t_mel
            diff = t_mel - np.sum(duration)
            if diff > 0:
                duration[-1] += diff
            elif diff < 0:
                for i in range(len(duration) - 1, -1, -1):
                    deduct = min(duration[i] - 1, -diff)
                    duration[i] -= deduct
                    diff += deduct
                    if diff == 0:
                        break

        return {
            "id": sample_id,
            "text": phone,
            "raw_text": raw_text,
            "speaker": 0,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> List[Any]:
    """Collates a list of samples into padded batch tensors."""
    ids = [b["id"] for b in batch]
    raw_texts = [b["raw_text"] for b in batch]
    speakers = torch.tensor([b["speaker"] for b in batch], dtype=torch.long)

    text_lens = torch.tensor([len(b["text"]) for b in batch], dtype=torch.long)
    max_text_len = int(torch.max(text_lens).item())

    mel_lens = torch.tensor([b["mel"].shape[0] for b in batch], dtype=torch.long)
    max_mel_len = int(torch.max(mel_lens).item())
    n_mel = batch[0]["mel"].shape[1]

    texts = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    mels = torch.zeros(len(batch), max_mel_len, n_mel, dtype=torch.float32)
    pitches = torch.zeros(len(batch), max_mel_len, dtype=torch.float32)
    energies = torch.zeros(len(batch), max_mel_len, dtype=torch.float32)
    durations = torch.zeros(len(batch), max_text_len, dtype=torch.long)

    for i, b in enumerate(batch):
        t_len = len(b["text"])
        m_len = b["mel"].shape[0]
        texts[i, :t_len] = torch.from_numpy(b["text"][:t_len])
        mels[i, :m_len, :] = torch.from_numpy(b["mel"][:m_len])
        pitches[i, :m_len] = torch.from_numpy(b["pitch"][:m_len])
        energies[i, :m_len] = torch.from_numpy(b["energy"][:m_len])
        d_len = min(t_len, len(b["duration"]))
        durations[i, :d_len] = torch.from_numpy(b["duration"][:d_len])

    return [
        ids,
        raw_texts,
        speakers,
        texts,
        text_lens,
        max_text_len,
        mels,
        mel_lens,
        max_mel_len,
        pitches,
        energies,
        durations,
    ]
