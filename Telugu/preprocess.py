"""
Telugu Preprocessing Pipeline for FastSpeech 2
===============================================

Preprocesses the IndicTTS Telugu dataset for FastSpeech 2 training:
  1. Loads audio files and transcripts
  2. Resamples to 22050 Hz
  3. Extracts mel-spectrograms
  4. Extracts pitch (F0) using PyWorld
  5. Extracts energy from mel-spectrogram
  6. Computes duration targets using CTC-Segmentation with IndicWav2Vec
  7. Normalizes pitch and energy statistics
  8. Saves preprocessed features as numpy arrays

Key difference from English:
  - Input representation: Grapheme-based (Telugu Unicode characters)
  - Duration alignment: CTC-Segmentation (MFA not supported for Telugu)
  - Text normalizer: Telugu-specific Unicode NFC normalization

Usage:
    python preprocess.py --config config.yaml --data_dir data/indictts_telugu

Requirements:
    pip install librosa soundfile pyworld numpy tqdm pyyaml transformers
    pip install ctc-segmentation  (optional, for duration alignment)

Author: FastSpeech 2 Telugu TTS Project
Licence: MIT
"""

import os
import sys
import json
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def parse_args():
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Preprocess IndicTTS Telugu for FastSpeech 2")
    parser.add_argument("--config", type=str, default=str(script_dir / "config.yaml"))
    parser.add_argument("--data_dir", type=str, default="data/indictts_telugu")
    parser.add_argument("--output_dir", type=str, default="data/preprocessed/telugu")
    parser.add_argument("--mfa_dir", type=str, default=None,
                        help="Directory containing IndicMFA TextGrid alignments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_hours", type=float, default=5.0)
    parser.add_argument("--val_hours", type=float, default=1.0)
    parser.add_argument("--compute_durations", action="store_true",
                        help="Run aligner (IndicMFA / CTC-segmentation) to compute durations")
    parser.add_argument("--download_indicmfa_models", action="store_true",
                        help="Download official AI4Bharat IndicMFA Telugu models and dictionary")
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# IndicTTS Telugu Dataset Loading
# ---------------------------------------------------------------------------

def load_indictts_manifest(data_dir: str) -> List[Dict]:
    """Load IndicTTS Telugu dataset metadata.
    
    IndicTTS structure (typical):
      data/
        female/
          wav/
            001.wav
            002.wav
          txt/
            001.txt
            002.txt
        male/
          wav/
          txt/
    
    Or flat structure:
      data/
        wavs/
        transcripts.txt  (format: file_id|transcript)
    
    Args:
        data_dir: Root directory of IndicTTS Telugu data
        
    Returns:
        List of dicts with keys: id, audio_path, text, speaker
    """
    data_dir = Path(data_dir)
    samples = []
    
    # Try common metadata file formats
    for meta_file in ["transcripts.txt", "metadata.txt", "text.txt", "transcript.txt"]:
        meta_path = data_dir / meta_file
        if meta_path.exists():
            logger.info(f"Found metadata: {meta_path}")
            samples = _load_indictts_txt_manifest(meta_path, data_dir)
            break
    
    if not samples:
        # Try scanning speaker subdirectories (female/male)
        for speaker_dir in data_dir.iterdir():
            if speaker_dir.is_dir():
                speaker_name = speaker_dir.name
                wav_dir = speaker_dir / "wav"
                txt_dir = speaker_dir / "txt"
                
                if wav_dir.exists() and txt_dir.exists():
                    for txt_file in txt_dir.glob("*.txt"):
                        wav_file = wav_dir / (txt_file.stem + ".wav")
                        if wav_file.exists():
                            with open(txt_file, "r", encoding="utf-8") as f:
                                text = f.read().strip()
                            samples.append({
                                "id": f"{speaker_name}_{txt_file.stem}",
                                "audio_path": str(wav_file),
                                "text": text,
                                "speaker": speaker_name,
                            })
    
    if not samples:
        # Fallback: scan all WAV+TXT pairs
        samples = _scan_telugu_wav_txt(data_dir)
    
    logger.info(f"Loaded {len(samples)} Telugu samples")
    return samples


def _load_indictts_txt_manifest(manifest_path: Path, data_dir: Path) -> List[Dict]:
    """Load from pipe-delimited or tab-delimited manifest."""
    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Try pipe delimiter first (file_id|text or file_id|speaker|text)
            if "|" in line:
                parts = line.split("|")
                file_id = parts[0].strip()
                text = parts[-1].strip()
                speaker = parts[1].strip() if len(parts) > 2 else "default"
            else:
                # Space/tab delimiter
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                file_id = parts[0]
                text = parts[1]
                speaker = "default"
            
            # Find WAV file
            wav_path = None
            for wav_dir in ["wavs", "wav", "audio", "."]:
                candidate = data_dir / wav_dir / f"{file_id}.wav"
                if candidate.exists():
                    wav_path = str(candidate)
                    break
            
            if wav_path:
                samples.append({
                    "id": file_id,
                    "audio_path": wav_path,
                    "text": text,
                    "speaker": speaker,
                })
    
    return samples


def _scan_telugu_wav_txt(data_dir: Path) -> List[Dict]:
    """Scan directory for WAV+TXT pairs."""
    samples = []
    for wav_path in data_dir.rglob("*.wav"):
        txt_path = wav_path.with_suffix(".txt")
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            samples.append({
                "id": wav_path.stem,
                "audio_path": str(wav_path),
                "text": text,
                "speaker": wav_path.parent.parent.name,
            })
    return samples


# ---------------------------------------------------------------------------
# Data Splitting
# ---------------------------------------------------------------------------

def compute_audio_duration(audio_path: str) -> float:
    """Compute audio file duration in seconds."""
    try:
        import soundfile as sf
        return sf.info(audio_path).duration
    except Exception:
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr
        except Exception as e:
            logger.warning(f"Cannot compute duration for {audio_path}: {e}")
            return 0.0


def select_data_split(
    samples: List[Dict],
    train_hours: float = 5.0,
    val_hours: float = 1.0,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Select train/validation split.
    
    For IndicTTS Telugu (typically 1-2 speakers), speaker-independent
    splitting is not feasible. We use utterance-level random splitting.
    
    Args:
        samples: All samples
        train_hours: Target training duration in hours
        val_hours: Target validation duration in hours
        seed: Random seed
        
    Returns:
        Tuple of (train_samples, val_samples)
    """
    logger.info("Computing audio durations for split selection...")
    
    for s in samples:
        if not s.get("duration"):
            s["duration"] = compute_audio_duration(s["audio_path"])
    
    valid = [s for s in samples if s.get("duration", 0) > 0.5]
    logger.info(f"{len(valid)} valid samples (duration > 0.5s)")
    
    rng = random.Random(seed)
    shuffled = valid.copy()
    rng.shuffle(shuffled)
    
    # Build train split
    train, val = [], []
    train_dur, val_dur = 0.0, 0.0
    target_train = train_hours * 3600
    target_val = val_hours * 3600
    
    for s in shuffled:
        if train_dur < target_train:
            train.append(s)
            train_dur += s["duration"]
        elif val_dur < target_val:
            val.append(s)
            val_dur += s["duration"]
        else:
            break
    
    logger.info(f"Telugu train: {len(train)} samples, {train_dur/3600:.2f}h")
    logger.info(f"Telugu val:   {len(val)} samples, {val_dur/3600:.2f}h")
    
    return train, val


# ---------------------------------------------------------------------------
# Audio Feature Extraction (same as English)
# ---------------------------------------------------------------------------

def load_audio(audio_path: str, target_sr: int = 22050) -> np.ndarray:
    import librosa
    y, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    return y.astype(np.float32)


def extract_mel_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    import librosa
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax,
        power=1.0,
    )
    return np.log(np.clip(mel, 1e-5, None)).astype(np.float32)


def extract_pitch(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 256,
    f0_min: float = 50.0,
    f0_max: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import pyworld as pw
        y64 = y.astype(np.float64)
        frame_ms = hop_length / sr * 1000
        f0, t = pw.dio(y64, sr, f0_floor=f0_min, f0_ceil=f0_max, frame_period=frame_ms)
        f0 = pw.stonemask(y64, f0, t, sr)
        voiced = (f0 > 0).astype(np.float32)
        return (f0 * voiced).astype(np.float32), voiced
    except ImportError:
        import librosa
        f0, voiced_flag, _ = librosa.pyin(y, fmin=f0_min, fmax=f0_max, sr=sr, hop_length=hop_length)
        return np.nan_to_num(f0, nan=0.0).astype(np.float32), voiced_flag.astype(np.float32)


def extract_energy(mel: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.exp(mel), axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# IndicMFA Duration Alignment (Primary Telugu Aligner)
# ---------------------------------------------------------------------------

def download_indicmfa_models(download_dir: str = "indicmfa_models") -> Tuple[str, str]:
    """Download official AI4Bharat IndicMFA Telugu acoustic model and dictionary.
    
    IndicMFA uses Grapheme-to-Grapheme (G2G) alignment for Telugu.
    Models are hosted on AI4Bharat IndicMFA GitHub releases.
    """
    import urllib.request
    dl_dir = Path(download_dir)
    dl_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://github.com/AI4Bharat/IndicMFA/releases/download/Telugu"
    model_name = "Telugu_Acoustic_Model.zip"
    dict_name = "Telugu_Dictionary_g2g.txt"
    
    model_path = dl_dir / model_name
    dict_path = dl_dir / dict_name
    
    if not model_path.exists():
        logger.info(f"Downloading IndicMFA Telugu Acoustic Model from {base_url}/{model_name}...")
        urllib.request.urlretrieve(f"{base_url}/{model_name}", str(model_path))
        logger.info(f"Saved: {model_path}")
    if not dict_path.exists():
        logger.info(f"Downloading IndicMFA Telugu G2G Dictionary from {base_url}/{dict_name}...")
        urllib.request.urlretrieve(f"{base_url}/{dict_name}", str(dict_path))
        logger.info(f"Saved: {dict_path}")
        
    return str(model_path), str(dict_path)


def parse_indicmfa_textgrid(
    textgrid_path: str,
    text: str,
    hop_length: int = 256,
    sr: int = 22050,
) -> np.ndarray:
    """Parse IndicMFA TextGrid to extract grapheme-level durations in frames.
    
    IndicMFA generates standard MFA TextGrids with a phones tier containing
    Telugu graphemes mapped directly from Telugu_Dictionary_g2g.txt.
    
    Args:
        textgrid_path: Path to .TextGrid file produced by IndicMFA
        text: Expected Telugu text
        hop_length: Hop length used for mel extraction (256)
        sr: Sampling rate (22050)
        
    Returns:
        Duration array [n_graphemes] in frames
    """
    import tgt
    tg = tgt.io.read_textgrid(textgrid_path)
    phone_tier = None
    for tier in tg.tiers:
        if "phones" in tier.name.lower() or "phone" in tier.name.lower() or "grapheme" in tier.name.lower():
            phone_tier = tier
            break
            
    if phone_tier is None and len(tg.tiers) > 0:
        phone_tier = tg.tiers[0]
        
    if phone_tier is None:
        raise ValueError(f"No phones/graphemes tier found in {textgrid_path}")
        
    frame_duration = hop_length / sr
    durations = []
    for interval in phone_tier:
        if interval.text not in ["", "sp", "sil", "SIL", "SPN", "<eps>"]:
            dur_sec = interval.duration()
            dur_frames = max(1, int(round(dur_sec / frame_duration)))
            durations.append(dur_frames)
            
    return np.array(durations, dtype=np.int64)


# ---------------------------------------------------------------------------
# CTC-Segmentation Duration Alignment (Telugu-specific Fallback)
# ---------------------------------------------------------------------------

def compute_ctc_durations(
    audio_path: str,
    text: str,
    model_name: str = "ai4bharat/indicwav2vec_v1_telugu",
    hop_length: int = 256,
    sr: int = 22050,
) -> Optional[np.ndarray]:
    """Compute grapheme-level durations using CTC-Segmentation.
    
    Uses AI4Bharat IndicWav2Vec for CTC logits, then applies
    CTC-Segmentation to find character-level frame alignments.
    
    Args:
        audio_path: Path to audio file
        text: Telugu text (normalized)
        model_name: HuggingFace model for CTC
        hop_length: Hop length for frame duration computation
        sr: Sampling rate
        
    Returns:
        Duration array [n_chars] in frames, or None if failed
    """
    try:
        import ctc_segmentation
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        import torch
        
        logger.info(f"Loading CTC model: {model_name}")
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        model.eval()
        
        # Load audio (model expects 16000 Hz)
        import librosa
        y, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        # Get CTC logits
        inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(0)  # [T_ctc, vocab]
        
        log_probs = torch.log_softmax(logits, dim=-1).numpy()
        
        # Get vocabulary
        vocab = processor.tokenizer.get_vocab()
        char_list = [None] * len(vocab)
        for ch, idx in vocab.items():
            char_list[idx] = ch
        
        # Prepare text for segmentation
        from Telugu.text_processing import telugu_text_normalize
        clean_text = telugu_text_normalize(text)
        
        # Run CTC segmentation
        config = ctc_segmentation.CtcSegmentationParameters()
        config.char_list = char_list
        config.index_duration = hop_length / sr  # seconds per CTC frame
        
        ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(
            config, [clean_text]
        )
        
        timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(
            config, log_probs, ground_truth_mat
        )
        
        # Convert to frame durations
        frame_duration_sec = hop_length / sr
        char_boundaries = ctc_segmentation.determine_utterance_segments(
            config, utt_begin_indices, char_probs, timings, [clean_text]
        )
        
        durations = []
        for start, end, prob in zip(timings[:-1], timings[1:], char_probs):
            dur_frames = max(1, int(round((end - start) / frame_duration_sec)))
            durations.append(dur_frames)
        
        return np.array(durations, dtype=np.int64)
    
    except ImportError as e:
        logger.warning(f"CTC-Segmentation not available: {e}")
        return None
    except Exception as e:
        logger.error(f"CTC-Segmentation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main Preprocessing
# ---------------------------------------------------------------------------

def preprocess_sample(
    sample: Dict,
    config: Dict,
    output_dir: str,
    mfa_dir: Optional[str] = None,
    compute_durations: bool = False,
) -> Optional[Dict]:
    """Preprocess a single Telugu audio sample."""
    sample_id = sample["id"]
    audio_path = sample["audio_path"]
    
    if not os.path.exists(audio_path):
        logger.warning(f"Missing: {audio_path}")
        return None
    
    try:
        cfg = config.get("preprocessing", {}).get("audio", {})
        sr = cfg.get("sampling_rate", 22050)
        n_fft = cfg.get("filter_length", 1024)
        hop = cfg.get("hop_length", 256)
        win = cfg.get("win_length", 1024)
        n_mels = cfg.get("n_mel_channels", 80)
        fmin = cfg.get("mel_fmin", 0.0)
        fmax = cfg.get("mel_fmax", 8000.0)
        
        y = load_audio(audio_path, target_sr=sr)
        mel = extract_mel_spectrogram(y, sr, n_fft, hop, win, n_mels, fmin, fmax)
        f0, voiced = extract_pitch(y, sr, hop)
        energy = extract_energy(mel)
        
        # Align lengths
        T = mel.shape[1]
        f0 = f0[:T] if len(f0) >= T else np.pad(f0, (0, T - len(f0)))
        energy = energy[:T] if len(energy) >= T else np.pad(energy, (0, T - len(energy)))
        
        # Save features
        out_dir = Path(output_dir)
        for subdir in ["mel", "pitch", "energy"]:
            (out_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        np.save(out_dir / "mel" / f"{sample_id}.npy", mel)
        np.save(out_dir / "pitch" / f"{sample_id}.npy", f0)
        np.save(out_dir / "energy" / f"{sample_id}.npy", energy)
        
        # Duration alignment: IndicMFA (Primary) or CTC-Segmentation (Fallback)
        has_duration = False
        duration = None
        
        if mfa_dir is not None:
            tg_candidates = [
                Path(mfa_dir) / f"{sample_id}.TextGrid",
                Path(mfa_dir) / f"{Path(audio_path).stem}.TextGrid",
            ]
            for tg_candidate in tg_candidates:
                if tg_candidate.exists():
                    try:
                        duration = parse_indicmfa_textgrid(str(tg_candidate), sample["text"], hop, sr)
                        break
                    except Exception as e:
                        logger.warning(f"Failed to parse TextGrid {tg_candidate}: {e}")
        
        if duration is None and compute_durations:
            aligner = config.get("preprocessing", {}).get("duration", {}).get(
                "aligner", config.get("alignment", {}).get("tool", "indicmfa")
            )
            if aligner == "indicmfa":
                logger.info("IndicMFA requires MFA alignment execution. Check TextGrids in mfa_dir.")
            else:
                ctc_model = config.get("preprocessing", {}).get("duration", {}).get(
                    "ctc_model", config.get("alignment", {}).get("ctc_fallback_model", "ai4bharat/indicwav2vec_v1_telugu")
                )
                duration = compute_ctc_durations(audio_path, sample["text"], ctc_model, hop, sr)
        
        if duration is not None:
            dur_dir = out_dir / "duration"
            dur_dir.mkdir(parents=True, exist_ok=True)
            np.save(dur_dir / f"{sample_id}.npy", duration)
            has_duration = True
        
        return {
            "id": sample_id,
            "text": sample["text"],
            "speaker": sample.get("speaker", "unknown"),
            "mel_frames": T,
            "audio_duration": len(y) / sr,
            "has_duration": has_duration,
        }
    
    except Exception as e:
        logger.error(f"Failed to preprocess {sample_id}: {e}", exc_info=True)
        return None


def run_preprocessing(args):
    """Main Telugu preprocessing entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    logger.info("=== Telugu FastSpeech 2 Preprocessing ===")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data dir: {args.data_dir}")
    
    config = load_config(args.config)
    
    if args.download_indicmfa_models:
        logger.info("Downloading official AI4Bharat IndicMFA Telugu models and dictionary...")
        download_indicmfa_models()
    
    # Build/save Telugu vocabulary
    logger.info("Building Telugu vocabulary...")
    from Telugu.text_processing import build_telugu_vocabulary, save_vocabulary
    
    samples = load_indictts_manifest(args.data_dir)
    
    if not samples:
        logger.error("No samples found!")
        sys.exit(1)
    
    all_texts = [s["text"] for s in samples]
    vocab = build_telugu_vocabulary(texts=all_texts)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    vocab_path = out_dir / "telugu_vocab.json"
    save_vocabulary(vocab, str(vocab_path))
    
    # Also save to Telugu/ directory for inference
    script_dir = Path(__file__).parent
    save_vocabulary(vocab, str(script_dir / "telugu_vocab.json"))
    
    logger.info(f"Telugu vocab size: {len(vocab)}")
    
    # Train/val split
    train_samples, val_samples = select_data_split(
        samples, args.train_hours, args.val_hours, args.seed
    )
    
    # Save manifests
    splits_dir = Path(os.path.dirname(args.config)) / "data_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        manifest_path = splits_dir / f"{split_name}.txt"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for s in split_samples:
                f.write(f"{s['id']}|{s['audio_path']}|{s['text']}|{s.get('speaker', 'unknown')}\n")
        logger.info(f"Saved {split_name} manifest: {manifest_path}")
    
    # Preprocess features
    from tqdm import tqdm
    metadata = []
    for sample in tqdm(train_samples + val_samples, desc="Preprocessing Telugu"):
        result = preprocess_sample(
            sample=sample,
            config=config,
            output_dir=args.output_dir,
            mfa_dir=args.mfa_dir,
            compute_durations=args.compute_durations,
        )
        if result:
            metadata.append(result)
    
    # Compute normalization stats
    pitch_files = list((out_dir / "pitch").glob("*.npy"))
    energy_files = list((out_dir / "energy").glob("*.npy"))
    
    if pitch_files:
        pitches = [np.load(f) for f in pitch_files]
        all_p = np.concatenate(pitches)
        nonzero_p = all_p[all_p > 0]
        p_mean = float(np.mean(nonzero_p)) if len(nonzero_p) > 0 else 0.0
        p_std = float(np.std(nonzero_p)) if len(nonzero_p) > 0 else 1.0
        np.save(out_dir / "pitch_mean.npy", np.array(p_mean))
        np.save(out_dir / "pitch_std.npy", np.array(p_std))
        logger.info(f"Telugu pitch stats: mean={p_mean:.2f} Hz, std={p_std:.2f} Hz")
    
    if energy_files:
        energies = [np.load(f) for f in energy_files]
        all_e = np.concatenate(energies)
        e_mean, e_std = float(np.mean(all_e)), float(np.std(all_e))
        np.save(out_dir / "energy_mean.npy", np.array(e_mean))
        np.save(out_dir / "energy_std.npy", np.array(e_std))
        logger.info(f"Telugu energy stats: mean={e_mean:.4f}, std={e_std:.4f}")
    
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Telugu preprocessing done: {len(metadata)} samples")


if __name__ == "__main__":
    args = parse_args()
    run_preprocessing(args)

