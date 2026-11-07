"""
English Preprocessing Pipeline for FastSpeech 2
=================================================

Preprocesses the Svarah dataset for FastSpeech 2 training:
  1. Loads audio files and transcripts
  2. Resamples to 22050 Hz
  3. Extracts mel-spectrograms
  4. Extracts pitch (F0) using PyWorld
  5. Extracts energy (RMS of mel frames)
  6. Computes duration targets from MFA TextGrid alignments
  7. Normalizes pitch and energy statistics
  8. Saves preprocessed features as numpy arrays

Usage:
    python preprocess.py --config config.yaml --data_dir data/svarah [--mfa]

Requirements:
    pip install librosa soundfile pyworld numpy tqdm pyyaml g2p_en

Author: FastSpeech 2 English TTS Project
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
    parser = argparse.ArgumentParser(description="Preprocess Svarah dataset for FastSpeech 2")
    parser.add_argument("--config", type=str, default=str(script_dir / "config.yaml"),
                        help="Path to English config.yaml")
    parser.add_argument("--data_dir", type=str, default="data/svarah",
                        help="Root directory of Svarah dataset")
    parser.add_argument("--output_dir", type=str, default="data/preprocessed/english",
                        help="Output directory for preprocessed features")
    parser.add_argument("--mfa_dir", type=str, default=None,
                        help="Directory containing MFA TextGrid alignments")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split")
    parser.add_argument("--train_hours", type=float, default=5.0,
                        help="Target training audio duration in hours")
    parser.add_argument("--val_hours", type=float, default=1.0,
                        help="Target validation audio duration in hours")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_svarah_manifest(data_dir: str) -> List[Dict]:
    """Load Svarah dataset metadata.
    
    Svarah is available on HuggingFace as ai4bharat/Svarah.
    Local structure may vary; this handles common formats:
      - TSV file with columns: file_id, transcript, speaker, duration
      - Or individual txt files alongside WAV files
    
    Args:
        data_dir: Root directory of Svarah dataset
        
    Returns:
        List of dicts with keys: id, audio_path, text, speaker, duration
    """
    data_dir = Path(data_dir)
    samples = []
    
    # Try loading from metadata.tsv or manifest.json
    for meta_file in ["metadata.tsv", "metadata.csv", "manifest.json", "train.tsv"]:
        meta_path = data_dir / meta_file
        if meta_path.exists():
            logger.info(f"Found metadata file: {meta_path}")
            if meta_file.endswith(".json"):
                samples = _load_json_manifest(meta_path)
            else:
                samples = _load_tsv_manifest(meta_path, data_dir)
            break
    
    if not samples:
        # Fallback: scan for WAV+txt pairs
        logger.info("No manifest found, scanning for WAV+txt pairs...")
        samples = _scan_wav_txt_pairs(data_dir)
    
    logger.info(f"Loaded {len(samples)} samples from {data_dir}")
    return samples


def _load_tsv_manifest(tsv_path: Path, data_dir: Path) -> List[Dict]:
    """Load from TSV file."""
    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Detect header
    if lines and "\t" in lines[0]:
        headers = lines[0].strip().split("\t")
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                row = dict(zip(headers, parts))
                sample = {
                    "id": row.get("id", row.get("file", parts[0])),
                    "audio_path": str(data_dir / "wavs" / f"{row.get('id', parts[0])}.wav"),
                    "text": row.get("text", row.get("transcript", row.get("sentence", parts[1]))),
                    "speaker": row.get("speaker", row.get("client_id", "unknown")),
                    "duration": float(row.get("duration", 0.0)),
                }
                samples.append(sample)
    return samples


def _load_json_manifest(json_path: Path) -> List[Dict]:
    """Load from JSON manifest."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def _scan_wav_txt_pairs(data_dir: Path) -> List[Dict]:
    """Scan directory for WAV+transcript pairs."""
    samples = []
    wav_files = list(data_dir.rglob("*.wav"))
    
    for wav_path in wav_files:
        txt_path = wav_path.with_suffix(".txt")
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            samples.append({
                "id": wav_path.stem,
                "audio_path": str(wav_path),
                "text": text,
                "speaker": wav_path.parent.name,
                "duration": 0.0,  # Will be computed
            })
    
    return samples


# ---------------------------------------------------------------------------
# Audio Duration Computation
# ---------------------------------------------------------------------------

def compute_audio_duration(audio_path: str, sampling_rate: int = None) -> float:
    """Compute audio duration in seconds.
    
    Args:
        audio_path: Path to WAV file
        sampling_rate: If None, reads from file
        
    Returns:
        Duration in seconds
    """
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr
        except Exception as e:
            logger.warning(f"Could not compute duration for {audio_path}: {e}")
            return 0.0


def select_data_split(
    samples: List[Dict],
    train_hours: float = 5.0,
    val_hours: float = 1.0,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Select train/validation split from samples.
    
    Strategy:
    1. Compute duration for all samples
    2. Shuffle with fixed seed
    3. Select first N samples for training until ~train_hours reached
    4. Select next M samples for validation until ~val_hours reached
    5. No overlap guaranteed by sequential selection
    
    Args:
        samples: All available samples
        train_hours: Target training duration
        val_hours: Target validation duration
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_samples, val_samples)
    """
    logger.info("Computing audio durations...")
    for sample in samples:
        if sample.get("duration", 0.0) == 0.0:
            sample["duration"] = compute_audio_duration(sample["audio_path"])
    
    # Filter out samples with zero duration or missing audio
    valid_samples = [s for s in samples if s.get("duration", 0.0) > 0.1]
    logger.info(f"{len(valid_samples)} / {len(samples)} samples have valid audio")
    
    # Shuffle with seed
    rng = random.Random(seed)
    shuffled = valid_samples.copy()
    rng.shuffle(shuffled)
    
    # Select training samples
    train_samples = []
    train_duration = 0.0
    train_target = train_hours * 3600  # Convert to seconds
    
    idx = 0
    while idx < len(shuffled) and train_duration < train_target:
        train_samples.append(shuffled[idx])
        train_duration += shuffled[idx]["duration"]
        idx += 1
    
    # Select validation samples (from remaining)
    val_samples = []
    val_duration = 0.0
    val_target = val_hours * 3600
    
    while idx < len(shuffled) and val_duration < val_target:
        val_samples.append(shuffled[idx])
        val_duration += shuffled[idx]["duration"]
        idx += 1
    
    # If dataset is small (e.g. smoke test), ensure validation has independent samples
    if len(val_samples) == 0 and len(train_samples) > 1:
        n_val = max(1, int(len(train_samples) * 0.2))
        val_samples = train_samples[-n_val:]
        train_samples = train_samples[:-n_val]
        train_duration = sum(s.get("duration", 0.0) for s in train_samples)
        val_duration = sum(s.get("duration", 0.0) for s in val_samples)
    
    logger.info(f"Train: {len(train_samples)} samples, {train_duration/3600:.2f}h")
    logger.info(f"Val:   {len(val_samples)} samples, {val_duration/3600:.2f}h")
    
    return train_samples, val_samples


# ---------------------------------------------------------------------------
# Audio Feature Extraction
# ---------------------------------------------------------------------------

def load_audio(audio_path: str, target_sr: int = 22050) -> np.ndarray:
    """Load and resample audio to target sampling rate.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sampling rate
        
    Returns:
        Audio signal as float32 numpy array, normalized to [-1, 1]
    """
    import librosa
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
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
    """Extract mel-spectrogram from audio signal.
    
    Args:
        y: Audio signal
        sr: Sampling rate
        n_fft: FFT window size
        hop_length: Hop length between frames
        win_length: Window length
        n_mels: Number of mel channels
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Mel-spectrogram [n_mels, T] in log scale
    """
    import librosa
    
    # Compute mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=1.0,  # magnitude spectrogram
    )
    
    # Log compression
    mel = np.log(np.clip(mel, 1e-5, None))
    
    return mel.astype(np.float32)  # [n_mels, T]


def extract_pitch(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 256,
    f0_min: float = 50.0,
    f0_max: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract pitch (F0) using PyWorld.
    
    Args:
        y: Audio signal (float64 for PyWorld)
        sr: Sampling rate
        hop_length: Hop length (must match mel-spectrogram)
        f0_min: Minimum F0 in Hz
        f0_max: Maximum F0 in Hz
        
    Returns:
        Tuple of (f0, voiced_mask) where f0 is [T] and voiced_mask is [T] bool
    """
    try:
        import pyworld as pw
        
        # PyWorld requires float64
        y_64 = y.astype(np.float64)
        
        frame_period = hop_length / sr * 1000  # ms
        
        # DIO algorithm for F0 extraction
        f0, t = pw.dio(y_64, sr, f0_floor=f0_min, f0_ceil=f0_max,
                       frame_period=frame_period)
        
        # StoneMask for F0 refinement
        f0 = pw.stonemask(y_64, f0, t, sr)
        
        # Voiced mask: f0 > 0
        voiced_mask = (f0 > 0).astype(np.float32)
        
        # Set unvoiced frames to 0
        f0 = f0 * voiced_mask
        
        return f0.astype(np.float32), voiced_mask
    
    except ImportError:
        logger.warning("PyWorld not installed. Using librosa pyin for pitch extraction.")
        return _extract_pitch_librosa(y, sr, hop_length, f0_min, f0_max)


def _extract_pitch_librosa(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    f0_min: float,
    f0_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback pitch extraction using librosa pyin."""
    import librosa
    
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=f0_min,
        fmax=f0_max,
        sr=sr,
        hop_length=hop_length,
    )
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    voiced_mask = voiced_flag.astype(np.float32)
    return f0, voiced_mask


def extract_energy(mel: np.ndarray) -> np.ndarray:
    """Extract energy as L2-norm of mel frame magnitudes.
    
    Args:
        mel: Log mel-spectrogram [n_mels, T]
        
    Returns:
        Energy array [T]
    """
    # Convert from log to linear scale before computing norm
    linear = np.exp(mel)
    energy = np.linalg.norm(linear, axis=0)  # [T]
    return energy.astype(np.float32)


# ---------------------------------------------------------------------------
# Duration Extraction from MFA TextGrids
# ---------------------------------------------------------------------------

def parse_textgrid_durations(
    textgrid_path: str,
    phonemes: List[str],
    hop_length: int = 256,
    sr: int = 22050,
) -> np.ndarray:
    """Parse MFA TextGrid to extract phoneme durations in frames.
    
    Args:
        textgrid_path: Path to .TextGrid file from MFA
        phonemes: List of expected phonemes for verification
        hop_length: Hop length used for mel extraction
        sr: Sampling rate
        
    Returns:
        Duration array [n_phonemes] in frames
    """
    import tgt
    
    tg = tgt.io.read_textgrid(textgrid_path)
    
    # Find the phones tier
    phone_tier = None
    for tier in tg.tiers:
        if "phones" in tier.name.lower() or "phone" in tier.name.lower():
            phone_tier = tier
            break
    
    if phone_tier is None:
        raise ValueError(f"No phones tier found in {textgrid_path}")
    
    # Extract durations in frames
    frame_duration = hop_length / sr  # seconds per frame
    durations = []
    
    for interval in phone_tier:
        if interval.text not in ["", "sp", "sil", "SIL", "SPN"]:
            duration_sec = interval.duration()
            duration_frames = int(np.round(duration_sec / frame_duration))
            duration_frames = max(1, duration_frames)  # At least 1 frame
            durations.append(duration_frames)
    
    return np.array(durations, dtype=np.int64)


# ---------------------------------------------------------------------------
# Statistics Computation
# ---------------------------------------------------------------------------

def compute_normalization_stats(
    feature_list: List[np.ndarray],
) -> Tuple[float, float]:
    """Compute mean and standard deviation for normalization.
    
    Args:
        feature_list: List of feature arrays
        
    Returns:
        Tuple of (mean, std)
    """
    all_values = np.concatenate(feature_list)
    # For pitch, exclude unvoiced frames (value == 0)
    nonzero = all_values[all_values > 0]
    if len(nonzero) == 0:
        return 0.0, 1.0
    return float(np.mean(nonzero)), float(np.std(nonzero))


# ---------------------------------------------------------------------------
# Main Preprocessing Function
# ---------------------------------------------------------------------------

def preprocess_sample(
    sample: Dict,
    config: Dict,
    output_dir: str,
    mfa_dir: Optional[str] = None,
) -> Optional[Dict]:
    """Preprocess a single audio sample.
    
    Args:
        sample: Sample dict with audio_path, text, id
        config: Preprocessing configuration
        output_dir: Output directory for features
        mfa_dir: Directory with MFA TextGrid files
        
    Returns:
        Metadata dict for the preprocessed sample, or None if failed
    """
    sample_id = sample["id"]
    audio_path = sample["audio_path"]
    
    if not os.path.exists(audio_path):
        logger.warning(f"Audio file not found: {audio_path}")
        return None
    
    try:
        cfg_audio = config.get("preprocessing", {}).get("audio", {})
        sr = cfg_audio.get("sampling_rate", 22050)
        n_fft = cfg_audio.get("filter_length", 1024)
        hop = cfg_audio.get("hop_length", 256)
        win = cfg_audio.get("win_length", 1024)
        n_mels = cfg_audio.get("n_mel_channels", 80)
        fmin = cfg_audio.get("mel_fmin", 0.0)
        fmax = cfg_audio.get("mel_fmax", 8000.0)
        
        # Load audio
        y = load_audio(audio_path, target_sr=sr)
        
        # Extract mel-spectrogram
        mel = extract_mel_spectrogram(y, sr, n_fft, hop, win, n_mels, fmin, fmax)
        
        # Extract pitch
        f0, voiced = extract_pitch(y, sr, hop)
        
        # Extract energy
        energy = extract_energy(mel)
        
        # Align lengths (mel, f0, energy may differ by 1-2 frames)
        T = mel.shape[1]
        f0 = f0[:T] if len(f0) >= T else np.pad(f0, (0, T - len(f0)))
        voiced = voiced[:T] if len(voiced) >= T else np.pad(voiced, (0, T - len(voiced)))
        energy = energy[:T] if len(energy) >= T else np.pad(energy, (0, T - len(energy)))
        
        # Save features
        out_dir = Path(output_dir)
        mel_dir = out_dir / "mel"
        pitch_dir = out_dir / "pitch"
        energy_dir = out_dir / "energy"
        for d in [mel_dir, pitch_dir, energy_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        np.save(mel_dir / f"{sample_id}.npy", mel)
        np.save(pitch_dir / f"{sample_id}.npy", f0)
        np.save(energy_dir / f"{sample_id}.npy", energy)
        
        # Duration from MFA (if available)
        duration = None
        if mfa_dir:
            tg_path = os.path.join(mfa_dir, f"{sample_id}.TextGrid")
            if os.path.exists(tg_path):
                try:
                    duration = parse_textgrid_durations(tg_path, [], hop, sr)
                    dur_dir = out_dir / "duration"
                    dur_dir.mkdir(parents=True, exist_ok=True)
                    np.save(dur_dir / f"{sample_id}.npy", duration)
                except Exception as e:
                    logger.warning(f"Duration extraction failed for {sample_id}: {e}")
        
        return {
            "id": sample_id,
            "text": sample["text"],
            "speaker": sample.get("speaker", "unknown"),
            "mel_frames": T,
            "audio_duration": len(y) / sr,
            "has_duration": duration is not None,
        }
    
    except Exception as e:
        logger.error(f"Failed to preprocess {sample_id}: {e}", exc_info=True)
        return None


def run_preprocessing(args):
    """Main preprocessing entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    logger.info("=== English FastSpeech 2 Preprocessing ===")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Random seed: {args.seed}")
    
    # Load config
    config = load_config(args.config)
    
    # Load dataset manifest
    logger.info("Loading dataset manifest...")
    samples = load_svarah_manifest(args.data_dir)
    
    if not samples:
        logger.error("No samples found! Check data directory and format.")
        sys.exit(1)
    
    # Select train/val split
    logger.info("Selecting train/val split...")
    train_samples, val_samples = select_data_split(
        samples,
        train_hours=args.train_hours,
        val_hours=args.val_hours,
        seed=args.seed,
    )
    
    # Save manifests to both data_splits and output_dir
    config_parent = Path(args.config).parent
    lang_root = config_parent.parent if config_parent.name == "config" else config_parent
    splits_dir = lang_root / "data_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        for target_dir in [splits_dir, out_dir]:
            manifest_path = target_dir / f"{split_name}.txt"
            with open(manifest_path, "w", encoding="utf-8") as f:
                for s in split_samples:
                    f.write(f"{s['id']}|{s['audio_path']}|{s['text']}|{s.get('speaker', 'unknown')}\n")
            logger.info(f"Saved {split_name} manifest: {manifest_path}")
    
    # Preprocess all samples
    logger.info("Preprocessing features...")
    all_samples = train_samples + val_samples
    metadata = []
    
    from tqdm import tqdm
    for sample in tqdm(all_samples, desc="Preprocessing"):
        result = preprocess_sample(
            sample=sample,
            config=config,
            output_dir=args.output_dir,
            mfa_dir=args.mfa_dir,
        )
        if result:
            metadata.append(result)
    
    # Compute normalization statistics
    logger.info("Computing normalization statistics...")
    output_dir = Path(args.output_dir)
    
    pitch_files = list((output_dir / "pitch").glob("*.npy"))
    energy_files = list((output_dir / "energy").glob("*.npy"))
    
    if pitch_files:
        pitches = [np.load(f) for f in pitch_files]
        p_mean, p_std = compute_normalization_stats(pitches)
        logger.info(f"Pitch stats: mean={p_mean:.2f} Hz, std={p_std:.2f} Hz")
        np.save(output_dir / "pitch_mean.npy", np.array(p_mean))
        np.save(output_dir / "pitch_std.npy", np.array(p_std))
    
    if energy_files:
        energies = [np.load(f) for f in energy_files]
        e_mean, e_std = compute_normalization_stats(energies)
        logger.info(f"Energy stats: mean={e_mean:.4f}, std={e_std:.4f}")
        np.save(output_dir / "energy_mean.npy", np.array(e_mean))
        np.save(output_dir / "energy_std.npy", np.array(e_std))
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Preprocessing complete: {len(metadata)} samples processed")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    run_preprocessing(args)

