"""
IndicTTS Telugu Dataset Acquisition and Split Preparation
=========================================================

Downloads and prepares the IndicTTS Telugu dataset (SPRINGLab/IndicTTS_Telugu):
  1. Streams or downloads utterances from Hugging Face
  2. Resamples audio to 22,050 Hz (16-bit mono PCM WAV)
  3. Validates Telugu Unicode transcripts and audio duration (0.5s - 12.0s)
  4. Accumulates 5.0 hours for training and 1.0 hour for validation
  5. Saves transcript (.txt/.lab) files alongside WAVs for IndicMFA alignment
  6. Writes manifests: Telugu/data_splits/train.txt & Telugu/data_splits/val.txt

Usage:
    python data_prep/prepare_indictts.py --output_dir data/indictts_telugu --train_hours 5.0 --val_hours 1.0
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_indictts")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare IndicTTS Telugu dataset for FastSpeech 2")
    parser.add_argument("--output_dir", type=str, default="data/indictts_telugu")
    parser.add_argument("--splits_dir", type=str, default="Telugu/data_splits")
    parser.add_argument("--train_hours", type=float, default=5.0)
    parser.add_argument("--val_hours", type=float, default=1.0)
    parser.add_argument("--target_sr", type=int, default=22050)
    parser.add_argument("--max_utterance_sec", type=float, default=12.0)
    parser.add_argument("--min_utterance_sec", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for quick testing")
    parser.add_argument("--smoke_test", action="store_true", help="Quick smoke test mode: downloads 10 train and 4 val independent genuine samples")
    return parser.parse_args()


def prepare_indictts(args):
    try:
        import datasets
        from datasets import load_dataset
        import soundfile as sf
        import librosa
        import io
    except ImportError as e:
        logger.error(f"Missing required library: {e}. Run: pip install datasets soundfile librosa")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    wavs_dir = out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    is_smoke = getattr(args, "smoke_test", False)
    target_train_count = 10 if is_smoke else None
    target_val_count = 4 if is_smoke else None

    target_train_sec = args.train_hours * 3600.0
    target_val_sec = args.val_hours * 3600.0

    if is_smoke:
        logger.info(f"[SMOKE TEST MODE] Target train: {target_train_count} samples, val: {target_val_count} samples")
    else:
        logger.info(f"Target train duration: {args.train_hours:.2f}h ({target_train_sec:.0f}s)")
        logger.info(f"Target val duration:   {args.val_hours:.2f}h ({target_val_sec:.0f}s)")
    logger.info("Loading IndicTTS Telugu stream from HuggingFace (SPRINGLab/IndicTTS_Telugu)...")

    dataset = load_dataset("SPRINGLab/IndicTTS_Telugu", split="train", streaming=True)
    dataset = dataset.cast_column("audio", datasets.Audio(decode=False))

    train_samples = []
    val_samples = []
    train_sec = 0.0
    val_sec = 0.0

    sample_idx = 0
    for item in dataset:
        if is_smoke:
            if len(train_samples) >= target_train_count and len(val_samples) >= target_val_count:
                break
        else:
            if args.max_samples and (len(train_samples) + len(val_samples)) >= args.max_samples:
                break
            if train_sec >= target_train_sec and val_sec >= target_val_sec:
                break

        sample_idx += 1
        sid = f"telugu_{sample_idx:06d}"
        audio_data = item.get("audio", {})
        text = item.get("transcription", item.get("text", "")).strip()

        if not audio_data or not text:
            continue

        array = None
        sr = args.target_sr

        if isinstance(audio_data, dict):
            if audio_data.get("bytes"):
                try:
                    array, sr = sf.read(io.BytesIO(audio_data["bytes"]))
                except Exception as e:
                    logger.warning(f"Failed to decode audio bytes for {sid}: {e}")
                    continue
            elif audio_data.get("path") and os.path.exists(audio_data["path"]):
                try:
                    array, sr = sf.read(audio_data["path"])
                except Exception as e:
                    logger.warning(f"Failed to read audio file for {sid}: {e}")
                    continue
            elif "array" in audio_data and audio_data["array"] is not None:
                array = np.array(audio_data["array"], dtype=np.float32)
                sr = audio_data.get("sampling_rate", 16000)

        if array is None or len(array) == 0:
            continue

        # Convert to mono if multi-channel
        if len(array.shape) > 1:
            array = np.mean(array, axis=-1)

        # Check duration
        duration = len(array) / sr
        if duration < args.min_utterance_sec or duration > args.max_utterance_sec:
            continue

        # Resample to target_sr if needed
        if sr != args.target_sr:
            array = librosa.resample(array.astype(np.float32), orig_sr=sr, target_sr=args.target_sr)
            sr = args.target_sr

        # Save audio
        wav_path = wavs_dir / f"{sid}.wav"
        sf.write(str(wav_path), array, sr, subtype="PCM_16")

        # Save transcript file (.txt / .lab) for MFA aligner
        txt_path = wavs_dir / f"{sid}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        entry = {
            "id": sid,
            "audio_path": str(wav_path).replace("\\", "/"),
            "text": text,
            "speaker": item.get("speaker", item.get("speaker_id", "telugu_speaker")),
            "duration": duration,
        }

        if is_smoke:
            if len(train_samples) < target_train_count:
                train_samples.append(entry)
                train_sec += duration
            elif len(val_samples) < target_val_count:
                val_samples.append(entry)
                val_sec += duration
        else:
            if train_sec < target_train_sec:
                train_samples.append(entry)
                train_sec += duration
            elif val_sec < target_val_sec:
                val_samples.append(entry)
                val_sec += duration

    logger.info(f"Collected Train: {len(train_samples)} samples ({train_sec/3600:.2f}h)")
    logger.info(f"Collected Val:   {len(val_samples)} samples ({val_sec/3600:.2f}h)")

    # Write manifests
    for name, samples in [("train", train_samples), ("val", val_samples)]:
        manifest_file = splits_dir / f"{name}.txt"
        with open(manifest_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(f"{s['id']}|{s['audio_path']}|{s['text']}|{s['speaker']}\n")
        logger.info(f"Saved manifest: {manifest_file} ({len(samples)} lines)")

    # Write split metadata JSON
    train_speakers = sorted(list({s['speaker'] for s in train_samples}))
    val_speakers = sorted(list({s['speaker'] for s in val_samples}))
    meta_file = splits_dir / "split_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_source": "SPRINGLab/IndicTTS_Telugu",
            "upstream_split": "train",
            "split_strategy": "deterministic_sequential",
            "mode": "smoke_test" if is_smoke else "full_dataset",
            "train_sample_count": len(train_samples),
            "val_sample_count": len(val_samples),
            "train_speakers": train_speakers,
            "val_speakers": val_speakers,
            "train_duration_sec": round(train_sec, 2),
            "val_duration_sec": round(val_sec, 2),
            "train_duration_hours": round(train_sec / 3600.0, 3),
            "val_duration_hours": round(val_sec / 3600.0, 3),
            "target_sr": args.target_sr,
        }, f, indent=2)
    logger.info(f"Saved Telugu split metadata: {meta_file}")


if __name__ == "__main__":
    args = parse_args()
    prepare_indictts(args)

