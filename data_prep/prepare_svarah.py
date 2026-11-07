"""
Svarah Dataset Acquisition and Split Preparation
=================================================

Downloads and prepares the Svarah Indian English dataset (ai4bharat/Svarah):
  1. Streams or downloads utterances from Hugging Face
  2. Resamples audio to 22,050 Hz (16-bit mono PCM WAV)
  3. Validates transcript and audio duration (0.5s - 12.0s)
  4. Accumulates 5.0 hours for training and 1.0 hour for validation
  5. Writes manifests: English/data_splits/train.txt & English/data_splits/val.txt

Usage:
    python data_prep/prepare_svarah.py --output_dir data/svarah --train_hours 5.0 --val_hours 1.0
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_svarah")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Svarah dataset for FastSpeech 2")
    parser.add_argument("--output_dir", type=str, default="data/svarah")
    parser.add_argument("--splits_dir", type=str, default="English/data_splits")
    parser.add_argument("--train_hours", type=float, default=5.0)
    parser.add_argument("--val_hours", type=float, default=1.0)
    parser.add_argument("--target_sr", type=int, default=22050)
    parser.add_argument("--max_utterance_sec", type=float, default=12.0)
    parser.add_argument("--min_utterance_sec", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for quick testing")
    parser.add_argument("--smoke_test", action="store_true", help="Quick smoke test mode: downloads 10 train and 4 val independent genuine samples")
    return parser.parse_args()


def prepare_svarah(args):
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

    hf_token = os.environ.get("HF_TOKEN")
    dataset_name = "ai4bharat/Svarah"

    try:
        logger.info(f"Attempting to load {dataset_name} stream from HuggingFace...")
        dataset = load_dataset(dataset_name, split="train", streaming=True, token=hf_token)
        dataset = dataset.cast_column("audio", datasets.Audio(decode=False))
        # Trigger first item to verify access
        next(iter(dataset))
        logger.info(f"Successfully connected to {dataset_name}.")
    except Exception as e:
        error_msg = (
            f"\n{'='*70}\n"
            f"ERROR: Cannot access required dataset '{dataset_name}': {e}\n"
            f"The assignment specifically requires the AI4Bharat Svarah dataset.\n"
            f"Svarah is a gated Hugging Face repository. To resolve access:\n"
            f"  1. Go to https://huggingface.co/datasets/ai4bharat/Svarah and accept the terms.\n"
            f"  2. Create an access token at https://huggingface.co/settings/tokens.\n"
            f"  3. Set HF_TOKEN environment variable (e.g. export HF_TOKEN='hf_...').\n"
            f"Per assignment instructions, automatic substitution of other datasets is prohibited.\n"
            f"{'='*70}\n"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

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
        sid = f"english_{sample_idx:06d}"
        audio_data = item.get("audio", {})
        audio_bytes = audio_data.get("bytes")
        text = item.get("transcription", item.get("text", "")).strip()

        if not audio_bytes or not text:
            continue

        try:
            array, sr = sf.read(io.BytesIO(audio_bytes))
        except Exception as e:
            logger.warning(f"Failed to decode audio bytes for {sid}: {e}")
            continue

        if len(array) == 0:
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

        # Save text file (.txt) alongside WAV for MFA aligner
        txt_path = wavs_dir / f"{sid}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        entry = {
            "id": sid,
            "audio_path": str(wav_path).replace("\\", "/"),
            "text": text,
            "speaker": item.get("speaker", item.get("speaker_id", "english_speaker")),
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


if __name__ == "__main__":
    args = parse_args()
    prepare_svarah(args)

