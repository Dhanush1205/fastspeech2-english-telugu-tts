"""
Svarah Dataset Acquisition and Split Preparation
=================================================

Downloads and prepares the Svarah Indian English dataset (ai4bharat/Svarah):
  1. Connects to the official AI4Bharat Svarah repository on Hugging Face.
     NOTE ON UPSTREAM SPLIT: AI4Bharat created Svarah as an ASR evaluation benchmark
     and published the entire 9.6h corpus under a single upstream split named 'test'
     (data/test-*.parquet). Per IIIT Hyderabad Voice AI assignment requirements,
     this script partitions the official 9.6h corpus into independent, speaker-disjoint
     training (~5.0h) and validation (~1.0h) subsets with zero speaker overlap.
  2. Resamples audio to 22,050 Hz (16-bit mono PCM WAV)
  3. Validates transcript and audio duration (0.5s - 12.0s)
  4. Strictly excludes all assignment evaluation test utterances (E1-E4)
  5. Implements speaker-disjoint partitioning (train_speakers ∩ val_speakers = ∅)
  6. Writes manifests: English/data_splits/train.txt & English/data_splits/val.txt
  7. Writes metadata: English/data_splits/split_metadata.json

Usage:
    # Full dataset preparation (~5h train / ~1h val)
    python data_prep/prepare_svarah.py --output_dir data/svarah --train_hours 5.0 --val_hours 1.0

    # Smoke test mode (10 train / 4 val independent genuine samples)
    python data_prep/prepare_svarah.py --smoke_test --output_dir data/svarah_smoke --splits_dir English/data_splits
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_svarah")

# Assignment mandatory test sentence keywords for exclusion protection
ASSIGNMENT_TEST_PATTERNS = [
    "antidisestablishmentarianism",
    "orchestra's conduct",
    "orchestra changed",
    "conduct an experiment",
    "rhyth  m",
    "rhythm of the orchestra",
    "andaroo",
    "10:45",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Svarah dataset for FastSpeech 2")
    parser.add_argument("--output_dir", type=str, default="data/svarah")
    parser.add_argument("--splits_dir", type=str, default="English/data_splits")
    parser.add_argument("--train_hours", type=float, default=5.0)
    parser.add_argument("--val_hours", type=float, default=1.0)
    parser.add_argument("--target_sr", type=int, default=22050)
    parser.add_argument("--max_utterance_sec", type=float, default=12.0)
    parser.add_argument("--min_utterance_sec", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for quick testing")
    parser.add_argument("--smoke_test", action="store_true", help="Quick smoke test mode: downloads 10 train and 4 val speaker-disjoint genuine samples")
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
        logger.info(f"[SMOKE TEST MODE] Target: {target_train_count} train samples, {target_val_count} val samples (speaker-disjoint)")
    else:
        logger.info(f"Target train duration: {args.train_hours:.2f}h ({target_train_sec:.0f}s)")
        logger.info(f"Target val duration:   {args.val_hours:.2f}h ({target_val_sec:.0f}s)")

    hf_token = os.environ.get("HF_TOKEN")
    dataset_name = "ai4bharat/Svarah"

    try:
        logger.info(f"Connecting to official {dataset_name} repository on Hugging Face...")
        # Note: ai4bharat/Svarah is published by AI4Bharat as an ASR benchmark in a single 9.6h split named 'test'.
        # We attempt split='train' first; if not found, we load the official 9.6h 'test' split.
        try:
            dataset = load_dataset(dataset_name, split="train", streaming=True, token=hf_token)
            next(iter(dataset))
            upstream_split = "train"
            logger.info(f"Connected to {dataset_name} using upstream split='train'.")
        except Exception as split_err:
            logger.info(f"{dataset_name} has no upstream 'train' split ({split_err}).")
            logger.info(f"Loading official 9.6h {dataset_name} corpus via upstream split='test'...")
            dataset = load_dataset(dataset_name, split="test", streaming=True, token=hf_token)
            upstream_split = "test"

        dataset = dataset.cast_column("audio", datasets.Audio(decode=False))
        # Trigger first item to verify access
        next(iter(dataset))
        logger.info(f"Successfully authenticated and connected to {dataset_name} (upstream split: '{upstream_split}').")
    except Exception as e:
        error_msg = (
            f"\n{'='*70}\n"
            f"ERROR: Cannot access required dataset '{dataset_name}': {e}\n"
            f"The assignment specifically requires the AI4Bharat Svarah dataset.\n"
            f"Svarah is a gated Hugging Face repository. To resolve access:\n"
            f"  1. Accept terms at: https://huggingface.co/datasets/ai4bharat/Svarah\n"
            f"  2. Create access token at: https://huggingface.co/settings/tokens\n"
            f"  3. Set HF_TOKEN environment variable (e.g. export HF_TOKEN='hf_...').\n"
            f"Per assignment instructions, automatic substitution of other datasets is prohibited.\n"
            f"{'='*70}\n"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    train_speakers = set()
    val_speakers = set()
    train_samples = []
    val_samples = []
    train_sec = 0.0
    val_sec = 0.0

    sample_idx = 0
    for item in dataset:
        # Check termination conditions
        if is_smoke:
            if len(train_samples) >= target_train_count and len(val_samples) >= target_val_count:
                break
        else:
            if args.max_samples and (len(train_samples) + len(val_samples)) >= args.max_samples:
                break
            if train_sec >= target_train_sec and val_sec >= target_val_sec:
                break

        audio_data = item.get("audio", {})
        text = item.get("transcription", item.get("text", "")).strip()

        if not audio_data or not text:
            continue

        # Condition 8: Test sentence protection (exclude any utterance matching assignment evaluation sentences)
        lower_text = text.lower()
        if any(pat in lower_text for pat in ASSIGNMENT_TEST_PATTERNS):
            logger.info(f"Excluded utterance matching assignment test pattern: '{text[:40]}...'")
            continue

        # Speaker ID from metadata
        speaker_id = str(item.get("speaker_id", item.get("speaker", "unknown_speaker"))).strip()

        # Speaker-Disjoint Assignment Logic
        assigned_split = None
        if is_smoke:
            if len(train_samples) < target_train_count:
                if speaker_id not in val_speakers:
                    assigned_split = "train"
            elif len(val_samples) < target_val_count:
                if speaker_id not in train_speakers:  # ZERO speaker leakage
                    assigned_split = "val"
        else:
            if train_sec < target_train_sec:
                if speaker_id not in val_speakers:
                    assigned_split = "train"
            elif val_sec < target_val_sec:
                if speaker_id not in train_speakers:  # ZERO speaker leakage
                    assigned_split = "val"

        if assigned_split is None:
            continue

        # Decode audio bytes or array
        array = None
        sr = args.target_sr

        if isinstance(audio_data, dict):
            if audio_data.get("bytes"):
                try:
                    array, sr = sf.read(io.BytesIO(audio_data["bytes"]))
                except Exception as e:
                    logger.warning(f"Failed to decode audio bytes: {e}")
                    continue
            elif audio_data.get("path") and os.path.exists(audio_data["path"]):
                try:
                    array, sr = sf.read(audio_data["path"])
                except Exception as e:
                    logger.warning(f"Failed to read audio file: {e}")
                    continue
            elif "array" in audio_data and audio_data["array"] is not None:
                array = np.array(audio_data["array"], dtype=np.float32)
                sr = audio_data.get("sampling_rate", 16000)

        if array is None or len(array) == 0:
            continue

        # Convert to mono if multi-channel
        if len(array.shape) > 1:
            array = np.mean(array, axis=-1)

        # Check duration bounds
        duration = len(array) / sr
        if duration < args.min_utterance_sec or duration > args.max_utterance_sec:
            continue

        # Resample to target_sr if needed
        if sr != args.target_sr:
            array = librosa.resample(array.astype(np.float32), orig_sr=sr, target_sr=args.target_sr)
            sr = args.target_sr

        sample_idx += 1
        sid = f"english_{sample_idx:06d}"

        # Save audio
        wav_path = wavs_dir / f"{sid}.wav"
        sf.write(str(wav_path), array, sr, subtype="PCM_16")

        # Save text file (.txt) alongside WAV for aligner
        txt_path = wavs_dir / f"{sid}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        entry = {
            "id": sid,
            "audio_path": str(wav_path).replace("\\", "/"),
            "text": text,
            "speaker": speaker_id,
            "duration": round(duration, 3),
        }

        if assigned_split == "train":
            train_speakers.add(speaker_id)
            train_samples.append(entry)
            train_sec += duration
        else:
            val_speakers.add(speaker_id)
            val_samples.append(entry)
            val_sec += duration

    # Integrity Assertions
    speaker_overlap = train_speakers.intersection(val_speakers)
    if len(speaker_overlap) > 0:
        raise AssertionError(f"Speaker leakage detected between train and val: {speaker_overlap}")

    train_ids = {s["id"] for s in train_samples}
    val_ids = {s["id"] for s in val_samples}
    id_overlap = train_ids.intersection(val_ids)
    if len(id_overlap) > 0:
        raise AssertionError(f"Duplicate utterance IDs detected: {id_overlap}")

    logger.info("=" * 70)
    logger.info(f"Svarah Split Preparation Completed (Upstream split: '{upstream_split}')")
    logger.info(f"Train: {len(train_samples)} samples ({train_sec/3600:.2f}h) across {len(train_speakers)} speakers")
    logger.info(f"Val:   {len(val_samples)} samples ({val_sec/3600:.2f}h) across {len(val_speakers)} speakers")
    logger.info(f"Speaker Overlap: {len(speaker_overlap)} (Strictly Speaker-Disjoint: {len(speaker_overlap) == 0})")
    logger.info("=" * 70)

    # Write manifests
    for name, samples in [("train", train_samples), ("val", val_samples)]:
        manifest_file = splits_dir / f"{name}.txt"
        with open(manifest_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(f"{s['id']}|{s['audio_path']}|{s['text']}|{s['speaker']}\n")
        logger.info(f"Saved manifest: {manifest_file} ({len(samples)} lines)")

    # Write metadata JSON
    metadata = {
        "dataset_source": dataset_name,
        "upstream_split": upstream_split,
        "split_strategy": "speaker_disjoint_sequential",
        "random_seed": args.seed,
        "mode": "smoke_test" if is_smoke else "full_dataset",
        "train_speaker_count": len(train_speakers),
        "val_speaker_count": len(val_speakers),
        "speaker_overlap_count": len(speaker_overlap),
        "train_speakers": sorted(list(train_speakers)),
        "val_speakers": sorted(list(val_speakers)),
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "train_duration_sec": round(train_sec, 2),
        "val_duration_sec": round(val_sec, 2),
        "train_duration_hours": round(train_sec / 3600.0, 3),
        "val_duration_hours": round(val_sec / 3600.0, 3),
        "target_sr": args.target_sr,
        "note": "Derived from official ai4bharat/Svarah 9.6h corpus (which upstream publishes exclusively under split='test') specifically for the IIIT-H TTS assignment with zero speaker leakage and test sentence protection."
    }

    meta_file = splits_dir / "split_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved split metadata: {meta_file}")


if __name__ == "__main__":
    args = parse_args()
    prepare_svarah(args)
