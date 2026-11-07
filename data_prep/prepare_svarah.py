"""
Svarah Dataset Preparation Script for FastSpeech 2 (English)
============================================================

This script downloads and partitions the AI4Bharat Svarah English dataset
into speaker-disjoint training (~5h) and validation (~1h) subsets.

Official Dataset Details:
  - Hugging Face ID: ai4bharat/Svarah
  - Benchmark details: 9.6 hours of Indian-accented English speech from 117 speakers
    across 65 districts in 19 states of India.
  - Published split: ai4bharat/Svarah is published exclusively as split='test'
    (the official ASR benchmark evaluation split).
  - Schema:
      * audio_filepath: datasets.features._torchcodec.AudioDecoder (or dict with array/bytes/path)
      * duration: float64 (ground-truth duration in seconds)
      * text: ground-truth English transcript string
      * native_place_state, native_place_district, primary_language, gender, age-group,
        highest_qualification, job_category, occupation_domain

Usage:
    python prepare_svarah.py --output_dir data/svarah --splits_dir English/data_splits
    python prepare_svarah.py --smoke_test --output_dir data/svarah_smoke --splits_dir English/data_splits
    python prepare_svarah.py --diagnose

Author: FastSpeech 2 English-Telugu TTS Project
"""

import os
import sys
import io
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
    parser.add_argument("--diagnose", action="store_true", help="Print schema, keys, types, AudioDecoder stats, and verification for the first 5 records without writing files")
    return parser.parse_args()


def extract_audio_and_sr(audio_data):
    """Robustly extract (numpy_array, orig_sr) from various HuggingFace audio representations.

    Supports:
      - datasets.features._torchcodec.AudioDecoder (modern datasets 4.x / torchcodec)
      - torchcodec.decoders.AudioDecoder
      - dict with 'array' and 'sampling_rate'
      - dict with raw 'bytes'
      - dict with 'path'
      - direct file path string
    """
    import soundfile as sf

    if audio_data is None:
        return None, None

    # Case 1: AudioDecoder object (datasets >= 4.0 / torchcodec)
    if hasattr(audio_data, "get_all_samples") or hasattr(audio_data, "get_samples"):
        try:
            if hasattr(audio_data, "get_all_samples"):
                samples = audio_data.get_all_samples()
            else:
                samples = audio_data.get_samples()

            # samples container exposes .data (torch.Tensor or ndarray) and .sample_rate (int)
            data = getattr(samples, "data", samples)
            sr = getattr(samples, "sample_rate", None)

            # Fallback for sample_rate from decoder metadata
            if sr is None and hasattr(audio_data, "metadata"):
                sr = getattr(audio_data.metadata, "sample_rate", None)
            if sr is None:
                sr = getattr(audio_data, "sample_rate", 16000)

            # Convert PyTorch Tensor to NumPy
            if hasattr(data, "detach"):
                arr = data.detach().cpu().numpy()
            elif isinstance(data, np.ndarray):
                arr = data
            else:
                arr = np.array(data, dtype=np.float32)

            # Handle multi-channel / tensor dimensions: TorchCodec typically produces [channels, samples]
            if arr.ndim > 1:
                # If shape is [channels, samples] (e.g. [1, N] or [2, N])
                if arr.shape[0] in [1, 2, 4] and arr.shape[0] < arr.shape[-1]:
                    arr = np.mean(arr, axis=0)
                elif arr.shape[-1] in [1, 2]:
                    arr = np.mean(arr, axis=-1)
                else:
                    arr = arr.flatten()

            arr = arr.astype(np.float32)
            sr = int(sr)

            if len(arr) > 0 and np.all(np.isfinite(arr)):
                return arr, sr
            else:
                logger.warning(f"AudioDecoder returned invalid array (len={len(arr)}, isfinite={np.all(np.isfinite(arr))})")
        except Exception as e:
            logger.warning(f"AudioDecoder decoding via get_all_samples() failed: {e}")

    # Case 2: Audio is decoded into a dictionary with 'array'
    if isinstance(audio_data, dict) and "array" in audio_data and audio_data["array"] is not None:
        try:
            data = audio_data["array"]
            if hasattr(data, "detach"):
                arr = data.detach().cpu().numpy()
            else:
                arr = np.array(data, dtype=np.float32)
            if arr.ndim > 1:
                arr = np.mean(arr, axis=-1)
            sr = int(audio_data.get("sampling_rate", 16000))
            if len(arr) > 0 and np.all(np.isfinite(arr)):
                return arr.astype(np.float32), sr
        except Exception as e:
            logger.warning(f"Failed to process dict array: {e}")

    # Case 3: Audio contains raw file bytes
    if isinstance(audio_data, dict) and audio_data.get("bytes"):
        try:
            arr, sr = sf.read(io.BytesIO(audio_data["bytes"]))
            if arr.ndim > 1:
                arr = np.mean(arr, axis=-1)
            if len(arr) > 0 and np.all(np.isfinite(arr)):
                return arr.astype(np.float32), sr
        except Exception as e:
            logger.debug(f"Failed to decode audio bytes: {e}")

    # Case 4: Audio dictionary contains local file path
    if isinstance(audio_data, dict) and audio_data.get("path") and os.path.exists(audio_data["path"]):
        try:
            arr, sr = sf.read(audio_data["path"])
            if arr.ndim > 1:
                arr = np.mean(arr, axis=-1)
            if len(arr) > 0 and np.all(np.isfinite(arr)):
                return arr.astype(np.float32), sr
        except Exception as e:
            logger.debug(f"Failed to read audio path: {e}")

    # Case 5: audio_data is a direct file path string
    if isinstance(audio_data, str) and os.path.exists(audio_data):
        try:
            arr, sr = sf.read(audio_data)
            if arr.ndim > 1:
                arr = np.mean(arr, axis=-1)
            if len(arr) > 0 and np.all(np.isfinite(arr)):
                return arr.astype(np.float32), sr
        except Exception as e:
            logger.debug(f"Failed to read audio string path: {e}")

    return None, None


def extract_speaker_id(item, audio_data=None):
    """Extract or determine speaker ID from actual Svarah dataset features."""
    # 1. Direct speaker columns
    for spk_key in ["speaker_id", "speaker", "client_id", "spk_id", "speaker_name"]:
        val = item.get(spk_key)
        if val is not None and str(val).strip():
            return str(val).strip()

    # 2. Path prefix if audio has a path
    path_val = None
    if isinstance(audio_data, dict):
        path_val = audio_data.get("path")
    elif isinstance(audio_data, str):
        path_val = audio_data

    if path_val:
        stem = Path(path_val).stem
        parts = stem.split("_")
        if len(parts) >= 2:
            candidate = "_".join(parts[:-1])
            if len(candidate) >= 3:
                return candidate

    # 3. Ground-truth demographic grouping from Svarah's 11 demographic features
    # Svarah features: native_place_state, native_place_district, primary_language, gender, age-group
    state = str(item.get("native_place_state") or "").strip().replace(" ", "_")
    district = str(item.get("native_place_district") or "").strip().replace(" ", "_")
    lang = str(item.get("primary_language") or "").strip().replace(" ", "_")
    gender = str(item.get("gender") or "").strip().replace(" ", "_")
    age = str(item.get("age-group") or "").strip().replace(" ", "_")

    demo_parts = [p for p in [state, district, lang, gender, age] if p]
    if demo_parts:
        return "_".join(demo_parts)

    return "unknown_speaker"


def diagnose_record(idx, item):
    """Deep inspection of a single record, verifying AudioDecoder decoding."""
    audio_obj = item.get("audio_filepath") or item.get("audio")
    txt = item.get("text") or item.get("transcription") or ""
    dur_field = item.get("duration")

    logger.info(f"\n--- [RECORD #{idx}] Svarah Record Diagnostic Inspection ---")
    logger.info(f"Top-level keys ({len(item.keys())}): {list(item.keys())}")
    for k, v in item.items():
        if k in ["audio_filepath", "audio"]:
            methods = [m for m in dir(v) if not m.startswith("_")]
            logger.info(f"  * {k} (type: {type(v).__name__}, module: {type(v).__module__}):")
            logger.info(f"    Available methods/attrs: {methods[:10]}")
            if hasattr(v, "metadata"):
                logger.info(f"    metadata: {getattr(v, 'metadata', None)}")
        else:
            logger.info(f"  * {k} ({type(v).__name__}): {repr(v)[:80]}")

    arr, sr = extract_audio_and_sr(audio_obj)
    spk = extract_speaker_id(item, audio_obj)

    if arr is not None:
        calc_dur = len(arr) / sr
        is_finite = bool(np.all(np.isfinite(arr)))
        arr_stats = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sample_rate": sr,
            "duration_sec": round(calc_dur, 3),
            "is_finite": is_finite,
            "min": round(float(np.min(arr)), 5),
            "max": round(float(np.max(arr)), 5),
            "mean": round(float(np.mean(arr)), 6),
            "std": round(float(np.std(arr)), 5),
        }
        logger.info(f"  [AUDIO DECODER VERIFICATION]")
        logger.info(f"    -> Decoded waveform: SUCCESS")
        logger.info(f"    -> Stats: {arr_stats}")
        if dur_field is not None:
            logger.info(f"    -> Discrepancy with upstream duration: {abs(calc_dur - float(dur_field)):.4f}s")
    else:
        logger.error(f"  [AUDIO DECODER VERIFICATION] FAILED: extract_audio_and_sr returned None!")

    logger.info(f"  [TEXT]: '{txt[:60]}...' (len: {len(txt)})")
    logger.info(f"  [SPEAKER]: '{spk}'")
    return arr is not None


def prepare_svarah(args):
    try:
        import datasets
        from datasets import load_dataset
        import soundfile as sf
        import librosa
    except ImportError as e:
        logger.error(f"Missing required library: {e}. Run: pip install datasets soundfile librosa")
        sys.exit(1)

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

        # Ensure raw audio bytes are preserved without requiring optional torchcodec binary
        import datasets
        try:
            dataset = dataset.cast_column("audio_filepath", datasets.Audio(decode=False))
        except Exception as cast_e:
            logger.debug(f"Could not cast audio_filepath to decode=False: {cast_e}")

        logger.info(f"Successfully connected to {dataset_name} (upstream split: '{upstream_split}').")
    except Exception as e:
        error_msg = (
            f"\n{'='*70}\n"
            f"ERROR: Cannot access required dataset '{dataset_name}': {e}\n"
            f"The assignment specifically requires the AI4Bharat Svarah dataset.\n"
            f"Please ensure HF_TOKEN is set with access to '{dataset_name}'.\n"
            f"{'='*70}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # If --diagnose flag is set, run schema and audio decoding inspection on 5 records and exit cleanly
    if getattr(args, "diagnose", False):
        logger.info("=" * 70)
        logger.info("RUNNING SVARAH DATASET DIAGNOSTIC INSPECTION (5 Records)")
        logger.info("=" * 70)
        diag_count = 0
        for item in dataset:
            diag_count += 1
            diagnose_record(diag_count, item)
            if diag_count >= 5:
                break
        logger.info("=" * 70)
        logger.info("DIAGNOSTIC INSPECTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        return

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

    train_samples = []
    val_samples = []
    train_speakers = set()
    val_speakers = set()
    train_sec = 0.0
    val_sec = 0.0

    sample_idx = 0
    inspected_count = 0
    max_inspections = 500 if is_smoke else 50000

    logger.info("Beginning Svarah sample collection & evaluation...")
    it = iter(dataset)
    try:
        for item in it:
            inspected_count += 1
            if inspected_count > max_inspections:
                logger.warning(f"Reached inspection limit ({max_inspections} records). Stopping stream iteration.")
                break

            # Termination conditions
            if is_smoke:
                if len(train_samples) >= target_train_count and len(val_samples) >= target_val_count:
                    break
            else:
                if args.max_samples and (len(train_samples) + len(val_samples)) >= args.max_samples:
                    break
                if train_sec >= target_train_sec and val_sec >= target_val_sec:
                    break

            is_early_record = inspected_count <= 10

            # For the first 5 records in smoke test mode, run full diagnostic inspection
            if is_smoke and inspected_count <= 5:
                diagnose_record(inspected_count, item)

            # Step 1: Audio extraction (official Svarah schema uses 'audio_filepath' as AudioDecoder)
            audio_data = None
            audio_source_key = None
            for key in ["audio_filepath", "audio", "wav", "audio_file"]:
                if key in item and item[key] is not None:
                    audio_data = item[key]
                    audio_source_key = key
                    break

            if not audio_data:
                if is_early_record:
                    logger.info(f"[Record #{inspected_count}] REJECTED: No audio found. Available keys: {list(item.keys())}")
                continue

            # Step 2: Text transcript extraction (official Svarah schema uses 'text')
            text = ""
            text_source_key = None
            for key in ["text", "transcription", "transcript", "sentence", "normalized_text"]:
                if key in item and item[key] is not None:
                    candidate = str(item[key]).strip()
                    if candidate:
                        text = candidate
                        text_source_key = key
                        break

            if not text:
                if is_early_record:
                    logger.info(f"[Record #{inspected_count}] REJECTED: Empty transcript text (audio key: '{audio_source_key}')")
                continue

            # Step 3: Test pattern exclusion
            lower_text = text.lower()
            if any(pat in lower_text for pat in ASSIGNMENT_TEST_PATTERNS):
                if is_early_record:
                    logger.info(f"[Record #{inspected_count}] REJECTED: Excluded assignment test pattern: '{text[:40]}...'")
                continue

            # Step 4: Speaker ID extraction
            speaker_id = extract_speaker_id(item, audio_data)

            # Step 5: Speaker-disjoint assignment logic
            assigned_split = None
            if is_smoke:
                if len(train_samples) < target_train_count:
                    if speaker_id not in val_speakers:
                        assigned_split = "train"
                    else:
                        if is_early_record:
                            logger.info(f"[Record #{inspected_count}] REJECTED for TRAIN: Speaker '{speaker_id}' already assigned to VAL")
                elif len(val_samples) < target_val_count:
                    if speaker_id not in train_speakers:  # ZERO speaker leakage
                        assigned_split = "val"
                    else:
                        if is_early_record:
                            logger.info(f"[Record #{inspected_count}] REJECTED for VAL: Speaker '{speaker_id}' already assigned to TRAIN")
            else:
                if train_sec < target_train_sec:
                    if speaker_id not in val_speakers:
                        assigned_split = "train"
                elif val_sec < target_val_sec:
                    if speaker_id not in train_speakers:  # ZERO speaker leakage
                        assigned_split = "val"

            if assigned_split is None:
                if is_early_record:
                    logger.info(f"[Record #{inspected_count}] REJECTED: No split available (target count/hours reached or speaker collision)")
                continue

            # Step 6: Audio decoding via AudioDecoder / torchcodec
            array, sr = extract_audio_and_sr(audio_data)
            if array is None or len(array) == 0:
                if is_early_record:
                    logger.info(f"[Record #{inspected_count}] REJECTED: Audio decoding failed from key '{audio_source_key}' (type: {type(audio_data)})")
                continue

            # Convert to mono if multi-channel
            if len(array.shape) > 1:
                array = np.mean(array, axis=-1)

            # Step 7: Duration bounds
            duration = len(array) / sr
            if duration < args.min_utterance_sec or duration > args.max_utterance_sec:
                if is_early_record:
                    logger.info(f"[Record #{inspected_count}] REJECTED: Duration {duration:.2f}s outside bounds [{args.min_utterance_sec}s, {args.max_utterance_sec}s]")
                continue

            # Record accepted!
            sample_idx += 1
            sid = f"english_{sample_idx:06d}"

            # Save genuine audio file (resample to target_sr if needed)
            if sr != args.target_sr:
                array = librosa.resample(array.astype(np.float32), orig_sr=sr, target_sr=args.target_sr)
                sr = args.target_sr

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
                "speaker": speaker_id,
                "duration": round(duration, 3),
            }

            if assigned_split == "train":
                train_speakers.add(speaker_id)
                train_samples.append(entry)
                train_sec += duration
                logger.info(f"[TRAIN {len(train_samples):02d}/{target_train_count or '5h'}] {sid} | Spk: {speaker_id[:25]} | Dur: {duration:.2f}s | Key: '{audio_source_key}' | '{text[:35]}...'")
            else:
                val_speakers.add(speaker_id)
                val_samples.append(entry)
                val_sec += duration
                logger.info(f"[VAL   {len(val_samples):02d}/{target_val_count or '1h'}] {sid} | Spk: {speaker_id[:25]} | Dur: {duration:.2f}s | Key: '{audio_source_key}' | '{text[:35]}...'")

    except OSError as e:
        if getattr(e, "errno", None) == 9:
            logger.debug(f"Handled transient socket close during streaming: {e}")
        else:
            raise

    # Integrity Assertions: Verify collection success
    if is_smoke:
        if len(train_samples) != target_train_count or len(val_samples) != target_val_count:
            raise RuntimeError(
                f"SMOKE TEST FAILED: Required {target_train_count} train and {target_val_count} val samples, "
                f"but only collected {len(train_samples)} train and {len(val_samples)} val samples from {dataset_name}! "
                f"Inspected {inspected_count} records."
            )
    else:
        if len(train_samples) == 0 or len(val_samples) == 0:
            raise RuntimeError(
                f"DATA PREPARATION FAILED: No samples collected for train ({len(train_samples)}) or val ({len(val_samples)})!"
            )

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
    logger.info(f"Speaker Overlap: {len(speaker_overlap)} (Strictly Speaker-Disjoint: True)")
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
        "sample_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
        },
    }
    meta_path = splits_dir / "split_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved split metadata: {meta_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    prepare_svarah(args)
