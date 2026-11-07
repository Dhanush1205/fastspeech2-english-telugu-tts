"""Prepare the SPRINGLab IndicTTS Telugu corpus for FastSpeech 2.

IndicTTS currently exposes AudioDecoder records with text and gender. It does
not expose a stable speaker identifier in its public dataset schema. We
therefore create a deterministic utterance-disjoint split and explicitly mark
speaker identity as unavailable; no speaker-disjoint claim is made.

The script writes manifests only after the requested targets are reached and
all emitted WAVs have been checked.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("prepare_indictts")


# Exact mandatory assignment test utterances are excluded from the corpus pool.
ASSIGNMENT_TEST_SENTENCES = (
    "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
    "పులి అడ  విలో నిశ్శబ్దంగా నడిచింది.",
    "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
)


def _normalize_for_test_match(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", normalized).strip()


_EXCLUDED_TEST_TEXTS = {
    _normalize_for_test_match(text) for text in ASSIGNMENT_TEST_SENTENCES
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare IndicTTS Telugu data for FastSpeech 2"
    )
    parser.add_argument("--output_dir", default="data/indictts_telugu")
    parser.add_argument("--splits_dir", default="Telugu/data_splits")
    parser.add_argument("--train_hours", type=float, default=5.0)
    parser.add_argument("--val_hours", type=float, default=1.0)
    parser.add_argument("--target_sr", type=int, default=22050)
    parser.add_argument("--max_utterance_sec", type=float, default=12.0)
    parser.add_argument("--min_utterance_sec", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _numpy_audio(data: Any) -> np.ndarray:
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    arr = np.asarray(data, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        # AudioDecoder / TorchCodec convention: [channels, samples].
        # SoundFile convention: [samples, channels].
        if arr.shape[0] <= 8 and arr.shape[1] > 8:
            arr = arr.mean(axis=0)
        elif arr.shape[1] <= 8 and arr.shape[0] > 8:
            arr = arr.mean(axis=1)
        else:
            raise ValueError(f"Ambiguous audio channel layout: {arr.shape}")
    elif arr.ndim != 1:
        raise ValueError(f"Expected mono/stereo audio, got shape {arr.shape}")
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("Audio is empty or contains NaN/Inf")
    peak = float(np.max(np.abs(arr)))
    if peak > 1.0001:
        if peak <= 32768:
            arr = arr / 32768.0
        else:
            raise ValueError(f"Unexpected audio amplitude peak: {peak}")
    return arr.astype(np.float32, copy=False)


def decode_audio(audio: Any) -> tuple[np.ndarray, int]:
    """Decode a Hugging Face AudioDecoder or a legacy Audio dictionary."""
    import soundfile as sf
    import io

    if audio is None:
        raise ValueError("Missing audio")
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        data = getattr(samples, "data", None)
        sr = getattr(samples, "sample_rate", None)
        if data is None:
            raise ValueError("AudioDecoder returned no sample data")
        if sr is None:
            metadata = getattr(audio, "metadata", None)
            sr = getattr(metadata, "sample_rate", None)
        if sr is None:
            raise ValueError("AudioDecoder did not expose a sample rate")
        return _numpy_audio(data), int(sr)

    if hasattr(audio, "get_samples"):
        samples = audio.get_samples()
        data = getattr(samples, "data", None)
        sr = getattr(samples, "sample_rate", None)
        if data is None or sr is None:
            raise ValueError("get_samples() did not expose data and sample_rate")
        return _numpy_audio(data), int(sr)

    if isinstance(audio, dict):
        if audio.get("bytes"):
            waveform, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            return _numpy_audio(waveform), int(sr)
        if audio.get("path") and os.path.isfile(audio["path"]):
            waveform, sr = sf.read(audio["path"], dtype="float32")
            return _numpy_audio(waveform), int(sr)
        if audio.get("array") is not None:
            sr = audio.get("sampling_rate")
            if not sr:
                raise ValueError("Audio array has no sampling_rate")
            return _numpy_audio(audio["array"]), int(sr)

    raise TypeError(f"Unsupported IndicTTS audio type: {type(audio)!r}")


def resample_audio(waveform: np.ndarray, source_sr: int, target_sr: int):
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive")
    if source_sr == target_sr:
        return waveform.astype(np.float32, copy=False)
    from scipy.signal import resample_poly
    from math import gcd

    divisor = gcd(source_sr, target_sr)
    result = resample_poly(
        waveform,
        target_sr // divisor,
        source_sr // divisor,
    )
    result = np.asarray(result, dtype=np.float32)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError("Resampling produced invalid audio")
    return result


def _write_manifest(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            text = row["text"].replace("|", " ").replace("\r", " ").replace("\n", " ")
            handle.write(f'{row["id"]}|{row["audio_path"]}|{text}|{row.get("speaker", "unknown")}\n')


def prepare_indictts(args):
    from datasets import load_dataset
    import soundfile as sf

    if args.train_hours <= 0 or args.val_hours <= 0:
        raise ValueError("train_hours and val_hours must be positive")
    if args.min_utterance_sec <= 0 or args.max_utterance_sec <= args.min_utterance_sec:
        raise ValueError("Invalid utterance duration bounds")

    smoke = args.smoke_test
    train_count_target = 10 if smoke else None
    val_count_target = 4 if smoke else None
    train_target_sec = args.train_hours * 3600.0
    val_target_sec = args.val_hours * 3600.0

    output = Path(args.output_dir)
    wavs_dir = output / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SPRINGLab/IndicTTS_Telugu train stream")
    dataset = load_dataset(
        "SPRINGLab/IndicTTS_Telugu",
        split="train",
        streaming=True,
    )
    logger.info("Dataset features: %s", dataset.features)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    train_seconds = 0.0
    val_seconds = 0.0
    accepted = 0
    rejected = 0
    source_index = 0

    for item in dataset:
        if smoke and len(train_rows) >= train_count_target and len(val_rows) >= val_count_target:
            break
        if not smoke:
            if args.max_samples is not None and accepted >= args.max_samples:
                break
            if train_seconds >= train_target_sec and val_seconds >= val_target_sec:
                break

        source_index += 1
        transcript = str(item.get("text") or item.get("transcription") or "").strip()
        if not transcript:
            rejected += 1
            continue
        if _normalize_for_test_match(transcript) in _EXCLUDED_TEST_TEXTS:
            rejected += 1
            logger.info("Excluded mandatory assignment evaluation utterance at record %d", source_index)
            continue
        try:
            waveform, source_sr = decode_audio(item.get("audio"))
            source_duration = len(waveform) / source_sr
            if not args.min_utterance_sec <= source_duration <= args.max_utterance_sec:
                rejected += 1
                continue
            waveform = resample_audio(waveform, source_sr, args.target_sr)
            duration = len(waveform) / args.target_sr

            sample_id = f"telugu_{source_index:06d}"
            wav_path = wavs_dir / f"{sample_id}.wav"
            txt_path = wavs_dir / f"{sample_id}.txt"
            sf.write(str(wav_path), waveform, args.target_sr, subtype="PCM_16")

            # Verify the physical output file, not just the in-memory waveform.
            check, check_sr = sf.read(str(wav_path), dtype="float32")
            if check_sr != args.target_sr or check.size == 0 or not np.isfinite(check).all():
                raise IOError(f"Saved WAV verification failed: {wav_path}")
            if abs(len(check) / check_sr - duration) > 1.0 / check_sr:
                raise IOError(f"Saved WAV duration mismatch: {wav_path}")
            txt_path.write_text(transcript, encoding="utf-8")
            if txt_path.read_text(encoding="utf-8") != transcript:
                raise IOError(f"Transcript round-trip verification failed: {txt_path}")

            gender_value = item.get("gender")
            gender_name = None
            gender_feature = getattr(dataset.features, "get", lambda _k: None)("gender")
            if gender_value is not None and gender_feature is not None and hasattr(gender_feature, "int2str"):
                try:
                    gender_name = str(gender_feature.int2str(int(gender_value))).strip().lower()
                except Exception:
                    gender_name = None
            elif gender_value is not None:
                gender_name = str(gender_value).strip().lower()
            # The inspected IndicTTS schema exposes gender but no stable speaker ID.
            # Keep the gender grouping as an explicit proxy for MFA corpus layout only.
            speaker_proxy = f"gender_{gender_name}" if gender_name in {"female", "male"} else "unknown"

            row = {
                "id": sample_id,
                "audio_path": wav_path.as_posix(),
                "text": transcript,
                "duration": duration,
                "source_index": source_index,
                "gender": gender_name,
                "speaker": speaker_proxy,
            }

            # Deterministic sequential allocation, with each source utterance
            # assigned exactly once. Telugu source schema does not expose IDs.
            if smoke:
                if len(train_rows) < train_count_target:
                    train_rows.append(row)
                    train_seconds += duration
                else:
                    val_rows.append(row)
                    val_seconds += duration
            elif train_seconds < train_target_sec:
                train_rows.append(row)
                train_seconds += duration
            else:
                val_rows.append(row)
                val_seconds += duration
            accepted += 1

            if accepted % 100 == 0:
                logger.info(
                    "Accepted=%d rejected=%d train=%.3fh val=%.3fh",
                    accepted, rejected, train_seconds / 3600, val_seconds / 3600
                )
        except Exception as exc:
            rejected += 1
            logger.warning("Rejected source record %d: %s", source_index, exc)

    if smoke:
        complete = len(train_rows) == train_count_target and len(val_rows) == val_count_target
    else:
        complete = train_seconds >= train_target_sec and val_seconds >= val_target_sec

    logger.info(
        "Collected train=%d (%.3fh), val=%d (%.3fh), accepted=%d, rejected=%d",
        len(train_rows), train_seconds / 3600,
        len(val_rows), val_seconds / 3600, accepted, rejected,
    )
    if not complete:
        raise RuntimeError(
            "The stream ended or max_samples was reached before all requested "
            "duration/sample targets were met. Existing manifests were not replaced."
        )

    train_ids = {row["id"] for row in train_rows}
    val_ids = {row["id"] for row in val_rows}
    overlap = train_ids & val_ids
    if overlap:
        raise RuntimeError(f"Utterance leakage detected: {sorted(overlap)[:10]}")

    _write_manifest(splits_dir / "train.txt", train_rows)
    _write_manifest(splits_dir / "val.txt", val_rows)
    metadata = {
        "dataset_source": "SPRINGLab/IndicTTS_Telugu",
        "upstream_split": "train",
        "split_strategy": "deterministic_utterance_disjoint_sequential",
        "speaker_identity_available": False,
        "speaker_disjoint": None,
        "speaker_grouping_field": "gender proxy",
        "speaker_note": (
            "The inspected source schema exposes audio, text, and gender but no "
            "stable speaker ID. The gender_female/gender_male values are used "
            "only as alignment-corpus speaker folders. They are not claimed as "
            "verified per-utterance speaker IDs; speaker-disjointness remains unverified."
        ),
        "mode": "smoke_test" if smoke else "full_dataset",
        "seed": args.seed,
        "train_sample_count": len(train_rows),
        "val_sample_count": len(val_rows),
        "train_duration_sec": round(train_seconds, 3),
        "val_duration_sec": round(val_seconds, 3),
        "train_duration_hours": round(train_seconds / 3600, 4),
        "val_duration_hours": round(val_seconds / 3600, 4),
        "target_train_hours": None if smoke else args.train_hours,
        "target_val_hours": None if smoke else args.val_hours,
        "target_sample_rate": args.target_sr,
        "accepted_records": accepted,
        "rejected_records": rejected,
        "utterance_overlap_count": len(overlap),
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
    }
    (splits_dir / "split_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Telugu preparation complete. Verified manifests: %s", splits_dir)


if __name__ == "__main__":
    prepare_indictts(parse_args())
