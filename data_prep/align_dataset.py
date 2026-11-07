"""Prepare speaker-organized MFA corpora and run language-specific alignment.

English uses the legacy ARPABET English US acoustic model/dictionary. Telugu
uses the official AI4Bharat IndicMFA G2G acoustic model and dictionary.

Each manifest row must contain: id|audio_path|text|speaker_group.
The fourth field is only a corpus-folder grouping label. It is not asserted to
be a verified speaker identity.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlretrieve
import unicodedata

import soundfile as sf

logger = logging.getLogger("align_dataset")


def parse_args():
    parser = argparse.ArgumentParser(description="Create MFA corpus and run forced alignment")
    parser.add_argument("--language", choices=["english", "telugu"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--mfa", default="mfa")
    parser.add_argument("--indicmfa_model_dir", default="indicmfa_models")
    parser.add_argument("--indicmfa_model", default=None)
    parser.add_argument("--indicmfa_dictionary", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path):
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("|", 3)
            if len(parts) != 4:
                raise ValueError(f"{path}:{line_no}: expected 4 pipe-separated fields")
            sample_id, audio_path, text, speaker_group = [part.strip() for part in parts]
            if not sample_id or not audio_path or not text:
                raise ValueError(f"{path}:{line_no}: empty ID, audio path, or transcript")
            if sample_id in seen:
                raise ValueError(f"{path}:{line_no}: duplicate sample ID {sample_id}")
            seen.add(sample_id)
            rows.append({
                "id": sample_id,
                "audio_path": Path(audio_path),
                "text": text,
                "speaker_group": speaker_group or "unknown",
            })
    if not rows:
        raise ValueError(f"No samples found in manifest {path}")
    return rows


def _safe_group_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    if not slug:
        slug = "unknown"
    return slug[:80]


def _hardlink_or_copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _english_lab(text: str) -> str:
    from English.text_processing import clean_english_text
    cleaned = clean_english_text(text)
    if not cleaned:
        raise ValueError("English transcript became empty after text normalization")
    return cleaned


def _telugu_lab(text: str) -> str:
    from Telugu.text_processing import telugu_text_normalize
    normalized = telugu_text_normalize(text, preserve_exact=False)
    letters = [
        char for char in normalized
        if 0x0C00 <= ord(char) <= 0x0C7F
    ]
    if not letters:
        raise ValueError("Telugu transcript contains no Telugu-block characters")
    # IndicMFA's G2G workflow requires letters separated by spaces.
    return " ".join(letters)


def build_corpus(rows, language: str, corpus_dir: Path, force=False):
    if corpus_dir.exists() and any(corpus_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Corpus directory {corpus_dir} is not empty. Use --force to rebuild it."
            )
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    lab_builder = _english_lab if language == "english" else _telugu_lab
    used_files = set()
    for row in rows:
        source = row["audio_path"]
        if not source.is_file():
            raise FileNotFoundError(f"Missing audio file for {row['id']}: {source}")
        info = sf.info(str(source))
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f"Invalid WAV header for {source}")
        lab_text = lab_builder(row["text"])

        group = _safe_group_name(row["speaker_group"])
        speaker_dir = corpus_dir / group
        wav_dest = speaker_dir / f'{row["id"]}.wav'
        lab_dest = speaker_dir / f'{row["id"]}.lab'
        if wav_dest in used_files or lab_dest in used_files:
            raise ValueError(f"Duplicate corpus filename for ID {row['id']}")
        _hardlink_or_copy(source, wav_dest)
        lab_dest.write_text(lab_text, encoding="utf-8")
        used_files.add(wav_dest)
        used_files.add(lab_dest)

    audio_count = len(list(corpus_dir.rglob("*.wav")))
    lab_count = len(list(corpus_dir.rglob("*.lab")))
    if audio_count != len(rows) or lab_count != len(rows):
        raise RuntimeError(
            f"Corpus build mismatch: expected {len(rows)} WAV/LAB pairs, "
            f"found {audio_count} WAV and {lab_count} LAB"
        )
    return audio_count


def _run(command):
    logger.info("Running: %s", " ".join(str(x) for x in command))
    subprocess.run([str(x) for x in command], check=True)


def _align_english(args, corpus_dir: Path, output_dir: Path):
    _run([args.mfa, "model", "download", "acoustic", "english_us_arpa"])
    _run([args.mfa, "model", "download", "dictionary", "english_us_arpa"])
    _run([
        args.mfa, "align", "--clean", "-j", str(args.jobs), "--use_mp",
        str(corpus_dir), "english_us_arpa", "english_us_arpa", str(output_dir),
    ])


def _align_telugu(args, corpus_dir: Path, output_dir: Path):
    model_dir = Path(args.indicmfa_model_dir)
    model_path = Path(args.indicmfa_model) if args.indicmfa_model else model_dir / "Telugu_Acoustic_Model.zip"
    dict_path = Path(args.indicmfa_dictionary) if args.indicmfa_dictionary else model_dir / "Telugu_Dictionary_g2g.txt"
    model_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://github.com/AI4Bharat/IndicMFA/releases/download/Telugu"

    if not model_path.is_file():
        logger.info("Downloading official IndicMFA Telugu acoustic model")
        urlretrieve(f"{base_url}/Telugu_Acoustic_Model.zip", str(model_path))
    if not dict_path.is_file():
        logger.info("Downloading official IndicMFA Telugu G2G dictionary")
        urlretrieve(f"{base_url}/Telugu_Dictionary_g2g.txt", str(dict_path))

    _run([
        args.mfa, "align", "--clean", "-j", str(args.jobs), "--use_mp",
        "--phone_set", "UNKNOWN", "--no_textgrid_cleanup",
        str(corpus_dir), str(dict_path), str(model_path), str(output_dir),
    ])


def main(args=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = args or parse_args()
    manifest = Path(args.manifest)
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    rows = read_manifest(manifest)
    count = build_corpus(rows, args.language, corpus_dir, force=args.force)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.language == "english":
        _align_english(args, corpus_dir, output_dir)
    else:
        _align_telugu(args, corpus_dir, output_dir)

    textgrids = list(output_dir.rglob("*.TextGrid")) + list(output_dir.rglob("*.textgrid"))
    if len(textgrids) != count:
        raise RuntimeError(
            f"Alignment incomplete: expected {count} TextGrids, found {len(textgrids)}. "
            "Inspect MFA output and OOV/unaligned utterance logs."
        )

    metadata = {
        "language": args.language,
        "manifest": manifest.as_posix(),
        "corpus_directory": corpus_dir.as_posix(),
        "output_directory": output_dir.as_posix(),
        "sample_count": count,
        "textgrid_count": len(textgrids),
        "speaker_folder_labels_are_identity": False,
    }
    (output_dir / "alignment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Alignment complete: %d TextGrids written to %s", len(textgrids), output_dir)


if __name__ == "__main__":
    main()
