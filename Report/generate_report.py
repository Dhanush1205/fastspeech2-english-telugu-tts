"""Generate an evidence-based FastSpeech 2 technical report.

The report intentionally distinguishes implementation claims from completed
experiments. Dataset durations, checkpoints, losses, generated audio, and plots
are reported only when corresponding files and metadata are present locally.
No placeholder observations or synthetic results are inserted.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)


ROOT = Path(".")
LANGUAGES = {
    "English": {
        "splits": ROOT / "English/data_splits",
        "checkpoints": ROOT / "English/checkpoints",
        "audio": ROOT / "Results/English/test_generated_audios",
        "plots": ROOT / "Results/English/variance_adaptor_plots",
        "loss": ROOT / "English/checkpoints/loss_history.npz",
        "sentences": ["E1", "E2a", "E2b", "E3", "E4"],
    },
    "Telugu": {
        "splits": ROOT / "Telugu/data_splits",
        "checkpoints": ROOT / "Telugu/checkpoints",
        "audio": ROOT / "Results/Telugu/test_generated_audios",
        "plots": ROOT / "Results/Telugu/variance_adaptor_plots",
        "loss": ROOT / "Telugu/checkpoints/loss_history.npz",
        "sentences": ["T1", "T2a", "T2b", "T3", "T4"],
    },
}


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _manifest_stats(split_dir: Path):
    train_path = split_dir / "train.txt"
    val_path = split_dir / "val.txt"
    if not train_path.is_file() or not val_path.is_file():
        return None

    def ids(path):
        result = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result.add(line.split("|", 1)[0].strip())
        return result

    train_ids = ids(train_path)
    val_ids = ids(val_path)
    return {
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "utterance_overlap": len(train_ids & val_ids),
    }


def _language_evidence(language: str, config: dict):
    meta = _read_json(config["splits"] / "split_metadata.json")
    manifests = _manifest_stats(config["splits"])
    train_hours = meta.get("train_duration_hours") if meta else None
    val_hours = meta.get("val_duration_hours") if meta else None
    full_targets_met = bool(
        meta
        and meta.get("mode") == "full_dataset"
        and isinstance(train_hours, (int, float))
        and isinstance(val_hours, (int, float))
        and train_hours >= 5.0
        and val_hours >= 1.0
        and manifests
        and manifests["utterance_overlap"] == 0
    )

    ckpts = []
    if config["checkpoints"].is_dir():
        ckpts = sorted(
            p.name for p in config["checkpoints"].glob("*")
            if p.is_file() and p.suffix in {".pt", ".pth"}
            or p.is_file() and p.name.endswith(".pth.tar")
        )

    generated = {
        sid: (config["audio"] / f"{sid}.wav").is_file()
        for sid in config["sentences"]
    }
    predictions = {
        sid: (config["audio"] / f"{sid}_predictions.npz").is_file()
        for sid in config["sentences"]
    }
    plot_files = list(config["plots"].rglob("*.png")) if config["plots"].is_dir() else []

    return {
        "metadata": meta,
        "manifests": manifests,
        "full_targets_met": full_targets_met,
        "train_hours": train_hours,
        "val_hours": val_hours,
        "checkpoints": ckpts,
        "loss_log": config["loss"].is_file(),
        "generated": generated,
        "predictions": predictions,
        "plot_count": len(plot_files),
    }


def _status(value: bool, yes="Verified", no="Pending"):
    return yes if value else no


def build_technical_report(pdf_path="Report/Technical_Report.pdf"):
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    evidence = {
        language: _language_evidence(language, config)
        for language, config in LANGUAGES.items()
    }

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="DocTitle", parent=styles["Title"], fontSize=19,
        leading=23, alignment=TA_CENTER, textColor=colors.HexColor("#17365D"),
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="Section", parent=styles["Heading2"], fontSize=13,
        leading=16, textColor=colors.HexColor("#17365D"),
        spaceBefore=12, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11
    ))

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=letter, rightMargin=0.6*inch,
        leftMargin=0.6*inch, topMargin=0.55*inch, bottomMargin=0.55*inch
    )
    story = []

    def para(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))

    story.append(Paragraph(
        "FastSpeech 2 for English and Telugu: Implementation and Experiment Report",
        styles["DocTitle"]
    ))
    para(f"Evidence snapshot generated on {date.today().isoformat()}.", "Small")
    para(
        "<b>Reporting rule:</b> This document reports only facts supported by "
        "repository files, split metadata, checkpoints, logs, generated audio, "
        "and plots present at generation time. Missing artifacts are marked "
        "pending; no listening observations or training outcomes are invented."
    )

    story.append(Paragraph("1. Objective and system design", styles["Section"]))
    para(
        "The project implements separate FastSpeech 2 pipelines for English "
        "and Telugu. The shared model uses a text encoder, variance adaptor "
        "(duration, pitch and energy predictors), mel decoder and PostNet. "
        "English text is represented with the repository's ARPABET/G2P path; "
        "Telugu text is represented with Unicode-normalized grapheme tokens. "
        "The two languages use separate preprocessing configurations, "
        "vocabularies, training configurations and checkpoints."
    )
    para(
        "Audio preparation targets 22,050 Hz mono WAV files. Acoustic features "
        "include mel spectrograms, pitch, energy and token/frame durations. "
        "Duration supervision depends on successful forced alignment; the "
        "selected alignment backend and its actual outputs must be recorded "
        "from the run rather than inferred from configuration alone."
    )

    story.append(Paragraph("2. Dataset preparation and split evidence", styles["Section"]))
    rows = [[
        "Language", "Dataset", "Train hours", "Validation hours",
        "Train / Val items", "Utterance overlap", "Speaker split"
    ]]
    for language, ev in evidence.items():
        meta = ev["metadata"] or {}
        manifests = ev["manifests"] or {}
        speaker_status = "Not verified"
        if meta.get("speaker_disjoint") is True:
            speaker_status = "Verified"
        elif meta.get("speaker_identity_available") is False:
            speaker_status = "Unavailable in source schema"
        elif meta.get("split_strategy") == "demographic_group_disjoint_sequential":
            speaker_status = "Demographic proxy only"
        rows.append([
            language,
            "AI4Bharat Svarah" if language == "English" else "SPRINGLab IndicTTS Telugu",
            f'{ev["train_hours"]:.3f}' if isinstance(ev["train_hours"], (int, float)) else "Not recorded",
            f'{ev["val_hours"]:.3f}' if isinstance(ev["val_hours"], (int, float)) else "Not recorded",
            f'{manifests.get("train_count", 0)} / {manifests.get("val_count", 0)}' if manifests else "No manifests",
            str(manifests.get("utterance_overlap", "Not checked")) if manifests else "Not checked",
            speaker_status,
        ])
    table = Table(rows, repeatRows=1, colWidths=[55, 95, 53, 58, 62, 63, 95])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#B8C2CC")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F3F6F9")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 6))
    for language, ev in evidence.items():
        status = "The 5h/1h full-data targets are met." if ev["full_targets_met"] else (
            "The full 5h/1h targets are not verified from the current metadata. "
            "Smoke-test subsets do not count as the assignment's full dataset."
        )
        para(f"<b>{language}:</b> {status}", "Small")
    para(
        "Svarah's public schema provides demographic fields but no verified "
        "unique speaker identifier in the observed records. Consequently, "
        "demographic composite groups must not be described as ground-truth "
        "speaker IDs. The observed IndicTTS schema contains audio, text and "
        "gender, but no stable speaker ID; its split can be verified by "
        "utterance ID only."
    )

    story.append(Paragraph("3. Experimental execution status", styles["Section"]))
    status_rows = [["Evidence", "English", "Telugu"]]
    for label, key in [
        ("Full 5h / 1h dataset target", "full_targets_met"),
        ("Checkpoint file present", "checkpoints"),
        ("Genuine loss history present", "loss_log"),
        ("All five assignment WAVs present", "generated"),
        ("All five prediction NPZ files present", "predictions"),
    ]:
        vals = []
        for language in ("English", "Telugu"):
            value = evidence[language][key]
            if key in ("generated", "predictions"):
                ok = all(value.values())
            elif key == "checkpoints":
                ok = bool(value)
            else:
                ok = bool(value)
            vals.append(_status(ok))
        status_rows.append([label] + vals)
    status_rows.append([
        "Variance-adaptor PNG count",
        str(evidence["English"]["plot_count"]),
        str(evidence["Telugu"]["plot_count"]),
    ])
    status_table = Table(status_rows, colWidths=[245, 125, 125], repeatRows=1)
    status_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#B8C2CC")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(status_table)
    para(
        "No claims about convergence, speech quality, pronunciation accuracy, "
        "or native-speaker preference are made unless there are actual model "
        "checkpoints, loss histories, generated waveforms, and documented "
        "human evaluation observations."
    )

    story.append(Paragraph("4. Required evaluation and analysis", styles["Section"]))
    para(
        "The assignment requires five English and five Telugu test utterances, "
        "with duration, pitch and energy prediction plots for every utterance. "
        "This corresponds to 10 WAV files, 10 genuine prediction files and "
        "30 variance-adaptor plots. The file-presence checks above describe "
        "what exists locally at report-generation time; a file's presence "
        "alone does not establish perceptual quality."
    )
    para(
        "Listening-based pronunciation analysis should be completed by the "
        "researcher and, ideally, native Telugu listeners. Record sentence ID, "
        "pronunciation issues, intelligibility and any unusual prosody. "
        "Do not infer these observations from plots alone."
    )

    story.append(Paragraph("5. Limitations and reproducibility", styles["Section"]))
    para(
        "Known data limitation: the inspected IndicTTS Telugu metadata does "
        "not provide a stable speaker identifier. The Svarah grouping used "
        "for separation is a demographic proxy unless the source provides "
        "additional speaker-level identifiers. Report the distinction "
        "explicitly when describing the evaluation protocol."
    )
    para(
        "Reproduction requires the repository revision, Python and package "
        "versions, dataset revision/split, random seed, exact manifest files, "
        "alignment tool and model version, acoustic feature settings, training "
        "configuration, checkpoint step, and inference command. Preserve "
        "these in experiment metadata alongside each run."
    )

    story.append(Paragraph("6. Current completion statement", styles["Section"]))
    all_full = all(ev["full_targets_met"] for ev in evidence.values())
    all_ckpt = all(bool(ev["checkpoints"]) for ev in evidence.values())
    all_audio = all(all(ev["generated"].values()) for ev in evidence.values())
    all_plots = sum(ev["plot_count"] for ev in evidence.values()) >= 30
    if all_full and all_ckpt and all_audio and all_plots:
        para(
            "The local artifact checks detect full-duration dataset metadata, "
            "checkpoints, all required WAV outputs and at least 30 variance "
            "plots. Inspect the experiment logs and listen to the generated "
            "audio before treating the experimental evaluation as complete."
        )
    else:
        para(
            "<b>Experimental work is incomplete.</b> One or more of the full "
            "dataset, checkpoint, generated-audio or plot requirements are "
            "not present in the current workspace. This report is a truthful "
            "implementation/status report, not a claim that model training "
            "or final synthesis has completed."
        )

    story.append(Spacer(1, 12))
    para(
        "References: Ren et al., FastSpeech 2 (2020), arXiv:2006.04558; "
        "the original FastSpeech 2 reference implementation by ming024; "
        "AI4Bharat Svarah; and SPRINGLab IndicTTS Telugu.",
        "Small"
    )

    doc.build(story)
    print(f"Evidence-based technical report created: {pdf_path}")


if __name__ == "__main__":
    build_technical_report()
