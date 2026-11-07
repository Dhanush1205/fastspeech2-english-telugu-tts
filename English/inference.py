"""
English FastSpeech 2 — Inference Pipeline
===========================================

Synthesizes speech from English text using trained FastSpeech 2 and vocoder.
Generates:
  1. Mel-spectrograms
  2. Synthesized audio (WAV) using HiFi-GAN
  3. Predicted duration, pitch, and energy features for variance adaptor analysis

Command-line usage:
    python inference.py \\
        --checkpoint checkpoints/english/200000.pth.tar \\
        --config config/model.yaml \\
        --output_dir ../Results/English/test_generated_audios

Licence: MIT
"""

import os
import sys
import json
import argparse
import logging
import yaml
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.fastspeech2 import FastSpeech2
from English.text_processing import EnglishPhonemizer, build_english_phoneme_vocabulary, ENGLISH_TEST_SENTENCES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    parser = argparse.ArgumentParser(description="English FastSpeech 2 Inference")
    parser.add_argument("-p", "--preprocess_config", type=str, default=str(script_dir / "config" / "preprocess.yaml"))
    parser.add_argument("-m", "--model_config", type=str, default=str(script_dir / "config" / "model.yaml"))
    parser.add_argument("-t", "--train_config", type=str, default=str(script_dir / "config" / "train.yaml"))
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pth.tar)")
    parser.add_argument("--output_dir", type=str, default=str(root_dir / "Results" / "English" / "test_generated_audios"))
    parser.add_argument("--stats_dir", type=str, default=str(root_dir / "data" / "preprocessed" / "english"))
    parser.add_argument("--d_control", type=float, default=1.0)
    parser.add_argument("--p_control", type=float, default=1.0)
    parser.add_argument("--e_control", type=float, default=1.0)
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint_arg: Optional[str], checkpoints_dir: Path) -> str:
    if checkpoint_arg and os.path.exists(checkpoint_arg):
        return checkpoint_arg
    ckpts = [p for p in checkpoints_dir.glob("*.pth.tar")] + [p for p in checkpoints_dir.glob("*.pth")]
    if ckpts:
        latest = max(ckpts, key=os.path.getmtime)
        return str(latest)
    if checkpoint_arg:
        return checkpoint_arg
    return str(checkpoints_dir / "checkpoint_final.pth.tar")


def load_model(preprocess_cfg, model_cfg, checkpoint_path, device):
    """Loads FastSpeech 2 model from a verified trained checkpoint.
    
    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        ValueError: If checkpoint is invalid or corrupt.
    """
    vocab = build_english_phoneme_vocabulary()
    vocab_size = len(vocab)
    model = FastSpeech2(preprocess_cfg, model_cfg, vocab_size=vocab_size).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            f"A valid trained checkpoint is required for inference. "
            f"Please train the model using English/train.py or download a verified checkpoint."
        )

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint format at {checkpoint_path}: expected dictionary.")

    state_dict = ckpt.get("model", ckpt)
    
    # Verify expected key components exist in state dict
    required_keys = ["encoder.src_word_emb.weight", "decoder.mel_linear.weight"]
    for k in required_keys:
        if k not in state_dict:
            raise ValueError(
                f"Corrupt checkpoint at {checkpoint_path}: missing required weight key '{k}'."
            )

    model.load_state_dict(state_dict)
    model.eval()
    return model, vocab


def synthesize_sample(
    model,
    text: str,
    vocab: dict,
    phonemizer: EnglishPhonemizer,
    stats: dict,
    device: torch.device,
    d_control: float = 1.0,
    p_control: float = 1.0,
    e_control: float = 1.0,
):
    """Generates predicted mel, pitch, energy, durations from text."""
    sequence = np.array(phonemizer.text_to_sequence(text, vocab))
    tokens = phonemizer.text_to_phonemes(text)
    
    text_tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
    src_len = torch.tensor([len(sequence)], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(
            speakers=None,
            texts=text_tensor,
            src_lens=src_len,
            max_src_len=len(sequence),
            p_control=p_control,
            e_control=e_control,
            d_control=d_control,
        )

    (
        mel_before,
        mel_after,
        p_pred,
        e_pred,
        log_d_pred,
        d_rounded,
        src_mask,
        mel_mask,
        _,
        mel_len,
    ) = output

    mel_len = int(mel_len[0].item())
    mel_out = mel_after[0, :mel_len].cpu().numpy()

    # De-normalize pitch & energy
    p_pred = p_pred[0, :mel_len].cpu().numpy() * stats.get("p_std", 1.0) + stats.get("p_mean", 0.0)
    e_pred = e_pred[0, :mel_len].cpu().numpy() * stats.get("e_std", 1.0) + stats.get("e_mean", 0.0)
    d_pred = d_rounded[0, :len(sequence)].cpu().numpy()

    return {
        "mel": mel_out,
        "pitch": p_pred,
        "energy": e_pred,
        "duration": d_pred,
        "tokens": tokens,
        "sequence": sequence,
    }


def invert_mel_to_audio(
    mel: np.ndarray,
    vocoder=None,
    device: str = "cpu",
    sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    allow_griffin_lim: bool = True,
) -> np.ndarray:
    """Invert predicted mel-spectrogram to audio waveform.
    
    Uses HiFi-GAN neural vocoder if available, with Griffin-Lim phase reconstruction
    as algorithmic fallback. Never generates synthetic tone / sine-wave fallbacks.
    
    Raises:
        RuntimeError: If waveform synthesis fails.
    """
    if vocoder is not None:
        try:
            mel_tensor = torch.from_numpy(mel.T).unsqueeze(0).to(device)
            with torch.no_grad():
                wav = vocoder(mel_tensor).squeeze().cpu().numpy()
            return wav.astype(np.float32)
        except Exception as e:
            logger.warning(f"HiFi-GAN vocoder inversion failed: {e}. Attempting Griffin-Lim fallback.")

    if not allow_griffin_lim:
        raise RuntimeError("Waveform inversion failed: no vocoder provided and Griffin-Lim fallback is disallowed.")

    try:
        import librosa
        linear_spec = librosa.feature.inverse.mel_to_stft(
            np.exp(mel.T),
            sr=sr,
            n_fft=n_fft,
            power=1.0,
        )
        wav = librosa.griffinlim(linear_spec, hop_length=hop_length, win_length=n_fft)
        return wav.astype(np.float32)
    except ImportError as e:
        raise RuntimeError(
            f"Waveform inversion failed: 'librosa' is not installed and no functional neural vocoder was provided. "
            f"Install librosa (`pip install librosa`) or provide a valid HiFi-GAN checkpoint to generate audio."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Waveform inversion failed: {e}") from e


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running English Inference on: {device}")

    # Load configs
    with open(args.preprocess_config) as f:
        prep_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        mod_cfg = yaml.safe_load(f)

    # Load normalization stats
    stats = {}
    p_mean_f = Path(args.stats_dir) / "pitch_mean.npy"
    p_std_f = Path(args.stats_dir) / "pitch_std.npy"
    e_mean_f = Path(args.stats_dir) / "energy_mean.npy"
    e_std_f = Path(args.stats_dir) / "energy_std.npy"
    if p_mean_f.exists(): stats["p_mean"] = float(np.load(p_mean_f))
    if p_std_f.exists(): stats["p_std"] = float(np.load(p_std_f))
    if e_mean_f.exists(): stats["e_mean"] = float(np.load(e_mean_f))
    if e_std_f.exists(): stats["e_std"] = float(np.load(e_std_f))

    # Load model
    ckpt_file = resolve_checkpoint_path(args.checkpoint, Path(__file__).parent / "checkpoints")
    model, vocab = load_model(prep_cfg, mod_cfg, ckpt_file, device)
    phonemizer = EnglishPhonemizer()

    # Load vocoder if available
    vocoder = None
    try:
        from hifigan.get_vocoder import get_vocoder
        vocoder = get_vocoder(device=device)
        logger.info("Loaded HiFi-GAN vocoder architecture")
    except Exception as e:
        logger.info(f"HiFi-GAN vocoder not loaded ({e}), using algorithmic inverter")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Results container for plotting
    predictions_dump = {}

    for sent_id, sentence in ENGLISH_TEST_SENTENCES.items():
        logger.info(f"Synthesizing [{sent_id}]: {sentence}")
        res = synthesize_sample(
            model=model,
            text=sentence,
            vocab=vocab,
            phonemizer=phonemizer,
            stats=stats,
            device=device,
            d_control=args.d_control,
            p_control=args.p_control,
            e_control=args.e_control,
        )

        # Invert to WAV
        wav = invert_mel_to_audio(res["mel"], vocoder=vocoder, device=device, sr=22050, hop_length=256)
        wav_path = out_dir / f"{sent_id}.wav"
        sf.write(str(wav_path), wav, 22050)
        logger.info(f"Saved audio: {wav_path} (Duration: {len(wav)/22050:.2f}s)")

        # Save predictions for variance adaptor plots
        np.savez(
            out_dir / f"{sent_id}_predictions.npz",
            pitch=res["pitch"],
            energy=res["energy"],
            duration=res["duration"],
            tokens=np.array(res["tokens"]),
        )

    logger.info("English synthesis complete!")


if __name__ == "__main__":
    main()

