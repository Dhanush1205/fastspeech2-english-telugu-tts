"""
Telugu FastSpeech 2 — Inference Pipeline
=========================================

Synthesizes speech from Telugu Unicode text using trained FastSpeech 2.
Preserves assignment test sentences exactly, including spacing and numerals.

Outputs:
  1. Mel-spectrograms
  2. Synthesized audio (WAV)
  3. Predicted duration, pitch, and energy features for variance adaptor analysis

Command-line usage:
    python inference.py \\
        --checkpoint checkpoints/telugu/200000.pth.tar \\
        --config config/model.yaml \\
        --output_dir ../Results/Telugu/test_generated_audios

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
from Telugu.text_processing import TeluguTokenizer, build_telugu_vocabulary, TELUGU_TEST_SENTENCES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    parser = argparse.ArgumentParser(description="Telugu FastSpeech 2 Inference")
    parser.add_argument("-p", "--preprocess_config", type=str, default=str(script_dir / "config" / "preprocess.yaml"))
    parser.add_argument("-m", "--model_config", type=str, default=str(script_dir / "config" / "model.yaml"))
    parser.add_argument("-t", "--train_config", type=str, default=str(script_dir / "config" / "train.yaml"))
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pth.tar)")
    parser.add_argument("--output_dir", type=str, default=str(root_dir / "Results" / "Telugu" / "test_generated_audios"))
    parser.add_argument("--stats_dir", type=str, default=str(root_dir / "data" / "preprocessed" / "telugu"))
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


def load_model(preprocess_cfg, model_cfg, checkpoint_path, device, vocab_path=None):
    """Loads Telugu FastSpeech 2 model from a verified trained checkpoint.
    
    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        ValueError: If checkpoint format is invalid.
    """
    if vocab_path and os.path.exists(vocab_path):
        tokenizer = TeluguTokenizer.from_vocab_file(vocab_path)
    else:
        # Fallback to test sentences vocab
        all_texts = list(TELUGU_TEST_SENTENCES.values())
        vocab = build_telugu_vocabulary(texts=all_texts)
        tokenizer = TeluguTokenizer(vocab)

    vocab_size = tokenizer.vocab_size
    model = FastSpeech2(preprocess_cfg, model_cfg, vocab_size=vocab_size).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            f"A valid trained checkpoint is required for Telugu synthesis. "
            f"Please train the model using Telugu/train.py or download a verified checkpoint."
        )

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint format at {checkpoint_path}: expected dictionary.")

    state_dict = ckpt.get("model", ckpt)
    
    required_keys = ["encoder.src_word_emb.weight", "mel_linear.weight"]
    for k in required_keys:
        if k not in state_dict:
            raise ValueError(
                f"Corrupt checkpoint at {checkpoint_path}: missing required weight key '{k}'."
            )

    state_dict = {k: v for k, v in state_dict.items() if not k.endswith(".pe")}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, tokenizer


def synthesize_sample(
    model,
    text: str,
    tokenizer: TeluguTokenizer,
    stats: dict,
    device: torch.device,
    d_control: float = 1.0,
    p_control: float = 1.0,
    e_control: float = 1.0,
):
    """Generates predicted mel, pitch, energy, durations from Telugu text."""
    sequence = np.array(tokenizer.text_to_sequence(text, add_eos=True, normalize=False))
    tokens = list(text) + ["<eos>"]  # Exact length alignment with sequence

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
    
    Uses HiFi-GAN vocoder if available, with Griffin-Lim phase reconstruction
    as fallback. Never generates synthetic tone / sine-wave fallbacks.
    
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
    logger.info(f"Running Telugu Inference on: {device}")

    with open(args.preprocess_config) as f:
        prep_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        mod_cfg = yaml.safe_load(f)

    stats = {}
    p_mean_f = Path(args.stats_dir) / "pitch_mean.npy"
    p_std_f = Path(args.stats_dir) / "pitch_std.npy"
    e_mean_f = Path(args.stats_dir) / "energy_mean.npy"
    e_std_f = Path(args.stats_dir) / "energy_std.npy"
    if p_mean_f.exists(): stats["p_mean"] = float(np.load(p_mean_f))
    if p_std_f.exists(): stats["p_std"] = float(np.load(p_std_f))
    if e_mean_f.exists(): stats["e_mean"] = float(np.load(e_mean_f))
    if e_std_f.exists(): stats["e_std"] = float(np.load(e_std_f))

    vocab_file = Path(args.stats_dir) / "telugu_vocab.json"
    ckpt_file = resolve_checkpoint_path(args.checkpoint, Path(__file__).parent / "checkpoints")
    model, tokenizer = load_model(prep_cfg, mod_cfg, ckpt_file, device, vocab_file)

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

    for sent_id, sentence in TELUGU_TEST_SENTENCES.items():
        logger.info(f"Synthesizing [{sent_id}]: {sentence}")
        res = synthesize_sample(
            model=model,
            text=sentence,
            tokenizer=tokenizer,
            stats=stats,
            device=device,
            d_control=args.d_control,
            p_control=args.p_control,
            e_control=args.e_control,
        )

        wav = invert_mel_to_audio(res["mel"], vocoder=vocoder, device=device, sr=22050, hop_length=256)
        wav_path = out_dir / f"{sent_id}.wav"
        sf.write(str(wav_path), wav, 22050)
        logger.info(f"Saved audio: {wav_path} (Duration: {len(wav)/22050:.2f}s)")

        np.savez(
            out_dir / f"{sent_id}_predictions.npz",
            pitch=res["pitch"],
            energy=res["energy"],
            duration=res["duration"],
            tokens=np.array(res["tokens"]),
        )

    logger.info("Telugu synthesis complete!")


if __name__ == "__main__":
    main()

