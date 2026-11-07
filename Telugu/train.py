"""
Telugu FastSpeech 2 — Training Script
=======================================

Training entry point for the Telugu FastSpeech 2 model on IndicTTS Telugu.
Identical structure to English/train.py with Telugu-specific adaptations:
  - Telugu grapheme vocabulary (not ARPABET phonemes)
  - Telugu text cleaner registered in model/text module

Usage:
    python train.py \\
        -p config/preprocess.yaml \\
        -m config/model.yaml \\
        -t config/train.yaml \\
        [--restore_step 50000]

Author: FastSpeech 2 Telugu TTS Project
Licence: MIT (original architecture from ming024/FastSpeech2)
"""

import os
import sys
import argparse
import logging
import yaml
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.fastspeech2 import FastSpeech2
from model.loss import FastSpeech2Loss
from model.dataset import FastSpeech2Dataset, collate_fn
from model.optimizer import ScheduledOptim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Telugu FastSpeech 2 on IndicTTS")
    parser.add_argument("-p", "--preprocess_config", type=str,
                        default="config/preprocess.yaml")
    parser.add_argument("-m", "--model_config", type=str,
                        default="config/model.yaml")
    parser.add_argument("-t", "--train_config", type=str,
                        default="config/train.yaml")
    parser.add_argument("--restore_step", type=int, default=0)
    return parser.parse_args()


def load_configs(args):
    def _load(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return _load(args.preprocess_config), _load(args.model_config), _load(args.train_config)


def get_vocab_size(preprocess_cfg: dict) -> int:
    """Load Telugu vocabulary size from JSON file created during preprocessing."""
    vocab_file = preprocess_cfg["preprocessing"]["text"].get("vocab_file")
    if vocab_file and os.path.exists(vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        logger.info(f"Telugu vocab loaded: {len(vocab)} tokens")
        return len(vocab)

    # Fallback: build vocabulary from the Telugu Unicode character set
    logger.warning("Vocab file not found, using default Telugu character set size (120)")
    return 120  # Conservative estimate; actual vocab computed during preprocessing


def build_model(preprocess_cfg, model_cfg):
    vocab_size = get_vocab_size(preprocess_cfg)
    model = FastSpeech2(
        preprocess_config=preprocess_cfg,
        model_config=model_cfg,
        vocab_size=vocab_size,
    )
    return model


def load_checkpoint(model, optimizer, ckpt_path, restore_step):
    ckpt_file = os.path.join(ckpt_path, f"{restore_step}.pth.tar")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
    logger.info(f"Restoring Telugu checkpoint from step {restore_step}")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return model, optimizer


def save_checkpoint(model, optimizer, step, ckpt_path):
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_file = os.path.join(ckpt_path, f"{step}.pth.tar")
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer._optimizer.state_dict()}, ckpt_file)
    logger.info(f"Saved Telugu checkpoint: {ckpt_file}")

    # Keep last 3
    all_ckpts = sorted(
        [f for f in os.listdir(ckpt_path) if f.endswith(".pth.tar")],
        key=lambda x: int(x.replace(".pth.tar", ""))
    )
    for old in all_ckpts[:-3]:
        os.remove(os.path.join(ckpt_path, old))


def evaluate(model, val_loader, loss_fn, device, step, writer):
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            output = model(*batch[2:])
            losses = loss_fn(batch, output)
            total_loss += losses[0].item()
            n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    logger.info(f"[Val-Telugu] Step {step} | Loss: {avg_loss:.4f}")
    if writer:
        writer.add_scalar("Loss/val", avg_loss, step)
    model.train()
    return avg_loss


def train(args):
    preprocess_cfg, model_cfg, train_cfg = load_configs(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    ckpt_path = train_cfg["path"]["ckpt_path"]
    log_path = train_cfg["path"]["log_path"]
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    batch_size = train_cfg["optimizer"]["batch_size"]
    total_step = train_cfg["step"]["total_step"]
    log_step = train_cfg["step"]["log_step"]
    val_step = train_cfg["step"]["val_step"]
    save_step = train_cfg["step"]["save_step"]
    grad_clip = train_cfg["optimizer"]["grad_clip_thresh"]

    logger.info("Loading Telugu datasets...")
    train_dataset = FastSpeech2Dataset(
        filename="train.txt",
        preprocess_config=preprocess_cfg,
        train_config=train_cfg,
        language="te",
    )
    val_dataset = FastSpeech2Dataset(
        filename="val.txt",
        preprocess_config=preprocess_cfg,
        train_config=train_cfg,
        language="te",
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    model = build_model(preprocess_cfg, model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Telugu model parameters: {n_params:,}")

    optimizer = ScheduledOptim(
        model=model,
        train_config=train_cfg,
        model_config=model_cfg,
        current_step=args.restore_step,
    )

    loss_fn = FastSpeech2Loss(preprocess_cfg, model_cfg)

    if args.restore_step > 0:
        model, optimizer = load_checkpoint(model, optimizer, ckpt_path, args.restore_step)

    writer = SummaryWriter(log_path)
    model.train()

    step = args.restore_step
    train_losses, val_losses = [], []
    train_iter = iter(train_loader)

    outer_bar = tqdm(total=total_step, desc="Telugu Training", position=0)
    outer_bar.update(step)

    while step < total_step:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = [x.to(device) if torch.is_tensor(x) else x for x in batch]
        output = model(*batch[2:])
        losses = loss_fn(batch, output)
        total_loss_val = losses[0]

        total_loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step_and_update_lr()
        optimizer.zero_grad()

        step += 1
        outer_bar.update(1)

        if step % log_step == 0:
            loss_item = total_loss_val.item()
            train_losses.append((step, loss_item))
            logger.info(
                f"Step {step}/{total_step} | Loss: {loss_item:.4f} | "
                f"Mel: {losses[1].item():.4f} | Dur: {losses[3].item():.4f} | "
                f"Pitch: {losses[4].item():.4f} | Energy: {losses[5].item():.4f}"
            )
            writer.add_scalar("Loss/train", loss_item, step)

        if step % val_step == 0:
            val_loss = evaluate(model, val_loader, loss_fn, device, step, writer)
            val_losses.append((step, val_loss))

        if step % save_step == 0:
            save_checkpoint(model, optimizer, step, ckpt_path)
            np.savez(
                os.path.join(ckpt_path, "loss_history.npz"),
                train_steps=np.array([x[0] for x in train_losses]),
                train_losses=np.array([x[1] for x in train_losses]),
                val_steps=np.array([x[0] for x in val_losses]),
                val_losses=np.array([x[1] for x in val_losses]),
            )

    outer_bar.close()
    save_checkpoint(model, optimizer, step, ckpt_path)
    np.savez(
        os.path.join(ckpt_path, "loss_history.npz"),
        train_steps=np.array([x[0] for x in train_losses]),
        train_losses=np.array([x[1] for x in train_losses]),
        val_steps=np.array([x[0] for x in val_losses]),
        val_losses=np.array([x[1] for x in val_losses]),
    )
    logger.info("Telugu training complete!")
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)

