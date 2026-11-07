"""
English FastSpeech 2 — Training Script
========================================

Training entry point for the English FastSpeech 2 model on the Svarah dataset.
Based on ming024/FastSpeech2 train.py with adaptations for Svarah.

Usage:
    python train.py \\
        -p config/preprocess.yaml \\
        -m config/model.yaml \\
        -t config/train.yaml \\
        [--restore_step 50000]

The training loop:
  1. Loads Dataset from preprocessed_path manifests
  2. Runs the FastSpeech2 forward pass
  3. Computes 5-component loss (mel, mel_postnet, duration, pitch, energy)
  4. Backpropagates with gradient clipping (max norm 1.0)
  5. Updates LR via Noam (warmup + anneal) scheduler
  6. Logs every log_step steps
  7. Saves checkpoint every save_step steps
  8. Runs validation every val_step steps

Author: FastSpeech 2 English TTS Project
Licence: MIT (original architecture from ming024/FastSpeech2)
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path so shared model code is importable
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
    parser = argparse.ArgumentParser(description="Train English FastSpeech 2 on Svarah")
    parser.add_argument("-p", "--preprocess_config", type=str,
                        default="config/preprocess.yaml")
    parser.add_argument("-m", "--model_config", type=str,
                        default="config/model.yaml")
    parser.add_argument("-t", "--train_config", type=str,
                        default="config/train.yaml")
    parser.add_argument("--restore_step", type=int, default=0,
                        help="Resume training from this checkpoint step")
    parser.add_argument("--total_step", type=int, default=None,
                        help="Override total_step from train.yaml (e.g. for benchmark runs)")
    parser.add_argument("--save_step", type=int, default=None,
                        help="Override save_step interval")
    parser.add_argument("--eval_step", type=int, default=None,
                        help="Override evaluation interval")
    parser.add_argument("--val_step", type=int, default=None,
                        help="Alias for --eval_step")
    parser.add_argument("--log_step", type=int, default=None,
                        help="Override logging interval")
    parser.add_argument("--drive_save_dir", type=str, default=None,
                        help="Persistent Google Drive directory to mirror checkpoints and logs")
    return parser.parse_args()


def load_configs(args):
    """Load all three configuration files."""
    def _load(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    preprocess_cfg = _load(args.preprocess_config)
    model_cfg = _load(args.model_config)
    train_cfg = _load(args.train_config)
    return preprocess_cfg, model_cfg, train_cfg


def get_vocab_size(preprocess_cfg: dict) -> int:
    """Get vocabulary size from the English phoneme symbol set."""
    # For English: use the fixed ARPABET phoneme vocabulary
    # This matches text/symbols.py in the reference implementation
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from text_processing import build_english_phoneme_vocabulary
    vocab = build_english_phoneme_vocabulary()
    return len(vocab)


def build_model(preprocess_cfg: dict, model_cfg: dict) -> FastSpeech2:
    """Instantiate FastSpeech2 with correct vocabulary size."""
    vocab_size = get_vocab_size(preprocess_cfg)
    logger.info(f"English vocabulary size: {vocab_size}")

    model = FastSpeech2(
        preprocess_config=preprocess_cfg,
        model_config=model_cfg,
        vocab_size=vocab_size,
    )
    return model


def sync_to_drive(src_file: str, drive_dir: Optional[str] = None):
    """Synchronize a checkpoint or log file to persistent Google Drive."""
    if not drive_dir or not os.path.exists(src_file):
        return
    try:
        import shutil
        os.makedirs(drive_dir, exist_ok=True)
        dst_file = os.path.join(drive_dir, os.path.basename(src_file))
        shutil.copyfile(src_file, dst_file)
        logger.info(f"[Drive Sync] Copied {os.path.basename(src_file)} to {drive_dir}")
    except Exception as e:
        logger.warning(f"[Drive Sync] Failed to copy {src_file} to Drive: {e}")


def load_checkpoint(model, optimizer, ckpt_path: str, restore_step: int, drive_save_dir: Optional[str] = None):
    """Load model and optimizer state from checkpoint, with Google Drive fallback."""
    ckpt_file = os.path.join(ckpt_path, f"{restore_step}.pth.tar")
    if not os.path.exists(ckpt_file) and drive_save_dir:
        drive_file = os.path.join(drive_save_dir, f"{restore_step}.pth.tar")
        if os.path.exists(drive_file):
            logger.info(f"Local checkpoint not found at {ckpt_file}. Restoring from Google Drive: {drive_file}")
            os.makedirs(ckpt_path, exist_ok=True)
            import shutil
            shutil.copyfile(drive_file, ckpt_file)
            ckpt_file = drive_file

    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    logger.info(f"Restoring checkpoint from step {restore_step}: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return model, optimizer


def save_checkpoint(model, optimizer, step: int, ckpt_path: str, config_metadata: dict = None, drive_save_dir: Optional[str] = None):
    """Save model and optimizer state with global step and configuration metadata."""
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_file = os.path.join(ckpt_path, f"{step}.pth.tar")
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer._optimizer.state_dict(),
    }
    if config_metadata:
        payload["config"] = config_metadata
    torch.save(payload, ckpt_file)
    logger.info(f"Saved checkpoint: {ckpt_file}")

    # Immediately mirror to Google Drive
    sync_to_drive(ckpt_file, drive_save_dir)

    # Keep only last 3 checkpoints to save disk space
    all_ckpts = sorted(
        [f for f in os.listdir(ckpt_path) if f.endswith(".pth.tar") and f.replace(".pth.tar", "").isdigit()],
        key=lambda x: int(x.replace(".pth.tar", ""))
    )
    for old_ckpt in all_ckpts[:-3]:
        os.remove(os.path.join(ckpt_path, old_ckpt))
        logger.debug(f"Removed old checkpoint: {old_ckpt}")


def evaluate(model, val_loader, loss_fn, device, step, writer):
    """Run one pass over validation set and log losses."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            output = model(*batch[2:])
            losses = loss_fn(batch, output)
            total_loss += losses[0].item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    logger.info(f"[Val] Step {step} | Loss: {avg_loss:.4f}")
    if writer:
        writer.add_scalar("Loss/val", avg_loss, step)

    model.train()
    return avg_loss


def train(args):
    """Main training function."""
    preprocess_cfg, model_cfg, train_cfg = load_configs(args)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Paths
    preprocessed_path = preprocess_cfg["path"]["preprocessed_path"]
    ckpt_path = train_cfg["path"]["ckpt_path"]
    log_path = train_cfg["path"]["log_path"]
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # Training hyperparameters
    batch_size = train_cfg["optimizer"]["batch_size"]
    total_step = args.total_step if args.total_step is not None else train_cfg["step"]["total_step"]
    log_step = args.log_step if args.log_step is not None else train_cfg["step"]["log_step"]
    val_step = args.val_step or args.eval_step if (args.val_step or args.eval_step) is not None else train_cfg["step"]["val_step"]
    save_step = args.save_step if args.save_step is not None else train_cfg["step"]["save_step"]
    grad_clip = train_cfg["optimizer"]["grad_clip_thresh"]
    drive_save_dir = args.drive_save_dir or os.environ.get("DRIVE_SAVE_DIR")
    if drive_save_dir:
        logger.info(f"Google Drive checkpoint synchronization enabled: {drive_save_dir}")

    # Build datasets
    logger.info("Loading datasets...")
    train_dataset = FastSpeech2Dataset(
        filename="train.txt",
        preprocess_config=preprocess_cfg,
        train_config=train_cfg,
        language="en",
    )
    val_dataset = FastSpeech2Dataset(
        filename="val.txt",
        preprocess_config=preprocess_cfg,
        train_config=train_cfg,
        language="en",
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            f"Cannot train with empty datasets: train={len(train_dataset)}, val={len(val_dataset)}. "
            "Verify manifests and preprocessing outputs."
        )
    # Smoke-test datasets may contain fewer items than the configured full-run batch size.
    # Reduce the batch size and keep the final partial batch instead of dropping all samples.
    effective_batch_size = min(batch_size, len(train_dataset))
    logger.info(
        f"Effective batch size: {effective_batch_size} "
        f"(configured={batch_size}, train_samples={len(train_dataset)})"
    )
    workers = 0 if (os.name == "nt" or device.type == "cpu") else 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_dataset)),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0 if (os.name == "nt" or device.type == "cpu") else 2,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Build model
    model = build_model(preprocess_cfg, model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Optimizer (Noam scheduler, as in reference implementation)
    optimizer = ScheduledOptim(
        model=model,
        train_config=train_cfg,
        model_config=model_cfg,
        current_step=args.restore_step,
    )

    # Loss function
    loss_fn = FastSpeech2Loss(preprocess_cfg, model_cfg)

    # Loss tracking for plots
    train_losses = []
    val_losses = []

    # Restore from checkpoint if requested
    if args.restore_step > 0:
        model, optimizer = load_checkpoint(model, optimizer, ckpt_path, args.restore_step, drive_save_dir)
        loss_path = os.path.join(ckpt_path, "loss_history.npz")
        if not os.path.exists(loss_path) and drive_save_dir and os.path.exists(os.path.join(drive_save_dir, "loss_history.npz")):
            loss_path = os.path.join(drive_save_dir, "loss_history.npz")
        if os.path.exists(loss_path):
            try:
                hist = np.load(loss_path)
                train_losses = list(zip(hist["train_steps"].tolist(), hist["train_losses"].tolist()))
                val_losses = list(zip(hist["val_steps"].tolist(), hist["val_losses"].tolist()))
                logger.info(f"Loaded existing loss history with {len(train_losses)} train and {len(val_losses)} val records.")
            except Exception as e:
                logger.warning(f"Could not load previous loss history: {e}")

    logger.info(f"Starting training from step {step} → {total_step}")

    outer_bar = tqdm(total=total_step, desc="Total Training", position=0)
    outer_bar.update(step)

    # Infinite dataloader loop
    train_iter = iter(train_loader)

    while step < total_step:
        # Get next batch, re-initialise iterator when exhausted
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Move to device
        batch = [x.to(device) if torch.is_tensor(x) else x for x in batch]

        # Forward pass
        output = model(*batch[2:])

        # Compute loss
        losses = loss_fn(batch, output)
        total_loss_val = losses[0]

        # Backward pass
        total_loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step_and_update_lr()
        optimizer.zero_grad()

        step += 1
        outer_bar.update(1)

        # Logging
        if step % log_step == 0:
            loss_item = total_loss_val.item()
            mel_loss = losses[1].item()
            dur_loss = losses[3].item()
            pitch_loss = losses[4].item()
            energy_loss = losses[5].item()

            train_losses.append((step, loss_item))

            logger.info(
                f"Step {step}/{total_step} | "
                f"Loss: {loss_item:.4f} | "
                f"Mel: {mel_loss:.4f} | "
                f"Dur: {dur_loss:.4f} | "
                f"Pitch: {pitch_loss:.4f} | "
                f"Energy: {energy_loss:.4f} | "
                f"LR: {optimizer.get_lr():.2e}"
            )

            if writer:
                writer.add_scalar("Loss/train", loss_item, step)
                writer.add_scalar("Loss/mel", mel_loss, step)
                writer.add_scalar("Loss/duration", dur_loss, step)
                writer.add_scalar("Loss/pitch", pitch_loss, step)
                writer.add_scalar("Loss/energy", energy_loss, step)
                writer.add_scalar("LR", optimizer.get_lr(), step)

        # Validation
        if step % val_step == 0:
            val_loss = evaluate(model, val_loader, loss_fn, device, step, writer)
            val_losses.append((step, val_loss))

        # Save checkpoint
        if step % save_step == 0:
            save_checkpoint(model, optimizer, step, ckpt_path, config_metadata={"preprocess": preprocess_cfg, "model": model_cfg, "train": train_cfg}, drive_save_dir=drive_save_dir)

            # Save loss history alongside checkpoint
            loss_path = os.path.join(ckpt_path, "loss_history.npz")
            np.savez(
                loss_path,
                train_steps=np.array([x[0] for x in train_losses]),
                train_losses=np.array([x[1] for x in train_losses]),
                val_steps=np.array([x[0] for x in val_losses]),
                val_losses=np.array([x[1] for x in val_losses]),
            )
            sync_to_drive(loss_path, drive_save_dir)

    outer_bar.close()
    logger.info("Training complete!")

    # Save final checkpoint
    save_checkpoint(model, optimizer, step, ckpt_path, config_metadata={"preprocess": preprocess_cfg, "model": model_cfg, "train": train_cfg}, drive_save_dir=drive_save_dir)
    final_ckpt = os.path.join(ckpt_path, "checkpoint_final.pth.tar")
    import shutil
    shutil.copyfile(os.path.join(ckpt_path, f"{step}.pth.tar"), final_ckpt)
    sync_to_drive(final_ckpt, drive_save_dir)

    # Save final loss history
    loss_path = os.path.join(ckpt_path, "loss_history.npz")
    np.savez(
        loss_path,
        train_steps=np.array([x[0] for x in train_losses]),
        train_losses=np.array([x[1] for x in train_losses]),
        val_steps=np.array([x[0] for x in val_losses]),
        val_losses=np.array([x[1] for x in val_losses]),
    )
    sync_to_drive(loss_path, drive_save_dir)
    logger.info(f"Loss history saved to {loss_path}")

    if writer:
        writer.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)

