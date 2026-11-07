"""
FastSpeech 2 Loss Function
==========================

Implements the multi-task loss objective for FastSpeech 2:
  1. Mel Spectrogram L1 Loss (before PostNet)
  2. Mel Spectrogram L1 Loss (after PostNet)
  3. Duration Loss: MSE in log-domain: MSE(log_pred_dur, log(target_dur + 1))
  4. Pitch Loss: MSE between predicted normalized pitch and target normalized pitch
  5. Energy Loss: MSE between predicted normalized energy and target normalized energy

Reference:
  - FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (Ren et al., 2020)
  - ming024/FastSpeech2/model/loss.py

Licence: MIT
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class FastSpeech2Loss(nn.Module):
    """Computes total multi-task loss for FastSpeech 2."""

    def __init__(self, preprocess_config: dict, model_config: dict):
        super(FastSpeech2Loss, self).__init__()
        self.loss_config = {
            "mel_loss_weight": 1.0,
            "postnet_mel_loss_weight": 1.0,
            "duration_loss_weight": 1.0,
            "pitch_loss_weight": 1.0,
            "energy_loss_weight": 1.0,
        }

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(
        self,
        batch: List[torch.Tensor],
        predictions: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                0: ids
                1: raw_texts
                2: speakers
                3: texts
                4: text_lens
                5: max_text_len
                6: mels
                7: mel_lens
                8: max_mel_len
                9: pitches
                10: energies
                11: durations
            predictions:
                0: mel_predictions (before postnet) [B, T_mel, n_mel]
                1: postnet_mel_predictions [B, T_mel, n_mel]
                2: pitch_predictions [B, T_mel]
                3: energy_predictions [B, T_mel]
                4: log_duration_predictions [B, T_src]
                5: duration_rounded [B, T_src]
                6: src_masks [B, T_src]
                7: mel_masks [B, T_mel]
                8: src_lens [B]
                9: mel_lens [B]
        """
        targets = batch
        (
            mel_targets,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = targets[6], targets[9], targets[10], targets[11]

        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions

        # Apply masks: False where valid, True where padding
        # In PyTorch attention masks: True = masked (padded)
        src_valid = ~src_masks
        mel_valid = ~mel_masks

        # 1 & 2. Mel Spectrogram L1 Loss
        mel_targets = mel_targets.masked_select(mel_valid.unsqueeze(-1))
        mel_predictions = mel_predictions.masked_select(mel_valid.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(mel_valid.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        # 3. Duration Loss (MSE in log domain)
        log_duration_targets = torch.log(duration_targets.float() + 1.0)
        log_duration_targets = log_duration_targets.masked_select(src_valid)
        log_duration_predictions = log_duration_predictions.masked_select(src_valid)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        # 4. Pitch Loss
        pitch_targets = pitch_targets.masked_select(mel_valid)
        pitch_predictions = pitch_predictions.masked_select(mel_valid)
        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)

        # 5. Energy Loss
        energy_targets = energy_targets.masked_select(mel_valid)
        energy_predictions = energy_predictions.masked_select(mel_valid)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)

        total_loss = (
            self.loss_config["mel_loss_weight"] * mel_loss
            + self.loss_config["postnet_mel_loss_weight"] * postnet_mel_loss
            + self.loss_config["duration_loss_weight"] * duration_loss
            + self.loss_config["pitch_loss_weight"] * pitch_loss
            + self.loss_config["energy_loss_weight"] * energy_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            duration_loss,
            pitch_loss,
            energy_loss,
        )
