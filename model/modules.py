"""
Variance Adaptor and Predictor Modules for FastSpeech 2
========================================================

Implements:
  - VariancePredictor (2-layer 1D Conv + LayerNorm + Dropout + Linear)
  - LengthRegulator (repeats encoder hidden states per duration)
  - VarianceAdaptor (combines Duration, Pitch, Energy predictors)

Preserved from ming024/FastSpeech2 (MIT License).

Author: ming024 / Adapted for FastSpeech 2 English-Telugu Project
Licence: MIT
"""

import os
import torch
import torch.nn as nn
import numpy as np


def get_mask_from_lengths(lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """Create boolean padding mask from sequence lengths.
    
    Args:
        lengths: [B] tensor of sequence lengths
        max_len: max sequence length (optional)
        
    Returns:
        [B, max_len] boolean mask (True where padded, False where valid)
    """
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return ~mask  # True = padded (standard in PyTorch attention)


class VariancePredictor(nn.Module):
    """2-layer 1D Convolutional predictor for duration, pitch, and energy.
    
    Structure:
      Input [B, T, d_model]
        → Conv1D(d_model → filter_size, kernel_size) → ReLU → LayerNorm → Dropout
        → Conv1D(filter_size → filter_size, kernel_size) → ReLU → LayerNorm → Dropout
        → Linear(filter_size → 1)
        → Output [B, T]
    """

    def __init__(self, model_config: dict):
        super(VariancePredictor, self).__init__()

        d_model = model_config["transformer"]["encoder_hidden"]
        filter_size = model_config["variance_predictor"]["filter_size"]
        kernel_size = model_config["variance_predictor"]["kernel_size"]
        dropout = model_config["variance_predictor"]["dropout"]

        self.conv_1 = nn.Sequential(
            nn.Conv1d(
                d_model, filter_size, kernel_size=kernel_size, padding=(kernel_size - 1) // 2
            ),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(
                filter_size, filter_size, kernel_size=kernel_size, padding=(kernel_size - 1) // 2
            ),
            nn.ReLU(),
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout),
        )
        self.linear = nn.Linear(filter_size, 1)

    def forward(self, encoder_output: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            encoder_output: [B, T, d_model]
            mask: [B, T] boolean mask (True = padded)
        Returns:
            [B, T] predictions (log durations, normalized pitch, or normalized energy)
        """
        out = encoder_output.contiguous().transpose(1, 2)
        out = self.conv_1[0](out)
        out = self.conv_1[1](out)
        out = out.transpose(1, 2)
        out = self.conv_1[2](out)
        out = self.conv_1[3](out)

        out = out.transpose(1, 2)
        out = self.conv_2[0](out)
        out = self.conv_2[1](out)
        out = out.transpose(1, 2)
        out = self.conv_2[2](out)
        out = self.conv_2[3](out)

        out = self.linear(out).squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class LengthRegulator(nn.Module):
    """Length Regulator: repeats encoder hidden states according to durations.
    
    During training: uses ground-truth durations from alignment.
    During inference: uses predicted durations from DurationPredictor.
    """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x: torch.Tensor, duration: torch.Tensor, max_len: int = None):
        """Expand x according to duration tensor.
        
        Args:
            x: [B, T_phone, d_model]
            duration: [B, T_phone] integers (number of frames per phone)
            max_len: optional cap on mel frames
            
        Returns:
            output: [B, T_mel, d_model]
            mel_len: [B] sequence lengths
        """
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.tensor(mel_len, dtype=torch.long, device=x.device)

    def expand(self, batch: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        out = list()
        for vec, d in zip(batch, predicted):
            d_val = max(0, int(d.item()))
            if d_val > 0:
                out.append(vec.expand(d_val, -1))
        if len(out) == 0:
            return torch.zeros(1, batch.shape[-1], device=batch.device)
        return torch.cat(out, 0)

    def forward(
        self,
        x: torch.Tensor,
        duration: torch.Tensor,
        max_len: int = None,
    ):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


def pad(tensors, max_len=None):
    """Pad a list of [T_i, D] tensors to [B, max_len, D]."""
    if max_len is None:
        max_len = max(t.size(0) for t in tensors)
    d = tensors[0].size(-1)
    batch = torch.zeros(len(tensors), max_len, d, device=tensors[0].device)
    for i, t in enumerate(tensors):
        length = min(t.size(0), max_len)
        batch[i, :length] = t[:length]
    return batch


class VarianceAdaptor(nn.Module):
    """FastSpeech 2 Variance Adaptor.
    
    Contains:
      - DurationPredictor + LengthRegulator
      - PitchPredictor + Pitch Embedding
      - EnergyPredictor + Energy Embedding
    """

    def __init__(self, preprocess_config: dict, model_config: dict):
        super(VarianceAdaptor, self).__init__()

        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        d_model = model_config["transformer"]["encoder_hidden"]
        n_bins = model_config["variance_embedding"]["n_bins"]

        # Embeddings for quantized pitch and energy
        self.pitch_bins = nn.Parameter(
            torch.linspace(0.0, 1.0, n_bins - 1), requires_grad=False
        )
        self.energy_bins = nn.Parameter(
            torch.linspace(0.0, 1.0, n_bins - 1), requires_grad=False
        )
        self.pitch_embedding = nn.Embedding(n_bins, d_model)
        self.energy_embedding = nn.Embedding(n_bins, d_model)

    def get_pitch_embedding(self, x, target, mask, control=1.0):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control=1.0):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor,
        mel_mask: torch.Tensor = None,
        max_len: int = None,
        pitch_target: torch.Tensor = None,
        energy_target: torch.Tensor = None,
        duration_target: torch.Tensor = None,
        p_control: float = 1.0,
        e_control: float = 1.0,
        d_control: float = 1.0,
    ):
        """
        Args:
            x: [B, T_phone, d_model] encoder hidden states
            src_mask: [B, T_phone] padding mask
            mel_mask: [B, T_mel] padding mask (training only)
            duration_target: [B, T_phone] ground truth durations (training only)
            pitch_target: [B, T_mel] ground truth pitch (training only)
            energy_target: [B, T_mel] ground truth energy (training only)
            *_control: prosody scaling factors (inference only)
        """
        # Duration prediction (in log domain)
        log_duration_prediction = self.duration_predictor(x, src_mask)

        if duration_target is not None:
            # Training: use ground truth duration
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            # Inference: use predicted duration (exp of log prediction)
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        # Pitch prediction and embedding
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            x, pitch_target, mel_mask, p_control
        )
        x = x + pitch_embedding

        # Energy prediction and embedding
        energy_prediction, energy_embedding = self.get_energy_embedding(
            x, energy_target, mel_mask, e_control
        )
        x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )
