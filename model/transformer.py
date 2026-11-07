"""
Transformer Modules for FastSpeech 2
=====================================

Feed-Forward Transformer (FFT) blocks, Multi-Head Attention, and Positional Encoding.
Preserved from ming024/FastSpeech2 (MIT License).

Architecture:
  - Sinusoidal Positional Encoding
  - Multi-Head Scaled Dot-Product Self-Attention
  - 1D-Convolutional Feed-Forward Network (Conv1D kernel=9, Conv1D kernel=1)
  - Layer Normalization and Dropout

Author: ming024 / Adapted for FastSpeech 2 English-Telugu Project
Licence: MIT
"""

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as described in 'Attention Is All You Need'."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        # Compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        if x.size(1) > self.pe.size(1):
            d_model = self.pe.size(2)
            new_max_len = max(x.size(1) + 1000, self.pe.size(1) * 2)
            position = torch.arange(0, new_max_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float, device=x.device) * -(np.log(10000.0) / d_model)
            )
            pe = torch.zeros(new_max_len, d_model, device=x.device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = pe.unsqueeze(0)
        return x + self.pe[:, : x.size(1), :]


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Args:
            q: [B, n_head, T_q, d_k]
            k: [B, n_head, T_k, d_k]
            v: [B, n_head, T_v, d_v]
            mask: [B, 1, T_q, T_k] or broadcastable
        """
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(torch.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k).transpose(1, 2)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k).transpose(1, 2)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T]

        output, attn = self.attention(q, k, v, mask=mask)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """1D-Convolutional Position-wise Feed-Forward Network as used in FastSpeech 2.
    
    Instead of linear layers, uses 1D convolutions with kernel sizes specified
    in model config (e.g. [9, 1]), giving local context to the feed-forward step.
    """

    def __init__(self, d_in: int, d_hid: int, kernel_size: list, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=kernel_size[0], padding=(kernel_size[0] - 1) // 2
        )
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=kernel_size[1], padding=(kernel_size[1] - 1) // 2
        )
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_in]
        Returns:
            [B, T, d_in]
        """
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(torch.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class FFTBlock(nn.Module):
    """Feed-Forward Transformer (FFT) Block:
    Multi-Head Attention + 1D-Conv Feed-Forward Network.
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        n_head: int,
        d_k: int,
        d_v: int,
        fft_conv1d_kernel: list,
        fft_conv1d_padding: list = None,
        dropout: float = 0.1,
    ):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size=fft_conv1d_kernel, dropout=dropout
        )

    def forward(
        self,
        enc_input: torch.Tensor,
        non_pad_mask: torch.Tensor = None,
        slf_attn_mask: torch.Tensor = None,
    ):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

