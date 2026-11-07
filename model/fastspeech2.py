"""
FastSpeech 2 Core Architecture
================================

Implements the complete FastSpeech 2 model:
  - Text Encoder (N FFTBlocks)
  - Variance Adaptor (Duration, Pitch, Energy predictors)
  - Mel Decoder (N FFTBlocks)
  - Linear Projection to 80-channel mel frames
  - PostNet (5-layer 1D Conv) for residual detail refinement

Preserved from ming024/FastSpeech2 (MIT License).

Author: ming024 / Adapted for FastSpeech 2 English-Telugu Project
Licence: MIT
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from model.transformer import FFTBlock, PositionalEncoding
from model.modules import VarianceAdaptor, get_mask_from_lengths


class Encoder(nn.Module):
    """Text Encoder: phoneme/grapheme embedding + positional encoding + N FFTBlocks."""

    def __init__(self, model_config: dict, vocab_size: int):
        super(Encoder, self).__init__()

        d_model = model_config["transformer"]["encoder_hidden"]
        n_layers = model_config["transformer"]["encoder_layer"]
        n_heads = model_config["transformer"]["encoder_head"]
        d_k = d_model // n_heads
        d_v = d_model // n_heads
        d_inner = model_config["transformer"]["conv_filter_size"]
        fft_kernel = model_config["transformer"]["conv_kernel_size"]
        dropout = model_config["transformer"]["encoder_dropout"]
        max_seq_len = model_config["transformer"].get("max_seq_len", 1000)

        self.src_word_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_enc = PositionalEncoding(d_model, max_len=max_seq_len)

        self.layer_stack = nn.ModuleList([
            FFTBlock(
                d_model=d_model,
                d_inner=d_inner,
                n_head=n_heads,
                d_k=d_k,
                d_v=d_v,
                fft_conv1d_kernel=fft_kernel,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def forward(self, src_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src_seq: [B, T_src] token indices
            mask: [B, T_src] boolean padding mask
        Returns:
            [B, T_src, d_model]
        """
        batch_size, max_len = src_seq.shape
        non_pad_mask = (~mask).unsqueeze(-1).float()  # [B, T, 1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)  # [B, T, T]

        enc_output = self.src_word_emb(src_seq)
        enc_output = self.position_enc(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
            )

        return enc_output


class Decoder(nn.Module):
    """Mel Decoder: positional encoding + N FFTBlocks + linear projection to mel."""

    def __init__(self, model_config: dict):
        super(Decoder, self).__init__()

        d_model = model_config["transformer"]["decoder_hidden"]
        n_layers = model_config["transformer"]["decoder_layer"]
        n_heads = model_config["transformer"]["decoder_head"]
        d_k = d_model // n_heads
        d_v = d_model // n_heads
        d_inner = model_config["transformer"]["conv_filter_size"]
        fft_kernel = model_config["transformer"]["conv_kernel_size"]
        dropout = model_config["transformer"]["decoder_dropout"]
        max_seq_len = model_config["transformer"].get("max_seq_len", 1000)

        self.position_enc = PositionalEncoding(d_model, max_len=max_seq_len)

        self.layer_stack = nn.ModuleList([
            FFTBlock(
                d_model=d_model,
                d_inner=d_inner,
                n_head=n_heads,
                d_k=d_k,
                d_v=d_v,
                fft_conv1d_kernel=fft_kernel,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def forward(self, enc_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc_seq: [B, T_mel, d_model]
            mask: [B, T_mel] boolean padding mask
        Returns:
            [B, T_mel, d_model]
        """
        batch_size, max_len = enc_seq.shape[:2]
        non_pad_mask = (~mask).unsqueeze(-1).float()
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        dec_output = self.position_enc(enc_seq)

        for dec_layer in self.layer_stack:
            dec_output, _ = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
            )

        return dec_output


class PostNet(nn.Module):
    """5-layer 1D Convolutional PostNet to refine predicted mel-spectrograms."""

    def __init__(self, n_mel_channels: int = 80, postnet_embedding_dim: int = 512,
                 postnet_kernel_size: int = 5, postnet_n_convolutions: int = 5):
        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=(postnet_kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
                nn.Tanh(),
                nn.Dropout(0.5),
            )
        )

        for _ in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=(postnet_kernel_size - 1) // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                    nn.Tanh(),
                    nn.Dropout(0.5),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=(postnet_kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(n_mel_channels),
                nn.Dropout(0.5),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, n_mel] mel-spectrogram
        Returns:
            [B, T, n_mel] residual
        """
        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = conv(x)
        return x.transpose(1, 2)


class FastSpeech2(nn.Module):
    """Complete FastSpeech 2 Model."""

    def __init__(
        self,
        preprocess_config: dict,
        model_config: dict,
        vocab_size: int,
    ):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        # 1. Text Encoder
        self.encoder = Encoder(model_config, vocab_size)

        # 2. Variance Adaptor
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)

        # 3. Mel Decoder
        self.decoder = Decoder(model_config)

        # 4. Mel Linear Projection
        d_model = model_config["transformer"]["decoder_hidden"]
        n_mel = preprocess_config.get("preprocessing", {}).get("audio", {}).get("n_mel_channels", 80)
        if "stft" in preprocess_config.get("preprocessing", {}):
            n_mel = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.mel_linear = nn.Linear(d_model, n_mel)

        # 5. PostNet
        self.postnet = PostNet(n_mel_channels=n_mel)

        # Multi-speaker embedding (optional)
        self.multi_speaker = model_config.get("multi_speaker", False)
        if self.multi_speaker:
            n_speakers = model_config.get("n_speakers", 2)
            self.speaker_emb = nn.Embedding(n_speakers, d_model)

    def forward(
        self,
        speakers: Optional[torch.Tensor],
        texts: torch.Tensor,
        src_lens: torch.Tensor,
        max_src_len: int,
        mels: Optional[torch.Tensor] = None,
        mel_lens: Optional[torch.Tensor] = None,
        max_mel_len: Optional[int] = None,
        p_targets: Optional[torch.Tensor] = None,
        e_targets: Optional[torch.Tensor] = None,
        d_targets: Optional[torch.Tensor] = None,
        p_control: float = 1.0,
        e_control: float = 1.0,
        d_control: float = 1.0,
    ) -> Tuple:
        """Forward pass.
        
        During training: mels, mel_lens, p_targets, e_targets, d_targets are provided.
        During inference: these are None; the variance adaptor predicts them.
        """
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        # 1. Text Encoder
        output = self.encoder(texts, src_masks)

        # Speaker conditioning (if enabled)
        if self.multi_speaker and speakers is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_len, -1)

        # 2. Variance Adaptor
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        # 3. Mel Decoder
        output = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        # 4. PostNet
        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )

