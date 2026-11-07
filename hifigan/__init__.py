"""
HiFi-GAN Neural Vocoder Package
================================

Implements the HiFi-GAN Generator architecture for Mel-spectrogram inversion.
Reference:
  HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis
  Jungil Kong, Jaehyeon Kim, Jaekyoung Bae (NeurIPS 2020)
  https://arxiv.org/abs/2010.05646
"""

from hifigan.models import Generator
from hifigan.get_vocoder import get_vocoder

__all__ = ["Generator", "get_vocoder"]
