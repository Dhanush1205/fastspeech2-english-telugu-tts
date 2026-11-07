"""
HiFi-GAN Vocoder Loader Utility
===============================

Loads the HiFi-GAN Generator with either pre-trained weights or fallback Griffin-Lim.
Compatible with ming024/FastSpeech2 utils/tools.py.

Licence: MIT
"""

import os
import json
import torch
from pathlib import Path
from hifigan.models import Generator


def get_vocoder(config_path=None, checkpoint_path=None, device="cpu"):
    """Loads HiFi-GAN generator model."""
    if config_path is None:
        config_path = str(Path(__file__).parent / "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    generator = Generator(config).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(state_dict["generator"] if "generator" in state_dict else state_dict)
    
    generator.eval()
    return generator

