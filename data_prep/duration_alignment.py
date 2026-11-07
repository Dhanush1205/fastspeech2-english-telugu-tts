"""Utilities for mapping forced-alignment intervals to FastSpeech 2 tokens."""
from __future__ import annotations

import numpy as np


SILENCE_LABELS = {"", "sp", "sil", "SIL", "SP", "SPN", "<eps>"}


def rescale_token_durations(durations, target_frames: int) -> np.ndarray:
    """Scale positive token durations so they sum exactly to the mel-frame count.

    Every token retains at least one frame. The relative aligned-duration
    proportions are preserved for the remaining frames using largest remainder
    allocation, so the result is deterministic and sums exactly to target_frames.
    """
    values = np.asarray(durations, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("No token durations supplied")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Token durations must be finite and strictly positive")
    if target_frames < values.size:
        raise ValueError(
            f"Mel length ({target_frames}) is shorter than token count ({values.size}); "
            "cannot assign at least one frame per token"
        )

    remaining = int(target_frames) - int(values.size)
    weights = np.maximum(values - 1.0, 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(values)
    quotas = remaining * weights / weights.sum()
    allocated = np.floor(quotas).astype(np.int64)
    left = remaining - int(allocated.sum())
    if left:
        order = np.argsort(-(quotas - allocated), kind="stable")
        allocated[order[:left]] += 1
    result = allocated + 1
    if int(result.sum()) != int(target_frames):
        raise AssertionError("Duration rescaling failed to conserve mel frames")
    return result.astype(np.int64)


def english_token_durations(expected_tokens, aligned_intervals, mel_frames: int, hop_length: int = 256, sample_rate: int = 22050) -> np.ndarray:
    """Map an MFA ARPABET phones tier onto the repository's G2P token sequence.

    Punctuation tokens are not represented in the ARPABET phones tier and get
    one frame each. One EOS token is appended. Aligned phone durations are
    required to match the expected G2P phone sequence exactly before rescaling.
    """
    import re

    expected = list(expected_tokens)
    is_phone = [bool(re.fullmatch(r"[A-Z]+[0-2]?", str(token))) for token in expected]
    expected_phones = [str(token).upper() for token, keep in zip(expected, is_phone) if keep]

    actual_labels = []
    aligned_durations = []
    for interval in aligned_intervals:
        label = str(interval.text).strip()
        if label in SILENCE_LABELS:
            continue
        actual_labels.append(label.upper())
        aligned_durations.append(max(1, int(round(interval.duration() * sample_rate / hop_length))))

    if actual_labels != expected_phones:
        limit = min(len(actual_labels), len(expected_phones))
        mismatch = next(
            (i for i in range(limit) if actual_labels[i] != expected_phones[i]),
            limit,
        )
        raise ValueError(
            "MFA phone sequence does not match the model's G2P sequence: "
            f"expected {len(expected_phones)} phones, aligned {len(actual_labels)}, "
            f"first mismatch index {mismatch}; expected={expected_phones[mismatch:mismatch+5]}, "
            f"aligned={actual_labels[mismatch:mismatch+5]}"
        )

    phone_iter = iter(aligned_durations)
    token_durations = [
        next(phone_iter) if keep else 1
        for keep in is_phone
    ]
    token_durations.append(1)  # EOS
    if next(phone_iter, None) is not None:
        raise ValueError("MFA produced more phone intervals than expected tokens")
    return rescale_token_durations(token_durations, mel_frames)


def telugu_token_durations(normalized_text: str, aligned_intervals, mel_frames: int, hop_length: int = 256, sample_rate: int = 22050) -> np.ndarray:
    """Map IndicMFA grapheme intervals to Unicode codepoint tokens plus EOS.

    The alignment corpus contains Telugu-block codepoints only. Whitespace,
    punctuation, ASCII digits and other tokenizer-visible symbols receive a
    one-frame placeholder; the final duration vector is rescaled to the mel
    frame count. This convention is recorded in metadata and is not a claim
    that these non-letter tokens received an acoustic alignment.
    """
    expected_chars = [ch for ch in normalized_text if 0x0C00 <= ord(ch) <= 0x0C7F]
    actual_chars = []
    aligned_durations = []
    for interval in aligned_intervals:
        label = str(interval.text).strip()
        if label in SILENCE_LABELS:
            continue
        normalized_label = label
        actual_chars.append(normalized_label)
        aligned_durations.append(max(1, int(round(interval.duration() * sample_rate / hop_length))))

    if actual_chars != expected_chars:
        limit = min(len(actual_chars), len(expected_chars))
        mismatch = next(
            (i for i in range(limit) if actual_chars[i] != expected_chars[i]),
            limit,
        )
        raise ValueError(
            "IndicMFA grapheme sequence does not match normalized Telugu text: "
            f"expected {len(expected_chars)} Telugu codepoints, aligned {len(actual_chars)}, "
            f"first mismatch index {mismatch}; expected={expected_chars[mismatch:mismatch+5]}, "
            f"aligned={actual_chars[mismatch:mismatch+5]}"
        )

    aligned_iter = iter(aligned_durations)
    durations = [
        next(aligned_iter) if 0x0C00 <= ord(ch) <= 0x0C7F else 1
        for ch in normalized_text
    ]
    durations.append(1)  # EOS
    if next(aligned_iter, None) is not None:
        raise ValueError("IndicMFA produced more grapheme intervals than expected")
    return rescale_token_durations(durations, mel_frames)
