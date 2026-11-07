"""
Telugu Text Processing for FastSpeech 2
========================================

Implements Telugu-specific text normalization and grapheme tokenization.
Uses Unicode NFC normalization and character-level vocabulary.

Telugu Unicode block: U+0C00–U+0C7F
Key script features:
  - Independent vowels: అ ఆ ఇ ఈ ఉ ఊ ఋ ఎ ఏ ఐ ఒ ఓ ఔ
  - Consonants: క ఖ గ ఘ ...
  - Vowel signs (mātras): ా ి ీ ు ూ ృ ే ై ో ై ...
  - Halant (virama): ్  (creates conjunct consonants)
  - Anusvara: ం
  - Visarga: ః
  - Nukta: ఀ

Design decision: Grapheme-based (not phoneme-based) because:
  1. No high-coverage Telugu phoneme dictionary available for IndicTTS
  2. Character-level tokenization is linguistically meaningful (aksharas)
  3. The FastSpeech 2 architecture supports arbitrary vocabulary sizes

Author: FastSpeech 2 Telugu TTS Project
Licence: MIT
"""

import re
import unicodedata
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Telugu Unicode ranges and character sets
# ---------------------------------------------------------------------------

# Telugu Unicode block: U+0C00 to U+0C7F
TELUGU_RANGE = (0x0C00, 0x0C7F)

# Independent vowels
TELUGU_VOWELS = "అఆఇఈఉఊఋఌఎఏఐఒఓఔ"

# Consonants (vyanjanas)
TELUGU_CONSONANTS = (
    "కఖగఘఙ"   # ka-varga
    "చఛజఝఞ"   # ca-varga
    "టఠడఢణ"   # ṭa-varga
    "తథదధన"   # ta-varga
    "పఫబభమ"   # pa-varga
    "యరలవశషసహళఴఱ"  # semi-vowels and sibilants
)

# Dependent vowel signs (mātras)
TELUGU_VOWEL_SIGNS = "ాిీుూృౄెేైొోౌ"

# Special characters
TELUGU_HALANT = "్"      # Virama / halant (creates conjuncts)
TELUGU_ANUSVARA = "ం"    # Anusvara (nasal)
TELUGU_VISARGA = "ః"     # Visarga
TELUGU_NUKTA = "ఀ"       # Nukta
TELUGU_CANDRABINDU = "ఁ" # Chandrabindu

# Digits (Telugu numerals)
TELUGU_DIGITS = "౦౧౨౩౪౫౬౭౮౯"

# Common punctuation used with Telugu text
TELUGU_PUNCTUATION = ".,!?;:-\"'()[]{}।॥"

# All special Telugu combining marks
TELUGU_COMBINING = TELUGU_HALANT + TELUGU_ANUSVARA + TELUGU_VISARGA + TELUGU_VOWEL_SIGNS

# Standard pad, end, start tokens
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPC_TOKEN = " "


# ---------------------------------------------------------------------------
# Text Cleaners / Normalizers
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Apply NFC Unicode normalization to Telugu text.
    
    NFC (Canonical Decomposition followed by Canonical Composition) ensures
    that combining characters are in a canonical form, which is important
    for consistent tokenization of Telugu vowel signs and conjuncts.
    
    Args:
        text: Raw Telugu Unicode string
        
    Returns:
        NFC-normalized string
    """
    return unicodedata.normalize("NFC", text)


def normalize_numbers(text: str) -> str:
    """Convert Telugu numerals to ASCII digits and normalize numeric patterns.
    
    Args:
        text: Text possibly containing Telugu numerals
        
    Returns:
        Text with Telugu numerals replaced by ASCII equivalents
    """
    # Replace Telugu numerals with ASCII
    telugu_digit_map = {
        "౦": "0", "౧": "1", "౨": "2", "౩": "3", "౪": "4",
        "౫": "5", "౬": "6", "౭": "7", "౮": "8", "౯": "9"
    }
    for tel, ascii_d in telugu_digit_map.items():
        text = text.replace(tel, ascii_d)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces, strip edges.
    
    Note: The assignment specifies preserving sentences exactly as provided.
    For the test sentences (T3 has "అడ  విలో"), the double space is preserved
    during preprocessing but normalized during inference tokenization.
    
    Args:
        text: Input text
        
    Returns:
        Whitespace-normalized text
    """
    # Replace multiple spaces with single space
    text = re.sub(r"  +", " ", text)
    return text.strip()


def remove_non_telugu_non_ascii(text: str) -> str:
    """Remove characters that are neither Telugu Unicode nor standard ASCII.
    
    Keeps: Telugu block (U+0C00-U+0C7F), ASCII printable, space
    Removes: Other Unicode ranges that may appear from copy-paste
    
    Args:
        text: Input text
        
    Returns:
        Filtered text
    """
    filtered = []
    for ch in text:
        cp = ord(ch)
        if 0x0C00 <= cp <= 0x0C7F:  # Telugu block
            filtered.append(ch)
        elif 0x20 <= cp <= 0x7E:    # ASCII printable
            filtered.append(ch)
        elif ch in [" ", "\t", "\n"]:
            filtered.append(" ")
        # else: discard
    return "".join(filtered)


def telugu_text_normalize(text: str, preserve_exact: bool = False) -> str:
    """Full Telugu text normalization pipeline.
    
    Args:
        text: Raw Telugu text
        preserve_exact: If True, skip whitespace normalization (for test sentences)
        
    Returns:
        Normalized Telugu text ready for tokenization
    """
    # Step 1: Unicode NFC normalization
    text = normalize_unicode(text)
    
    # Step 2: Normalize Telugu/ASCII numerals
    text = normalize_numbers(text)
    
    # Step 3: Whitespace normalization (optional for test sentences)
    if not preserve_exact:
        text = normalize_whitespace(text)
    
    return text.strip()


# ---------------------------------------------------------------------------
# Vocabulary Construction
# ---------------------------------------------------------------------------

def build_telugu_vocabulary(texts: Optional[List[str]] = None) -> Dict[str, int]:
    """Build a Telugu character vocabulary.
    
    Constructs vocab from:
    1. All known Telugu Unicode characters in the script range
    2. Common punctuation used in TTS
    3. Special tokens: pad, eos, unk
    
    Args:
        texts: Optional list of texts to scan for additional characters
        
    Returns:
        Dictionary mapping character to integer index
    """
    # Start with special tokens
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, EOS_TOKEN: 2, SPC_TOKEN: 3}
    
    idx = 4
    
    # Add all Telugu characters from the Unicode block that are commonly used
    # Vowels
    for ch in TELUGU_VOWELS:
        if ch not in vocab:
            vocab[ch] = idx
            idx += 1
    
    # Consonants
    for ch in TELUGU_CONSONANTS:
        if ch not in vocab:
            vocab[ch] = idx
            idx += 1
    
    # Vowel signs (dependent)
    for ch in TELUGU_VOWEL_SIGNS:
        if ch not in vocab:
            vocab[ch] = idx
            idx += 1
    
    # Halant and special marks
    for ch in [TELUGU_HALANT, TELUGU_ANUSVARA, TELUGU_VISARGA,
               TELUGU_NUKTA, TELUGU_CANDRABINDU]:
        if ch not in vocab:
            vocab[ch] = idx
            idx += 1
    
    # ASCII digits
    for ch in "0123456789":
        if ch not in vocab:
            vocab[ch] = idx
            idx += 1
    
    # Common punctuation
    for ch in ".,!?;:-\"'()[]{}।॥%":
        if ch not in vocab:
            vocab[ch] = idx
            idx += 1
    
    # Scan additional texts for any characters we missed
    if texts:
        for text in texts:
            normalized = normalize_unicode(text)
            for ch in normalized:
                if ch not in vocab:
                    logger.debug(f"Adding unseen character to vocab: U+{ord(ch):04X} '{ch}'")
                    vocab[ch] = idx
                    idx += 1
    
    logger.info(f"Telugu vocabulary size: {len(vocab)}")
    return vocab


def save_vocabulary(vocab: Dict[str, int], path: str) -> None:
    """Save vocabulary to JSON file.
    
    Args:
        vocab: Vocabulary dictionary
        path: Output file path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    logger.info(f"Vocabulary saved to {path}")


def load_vocabulary(path: str) -> Dict[str, int]:
    """Load vocabulary from JSON file.
    
    Args:
        path: Path to vocabulary JSON file
        
    Returns:
        Vocabulary dictionary
    """
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    logger.info(f"Vocabulary loaded from {path} ({len(vocab)} tokens)")
    return vocab


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TeluguTokenizer:
    """Character-level tokenizer for Telugu text.
    
    Handles Telugu Unicode grapheme clusters by treating each Unicode
    code point as a separate token. This is consistent with how the
    reference FastSpeech 2 implementation handles phoneme sequences:
    each element of the sequence is an index into an embedding table.
    
    For Telugu:
      - Each character (including combining marks like vowel signs, halant)
        is a separate token.
      - This matches how FastSpeech 2 handles phoneme sequences.
      - Spaces between words produce a space token.
    
    Design note:
      A more linguistically accurate approach would segment by aksharas
      (syllable clusters), but that requires a dedicated Telugu aksharas
      segmentation library. Character-level tokenization is chosen for
      simplicity and robustness.
    """
    
    def __init__(self, vocab: Dict[str, int]):
        """Initialize tokenizer with vocabulary.
        
        Args:
            vocab: Character-to-index mapping
        """
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.pad_id = vocab[PAD_TOKEN]
        self.eos_id = vocab[EOS_TOKEN]
        self.unk_id = vocab[UNK_TOKEN]
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def text_to_sequence(
        self,
        text: str,
        add_eos: bool = True,
        normalize: bool = True
    ) -> List[int]:
        """Convert Telugu text to sequence of token indices.
        
        Args:
            text: Input Telugu text
            add_eos: Whether to append EOS token
            normalize: Whether to apply normalization
            
        Returns:
            List of integer token indices
        """
        if normalize:
            text = telugu_text_normalize(text, preserve_exact=False)
        
        sequence = []
        for ch in text:
            if ch in self.vocab:
                sequence.append(self.vocab[ch])
            elif ch == " ":
                sequence.append(self.vocab.get(SPC_TOKEN, self.unk_id))
            else:
                logger.warning(f"Unknown character: U+{ord(ch):04X} '{ch}', using UNK")
                sequence.append(self.unk_id)
        
        if add_eos:
            sequence.append(self.eos_id)
        
        return sequence
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """Convert sequence of token indices back to text.
        
        Args:
            sequence: List of token indices
            
        Returns:
            Reconstructed text string
        """
        return "".join(
            self.inv_vocab.get(idx, UNK_TOKEN)
            for idx in sequence
            if idx not in [self.pad_id, self.eos_id]
        )
    
    @classmethod
    def from_vocab_file(cls, vocab_path: str) -> "TeluguTokenizer":
        """Load tokenizer from saved vocabulary file.
        
        Args:
            vocab_path: Path to vocabulary JSON file
            
        Returns:
            Initialized tokenizer
        """
        vocab = load_vocabulary(vocab_path)
        return cls(vocab)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def is_telugu_char(ch: str) -> bool:
    """Check if a character is in the Telugu Unicode block.
    
    Args:
        ch: Single character
        
    Returns:
        True if Telugu character
    """
    cp = ord(ch)
    return 0x0C00 <= cp <= 0x0C7F


def count_telugu_chars(text: str) -> int:
    """Count Telugu characters in text.
    
    Args:
        text: Input text
        
    Returns:
        Count of Telugu Unicode characters
    """
    return sum(1 for ch in text if is_telugu_char(ch))


def validate_telugu_text(text: str) -> Tuple[bool, List[str]]:
    """Validate that text is usable Telugu for TTS.
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings = []
    
    # Check non-empty
    normalized = normalize_unicode(text).strip()
    if not normalized:
        return False, ["Empty text after normalization"]
    
    # Check has Telugu content
    telugu_count = count_telugu_chars(normalized)
    if telugu_count == 0:
        warnings.append("No Telugu characters found in text")
    
    # Check for very long sequences
    if len(normalized) > 300:
        warnings.append(f"Text is very long ({len(normalized)} chars), may exceed model capacity")
    
    return True, warnings


# ---------------------------------------------------------------------------
# Test Sentence Processing
# ---------------------------------------------------------------------------

# Test sentences from assignment PDF — preserved exactly as specified
TELUGU_TEST_SENTENCES = {
    "T1": "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    "T2a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    "T2b": "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
    # T3: Note double space in "అడ  విలో" — preserved from PDF
    "T3": "పులి అడ  విలో నిశ్శబ్దంగా నడిచింది.",
    # T4: Line breaks in PDF reconstructed as single sentence
    "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
}


if __name__ == "__main__":
    # Quick test of the Telugu text processing pipeline
    logging.basicConfig(level=logging.INFO)
    
    print("=== Telugu Text Processing Test ===\n")
    
    # Build vocabulary
    all_texts = list(TELUGU_TEST_SENTENCES.values())
    vocab = build_telugu_vocabulary(texts=all_texts)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create tokenizer
    tokenizer = TeluguTokenizer(vocab)
    
    # Test each sentence
    for sent_id, sentence in TELUGU_TEST_SENTENCES.items():
        print(f"\n{sent_id}: {sentence}")
        
        # Normalize
        normalized = telugu_text_normalize(sentence)
        print(f"  Normalized: {normalized}")
        
        # Tokenize
        sequence = tokenizer.text_to_sequence(sentence)
        print(f"  Tokens ({len(sequence)}): {sequence[:20]}{'...' if len(sequence) > 20 else ''}")
        
        # Reconstruct
        reconstructed = tokenizer.sequence_to_text(sequence)
        print(f"  Reconstructed: {reconstructed}")
        
        # Validate
        is_valid, warnings = validate_telugu_text(sentence)
        print(f"  Valid: {is_valid}, Warnings: {warnings}")

