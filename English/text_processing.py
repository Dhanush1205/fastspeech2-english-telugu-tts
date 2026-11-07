"""
English Text Processing for FastSpeech 2
==========================================

Implements English text normalization and phonemization for Svarah dataset.
Uses g2p_en for grapheme-to-phoneme conversion.

Design decision:
  Phoneme-based input (ARPABET) is used for English because:
  1. The reference implementation (ming024/FastSpeech2) is phoneme-based
  2. MFA forced alignment works with CMU pronunciation dictionary (ARPABET)
  3. Phoneme-level duration targets are more precise
  4. English has well-established G2P resources (g2p_en, CMU dict)

Author: FastSpeech 2 English TTS Project
Licence: MIT
"""

import re
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ARPABET phoneme set (from CMU dictionary + extensions)
# Compatible with ming024/FastSpeech2 symbols
# ---------------------------------------------------------------------------

# Consonants (ARPABET)
_CONSONANTS = [
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M",
    "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y",
    "Z", "ZH"
]

# Vowels (ARPABET, with stress markers 0, 1, 2)
_VOWELS = [
    "AA0", "AA1", "AA2",
    "AE0", "AE1", "AE2",
    "AH0", "AH1", "AH2",
    "AO0", "AO1", "AO2",
    "AW0", "AW1", "AW2",
    "AY0", "AY1", "AY2",
    "EH0", "EH1", "EH2",
    "ER0", "ER1", "ER2",
    "EY0", "EY1", "EY2",
    "IH0", "IH1", "IH2",
    "IY0", "IY1", "IY2",
    "OW0", "OW1", "OW2",
    "OY0", "OY1", "OY2",
    "UH0", "UH1", "UH2",
    "UW0", "UW1", "UW2",
]

# Special tokens
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPC_TOKEN = " "

# Punctuation marks (some TTS systems include these in the phoneme sequence)
_PUNCTUATION = "!',-.?;:"


def build_english_phoneme_vocabulary() -> Dict[str, int]:
    """Build English phoneme vocabulary (ARPABET-based).
    
    Returns:
        Dictionary mapping phoneme/symbol to integer index
    """
    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
        EOS_TOKEN: 2,
        SPC_TOKEN: 3,
    }
    idx = 4
    
    for punc in _PUNCTUATION:
        vocab[punc] = idx
        idx += 1
    
    for phone in _CONSONANTS:
        vocab[phone] = idx
        idx += 1
    
    for phone in _VOWELS:
        vocab[phone] = idx
        idx += 1
    
    logger.info(f"English phoneme vocabulary size: {len(vocab)}")
    return vocab


# ---------------------------------------------------------------------------
# Text Cleaners
# ---------------------------------------------------------------------------

def normalize_numbers(text: str) -> str:
    """Expand numbers in text to spoken words.
    
    Handles common patterns:
    - Times: 10:45 → "ten forty-five"
    - Years: 2026 → "twenty twenty-six" (or "two thousand twenty-six")
    - Ordinals: 12th → "twelfth"
    
    Note: For the test sentence E4, specific handling of "10:45 AM" and
    "12 November 2026" is implemented.
    
    Args:
        text: Input English text
        
    Returns:
        Text with numbers expanded
    """
    # Time pattern: HH:MM AM/PM
    def expand_time(match):
        time_str = match.group(0)
        parts = re.match(r"(\d+):(\d+)\s*(AM|PM)?", time_str)
        if parts:
            hour = int(parts.group(1))
            minute = int(parts.group(2))
            period = parts.group(3) or ""
            
            hour_words = _number_to_words(hour)
            if minute == 0:
                min_words = "o'clock"
            elif minute < 10:
                min_words = f"oh {_number_to_words(minute)}"
            else:
                min_words = _number_to_words(minute)
            
            return f"{hour_words} {min_words} {period}".strip()
        return time_str
    
    text = re.sub(r"\d+:\d+\s*(?:AM|PM)?", expand_time, text)
    
    # Ordinals: 12th, 1st, 2nd, 3rd
    def expand_ordinal(match):
        num = int(match.group(1))
        return _ordinal_to_words(num)
    
    text = re.sub(r"(\d+)(?:st|nd|rd|th)\b", expand_ordinal, text)
    
    # Plain numbers (4+ digits = year/large number)
    def expand_number(match):
        num = int(match.group(0))
        return _number_to_words(num)
    
    text = re.sub(r"\b\d+\b", expand_number, text)
    
    return text


def _number_to_words(n: int) -> str:
    """Convert integer to English words.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Word representation
    """
    ones = ["", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen",
            "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
            "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty",
            "sixty", "seventy", "eighty", "ninety"]
    
    if n < 0:
        return "minus " + _number_to_words(-n)
    if n < 20:
        return ones[n]
    if n < 100:
        return tens[n // 10] + ("-" + ones[n % 10] if n % 10 else "")
    if n < 1000:
        rest = n % 100
        return ones[n // 100] + " hundred" + (" " + _number_to_words(rest) if rest else "")
    if n < 10000:
        rest = n % 1000
        return _number_to_words(n // 1000) + " thousand" + (" " + _number_to_words(rest) if rest else "")
    return str(n)  # fallback for very large numbers


def _ordinal_to_words(n: int) -> str:
    """Convert integer to ordinal English words.
    
    Args:
        n: Positive integer
        
    Returns:
        Ordinal word (e.g., 12 → "twelfth")
    """
    ordinals = {
        1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
        6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
        11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
        15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
        19: "nineteenth", 20: "twentieth",
    }
    if n in ordinals:
        return ordinals[n]
    words = _number_to_words(n)
    if words.endswith("y"):
        return words[:-1] + "ieth"
    return words + "th"


def expand_abbreviations(text: str) -> str:
    """Expand common abbreviations.
    
    Args:
        text: Input text
        
    Returns:
        Text with abbreviations expanded
    """
    abbrevs = [
        (r"\bMr\.\b", "Mister"),
        (r"\bMrs\.\b", "Missus"),
        (r"\bDr\.\b", "Doctor"),
        (r"\bProf\.\b", "Professor"),
        (r"\bAM\b", "A M"),
        (r"\bPM\b", "P M"),
        (r"\bvs\.\b", "versus"),
        (r"\betc\.\b", "et cetera"),
    ]
    for pattern, replacement in abbrevs:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_apostrophes(text: str) -> str:
    """Normalize contractions and possessives.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    contractions = [
        (r"won't", "will not"),
        (r"can't", "cannot"),
        (r"n't\b", " not"),
        (r"'re\b", " are"),
        (r"'s\b", "'s"),   # Keep possessive as-is
        (r"'ll\b", " will"),
        (r"'ve\b", " have"),
        (r"'d\b", " would"),
        (r"'m\b", " am"),
    ]
    for pattern, replacement in contractions:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def clean_english_text(text: str) -> str:
    """Full English text normalization pipeline.
    
    Args:
        text: Raw English text
        
    Returns:
        Cleaned text ready for phonemization
    """
    text = text.strip()
    
    # Normalize unicode (handle fancy quotes, dashes, etc.)
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    
    # Expand abbreviations
    text = expand_abbreviations(text)
    
    # Normalize contractions
    text = normalize_apostrophes(text)
    
    # Expand numbers
    text = normalize_numbers(text)
    
    # Clean remaining punctuation (keep space, letters, common punc)
    text = re.sub(r"[^\w\s\-',!?.]", " ", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text.lower()


# ---------------------------------------------------------------------------
# Phonemizer
# ---------------------------------------------------------------------------

class EnglishPhonemizer:
    """English grapheme-to-phoneme converter using g2p_en.
    
    Falls back to character-level output if g2p_en is not installed.
    """
    
    def __init__(self):
        self._g2p = None
        self._load_g2p()
    
    def _load_g2p(self):
        """Try to load g2p_en with required NLTK taggers."""
        try:
            import nltk
            for resource in ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict"]:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass
            from g2p_en import G2p
            self._g2p = G2p()
            logger.info("g2p_en loaded successfully")
        except ImportError:
            logger.warning(
                "g2p_en not installed. Run: pip install g2p_en\n"
                "Falling back to character-level tokenization."
            )
    
    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert text to phoneme sequence.
        
        Args:
            text: Cleaned English text (lowercase)
            
        Returns:
            List of ARPABET phoneme strings
        """
        if self._g2p is None:
            # Fallback: character-level
            return list(text)
        
        phonemes = self._g2p(text)
        # g2p_en returns list of phonemes and punctuation
        # Filter to keep only valid phonemes and spaces
        result = []
        for ph in phonemes:
            if ph.strip():
                result.append(ph)
        return result
    
    def phonemes_to_sequence(
        self,
        phonemes: List[str],
        vocab: Dict[str, int],
        add_eos: bool = True
    ) -> List[int]:
        """Convert phoneme list to index sequence.
        
        Args:
            phonemes: List of phoneme strings
            vocab: Phoneme vocabulary
            add_eos: Append EOS token
            
        Returns:
            List of integer indices
        """
        unk_id = vocab.get(UNK_TOKEN, 1)
        sequence = [vocab.get(ph, unk_id) for ph in phonemes]
        if add_eos:
            sequence.append(vocab.get(EOS_TOKEN, 2))
        return sequence
    
    def text_to_sequence(
        self,
        text: str,
        vocab: Dict[str, int],
        normalize: bool = True,
        add_eos: bool = True
    ) -> List[int]:
        """Full pipeline: text → phonemes → indices.
        
        Args:
            text: Raw English text
            vocab: Phoneme vocabulary
            normalize: Apply text normalization
            add_eos: Append EOS token
            
        Returns:
            List of integer token indices
        """
        if normalize:
            text = clean_english_text(text)
        
        phonemes = self.text_to_phonemes(text)
        return self.phonemes_to_sequence(phonemes, vocab, add_eos)


# ---------------------------------------------------------------------------
# Test Sentences
# ---------------------------------------------------------------------------

# English test sentences from assignment PDF — preserved exactly
ENGLISH_TEST_SENTENCES = {
    "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
    # E3: Note the double space in "rhyth  m" — preserved from PDF
    "E3": "The rhyth  m of the orchestra changed unexpectedly.",
    # E4: Contains "andaroo" (Telugu-influenced English word) and time/date
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== English Text Processing Test ===\n")
    
    phonemizer = EnglishPhonemizer()
    vocab = build_english_phoneme_vocabulary()
    
    for sent_id, sentence in ENGLISH_TEST_SENTENCES.items():
        print(f"\n{sent_id}: {sentence}")
        cleaned = clean_english_text(sentence)
        print(f"  Cleaned: {cleaned}")
        
        try:
            phonemes = phonemizer.text_to_phonemes(cleaned)
            print(f"  Phonemes ({len(phonemes)}): {phonemes[:15]}{'...' if len(phonemes) > 15 else ''}")
            sequence = phonemizer.phonemes_to_sequence(phonemes, vocab)
            print(f"  Sequence length: {len(sequence)}")
        except Exception as e:
            print(f"  Error: {e}")

