"""Unit tests for real IndicTTS audio decoding and resampling."""
import unittest
import numpy as np

from data_prep.prepare_indictts import decode_audio, resample_audio


class AudioSamples:
    def __init__(self, data, sample_rate):
        self.data = data
        self.sample_rate = sample_rate


class MockAudioDecoder:
    def __init__(self, data, sample_rate):
        self.samples = AudioSamples(data, sample_rate)

    def get_all_samples(self):
        return self.samples


class TestIndicTTSAudio(unittest.TestCase):
    def test_audio_decoder_mono(self):
        waveform = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
        audio = MockAudioDecoder(waveform, 16000)
        decoded, sr = decode_audio(audio)
        self.assertEqual(sr, 16000)
        self.assertEqual(decoded.shape, (16000,))
        np.testing.assert_allclose(decoded, waveform)

    def test_audio_decoder_stereo_torchcodec_layout(self):
        waveform = np.stack([
            np.ones(16000, dtype=np.float32) * 0.25,
            np.ones(16000, dtype=np.float32) * 0.75,
        ])
        decoded, sr = decode_audio(MockAudioDecoder(waveform, 16000))
        self.assertEqual(sr, 16000)
        self.assertEqual(decoded.shape, (16000,))
        self.assertAlmostEqual(float(decoded.mean()), 0.5, places=5)

    def test_resampling_changes_sample_count_and_preserves_duration(self):
        waveform = np.zeros(16000, dtype=np.float32)
        converted = resample_audio(waveform, 16000, 22050)
        self.assertEqual(len(converted), 22050)
        self.assertTrue(np.isfinite(converted).all())

    def test_reject_missing_audio(self):
        with self.assertRaises(ValueError):
            decode_audio(None)

    def test_reject_invalid_waveform(self):
        audio = MockAudioDecoder(np.array([0.0, np.nan], dtype=np.float32), 16000)
        with self.assertRaises(ValueError):
            decode_audio(audio)


class TestSvarahAudio(unittest.TestCase):
    def test_audio_decoder_mono(self):
        from data_prep.prepare_svarah import extract_audio_and_sr

        waveform = np.linspace(-0.4, 0.4, 16000, dtype=np.float32)
        arr, sr = extract_audio_and_sr(MockAudioDecoder(waveform, 16000))
        self.assertEqual(sr, 16000)
        self.assertEqual(arr.shape, (16000,))
        np.testing.assert_allclose(arr, waveform)

    def test_demographic_fallback_is_explicitly_a_proxy(self):
        from data_prep.prepare_svarah import extract_speaker_id

        item = {
            "native_place_state": "Telangana",
            "native_place_district": "Hyderabad",
            "primary_language": "Telugu",
            "gender": "Female",
            "age-group": "18-30",
        }
        proxy = extract_speaker_id(item)
        self.assertIn("Telangana", proxy)
        self.assertIn("Hyderabad", proxy)
        self.assertIn("Telugu", proxy)
        # This function can only construct a metadata group from this schema;
        # it is not evidence of a stable source speaker identifier.
        self.assertNotEqual(proxy, "unknown_speaker")



class MockInterval:
    def __init__(self, text, duration):
        self.text = text
        self._duration = duration

    def duration(self):
        return self._duration


class TestDurationAlignment(unittest.TestCase):
    def test_rescale_preserves_frame_total(self):
        from data_prep.duration_alignment import rescale_token_durations

        result = rescale_token_durations([2, 3, 5], target_frames=100)
        self.assertEqual(result.dtype, np.int64)
        self.assertTrue(np.all(result >= 1))
        self.assertEqual(int(result.sum()), 100)

    def test_rescale_rejects_more_tokens_than_mel_frames(self):
        from data_prep.duration_alignment import rescale_token_durations

        with self.assertRaises(ValueError):
            rescale_token_durations([1, 1, 1], target_frames=2)

    def test_english_duration_vector_matches_g2p_tokens_and_eos(self):
        from data_prep.duration_alignment import english_token_durations

        tokens = ["HH", "AH0", " ", "EH1", "."]
        intervals = [
            MockInterval("HH", 0.05),
            MockInterval("AH0", 0.08),
            MockInterval("EH1", 0.07),
        ]
        result = english_token_durations(tokens, intervals, mel_frames=40)
        self.assertEqual(len(result), len(tokens) + 1)
        self.assertEqual(int(result.sum()), 40)

    def test_english_duration_rejects_phone_mismatch(self):
        from data_prep.duration_alignment import english_token_durations

        with self.assertRaises(ValueError):
            english_token_durations(
                ["HH", "AH0"],
                [MockInterval("HH", 0.05), MockInterval("AE1", 0.08)],
                mel_frames=20,
            )

    def test_telugu_duration_vector_matches_unicode_tokens_and_eos(self):
        from data_prep.duration_alignment import telugu_token_durations

        normalized = "అ"
        intervals = [MockInterval("అ", 0.1)]
        result = telugu_token_durations(normalized, intervals, mel_frames=20)
        self.assertEqual(len(result), len(normalized) + 1)
        self.assertEqual(int(result.sum()), 20)

    def test_telugu_duration_rejects_grapheme_mismatch(self):
        from data_prep.duration_alignment import telugu_token_durations

        with self.assertRaises(ValueError):
            telugu_token_durations(
                "అ", [MockInterval("ఆ", 0.1)], mel_frames=20
            )



if __name__ == "__main__":
    unittest.main()
