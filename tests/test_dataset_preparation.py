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



if __name__ == "__main__":
    unittest.main()
