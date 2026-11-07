"""
Unit and Integration Tests for FastSpeech 2
===========================================

Tests:
  1. Text Normalization and Tokenization (English & Telugu)
  2. Character / Phoneme Vocabulary generation
  3. Model Components (Encoder, Decoder, VarianceAdaptor, LengthRegulator)
  4. FastSpeech2 full forward pass
  5. FastSpeech2 loss computation
  6. Batch collation and tensor masking
"""

import unittest
import torch
import numpy as np

from English.text_processing import (
    clean_english_text,
    build_english_phoneme_vocabulary,
    EnglishPhonemizer,
    ENGLISH_TEST_SENTENCES,
)
from Telugu.text_processing import (
    telugu_text_normalize,
    build_telugu_vocabulary,
    TeluguTokenizer,
    TELUGU_TEST_SENTENCES,
)
from model.fastspeech2 import FastSpeech2
from model.loss import FastSpeech2Loss
from model.dataset import collate_fn


class TestFastSpeech2(unittest.TestCase):

    def setUp(self):
        self.mock_preprocess_config = {
            "dataset": "Test",
            "path": {"preprocessed_path": "mock"},
            "preprocessing": {
                "audio": {"n_mel_channels": 80},
                "mel": {"n_mel_channels": 80},
                "pitch": {"normalization": True},
                "energy": {"normalization": True},
            },
        }
        self.mock_model_config = {
            "transformer": {
                "encoder_layer": 2,
                "encoder_head": 2,
                "encoder_hidden": 64,
                "decoder_layer": 2,
                "decoder_head": 2,
                "decoder_hidden": 64,
                "conv_filter_size": 128,
                "conv_kernel_size": [5, 1],
                "encoder_dropout": 0.1,
                "decoder_dropout": 0.1,
                "max_seq_len": 500,
            },
            "variance_predictor": {
                "filter_size": 64,
                "kernel_size": 3,
                "dropout: 0.1": 0.1,
                "dropout": 0.1,
            },
            "variance_embedding": {
                "n_bins": 256,
                "pitch_quantization": "linear",
                "energy_quantization": "linear",
            },
            "multi_speaker": False,
        }

    def test_english_text_cleaning(self):
        text = "The meeting will start at 10:45 AM on 12 November 2026."
        cleaned = clean_english_text(text)
        self.assertTrue(len(cleaned) > 0)
        self.assertIn("forty-five", cleaned)
        self.assertIn("twelve", cleaned)

        ordinal_test = clean_english_text("The 12th of November.")
        self.assertIn("twelfth", ordinal_test)

    def test_telugu_text_normalization(self):
        for sid, text in TELUGU_TEST_SENTENCES.items():
            norm = telugu_text_normalize(text)
            self.assertTrue(len(norm) > 0)

    def test_telugu_tokenization(self):
        vocab = build_telugu_vocabulary(list(TELUGU_TEST_SENTENCES.values()))
        tokenizer = TeluguTokenizer(vocab)
        test_str = TELUGU_TEST_SENTENCES["T1"]
        seq = tokenizer.text_to_sequence(test_str)
        self.assertTrue(len(seq) > 0)
        reconstructed = tokenizer.sequence_to_text(seq)
        self.assertTrue(len(reconstructed) > 0)

    def test_model_forward_training(self):
        vocab_size = 100
        model = FastSpeech2(self.mock_preprocess_config, self.mock_model_config, vocab_size)
        
        batch_size = 2
        seq_len = 10
        mel_len = 30
        n_mel = 80

        texts = torch.randint(1, vocab_size, (batch_size, seq_len))
        src_lens = torch.tensor([seq_len, seq_len - 2], dtype=torch.long)
        mels = torch.randn(batch_size, mel_len, n_mel)
        mel_lens = torch.tensor([mel_len, mel_len - 5], dtype=torch.long)
        p_targets = torch.randn(batch_size, mel_len)
        e_targets = torch.randn(batch_size, mel_len)
        d_targets = torch.randint(1, 5, (batch_size, seq_len))

        output = model(
            speakers=None,
            texts=texts,
            src_lens=src_lens,
            max_src_len=seq_len,
            mels=mels,
            mel_lens=mel_lens,
            max_mel_len=mel_len,
            p_targets=p_targets,
            e_targets=e_targets,
            d_targets=d_targets,
        )

        mel_pred, postnet_pred, p_pred, e_pred, log_d_pred, d_round, src_mask, mel_mask, _, _ = output
        self.assertEqual(mel_pred.shape, (batch_size, mel_len, n_mel))
        self.assertEqual(postnet_pred.shape, (batch_size, mel_len, n_mel))

        # Test loss computation
        batch = [
            ["id1", "id2"],
            ["t1", "t2"],
            torch.tensor([0, 0]),
            texts,
            src_lens,
            seq_len,
            mels,
            mel_lens,
            mel_len,
            p_targets,
            e_targets,
            d_targets,
        ]

        loss_fn = FastSpeech2Loss(self.mock_preprocess_config, self.mock_model_config)
        losses = loss_fn(batch, output)
        total_loss = losses[0]
        self.assertTrue(total_loss.item() > 0)

    def test_model_forward_inference(self):
        vocab_size = 80
        model = FastSpeech2(self.mock_preprocess_config, self.mock_model_config, vocab_size)
        model.eval()

        texts = torch.randint(1, vocab_size, (1, 15))
        src_lens = torch.tensor([15], dtype=torch.long)

        with torch.no_grad():
            output = model(
                speakers=None,
                texts=texts,
                src_lens=src_lens,
                max_src_len=15,
            )

        mel_pred, postnet_pred, p_pred, e_pred, log_d_pred, d_round, _, _, _, mel_len = output
        self.assertTrue(mel_len.item() > 0)
        self.assertEqual(mel_pred.shape[1], mel_len.item())


    def test_hifigan_generator_forward(self):
        from hifigan.get_vocoder import get_vocoder
        vocoder = get_vocoder(device="cpu")
        dummy_mel = torch.randn(1, 80, 50)
        with torch.no_grad():
            wav = vocoder(dummy_mel)
        self.assertEqual(wav.shape[0], 1)
        self.assertEqual(wav.shape[1], 1)
        # Expected upsampled length: 50 * 256 = 12800 samples
        self.assertEqual(wav.shape[2], 50 * 256)

    def test_checkpoint_validation(self):
        """Verify that load_model rejects missing and invalid checkpoints without random fallback."""
        from English.inference import load_model as load_english_model
        from Telugu.inference import load_model as load_telugu_model
        import tempfile
        import os

        # Missing checkpoint must raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            load_english_model(self.mock_preprocess_config, self.mock_model_config, "nonexistent_path/chkpt.pth.tar", "cpu")

        with self.assertRaises(FileNotFoundError):
            load_telugu_model(self.mock_preprocess_config, self.mock_model_config, "nonexistent_path/chkpt.pth.tar", "cpu")

        # Invalid checkpoint format must raise ValueError
        with tempfile.NamedTemporaryFile(suffix=".pth.tar", delete=False) as f:
            invalid_path = f.name
            torch.save({"unrelated_key": 123}, invalid_path)

        try:
            with self.assertRaises(ValueError):
                load_english_model(self.mock_preprocess_config, self.mock_model_config, invalid_path, "cpu")
            with self.assertRaises(ValueError):
                load_telugu_model(self.mock_preprocess_config, self.mock_model_config, invalid_path, "cpu")
        finally:
            if os.path.exists(invalid_path):
                os.remove(invalid_path)

    def test_audio_inversion_no_synthetic_fallback(self):
        """Verify audio inversion raises RuntimeError rather than generating sine waves."""
        from English.inference import invert_mel_to_audio as invert_english_mel
        from Telugu.inference import invert_mel_to_audio as invert_telugu_mel

        dummy_mel = np.random.randn(80, 50)

        # Neither vocoder nor Griffin-Lim allowed should raise RuntimeError
        with self.assertRaises(RuntimeError):
            invert_english_mel(dummy_mel, vocoder=None, allow_griffin_lim=False)

        with self.assertRaises(RuntimeError):
            invert_telugu_mel(dummy_mel, vocoder=None, allow_griffin_lim=False)

    def test_plotting_missing_prediction_error(self):
        """Verify that plot utilities fail cleanly when prediction or history files are missing."""
        from utils.plot_variance import load_predictions
        from utils.plot_losses import load_loss_history

        with self.assertRaises(FileNotFoundError):
            load_predictions("nonexistent_dir/pred.npz")

        with self.assertRaises(FileNotFoundError):
            load_loss_history("nonexistent_dir/loss_history.npz")

    def test_dataset_strict_validation(self):
        """Verify FastSpeech2Dataset raises FileNotFoundError for missing preprocessed features."""
        import tempfile
        import os
        from model.dataset import FastSpeech2Dataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = os.path.join(tmp_dir, "train.txt")
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write("sample_01|dummy.wav|Hello world|speaker1\n")

            prep_cfg = {
                "dataset": "Test",
                "path": {"preprocessed_path": tmp_dir, "raw_path": tmp_dir},
                "preprocessing": {"audio": {"n_mel_channels": 80}},
            }
            train_cfg = {}

            ds = FastSpeech2Dataset(
                filename="train.txt",
                preprocess_config=prep_cfg,
                train_config=train_cfg,
                language="en",
            )
            self.assertEqual(len(ds), 1)

            # Accessing item without preprocessed mel/pitch/energy must raise FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                _ = ds[0]

    def test_audio_decoder_extraction(self):
        """Verify extract_audio_and_sr correctly decodes AudioDecoder objects (torchcodec / datasets 4+)."""
        from data_prep.prepare_svarah import extract_audio_and_sr

        class MockAudioSamples:
            def __init__(self, tensor, sample_rate):
                self.data = tensor
                self.sample_rate = sample_rate

        class MockAudioDecoder:
            def __init__(self, tensor, sample_rate):
                self._samples = MockAudioSamples(tensor, sample_rate)

            def get_all_samples(self):
                return self._samples

        # 1. Test mono tensor [1, 16000]
        mono_tensor = torch.randn(1, 16000)
        mono_decoder = MockAudioDecoder(mono_tensor, 16000)
        arr, sr = extract_audio_and_sr(mono_decoder)
        self.assertIsNotNone(arr)
        self.assertEqual(sr, 16000)
        self.assertEqual(arr.shape, (16000,))
        self.assertTrue(np.all(np.isfinite(arr)))

        # 2. Test stereo tensor [2, 16000] -> converted to 1D mono
        stereo_tensor = torch.randn(2, 16000)
        stereo_decoder = MockAudioDecoder(stereo_tensor, 16000)
        arr_stereo, sr_stereo = extract_audio_and_sr(stereo_decoder)
        self.assertIsNotNone(arr_stereo)
        self.assertEqual(sr_stereo, 16000)
        self.assertEqual(arr_stereo.shape, (16000,))
        self.assertTrue(np.all(np.isfinite(arr_stereo)))

        # 3. Test legacy dict with 'array'
        dict_audio = {"array": np.zeros(8000, dtype=np.float32), "sampling_rate": 8000}
        arr_dict, sr_dict = extract_audio_and_sr(dict_audio)
        self.assertIsNotNone(arr_dict)
        self.assertEqual(sr_dict, 8000)
        self.assertEqual(arr_dict.shape, (8000,))

        # 4. Test None / invalid audio
        arr_none, sr_none = extract_audio_and_sr(None)
        self.assertIsNone(arr_none)
        self.assertIsNone(sr_none)


if __name__ == "__main__":
    unittest.main()
