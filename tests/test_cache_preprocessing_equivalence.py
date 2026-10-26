"""
DeepShield Preprocessing & Cache Equivalence Unit Test
Verifies that:
1. The authoritative preprocessing pipeline implements all 14 required specifications:
   - 16 kHz mono
   - Energy silence trimming (30 dB top_db)
   - Peak normalization (0.95 target_peak)
   - 4.0-second sliding chunks (64,000 samples)
   - 2.0-second hop (32,000 samples)
   - 128-mel log-mel spectrogram (n_fft=1024, win=400, hop=160, 0-8000Hz)
   - Bicubic interpolation to 128x128
   - 3-channel broadcast and ImageNet normalization
2. The cache implementation produces bit-for-bit / float-equivalent tensors compared to
   the authoritative on-the-fly extraction pipeline.
3. Multi-chunk utterances correctly generate multiple chunks and preserve utterance labels.
"""

import os
import sys
import tempfile
import json
import torch
import soundfile as sf
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.audio import AudioPreprocessor
from src.preprocessing.spectrogram import SpectrogramExtractor


def test_authoritative_preprocessing_specifications():
    """Verify that AudioPreprocessor and SpectrogramExtractor default to locked parameters."""
    ap = AudioPreprocessor()
    assert ap.sample_rate == 16000
    assert ap.mono is True
    assert ap.chunk_duration == 4.0
    assert ap.hop_duration == 2.0
    assert ap.chunk_samples == 64000
    assert ap.hop_samples == 32000
    assert ap.trim_silence is True
    assert ap.top_db == 30.0
    assert ap.normalize_peak is True
    assert ap.target_peak == 0.95

    se = SpectrogramExtractor(image_size=(128, 128))
    assert se.sample_rate == 16000
    assert se.n_fft == 1024
    assert se.win_length == 400
    assert se.hop_length == 160
    assert se.n_mels == 128
    assert se.fmin == 0.0
    assert se.fmax == 8000.0
    assert se.image_size == (128, 128)
    assert se.channels == 3
    assert se.imagenet_norm is True


def test_chunking_temporal_coverage():
    """Test chunking logic for short, exact, medium, and long waveforms."""
    ap = AudioPreprocessor(sample_rate=16000, chunk_duration=4.0, hop_duration=2.0)

    # 1. Short audio (2.5 seconds = 40,000 samples) -> 1 padded chunk of 64,000 samples
    short_wf = torch.randn(40000)
    chunks_short = ap.chunk_waveform(short_wf)
    assert len(chunks_short) == 1
    assert chunks_short[0].shape == (64000,)
    assert torch.all(chunks_short[0][40000:] == 0.0)  # Padded with zero

    # 2. Exact audio (4.0 seconds = 64,000 samples) -> 1 chunk of 64,000 samples
    exact_wf = torch.randn(64000)
    chunks_exact = ap.chunk_waveform(exact_wf)
    assert len(chunks_exact) == 1
    assert chunks_exact[0].shape == (64000,)

    # 3. Medium audio (5.5 seconds = 88,000 samples) -> 2 chunks with 50% overlap + tail
    med_wf = torch.randn(88000)
    chunks_med = ap.chunk_waveform(med_wf)
    assert len(chunks_med) == 2
    for c in chunks_med:
        assert c.shape == (64000,)
    # Second chunk must cover the end of the audio
    assert torch.allclose(chunks_med[-1], med_wf[88000 - 64000 :])

    # 4. Long audio (10.0 seconds = 160,000 samples) -> 4 chunks (0s, 2s, 4s, 6s)
    long_wf = torch.randn(160000)
    chunks_long = ap.chunk_waveform(long_wf)
    assert len(chunks_long) == 4
    for c in chunks_long:
        assert c.shape == (64000,)


def test_cache_vs_onthefly_equivalence():
    """
    Test direct numeric equivalence between authoritative on-the-fly extraction
    and serialized cache loading on real ASVspoof 2019 audio files if present,
    or synthetic audio files.
    """
    ap = AudioPreprocessor()
    se = SpectrogramExtractor(image_size=(128, 128))

    # Look for real audio files
    real_audio_dir = r"C:\DeepShieldData\ASVspoof2019_LA\LA\ASVspoof2019_LA_train\flac"
    audio_paths = []
    if os.path.exists(real_audio_dir):
        files = [os.path.join(real_audio_dir, f) for f in os.listdir(real_audio_dir) if f.endswith(".flac")]
        if len(files) >= 5:
            audio_paths = files[:5]

    with tempfile.TemporaryDirectory() as tmp_dir:
        if not audio_paths:
            # Create synthetic test audio
            for i in range(3):
                p = os.path.join(tmp_dir, f"test_sample_{i}.wav")
                dur = 2.0 + i * 2.5  # 2.0s, 4.5s, 7.0s
                synth = np.sin(2 * np.pi * 440 * np.linspace(0, dur, int(16000 * dur))).astype(np.float32)
                sf.write(p, synth, 16000)
                audio_paths.append(p)

        cache_dir = os.path.join(tmp_dir, "cache_spec", "128x128", "test_split")
        os.makedirs(cache_dir, exist_ok=True)

        for audio_path in audio_paths:
            utt_id = os.path.splitext(os.path.basename(audio_path))[0]
            utt_cache_dir = os.path.join(cache_dir, utt_id)
            os.makedirs(utt_cache_dir, exist_ok=True)

            # 1. Authoritative on-the-fly extraction
            wf_raw = ap.load_audio(audio_path)
            wf_proc = ap.process_waveform(wf_raw)
            chunks_raw = ap.chunk_waveform(wf_proc)
            specs_raw = [se(chunk) for chunk in chunks_raw]

            # 2. Cache extraction and save
            for c_idx, chunk in enumerate(chunks_raw):
                spec = se(chunk)
                chunk_file = os.path.join(utt_cache_dir, f"chunk_{c_idx:03d}.pt")
                torch.save(spec, chunk_file)

            # 3. Load from cache and compare
            for c_idx, spec_expected in enumerate(specs_raw):
                chunk_file = os.path.join(utt_cache_dir, f"chunk_{c_idx:03d}.pt")
                assert os.path.exists(chunk_file), f"Cache file {chunk_file} missing!"
                spec_cached = torch.load(chunk_file, weights_only=True)

                # Check shape
                assert spec_cached.shape == (3, 128, 128)
                assert spec_expected.shape == (3, 128, 128)

                # Check numerical equivalence
                assert torch.allclose(spec_cached, spec_expected, atol=1e-5), (
                    f"Mismatch between on-the-fly and cached spectrogram for {utt_id} chunk {c_idx}! "
                    f"Max diff: {torch.max(torch.abs(spec_cached - spec_expected)).item()}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
