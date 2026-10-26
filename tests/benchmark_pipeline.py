"""
Benchmark Data Loading and End-to-End Pipeline Performance
Measures:
A. CPU Spectrogram Generation throughput (samples/sec)
B. PyTorch DataLoader batching throughput
C. Model Inference throughput
D. Full Training Step (Forward + Loss + AMP Backward + Step)
Identifies the primary computational bottleneck (I/O vs Preprocessing vs GPU Compute).
"""

import sys
import os
import time
import shutil
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.audio import AudioPreprocessor
from src.preprocessing.spectrogram import SpectrogramExtractor
from src.models.efficientnet_model import EfficientNetDeepShield
from src.models.vit_model import VisionTransformerDeepShield


class SyntheticAudioDataset(Dataset):
    """Generates synthetic 4-second audio utterances on disk or in memory."""

    def __init__(self, root_dir: str, num_samples: int = 500, sample_rate: int = 16000):
        self.root_dir = root_dir
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.file_list = []
        self.labels = []
        
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        self.spec_extractor = SpectrogramExtractor(sample_rate=sample_rate, image_size=(128, 128))

        os.makedirs(root_dir, exist_ok=True)
        import soundfile as sf
        # Generate 500 fake audio files
        for i in range(num_samples):
            fpath = os.path.join(root_dir, f"syn_{i:04d}.wav")
            label = 0 if i % 2 == 0 else 1
            if not os.path.exists(fpath):
                # 4-second audio signal
                t = torch.linspace(0, 4.0, int(sample_rate * 4.0))
                freq = 200 + (i % 50) * 20
                waveform = (0.5 * torch.sin(2 * torch.pi * freq * t)).numpy()
                sf.write(fpath, waveform, sample_rate)
            self.file_list.append(fpath)
            self.labels.append(label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        fpath = self.file_list[idx]
        label = self.labels[idx]

        # Load audio & process
        waveform = self.preprocessor.load_audio(fpath)
        processed_wf = self.preprocessor.process_waveform(waveform)
        chunks = self.preprocessor.chunk_waveform(processed_wf)
        chunk = chunks[0]

        # Convert to 128x128 spectrogram
        spec = self.spec_extractor(chunk)
        return spec, torch.tensor(label, dtype=torch.long)


def run_pipeline_benchmark(num_samples: int = 200, batch_size: int = 32):
    print("=" * 60)
    print(f"RUNNING PIPELINE PERFORMANCE BENCHMARK ({num_samples} samples, BS={batch_size})")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp(prefix="deepshield_bench_")
    try:
        # 1. Create dataset
        print(f"Creating synthetic audio files in temporary location: {temp_dir}")
        t0 = time.perf_counter()
        dataset = SyntheticAudioDataset(root_dir=temp_dir, num_samples=num_samples)
        t_creation = time.perf_counter() - t0
        print(f"Dataset generated in {t_creation:.2f}s.")

        # Benchmark A: Standalone Spectrogram Generation on CPU
        print("\n--- Benchmark A: CPU Spectrogram Extraction Throughput ---")
        preprocessor = AudioPreprocessor()
        spec_extractor = SpectrogramExtractor(image_size=(128, 128))
        dummy_audio = torch.randn(64000)
        
        t0 = time.perf_counter()
        iters = 100
        for _ in range(iters):
            wf = preprocessor.process_waveform(dummy_audio)
            chunk = preprocessor.chunk_waveform(wf)[0]
            _ = spec_extractor(chunk)
        t_spec = time.perf_counter() - t0
        spec_throughput = iters / t_spec
        print(f"CPU Spectrogram Throughput: {spec_throughput:.1f} spectrograms/sec ({1000/spec_throughput:.2f} ms/sample)")

        # Benchmark B: PyTorch DataLoader Batch Throughput
        print("\n--- Benchmark B: DataLoader Pipeline Throughput ---")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        t0 = time.perf_counter()
        total_batches = 0
        total_items = 0
        for x, y in loader:
            total_batches += 1
            total_items += x.shape[0]
        t_loader = time.perf_counter() - t0
        loader_throughput = total_items / t_loader
        print(f"DataLoader (Audio Load + Spec Extract): {loader_throughput:.1f} samples/sec ({t_loader:.2f}s for {total_items} items)")

        # Benchmark C & D: Full End-to-End Training Loop Step
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n--- Benchmark C & D: Full Training Step on {device} ---")
        
        for name, model_cls, kwargs in [
            ("EfficientNet-B0", EfficientNetDeepShield, {"num_classes": 2, "pretrained": True}),
            ("ViT-B/16 (128x128)", VisionTransformerDeepShield, {"model_name": "vit_base_patch16_224", "num_classes": 2, "pretrained": True, "img_size": (128, 128)})
        ]:
            model = model_cls(**kwargs).to(device)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            step_count = 0
            
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    with torch.amp.autocast("cuda"):
                        out = model(x)
                        loss = criterion(out, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
                step_count += 1

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_train = time.perf_counter() - t0
            train_throughput = total_items / t_train
            print(f"{name} Full End-to-End Training Epoch: {train_throughput:.1f} samples/sec ({t_train:.2f}s total)")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("\nCleaned up synthetic benchmarking files.")


if __name__ == "__main__":
    run_pipeline_benchmark()
