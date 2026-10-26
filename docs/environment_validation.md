# DeepShield — Environment & Hardware Validation Report

**Date:** October 5, 2026  
**Status:** VALIDATED & FROZEN  
**Hardware Platform:** NVIDIA GeForce RTX 5050 Laptop GPU (8,151 MiB / 8 GB VRAM)  
**Host Architecture:** Blackwell (sm_120), Compute Capability (12, 0)

---

## 1. Environment & CUDA Verification

| Component | Measured Value | Verification Status |
| :--- | :--- | :--- |
| **Python Version** | 3.12.7 (64-bit AMD64) | PASSED |
| **PyTorch Version** | 2.14.1+cu130 | PASSED |
| **Torchvision Version** | 0.29.1+cu130 | PASSED |
| **Torchaudio Version** | 2.11.0+cu130 | PASSED |
| **PyTorch Reported CUDA** | 13.0 | PASSED |
| **torch.cuda.is_available()** | True | PASSED |
| **GPU Device Name** | NVIDIA GeForce RTX 5050 Laptop GPU | PASSED |
| **CUDA Device Count** | 1 | PASSED |
| **CUDA Device Capability** | (12, 0) (Blackwell sm_120) | PASSED |
| **NVIDIA Driver Version** | 592.82 (CUDA 13.1 driver) | PASSED |
| **CUDA Tensor Matmul** | x = torch.randn(10,10,device=\'cuda\'); y = x @ x | PASSED (Device: cuda:0) |

---

## 2. Unit Test Suite Results

All tests executed with zero failures:

1. **tests/test_spectrogram_shape.py** — **PASSED**
   - Input: 64,000 audio samples (4.0s @ 16 kHz)
   - Extraction: Log-mel filterbanks (128 mel bins, 1024 FFT, 160 hop, 400 win, 0-8000 Hz)
   - Output Shape: torch.Size([3, 128, 128])
   - Numeric Stability: All finite (min=-2.1179, max=2.6400, standard ImageNet normalized)

2. **tests/test_vit_128.py** — **PASSED**
   - ViT Model Backbone: vit_base_patch16_224
   - Total Parameters: **85,698,818 (85.70 M)**
   - Trainable Parameters: **85,698,818 (85.70 M)**
   - Input Tensor: torch.Size([2, 3, 128, 128])
   - Patch Size: 16 x 16
   - Spatial Grid: 8 x 8 = 64 patches
   - Visual Tokens: **65 tokens** (64 patch tokens + 1 [CLS] token)
   - Positional Embedding Interpolation: **Verified** (2D bicubic interpolation from 14x14 grid down to 8x8 grid, preserving [CLS])
   - Pretrained Weights: Loaded successfully (ViT_B_16_Weights.IMAGENET1K_V1)
   - Output Shape: torch.Size([2, 2])

3. **tests/test_model_parity.py** — **PASSED**
   - Identical input [4, 3, 128, 128] fed to both models.
   - EfficientNet-B0 Output: torch.Size([4, 2])
   - ViT-B/16 Output: torch.Size([4, 2])
   - Both models consume the exact same spectrogram tensor with zero mismatch.

4. **tests/test_dataset.py** — **PASSED**
   - Dataset abstractions for ASVspoof 2019 LA and ASVspoof 2021 DF verified.
   - Inverse-frequency class weighting function operational.

---

## 3. Empirical Latency & Memory Benchmarks

Measured on NVIDIA GeForce RTX 5050 (8 GB VRAM) using tests/benchmark_amp.py (30 batches after 10 warmup passes):

| Model | Batch Size | Input Resolution | AMP Enabled | GPU VRAM Allocated (MB) | GPU VRAM Reserved (MB) | Inference Time (ms/batch) | Throughput (samples/sec) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | 8 | 128x128 | **True** | 49.0 | 74.0 | 19.92 | 401.6 |
| **EfficientNet-B0** | 16 | 128x128 | **True** | 50.5 | 104.0 | 18.56 | 862.0 |
| **EfficientNet-B0** | 32 | 128x128 | **True** | 53.5 | 162.0 | 20.11 | **1,591.5** |
| **EfficientNet-B0** | 8 | 128x128 | **False** | 49.0 | 94.0 | 13.95 | 573.5 |
| **EfficientNet-B0** | 16 | 128x128 | **False** | 50.5 | 138.0 | 13.02 | 1,229.2 |
| **EfficientNet-B0** | 32 | 128x128 | **False** | 53.5 | 214.0 | 12.69 | **2,521.4** |
| **ViT-B/16** | 8 | 128x128 | **True** | 360.9 | 596.0 | 12.56 | 636.8 |
| **ViT-B/16** | 16 | 128x128 | **True** | 361.9 | 594.0 | 14.88 | 1,075.5 |
| **ViT-B/16** | 32 | 128x128 | **True** | 365.4 | 642.0 | 28.83 | **1,110.1** |
| **ViT-B/16** | 8 | 128x128 | **False** | 360.9 | 414.0 | 20.36 | 393.0 |
| **ViT-B/16** | 16 | 128x128 | **False** | 361.9 | 462.0 | 38.15 | 419.4 |
| **ViT-B/16** | 32 | 128x128 | **False** | 365.4 | 522.0 | 77.15 | 414.8 |

### Key Benchmark Observations:
1. **ViT Speedup with AMP:** On Blackwell Tensor Cores, ViT-B/16 with AMP achieves **1,110 samples/sec** at Batch Size 32 (2.67x faster than FP32 414.8 samples/sec).
2. **VRAM Footprint:** ViT-B/16 allocates only **365.4 MB** of VRAM (Peak Reserved: 642.0 MB), easily fitting within the 8 GB budget and leaving over 7 GB for gradient activations during training.
3. **Sequence Length Reduction:** Interpolating from 224x224 (197 tokens) to 128x128 (65 tokens) yields a theoretical (197/65)^2 ≈ 9.2x reduction in self-attention FLOPs.

---

## 4. End-to-End Pipeline & DataLoader Throughput

Measured using tests/benchmark_pipeline.py (200 synthetic 4.0s utterances):

- **CPU Spectrogram Generation Throughput:** **357.5 spectrograms/sec** (2.80 ms/sample).
- **PyTorch DataLoader Throughput (Audio Load + STFT + Mel Filterbank):** **94.2 samples/sec** (2.12 s for 200 items).
- **Full End-to-End Training Epoch (DataLoader + GPU Transfer + Forward + AMP Backward + AdamW Step):**
  - **EfficientNet-B0:** 25.8 samples/sec
  - **ViT-B/16 (128x128):** 135.6 samples/sec

---

## 5. Frozen Dependencies

Dependencies are locked in requirements-lock.txt (71 packages):
- torch==2.14.1+cu130
- torchvision==0.29.1+cu130
- torchaudio==2.11.0+cu130
- timm==1.0.30
- librosa==1.0.0
- soundfile==0.14.0
- scikit-learn==1.9.1
- pandas==3.0.6
- Flask==3.1.3
- pytest==9.1.1
