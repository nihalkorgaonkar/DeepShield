# DeepShield — Speech Deepfake & Voice Spoofing Detection Research Platform

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14.2-black.svg)](https://nextjs.org/)
[![Research Guardrails](https://img.shields.io/badge/Academic%20Integrity-Zero%20Fabrication-success.svg)](GEMINI.md)

DeepShield is a cybersecurity research platform providing empirical comparison and real-time inference between a **Convolutional Neural Network baseline (EfficientNet-B0)** and a **Vision Transformer (ViT-B/16)** on the ASVspoof 2019 Logical Access (LA) and ASVspoof 2021 Deepfake (DF) benchmarks under strict academic integrity protocols.

---

## 1. System Architecture

```
                                      ┌─────────────────────────────────────────────────────────┐
                                      │              Next.js 14 Web Dashboard                   │
                                      │   (TypeScript, Tailwind CSS, Recharts, Lucide Icons)    │
                                      └────────────────────────────┬────────────────────────────┘
                                                                   │ HTTP / JSON / Multipart
                                                                   ▼
                                      ┌─────────────────────────────────────────────────────────┐
                                      │               FastAPI Python Backend                    │
                                      │              (Uvicorn on Port 8000)                     │
                                      └────────────────────────────┬────────────────────────────┘
                                                                   │
                                ┌──────────────────────────────────┴──────────────────────────────────┐
                                ▼                                                                     ▼
    ┌───────────────────────────────────────────────┐                 ┌───────────────────────────────────────────────┐
    │       Locked 14-Point Audio Preprocessor      │                 │            PyTorch Neural Backbones           │
    │  • 16 kHz mono resampling                     │                 │  • EfficientNet-B0 (4.01M params)             │
    │  • 4.0s sliding window (2.0s hop = 50% overlap│                 │  • ViT-B/16 with 2D Bicubic Pos-Embed (85.8M) │
    │  • 30 dB silence trimming, 0.95 peak norm     │                 │  • CUDA RTX 5050 FP16 AMP / CPU fallback      │
    │  • 128 × 128 log-mel ImageNet 3-channel spec  │                 │  • Chunk-probability arithmetic mean pooling  │
    └───────────────────────────────────────────────┘                 └───────────────────────────────────────────────┘
```

---

## 2. Model Specifications & Cryptographic Provenance

All model weights are verified using cryptographic SHA-256 hashes to guarantee provenance and reproducibility:

| Model Identity | Family | Parameters | Input Resolution | Frozen Dev Threshold ($\tau$) | Checkpoint Path | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `efficientnet_b0` | CNN Baseline | 4.01 Million | $128 \times 128 \times 3$ | `0.967773` | `models/checkpoints/best_efficientnet_b0.pt` | `1bd3841c7f0568e7d62783df4015dd0f25656366889cdf86ad34c40d6f211b69` |
| `vit_b16` | Vision Transformer | 85.8 Million | $128 \times 128 \times 3$ | `0.986328` | `models/checkpoints/best_vit_b16.pt` | `14dfd3451c24499d17e705b1a0bc40fa7c8c1f6ef880e6c4bba2bd4570e799d0` |

### Benchmark Results (ASVspoof 2019 LA Official Partition)

| Evaluation Metric | CNN Baseline (EfficientNet-B0) | Vision Transformer (ViT-B/16) | Protocol Standard |
| :--- | :--- | :--- | :--- |
| **Development EER** | **6.44%** | **1.55%** | Calibrated on known vocoders (A01–A06) |
| **Evaluation EER** | **21.49%** | **27.08%** | Tested on 13 unseen vocoders (A07–A19) |
| **ROC-AUC** | **0.8682** | **0.8201** | Threshold-independent discrimination |
| **Balanced Accuracy** | **73.58%** | **72.15%** | Evaluated strictly at frozen dev $\tau$ |
| **Macro F1 Score** | **0.4739** | **0.4693** | Unweighted harmonic mean of precision/recall |
| **Inference Latency** | **1.22 ms** / utterance | **1.18 ms** / utterance | NVIDIA RTX 5050 (AMP autocast) |

> **Critical Research Alert:** Both architectures exhibit a 5-orders-of-magnitude threshold shift between development ($\tau \approx 0.97 - 0.99$) and evaluation operating points ($\tau \approx 10^{-6} - 10^{-5}$). In accordance with the **DeepShield Zero Fabrication Directive**, thresholds are frozen on development data and NEVER tuned against official evaluation splits.

---

## 3. Prerequisites

- **Operating System:** Windows 10/11 (64-bit)
- **Python:** Python 3.12.x (installed in local `.venv`)
- **Node.js:** Node.js v18.0.0+ (Tested on v25.0.0, npm 11.6.2)
- **GPU (Recommended):** NVIDIA GPU with CUDA 12.x / 13.x (CPU fallback supported automatically)

---

## 4. Quick Startup Instructions

### Step 1: Start the FastAPI Python Backend

Open a PowerShell terminal in the repository root:

```powershell
# Option A: Using launcher script
.\run_backend.ps1

# Option B: Manual uvicorn launch
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

The backend server will start on: **`http://127.0.0.1:8000`** (Interactive OpenAPI docs at **`http://127.0.0.1:8000/docs`**).

### Step 2: Start the Next.js Frontend Dashboard

Open a **second** PowerShell terminal in the repository root:

```powershell
# Option A: Using launcher script
.\run_frontend.ps1

# Option B: Manual npm launch
cd frontend
cmd /c "npm run dev"
```

The web dashboard will start on: **`http://localhost:3000`**.

### Option C: 1-Click Dual Launch

To launch both backend and frontend simultaneously in separate PowerShell windows:

```powershell
.\run_all.ps1
```

---

## 5. Step-by-Step Walkthrough: Verifying Live Model Inference

1. Open **`http://localhost:3000`** in your browser.
2. Ensure the top status bar displays **`Backend API: Online`** and **`Compute Device: CUDA`** (or `CPU`).
3. Click **Audio Detection** in the left sidebar navigation.
4. You have two testing options:
   - **Option A (Instant 1-Click Verification):** Click either **"Load Genuine Speech"** or **"Load Synthetic Speech (TTS A01)"** in the top sample bar.
   - **Option B (Custom Upload):** Drag-and-drop any `.wav`, `.flac`, or `.mp3` recording into the upload zone.
5. Review the audio waveform envelope and metadata (duration, sample rate, channels, RMS energy).
6. Select **"Dual-Model Comparison"** to execute both EfficientNet-B0 and ViT-B/16 side-by-side.
7. Click **"Run Dual Comparative Inference"**.
8. Observe the actual results returned in real time:
   - Individual model predictions (`GENUINE SPEECH` vs `SYNTHETIC / SPOOF SPEECH`)
   - Spoof posterior probabilities ($P_{\text{spoof}}$) and genuine probabilities
   - Frozen decision thresholds ($\tau_{\text{eff}} = 0.967773$, $\tau_{\text{vit}} = 0.986328$)
   - Consensus status: `MODELS IN AGREEMENT` or `MODEL DISAGREEMENT DETECTED`
   - Real inference latency in milliseconds (typically 20–150 ms depending on audio length)
   - Temporal chunk decomposition breakdown across 4.0s sliding windows

---

## 6. Dashboard Features & Navigation Sections

| Section | Description |
| :--- | :--- |
| **1. Overview** | Research executive summary, live GPU telemetry, model capacity comparison chart, and threshold shift alerts. |
| **2. Audio Detection** | Drag-and-drop audio detector, waveform visualization, dual-model comparison, and chunk breakdown. |
| **3. Model Comparison** | Architectural comparison, parameter counts, side-by-side confusion matrices, and ROC-AUC benchmarks. |
| **4. Evaluation Lab** | Execute reproducible diagnostic evaluation runs on ASVspoof 2019 LA with live progress tracking and JSON export. |
| **5. Robustness Testing** | Stress-test audio against additive Gaussian noise (20dB, 10dB, 5dB), volume variations ($\pm 6$dB), and telephony 8 kHz filtering. |
| **6. Explainability** | Visual acoustic heatmaps comparing CNN Grad-CAM formants (0–2 kHz) against ViT self-attention rollout maps. |
| **7. Dataset Explorer** | Local filesystem inventory, train/dev/eval sample counts, attack algorithms, and preprocessed cache health. |
| **8. System Diagnostics** | Telemetry health checks, CUDA driver, allocated VRAM, and a live "Test Model Forward Pass" latency probe. |
| **9. Research Reports** | Interactive markdown document reader for `final_report.md`, `final_audit.md`, and provenance logs. |
| **10. Settings** | Review locked 14-point preprocessing invariants and configure default detection preferences. |

---

## 7. Running Automated Tests

DeepShield includes a comprehensive automated test suite verifying model loading, API response schemas, and real inference:

```powershell
# Run full backend pytest suite
.\.venv\Scripts\pytest.exe tests/test_fastapi_backend.py -v

# Run metric conventions test
.\.venv\Scripts\pytest.exe tests/test_metric_conventions.py -v

# Run preprocessing parity test
.\.venv\Scripts\pytest.exe tests/test_preprocessing_pipeline.py -v
```

All 10 integration tests in `test_fastapi_backend.py` run against real model weights on CUDA/CPU and pass with exit code 0.

---

## 8. Troubleshooting Guide

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **`npm.ps1 cannot be loaded because running scripts is disabled`** | Windows PowerShell script execution policy blocks `.ps1` files. | Run commands via `cmd /c "npm run dev"` or use the provided `run_frontend.ps1` launcher. |
| **`CUDA out of memory`** | Large audio files with many chunks evaluated concurrently. | The backend automatically limits batch size and employs AMP FP16 autocast. If running other GPU tasks, close them first. |
| **`Backend connection failed`** | FastAPI server is not running on port 8000. | Verify that `run_backend.ps1` is active and navigating to `http://127.0.0.1:8000/api/health` returns status 200. |
| **`ASVspoof 2021 DF: DOCUMENTED_MISSING_AUDIO`** | Raw FLAC evaluation audio is not downloaded locally. | This is an intentional research integrity state documented in `results/generalization_2021_df_status.json`. In strict compliance with the Zero Fabrication Directive, no synthetic results are manufactured. |

---

## 9. Academic Integrity & Research Guardrails

DeepShield enforces three immutable research rules:
1. **Zero Fabrication Directive:** Metrics, predictions, and hashes originate strictly from unedited execution logs.
2. **Evaluation Partition Isolation:** The evaluation split is accessed solely in read-only inference mode; thresholds are never recalibrated on evaluation data.
3. **Artifact Immutability:** Pre-trained checkpoints and frozen thresholds in `results/thresholds.json` are cryptographically audited and never overwritten implicitly.
