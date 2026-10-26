# DeepShield Project Audit & Baseline Methodology Review

**Project Name:** DeepShield — Vision Transformer Based Speech Deepfake Detection  
**Author / Role:** Lead ML Engineer & Research Assistant  
**Date:** October 2026  
**Status:** Phase 0 Audit & Gap Analysis Completed  

---

## 1. Executive Summary

The DeepShield project was originally designed as an audio deepfake detection system leveraging log-mel spectrogram representations and CNN backbones (EfficientNet-B0), evaluated on the ASVspoof 2019 Logical Access (LA) dataset, served via a Flask REST API, and visualized through a React interface.

Following faculty guidance, the project is expanding into a comprehensive comparative research study investigating whether **Vision Transformers (ViT)** operating on spectrogram representations provide improved detection accuracy, better cross-attack generalization, and enhanced robustness over CNNs (EfficientNet-B0), evaluated on both **ASVspoof 2019 LA** and the **ASVspoof 2021 Deepfake (DF)** benchmark.

This audit evaluates the current repository state, details the baseline specifications from the original DeepShield paper, identifies critical methodological inconsistencies and pitfalls, and outlines the system requirements and technical roadmap.

---

## 2. Baseline Architecture & Methodology Review

Based on the original DeepShield paper and specification:

### 2.1 Audio Preprocessing Pipeline
| Parameter | Paper Specification | DeepShield Standard |
|---|---|---|
| **Sampling Rate** | 16,000 Hz (16 kHz) | 16 kHz Mono |
| **Chunking Strategy** | 4.0-second sliding windows | 64,000 samples / chunk |
| **Window Hop** | 2.0-second overlap | 32,000 samples hop |
| **Silence Handling** | Silence trimming / thresholding | Energy-based voice activity / top DB trimming |
| **Peak Normalization** | Waveform peak normalization to $[-1.0, 1.0]$ | Standardized scaling |
| **STFT Window Size** | 400 samples (25 ms) | Hann window |
| **STFT Hop Length** | 160 samples (10 ms) | 160 samples |
| **FFT Points ($N_{fft}$)** | 1024 | 1024 points |
| **Mel Filter Banks ($N_{mels}$)** | 128 | 128 mel frequency bins |
| **Frequency Range** | 0 Hz – 8000 Hz ($f_{min}=0, f_{max}=8000$) | Nyquist frequency limit |
| **Representation** | Log-mel spectrogram ($\log(1 + 10000 \cdot \text{Mel})$ or $\text{dB}$) | Power to dB conversion |
| **Tensor Normalization** | ImageNet mean $([0.485, 0.456, 0.406])$ & std $([0.229, 0.224, 0.225])$ | 3-channel RGB mapped |

### 2.2 Model Architecture: Baseline (EfficientNet-B0)
- **Backbone:** ImageNet-1k pretrained EfficientNet-B0 (`torchvision.models.efficientnet_b0` or `timm`).
- **Feature Extractor:** MBConv blocks extracting 1280-dimensional feature representations.
- **Classification Head:** Global Average Pooling $\rightarrow$ Dropout ($p=0.2$) $\rightarrow$ Linear Classifier (`Linear(1280, 2)`).
- **Target Classes:** Binary Classification:
  - `0`: Bonafide (Genuine human speech)
  - `1`: Spoof (Synthesized / Voice-converted / Deepfake speech)
- **Inference Aggregation:** For multi-chunk utterances, probability score averaging across 4-second chunks with threshold application.

### 2.3 Training & Optimization Protocol
- **Loss Function:** Weighted Cross-Entropy Loss to account for class imbalance (in ASVspoof 2019 LA, bonafide:spoof ratio is roughly $1:9$).
- **Optimizer:** AdamW with initial learning rate $\eta = 1\times 10^{-4}$ and weight decay $1\times 10^{-2}$.
- **Learning Rate Scheduler:** Cosine Annealing / ReduceLROnPlateau with patience.
- **Validation:** Validation loss and EER / F1 monitoring with Early Stopping (patience: 5–7 epochs).
- **Threshold Optimization:** Optimal decision threshold $\tau^*$ computed on the **Development (Dev)** split to minimize classification error or equalize False Accept / False Reject rates, strictly frozen prior to Evaluation split testing.

---

## 3. Audit of Existing Repository State

### 3.1 Directory Scan Results
Inspection of the current workspace (`c:\Users\parth\OneDrive\Desktop\DeepShield`) indicates:
- **Repository State:** Clean / uninitialized workspace.
- **Existing Source Files:** None present in the active folder.
- **Existing Models / Checkpoints:** No saved checkpoint files (`.pt`, `.pth`, `.onnx`).
- **Virtual Environment:** Python 3.12.7 detected globally; PyTorch and scientific audio packages (`torchaudio`, `librosa`, `timm`) are not yet installed in a dedicated environment.
- **Hardware Acceleration:** NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM, Driver 592.82, CUDA 13.1 support) is available on the host machine.
- **Storage Availability:** 285.99 GB free disk space on `C:\`.

---

## 4. Methodological Gap Analysis & Critical Pitfalls

To transform DeepShield into an academic-grade research project, the following common methodological pitfalls must be avoided:

### 4.1 Inconsistency in Spectrogram Input Resolution
- **Issue:** A 4.0-second audio chunk at 16 kHz with `hop_length=160` generates an STFT matrix of $128 \times 401$ time-frequency bins. The original paper mentions $128\times 128$ spectrogram images.
- **Risk:** Arbitrarily squashing $401$ temporal frames down to $128$ introduces non-linear time compression and loses subtle temporal phase and modulation artifacts crucial for deepfake detection.
- **Standardized Solution:** Use an explicit, fixed spatial resizing pipeline (e.g., $224 \times 224$ via bicubic interpolation or anti-aliased resizing) that aligns with the native patch grid ($14\times 14$ patches for $16\times 16$ patch size) of standard Vision Transformers (ViT-B/16) and EfficientNet-B0 input resolutions. **Crucially, the exact same resolution and interpolation must be fed to both EfficientNet-B0 and ViT.**

### 4.2 Data Leakage & Split Violations
- **Issue:** Random $k$-fold cross-validation or merging train/dev/eval splits leaks speaker characteristics and synthetic speech generation algorithms (vocoders/acoustic models) between train and test sets.
- **Rule:** Strict adherence to official ASVspoof protocol files:
  - **ASVspoof 2019 LA:** Train (A01–A06), Dev (A01–A06), Eval (A07–A19: 13 unseen attacks).
  - **ASVspoof 2021 DF:** Evaluation only on unseen compression codecs, acoustic channels, and advanced neural vocoders.
  - Decision threshold $\tau^*$ must be chosen **solely on the Dev split** and never on the Eval split.

### 4.3 Evaluation Metric Rigor
- Accuracy is misleading in ASVspoof due to severe class imbalance (e.g. 90% spoof).
- Must compute and report:
  - **Equal Error Rate (EER)** — primary biometrics/countermeasure metric.
  - **Balanced Accuracy (BAcc)** — $\frac{1}{2}(\text{TPR} + \text{TNR})$.
  - **Precision, Recall, F1-Score** (for both bonafide and spoof classes).
  - **ROC-AUC** and **PR-AUC**.
  - **Confusion Matrix** with explicit False Positive Rate (FPR) and False Negative Rate (FNR).

### 4.4 Interpretability and Explainability
- EfficientNet requires **Grad-CAM** operating on the final convolutional layer (`conv_head` / `features[8]`).
- Vision Transformers require **Transformer Attention Rollout** or **Attention Attribution** (extracting self-attention weights across the $12$ multi-head attention layers and mapping the `[CLS]` token attention back to the $14\times 14$ patch grid).
- Maps must be clearly documented as visual attribution aids, avoiding unsubstantiated claims of causal explanation.

---

## 5. Missing Components Inventory

| Component | Status | Required Implementation |
|---|---|---|
| **Python Environment & Dependencies** | Missing | `requirements.txt` with `torch`, `torchvision`, `torchaudio`, `timm`, `librosa`, `scikit-learn`, `flask`, `flask-cors`, `pytest`, `pyyaml`, `matplotlib`, `seaborn` |
| **Config System** | Missing | `configs/preprocessing.yaml`, `configs/experiment.yaml`, `configs/vit.yaml`, `configs/efficientnet.yaml` |
| **Audio & Spectrogram Preprocessing** | Missing | `src/preprocessing/audio.py`, `src/preprocessing/spectrogram.py` (strict parity module) |
| **Dataset & Protocol Loaders** | Missing | `src/data/asvspoof2019.py`, `src/data/asvspoof2021.py`, `src/data/dataset.py` with dynamic tensor generation & smoke-test subsampling (`--fraction`) |
| **EfficientNet-B0 Baseline** | Missing | `src/models/efficientnet_model.py`, `src/training/train_efficientnet.py` |
| **Vision Transformer (ViT-B/16)** | Missing | `src/models/vit_model.py`, `src/training/train_vit.py` |
| **Evaluation & Metrics Suite** | Missing | `src/evaluation/evaluate.py`, `src/evaluation/metrics.py` (EER, ROC, PR, Confusion Matrix) |
| **Threshold Optimization** | Missing | `src/evaluation/threshold.py` (Dev-only optimization $\rightarrow$ `results/thresholds.json`) |
| **Explainability Suite** | Missing | `src/explainability/gradcam.py` (CNN) and `src/explainability/vit_attention.py` (ViT) |
| **Robustness & Ablation Suite** | Missing | `src/evaluation/robustness.py` (noise, SNR, codecs, volume perturbation) |
| **Backend REST API** | Missing | `src/api/app.py` (Flask `/api/predict` with dual model selection & chunk aggregation) |
| **Frontend UI** | Missing | Modern React interface with waveform, spectrogram, real-time audio recording, confidence gauge, explainability overlay, and model comparison dashboard |
| **Automated Unit Tests** | Missing | `tests/test_preprocessing.py`, `tests/test_models.py`, `tests/test_dataset.py`, `tests/test_api.py` |
| **Research Documentation** | Missing | `docs/experiment_plan.md`, `docs/dataset_protocol.md`, `docs/asvspoof2021_evaluation.md`, `docs/asvspoof5_feasibility.md`, `docs/research_summary.md` |

---

## 6. System Resource & Hardware Feasibility

### 6.1 Storage Requirements
- **ASVspoof 2019 LA:**
  - `LA.zip` (Train + Dev + Eval flac audio + protocol labels): $\sim 6.5 \text{ GB}$ (uncompressed: $\sim 15 \text{ GB}$).
- **ASVspoof 2021 DF:**
  - Evaluation subset / full keys: Full audio archive is $\sim 45 \text{ GB}$.
  - Storage feasibility: $285.99\text{ GB}$ available space is more than sufficient.
- **Checkpoint & Artifact Storage:**
  - EfficientNet-B0 weights: $\sim 20\text{ MB}$ per checkpoint.
  - ViT-B/16 weights: $\sim 330\text{ MB}$ per checkpoint.
  - Total model checkpoints + evaluation caches: $< 2\text{ GB}$.

### 6.2 GPU Compute Feasibility
- **Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU with 8,151 MiB (8 GB) VRAM.
- **EfficientNet-B0 ($5.3\text{M}$ params):**
  - VRAM consumption with batch size 32 at $224\times 224$: $\sim 1.2\text{ GB}$.
  - Training speed: $\sim 40\text{--}60\text{ sec/epoch}$ on full ASVspoof 2019 Train split.
- **ViT-B/16 ($86.6\text{M}$ params):**
  - VRAM consumption with batch size 32 at $224\times 224$ (mixed precision FP16/AMP): $\sim 3.8\text{--}4.5\text{ GB}$.
  - Training speed: $\sim 2.5\text{--}4.0\text{ min/epoch}$ on full ASVspoof 2019 Train split.
  - Fits comfortably within the 8 GB VRAM budget.

---

## 7. Recommended Modifications & Implementation Roadmap

1. **Establish Research Documentation:**
   - Create `docs/experiment_plan.md` (Formal plan for Experiments A, B, C, D).
   - Create `docs/dataset_protocol.md` (Official protocol specifications and partition guidelines).
   - Create `docs/asvspoof5_feasibility.md` (ASVspoof 5 feasibility and compute analysis).
2. **Environment & Dependency Setup:**
   - Create `requirements.txt` and establish PyTorch + CUDA virtual environment.
3. **Core Modules (Strict Preprocessing Parity):**
   - Implement `src/preprocessing/audio.py` and `src/preprocessing/spectrogram.py`.
   - Implement `src/data/` with on-the-fly spectrogram generation and fast `--fraction` smoke testing.
   - Implement `src/models/` for EfficientNet-B0 and ViT-B/16.
4. **Execution & Evaluation:**
   - Implement training pipelines with early stopping, checkpointing, and dev threshold tuning.
   - Implement evaluation suite with EER, AUC, F1, and confusion matrix export.
   - Implement explainability and robustness modules.
   - Build Flask backend and React frontend.
