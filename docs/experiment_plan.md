# DeepShield Research Experiment Plan

**Project:** DeepShield — Vision Transformer vs. CNN for Speech Deepfake Detection  
**Document Version:** 1.0  
**Status:** Approved for Implementation (Subject to Pre-Training Sign-off)  

---

## 1. Research Motivation & Core Questions

Deepfake audio generation has rapidly evolved through neural acoustic models, diffusion synthesizers, and neural vocoders (e.g., HiFi-GAN, WaveGlow, BigVGAN). Traditional deepfake countermeasures rely heavily on Convolutional Neural Networks (CNNs) like EfficientNet-B0 or ResNet architectures applied to time-frequency representations (log-mel spectrograms).

While CNNs capture local time-frequency spectral cues through localized receptive fields, they lack explicit global temporal-spectral attention. **Vision Transformers (ViTs)** utilize self-attention to model long-range relationships across time frames and frequency sub-bands simultaneously.

### 1.1 Primary Research Question
> *"Can a Vision Transformer operating on the identical $128 \times 128$ log-mel spectrogram representation provide improved deepfake speech detection and/or superior generalization to unseen spoofing attacks compared with the EfficientNet-B0 baseline?"*

### 1.2 Secondary Research Questions
1. **Cross-Dataset Generalization:** How does detection performance degrade when moving from controlled conditions (ASVspoof 2019 LA) to noisy, multi-codec transmission environments (ASVspoof 2021 DF)?
2. **Failure Mode Analysis:** Do ViTs exhibit lower False Positive Rates (FPR) on bona fide speech or lower False Negative Rates (FNR) on advanced neural vocoders?
3. **Computational Trade-off:** What is the penalty in parameters, FLOPs, GPU memory, and CPU inference latency when adopting ViT over EfficientNet-B0?

---

## 2. Dataset Clarification (2019 vs 2021 vs 2020/2022 vs ASVspoof 5)

| Edition | Official Name | Release Year | Role in DeepShield Experiment |
|---|---|---|---|
| **ASVspoof 2019** | ASVspoof 2019 Logical Access (LA) | 2019 | **Core Controlled Benchmark** (Train, Dev, Eval with 13 unseen attacks A07–A19). |
| **ASVspoof 2020/2022** | *Non-existent as separate official challenges* | N/A | **Clarification:** The official challenges were held in 2015, 2017, 2019, 2021, and 2024. Faculty references to 2020–2022 refer to the 2021 benchmark period. |
| **ASVspoof 2021** | ASVspoof 2021 Deepfake (DF) | 2021 | **Generalization Evaluation Benchmark** (Evaluates models trained on 2019 LA against diverse lossy compression codecs). |
| **ASVspoof 5** | ASVspoof 5 (Track 1 & Track 2) | 2024 | **Future Work / Large-Scale Benchmark** (Detailed in feasibility study). |

---

## 3. Formal Experiment Matrix

To ensure absolute academic rigor, all primary experiments (**A, B, C, D**) share the **exact same $128 \times 128$ log-mel spectrogram input and identical audio preprocessing pipeline**.

| Experiment ID | Dataset Split Protocol | Input Representation | Model Architecture | Primary Objective |
|---|---|---|---|---|
| **Experiment A** | ASVspoof 2019 LA (Train/Dev $\rightarrow$ Eval) | $128 \times 128$ log-mel | **EfficientNet-B0** | **Original DeepShield Baseline** |
| **Experiment B** | ASVspoof 2019 LA (Train/Dev $\rightarrow$ Eval) | $128 \times 128$ log-mel | **Vision Transformer (ViT)** | **Faculty-Suggested Comparison** (Same data & representation) |
| **Experiment C** | Train: 2019 LA (Train/Dev) $\rightarrow$ Eval: 2021 DF | $128 \times 128$ log-mel | **EfficientNet-B0** | **Generalization Baseline** (Multi-codec robustness) |
| **Experiment D** | Train: 2019 LA (Train/Dev) $\rightarrow$ Eval: 2021 DF | $128 \times 128$ log-mel | **Vision Transformer (ViT)** | **Generalization Transformer** (Cross-codec robustness) |
| **Experiment E** *(Ablation)* | ASVspoof 2019 LA (Train/Dev $\rightarrow$ Eval) | $224 \times 224$ log-mel | **ViT-B/16** | **Resolution Ablation** ($128\times 128$ vs $224\times 224$ ViT) |
| **Experiment F** *(Optional/Future)* | ASVspoof 5 (Track 1 Dev Subset) | $128 \times 128$ log-mel | **ViT** | **Next-Gen Benchmark Exploration** |

---

## 4. ViT Input Handling for $128 \times 128$ Spectrograms

Standard Vision Transformers (e.g., standard ViT-B/16) were originally trained on $224 \times 224$ images with $16 \times 16$ patches ($14 \times 14 = 196$ tokens). Because the DeepShield baseline strictly operates on $128 \times 128$ spectrograms, we investigated three technical approaches:

### 4.1 Comparative Analysis of Input Handling Strategies

1. **Strategy 1: Pretrained ViT with 2D Bicubic Positional Embedding Interpolation (Selected Primary Strategy)**
   - **Mechanism:** Using ImageNet-pretrained ViT weights (e.g., `vit_base_patch16_224` or `vit_small_patch16_224` via `timm` / `torchvision`), the input image is kept at native $128 \times 128$.
   - **Patch Grid:** With $16 \times 16$ patches, a $128 \times 128$ spectrogram produces exactly $(128/16) \times (128/16) = 8 \times 8 = 64$ patch tokens.
   - **Positional Embedding Adaptation:** The original $14 \times 14$ grid of learnable positional embeddings is resampled to $8 \times 8$ using 2D bicubic interpolation prior to adding the `[CLS]` token.
   - **Advantages:** Retains rich ImageNet low-level edge and texture representations while operating natively on $128\times 128$ without artificial audio stretching; reduces transformer sequence length from 197 to 65 tokens, accelerating inference by $\sim 3\times$.

2. **Strategy 2: ViT with $8\times 8$ Patch Size (`patch_size=8`)**
   - **Mechanism:** On $128 \times 128$ input, $8 \times 8$ patches yield $(128/8) \times (128/8) = 16 \times 16 = 256$ tokens.
   - **Trade-off:** High spatial/spectral granularity, but quadruples self-attention computation ($O(N^2)$ where $N=256$).

3. **Strategy 3: Resizing Spectrogram to $224 \times 224$ (Preserved as Ablation Experiment E)**
   - **Mechanism:** Bilinear/bicubic interpolation of the $128\times 128$ spectrogram to $224\times 224$.
   - **Trade-off:** Allows unmodified $14\times 14$ positional embeddings, but alters the original representation frequency bins.

**Conclusion:** Strategy 1 is selected for **Experiment B and D**, preserving exact representation parity with EfficientNet-B0.

---

## 5. Standardized Audio Preprocessing Pipeline

To guarantee fairness, both models will consume batches generated by the exact same preprocessing pipeline:

```
Raw Audio (.flac / .wav)
  │
  ▼
1. Resample to 16,000 Hz Mono
  │
  ▼
2. Silence Trimming (top_db = 30 dB)
  │
  ▼
3. Peak Normalization (range [-1.0, 1.0])
  │
  ▼
4. Chunking (4.0s window = 64,000 samples; 2.0s hop = 32,000 samples)
  │
  ▼
5. Short-Time Fourier Transform (STFT):
   - n_fft = 1024
   - win_length = 400 (25 ms Hann window)
   - hop_length = 160 (10 ms hop)
  │
  ▼
6. Mel Filterbank:
   - n_mels = 128
   - fmin = 0 Hz, fmax = 8000 Hz
  │
  ▼
7. Log-Mel Conversion: S_db = 10 * log10(Mel + 1e-6)
  │
  ▼
8. Spatial Standardization: Resize/Crop to 128 × 128
  │
  ▼
9. 3-Channel Broadcast & ImageNet Normalization:
   - Channel Repeat: (1, 128, 128) → (3, 128, 128)
   - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
```

---

## 6. Training & Optimization Methodology

### 6.1 Hyperparameters
| Parameter | EfficientNet-B0 (Exp A / C) | Vision Transformer (Exp B / D) |
|---|---|---|
| **Backbone** | `efficientnet_b0` (ImageNet-1k) | `vit_base_patch16_224` (ImageNet-1k, pos-interpolated) |
| **Input Shape** | $(3, 128, 128)$ | $(3, 128, 128)$ |
| **Output Head** | `Linear(1280, 2)` | `Linear(768, 2)` |
| **Optimizer** | AdamW | AdamW |
| **Learning Rate** | $1 \times 10^{-4}$ | $5 \times 10^{-5}$ (lower LR for transformer stability) |
| **Weight Decay** | $1 \times 10^{-2}$ | $1 \times 10^{-2}$ |
| **LR Scheduler** | Cosine Annealing ($\eta_{min} = 10^{-6}$) | Cosine Annealing ($\eta_{min} = 10^{-6}$) |
| **Batch Size** | 32 | 32 |
| **Precision** | Mixed Precision (FP16 / AMP) | Mixed Precision (FP16 / AMP) |
| **Max Epochs** | 25 | 25 |
| **Early Stopping** | Patience = 6 epochs on Dev EER | Patience = 6 epochs on Dev EER |

### 6.2 Loss Function & Class Imbalance
In ASVspoof 2019 LA training set, bonafide utterances ($N_0 = 2,580$) and spoof utterances ($N_1 = 22,800$) exhibit a ratio of $\sim 1:8.84$.
We use inverse class frequency weighted Cross-Entropy Loss:
$$w_c = \frac{N_{\text{total}}}{2 \cdot N_c}, \quad \mathcal{L} = -\sum_{c=0}^1 w_c y_c \log(\hat{y}_c)$$

---

## 7. Decision Threshold Optimization Protocol

1. **Strict Dev Split Optimization:**  
   During validation on the ASVspoof 2019 Development split, predicted spoof probabilities $\hat{p}_i \in [0, 1]$ are evaluated over candidate thresholds $\tau \in [0.01, 0.99]$.
2. **Threshold Criterion:**  
   Optimal threshold $\tau^*$ is chosen to minimize Equal Error Rate (EER) or maximize Balanced Accuracy on the **Development split only**.
3. **Threshold Freezing:**  
   The chosen $\tau^*$ is serialized to `results/thresholds.json`.
4. **Evaluation Execution:**  
   The frozen threshold $\tau^*$ is applied unconditionally to the ASVspoof 2019 LA Evaluation split (Exp A & B) and ASVspoof 2021 DF Evaluation split (Exp C & D). **No threshold tuning is ever performed on evaluation data.**

---

## 8. Evaluation Metrics Suite

For each experiment, the evaluation suite computes and exports:
1. **Equal Error Rate (EER):** Primary biometrics standard (operating point where $\text{FPR} = \text{FNR}$).
2. **Balanced Accuracy (BAcc):** $\frac{1}{2}(\text{Sensitivity} + \text{Specificity})$.
3. **Precision, Recall, and F1-Score:** Separately for bona fide and spoof classes, plus macro-averaged.
4. **ROC-AUC & PR-AUC:** Area under receiver operating characteristic and precision-recall curves.
5. **Confusion Matrix:** True Positives, True Negatives, False Positives (bona fide flagged as spoof), False Negatives (spoof missed as bona fide).
6. **Per-Attack Breakdown (ASVspoof 2019 LA Eval):** Breakdown across individual unseen attacks A07 through A19.
7. **Computational Efficiency:** Parameter count (M), Checkpoint size (MB), GPU VRAM footprint (MB), GPU batch inference latency (ms), and single-sample CPU inference latency (ms).

---

## 9. Explainability & Interpretability Plan

- **EfficientNet-B0:** Grad-CAM applied to the final feature extraction layer (`features[-1]`), projecting gradient-weighted feature maps onto the log-mel spectrogram.
- **Vision Transformer:** Attention Rollout / Attention Attribution across all 12 self-attention heads, computing token-to-patch relevance from the `[CLS]` token to the $8\times 8$ patch grid.
- **Visualization Output:** Dual plot showing the raw time-frequency spectrogram alongside the heatmap overlay, highlighting whether the decision was driven by high-frequency spectral artifacts, formant discontinuities, or silence regions.

---

## 10. Robustness & Ablation Protocols

### 10.1 Ablation Matrix
1. Pretrained ImageNet weights vs. Random initialization (EfficientNet & ViT).
2. Class-weighted Cross-Entropy vs. Unweighted Cross-Entropy.
3. Input resolution: $128 \times 128$ (Exp B) vs $224 \times 224$ (Exp E).

### 10.2 Robustness Stress Testing
Synthetic perturbations applied to evaluation audio to assess real-world degradation:
- **Additive Gaussian Noise:** SNR levels at 20 dB, 10 dB, 5 dB, 0 dB.
- **Volume Perturbation:** Gain scaling by $\pm 6 \text{ dB}$.
- **Resampling Artifacts:** 16 kHz $\rightarrow$ 8 kHz (telephony) $\rightarrow$ 16 kHz.
- **Lossy Compression:** MP3 / AAC compression at 64 kbps and 32 kbps.
