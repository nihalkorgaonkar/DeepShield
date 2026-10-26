# ASVspoof 5 Benchmark Feasibility & Resource Assessment

**Project:** DeepShield — Vision Transformer vs. CNN for Speech Deepfake Detection  
**Author:** Lead ML Engineer & Research Assistant  
**Date:** October 2026  
**Document Status:** Formal Feasibility Report  

---

## 1. Overview of the ASVspoof 5 Benchmark (2024)

ASVspoof 5 is the fifth official challenge edition organized by the ASVspoof Consortium in 2024, expanding speech deepfake detection into broader acoustic, multilingual, and musical domains:
- **Track 1 (Countermeasure / CM):** Detection of synthetic and voice-converted speech and vocal singing across diverse international languages, modern diffusion models, and multispeaker acoustic contexts.
- **Track 2 (In-the-Wild Audio Deepfakes):** Detection of spoofed audio captured in realistic acoustic environments with natural reverberation and background soundscapes.

---

## 2. Dataset Scale & Storage Analysis

### 2.1 Archive and Disk Space Requirements
| Partition | Audio Duration | Raw Compressed Archive | Extracted FLAC Storage |
|---|---|---|---|
| **Track 1 Training Split** | $\sim 350\text{ hours}$ | $\sim 50\text{ GB}$ | $\sim 110\text{ GB}$ |
| **Track 1 Development Split** | $\sim 100\text{ hours}$ | $\sim 15\text{ GB}$ | $\sim 32\text{ GB}$ |
| **Track 1 Evaluation Split** | $\sim 700\text{ hours}$ | $\sim 95\text{ GB}$ | $\sim 210\text{ GB}$ |
| **Track 2 In-the-Wild Partitions** | $\sim 250\text{ hours}$ | $\sim 40\text{ GB}$ | $\sim 85\text{ GB}$ |
| **Total ASVspoof 5 Full Archive** | **$> 1,400\text{ hours}$** | **$\sim 200\text{ GB}$** | **$> 437\text{ GB}$** |

### 2.2 Host Machine Storage Comparison
- **Host Free Disk Space:** **285.99 GB** on `C:\`.
- **Full ASVspoof 5 Extraction Requirement:** $> 437\text{ GB}$ (or $> 250\text{ GB}$ even for Track 1 alone when accounting for simultaneous zip archives and extracted audio).
- **Result:** Downloading and uncompressing the full ASVspoof 5 dataset would risk saturating host storage and causing system crashes.

---

## 3. Compute & GPU VRAM Feasibility Analysis

### 3.1 Host Compute Profile
- **GPU:** NVIDIA GeForce RTX 5050 Laptop GPU (8,151 MiB VRAM / 8 GB).
- **Memory Bandwidth & Architecture:** 128-bit bus, entry-level mobile RTX accelerator.

### 3.2 Training Throughput & Duration Estimates
- **EfficientNet-B0 on full ASVspoof 5 ($> 1\text{M}$ chunks):**
  - Estimated epoch time: $\sim 35\text{--}45\text{ minutes/epoch}$.
  - Full 25-epoch convergence: $\sim 15\text{--}18\text{ hours}$.
- **ViT-B/16 on full ASVspoof 5 ($> 1\text{M}$ chunks):**
  - Estimated epoch time: $\sim 2.5\text{--}3.5\text{ hours/epoch}$.
  - Full 25-epoch convergence: $\sim 65\text{--}85\text{ hours}$ ($\sim 3.5\text{ days}$ of non-stop GPU thermal load on a laptop chassis).

---

## 4. Academic Evaluation & Feasibility Verdict

### 4.1 Comparison of Benchmarks for DeepShield Objectives
| Criterion | ASVspoof 2019 LA | ASVspoof 2021 DF | ASVspoof 5 (2024) |
|---|---|---|---|
| **Academic Baseline Comparability** | Gold standard (comparable with $> 250$ published papers). | Official standard for codec & transmission generalization. | Emerging; comparisons with historical baselines are non-standard. |
| **Direct DeepShield Paper Parity** | **Exact match** to original DeepShield baseline. | Natural generalization benchmark for 2019-trained models. | Substantially different acoustic domain (singing / multilingual). |
| **Local Storage Budget** | $\sim 15\text{ GB}$ (Comfortable) | $\sim 45\text{ GB}$ (Comfortable) | $> 437\text{ GB}$ (**Exceeds Disk Capacity**) |
| **RTX 5050 8 GB Feasibility** | **Optimal** ($\sim 2\text{--}3\text{ min/epoch}$) | **Optimal for Inference** | **Marginal / High Thermal Risk** |

---

## 5. Strategic Recommendation

1. **Primary Research Focus (Experiments A, B, C, D):**  
   Maintain **ASVspoof 2019 LA** as the primary in-domain controlled benchmark and **ASVspoof 2021 DF** as the official cross-codec generalization evaluation benchmark. This directly satisfies the faculty request while guaranteeing 100% scientific reproducibility and safety.
2. **Designation for ASVspoof 5 (Experiment F):**  
   Classify ASVspoof 5 as **Future Work / Optional Pilot Study**. If a pilot exploration is requested, a documented 5% stratified development slice of Track 1 may be evaluated, clearly documented as a development subset rather than the full official benchmark.
