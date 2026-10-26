# Official ASVspoof Dataset Protocol & Evaluation Guidelines

**Project:** DeepShield — Vision Transformer vs. CNN for Speech Deepfake Detection  
**Author:** Lead ML Engineer & Research Assistant  
**Date:** October 2026  
**Document Status:** Official Protocol Reference  

---

## 1. Challenge Editions & Official Sources

The Automatic Speaker Verification and Spoofing (ASVspoof) initiative is the benchmark community initiative for voice biometrics and speech deepfake detection. Official challenges were conducted in **2015, 2017, 2019, 2021, and 2024 (ASVspoof 5)**.

> [!IMPORTANT]
> **Clarification Regarding "ASVspoof 2020 / 2022":**  
> There are no official challenge editions named "ASVspoof 2020" or "ASVspoof 2022". In academic literature, citations mentioning 2020–2022 refer to research conducted during the **ASVspoof 2021** challenge cycle. In accordance with strict academic guidelines, DeepShield uses the official **ASVspoof 2019 LA** and **ASVspoof 2021 DF** challenge datasets.

### 1.1 Official Dataset Repositories & Licenses
- **Host Institution:** University of Edinburgh DataShare / ASVspoof Consortium.
- **ASVspoof 2019 Logical Access:** Edinburgh DataShare ([https://datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336)).
- **ASVspoof 2021 Deepfake:** Edinburgh DataShare ([https://datashare.ed.ac.uk/handle/10283/4056](https://datashare.ed.ac.uk/handle/10283/4056)) and ASVspoof Official Evaluation Keys Repository.
- **License:** Open Data Commons Attribution License (ODC-By) / Academic Research Use.

---

## 2. ASVspoof 2019 Logical Access (LA) Benchmark

The ASVspoof 2019 Logical Access track evaluates countermeasures against synthetic speech (Text-to-Speech / TTS) and voice conversion (VC).

### 2.1 Audio Specifications
- **Sampling Rate:** 16,000 Hz (16 kHz), 16-bit PCM.
- **Format:** Lossless FLAC (`.flac`).
- **Channel:** Single-channel mono.

### 2.2 Split Structure & Statistics
| Partition | Bonafide Utterances | Spoof Utterances | Total Utterances | Male / Female Speakers | Spoof Attack Systems |
|---|---|---|---|---|---|
| **Train** | 2,580 | 22,800 | 25,380 | 20 speakers (8 M / 12 F) | A01, A02, A03, A04, A05, A06 |
| **Development** | 2,548 | 22,296 | 24,844 | 20 speakers (8 M / 12 F) | A01, A02, A03, A04, A05, A06 |
| **Evaluation** | 7,355 | 63,882 | 71,237 | 67 speakers (28 M / 39 F) | A07–A19 (**13 Unseen Attacks**) |
| **Total** | **12,483** | **108,978** | **121,461** | **107 speakers** | **19 Attack Systems** |

### 2.3 Attack Systems Breakdown
- **Known Attacks (Train & Dev: A01–A06):**
  - **A01:** Neural network based TTS (WaveNet vocoder).
  - **A02:** Neural network based TTS (World vocoder).
  - **A03:** Neural network based TTS (Feed-forward / Mel-spectrogram).
  - **A04:** Waveform concatenation based TTS.
  - **A05:** Voice conversion (VC) using VCC2018 baseline (World vocoder).
  - **A06:** Voice conversion (VC) using spectral transfer.
- **Unseen Attacks (Evaluation: A07–A19):**
  - Advanced neural TTS/VC systems, including GAN-based vocoders, modern acoustic models, waveform filtering, and high-fidelity synthesis architectures not present in the training set.

### 2.4 Protocol File Structure
The official protocol files (`ASVspoof2019.LA.cm.train.trn.txt`, `ASVspoof2019.LA.cm.dev.trl.txt`, `ASVspoof2019.LA.cm.eval.trl.txt`) follow the 5-column space-delimited standard:

```text
[SPEAKER_ID] [AUDIO_FILE_NAME] [ENVIRONMENT_ID] [ATTACK_ID] [KEY]
LA_0079 LA_D_1047731 - - bonafide
LA_0079 LA_D_1105538 - A06 spoof
LA_0079 LA_D_1125976 - A05 spoof
```

---

## 3. ASVspoof 2021 Deepfake (DF) Evaluation Benchmark

The ASVspoof 2021 Deepfake (DF) task evaluates speech deepfake detection under **uncontrolled, noisy, compressed, and multi-codec transmission conditions**.

### 3.1 Official Challenge Protocol & Training Rules
- **No Native 2021 DF Training Set:** Under official ASVspoof 2021 DF guidelines, countermeasure (CM) systems are trained on the **ASVspoof 2019 LA training and development partitions** (or permitted external clean speech) and evaluated on the **2021 DF evaluation partition**.
- **Evaluation Partition Size:** 611,829 audio trials.
- **Transmission & Compression Impairments:**
  - Lossy audio compression codecs: MP3, AAC, OGG Vorbis, OPUS.
  - Variable bitrates: High bitrate (128–256 kbps), medium bitrate (64 kbps), and extreme low-bitrate compression (16–32 kbps).
  - Telephony and internet streaming channels.
  - Advanced neural vocoders (HiFi-GAN, WaveGlow, MelGAN, etc.).

### 3.2 Evaluation Keys & Trial Structure
The protocol file `ASVspoof2021.DF.cm.eval.trial.txt` specifies trial files, and `ASVspoof2021.DF.cm.eval.keys.txt` contains the ground-truth metadata:
```text
[AUDIO_FILE_NAME] [SOURCE_DATASET] [VOCODER] [SRC_KEY] [KEY] [COMPRESSION_CODEC] [BITRATE]
DF_E_2000011 asvspoof2019 wavenet spoof spoof mp3 64k
DF_E_2000012 vctk none bonafide bonafide ogg 32k
```

### 3.3 Primary Challenge Metric
The official metric for the ASVspoof 2021 DF task is the **Equal Error Rate (EER)**.

---

## 4. Data Integrity & Anti-Leakage Rules

To maintain absolute scientific validity, the DeepShield experiment strictly enforces the following four rules:

1. **Rule 1 — Zero Evaluation Leakage:**  
   No sample, spectrogram, or feature from the ASVspoof 2019 LA Evaluation split or ASVspoof 2021 DF Evaluation split will ever enter the training or validation pipelines.
2. **Rule 2 — Independent Threshold Freezing:**  
   The decision threshold $\tau^*$ is optimized **strictly on the ASVspoof 2019 Development split**. Once computed, it is frozen and stored in `results/thresholds.json`. The evaluation sets are scored with this fixed threshold without recalculation.
3. **Rule 3 — Official Protocol Compliance:**  
   No random splitting or $k$-fold partitioning of the official evaluation files is permitted. Official ground truth files are parsed directly.
4. **Rule 4 — Identical Preprocessing Parity:**  
   Both EfficientNet-B0 and Vision Transformer must receive identically preprocessed $128 \times 128$ log-mel spectrogram tensors generated with identical FFT, hop, and normalization parameters.

---

## 5. Summary of Experiment Data Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           ASVspoof 2019 LA                                │
│                                                                           │
│  ┌───────────────────────┐   ┌───────────────────────┐                    │
│  │      Train Split      │   │   Development Split   │                    │
│  │ (25,380 audio files)  │   │ (24,844 audio files)  │                    │
│  └──────────┬────────────┘   └──────────┬────────────┘                    │
│             │                           │                                 │
│             ▼                           ▼                                 │
│      Model Training            Validation & Threshold                     │
│      (Weighted CE)             Optimization (τ* on Dev)                   │
└─────────────┬───────────────────────────┬─────────────────────────────────┘
              │                           │
              ▼                           ▼
    ┌───────────────────┐       ┌───────────────────┐
    │  EfficientNet-B0  │       │ VisionTransformer │
    │  (Exp A Baseline) │       │ (Exp B Candidate) │
    └─────────┬─────────┘       └─────────┬─────────┘
              │                           │
              ├───────────────────────────┴───────────────────────┐
              │ Apply Frozen Threshold τ*                         │
              ▼                                                   ▼
┌───────────────────────────────┐               ┌───────────────────────────────────┐
│     ASVspoof 2019 LA Eval     │               │       ASVspoof 2021 DF Eval       │
│    (71,237 audio files)       │               │      (611,829 audio trials)       │
│                               │               │                                   │
│  • Primary In-Domain Benchmark│               │  • Cross-Dataset Generalization   │
│  • 13 Unseen Attacks (A07-A19)│               │  • Multi-Codec & Transmission     │
│  • Exp A (CNN) vs Exp B (ViT) │               │  • Exp C (CNN) vs Exp D (ViT)     │
└───────────────────────────────┘               └───────────────────────────────────┘
```
