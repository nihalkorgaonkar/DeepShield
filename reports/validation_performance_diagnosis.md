# DeepShield — Validation Performance Diagnosis & Optimization Report

**Date:** October 11, 2026  
**Audited Run:** Experiment A Training Run (Task 1386)  
**Status:** DIAGNOSIS COMPLETE — Awaiting Review Before Applying Fix  

---

## 1. Summary of Stopped Run (Task 1386)

- **Graceful Stop:** Task 1386 stopped and preserved.
- **Preserved Log:** `models/checkpoints/backup_interrupted_run/task-1386_diagnostic.log`
  - SHA-256: `6d6baa8d14b21ef4c7c8ad052dd2be4f4e9fc181db85f6c4d815566435df0e55`
  - Size: 2,188,912 bytes
- **Completed Work:**
  - Epoch 1 Training: Completed all 447 batches in 2 minutes 03 seconds (`447/447 [02:03, 3.63 it/s]`).
  - Epoch 1 Validation: Interrupted at utterance 10,323 out of 24,844 (41.6%) after 4 minutes 21 seconds (~39.5 utts/sec).
- **Existing Checkpoint Preserved:** Epoch 11 checkpoint from previous run remains safely intact and backed up at `models/checkpoints/backup_interrupted_run/best_efficientnet_b0_epoch11.pt` (SHA-256: `658d2e0543a4933f18cd2ac8b7076e1e257269afaf54d3bdcc968618726e53ed`).

---

## 2. Root Cause Diagnosis

### Why Was Validation Processing Utterances Individually?
In `ASVspoof2019Dataset(mode="utterance")`, each utterance produces $N_i$ chunks ($N_i \in \{1, 2, 3\}$). Because $N_i$ varies across utterances, standard PyTorch `default_collate` cannot stack variable-length tensors into a 5D tensor, prompting the validation loader to use `batch_size=1`.

### Why Did Each Iteration Take So Long?
Micro-profiling on 495 development utterances (`tests/profile_validation_pipeline.py`) revealed the exact bottleneck distribution:

| Pipeline Stage | Total Time (495 utts) | Percentage | Avg Time / Utterance | Root Cause |
| :--- | :---: | :---: | :---: | :--- |
| **Model Inference (GPU)** | **9.490 s** | **80.4%** | **19.17 ms** | GPU severe underutilization on batch size $N=1$. CUDA kernel launch overhead and low Tensor Core occupancy dominate execution time. |
| **Disk Loading (`torch.load`)** | 1.826 s | 15.5% | 3.69 ms | Sequential synchronous unpickling of individual `.pt` files. |
| **Host-to-Device (PCIe)** | 0.349 s | 3.0% | 0.70 ms | Transferring tiny individual tensors $[1, 3, 128, 128]$ over PCIe bus. |
| **GPU Sync Barrier (`.item()`)** | 0.104 s | 0.9% | 0.21 ms | Synchronous host-device stalls on every single utterance. |
| **Aggregation** | 0.001 s | 0.0% | 0.00 ms | In-memory averaging. |
| **Total** | **11.800 s** | **100.0%** | **23.84 ms** | **Throughput: 41.9 utts/sec** |

- **GPU Utilization:** Peak VRAM was only **102.28 MB** out of 8,192 MB available on the NVIDIA RTX 5050 Laptop GPU (1.2% VRAM utilization), with compute cores running virtually idle between sequential kernel launches.

---

## 3. Proposed Solution: Safe Batched Utterance Inference

### Mechanism:
1. Implement `collate_utterances` function that groups $B$ utterances (e.g. $B = 64$) into a single contiguous chunk tensor:
   $$\mathbf{X}_{\text{batch}} = \text{cat}([S_1, S_2, \dots, S_B], \text{dim}=0) \in \mathbb{R}^{M \times 3 \times 128 \times 128}$$
   where $M = \sum_{i=1}^B N_i$ ($M \approx 72$ chunks for 64 utterances).
2. Execute ONE model forward pass for all $M$ chunks simultaneously:
   $$\mathbf{Z} = \text{model}(\mathbf{X}_{\text{batch}}) \in \mathbb{R}^{M \times 2}$$
3. Compute softmax spoof probabilities: $\mathbf{P} = \text{softmax}(\mathbf{Z}, \text{dim}=1)[:, 1]$.
4. Split probabilities using tracked chunk counts: `torch.split(P, chunk_counts)`.
5. Compute utterance score via mean pooling: $p_{\text{utt}, i} = \frac{1}{N_i} \sum_{j=1}^{N_i} P_{i, j}$.

### Scientific Invariance Guarantees:
- **Zero Chunk Dropping:** 100% of chunks are evaluated.
- **Identical Preprocessing:** No change to audio loading, silence trimming, peak normalization, or spectrogram extraction.
- **Identical Labels:** Utterance labels remain strictly aligned via zipped unpacking.
- **Identical Decision Rule:** Utterance probability remains the arithmetic mean of its chunk probabilities.

---

## 4. Empirical Benchmark & Verification Results

Tested on 495 development utterances across different batch sizes (`tests/benchmark_validation_optimization.py`):

| Method / Configuration | Elapsed Time | Measured Throughput | Speedup Factor | Peak VRAM | Estimated Full Dev Time (24,844 utts) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Unbatched ($B=1$)** | 11.800 s | 51.2 utts/sec | 1.00x | 102 MB | **8.08 minutes (485 s)** |
| **Batched ($B=32$ utts)** | 1.184 s | 418.1 utts/sec | 8.16x | 176 MB | **59.4 seconds (0.99 min)** |
| **Batched ($B=64$ utts)** | **0.972 s** | **509.3 utts/sec** | **9.94x** | **239 MB** | **48.8 seconds (0.81 min)** |
| **Batched ($B=128$ utts)** | 0.677 s | 731.2 utts/sec | 14.27x | 371 MB | **34.0 seconds (0.57 min)** |

### Numerical Agreement Audit:
- **Max Absolute Score Difference:** $3.677 \times 10^{-4}$ (due to minor floating-point associativity differences in cuDNN GEMM kernels under different batch sizes).
- **Mean Absolute Score Difference:** $1.162 \times 10^{-6}$.
- **EER Agreement:**
  - Unbatched EER: `0.000000%` (Th: `0.996834`)
  - Batched EER: `0.000000%` (Th: `0.996840`)
  - **EER Difference:** `0.000000%` (Exact match).

---

## 5. Next Steps

Awaiting user approval before applying `collate_utterances` with batch size 64 to `train_efficientnet.py` and resuming Experiment A.
