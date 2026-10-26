---
name: deepshield-experiment-manager
description: Manage, prepare, and govern DeepShield experiments (CNN vs ViT, cross-dataset generalization, ablations). Use for protocol verification, resource checks, conditional smoke tests, run preparation, and experiment provenance before full execution.
---

# DeepShield Experiment Management & Governance Runbook

Use this skill to prepare, validate, preflight, and govern model training runs and evaluations across all DeepShield experiments (Experiments A, B, C, D, E).

## 1. Active Research Gate Verification
Before preparing or launching Experiment B (ViT-B/16 on ASVspoof 2019 LA):
1. **Check Status of Threshold Discrepancy Investigation:** Confirm whether the Experiment A threshold discrepancy ($\tau_{\text{dev}} = 0.9678$ vs $\tau_{\text{eval}} \approx 2.8 \times 10^{-6}$) has been investigated and documented.
2. **Enforce Gate:** If the investigation is unresolved, **DO NOT proceed with full ViT training**. Report the blocker to the user.

## 2. Protocol Verification & Preflight Audit
When preparing any approved experiment:
1. **Verify Experiment Parameters:**
   - Model architecture (`efficientnet` vs. `vit`).
   - Image resolution (strictly $128 \times 128$ for primary experiments; $224 \times 224$ only for Experiment E ablation).
   - Training hyperparameters: optimizer (AdamW), learning rate ($10^{-4}$ for CNN, $5 \times 10^{-5}$ or $10^{-4}$ for ViT), batch size (64), cosine annealing scheduler, early stopping patience (6).
   - Class weighting: inverse-frequency weights derived from training split.
2. **Verify Hardware & Concurrency:**
   - Run `nvidia-smi` to confirm GPU memory availability (8 GB VRAM on RTX 5050).
   - Ensure NO other training or evaluation processes are active via `manage_task` with action `list`.
3. **Verify Dataset Caches:**
   - Confirm preprocessed feature cache exists at `C:\DeepShieldData\cache_spec\128x128`.

## 3. Conditional & Isolated Smoke-Test Procedure
Before proposing or running a smoke test:
1. **Inspect CLI Arguments:** Run `.venv\Scripts\python.exe src\training\train_<model>.py --help` or inspect the `ArgumentParser` in code to confirm which arguments the target script actually supports (e.g. `--fraction`, `--epochs`, `--save_dir`, `--val_batch_size`).
2. **Verify Output Isolation:**
   - Confirm that the script accepts a custom save directory argument (e.g., `--save_dir`).
   - Specify an isolated temporary directory (e.g., `--save_dir tmp/smoke_test/checkpoints`).
   - Ensure the script will NOT write to `results/thresholds.json` or overwrite production checkpoints.
   - **If isolation cannot be guaranteed, STOP and ask the user before running.**
3. **Execute Verified Smoke Test:** Run with bounded subsample and isolated output:
   ```powershell
   .venv\Scripts\python.exe -u src\training\train_<model>.py --fraction 0.02 --epochs 1 --save_dir tmp/smoke_test/
   ```
4. **Verify Smoke-Test Outcomes:**
   - Dataset loads without error.
   - Forward and backward passes complete with AMP `GradScaler`.
   - Loss is finite (no NaN / Inf).
   - Utterance-level validation loop completes.
   - Temporary checkpoint serialized inside `tmp/smoke_test/` without touching production files.

## 4. Threshold Lifecycle & Freezing Governance
1. **No Writes During Training:** Do not write or serialize thresholds to `results/thresholds.json` during ordinary training loops or smoke tests.
2. **Development Selection:** Record the optimal development threshold in the training run log or history JSON (`results/training_history_<model>.json`).
3. **Freezing Preflight:**
   - Inspect `results/thresholds.json` to verify current contents and ensure existing entries (e.g. `"efficientnet_b0"`) are preserved.
   - Propose an uncollided model key (e.g. `vit_b16_128_seed42`).
4. **Approval Gate:** Present the proposed threshold entry and target file to the user. **Write to `results/thresholds.json` only after receiving explicit user approval.**

## 5. Human Approval Gate
Once the smoke test passes:
1. Formulate the exact execution command for the full run.
2. Document random seed, hyperparameters, expected epoch count, and checkpoint output paths.
3. **STOP and present the preflight evidence to the user. Await explicit approval before launching.**

## 6. Execution Monitoring & Provenance Logging
During and after approved runs:
1. Monitor progress at epoch boundaries.
2. Record training loss, dev loss, dev EER, and learning rate.
3. Save complete history to `results/training_history_<model>.json`.
4. Log SHA-256 checksums of the resulting best checkpoint (`models/checkpoints/best_<model>.pt`).
