---
name: deepshield-research-guardrails
description: Mandatory research safeguards, protocol invariants, and integrity guardrails for the DeepShield project.
trigger: always_on
---

# DeepShield Mandatory Research Guardrails

All operations within this codebase must strictly adhere to these immutable research guardrails:

## 1. Zero Fabrication Directive (Absolute Priority)
- **Never fabricate results, metrics, logs, test outcomes, hashes, citations, or claims of completion.**
- Every reported metric (EER, ROC-AUC, accuracy, precision, recall, F1, confusion matrix) must originate from direct, unedited execution logs or persisted CSV records.
- If a task, test, or experiment was merely launched or partially executed, it must NEVER be reported as successful or complete.

## 2. Evaluation Partition Isolation (Zero Leakage)
- **The official evaluation split (`ASVspoof2019_LA_eval`, `ASVspoof2021_DF_eval`) must NEVER be used to tune thresholds, select checkpoints, optimize hyperparameters, or make model-selection decisions.**
- A model may be evaluated against the evaluation partition at most once per approved experimental protocol.

## 3. Threshold Lifecycle & Freezing Integrity
- **Separation of Stages:**
  1. *Threshold Selection:* Computed strictly on Development partition (`dev`) validation scores during or after training.
  2. *Threshold Freezing:* An explicit, gated serialization step saving to `results/thresholds.json`. Must never be performed implicitly during training loops or smoke tests.
  3. *Official Evaluation:* Read-only consumption of the frozen threshold.
- **Pre-Serialization Verification:** Before writing any threshold:
  - Verify that the target file exists and inspect its current schema.
  - Confirm the model key is distinct and uncollided (e.g., `vit_b16_128_seed42`).
  - Preserve all existing entries in `results/thresholds.json`.
  - Require explicit human approval for the freeze/write step.
- **No Overwriting & No Recalibration:** Never overwrite existing threshold entries without explicit approval. Never tune, adjust, or recalibrate decision thresholds against evaluation data.

## 4. Current Research Gate — Experiment A Threshold Discrepancy
- **Active Blocker:** Full training of the Vision Transformer (Experiment B) must NOT begin until the Experiment A threshold discrepancy is diagnosed, audited, and documented.
- **The Discrepancy:** The frozen development threshold was $\tau_{\text{dev}} = 0.9677734375$, whereas the optimal post-hoc evaluation EER threshold was $\tau_{\text{eval}} \approx 2.801418 \times 10^{-6}$ (a difference of over 5 orders of magnitude).
- **Audit Mandate:**
  - Inspect the actual metric code (`compute_eer`, `roc_curve`), label mapping (`bonafide=0`, `spoof=1`), score orientation ($p_{\text{spoof}}$ pooling), and saved prediction logs first.
  - Clearly separate directly verified empirical findings from hypotheses. Do not assume the cause of the discrepancy prior to inspection.
- **Integrity Rule:** Do NOT alter recorded evaluation metrics in `results/model_comparison.csv` or tune on evaluation data to make results look consistent. Document empirical facts objectively.

## 5. Human Approval Gates for Expensive / Consequential Runs
- **Never launch full training runs or official evaluation passes without explicit human approval for that specific run.**
- **Smoke Test Isolation Requirement:**
  - Smoke tests must write exclusively to isolated temporary output paths (e.g., `--save_dir tmp/smoke_test/`).
  - Smoke tests must NEVER overwrite or write to production checkpoints (`models/checkpoints/best_*.pt`), frozen thresholds (`results/thresholds.json`), evaluation records (`results/model_comparison.csv`), or audit reports (`reports/*.md`).
  - If isolation cannot be verified or guaranteed via the target script's CLI arguments, **STOP and ask the user before running the smoke test.**
- Proceed to full training or evaluation only upon receiving explicit approval following a verified smoke test.

## 6. Artifact Preservation & Immutability
- **Never overwrite or delete frozen checkpoints (`models/checkpoints/best_*.pt`), frozen thresholds (`results/thresholds.json`), evaluation records (`results/model_comparison.csv`), or audit reports (`reports/*.md`) without explicit human approval.**
- Always record and verify cryptographic SHA-256 checksums before loading or referencing checkpoints.

## 7. Experimental Protocol & Preprocessing Invariants
- **Preserve the locked 14-point audio preprocessing pipeline:** 16 kHz mono sampling, 4.0s sliding-window chunks with 2.0s hop, unweighted mean-probability utterance pooling, and $128 \times 128$ log-mel spectrogram generation.
- Preserve class label conventions: `0 = bonafide` (authentic), `1 = spoof` (synthetic/converted).
- Preserve score conventions: $p_{\text{spoof}} = \text{softmax}(\text{logits})[:, 1] \in [0, 1]$. Positive score indicates synthetic speech.

## 8. Discrepancy Investigation & Empirical Honesty
- Investigate all anomalies, regressions, or unexpected metrics immediately. Document the evidence.
- **Never alter code, results, or data to make them appear consistent.**
- Clearly distinguish observed facts, empirical measurements, hypotheses, and unresolved questions in all reporting.

## 9. Resource & Concurrency Governance
- Check active processes via `manage_task`, GPU memory via `nvidia-smi`, host RAM, and disk space before expensive operations.
- **Strictly enforce at most ONE GPU-intensive job at any given time.**

## 10. Non-Git Environment & Versioning Constraint
- The workspace root is not a Git repository (`git status` fails).
- **Do not initialize Git (`git init`) or alter repository configuration without explicit approval.**
- Use filesystem metadata and cryptographic SHA-256 checksums to verify artifact provenance and change history.

## 11. Autonomous vs. Gated Scope Boundary
- **Autonomous Operations Permitted:** Routine software development, bug fixing, unit test execution, static analysis, non-destructive debugging, and documentation maintenance within approved constraints.
- **Gated Operations Requiring Prior Approval:** Full training runs, evaluation partition inference, threshold freezing, modifying experiment plans, deleting/overwriting checkpoints, and initializing Git.

## 12. Data Privacy & Exfiltration Prohibition
- Do not upload project audio data, feature caches, or checkpoint weights to external web services or third-party APIs without explicit authorization.
