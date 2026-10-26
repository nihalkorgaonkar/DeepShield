---
name: deepshield-scientific-auditor
description: Perform rigorous, read-only scientific audits of DeepShield experiments, checkpoints, thresholds, evaluation metrics, and artifact provenance. Use when verifying training completion, checking zero-leakage protocols, validating metric calculations, and auditing research claims.
---

# DeepShield Scientific Auditor Runbook

Use this skill to conduct independent, strictly read-only verification of experimental completion, checkpoint integrity, threshold calibration, metric calculations, and evaluation artifacts.

## 1. Strictly Read-Only Audit Constraint
- Audits are purely observational, evidence-gathering investigations.
- **Never modify, delete, or overwrite code, checkpoints, thresholds, evaluation outputs, or existing audit reports during an audit.**
- **Default Policy:** All audits are strictly read-only. Do NOT create or edit report files in `reports/` unless separately and explicitly authorized by the user. Present all findings, tables, and verification matrices directly in chat.

## 2. Investigation Protocol: Experiment A Threshold Discrepancy
When auditing the discrepancy between $\tau_{\text{dev}} = 0.967773$ and $\tau_{\text{eval}} \approx 2.801418 \times 10^{-6}$:
1. **Evidence-First Code Inspection:**
   - Inspect `src/evaluation/metrics.py` (`compute_eer`, `roc_curve`, threshold indexing).
   - Inspect `src/data/asvspoof2019.py` to confirm ground-truth label assignments (`bonafide=0`, `spoof=1`).
   - Inspect `src/training/train_efficientnet.py` and `src/evaluation/evaluate.py` to verify score computation (`softmax[:, 1]` pooling).
2. **Audit Available Predictions & Distributions:**
   - Inspect available prediction logs and score distributions for Dev and Eval partitions.
   - Check minimum, maximum, median, and quartile values of scores for bona fide and spoof utterances.
3. **Separation of Verified Findings vs. Hypotheses:**
   - Clearly delineate **Directly Verified Facts** (observed lines of code, numerical values, array shapes, score ranges).
   - Clearly delineate **Working Hypotheses** (e.g., dynamic range compression, domain shift on unseen vocoders, score orientation).
   - **Do NOT assume the cause of the discrepancy prior to gathering empirical proof.**
4. **Zero Contamination Rule:**
   - Do NOT tune any threshold against evaluation data.
   - Do NOT modify `results/model_comparison.csv` or existing audit reports.

## 3. Checkpoint Reload & Integrity Audit
1. Verify checkpoint existence, file size, and timestamp.
2. Calculate cryptographic SHA-256 checksum:
   ```powershell
   Get-FileHash -Algorithm SHA256 <checkpoint_path>
   ```
3. Execute programmatic reload test in Python:
   - Load checkpoint state dictionary onto CUDA device.
   - Verify `<All keys matched successfully>`.
   - Inspect recorded metadata: `epoch`, `dev_eer`, `dev_threshold`, `seed`.
   - Run dummy forward pass with random tensor: verify finite logits and softmax probabilities strictly summing to 1.0.

## 4. Threshold Lifecycle & Non-Overwriting Audit
1. Inspect `results/thresholds.json`.
2. Confirm that recorded thresholds were derived strictly on the Development partition (`dev`) via `min_eer` or approved criteria.
3. Verify that evaluation data was never accessed during threshold tuning.
4. Ensure existing threshold entries are preserved intact.

## 5. Evaluation Metrics & Confusion Matrix Audit
When auditing an official evaluation run:
1. Verify total evaluated utterances matches official split ground truth (e.g., exactly 71,237 utterances for ASVspoof 2019 LA Eval: 7,355 bonafide, 63,882 spoof).
2. Verify confusion matrix integrity:
   $$\text{TN} + \text{FP} = \text{Total Bonafide}$$
   $$\text{FN} + \text{TP} = \text{Total Spoof}$$
   $$\text{Total} = \text{TN} + \text{FP} + \text{FN} + \text{TP}$$
3. Verify rate calculations:
   $$\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}}, \quad \text{FNR} = \frac{\text{FN}}{\text{TP} + \text{FN}}$$
4. Clearly bifurcate:
   - **Threshold-Independent Metrics:** EER, ROC-AUC, PR-AUC.
   - **Threshold-Dependent Metrics:** Accuracy, Balanced Accuracy, Macro F1, Precision, Recall at frozen $\tau$.

## 6. Audit Reporting
- Present all findings, tables, verification matrices, and identified risks directly in chat.
- Highlight unresolved questions and specify the exact decision gate requiring human review before any next steps.
