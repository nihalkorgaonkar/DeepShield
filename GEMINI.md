# DeepShield — Speech Deepfake Detection Research Rules

You are operating inside the **DeepShield** research workspace (`c:\Users\parth\OneDrive\Desktop\DeepShield`).
This project conducts an empirical comparison between a CNN Baseline (EfficientNet-B0) and a Vision Transformer (ViT-B/16) on ASVspoof 2019 LA and ASVspoof 2021 DF under strict academic integrity protocols.

@[deepshield-research-guardrails](.agents/rules/deepshield-research-guardrails.md)

## Active Research Gate (Current Priority)
- **Experiment A Threshold Discrepancy Investigation:** Before launching full ViT training (Phase 8), conduct an audit investigating why the frozen development threshold was $\tau_{\text{dev}} = 0.967773$ while the post-hoc evaluation EER operating threshold was $\tau_{\text{eval}} \approx 2.801418 \times 10^{-6}$.
  - Inspect actual metric implementation, label mapping, score orientation, and available predictions first.
  - Strictly separate directly verified empirical findings from hypotheses; do not assume the cause.
  - Do not tune against evaluation data or alter recorded results.

## Operational Defaults & Constraints
1. **Python Environment:** Always execute via the local virtual environment: `.venv\Scripts\python.exe`.
2. **Hardware Constraints:** NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM, sm_120 Blackwell). Enforce single-stream execution (at most 1 active GPU job).
3. **Repository Status:** This workspace is **not currently a Git repository**. Do not run `git init` or modify VCS configuration without explicit approval. Use SHA-256 hashes and file timestamps for artifact provenance.
4. **Data Root:** `C:\DeepShieldData\ASVspoof2019_LA` | Preprocessed Cache: `C:\DeepShieldData\cache_spec\128x128`.
5. **Threshold Integrity:** Existing entries in `results/thresholds.json` must NEVER be silently overwritten. Thresholds are not written during ordinary training or smoke tests; freezing requires explicit human approval.
6. **Smoke Test Isolation:** Smoke tests must write to isolated temporary directories (`tmp/smoke_test/`) and must never touch production checkpoints, thresholds, or evaluation CSVs. If isolation cannot be guaranteed, stop and ask the user.
7. **Read-Only Audits:** Audits are strictly read-only by default. Do not generate or modify files in `reports/` unless explicitly authorized.
8. **Skills Available:**
   - Use `/deepshield-implementation` for routine coding, unit testing, and non-destructive maintenance.
   - Use `/deepshield-experiment-manager` for preflight checks, conditional smoke tests, and training preparation.
   - Use `/deepshield-scientific-auditor` for read-only metric verification, discrepancy audits, and integrity checks.
