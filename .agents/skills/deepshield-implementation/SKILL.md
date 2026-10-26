---
name: deepshield-implementation
description: Routine software implementation, unit testing, documentation, and non-destructive debugging for DeepShield audio deepfake detection. Use for code maintenance, bug fixes, refactoring, writing unit tests, and updating documentation within approved research constraints.
---

# DeepShield Implementation & Maintenance Runbook

Use this skill for routine coding, unit test implementation, documentation updates, and non-destructive debugging within the DeepShield codebase.

## 1. Pre-Implementation Checklist
1. Verify active workspace is `c:\Users\parth\OneDrive\Desktop\DeepShield`.
2. Confirm the Python virtual environment is active: `.venv\Scripts\python.exe`.
3. Note that the project is not a Git repository; do not attempt `git commit`, `git checkout`, or `git init`.
4. Ensure no background training or evaluation jobs are currently running via `manage_task` with action `list`.

## 2. Implementation Standards & Constraints
- **Preserve Preprocessing Invariants:** Any changes touching `src/preprocessing/` or `src/data/` must strictly preserve the 14-point audio extraction protocol ($128 \times 128$ log-mel, 16 kHz mono, 4.0s window, 2.0s hop).
- **Preserve Interface Contracts:**
  - Utterance-level datasets must yield `batched_chunks`, `chunk_counts`, `labels`, `utterance_ids`, and `attack_ids` when using `collate_utterances`.
  - Models must accept `(B, 3, 128, 128)` tensors and produce `(B, 2)` unnormalized logits.
- **Docstring & Type Annotation Integrity:** Preserve existing docstrings, comments, and type signatures unless explicitly refactoring.

## 3. Verification & Testing Workflow
After making modifications:
1. Run syntax and lint checks:
   ```powershell
   .venv\Scripts\python.exe -m py_compile <modified_file_path>
   ```
2. Run targeted unit tests:
   ```powershell
   .venv\Scripts\pytest.exe tests/test_<relevant_test>.py -v
   ```
3. Confirm all tests pass with exit code 0.
4. If modifying models or data loading, execute `tests/test_model_parity.py` and `tests/test_batched_validation_equivalence.py`.

## 4. Documentation & Boundaries
- Update relevant markdown documents in `docs/` or docstrings if architectural interfaces change.
- **Do not modify empirical values in `results/model_comparison.csv` or overwrite audit reports in `reports/` during routine implementation.**
