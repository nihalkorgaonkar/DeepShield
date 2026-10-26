# DeepShield Scientific Integrity & Provenance Audit

**Audit Date:** 2026-10-11 11:24:03 UTC  
**Governing Standard:** DeepShield Research Guardrails (Zero Fabrication Directive)  

---

## 1. Cryptographic Checksum Provenance Table

| Artifact Path | Expected Baseline Hash | Verified Live Hash | Verification |
| :--- | :--- | :--- | :--- |
| `models/checkpoints/best_efficientnet_b0.pt` | `1bd3841c7f0568e7d62783df4015dd0f25656366889cdf86ad34c40d6f211b69` | `1bd3841c7f0568e7d62783df4015dd0f25656366889cdf86ad34c40d6f211b69` | **PASS** |
| `results/thresholds.json` | `17c61fced09aee820785cfe1e878b942ea2904a3ab48e262d2c41f3922fdce6d` | `17c61fced09aee820785cfe1e878b942ea2904a3ab48e262d2c41f3922fdce6d` | **PASS** |
| `results/model_comparison.csv` | `61079ff7bdfa0c62f9aa8688514c55f16c3d1cc1eabcf5d3f9ab34a6d61783ea` | `61079ff7bdfa0c62f9aa8688514c55f16c3d1cc1eabcf5d3f9ab34a6d61783ea` | **PASS** |
| `reports/experiment_A_audit.md` | `9251818075b837055f0c74ca02662b40aae754d7a0412ec1c0e8e011df54eddb` | `9251818075b837055f0c74ca02662b40aae754d7a0412ec1c0e8e011df54eddb` | **PASS** |

### Generated Vision Transformer Artifacts:
| Artifact Path | SHA-256 Checksum |
| :--- | :--- |
| `models/checkpoints/best_vit_b16.pt` | `14dfd3451c24499d17e705b1a0bc40fa7c8c1f6ef880e6c4bba2bd4570e799d0` |
| `models/checkpoints/latest_vit_b16.pt` | `f54dd0f9b9485e775a25f2e48d3548481b6b1158e8ed8330646da4a5b186debc` |
| `results/thresholds_vit.json` | `8de1e67a29e8bb4e632c2d9c181d49b80fc0fbe5f08cdc371e216258d25f3263` |
| `results/model_comparison_vit.csv` | `7f6a963219ba7bc30553235419f317f1cfb45424f298ec1789704e9dddde5a2a` |

---

## 2. Partition Isolation & Scientific Verification

1. **Development Threshold Freezing:** Thresholds were calibrated strictly using development split scores.
2. **Evaluation Split Single-Pass Consumption:** The official evaluation partition was accessed in read-only inference mode without hyperparameter tuning or post-hoc threshold searching.
3. **Immutability of Baseline:** Baseline artifacts were preserved without byte modification.
