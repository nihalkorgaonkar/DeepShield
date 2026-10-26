"""
DeepShield Autonomous Research Orchestrator — Task Registry
Defines declarative tasks, dependencies, CLI commands, artifact verifiers,
and execution constraints for the DeepShield research roadmap.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import os
import hashlib


@dataclass
class ResearchTask:
    task_id: str
    name: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    command: List[str] = field(default_factory=list)
    expected_artifacts: List[str] = field(default_factory=list)
    is_gpu_intensive: bool = False
    timeout_seconds: int = 7200
    max_retries: int = 2
    retry_delay_seconds: int = 15
    is_scientific_gate: bool = False

    def verify_artifacts(self) -> bool:
        """Verifies that all expected artifacts exist, are non-empty, and accessible."""
        for path in self.expected_artifacts:
            if not os.path.exists(path):
                return False
            if os.path.getsize(path) == 0:
                return False
        return True


def compute_sha256(path: str) -> str:
    """Computes SHA-256 hash of a file."""
    if not os.path.exists(path):
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


# Immutable Experiment A Baseline Hashes
PROTECTED_BASELINE_HASHES = {
    "models/checkpoints/best_efficientnet_b0.pt": "1bd3841c7f0568e7d62783df4015dd0f25656366889cdf86ad34c40d6f211b69",
    "results/thresholds.json": "17c61fced09aee820785cfe1e878b942ea2904a3ab48e262d2c41f3922fdce6d",
    "results/model_comparison.csv": "61079ff7bdfa0c62f9aa8688514c55f16c3d1cc1eabcf5d3f9ab34a6d61783ea",
    "reports/experiment_A_audit.md": "9251818075b837055f0c74ca02662b40aae754d7a0412ec1c0e8e011df54eddb",
}


def verify_protected_baselines() -> Tuple[bool, Dict[str, str]]:
    """Confirms all four Experiment A baseline artifacts match their immutable SHA-256 checksums."""
    results = {}
    all_passed = True
    for path, expected_hash in PROTECTED_BASELINE_HASHES.items():
        actual_hash = compute_sha256(path)
        results[path] = actual_hash
        if actual_hash != expected_hash:
            all_passed = False
    return all_passed, results


def get_task_registry(python_exe: str = ".venv\\Scripts\\python.exe") -> Dict[str, ResearchTask]:
    """Returns the full task registry for the DeepShield autonomous controller."""
    tasks = {
        "TASK_01_AUDIT_EXPERIMENT_A": ResearchTask(
            task_id="TASK_01_AUDIT_EXPERIMENT_A",
            name="Experiment A Threshold Discrepancy Audit",
            description="Performs a strictly read-only scientific audit on development data investigating the threshold shift.",
            dependencies=[],
            command=[python_exe, "-u", "scripts/audit_experiment_a_discrepancy.py"],
            expected_artifacts=["reports/experiment_A_discrepancy_audit.md"],
            is_gpu_intensive=True,
            timeout_seconds=900,
            max_retries=1,
            is_scientific_gate=True,
        ),
        "TASK_02_HARDEN_VIT_PIPELINE": ResearchTask(
            task_id="TASK_02_HARDEN_VIT_PIPELINE",
            name="ViT Pipeline Parity Verification",
            description="Validates ViT-B/16 architecture compatibility and runs unit tests for model parity.",
            dependencies=["TASK_01_AUDIT_EXPERIMENT_A"],
            command=[python_exe, "tests/test_vit_128.py"],
            expected_artifacts=[],
            is_gpu_intensive=False,
            timeout_seconds=300,
            max_retries=1,
        ),
        "TASK_03_TRAIN_VIT": ResearchTask(
            task_id="TASK_03_TRAIN_VIT",
            name="Experiment B — ViT-B/16 Full Training",
            description="Trains Vision Transformer on ASVspoof 2019 LA for 25 epochs with early stopping.",
            dependencies=["TASK_02_HARDEN_VIT_PIPELINE"],
            command=[
                python_exe, "-u", "src/training/train_vit.py",
                "--batch_size", "64",
                "--val_batch_size", "64",
                "--epochs", "25",
                "--lr", "1e-4",
                "--seed", "42",
                "--patience", "6",
                "--resume", "models/checkpoints/latest_vit_b16.pt",
            ],
            expected_artifacts=[
                "models/checkpoints/best_vit_b16.pt",
                "results/training_history_vit_b16.json",
            ],
            is_gpu_intensive=True,
            timeout_seconds=14400,  # 4 hours
            max_retries=2,
        ),
        "TASK_04_FREEZE_VIT_THRESHOLD": ResearchTask(
            task_id="TASK_04_FREEZE_VIT_THRESHOLD",
            name="ViT Development Threshold Freezing",
            description="Selects and serializes the optimal development threshold for ViT into results/thresholds_vit.json.",
            dependencies=["TASK_03_TRAIN_VIT"],
            command=[
                python_exe, "-c",
                "import json, torch, os, time; "
                "ckpt = torch.load('models/checkpoints/best_vit_b16.pt', map_location='cpu'); "
                "th = ckpt.get('dev_threshold', 0.5); "
                "eer = ckpt.get('dev_eer', 0.0); "
                "data = {'vit_b16': {'threshold': round(float(th), 6), 'dev_eer': round(float(eer), 6), 'development_dataset': 'ASVspoof2019_LA_dev', 'optimization_criterion': 'min_eer', 'random_seed': 42, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}}; "
                "os.makedirs('results', exist_ok=True); "
                "open('results/thresholds_vit.json', 'w', encoding='utf-8').write(json.dumps(data, indent=2)); "
                "print(f'Serialized ViT dev threshold {th:.6f} to results/thresholds_vit.json')"
            ],
            expected_artifacts=["results/thresholds_vit.json"],
            is_gpu_intensive=False,
            timeout_seconds=300,
            max_retries=1,
            is_scientific_gate=True,
        ),
        "TASK_05_EVAL_VIT_2019_LA": ResearchTask(
            task_id="TASK_05_EVAL_VIT_2019_LA",
            name="Official ASVspoof 2019 LA Evaluation (ViT)",
            description="Executes a single official evaluation pass of ViT across all 71,237 evaluation utterances under the frozen threshold, saving to results/model_comparison_vit.csv.",
            dependencies=["TASK_04_FREEZE_VIT_THRESHOLD"],
            command=[
                python_exe, "src/evaluation/evaluate.py",
                "--model_type", "vit",
                "--checkpoint", "models/checkpoints/best_vit_b16.pt",
                "--threshold_file", "results/thresholds_vit.json",
                "--output_csv", "results/model_comparison_vit.csv",
                "--split", "eval",
                "--batch_size", "64",
            ],
            expected_artifacts=["results/model_comparison_vit.csv"],
            is_gpu_intensive=True,
            timeout_seconds=1800,  # 30 minutes
            max_retries=1,
            is_scientific_gate=True,
        ),
        "TASK_06_COMPARATIVE_ANALYSIS": ResearchTask(
            task_id="TASK_06_COMPARATIVE_ANALYSIS",
            name="Empirical Model Comparison (A vs B)",
            description="Generates publication-quality figures comparing EfficientNet-B0 and ViT-B/16 on ASVspoof 2019 LA.",
            dependencies=["TASK_05_EVAL_VIT_2019_LA"],
            command=[python_exe, "scripts/generate_plots.py"],
            expected_artifacts=["outputs/figures/model_comparison_bar.png"],
            is_gpu_intensive=False,
            timeout_seconds=600,
            max_retries=2,
        ),
        "TASK_07_ROBUSTNESS_EVALUATION": ResearchTask(
            task_id="TASK_07_ROBUSTNESS_EVALUATION",
            name="Acoustic Corruptions Robustness Suite",
            description="Evaluates noise, volume, and telephony perturbation stress tests on the dev partition.",
            dependencies=["TASK_05_EVAL_VIT_2019_LA"],
            command=[python_exe, "scripts/run_robustness.py"],
            expected_artifacts=[
                "results/robustness_results.json",
                "outputs/figures/robustness_comparison.png",
            ],
            is_gpu_intensive=True,
            timeout_seconds=1800,
            max_retries=2,
        ),
        "TASK_08_EXPLAINABILITY_ANALYSIS": ResearchTask(
            task_id="TASK_08_EXPLAINABILITY_ANALYSIS",
            name="Explainability & Attention Maps Generation",
            description="Generates Grad-CAM for CNN and self-attention rollouts for ViT.",
            dependencies=["TASK_05_EVAL_VIT_2019_LA"],
            command=[python_exe, "scripts/run_explainability.py"],
            expected_artifacts=[
                "outputs/figures/explainability_comparison.png",
                "outputs/figures/explainability_summary.json",
            ],
            is_gpu_intensive=False,
            timeout_seconds=600,
            max_retries=2,
        ),
        "TASK_09_GENERALIZATION_2021_DF": ResearchTask(
            task_id="TASK_09_GENERALIZATION_2021_DF",
            name="ASVspoof 2021 DF Cross-Dataset Preflight & Eval",
            description="Audits 2021 DF dataset availability. Documents missing audio files without fabrication.",
            dependencies=["TASK_05_EVAL_VIT_2019_LA"],
            command=[python_exe, "scripts/audit_generalization_2021_df.py"],
            expected_artifacts=["results/generalization_2021_df_status.json"],
            is_gpu_intensive=False,
            timeout_seconds=300,
            max_retries=1,
            is_scientific_gate=False,
        ),
        "TASK_10_FINAL_DELIVERABLES": ResearchTask(
            task_id="TASK_10_FINAL_DELIVERABLES",
            name="Final Research Report & Artifact Quality Audit",
            description="Synthesizes all empirical findings, metric tables, and checksums into docs/final_report.md and docs/final_audit.md.",
            dependencies=["TASK_06_COMPARATIVE_ANALYSIS", "TASK_07_ROBUSTNESS_EVALUATION", "TASK_08_EXPLAINABILITY_ANALYSIS", "TASK_09_GENERALIZATION_2021_DF"],
            command=[python_exe, "scripts/build_final_deliverables.py"],
            expected_artifacts=["docs/final_report.md", "docs/final_audit.md"],
            is_gpu_intensive=False,
            timeout_seconds=600,
            max_retries=1,
        ),
    }
    return tasks
