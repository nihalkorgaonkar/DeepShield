"""
Report Service
Reads and parses Markdown reports and experiment CSV/JSON artifacts.
Provides structured summaries, markdown content, and downloadable audit packages.
"""

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from backend.config import (
    DOCS_DIR,
    GENERALIZATION_STATUS_JSON,
    MODEL_COMPARISON_CSV,
    MODEL_COMPARISON_VIT_CSV,
    REPORTS_DIR,
    RESULTS_DIR,
    ROBUSTNESS_RESULTS_JSON,
    TRAINING_HISTORY_EFF,
    TRAINING_HISTORY_VIT,
)
from backend.schemas.models import MetricComparisonRecord, MetricsSummaryResponse
from backend.schemas.reports import ReportDetail, ReportSummary

logger = logging.getLogger("deepshield.reports")


class ReportService:
    def __init__(self):
        self.docs_dir = DOCS_DIR
        self.results_dir = RESULTS_DIR

    def list_reports(self) -> List[ReportSummary]:
        reports = [
            ReportSummary(
                id="final_report",
                title="DeepShield Final Research Report: ViT vs. EfficientNet-B0",
                category="Evaluation",
                description="Comprehensive empirical comparison between CNN baseline and Vision Transformer on ASVspoof 2019 LA.",
                filename="final_report.md",
                last_modified=self._get_mtime(self.docs_dir / "final_report.md"),
                size_bytes=self._get_size(self.docs_dir / "final_report.md"),
            ),
            ReportSummary(
                id="final_audit",
                title="Scientific Integrity & Cryptographic Provenance Audit",
                category="Audit",
                description="Zero Fabrication Directive validation, SHA-256 checkpoint verification, and partition isolation proof.",
                filename="final_audit.md",
                last_modified=self._get_mtime(self.docs_dir / "final_audit.md"),
                size_bytes=self._get_size(self.docs_dir / "final_audit.md"),
            ),
            ReportSummary(
                id="experiment_A_audit",
                title="Experiment A Threshold Discrepancy & Diagnostic Investigation",
                category="Audit",
                description="Deep investigation of the 5-orders-of-magnitude shift between development threshold and evaluation EER operating point.",
                filename="reports/experiment_A_audit.md",
                last_modified=self._get_mtime(REPORTS_DIR / "experiment_A_audit.md"),
                size_bytes=self._get_size(REPORTS_DIR / "experiment_A_audit.md"),
            ),
            ReportSummary(
                id="dataset_protocol",
                title="ASVspoof 2019 & 2021 Dataset Protocol Standards",
                category="Protocol",
                description="14-point audio extraction invariants, class labels, and mathematical score aggregation definitions.",
                filename="dataset_protocol.md",
                last_modified=self._get_mtime(self.docs_dir / "dataset_protocol.md"),
                size_bytes=self._get_size(self.docs_dir / "dataset_protocol.md"),
            ),
            ReportSummary(
                id="generalization_status",
                title="ASVspoof 2021 DF Cross-Dataset Generalization Status",
                category="Generalization",
                description="Official status documenting absence of raw evaluation FLAC audio in accordance with Zero Fabrication Directive.",
                filename="generalization_2021_df_status.json",
                last_modified=self._get_mtime(GENERALIZATION_STATUS_JSON),
                size_bytes=self._get_size(GENERALIZATION_STATUS_JSON),
            ),
        ]
        return reports

    def get_report_detail(self, report_id: str) -> Optional[ReportDetail]:
        report_map = {
            "final_report": (self.docs_dir / "final_report.md", "Evaluation", "DeepShield Final Research Report: ViT vs. EfficientNet-B0"),
            "final_audit": (self.docs_dir / "final_audit.md", "Audit", "Scientific Integrity & Cryptographic Provenance Audit"),
            "experiment_A_audit": (REPORTS_DIR / "experiment_A_audit.md", "Audit", "Experiment A Threshold Discrepancy Investigation"),
            "dataset_protocol": (self.docs_dir / "dataset_protocol.md", "Protocol", "ASVspoof 2019 & 2021 Dataset Protocol Standards"),
            "generalization_status": (GENERALIZATION_STATUS_JSON, "Generalization", "ASVspoof 2021 DF Generalization Resource Status"),
        }

        if report_id not in report_map:
            return None

        file_path, category, title = report_map[report_id]
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # If JSON, wrap in markdown code block for clean reading
            if file_path.suffix == ".json":
                content = f"```json\n{content}\n```"

            return ReportDetail(
                id=report_id,
                title=title,
                category=category,
                filename=file_path.name,
                markdown_content=content,
                last_modified=self._get_mtime(file_path),
                associated_artifacts=[
                    "results/model_comparison.csv",
                    "results/model_comparison_vit.csv",
                    "results/thresholds.json",
                    "results/thresholds_vit.json",
                ],
            )
        except Exception as e:
            logger.error(f"Error reading report {report_id}: {e}")
            return None

    def get_metrics_summary(self) -> MetricsSummaryResponse:
        """Parses model_comparison.csv and model_comparison_vit.csv to produce authoritative metrics."""
        eff_rec = self._parse_comparison_csv(MODEL_COMPARISON_CSV)
        vit_rec = self._parse_comparison_csv(MODEL_COMPARISON_VIT_CSV)

        discrepancy_analysis = {
            "efficientnet_dev_threshold": "0.967773 (min_eer on Dev, Dev EER = 6.44%)",
            "efficientnet_eval_eer_threshold": "2.801418e-06 (optimal post-hoc threshold on Eval, Eval EER = 21.49%)",
            "vit_dev_threshold": "0.986328 (min_eer on Dev, Dev EER = 1.55%)",
            "vit_eval_eer_threshold": "1.215935e-05 (optimal post-hoc threshold on Eval, Eval EER = 27.08%)",
            "verified_audit_findings": (
                "Both models exhibit a 5-orders-of-magnitude threshold shift between Development and Evaluation. "
                "Investigation confirmed: (1) Label mappings are consistent (0=bonafide, 1=spoof); (2) Score orientation is consistent "
                "(P_spoof = softmax[:, 1]); (3) Evaluation set introduces 13 unseen speech synthesis algorithms (A07–A19) "
                "yielding lower raw neural confidence than the 6 known training algorithms (A01–A06). "
                "Zero Fabrication Directive strictly forbids tuning or shifting thresholds on Evaluation data."
            ),
        }

        warning_details = (
            "CRITICAL RESEARCH ALERT: Operating thresholds optimized on the development split (τ ≈ 0.97 - 0.99) "
            "exhibit substantial divergence from optimal post-hoc operating points on the evaluation split (τ ≈ 10⁻⁶ - 10⁻⁵). "
            "Evaluation accuracy at frozen threshold reflects severe domain shift to unseen speech synthesis algorithms."
        )

        return MetricsSummaryResponse(
            efficientnet_baseline=eff_rec,
            vit_b16_model=vit_rec,
            discrepancy_analysis=discrepancy_analysis,
            discrepancy_warning=True,
            warning_details=warning_details,
        )

    @staticmethod
    def _parse_comparison_csv(filepath: Path) -> Optional[MetricComparisonRecord]:
        if not filepath.exists():
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                row = next(reader, None)
                if not row:
                    return None

                return MetricComparisonRecord(
                    model=row.get("model", ""),
                    dataset=row.get("dataset", ""),
                    split=row.get("split", ""),
                    parameters_m=float(row.get("parameters_m", 0.0)),
                    inference_time_ms_per_utterance=float(row.get("inference_time_ms_per_utterance", 0.0)),
                    eer=float(row.get("eer", 0.0)),
                    eer_percent=round(float(row.get("eer", 0.0)) * 100.0, 2),
                    eer_threshold=float(row.get("eer_threshold", 0.0)),
                    eval_threshold=float(row.get("eval_threshold", 0.0)),
                    accuracy=float(row.get("accuracy", 0.0)),
                    accuracy_percent=round(float(row.get("accuracy", 0.0)) * 100.0, 2),
                    balanced_accuracy=float(row.get("balanced_accuracy", 0.0)),
                    balanced_accuracy_percent=round(float(row.get("balanced_accuracy", 0.0)) * 100.0, 2),
                    f1_macro=float(row.get("f1_macro", 0.0)),
                    precision_macro=float(row.get("precision_macro", 0.0)),
                    recall_macro=float(row.get("recall_macro", 0.0)),
                    bonafide_precision=float(row.get("bonafide_precision", 0.0)),
                    bonafide_recall=float(row.get("bonafide_recall", 0.0)),
                    bonafide_f1=float(row.get("bonafide_f1", 0.0)),
                    spoof_precision=float(row.get("spoof_precision", 0.0)),
                    spoof_recall=float(row.get("spoof_recall", 0.0)),
                    spoof_f1=float(row.get("spoof_f1", 0.0)),
                    roc_auc=float(row.get("roc_auc", 0.0)),
                    pr_auc=float(row.get("pr_auc", 0.0)),
                    true_negatives=int(float(row.get("true_negatives", 0))),
                    false_positives=int(float(row.get("false_positives", 0))),
                    false_negatives=int(float(row.get("false_negatives", 0))),
                    true_positives=int(float(row.get("true_positives", 0))),
                    false_positive_rate=float(row.get("false_positive_rate", 0.0)),
                    false_negative_rate=float(row.get("false_negative_rate", 0.0)),
                )
        except Exception as e:
            logger.error(f"Failed to parse CSV {filepath}: {e}")
            return None

    @staticmethod
    def _get_mtime(p: Path) -> str:
        if p.exists():
            return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(os.path.getmtime(p)))
        return "Unknown"

    @staticmethod
    def _get_size(p: Path) -> int:
        if p.exists():
            return os.path.getsize(p)
        return 0
