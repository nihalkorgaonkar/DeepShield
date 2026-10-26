"""
Evaluation Service
Executes controlled evaluation and diagnostic runs on ASVspoof 2019 partitions
without modifying or overwriting baseline comparison artifacts.
Calculates authentic biometric and classification metrics.
"""

import json
import logging
import os
import threading
import time
import uuid
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.config import DEFAULT_CACHE_DIR, DEFAULT_DATA_ROOT, DEVICE, EVAL_RUNS_DIR
from backend.schemas.evaluation import (
    ConfusionMatrixData,
    EvaluationProgress,
    EvaluationRequest,
    EvaluationRunResult,
)
from backend.services.model_service import ModelService
from src.data.asvspoof2019 import ASVspoof2019Dataset, collate_utterances
from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger("deepshield.evaluation")


class EvaluationService:
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.device = model_service.device
        self.active_runs: Dict[str, EvaluationProgress] = {}
        self.completed_runs: Dict[str, EvaluationRunResult] = {}
        self._runs_lock = threading.Lock()

        # Load existing run files from disk if any
        self._load_saved_runs()

    def _load_saved_runs(self):
        if not EVAL_RUNS_DIR.exists():
            return
        for file in EVAL_RUNS_DIR.glob("run_*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    res = EvaluationRunResult(**data)
                    self.completed_runs[res.run_id] = res
            except Exception as e:
                logger.warning(f"Could not load evaluation run file {file}: {e}")

    def get_run_status(self, run_id: str) -> Optional[EvaluationProgress]:
        with self._runs_lock:
            return self.active_runs.get(run_id)

    def get_run_result(self, run_id: str) -> Optional[EvaluationRunResult]:
        with self._runs_lock:
            return self.completed_runs.get(run_id)

    def list_all_runs(self) -> List[EvaluationRunResult]:
        with self._runs_lock:
            return sorted(
                list(self.completed_runs.values()),
                key=lambda r: r.timestamp,
                reverse=True,
            )

    def start_evaluation(self, req: EvaluationRequest) -> str:
        """Launches a background thread to evaluate the specified model subset."""
        run_id = f"run_{uuid.uuid4().hex[:8]}"

        progress = EvaluationProgress(
            run_id=run_id,
            status="queued",
            model=req.model,
            dataset=req.dataset,
            split=req.split,
            processed_samples=0,
            total_samples=req.max_samples,
            percent_complete=0.0,
            current_time_ms=0.0,
        )

        with self._runs_lock:
            self.active_runs[run_id] = progress

        # Launch background worker
        worker = threading.Thread(
            target=self._run_eval_worker,
            args=(run_id, req),
            daemon=True,
        )
        worker.start()

        return run_id

    def _run_eval_worker(self, run_id: str, req: EvaluationRequest):
        logger.info(f"Starting evaluation job {run_id}: model={req.model}, split={req.split}, limit={req.max_samples}")
        t_start = time.perf_counter()

        with self._runs_lock:
            self.active_runs[run_id].status = "running"

        try:
            norm_id = "efficientnet_b0" if "eff" in req.model.lower() else "vit_b16"
            model = self.model_service.get_model(norm_id)
            if model is None:
                raise RuntimeError(f"Model '{norm_id}' is not loaded.")

            # Load dataset in utterance mode
            # Calculate fraction to roughly match max_samples
            dataset = ASVspoof2019Dataset(
                data_root=str(DEFAULT_DATA_ROOT),
                split=req.split,
                cache_dir=str(DEFAULT_CACHE_DIR),
                mode="utterance",
            )

            if len(dataset) == 0:
                raise RuntimeError(f"No samples found in ASVspoof2019 dataset for split '{req.split}'.")

            # Slice entries to requested limit while maintaining class balance
            bonafide_entries = [e for e in dataset.entries if e["label"] == 0]
            spoof_entries = [e for e in dataset.entries if e["label"] == 1]
            n_samples = min(req.max_samples, len(dataset))
            half = max(1, n_samples // 2)
            sel_entries = bonafide_entries[:half] + spoof_entries[: min(len(spoof_entries), n_samples - half)]
            dataset.entries = sel_entries

            total_samples = len(dataset)
            with self._runs_lock:
                self.active_runs[run_id].total_samples = total_samples

            loader = DataLoader(
                dataset,
                batch_size=req.batch_size,
                shuffle=False,
                collate_fn=collate_utterances,
                num_workers=0,
            )

            y_true_list = []
            y_scores_list = []
            processed_count = 0

            with self.model_service._infer_lock:
                model.eval()
                with torch.no_grad():
                    for batch in loader:
                        batched_chunks = batch["batched_chunks"].to(self.device)
                        chunk_counts = batch["chunk_counts"]
                        labels = batch["labels"].tolist()

                        if self.device.type == "cuda":
                            with torch.amp.autocast("cuda"):
                                logits = model(batched_chunks)
                        else:
                            logits = model(batched_chunks)

                        probs = torch.softmax(logits, dim=1)[:, 1]  # Spoof posterior probabilities

                        # Average across chunks per utterance
                        offset = 0
                        for count, label in zip(chunk_counts, labels):
                            utt_prob = float(torch.mean(probs[offset : offset + count]).item())
                            offset += count
                            y_true_list.append(label)
                            y_scores_list.append(utt_prob)

                        processed_count += len(labels)
                        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

                        with self._runs_lock:
                            if run_id in self.active_runs:
                                self.active_runs[run_id].processed_samples = processed_count
                                self.active_runs[run_id].percent_complete = round(
                                    (processed_count / total_samples) * 100.0, 1
                                )
                                self.active_runs[run_id].current_time_ms = round(elapsed_ms, 1)

            # Compute evaluation metrics
            y_true_arr = np.array(y_true_list, dtype=int)
            y_scores_arr = np.array(y_scores_list, dtype=float)

            threshold_used = (
                float(req.threshold)
                if req.threshold is not None
                else self.model_service.get_frozen_threshold(norm_id)
            )
            th_source = "Custom Input" if req.threshold is not None else "Frozen Development Threshold"

            metrics = compute_all_metrics(y_true_arr, y_scores_arr, threshold=threshold_used)

            # Score distributions by class
            bona_scores = y_scores_arr[y_true_arr == 0]
            spoof_scores = y_scores_arr[y_true_arr == 1]

            score_dist = {
                "bonafide": {
                    "count": float(len(bona_scores)),
                    "min": float(np.min(bona_scores)) if len(bona_scores) > 0 else 0.0,
                    "max": float(np.max(bona_scores)) if len(bona_scores) > 0 else 0.0,
                    "mean": float(np.mean(bona_scores)) if len(bona_scores) > 0 else 0.0,
                    "median": float(np.median(bona_scores)) if len(bona_scores) > 0 else 0.0,
                    "std": float(np.std(bona_scores)) if len(bona_scores) > 0 else 0.0,
                },
                "spoof": {
                    "count": float(len(spoof_scores)),
                    "min": float(np.min(spoof_scores)) if len(spoof_scores) > 0 else 0.0,
                    "max": float(np.max(spoof_scores)) if len(spoof_scores) > 0 else 0.0,
                    "mean": float(np.mean(spoof_scores)) if len(spoof_scores) > 0 else 0.0,
                    "median": float(np.median(spoof_scores)) if len(spoof_scores) > 0 else 0.0,
                    "std": float(np.std(spoof_scores)) if len(spoof_scores) > 0 else 0.0,
                },
            }

            warnings = []
            if req.split == "eval" and abs(metrics["eer_threshold"] - threshold_used) > 0.5:
                warnings.append(
                    f"Threshold Discrepancy Observed: Optimal post-hoc EER threshold on this subset "
                    f"is {metrics['eer_threshold']:.6e}, which differs significantly from frozen threshold {threshold_used:.6f}. "
                    "This threshold shift is consistent with known distribution dynamics between Dev and Eval."
                )

            total_sec = round(time.perf_counter() - t_start, 2)
            avg_latency = round((total_sec * 1000.0) / max(1, total_samples), 2)

            result = EvaluationRunResult(
                run_id=run_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                model=norm_id,
                dataset=req.dataset,
                split=req.split,
                total_utterances=total_samples,
                bonafide_count=int(np.sum(y_true_arr == 0)),
                spoof_count=int(np.sum(y_true_arr == 1)),
                threshold_used=round(threshold_used, 6),
                threshold_source=th_source,
                eer=round(metrics["eer"], 6),
                eer_percent=round(metrics["eer"] * 100.0, 2),
                eer_threshold=round(metrics["eer_threshold"], 6),
                accuracy=round(metrics["accuracy"], 4),
                accuracy_percent=round(metrics["accuracy"] * 100.0, 2),
                balanced_accuracy=round(metrics["balanced_accuracy"], 4),
                balanced_accuracy_percent=round(metrics["balanced_accuracy"] * 100.0, 2),
                macro_f1=round(metrics["f1_macro"], 4),
                macro_precision=round(metrics["precision_macro"], 4),
                macro_recall=round(metrics["recall_macro"], 4),
                roc_auc=round(metrics["roc_auc"], 4) if not np.isnan(metrics["roc_auc"]) else 0.5,
                pr_auc=round(metrics["pr_auc"], 4) if not np.isnan(metrics["pr_auc"]) else 0.5,
                confusion_matrix=ConfusionMatrixData(
                    true_negatives=metrics["true_negatives"],
                    false_positives=metrics["false_positives"],
                    false_negatives=metrics["false_negatives"],
                    true_positives=metrics["true_positives"],
                    fpr=round(metrics["false_positive_rate"], 4),
                    fnr=round(metrics["false_negative_rate"], 4),
                ),
                avg_latency_ms_per_utterance=avg_latency,
                total_duration_sec=total_sec,
                score_distribution=score_dist,
                warnings=warnings,
            )

            # Persist run output as a separate artifact without overwriting model_comparison.csv
            run_file = EVAL_RUNS_DIR / f"{run_id}.json"
            with open(run_file, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2)

            with self._runs_lock:
                self.completed_runs[run_id] = result
                self.active_runs[run_id].status = "completed"
                self.active_runs[run_id].percent_complete = 100.0

            logger.info(f"Evaluation run {run_id} completed successfully in {total_sec}s. EER: {result.eer_percent}%")

        except Exception as e:
            logger.error(f"Evaluation run {run_id} failed: {e}", exc_info=True)
            with self._runs_lock:
                if run_id in self.active_runs:
                    self.active_runs[run_id].status = "failed"
                    self.active_runs[run_id].error_message = str(e)
