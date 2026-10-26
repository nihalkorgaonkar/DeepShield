"""
Pydantic schemas for Model metadata, comparison, and metrics summaries.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CheckpointMetadata(BaseModel):
    model_name: str
    exists: bool
    path: str
    size_bytes: int = 0
    size_mb: float = 0.0
    sha256_hash: str = ""
    epoch: Optional[int] = None
    dev_eer: Optional[float] = None
    dev_threshold: Optional[float] = None
    seed: Optional[int] = None
    last_modified: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    name: str
    architecture: str
    family: str  # "CNN" or "Vision Transformer"
    parameters: int
    parameters_million: float
    input_resolution: str
    feature_extractor: str
    classification_head: str
    frozen_threshold: float
    threshold_criterion: str
    dev_eer: float
    eval_eer: Optional[float] = None
    eval_roc_auc: Optional[float] = None
    avg_inference_latency_ms: Optional[float] = None
    checkpoint: CheckpointMetadata


class MetricComparisonRecord(BaseModel):
    model: str
    dataset: str
    split: str
    parameters_m: float
    inference_time_ms_per_utterance: float
    eer: float
    eer_percent: float
    eer_threshold: float
    eval_threshold: float
    accuracy: float
    accuracy_percent: float
    balanced_accuracy: float
    balanced_accuracy_percent: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    bonafide_precision: float
    bonafide_recall: float
    bonafide_f1: float
    spoof_precision: float
    spoof_recall: float
    spoof_f1: float
    roc_auc: float
    pr_auc: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    false_positive_rate: float
    false_negative_rate: float


class MetricsSummaryResponse(BaseModel):
    efficientnet_baseline: Optional[MetricComparisonRecord] = None
    vit_b16_model: Optional[MetricComparisonRecord] = None
    discrepancy_analysis: Dict[str, str] = Field(default_factory=dict)
    discrepancy_warning: bool = False
    warning_details: Optional[str] = None
