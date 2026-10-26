"""
Pydantic schemas for Evaluation Lab and diagnostic runs.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    model: str = Field(default="efficientnet_b0", description="'efficientnet_b0' or 'vit_b16'")
    dataset: str = Field(default="ASVspoof2019_LA", description="Dataset identifier")
    split: str = Field(default="dev", description="'dev' or 'eval'")
    max_samples: int = Field(default=100, ge=10, le=2000, description="Diagnostic subset sample limit")
    threshold: Optional[float] = Field(default=None, description="Custom threshold; defaults to frozen dev threshold")
    batch_size: int = Field(default=16, ge=1, le=64)


class EvaluationProgress(BaseModel):
    run_id: str
    status: str  # "queued", "running", "completed", "failed"
    model: str
    dataset: str
    split: str
    processed_samples: int
    total_samples: int
    percent_complete: float
    current_time_ms: float
    error_message: Optional[str] = None


class ConfusionMatrixData(BaseModel):
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    fpr: float
    fnr: float


class EvaluationRunResult(BaseModel):
    run_id: str
    timestamp: str
    model: str
    dataset: str
    split: str
    total_utterances: int
    bonafide_count: int
    spoof_count: int
    threshold_used: float
    threshold_source: str
    eer: float
    eer_percent: float
    eer_threshold: float
    accuracy: float
    accuracy_percent: float
    balanced_accuracy: float
    balanced_accuracy_percent: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: ConfusionMatrixData
    avg_latency_ms_per_utterance: float
    total_duration_sec: float
    score_distribution: Dict[str, Dict[str, float]]
    warnings: List[str] = Field(default_factory=list)
