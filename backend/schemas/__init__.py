"""
DeepShield Schemas
"""

from .models import (
    CheckpointMetadata,
    ModelInfo,
    MetricComparisonRecord,
    MetricsSummaryResponse,
)
from .detection import (
    ChunkScore,
    AudioMetadata,
    DetectionResult,
    DetectionComparisonResult,
)
from .evaluation import (
    EvaluationRequest,
    EvaluationProgress,
    EvaluationRunResult,
    ConfusionMatrixData,
)
from .datasets import (
    DatasetSplitInfo,
    DatasetInfo,
    DatasetsResponse,
)
from .robustness import (
    PerturbationResult,
    RobustnessTestResponse,
)
from .reports import (
    ReportSummary,
    ReportDetail,
)
from .diagnostics import (
    DiagnosticCheckItem,
    SystemHealthResponse,
    FullDiagnosticResponse,
)

__all__ = [
    "CheckpointMetadata",
    "ModelInfo",
    "MetricComparisonRecord",
    "MetricsSummaryResponse",
    "ChunkScore",
    "AudioMetadata",
    "DetectionResult",
    "DetectionComparisonResult",
    "EvaluationRequest",
    "EvaluationProgress",
    "EvaluationRunResult",
    "ConfusionMatrixData",
    "DatasetSplitInfo",
    "DatasetInfo",
    "DatasetsResponse",
    "PerturbationResult",
    "RobustnessTestResponse",
    "ReportSummary",
    "ReportDetail",
    "DiagnosticCheckItem",
    "SystemHealthResponse",
    "FullDiagnosticResponse",
]
