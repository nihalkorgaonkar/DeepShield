"""
Pydantic schemas for Audio Detection and Dual-Model Comparison.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class ChunkScore(BaseModel):
    chunk_index: int
    start_time_sec: float
    end_time_sec: float
    spoof_score: float
    bonafide_score: float
    is_spoof: bool


class AudioMetadata(BaseModel):
    filename: str
    duration_sec: float
    sample_rate_hz: int
    num_channels: int
    num_samples: int
    file_size_bytes: int
    file_size_formatted: str
    rms_energy: float
    waveform_downsampled: List[float] = Field(
        default_factory=list,
        description="Downsampled waveform envelope (e.g. 100 points) for client-side rendering"
    )


class DetectionResult(BaseModel):
    model_id: str
    model_name: str
    prediction: str  # "bonafide" or "spoof"
    prediction_label: str  # "Genuine Speech" or "Synthetic / Spoof Speech"
    spoof_score: float
    bonafide_score: float
    confidence_percent: float
    configured_threshold: float
    is_above_threshold: bool
    total_inference_time_ms: float
    num_chunks_analyzed: int
    chunk_scores: List[ChunkScore]
    audio_metadata: AudioMetadata
    preprocessing_summary: Dict[str, str]
    disclaimer: str = (
        "DeepShield provides research-oriented audio spoofing estimates. "
        "Predictions may be incorrect, particularly on audio types or recording conditions "
        "outside the training distribution. Do not treat a single prediction as definitive proof of authenticity."
    )


class DetectionComparisonResult(BaseModel):
    efficientnet_result: DetectionResult
    vit_result: DetectionResult
    agreement: bool
    agreement_status: str  # "AGREEMENT" or "DISAGREEMENT"
    disagreement_warning: Optional[str] = None
    faster_model: str
    latency_difference_ms: float
    audio_metadata: AudioMetadata
    disclaimer: str = (
        "DeepShield provides research-oriented audio spoofing estimates. "
        "Predictions may be incorrect, particularly on audio types or recording conditions "
        "outside the training distribution. Do not treat a single prediction as definitive proof of authenticity."
    )
