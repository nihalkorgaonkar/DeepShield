"""
Pydantic schemas for Robustness stress-testing.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PerturbationResult(BaseModel):
    condition_name: str
    description: str
    parameter_description: str
    spoof_score: float
    bonafide_score: float
    prediction: str
    score_delta_from_clean: float
    waveform_downsampled: List[float] = Field(default_factory=list)


class RobustnessTestResponse(BaseModel):
    filename: str
    model_id: str
    clean_score: float
    clean_prediction: str
    perturbations: List[PerturbationResult]
    benchmark_dataset_summary: Optional[Dict[str, Dict[str, float]]] = None
    disclaimer: str = (
        "Individual audio robustness tests illustrate single-sample acoustic sensitivity. "
        "They do not constitute a population-level generalization benchmark."
    )
