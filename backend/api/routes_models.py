"""
Model Metadata and Metrics Comparison API Routes
"""

from typing import Dict, List
from fastapi import APIRouter, Depends

from backend.schemas.models import ModelInfo, MetricsSummaryResponse
from backend.services.model_service import ModelService
from backend.services.report_service import ReportService

router = APIRouter(tags=["Models & Metrics"])


def get_model_service():
    return ModelService()


def get_report_service():
    return ReportService()


@router.get("/models", response_model=Dict[str, ModelInfo])
def get_models(model_service: ModelService = Depends(get_model_service)):
    """Returns architecture, parameters, input dimensions, and checkpoint metadata for both models."""
    return model_service.get_models_info()


@router.get("/metrics/summary", response_model=MetricsSummaryResponse)
def get_metrics_summary(report_service: ReportService = Depends(get_report_service)):
    """
    Parses and returns authoritative metrics from model_comparison.csv and model_comparison_vit.csv.
    Includes discrepancy analysis regarding the 5-orders-of-magnitude threshold shift.
    """
    return report_service.get_metrics_summary()
