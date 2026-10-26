"""
Evaluation Lab API Routes
Manages diagnostic evaluation runs on ASVspoof 2019 partitions.
"""

from typing import List, Union
from fastapi import APIRouter, Depends, HTTPException, status

from backend.schemas.evaluation import (
    EvaluationProgress,
    EvaluationRequest,
    EvaluationRunResult,
)
from backend.services.evaluation_service import EvaluationService
from backend.services.model_service import ModelService

router = APIRouter(tags=["Evaluation Lab"])

_model_service = ModelService()
_evaluation_service = EvaluationService(_model_service)


def get_evaluation_service():
    return _evaluation_service


@router.post("/evaluate")
def start_evaluation(
    req: EvaluationRequest,
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    """
    Initiates an asynchronous diagnostic evaluation run on a partition of ASVspoof 2019.
    Returns a unique run_id to monitor progress.
    """
    try:
        run_id = evaluation_service.start_evaluation(req)
        return {"run_id": run_id, "status": "queued", "message": f"Evaluation run {run_id} initiated."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start evaluation run: {str(e)}",
        )


@router.get("/evaluate/{run_id}", response_model=Union[EvaluationRunResult, EvaluationProgress])
def get_evaluation_status_or_result(
    run_id: str,
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    """
    Retrieves the status (if still running) or completed metrics result of an evaluation run.
    """
    # Check if completed first
    result = evaluation_service.get_run_result(run_id)
    if result:
        return result

    # Check if in progress
    progress = evaluation_service.get_run_status(run_id)
    if progress:
        return progress

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Evaluation run '{run_id}' not found.",
    )


@router.get("/evaluate", response_model=List[EvaluationRunResult])
def list_evaluation_runs(
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    """Returns a list of all historical completed evaluation runs."""
    return evaluation_service.list_all_runs()
