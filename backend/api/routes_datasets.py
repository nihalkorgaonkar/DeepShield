"""
Dataset Explorer API Routes
"""

from fastapi import APIRouter, Depends

from backend.schemas.datasets import DatasetsResponse
from backend.services.dataset_service import DatasetService

router = APIRouter(tags=["Datasets"])

_dataset_service = DatasetService()


def get_dataset_service():
    return _dataset_service


@router.get("/datasets", response_model=DatasetsResponse)
def get_datasets(dataset_service: DatasetService = Depends(get_dataset_service)):
    """
    Returns filesystem inventory, partition sample counts, attack algorithms,
    and preprocessed spectrogram cache status for ASVspoof 2019 and 2021.
    """
    return dataset_service.get_all_datasets_info()
