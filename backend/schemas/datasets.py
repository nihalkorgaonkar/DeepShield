"""
Pydantic schemas for Dataset Explorer and status.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class DatasetSplitInfo(BaseModel):
    split: str
    available: bool
    bonafide_count: int
    spoof_count: int
    total_count: int
    cached_chunks_count: Optional[int] = None
    has_manifest: bool
    audio_dir_exists: bool
    protocol_file_exists: bool
    attack_types: List[str] = Field(default_factory=list)


class DatasetInfo(BaseModel):
    id: str
    name: str
    description: str
    status: str  # "AVAILABLE", "PARTIAL", "DOCUMENTED_MISSING_AUDIO"
    data_root: str
    splits: List[DatasetSplitInfo]
    notes: str
    zero_fabrication_statement: Optional[str] = None


class DatasetsResponse(BaseModel):
    datasets: List[DatasetInfo]
    preprocessed_cache_dir: str
    cache_available: bool
