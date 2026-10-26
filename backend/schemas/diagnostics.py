"""
Pydantic schemas for System Diagnostics and Health Checks.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class DiagnosticCheckItem(BaseModel):
    name: str
    category: str
    status: str  # "PASS", "WARN", "FAIL", "INFO"
    message: str
    details: Optional[Dict[str, Any]] = None


class SystemHealthResponse(BaseModel):
    status: str  # "healthy", "degraded", "error"
    timestamp: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_device_name: Optional[str] = None
    cuda_device_count: int = 0
    vram_total_gb: Optional[float] = None
    vram_allocated_gb: Optional[float] = None
    vram_reserved_gb: Optional[float] = None
    active_device: str
    loaded_models: List[str] = Field(default_factory=list)
    dataset_status: str
    preprocessed_cache_status: str


class FullDiagnosticResponse(BaseModel):
    overall_health: str
    timestamp: str
    system_health: SystemHealthResponse
    checks: List[DiagnosticCheckItem]
    recent_errors: List[str] = Field(default_factory=list)
