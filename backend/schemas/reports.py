"""
Pydantic schemas for Research Reports and Document exploration.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ReportSummary(BaseModel):
    id: str
    title: str
    category: str  # "Audit", "Evaluation", "Protocol", "Generalization"
    description: str
    filename: str
    last_modified: str
    size_bytes: int


class ReportDetail(BaseModel):
    id: str
    title: str
    category: str
    filename: str
    markdown_content: str
    last_modified: str
    associated_artifacts: List[str] = Field(default_factory=list)
