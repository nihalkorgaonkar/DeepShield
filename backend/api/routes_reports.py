"""
Research Reports API Routes
Serves research documents, audit reports, and downloadable evaluation packages.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from backend.schemas.reports import ReportDetail, ReportSummary
from backend.services.report_service import ReportService

router = APIRouter(tags=["Research Reports"])

_report_service = ReportService()


def get_report_service():
    return _report_service


@router.get("/reports", response_model=List[ReportSummary])
def list_reports(report_service: ReportService = Depends(get_report_service)):
    """Returns summaries of all research documents and audit logs available in DeepShield."""
    return report_service.list_reports()


@router.get("/reports/{report_id}", response_model=ReportDetail)
def get_report(report_id: str, report_service: ReportService = Depends(get_report_service)):
    """Fetches the full markdown content and metadata for a specific research report."""
    detail = report_service.get_report_detail(report_id)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{report_id}' not found.",
        )
    return detail
