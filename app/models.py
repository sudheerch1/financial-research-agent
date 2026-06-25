"""
Pydantic schemas for the financial-research-agent FastAPI wrapper.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Body for POST /run — kicks off a new graph execution."""
    query: str = Field(..., description="The research question, e.g. 'Compare AAPL and MSFT FY2024 revenue'")
    companies: Optional[list[str]] = Field(
        default=None, description="Optional explicit ticker/company list, e.g. ['AAPL', 'MSFT']"
    )


class RunResponse(BaseModel):
    """Response for POST /run."""
    run_id: str
    status: str


class StatusResponse(BaseModel):
    """Response for GET /status/{run_id}."""
    run_id: str
    status: str
    current_node: Optional[str] = None
    interrupt_payload: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class ResumeRequest(BaseModel):
    """Body for POST /resume/{run_id} — answers the human-review interrupt."""
    decision: str = Field(..., description="e.g. 'approve', 'reject', or free-text human feedback")
    notes: Optional[str] = None


class ResumeResponse(BaseModel):
    run_id: str
    status: str


class ResultResponse(BaseModel):
    """Response for GET /result/{run_id}."""
    run_id: str
    status: str
    report: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    error: Optional[str] = None
