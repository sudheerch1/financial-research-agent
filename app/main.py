"""
FastAPI wrapper for the financial-research-agent LangGraph.

Stub stage: routes, models, and run-state plumbing are wired and testable
end-to-end with a fake graph runner. Swap `fake_run_graph` /
`fake_resume_graph` for real calls into your compiled graph
(graph.invoke / graph.stream + Command(resume=...)) in the next pass.

Run locally:
    uvicorn app.main:app --reload --port 8000

Test with PowerShell:
    Invoke-RestMethod -Uri http://localhost:8000/run -Method Post `
        -ContentType "application/json" `
        -Body (@{query="Compare AAPL and MSFT FY2024 revenue"} | ConvertTo-Json)
"""

from __future__ import annotations

import time

from fastapi import BackgroundTasks, FastAPI, HTTPException

from app.models import (
    ResultResponse,
    ResumeRequest,
    ResumeResponse,
    RunRequest,
    RunResponse,
    StatusResponse,
)
from app.run_store import RunStatus, run_store

app = FastAPI(title="Financial Research Agent API", version="0.1.0")


# ---------------------------------------------------------------------------
# Graph execution — STUBBED. Replace these two functions with real LangGraph
# calls. Keep the signatures the same so the route handlers don't change.
# ---------------------------------------------------------------------------

def fake_run_graph(run_id: str, query: str) -> None:
    """Stand-in for graph.invoke(...). Simulates: plan -> rag_search -> ...
    -> human_review interrupt. Replace body with real graph.stream() call,
    persisting graph_thread_config (e.g. {"configurable": {"thread_id": run_id}})
    so /resume can reconnect to the same checkpoint."""
    run_store.update_run(run_id, status=RunStatus.RUNNING, current_node="node_plan")
    time.sleep(0.5)  # placeholder for real node latency

    run_store.update_run(run_id, current_node="node_rag_search")
    time.sleep(0.5)

    # Simulate hitting the human-in-the-loop interrupt
    run_store.update_run(
        run_id,
        status=RunStatus.PAUSED_FOR_REVIEW,
        current_node="node_human_review",
        interrupt_payload={
            "reason": "conflict_detected",
            "summary": "STUB: RAG figure disagrees with web-search figure.",
        },
        graph_thread_config={"configurable": {"thread_id": run_id}},
    )


def fake_resume_graph(run_id: str, decision: str, notes: str | None) -> None:
    """Stand-in for graph.invoke(Command(resume=...), config=graph_thread_config).
    Replace with the real resume call using the stored graph_thread_config."""
    run_store.update_run(run_id, status=RunStatus.RUNNING, current_node="node_write_report")
    time.sleep(0.5)

    run_store.update_run(
        run_id,
        status=RunStatus.COMPLETED,
        current_node=None,
        interrupt_payload=None,
        result={
            "report": f"STUB REPORT (decision={decision!r}, notes={notes!r}).",
            "metadata": {"mode": "report_mode_stub"},
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/run", response_model=RunResponse)
def run(request: RunRequest, background_tasks: BackgroundTasks) -> RunResponse:
    record = run_store.create_run(input_payload=request.model_dump())
    background_tasks.add_task(fake_run_graph, record.run_id, request.query)
    return RunResponse(run_id=record.run_id, status=record.status.value)


@app.get("/status/{run_id}", response_model=StatusResponse)
def status(run_id: str) -> StatusResponse:
    record = run_store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return StatusResponse(**record.to_status_dict())


@app.post("/resume/{run_id}", response_model=ResumeResponse)
def resume(run_id: str, request: ResumeRequest, background_tasks: BackgroundTasks) -> ResumeResponse:
    record = run_store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    if record.status != RunStatus.PAUSED_FOR_REVIEW:
        raise HTTPException(
            status_code=409,
            detail=f"run_id is in status '{record.status.value}', not awaiting resume",
        )
    background_tasks.add_task(fake_resume_graph, run_id, request.decision, request.notes)
    run_store.update_run(run_id, status=RunStatus.RUNNING)
    return ResumeResponse(run_id=run_id, status=RunStatus.RUNNING.value)


@app.get("/result/{run_id}", response_model=ResultResponse)
def result(run_id: str) -> ResultResponse:
    record = run_store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    if record.status != RunStatus.COMPLETED:
        return ResultResponse(run_id=run_id, status=record.status.value, error=record.error)
    return ResultResponse(
        run_id=run_id,
        status=record.status.value,
        report=record.result.get("report") if record.result else None,
        metadata=record.result.get("metadata") if record.result else None,
    )
