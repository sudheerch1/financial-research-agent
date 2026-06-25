"""
FastAPI wrapper for the financial-research-agent LangGraph.

Wired to the real compiled graph (src.agent.graph.agent_graph), using the
same stream()/interrupt()/Command(resume=...) pattern as run_agent.py.
Each run_id becomes the LangGraph thread_id, so the DynamoDB checkpointer
keeps the actual graph state durable across requests; run_store.py only
tracks API-facing status for polling.

Run locally:
    uvicorn app.main:app --reload --port 8000

Test with PowerShell:
    Invoke-RestMethod -Uri http://localhost:8000/run -Method Post `
        -ContentType "application/json" `
        -Body (@{query="Compare AAPL and MSFT FY2024 revenue"} | ConvertTo-Json)
"""

from __future__ import annotations

from fastapi import BackgroundTasks, FastAPI, HTTPException
from langgraph.types import Command

from app.models import (
    ResultResponse,
    ResumeRequest,
    ResumeResponse,
    RunRequest,
    RunResponse,
    StatusResponse,
)
from app.run_store import RunStatus, run_store
from src.agent.graph import agent_graph
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Financial Research Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Graph execution — wired to the real compiled graph (DynamoDB-checkpointed).
# Mirrors the stream/interrupt pattern from run_agent.py exactly, so behavior
# matches what's already been validated by hand at the CLI.
# ---------------------------------------------------------------------------

def _stream_and_track(run_id: str, graph_input: dict | Command, config: dict) -> None:
    """Drives one leg of graph execution (either the initial run or a resume)
    via stream(), updating run_store as each node completes. Stops and marks
    PAUSED_FOR_REVIEW if an interrupt is hit; marks COMPLETED on final_report."""
    try:
        for step in agent_graph.stream(graph_input, config=config, stream_mode="updates"):
            if "__interrupt__" in step:
                payload = step["__interrupt__"][0].value
                run_store.update_run(
                    run_id,
                    status=RunStatus.PAUSED_FOR_REVIEW,
                    current_node="node_human_review",
                    interrupt_payload=payload,
                    graph_thread_config=config,
                )
                return

            for node_name, output in step.items():
                run_store.update_run(run_id, current_node=node_name)
                if output and "final_report" in output:
                    run_store.update_run(
                        run_id,
                        status=RunStatus.COMPLETED,
                        current_node=None,
                        interrupt_payload=None,
                        result={
                            "report": output["final_report"],
                            "metadata": {"report_mode": output.get("report_mode")},
                        },
                    )
                    return
    except Exception as exc:  # noqa: BLE001 — surface any graph-level failure via /status
        run_store.update_run(run_id, status=RunStatus.FAILED, error=str(exc))


def run_graph(run_id: str, question: str) -> None:
    config = {"configurable": {"thread_id": run_id}}
    run_store.update_run(run_id, status=RunStatus.RUNNING, current_node="node_plan", graph_thread_config=config)
    _stream_and_track(run_id, {"question": question}, config)


def resume_graph(run_id: str, decision: str) -> None:
    record = run_store.get_run(run_id)
    config = record.graph_thread_config
    run_store.update_run(run_id, status=RunStatus.RUNNING, current_node="node_write_report")
    _stream_and_track(run_id, Command(resume=decision), config)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/run", response_model=RunResponse)
def run(request: RunRequest, background_tasks: BackgroundTasks) -> RunResponse:
    record = run_store.create_run(input_payload=request.model_dump())
    background_tasks.add_task(run_graph, record.run_id, request.query)
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
    background_tasks.add_task(resume_graph, run_id, request.decision)
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
