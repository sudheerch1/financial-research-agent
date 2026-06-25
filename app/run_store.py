"""
In-memory run state store for the FastAPI wrapper.

Swap-out point for DynamoDB later: keep the same get/set/list interface
so callers (main.py) never need to change when the backing store does.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from threading import Lock


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED_FOR_REVIEW = "paused_for_review"   # interrupt() hit, awaiting /resume
    COMPLETED = "completed"
    FAILED = "failed"


class RunRecord:
    def __init__(self, run_id: str, input_payload: dict[str, Any]):
        self.run_id = run_id
        self.input_payload = input_payload
        self.status: RunStatus = RunStatus.PENDING
        self.current_node: Optional[str] = None
        self.interrupt_payload: Optional[dict[str, Any]] = None  # what node_human_review asked
        self.result: Optional[dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        # Holds whatever LangGraph needs to resume (e.g. thread_id / checkpoint config)
        self.graph_thread_config: Optional[dict[str, Any]] = None

    def to_status_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "current_node": self.current_node,
            "interrupt_payload": self.interrupt_payload,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class InMemoryRunStore:
    """Thread-safe in-memory store. Single-process only — fine for the demo,
    not for ECS Fargate with >1 task (DynamoDB migration note for later)."""

    def __init__(self):
        self._runs: dict[str, RunRecord] = {}
        self._lock = Lock()

    def create_run(self, input_payload: dict[str, Any]) -> RunRecord:
        run_id = str(uuid.uuid4())
        record = RunRecord(run_id, input_payload)
        with self._lock:
            self._runs[run_id] = record
        return record

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            return self._runs.get(run_id)

    def update_run(self, run_id: str, **fields: Any) -> Optional[RunRecord]:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return None
            for key, value in fields.items():
                setattr(record, key, value)
            record.updated_at = datetime.now(timezone.utc)
            return record

    def list_runs(self) -> list[RunRecord]:
        with self._lock:
            return list(self._runs.values())


# Module-level singleton — one store per process.
# TODO(post-demo): replace with DynamoDB-backed implementation behind the same interface.
run_store = InMemoryRunStore()
