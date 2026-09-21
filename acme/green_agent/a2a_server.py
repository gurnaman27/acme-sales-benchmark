"""A2A-compatible HTTP interface for the Acme Sales Green Agent.

Provides AgentBeats / A2A specification endpoints:
  - GET  /.well-known/agent-card.json -> Metadata & capability card
  - GET  /health                      -> Health check
  - POST /run                        -> Execute benchmark run on request
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from acme.green_agent.cli import _build_agent
from acme.green_agent.runner import Runner
from acme.tasks.task_library import TASK_LIBRARY, get_task_by_id


app = FastAPI(
    title="Acme Sales Green Agent Benchmark API",
    version="0.1.0",
    description="A2A-compatible API for benchmark execution and evaluation.",
)


class RunRequest(BaseModel):
    agent_name: str = "simple"
    task_ids: list[str] | None = None
    seed: int = 42
    k: int = 1


class TaskResultOutput(BaseModel):
    task_id: str
    category: str
    difficulty: int
    success: bool
    score: float
    tool_call_count: int
    policy_violations_introduced: int
    failure_type: str | None = None


class RunResponse(BaseModel):
    agent_name: str
    total_tasks: int
    successful_tasks: int
    success_rate: float
    mean_score: float
    results: list[TaskResultOutput]


@app.get("/.well-known/agent-card.json")
def agent_card() -> dict[str, Any]:
    """Return standard A2A agent card metadata."""
    return {
        "name": "acme-sales-green-agent",
        "version": "0.1.0",
        "description": "Benchmark evaluation agent for Acme Sales CRM environment",
        "capabilities": ["benchmark", "evaluate", "pass_at_k"],
        "endpoints": {
            "agent_card": "/.well-known/agent-card.json",
            "health": "/health",
            "run": "/run",
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "acme-sales-green-agent"}


@app.post("/run", response_model=RunResponse)
def run_benchmark(req: RunRequest) -> RunResponse:
    """Execute a benchmark evaluation request."""
    try:
        agent = _build_agent(req.agent_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if req.task_ids:
        tasks = []
        for tid in req.task_ids:
            t = get_task_by_id(tid)
            if t is None:
                raise HTTPException(
                    status_code=400, detail=f"Unknown task ID: {tid}"
                )
            tasks.append(t)
    else:
        tasks = TASK_LIBRARY

    runner = Runner(seed=req.seed)
    res = runner.run_all(agent, tasks=tasks)

    formatted_results = [
        TaskResultOutput(
            task_id=r.task_id,
            category=r.category,
            difficulty=r.difficulty,
            success=r.success,
            score=round(r.score, 4),
            tool_call_count=r.tool_call_count,
            policy_violations_introduced=r.policy_violations_introduced,
            failure_type=r.failure_type.value if r.failure_type else None,
        )
        for r in res.results
    ]

    return RunResponse(
        agent_name=agent.name,
        total_tasks=res.total_tasks,
        successful_tasks=res.successful_tasks,
        success_rate=round(res.success_rate, 4),
        mean_score=round(res.mean_score, 4),
        results=formatted_results,
    )
