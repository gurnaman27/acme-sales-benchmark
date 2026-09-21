"""Result models for the evaluator.

`TaskResult` — one task's evaluation outcome.
`BenchmarkResult` — aggregation over many tasks, with helpers for
per-category, per-difficulty, and failure-type breakdowns.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from acme.evaluation.failure_taxonomy import FailureType


class TaskResult(BaseModel):
    """Evaluation result for a single benchmark task."""
    task_id: str
    category: str
    difficulty: int
    success: bool
    score: float                     # composite score in [0, 1]

    # Sub-scores (each in [0, 1])
    state_correctness: float
    required_tools_score: float
    policy_compliance_score: float
    efficiency_score: float
    no_forbidden_score: float

    # Diagnostics
    policy_violations_before: int
    policy_violations_after: int
    policy_violations_introduced: int
    tool_call_count: int
    successful_tool_calls: int
    failed_tool_calls: int

    # Failure classification
    failure_type: FailureType | None = None
    failure_details: dict = Field(default_factory=dict)

    notes: str = ""


class BenchmarkResult(BaseModel):
    """Aggregate result across a set of tasks."""
    results: list[TaskResult] = Field(default_factory=list)

    @property
    def total_tasks(self) -> int:
        return len(self.results)

    @property
    def successful_tasks(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.successful_tasks / self.total_tasks

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / self.total_tasks

    @property
    def mean_tool_calls(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.tool_call_count for r in self.results) / self.total_tasks

    def by_category(self) -> dict[str, dict]:
        groups: dict[str, list[TaskResult]] = {}
        for r in self.results:
            groups.setdefault(r.category, []).append(r)
        return {
            cat: {
                "count": len(rs),
                "successes": sum(1 for r in rs if r.success),
                "success_rate": sum(1 for r in rs if r.success) / len(rs),
                "mean_score": sum(r.score for r in rs) / len(rs),
            }
            for cat, rs in groups.items()
        }

    def by_difficulty(self) -> dict[int, dict]:
        groups: dict[int, list[TaskResult]] = {}
        for r in self.results:
            groups.setdefault(r.difficulty, []).append(r)
        return {
            lvl: {
                "count": len(rs),
                "successes": sum(1 for r in rs if r.success),
                "success_rate": sum(1 for r in rs if r.success) / len(rs),
                "mean_score": sum(r.score for r in rs) / len(rs),
            }
            for lvl, rs in groups.items()
        }

    def failure_breakdown(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            if r.failure_type is not None:
                key = r.failure_type.value
                counts[key] = counts.get(key, 0) + 1
        return counts


__all__ = ["TaskResult", "BenchmarkResult"]