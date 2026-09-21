"""Export results to JSON and CSV.

Both functions write to a file path. Caller is responsible for
creating the parent directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from acme.evaluation.metrics import BenchmarkResult
from acme.green_agent.runner import ReliabilityResult


def export_results_json(result: BenchmarkResult, path: str | Path) -> None:
    """Write a BenchmarkResult as a structured JSON file."""
    path = Path(path)
    data = {
        "summary": {
            "total_tasks": result.total_tasks,
            "successful_tasks": result.successful_tasks,
            "success_rate": round(result.success_rate, 4),
            "mean_score": round(result.mean_score, 4),
            "mean_tool_calls": round(result.mean_tool_calls, 4),
        },
        "by_category": result.by_category(),
        "by_difficulty": {
            str(k): v for k, v in result.by_difficulty().items()
        },
        "failure_breakdown": result.failure_breakdown(),
        "tasks": [r.model_dump(mode="json") for r in result.results],
    }
    path.write_text(json.dumps(data, indent=2))


def export_results_csv(result: BenchmarkResult, path: str | Path) -> None:
    """Write a BenchmarkResult as a per-task CSV file."""
    path = Path(path)
    fieldnames = [
        "task_id", "category", "difficulty",
        "success", "score",
        "state_correctness", "required_tools_score",
        "policy_compliance_score", "efficiency_score", "no_forbidden_score",
        "policy_violations_introduced",
        "tool_call_count", "successful_tool_calls", "failed_tool_calls",
        "failure_type",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in result.results:
            writer.writerow({
                "task_id": r.task_id,
                "category": r.category,
                "difficulty": r.difficulty,
                "success": r.success,
                "score": r.score,
                "state_correctness": r.state_correctness,
                "required_tools_score": r.required_tools_score,
                "policy_compliance_score": r.policy_compliance_score,
                "efficiency_score": r.efficiency_score,
                "no_forbidden_score": r.no_forbidden_score,
                "policy_violations_introduced": r.policy_violations_introduced,
                "tool_call_count": r.tool_call_count,
                "successful_tool_calls": r.successful_tool_calls,
                "failed_tool_calls": r.failed_tool_calls,
                "failure_type": r.failure_type.value if r.failure_type else "",
            })


def export_reliability_json(
    results: list[ReliabilityResult],
    path: str | Path,
) -> None:
    """Write per-task reliability results as JSON."""
    path = Path(path)
    data = {
        "total_tasks": len(results),
        "overall_pass_at_1": (
            sum(1 for r in results if r.pass_at_1) / len(results)
            if results else 0.0
        ),
        "overall_pass_at_k": (
            sum(1 for r in results if r.pass_at_k) / len(results)
            if results else 0.0
        ),
        "overall_consistency": (
            sum(r.consistency for r in results) / len(results)
            if results else 0.0
        ),
        "tasks": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(data, indent=2))


__all__ = [
    "export_results_json",
    "export_results_csv",
    "export_reliability_json",
]