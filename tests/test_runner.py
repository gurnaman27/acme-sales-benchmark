"""Tests for the batch runner, reliability computation, and export."""

import json
from pathlib import Path

import pytest

from acme.agents.baseline_simple import SimpleAgent
from acme.green_agent.cli import main
from acme.green_agent.export import (
    export_reliability_json,
    export_results_csv,
    export_results_json,
)
from acme.green_agent.runner import Runner
from acme.tasks.task_library import TASK_LIBRARY, get_task_by_id
from acme.tasks.task_schema import (
    BenchmarkTask,
    ExpectedOutcome,
    TaskCategory,
)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def test_runner_runs_full_library():
    result = Runner(seed=42).run_all(SimpleAgent())
    assert result.total_tasks == 30
    assert result.success_rate == 1.0
    assert result.mean_score == 1.0


def test_runner_subset_of_tasks():
    subset = [t for t in TASK_LIBRARY if t.task_id in {"T001", "T004"}]
    result = Runner(seed=42).run_all(SimpleAgent(), tasks=subset)
    assert result.total_tasks == 2
    assert result.success_rate == 1.0


def test_runner_accepts_empty_tasks_list():
    result = Runner(seed=42).run_all(SimpleAgent(), tasks=[])
    assert result.total_tasks == 0
    assert result.success_rate == 0.0


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------

def test_reliability_deterministic_agent_is_perfect():
    """SimpleAgent with same seed is deterministic — consistency should be 1.0."""
    task = get_task_by_id("T004")
    rel = Runner(seed=42).run_with_reliability(task, SimpleAgent(), k=5)
    assert rel.task_id == "T004"
    assert rel.runs == 5
    assert rel.successes == 5
    assert rel.pass_at_1 is True
    assert rel.pass_at_k is True
    assert rel.consistency == 1.0
    assert rel.score_stddev == 0.0


def test_reliability_failing_task():
    """Task with no handler → all runs fail."""
    task = BenchmarkTask(
        task_id="T_FAKE",
        category=TaskCategory.FOLLOW_UP,
        difficulty=1,
        instruction="N/A",
        expected_outcome=ExpectedOutcome(required_tool_calls=["get_customer"]),
    )
    rel = Runner(seed=42).run_with_reliability(task, SimpleAgent(), k=3)
    assert rel.runs == 3
    assert rel.successes == 0
    assert rel.pass_at_1 is False
    assert rel.pass_at_k is False
    assert rel.consistency == 0.0


def test_reliability_k_must_be_positive():
    task = get_task_by_id("T004")
    with pytest.raises(ValueError):
        Runner(seed=42).run_with_reliability(task, SimpleAgent(), k=0)


def test_run_all_with_reliability_returns_one_per_task():
    results = Runner(seed=42).run_all_with_reliability(SimpleAgent(), k=2)
    assert len(results) == len(TASK_LIBRARY)
    for r in results:
        assert r.runs == 2
        assert r.successes == 2


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_export_json(tmp_path: Path):
    result = Runner(seed=42).run_all(SimpleAgent())
    out = tmp_path / "results.json"
    export_results_json(result, out)

    assert out.exists()
    data = json.loads(out.read_text())

    assert data["summary"]["total_tasks"] == 30
    assert data["summary"]["success_rate"] == 1.0
    assert "by_category" in data
    assert "by_difficulty" in data
    assert "tasks" in data
    assert len(data["tasks"]) == 30


def test_export_csv(tmp_path: Path):
    result = Runner(seed=42).run_all(SimpleAgent())
    out = tmp_path / "results.csv"
    export_results_csv(result, out)

    assert out.exists()
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 31  # header + 30 tasks
    assert "task_id" in lines[0]
    assert "T001" in lines[1]  # first task row


def test_export_reliability_json(tmp_path: Path):
    rel = Runner(seed=42).run_all_with_reliability(SimpleAgent(), k=3)
    out = tmp_path / "reliability.json"
    export_reliability_json(rel, out)

    assert out.exists()
    data = json.loads(out.read_text())
    assert data["total_tasks"] == 30
    assert data["overall_consistency"] == 1.0
    assert len(data["tasks"]) == 30


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_runs_full_library(capsys):
    exit_code = main(["--agent", "simple"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "ACME SALES BENCHMARK" in out
    assert "Success" in out
    assert "30/30" in out


def test_cli_subset_of_tasks(capsys):
    exit_code = main(["--agent", "simple", "--tasks", "T001,T004"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "2/2" in out


def test_cli_unknown_agent(capsys):
    exit_code = main(["--agent", "does_not_exist"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "Unknown agent" in err


def test_cli_unknown_task(capsys):
    exit_code = main(["--agent", "simple", "--tasks", "T999"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "Unknown task" in err


def test_cli_writes_output(tmp_path: Path, capsys):
    exit_code = main([
        "--agent", "simple",
        "--output", str(tmp_path),
    ])
    assert exit_code == 0
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.csv").exists()


def test_cli_reliability_mode(tmp_path: Path, capsys):
    exit_code = main([
        "--agent", "simple",
        "--k", "3",
        "--output", str(tmp_path),
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "pass@1" in out
    assert "pass@3" in out
    assert "consistency" in out
    assert (tmp_path / "reliability.json").exists()
    assert (tmp_path / "results.json").exists()