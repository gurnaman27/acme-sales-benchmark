"""End-to-end tests for the Green Agent + SimpleAgent pipeline."""

from acme.agents.baseline_simple import SimpleAgent
from acme.evaluation.failure_taxonomy import FailureType
from acme.green_agent.green_agent import GreenAgent
from acme.tasks.task_library import TASK_LIBRARY, get_task_by_id


# ---------------------------------------------------------------------------
# Single-task runs
# ---------------------------------------------------------------------------

def test_green_agent_runs_l1_task():
    task = get_task_by_id("T004")
    agent = SimpleAgent()
    result = GreenAgent(seed=42).run_task(task, agent)
    assert result.task_id == "T004"
    assert result.success is True
    assert result.score == 1.0


def test_green_agent_runs_l2_task():
    task = get_task_by_id("T014")
    agent = SimpleAgent()
    result = GreenAgent(seed=42).run_task(task, agent)
    assert result.task_id == "T014"
    assert result.success is True


def test_green_agent_detects_missing_handler():
    """A task with no handler should be classified as a planning failure."""
    # Build a synthetic task the SimpleAgent has no handler for
    from acme.tasks.task_schema import (
        BenchmarkTask, ExpectedOutcome, TaskCategory,
    )
    task = BenchmarkTask(
        task_id="T_FAKE",
        category=TaskCategory.FOLLOW_UP,
        difficulty=1,
        instruction="Does not matter",
        expected_outcome=ExpectedOutcome(
            required_tool_calls=["get_customer"],
        ),
    )
    result = GreenAgent(seed=42).run_task(task, SimpleAgent())
    assert result.success is False
    assert result.failure_type == FailureType.PLANNING


def test_green_agent_uses_reference_time_seed():
    """Runs with seed=42 should produce identical results for identical tasks."""
    task = get_task_by_id("T004")
    r1 = GreenAgent(seed=42).run_task(task, SimpleAgent())
    r2 = GreenAgent(seed=42).run_task(task, SimpleAgent())
    assert r1.score == r2.score
    assert r1.success == r2.success
    assert r1.tool_call_count == r2.tool_call_count


def test_green_agent_different_seeds_can_differ():
    """Runs with different seeds may produce different states."""
    task = get_task_by_id("T002")  # Healthcare count
    r42 = GreenAgent(seed=42).run_task(task, SimpleAgent())
    r7 = GreenAgent(seed=7).run_task(task, SimpleAgent())
    # Both should still succeed — but the exact count may differ
    assert r42.success is True
    assert r7.success is True


# ---------------------------------------------------------------------------
# Full library run
# ---------------------------------------------------------------------------

def test_green_agent_runs_entire_library():
    """The SimpleAgent should solve every L1-L4 task it has handlers for."""
    agent = SimpleAgent()
    ga = GreenAgent(seed=42)

    results = []
    for task in TASK_LIBRARY:
        results.append(ga.run_task(task, agent))

    assert len(results) == len(TASK_LIBRARY)
    # The SimpleAgent handles all 30 tasks, so all should pass
    passed = sum(1 for r in results if r.success)
    assert passed == 30, (
        f"Expected 30/30 pass, got {passed}. "
        f"Failed: {[r.task_id for r in results if not r.success]}"
    )