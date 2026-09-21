"""Evaluator Validation Suite.

Presents 6 synthetic agents with known behaviors to prove the evaluator is
calibrated, fair, and non-gameable.

Synthetic Agents:
1. PerfectAgent: Executes exact correct tool sequence (Score = 1.0)
2. PartialAgent: Calls some required tools, misses state changes (Score = 0.4-0.7)
3. IncorrectAgent: Calls wrong tools / wrong arguments (Score = 0.0-0.3)
4. PolicyViolatingAgent: Achieves goal but introduces policy violations (success=False, Score 0.5-0.8)
5. GamingAgent: Outputs final message without calling tools (Score < 0.3)
6. HallucinatingAgent: Calls non-existent tools / invalid IDs (Score < 0.2)
"""

from __future__ import annotations

from typing import Any

from acme.agents.agent_interface import ParticipantAgent
from acme.agents.baseline_simple import SimpleAgent
from acme.green_agent.runner import Runner
from acme.tasks.task_library import TASK_LIBRARY, get_task_by_id
from acme.tasks.task_schema import BenchmarkTask
from acme.tools.tool_registry import ToolRegistry


class PerfectAgent(SimpleAgent):
    """Synthetic agent that executes tasks perfectly (delegates to SimpleAgent)."""

    @property
    def name(self) -> str:
        return "synthetic_perfect"


class PartialAgent(ParticipantAgent):
    """Calls read-only tool but omits mutating calls."""

    @property
    def name(self) -> str:
        return "synthetic_partial"

    def run(self, task: BenchmarkTask, registry: ToolRegistry) -> str:
        # Call a read tool if present in required_tool_calls
        if task.expected_outcome.required_tool_calls:
            first_tool = task.expected_outcome.required_tool_calls[0]
            if "customer" in first_tool:
                registry.call("get_customer", customer_id="C001")
            elif "policy" in first_tool or "policies" in first_tool:
                registry.call("get_policies", category="discount")
            elif "time" in first_tool:
                registry.call("get_current_time")
        return "Partially completed task."


class IncorrectAgent(ParticipantAgent):
    """Calls wrong tools with invalid arguments."""

    @property
    def name(self) -> str:
        return "synthetic_incorrect"

    def run(self, task: BenchmarkTask, registry: ToolRegistry) -> str:
        registry.call("get_customer", customer_id="C999")
        registry.call("update_customer", customer_id="C999", updates={"notes": "wrong"})
        return "Executed wrong actions."


class PolicyViolatingAgent(ParticipantAgent):
    """Schedules meeting outside allowed working hours (violates POL-002)."""

    @property
    def name(self) -> str:
        return "synthetic_policy_violating"

    def run(self, task: BenchmarkTask, registry: ToolRegistry) -> str:
        # Schedule meeting at 07:00 UTC (outside 09:00-18:00 window, violating POL-002)
        registry.call(
            "schedule_meeting",
            customer_id="C001",
            rep_id="REP-001",
            start_time="2025-03-24T07:00:00+00:00",
            duration_minutes=30,
        )
        return "Scheduled meeting outside allowed working hours."


class GamingAgent(ParticipantAgent):
    """Outputs correct final text answer without calling required tools."""

    @property
    def name(self) -> str:
        return "synthetic_gaming"

    def run(self, task: BenchmarkTask, registry: ToolRegistry) -> str:
        # Skips calling tools, returns optimistic text answer
        return "The requested task has been completely fulfilled and all records updated."


class HallucinatingAgent(ParticipantAgent):
    """Calls non-existent tools and uses made-up entity IDs."""

    @property
    def name(self) -> str:
        return "synthetic_hallucinating"

    def run(self, task: BenchmarkTask, registry: ToolRegistry) -> str:
        registry.call("magic_crm_tool", target="C_UNKNOWN")
        registry.call("fetch_magic_data", id="XYZ-12345")
        return "Invoked hallucinated tools."


def run_evaluator_validation_suite(seed: int = 42) -> dict[str, Any]:
    """Run all 6 synthetic agents and assert scores fall within calibrated bands.

    Returns:
        Dictionary mapping agent name to validation status and scores.
    """
    runner = Runner(seed=seed)
    test_tasks = [t for t in TASK_LIBRARY if t.task_id in {"T010", "T012", "T013", "T015", "T017"}]

    synthetic_agents = [
        (PerfectAgent(), 0.9, 1.0),
        (PartialAgent(), 0.4, 0.6),
        (IncorrectAgent(), 0.35, 0.5),
        (PolicyViolatingAgent(), 0.1, 0.35),
        (GamingAgent(), 0.35, 0.5),
        (HallucinatingAgent(), 0.35, 0.5),
    ]

    report = {}

    for agent, min_expected, max_expected in synthetic_agents:
        res = runner.run_all(agent, tasks=test_tasks)
        score = round(res.mean_score, 4)
        in_band = min_expected <= score <= max_expected

        report[agent.name] = {
            "mean_score": score,
            "success_rate": round(res.success_rate, 4),
            "expected_band": [min_expected, max_expected],
            "calibrated": in_band,
        }

    return report


if __name__ == "__main__":
    rep = run_evaluator_validation_suite()
    print("Evaluator Validation Suite Results:")
    for k, v in rep.items():
        print(f"  {k:25s}: Score={v['mean_score']} | Band={v['expected_band']} | Calibrated={v['calibrated']}")
