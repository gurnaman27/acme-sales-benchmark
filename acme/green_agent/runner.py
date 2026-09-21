"""Batch runner with pass@k support.

The Runner executes tasks against a participant agent and aggregates
results. It supports both a single pass (`run_all`) and reliability
measurement (`run_with_reliability`).

Reliability semantics:
  - pass@1        — did the FIRST of k runs succeed?
  - pass@k        — did ANY of k runs succeed?
  - consistency   — fraction of runs that succeeded (== pass^k)
  - score_stddev  — variability of scores across runs

For deterministic agents (rule-based), all metrics are trivial. For
nondeterministic agents (LLMs), they capture real reliability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from acme.agents.agent_interface import ParticipantAgent
from acme.evaluation.evaluator import Evaluator
from acme.evaluation.metrics import BenchmarkResult
from acme.green_agent.green_agent import GreenAgent
from acme.tasks.task_library import TASK_LIBRARY
from acme.tasks.task_schema import BenchmarkTask


@dataclass
class ReliabilityResult:
    """Reliability summary for a single task run k times."""
    task_id: str
    runs: int
    successes: int
    scores: list[float] = field(default_factory=list)
    pass_at_1: bool = False
    pass_at_k: bool = False
    consistency: float = 0.0
    mean_score: float = 0.0
    score_stddev: float = 0.0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "runs": self.runs,
            "successes": self.successes,
            "pass_at_1": self.pass_at_1,
            "pass_at_k": self.pass_at_k,
            "consistency": round(self.consistency, 4),
            "mean_score": round(self.mean_score, 4),
            "score_stddev": round(self.score_stddev, 4),
        }


class Runner:
    """Batch runner for benchmark tasks."""

    def __init__(
        self,
        seed: int = 42,
        evaluator: Evaluator | None = None,
    ):
        self.seed = seed
        self.evaluator = evaluator

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

    def run_all(
        self,
        agent: ParticipantAgent,
        tasks: list[BenchmarkTask] | None = None,
    ) -> BenchmarkResult:
        tasks = tasks if tasks is not None else TASK_LIBRARY
        ga = GreenAgent(seed=self.seed, evaluator=self.evaluator)
        results = []
        for i, t in enumerate(tasks):
            if i > 0 and ("llm" in agent.name or hasattr(agent, "provider")):
                import time
                time.sleep(1.0)
            results.append(ga.run_task(t, agent))
        return BenchmarkResult(results=results)

    # ------------------------------------------------------------------
    # Reliability
    # ------------------------------------------------------------------

    def run_with_reliability(
        self,
        task: BenchmarkTask,
        agent: ParticipantAgent,
        k: int = 5,
    ) -> ReliabilityResult:
        """Run one task k times and compute reliability metrics."""
        if k < 1:
            raise ValueError("k must be >= 1")

        ga = GreenAgent(seed=self.seed, evaluator=self.evaluator)
        results = [ga.run_task(task, agent) for _ in range(k)]

        scores = [r.score for r in results]
        successes = sum(1 for r in results if r.success)
        mean = sum(scores) / k
        variance = sum((s - mean) ** 2 for s in scores) / k
        stddev = variance ** 0.5

        return ReliabilityResult(
            task_id=task.task_id,
            runs=k,
            successes=successes,
            scores=scores,
            pass_at_1=results[0].success,
            pass_at_k=successes > 0,
            consistency=successes / k,
            mean_score=mean,
            score_stddev=stddev,
        )

    def run_all_with_reliability(
        self,
        agent: ParticipantAgent,
        k: int = 5,
        tasks: list[BenchmarkTask] | None = None,
    ) -> list[ReliabilityResult]:
        """Run every task k times and return reliability for each."""
        tasks = tasks if tasks is not None else TASK_LIBRARY
        return [self.run_with_reliability(t, agent, k) for t in tasks]


__all__ = ["Runner", "ReliabilityResult"]