"""Green Agent — orchestrates one benchmark task end-to-end.

Workflow:
    1. Generate a fresh seed state
    2. Apply the task's initial_state_patch
    3. Snapshot the "before" state
    4. Build a ToolRegistry with the task's allowed_tools
    5. Run the participant agent
    6. Capture the "after" state and action log
    7. Evaluate
    8. Return a TaskResult
"""

from __future__ import annotations

from acme.agents.agent_interface import ParticipantAgent
from acme.environment.data_generator import generate_seed_data
from acme.evaluation.evaluator import Evaluator
from acme.evaluation.metrics import TaskResult
from acme.tasks.task_schema import BenchmarkTask
from acme.tools.tool_registry import ToolRegistry


class GreenAgent:
    """Runs a single benchmark task against a participant agent."""

    def __init__(
        self,
        seed: int = 42,
        evaluator: Evaluator | None = None,
    ):
        """
        Args:
            seed: Random seed for state generation. Fixed per run for
                reproducibility; vary across k-repetition runs.
            evaluator: Custom evaluator instance. Defaults to stock
                Evaluator with default weights.
        """
        self.seed = seed
        self.evaluator = evaluator or Evaluator()

    def run_task(
        self,
        task: BenchmarkTask,
        agent: ParticipantAgent,
    ) -> TaskResult:
        """Run one task against one participant agent."""
        # 1. Initialize the environment
        state = generate_seed_data(seed=self.seed)
        state.apply_patch(task.initial_state_patch)

        # 2. Snapshot the "before" state for policy-delta computation
        before_state = state.snapshot()

        # 3. Build the registry with per-task access control
        allowed = task.available_tools if task.available_tools else None
        registry = ToolRegistry(state, allowed_tools=allowed)

        # 4. Run the agent
        final_message = agent.run(task, registry)

        # 5. Evaluate
        result = self.evaluator.evaluate(
            task=task,
            before_state=before_state,
            after_state=state,
            action_log=registry.log,
        )

        # 6. Attach debugging info
        result.notes = f"final_message={final_message[:200]!r}"
        return result


__all__ = ["GreenAgent"]