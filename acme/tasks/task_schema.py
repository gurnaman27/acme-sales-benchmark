"""Schema for benchmark tasks.

A `BenchmarkTask` describes:
  - what the agent should do (instruction)
  - what state to start from (initial_state_patch)
  - what tools they can use (available_tools)
  - what success looks like (expected_outcome)

The evaluator (Phase 3) reads `expected_outcome` and compares it against
the final state and action log.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TaskCategory(str, Enum):
    FOLLOW_UP = "follow_up"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    OPPORTUNITY_MGMT = "opportunity_mgmt"
    SCHEDULING = "scheduling"
    POLICY_CONFLICT = "policy_conflict"


class ExpectedOutcome(BaseModel):
    """What the final state should look like if the task is solved correctly.

    Attributes:
        required_state: Each key is a collection name (e.g., "opportunities").
            The value is a list of "matcher" dicts. Every matcher must be
            satisfied by at least one entity in the collection (all fields
            in the matcher must match that entity).

            Example::

                {"opportunities": [
                    {"customer_id": "C005", "next_action": "send_proposal"}
                ]}

            Means: at least one opportunity in the final state must have
            customer_id == "C005" AND next_action == "send_proposal".

        forbidden_state: Same structure as required_state, but inverted —
            if any entity matches, the task fails.

        required_tool_calls: Tool names that must appear in the action log.

        forbidden_tool_calls: Tool names that must NOT appear.

        max_policy_violations: Maximum number of policy violations allowed
            in the final state (default 0).

        max_tool_calls: Optional efficiency budget. If set and exceeded,
            the efficiency score is penalized (does not fail the task).
    """
    required_state: dict = Field(default_factory=dict)
    forbidden_state: dict = Field(default_factory=dict)
    required_tool_calls: list[str] = Field(default_factory=list)
    forbidden_tool_calls: list[str] = Field(default_factory=list)
    max_policy_violations: int = 0
    max_tool_calls: int | None = None


class BenchmarkTask(BaseModel):
    """A single benchmark task.

    Attributes:
        task_id: Unique identifier (e.g., "T001").
        category: One of the TaskCategory values.
        difficulty: 1 (basic) to 4 (complex).
        instruction: Natural language task for the agent.
        initial_state_patch: Task-specific modifications to the seed data.
        available_tools: Whitelist of tool names. Empty list = all tools.
        constraints: Human-readable constraints (for documentation).
        expected_outcome: Structured success criteria.
        edge_cases: Tags for edge cases this task exercises.
        notes: Author notes.
    """
    task_id: str
    category: TaskCategory
    difficulty: int = Field(ge=1, le=4)
    instruction: str
    initial_state_patch: dict = Field(default_factory=dict)
    available_tools: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    expected_outcome: ExpectedOutcome
    edge_cases: list[str] = Field(default_factory=list)
    notes: str = ""


__all__ = ["TaskCategory", "ExpectedOutcome", "BenchmarkTask"]