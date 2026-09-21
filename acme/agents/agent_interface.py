"""Abstract interface for participant agents.

A participant agent is the thing being evaluated. It receives a task
instruction and a ToolRegistry, and it does whatever it wants with
them. The Green Agent doesn't care about the internal architecture —
only that the agent eventually returns a final message.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from acme.tasks.task_schema import BenchmarkTask
from acme.tools.tool_registry import ToolRegistry


class ParticipantAgent(ABC):
    """Base class for all participant agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """A human-readable identifier (used in reports)."""

    @abstractmethod
    def run(self, task: BenchmarkTask, registry: ToolRegistry) -> str:
        """Execute the task using the provided tool registry.

        Args:
            task: The benchmark task to solve.
            registry: Tool registry bound to the current environment state.

        Returns:
            The agent's final message (may be empty, may be a claim —
            the evaluator ignores it).
        """