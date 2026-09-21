"""MLP-based participant agent using neural network inference.

Executes tasks using trained NumPy MLP models (Standard, Deep, Residual, Ensemble)
trained via imitation learning.
"""

from __future__ import annotations

import os
from typing import Any

from acme.agents.agent_interface import ParticipantAgent
from acme.agents.feature_encoder import (
    ACTION_TEMPLATES,
    FEATURE_DIM,
    N_ACTIONS,
    FeatureEncoder,
)
from acme.agents.mlp_models import (
    DeepMLP,
    EnsembleMLP,
    ResidualMLP,
    StandardMLP,
)
from acme.tasks.task_schema import BenchmarkTask
from acme.tools.tool_registry import ToolRegistry


class MLPAgent(ParticipantAgent):
    """Neural network agent powered by pure-NumPy MLP models."""

    def __init__(self, variant: str = "standard", model_path: str | None = None):
        self.variant = variant
        self._name = f"mlp_{variant}"
        self.encoder = FeatureEncoder()

        if variant == "deep":
            self.model: Any = DeepMLP(FEATURE_DIM, N_ACTIONS)
        elif variant == "residual":
            self.model = ResidualMLP(FEATURE_DIM, N_ACTIONS)
        elif variant == "ensemble":
            self.model = EnsembleMLP(FEATURE_DIM, N_ACTIONS)
        else:
            self.model = StandardMLP(FEATURE_DIM, N_ACTIONS)

        if model_path is None:
            model_path = os.path.join("models", f"mlp_{variant}.npz")

        if os.path.exists(model_path):
            self.model.load(model_path)
        else:
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. "
                f"Run `python -m acme.agents.imitation_trainer` first."
            )

    @property
    def name(self) -> str:
        return self._name

    def run(self, task: BenchmarkTask, registry: ToolRegistry) -> str:
        tool_history = []
        max_steps = min(task.expected_outcome.max_tool_calls or 10, 10)

        for step in range(max_steps):
            x_vec = self.encoder.encode_state(task, step_number=step, tool_history=tool_history)
            action_idx = self.model.predict(x_vec)

            if action_idx < 0 or action_idx >= len(ACTION_TEMPLATES):
                action_idx = len(ACTION_TEMPLATES) - 1

            template = ACTION_TEMPLATES[action_idx]

            if template.get("type") == "final":
                return template.get("message", "Task complete.")

            tool_name = template.get("name")
            tool_args = template.get("args", {})

            if not tool_name:
                return "Task complete."

            # Execute tool
            res = registry.call(tool_name, **tool_args)
            tool_history.append(tool_name)

            if not res.success:
                # If tool failed, try simple fallback or loop
                continue

        return f"Completed task {task.task_id} after {len(tool_history)} steps."
