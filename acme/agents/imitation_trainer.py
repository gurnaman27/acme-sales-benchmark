"""Imitation Learning Trainer for MLP Agents.

Collects expert trajectories from SimpleAgent running on all 30 tasks,
featurizes them into (X, Y) training pairs, trains the MLP model, and
saves the model weights.
"""

from __future__ import annotations

import os
import numpy as np

from acme.agents.baseline_simple import SimpleAgent
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
from acme.environment.data_generator import generate_seed_data
from acme.tasks.task_library import TASK_LIBRARY
from acme.tools.tool_registry import ToolRegistry


def match_action_to_index(tool_name: str, args: dict) -> int:
    """Find closest action template index for a logged tool call."""
    for idx, t in enumerate(ACTION_TEMPLATES):
        if t.get("type") == "tool" and t.get("name") == tool_name:
            # Match tool name
            t_args = t.get("args", {})
            # Check key subset match
            if all(k in args and args[k] == v for k, v in t_args.items()):
                return idx
    # Fallback to first matching tool name
    for idx, t in enumerate(ACTION_TEMPLATES):
        if t.get("type") == "tool" and t.get("name") == tool_name:
            return idx
    # Final fallback: last action index (FINAL)
    return N_ACTIONS - 1


def collect_expert_dataset(encoder: FeatureEncoder) -> tuple[np.ndarray, np.ndarray]:
    """Run SimpleAgent on all 30 tasks and extract (state_features, action_index) pairs."""
    agent = SimpleAgent()
    x_list = []
    y_list = []

    for task in TASK_LIBRARY:
        state = generate_seed_data(seed=42)
        if task.initial_state_patch:
            state.apply_patch(task.initial_state_patch)
        registry = ToolRegistry(state)

        # Intercept tool calls step by step
        tool_history = []
        step_num = 0

        # Run handler
        agent.run(task, registry)

        # Convert logged calls to dataset pairs
        for entry in registry.log:
            x_vec = encoder.encode_state(task, step_number=step_num, tool_history=tool_history)
            y_idx = match_action_to_index(entry.tool, entry.args)
            x_list.append(x_vec)
            y_list.append(y_idx)

            tool_history.append(entry.tool)
            step_num += 1

        # Add final action step
        x_vec = encoder.encode_state(task, step_number=step_num, tool_history=tool_history)
        y_list.append(N_ACTIONS - 1)  # FINAL
        x_list.append(x_vec)

    X = np.array(x_list, dtype=np.float32)
    Y = np.array(y_list, dtype=np.int64)
    return X, Y


def train_mlp_agent(variant: str = "standard", epochs: int = 50, models_dir: str = "models") -> str:
    """Train an MLP variant using imitation learning and save weights.

    Args:
        variant: 'standard' | 'deep' | 'residual' | 'ensemble'
        epochs: Number of training epochs
        models_dir: Directory to save model files

    Returns:
        Path to the saved model file.
    """
    os.makedirs(models_dir, exist_ok=True)
    encoder = FeatureEncoder()
    X, Y = collect_expert_dataset(encoder)

    save_path = os.path.join(models_dir, f"mlp_{variant}.npz")

    if variant == "deep":
        model = DeepMLP(FEATURE_DIM, N_ACTIONS)
    elif variant == "residual":
        model = ResidualMLP(FEATURE_DIM, N_ACTIONS)
    elif variant == "ensemble":
        model = EnsembleMLP(FEATURE_DIM, N_ACTIONS)
    else:
        model = StandardMLP(FEATURE_DIM, N_ACTIONS)

    model.fit(X, Y, epochs=epochs)
    model.save(save_path)
    return save_path


if __name__ == "__main__":
    for v in ["standard", "deep", "residual", "ensemble"]:
        path = train_mlp_agent(variant=v, epochs=30)
        print(f"Trained MLP agent variant '{v}' -> {path}")
