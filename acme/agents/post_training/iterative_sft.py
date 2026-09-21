"""Iterative Rejection Sampling / Self-Taught Reasoner (STaR) Post-Training Loop.

Performs self-improvement by:
  1. Generating rollouts with the current MLP policy
  2. Filtering for successful rollouts (rejection sampling via Evaluator)
  3. Fine-tuning the MLP policy on its own successful trajectories
  4. Iterating over multiple rounds (default: 3 iterations)
"""

from __future__ import annotations

import os

import numpy as np

from acme.agents.feature_encoder import FEATURE_DIM, N_ACTIONS, FeatureEncoder
from acme.agents.imitation_trainer import match_action_to_index
from acme.agents.mlp_agent import MLPAgent
from acme.agents.mlp_models import (
    DeepMLP,
    EnsembleMLP,
    ResidualMLP,
    StandardMLP,
)
from acme.environment.data_generator import generate_seed_data
from acme.green_agent.green_agent import GreenAgent
from acme.tasks.task_library import TASK_LIBRARY
from acme.tools.tool_registry import ToolRegistry


def iterative_sft_mlp(
    variant: str = "deep",
    iterations: int = 3,
    epochs_per_iter: int = 15,
    lr: float = 0.0005,
    models_dir: str = "models",
    seed: int = 42,
) -> str:
    """Run iterative rejection sampling (STaR) self-training on an MLP agent.

    Args:
        variant: 'deep' | 'residual' | 'standard' | 'ensemble'
        iterations: Number of self-improvement rounds
        epochs_per_iter: Training epochs per iteration
        lr: Fine-tuning learning rate
        models_dir: Model checkpoint directory
        seed: Random seed

    Returns:
        Path to the final STaR-trained model file.
    """
    os.makedirs(models_dir, exist_ok=True)
    encoder = FeatureEncoder()
    ga = GreenAgent(seed=seed)

    init_path = os.path.join(models_dir, f"mlp_{variant}.npz")
    final_path = os.path.join(models_dir, f"mlp_{variant}_star.npz")

    if variant == "deep":
        model = DeepMLP(FEATURE_DIM, N_ACTIONS)
    elif variant == "residual":
        model = ResidualMLP(FEATURE_DIM, N_ACTIONS)
    elif variant == "ensemble":
        model = EnsembleMLP(FEATURE_DIM, N_ACTIONS)
    else:
        model = StandardMLP(FEATURE_DIM, N_ACTIONS)

    if os.path.exists(init_path):
        model.load(init_path)

    for iter_idx in range(1, iterations + 1):
        print(f"--- STaR Iteration {iter_idx}/{iterations} for MLP ({variant}) ---")

        # Temporarily save current weights so MLPAgent can load them
        temp_path = os.path.join(models_dir, f"mlp_{variant}_temp.npz")
        model.save(temp_path)

        # Build MLPAgent wrapper pointing to temp weights
        agent = MLPAgent(variant=variant, model_path=temp_path)

        # Collect successful rollouts
        successful_trajectories = []
        for task in TASK_LIBRARY:
            res = ga.run_task(task, agent)
            if res.success:
                state = generate_seed_data(seed=seed)
                if task.initial_state_patch:
                    state.apply_patch(task.initial_state_patch)
                registry = ToolRegistry(state)
                agent.run(task, registry)
                successful_trajectories.append((task, registry.log))

        print(f"  Iteration {iter_idx}: {len(successful_trajectories)}/{len(TASK_LIBRARY)} tasks solved.")

        if not successful_trajectories:
            print("  No successful rollouts found in this iteration. Skipping update.")
            continue

        # Featurize kept rollouts
        x_list = []
        y_list = []
        for task, log in successful_trajectories:
            history: list[str] = []
            for step, entry in enumerate(log):
                x_vec = encoder.encode_state(task, step_number=step, tool_history=history)
                y_idx = match_action_to_index(entry.tool, entry.args)
                x_list.append(x_vec)
                y_list.append(y_idx)
                history.append(entry.tool)

            x_vec = encoder.encode_state(task, step_number=len(log), tool_history=history)
            x_list.append(x_vec)
            y_list.append(N_ACTIONS - 1)

        X = np.array(x_list, dtype=np.float32)
        Y = np.array(y_list, dtype=np.int64)

        # Fine-tune model
        model.fit(X, Y, epochs=epochs_per_iter, lr=lr)

        iter_path = os.path.join(models_dir, f"mlp_{variant}_star_iter{iter_idx}.npz")
        model.save(iter_path)

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    model.save(final_path)
    print(f"STaR iterative post-training complete! Saved to '{final_path}'")
    return final_path


if __name__ == "__main__":
    iterative_sft_mlp(variant="deep", iterations=3, epochs_per_iter=15)
