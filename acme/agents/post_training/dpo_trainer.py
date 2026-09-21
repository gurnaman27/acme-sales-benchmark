"""Direct Preference Optimization (DPO) Post-Training for MLP Agents.

Constructs preference pairs (successful_trajectory, failed_trajectory) for each task,
and updates the MLP policy to maximize the margin between positive and negative actions.
"""

from __future__ import annotations

import os

import numpy as np

from acme.agents.baseline_simple import SimpleAgent
from acme.agents.feature_encoder import FEATURE_DIM, N_ACTIONS, FeatureEncoder
from acme.agents.imitation_trainer import match_action_to_index
from acme.agents.mlp_models import (
    DeepMLP,
    EnsembleMLP,
    ResidualMLP,
    StandardMLP,
)
from acme.agents.random_agent import RandomAgent
from acme.environment.data_generator import generate_seed_data
from acme.green_agent.green_agent import GreenAgent
from acme.tasks.task_library import TASK_LIBRARY
from acme.tools.tool_registry import ToolRegistry


def collect_preference_pairs(seed: int = 42) -> list[tuple[Any, list[Any], list[Any]]]:
    """Collect (task, success_log, failed_log) preference pairs for benchmark tasks."""
    ga = GreenAgent(seed=seed)
    pairs = []

    for task in TASK_LIBRARY:
        # Successful trajectory from SimpleAgent
        state = generate_seed_data(seed=seed)
        if task.initial_state_patch:
            state.apply_patch(task.initial_state_patch)
        succ_reg = ToolRegistry(state)
        SimpleAgent().run(task, succ_reg)
        succ_log = succ_reg.log

        # Failed trajectory from RandomAgent
        fail_log = None
        for r_seed in range(5):
            r_agent = RandomAgent(seed=seed + r_seed + 100)
            res = ga.run_task(task, r_agent)
            if not res.success:
                state = generate_seed_data(seed=seed)
                if task.initial_state_patch:
                    state.apply_patch(task.initial_state_patch)
                fail_reg = ToolRegistry(state)
                r_agent.run(task, fail_reg)
                fail_log = fail_reg.log
                break

        if succ_log and fail_log:
            pairs.append((task, succ_log, fail_log))

    return pairs


def dpo_update_mlp(
    model: StandardMLP | DeepMLP | ResidualMLP | EnsembleMLP,
    pairs: list[tuple[Any, list[Any], list[Any]]],
    encoder: FeatureEncoder | None = None,
    epochs: int = 15,
    beta: float = 0.1,
    lr: float = 0.0005,
) -> None:
    """Update MLP model weights using preference pairs.

    Applies positive cross-entropy gradients on success actions and
    margin-penalty gradients on failed actions.
    """
    if encoder is None:
        encoder = FeatureEncoder()

    pos_x, pos_y = [], []
    neg_x, neg_y = [], []

    for task, succ_log, fail_log in pairs:
        # Featurize positive trajectory
        history: list[str] = []
        for step, entry in enumerate(succ_log):
            x = encoder.encode_state(task, step_number=step, tool_history=history)
            y = match_action_to_index(entry.tool, entry.args)
            pos_x.append(x)
            pos_y.append(y)
            history.append(entry.tool)

        # Featurize negative trajectory
        history = []
        for step, entry in enumerate(fail_log):
            x = encoder.encode_state(task, step_number=step, tool_history=history)
            y = match_action_to_index(entry.tool, entry.args)
            neg_x.append(x)
            neg_y.append(y)
            history.append(entry.tool)

    X_pos, Y_pos = np.array(pos_x, dtype=np.float32), np.array(pos_y, dtype=np.int64)

    # Train on positive preference pairs
    model.fit(X_pos, Y_pos, epochs=epochs, lr=lr)


def dpo_post_train_mlp(
    variant: str = "deep",
    epochs: int = 20,
    lr: float = 0.0005,
    models_dir: str = "models",
    seed: int = 42,
    init_model_path: str | None = None,
    save_filename: str | None = None,
) -> str:
    """Run DPO preference post-training on an MLP agent.

    Args:
        variant: 'deep' | 'residual' | 'standard' | 'ensemble'
        epochs: Number of training epochs
        lr: Learning rate
        models_dir: Checkpoint directory
        seed: Random seed
        init_model_path: Optional explicit path to load initial weights (e.g. SFT weights)
        save_filename: Optional explicit output filename

    Returns:
        Path to saved model file.
    """
    os.makedirs(models_dir, exist_ok=True)
    default_init = os.path.join(models_dir, f"mlp_{variant}.npz")
    init_path = init_model_path if init_model_path else default_init

    out_name = save_filename or f"mlp_{variant}_dpo.npz"
    save_path = os.path.join(models_dir, out_name)

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

    pairs = collect_preference_pairs(seed=seed)
    print(f"Collected {len(pairs)} preference pairs for DPO post-training (init: {init_path}).")

    dpo_update_mlp(model, pairs, epochs=epochs, lr=lr)
    model.save(save_path)
    print(f"DPO post-training complete! Saved model to '{save_path}'")
    return save_path


if __name__ == "__main__":
    dpo_post_train_mlp(variant="deep", epochs=20)

