"""Supervised Fine-Tuning (SFT) Trainer for MLP Agents.

Loads an imitation-trained MLP checkpoint, collects multi-source success trajectories,
and continues training (fine-tuning) with a lower learning rate.
"""

from __future__ import annotations

import os

from acme.agents.feature_encoder import FEATURE_DIM, N_ACTIONS, FeatureEncoder
from acme.agents.mlp_models import (
    DeepMLP,
    EnsembleMLP,
    ResidualMLP,
    StandardMLP,
)
from acme.agents.post_training.collect_successes import collect_augmented_dataset


def sft_mlp_agent(
    variant: str = "deep",
    epochs: int = 30,
    lr: float = 0.0005,
    models_dir: str = "models",
) -> str:
    """Perform SFT post-training on an existing MLP variant.

    Args:
        variant: 'standard' | 'deep' | 'residual' | 'ensemble'
        epochs: Number of fine-tuning epochs
        lr: Fine-tuning learning rate (lower than initial training)
        models_dir: Directory to read/save model files

    Returns:
        Path to the fine-tuned model file.
    """
    os.makedirs(models_dir, exist_ok=True)
    encoder = FeatureEncoder()
    X, Y = collect_augmented_dataset(encoder)

    init_path = os.path.join(models_dir, f"mlp_{variant}.npz")
    save_path = os.path.join(models_dir, f"mlp_{variant}_sft.npz")

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

    # Continue training (fine-tuning)
    model.fit(X, Y, epochs=epochs, lr=lr)
    model.save(save_path)
    print(f"SFT complete! Saved post-trained model to '{save_path}'")
    return save_path


if __name__ == "__main__":
    for v in ["deep", "residual", "standard", "ensemble"]:
        sft_mlp_agent(variant=v, epochs=30, lr=0.0005)
