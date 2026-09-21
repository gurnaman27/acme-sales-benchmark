"""Tests for Post-Training modules (SFT, STaR, DPO, dataset collection)."""

import os
from unittest.mock import patch

import pytest

from acme.agents.feature_encoder import FeatureEncoder
from acme.agents.mlp_agent import MLPAgent
from acme.agents.mlp_models import DeepMLP, ResidualMLP, StandardMLP
from acme.agents.post_training.collect_successes import (
    collect_augmented_dataset,
    collect_success_trajectories,
    export_openai_finetune_jsonl,
)
from acme.agents.post_training.dpo_trainer import (
    collect_preference_pairs,
    dpo_post_train_mlp,
)
from acme.agents.post_training.iterative_sft import iterative_sft_mlp
from acme.agents.post_training.sft_trainer import sft_mlp_agent


from acme.agents.feature_encoder import FEATURE_DIM, N_ACTIONS, FeatureEncoder


def test_collect_success_trajectories():
    trajectories = collect_success_trajectories(seed=42)
    assert len(trajectories) >= 30
    task, log, name = trajectories[0]
    assert task.task_id is not None
    assert isinstance(log, list)
    assert name == "simple_rule_based"


def test_collect_augmented_dataset():
    encoder = FeatureEncoder()
    X, Y = collect_augmented_dataset(encoder, seed=42)
    assert X.shape[0] > 0
    assert X.shape[1] == FEATURE_DIM
    assert Y.shape[0] == X.shape[0]


def test_export_openai_finetune_jsonl(tmp_path):
    out_file = str(tmp_path / "openai_finetune.jsonl")
    result_path = export_openai_finetune_jsonl(out_path=out_file, seed=42)
    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 0


def test_sft_mlp_agent(tmp_path):
    models_dir = str(tmp_path / "models")
    # First create baseline standard model
    std_path = str(tmp_path / "models" / "mlp_standard.npz")
    os.makedirs(models_dir, exist_ok=True)
    m = StandardMLP(FEATURE_DIM, N_ACTIONS)
    m.save(std_path)

    res_path = sft_mlp_agent(variant="standard", epochs=2, lr=0.001, models_dir=models_dir)
    assert os.path.exists(res_path)



def test_collect_preference_pairs():
    pairs = collect_preference_pairs(seed=42)
    assert len(pairs) > 0
    task, succ, fail = pairs[0]
    assert task is not None
    assert len(succ) > 0


def test_dpo_post_train_mlp(tmp_path):
    models_dir = str(tmp_path / "models")
    res_path = dpo_post_train_mlp(variant="standard", epochs=2, lr=0.001, models_dir=models_dir)
    assert os.path.exists(res_path)


def test_iterative_sft_mlp(tmp_path):
    models_dir = str(tmp_path / "models")
    res_path = iterative_sft_mlp(
        variant="standard", iterations=1, epochs_per_iter=2, models_dir=models_dir, seed=42
    )
    assert os.path.exists(res_path)
