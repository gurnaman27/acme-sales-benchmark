"""Collect successful trajectories across multiple source agents for post-training / SFT.

Collects trajectory data from:
  1. SimpleAgent (30/30 tasks)
  2. High-performing MLP checkpoints
  3. LLM agents (when API keys are present)

Outputs:
  - (X, Y) state-action pairs for MLP SFT
  - JSONL format for OpenAI Fine-Tuning API
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from acme.agents.baseline_simple import SimpleAgent
from acme.agents.feature_encoder import N_ACTIONS, FeatureEncoder
from acme.agents.imitation_trainer import match_action_to_index
from acme.agents.llm_agent import _SYSTEM_PROMPT, _build_tool_list
from acme.environment.data_generator import generate_seed_data
from acme.green_agent.green_agent import GreenAgent
from acme.tasks.task_library import TASK_LIBRARY
from acme.tools.tool_registry import ToolRegistry


def collect_success_trajectories(seed: int = 42) -> list[tuple[Any, list[Any], str]]:
    """Return list of (task, action_log, source_name) for successful runs."""
    sources: list[tuple[str, Any]] = [
        ("simple_rule_based", SimpleAgent()),
    ]

    # Try loading OpenAI agent if key present
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from acme.agents.llm_agent import LLMAgent
            from acme.agents.providers.openai import OpenAIProvider
            sources.append(("llm_openai", LLMAgent(OpenAIProvider())))
        except Exception:
            pass

    # Try loading DeepSeek agent if key present
    if os.environ.get("DEEPSEEK_API_KEY"):
        try:
            from acme.agents.llm_agent import LLMAgent
            from acme.agents.providers.deepseek import DeepSeekProvider
            sources.append(("llm_deepseek", LLMAgent(DeepSeekProvider())))
        except Exception:
            pass

    ga = GreenAgent(seed=seed)
    trajectories = []

    for name, agent in sources:
        for task in TASK_LIBRARY:
            res = ga.run_task(task, agent)
            if res.success:
                state = generate_seed_data(seed=seed)
                if task.initial_state_patch:
                    state.apply_patch(task.initial_state_patch)
                registry = ToolRegistry(state)
                agent.run(task, registry)
                trajectories.append((task, registry.log, name))

    return trajectories


def collect_augmented_dataset(
    encoder: FeatureEncoder | None = None, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Featurize success trajectories into (X, Y) dataset for MLP fine-tuning."""
    if encoder is None:
        encoder = FeatureEncoder()

    trajectories = collect_success_trajectories(seed=seed)

    x_list = []
    y_list = []

    for task, log, source in trajectories:
        history: list[str] = []
        for step, entry in enumerate(log):
            x_vec = encoder.encode_state(task, step_number=step, tool_history=history)
            y_idx = match_action_to_index(entry.tool, entry.args)
            x_list.append(x_vec)
            y_list.append(y_idx)
            history.append(entry.tool)

        # Final terminal action step
        x_vec = encoder.encode_state(task, step_number=len(log), tool_history=history)
        x_list.append(x_vec)
        y_list.append(N_ACTIONS - 1)

    X = np.array(x_list, dtype=np.float32)
    Y = np.array(y_list, dtype=np.int64)
    return X, Y


def export_openai_finetune_jsonl(out_path: str = "data/openai_finetune.jsonl", seed: int = 42) -> str:
    """Export success trajectories in OpenAI Fine-Tuning JSONL format."""
    trajectories = collect_success_trajectories(seed=seed)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for task, log, source in trajectories:
            state = generate_seed_data(seed=seed)
            if task.initial_state_patch:
                state.apply_patch(task.initial_state_patch)
            registry = ToolRegistry(state)

            messages = [
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT.format(tool_list=_build_tool_list(registry)),
                },
                {"role": "user", "content": f"TASK: {task.instruction}"},
            ]

            for entry in log:
                call = {"name": entry.tool, "args": entry.args}
                messages.append({
                    "role": "assistant",
                    "content": f"TOOL_CALL: {json.dumps(call)}",
                })
                messages.append({
                    "role": "user",
                    "content": "Tool result: OK",
                })

            messages.append({
                "role": "assistant",
                "content": "FINAL: Task completed successfully.",
            })

            f.write(json.dumps({"messages": messages}) + "\n")
            count += 1

    print(f"Exported {count} training trajectories to '{out_path}'")
    return out_path


__all__ = [
    "collect_success_trajectories",
    "collect_augmented_dataset",
    "export_openai_finetune_jsonl",
]
