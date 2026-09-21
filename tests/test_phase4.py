"""Phase 4 — Comprehensive tests for multi-agent comparison infrastructure.

Tests:
  - RandomAgent behavior and bounds
  - MLPAgent variants (Standard, Deep, Residual, Ensemble)
  - LLMAgent response parsing (TOOL_CALL, FINAL, multi-line, error recovery)
  - LLM Provider protocol compliance
  - Feature encoder consistency
  - Imitation trainer data collection
  - Comparison experiment structure
  - Failure analysis module
  - CLI argument parsing
  - End-to-end agent run verification
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from acme.agents.agent_interface import ParticipantAgent
from acme.agents.baseline_simple import SimpleAgent
from acme.agents.feature_encoder import (
    ACTION_TEMPLATES,
    FEATURE_DIM,
    N_ACTIONS,
    FeatureEncoder,
)
from acme.agents.llm_agent import LLMAgent, _parse_response, _build_tool_list
from acme.agents.mlp_agent import MLPAgent
from acme.agents.mlp_models import (
    BaseMLP,
    DeepMLP,
    EnsembleMLP,
    ResidualMLP,
    StandardMLP,
    softmax,
    relu,
)
from acme.agents.random_agent import RandomAgent
from acme.environment.data_generator import generate_seed_data
from acme.evaluation.evaluator import Evaluator
from acme.evaluation.metrics import BenchmarkResult, TaskResult
from acme.green_agent.runner import Runner
from acme.tasks.task_library import TASK_LIBRARY, get_task_by_id
from acme.tools.tool_registry import ToolRegistry

import numpy as np


# ======================================================================
# RandomAgent
# ======================================================================


class TestRandomAgent:
    """Tests for the random baseline agent."""

    def test_is_participant(self):
        agent = RandomAgent(seed=42)
        assert isinstance(agent, ParticipantAgent)

    def test_name(self):
        agent = RandomAgent(seed=42)
        assert agent.name == "random_baseline"

    def test_runs_without_crash(self):
        """Random agent should complete all 30 tasks without raising."""
        agent = RandomAgent(seed=42)
        runner = Runner(seed=42)
        result = runner.run_all(agent)
        assert result.total_tasks == 30
        assert 0.0 <= result.mean_score <= 1.0

    def test_score_below_simple(self):
        """Random agent should score strictly below SimpleAgent."""
        runner = Runner(seed=42)
        random_result = runner.run_all(RandomAgent(seed=42))
        simple_result = runner.run_all(SimpleAgent())
        assert random_result.mean_score < simple_result.mean_score

    def test_deterministic_with_seed(self):
        """Same seed should give same results."""
        runner = Runner(seed=42)
        r1 = runner.run_all(RandomAgent(seed=42))
        r2 = runner.run_all(RandomAgent(seed=42))
        assert r1.mean_score == r2.mean_score


# ======================================================================
# MLP Models
# ======================================================================


class TestMLPModels:
    """Tests for the 4 MLP model variants."""

    def test_standard_mlp_output_shape(self):
        model = StandardMLP(64, 10)
        x = np.random.randn(64).astype(np.float32)
        probs = model.predict_proba(x)
        assert probs.shape == (10,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_deep_mlp_output_shape(self):
        model = DeepMLP(64, 10)
        x = np.random.randn(64).astype(np.float32)
        probs = model.predict_proba(x)
        assert probs.shape == (10,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_residual_mlp_output_shape(self):
        model = ResidualMLP(64, 10)
        x = np.random.randn(64).astype(np.float32)
        probs = model.predict_proba(x)
        assert probs.shape == (10,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_ensemble_mlp_output_shape(self):
        model = EnsembleMLP(64, 10)
        x = np.random.randn(64).astype(np.float32)
        probs = model.predict_proba(x)
        assert probs.shape == (10,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_standard_mlp_predict(self):
        model = StandardMLP(64, 10)
        x = np.random.randn(64).astype(np.float32)
        idx = model.predict(x)
        assert 0 <= idx < 10

    def test_standard_mlp_fit(self):
        model = StandardMLP(64, 10)
        X = np.random.randn(20, 64).astype(np.float32)
        Y = np.random.randint(0, 10, 20).astype(np.int64)
        model.fit(X, Y, epochs=2, batch_size=4)
        # Should not crash and probs should still sum to 1
        probs = model.predict_proba(X[0])
        assert abs(probs.sum() - 1.0) < 1e-4

    def test_residual_mlp_fit(self):
        model = ResidualMLP(64, 10)
        X = np.random.randn(20, 64).astype(np.float32)
        Y = np.random.randint(0, 10, 20).astype(np.int64)
        model.fit(X, Y, epochs=2, batch_size=4)
        probs = model.predict_proba(X[0])
        assert abs(probs.sum() - 1.0) < 1e-4

    def test_ensemble_mlp_fit(self):
        model = EnsembleMLP(64, 10)
        X = np.random.randn(20, 64).astype(np.float32)
        Y = np.random.randint(0, 10, 20).astype(np.int64)
        model.fit(X, Y, epochs=2)
        probs = model.predict_proba(X[0])
        assert abs(probs.sum() - 1.0) < 1e-4

    def test_save_load_standard(self, tmp_path):
        model = StandardMLP(64, 10)
        x = np.random.randn(64).astype(np.float32)
        pred_before = model.predict(x)
        path = str(tmp_path / "model.npz")
        model.save(path)

        model2 = StandardMLP(64, 10, seed=99)
        model2.load(path)
        pred_after = model2.predict(x)
        assert pred_before == pred_after

    def test_save_load_residual(self, tmp_path):
        model = ResidualMLP(64, 10)
        x = np.random.randn(64).astype(np.float32)
        pred_before = model.predict(x)
        path = str(tmp_path / "residual.npz")
        model.save(path)

        model2 = ResidualMLP(64, 10, seed=99)
        model2.load(path)
        pred_after = model2.predict(x)
        assert pred_before == pred_after

    def test_save_load_ensemble(self, tmp_path):
        model = EnsembleMLP(64, 10)
        x = np.random.randn(64).astype(np.float32)
        pred_before = model.predict(x)
        model.save(str(tmp_path / "ensemble.npz"))

        model2 = EnsembleMLP(64, 10)
        model2.load(str(tmp_path / "ensemble.npz"))
        pred_after = model2.predict(x)
        assert pred_before == pred_after


class TestSoftmaxRelu:
    """Tests for activation functions."""

    def test_softmax_1d(self):
        x = np.array([1.0, 2.0, 3.0])
        p = softmax(x)
        assert abs(p.sum() - 1.0) < 1e-6
        assert (p > 0).all()

    def test_softmax_large_values(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        p = softmax(x)
        assert abs(p.sum() - 1.0) < 1e-5
        assert np.isfinite(p).all()

    def test_relu(self):
        x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        r = relu(x)
        np.testing.assert_array_equal(r, [0, 0, 0, 1, 2])


# ======================================================================
# Feature Encoder
# ======================================================================


class TestFeatureEncoder:
    """Tests for the task state featurizer."""

    def test_output_dim(self):
        encoder = FeatureEncoder()
        task = TASK_LIBRARY[0]
        vec = encoder.encode_state(task)
        assert vec.shape == (FEATURE_DIM,)

    def test_task_id_one_hot(self):
        encoder = FeatureEncoder()
        for i, task in enumerate(TASK_LIBRARY):
            vec = encoder.encode_state(task)
            assert vec[i] == 1.0

    def test_different_tasks_different_vectors(self):
        encoder = FeatureEncoder()
        v1 = encoder.encode_state(TASK_LIBRARY[0])
        v2 = encoder.encode_state(TASK_LIBRARY[1])
        assert not np.allclose(v1, v2)

    def test_step_number_encoded(self):
        encoder = FeatureEncoder()
        task = TASK_LIBRARY[0]
        v0 = encoder.encode_state(task, step_number=0)
        v5 = encoder.encode_state(task, step_number=5)
        assert not np.allclose(v0, v5)

    def test_action_templates_count(self):
        assert N_ACTIONS == len(ACTION_TEMPLATES)
        assert N_ACTIONS > 30  # Should have many action templates


# ======================================================================
# MLPAgent
# ======================================================================


class TestMLPAgent:
    """Tests for the neural network participant agent."""

    def test_is_participant(self):
        agent = MLPAgent(variant="standard")
        assert isinstance(agent, ParticipantAgent)

    def test_name_variants(self):
        assert MLPAgent("standard").name == "mlp_standard"
        assert MLPAgent("deep").name == "mlp_deep"
        assert MLPAgent("residual").name == "mlp_residual"
        assert MLPAgent("ensemble").name == "mlp_ensemble"

    def test_runs_on_all_tasks(self):
        """MLP agent should complete all 30 tasks without crashing."""
        agent = MLPAgent(variant="standard")
        runner = Runner(seed=42)
        result = runner.run_all(agent)
        assert result.total_tasks == 30


# ======================================================================
# LLM Agent — Response Parsing
# ======================================================================


class TestLLMResponseParsing:
    """Tests for the ReAct-style response parser."""

    def test_parse_final(self):
        kind, payload = _parse_response("FINAL: Task completed successfully.")
        assert kind == "final"
        assert payload == "Task completed successfully."

    def test_parse_final_case_insensitive(self):
        kind, payload = _parse_response("final: Done.")
        assert kind == "final"
        assert payload == "Done."

    def test_parse_tool_call(self):
        kind, payload = _parse_response(
            'TOOL_CALL: {"name": "get_customer", "args": {"customer_id": "C001"}}'
        )
        assert kind == "tool"
        assert payload["name"] == "get_customer"
        assert payload["args"]["customer_id"] == "C001"

    def test_parse_tool_call_no_args(self):
        kind, payload = _parse_response(
            'TOOL_CALL: {"name": "get_current_time"}'
        )
        assert kind == "tool"
        assert payload["name"] == "get_current_time"
        assert payload["args"] == {}

    def test_parse_error_no_match(self):
        kind, payload = _parse_response("I think we should do something.")
        assert kind == "error"

    def test_parse_invalid_json(self):
        kind, payload = _parse_response("TOOL_CALL: {not json}")
        assert kind == "error"
        assert "Invalid JSON" in payload

    def test_parse_missing_name(self):
        kind, payload = _parse_response('TOOL_CALL: {"args": {}}')
        assert kind == "error"
        assert "missing 'name'" in payload

    def test_parse_multiline_with_tool_call(self):
        """LLMs often include reasoning before the tool call line."""
        text = (
            "Let me check the customer first.\n\n"
            'TOOL_CALL: {"name": "get_customer", "args": {"customer_id": "C001"}}'
        )
        kind, payload = _parse_response(text)
        assert kind == "tool"
        assert payload["name"] == "get_customer"

    def test_parse_multiline_with_final(self):
        text = (
            "After reviewing, I can provide my answer.\n\n"
            "FINAL: The customer has 3 interactions."
        )
        kind, payload = _parse_response(text)
        assert kind == "final"
        assert "3 interactions" in payload


class TestLLMAgentWithMockProvider:
    """Tests for the LLMAgent using a mock provider."""

    def _make_mock_provider(self, responses: list[str]):
        """Create a mock provider that returns predefined responses."""
        provider = MagicMock()
        provider.name = "mock"
        provider.generate = MagicMock(side_effect=responses)
        return provider

    def test_simple_final_response(self):
        provider = self._make_mock_provider(["FINAL: Done with task."])
        agent = LLMAgent(provider)
        task = TASK_LIBRARY[0]
        state = generate_seed_data(seed=42)
        registry = ToolRegistry(state)
        result = agent.run(task, registry)
        assert result == "Done with task."

    def test_tool_call_then_final(self):
        provider = self._make_mock_provider([
            'TOOL_CALL: {"name": "get_current_time", "args": {}}',
            "FINAL: The date is 2025-03-15.",
        ])
        agent = LLMAgent(provider)
        task = get_task_by_id("T005")
        state = generate_seed_data(seed=42)
        registry = ToolRegistry(state)
        result = agent.run(task, registry)
        assert "2025-03-15" in result
        assert len(registry.log) == 1  # One tool call was made

    def test_error_recovery(self):
        provider = self._make_mock_provider([
            "I don't understand the format...",  # triggers parse error
            "FINAL: OK done.",  # recovery
        ])
        agent = LLMAgent(provider)
        task = TASK_LIBRARY[0]
        state = generate_seed_data(seed=42)
        registry = ToolRegistry(state)
        result = agent.run(task, registry)
        assert result == "OK done."

    def test_max_steps_exceeded(self):
        # Always return a bad response
        provider = self._make_mock_provider(
            ["bad response"] * 50
        )
        agent = LLMAgent(provider, max_steps=3)
        task = TASK_LIBRARY[0]
        state = generate_seed_data(seed=42)
        registry = ToolRegistry(state)
        result = agent.run(task, registry)
        assert "max steps" in result

    def test_provider_exception_handled(self):
        provider = self._make_mock_provider([])
        provider.generate.side_effect = RuntimeError("API timeout")
        agent = LLMAgent(provider)
        task = TASK_LIBRARY[0]
        state = generate_seed_data(seed=42)
        registry = ToolRegistry(state)
        result = agent.run(task, registry)
        assert "provider error" in result

    def test_tool_list_includes_descriptions(self):
        state = generate_seed_data(seed=42)
        registry = ToolRegistry(state)
        tool_list = _build_tool_list(registry)
        assert "get_customer" in tool_list
        assert "schedule_meeting" in tool_list
        assert "customer_id" in tool_list  # parameter docs included


# ======================================================================
# LLM Providers
# ======================================================================


class TestProviders:
    """Tests for provider instantiation and protocol compliance."""

    def test_openai_provider_init(self):
        from acme.agents.providers.openai import OpenAIProvider
        p = OpenAIProvider(api_key="test-key")
        assert p.name.startswith("openai_")

    def test_deepseek_provider_init(self):
        from acme.agents.providers.deepseek import DeepSeekProvider
        p = DeepSeekProvider(api_key="test-key")
        assert p.name.startswith("deepseek_")

    def test_compatible_provider_init(self):
        from acme.agents.providers.openai_compatible import OpenAICompatibleProvider
        p = OpenAICompatibleProvider()
        assert p.name.startswith("compatible_")

    def test_openai_raises_without_key(self):
        from acme.agents.providers.openai import OpenAIProvider
        p = OpenAIProvider(api_key=None)
        p.api_key = None
        with pytest.raises(ValueError, match="API key missing"):
            p.generate([{"role": "user", "content": "test"}])


# ======================================================================
# Imitation Trainer
# ======================================================================


class TestImitationTrainer:
    """Tests for the imitation learning data collection."""

    def test_action_matching(self):
        from acme.agents.imitation_trainer import match_action_to_index
        idx = match_action_to_index("get_customer", {"customer_id": "C001"})
        assert 0 <= idx < N_ACTIONS

    def test_action_matching_fallback(self):
        from acme.agents.imitation_trainer import match_action_to_index
        # Unknown tool should fall back to FINAL action
        idx = match_action_to_index("nonexistent_tool", {})
        assert idx == N_ACTIONS - 1

    def test_expert_dataset_collection(self):
        from acme.agents.imitation_trainer import collect_expert_dataset
        encoder = FeatureEncoder()
        X, Y = collect_expert_dataset(encoder)
        assert X.shape[0] > 0
        assert X.shape[1] == FEATURE_DIM
        assert Y.shape[0] == X.shape[0]
        assert all(0 <= y < N_ACTIONS for y in Y)

    def test_train_standard_mlp(self, tmp_path):
        from acme.agents.imitation_trainer import train_mlp_agent
        path = train_mlp_agent("standard", epochs=2, models_dir=str(tmp_path))
        assert os.path.exists(path)


# ======================================================================
# Comparison Experiment
# ======================================================================


class TestComparisonExperiment:
    """Tests for the multi-agent comparison infrastructure."""

    def test_available_agents_includes_all_families(self):
        from acme.experiments.run_comparison import get_available_participant_agents
        agents = get_available_participant_agents()
        names = [a.name for a in agents]
        assert "simple_rule_based" in names
        assert "random_baseline" in names
        assert "mlp_standard" in names

    def test_run_comparison_outputs(self, tmp_path):
        from acme.experiments.run_comparison import run_comparison_experiment
        # Run with just SimpleAgent and RandomAgent for speed
        result = run_comparison_experiment(out_dir=str(tmp_path), seed=42)
        assert result["agents_evaluated"] >= 2
        assert os.path.exists(tmp_path / "results.json")
        assert os.path.exists(tmp_path / "results.csv")
        assert os.path.exists(tmp_path / "comparison_table.md")

    def test_comparison_json_structure(self, tmp_path):
        from acme.experiments.run_comparison import run_comparison_experiment
        run_comparison_experiment(out_dir=str(tmp_path), seed=42)

        with open(tmp_path / "results.json") as f:
            data = json.load(f)

        assert "benchmark_size" in data
        assert data["benchmark_size"] == 30
        assert "summary" in data
        assert "per_agent_details" in data

        for agent_summary in data["summary"]:
            assert "agent" in agent_summary
            assert "family" in agent_summary
            assert "by_difficulty" in agent_summary
            assert "by_category" in agent_summary


# ======================================================================
# Failure Analysis
# ======================================================================


class TestFailureAnalysis:
    """Tests for the failure analysis module."""

    def test_analyze_missing_file(self):
        from acme.experiments.failure_analysis import analyze_failures
        result = analyze_failures(results_json_path="/nonexistent/path.json")
        assert "error" in result

    def test_analyze_valid_results(self, tmp_path):
        # Generate results first
        from acme.green_agent.export import export_results_json
        runner = Runner(seed=42)
        result = runner.run_all(RandomAgent(seed=42))
        results_path = tmp_path / "results.json"
        export_results_json(result, results_path)

        from acme.experiments.failure_analysis import analyze_failures
        analysis = analyze_failures(
            results_json_path=str(results_path),
            out_dir=str(tmp_path),
        )
        assert "total_tasks" in analysis
        assert "failed_count" in analysis
        assert os.path.exists(tmp_path / "failure_analysis.json")
        assert os.path.exists(tmp_path / "failure_analysis.md")


# ======================================================================
# CLI
# ======================================================================


class TestCLI:
    """Tests for the command-line interface argument parsing."""

    def test_build_agent_simple(self):
        from acme.green_agent.cli import _build_agent
        agent = _build_agent("simple")
        assert isinstance(agent, SimpleAgent)

    def test_build_agent_random(self):
        from acme.green_agent.cli import _build_agent
        agent = _build_agent("random")
        assert isinstance(agent, RandomAgent)

    def test_build_agent_mlp(self):
        from acme.green_agent.cli import _build_agent
        for variant in ["mlp_standard", "mlp_deep", "mlp_residual", "mlp_ensemble"]:
            agent = _build_agent(variant)
            assert isinstance(agent, MLPAgent)

    def test_build_agent_unknown(self):
        from acme.green_agent.cli import _build_agent
        with pytest.raises(ValueError, match="Unknown agent"):
            _build_agent("nonexistent")

    def test_select_tasks_all(self):
        from acme.green_agent.cli import _select_tasks
        tasks = _select_tasks("all")
        assert len(tasks) == 30

    def test_select_tasks_specific(self):
        from acme.green_agent.cli import _select_tasks
        tasks = _select_tasks("T001,T002")
        assert len(tasks) == 2
        assert {t.task_id for t in tasks} == {"T001", "T002"}

    def test_select_tasks_unknown(self):
        from acme.green_agent.cli import _select_tasks
        with pytest.raises(ValueError, match="Unknown task IDs"):
            _select_tasks("T999")


# ======================================================================
# End-to-End Integration
# ======================================================================


class TestEndToEnd:
    """End-to-end integration tests for Phase 4."""

    def test_simple_agent_100_percent(self):
        """SimpleAgent should score 100% on all 30 tasks."""
        runner = Runner(seed=42)
        result = runner.run_all(SimpleAgent())
        assert result.total_tasks == 30
        assert result.success_rate == 1.0

    def test_all_difficulty_levels_covered(self):
        """Verify all 4 difficulty levels are represented."""
        runner = Runner(seed=42)
        result = runner.run_all(SimpleAgent())
        difficulties = result.by_difficulty()
        assert set(difficulties.keys()) == {1, 2, 3, 4}
        assert difficulties[1]["count"] == 8  # L1
        assert difficulties[2]["count"] == 7  # L2
        assert difficulties[3]["count"] == 8  # L3
        assert difficulties[4]["count"] == 7  # L4

    def test_all_categories_covered(self):
        """Verify all 5 task categories are represented."""
        runner = Runner(seed=42)
        result = runner.run_all(SimpleAgent())
        categories = result.by_category()
        assert len(categories) == 5

    def test_validation_suite_all_calibrated(self):
        """Validation suite should pass for all 6 synthetic agents."""
        from acme.validation.validation_suite import run_evaluator_validation_suite
        report = run_evaluator_validation_suite(seed=42)
        assert len(report) == 6
        for agent_name, metrics in report.items():
            assert metrics["calibrated"], f"{agent_name} not calibrated: score={metrics['mean_score']}"

    def test_random_agent_well_below_simple(self):
        """Random agent should be well below 50% success rate."""
        runner = Runner(seed=42)
        result = runner.run_all(RandomAgent(seed=42))
        assert result.success_rate < 0.5

    def test_benchmark_result_serializable(self):
        """BenchmarkResult should be JSON-serializable."""
        runner = Runner(seed=42)
        result = runner.run_all(SimpleAgent())
        # Should not raise
        data = [r.model_dump(mode="json") for r in result.results]
        json.dumps(data)


class TestA2AServer:
    """Tests for Phase 5 A2A FastAPI server endpoint."""

    def test_agent_card_endpoint(self):
        from fastapi.testclient import TestClient
        from acme.green_agent.a2a_server import app

        client = TestClient(app)
        res = client.get("/.well-known/agent-card.json")
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == "acme-sales-green-agent"
        assert "run" in data["endpoints"]

    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from acme.green_agent.a2a_server import app

        client = TestClient(app)
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

    def test_run_endpoint_simple_agent(self):
        from fastapi.testclient import TestClient
        from acme.green_agent.a2a_server import app

        client = TestClient(app)
        res = client.post(
            "/run",
            json={"agent_name": "simple", "task_ids": ["T001", "T002"], "seed": 42},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["agent_name"] == "simple_rule_based"
        assert data["total_tasks"] == 2
        assert data["success_rate"] == 1.0

