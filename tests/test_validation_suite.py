"""Tests for the Evaluator Validation Suite."""

from acme.validation.validation_suite import run_evaluator_validation_suite


def test_evaluator_calibration_on_synthetic_agents():
    """Verify that all 6 synthetic agents produce scores within expected calibrated bands."""
    report = run_evaluator_validation_suite(seed=42)
    for agent_name, status in report.items():
        assert status["calibrated"], (
            f"Evaluator calibration failure for '{agent_name}': "
            f"score {status['mean_score']} outside expected band {status['expected_band']}"
        )
