"""Failure classification for benchmark tasks.

When a task fails, we want more than "it failed" — we want to know
*why*. This module defines a taxonomy and a classifier that reads the
action log + sub-scores to assign the most explanatory failure type.
"""

from __future__ import annotations

from enum import Enum


class FailureType(str, Enum):
    """Categories of agent failure."""
    RETRIEVAL = "retrieval"
    PLANNING = "planning"
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    STATE_TRACKING = "state_tracking"
    POLICY_VIOLATION = "policy_violation"
    SCHEDULING = "scheduling"
    EXECUTION = "execution"
    EFFICIENCY = "efficiency"
    HALLUCINATION = "hallucination"


def classify_failure(
    *,
    action_log,
    state_score: float,
    tools_score: float,
    policy_score: float,
    forbidden_score: float,
    state_details: dict,
    tools_details: dict,
    policy_details: dict,
    forbidden_details: dict,
) -> tuple[FailureType, dict]:
    """Return the dominant failure type and supporting details.

    Priority order (first match wins):
        1. POLICY_VIOLATION — introduced violations
        2. TOOL_USE — called a forbidden tool
        3. RETRIEVAL — any NOT_FOUND error during lookup
        4. PLANNING — missing required tools
        5. EXECUTION — state doesn't match despite tools being used
        6. EFFICIENCY — excessive extra calls
    """
    # 1. Policy violation (most severe — broke a rule)
    if policy_details["introduced"] > policy_details["budget"]:
        return FailureType.POLICY_VIOLATION, {
            "violations_introduced": policy_details["introduced"],
            "budget": policy_details["budget"],
        }

    # 2. Forbidden actions
    if forbidden_details["forbidden_tools_called"]:
        return FailureType.TOOL_USE, {
            "forbidden_tools_called": forbidden_details["forbidden_tools_called"],
        }
    if forbidden_details["forbidden_state_matches"]:
        return FailureType.TOOL_USE, {
            "forbidden_state_matches": forbidden_details["forbidden_state_matches"],
        }

    # 3. Retrieval — any lookup that returned NOT_FOUND
    not_found_calls = [
        e for e in action_log if e.result_error_code == "NOT_FOUND"
    ]
    if not_found_calls:
        return FailureType.RETRIEVAL, {
            "not_found_calls": [
                {"tool": e.tool, "args": e.args} for e in not_found_calls
            ],
        }

    # 4. Planning — didn't call all required tools
    if tools_details["missing"]:
        return FailureType.PLANNING, {
            "missing_tools": tools_details["missing"],
        }

    # 5. Execution — state doesn't match despite tools being used
    unsatisfied = [d for d in state_details["details"] if not d["matched"]]
    if unsatisfied:
        return FailureType.EXECUTION, {
            "unsatisfied_matchers": unsatisfied,
        }

    # 6. Efficiency — excessive extra calls
    return FailureType.EFFICIENCY, {}


__all__ = ["FailureType", "classify_failure"]