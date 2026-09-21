"""Policy query tools for the Acme Sales benchmark.

Agents can use these to ask "would this action violate policy?"
before taking it. This is the agent's introspection layer.
"""

from __future__ import annotations

from datetime import datetime

from acme.environment.policies import PolicyEngine
from acme.environment.state import EnvironmentState
from acme.tools.tool_types import (
    ToolResult,
    ERR_INVALID_ARGS,
    ERR_TOOL_EXCEPTION,
)


_DATETIME_PARAMS = {"contact_time", "proposed_start", "start_time", "datetime_start"}


def _try_parse_datetime(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    return value


def check_policy(
    state: EnvironmentState,
    action_type: str,
    **parameters,
) -> ToolResult:
    """Run a policy check for a specific action.

    Args:
        action_type: One of:
            - "discount"              (requires: discount_pct; optional: approver_role)
            - "contact_hours"         (requires: contact_time, customer_id)
            - "product_eligibility"   (requires: product_id, customer_id)
            - "post_demo_followup"    (no args)
            - "scheduling"            (requires: rep_id, proposed_start; optional: duration_minutes)
            - "rep_assignment"        (requires: customer_id, rep_id)
            - "quarterly_review"      (no args)
        **parameters: Keyword arguments matching the underlying policy method.

    Returns:
        ToolResult with `data = {"action_type", "passed", "violation_count", "results"}`.
    """
    engine = PolicyEngine(state)

    action_map = {
        "discount": engine.check_discount,
        "contact_hours": engine.check_contact_hours,
        "product_eligibility": engine.check_product_eligibility,
        "post_demo_followup": engine.check_post_demo_followup,
        "scheduling": engine.check_scheduling,
        "rep_assignment": engine.check_rep_assignment,
        "quarterly_review": engine.check_quarterly_review,
    }

    if action_type not in action_map:
        return ToolResult.fail(
            error=f"Unknown action_type '{action_type}'. Valid: {sorted(action_map.keys())}",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )

    # Parse datetime-like parameters
    for key in list(parameters.keys()):
        if key in _DATETIME_PARAMS:
            parameters[key] = _try_parse_datetime(parameters[key])

    try:
        report = action_map[action_type](**parameters)
    except TypeError as e:
        return ToolResult.fail(
            error=f"Invalid parameters for '{action_type}': {e}",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )
    except Exception as e:
        return ToolResult.fail(
            error=f"Policy check '{action_type}' raised: {type(e).__name__}: {e}",
            error_code=ERR_TOOL_EXCEPTION,
            timestamp=state.reference_time,
        )

    return ToolResult.ok(
        data={
            "action_type": action_type,
            "passed": report.passed,
            "violation_count": report.violation_count,
            "results": [
                {
                    "rule_id": r.rule_id,
                    "rule_name": r.rule_name,
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity,
                }
                for r in report.results
            ],
        },
        timestamp=state.reference_time,
    )


def get_policies(
    state: EnvironmentState,
    category: str | None = None,
) -> ToolResult:
    """List business policies.

    Args:
        category: Optional filter on `rule_type.value`
                  (e.g., "discount", "scheduling", "contact").
    """
    results = []
    for rule in state.policy_rules.values():
        if not rule.active:
            continue
        if category and rule.rule_type.value != category:
            continue
        results.append(rule)

    results.sort(key=lambda r: r.rule_id)

    return ToolResult.ok(
        data={
            "count": len(results),
            "policies": [r.model_dump(mode="json") for r in results],
        },
        timestamp=state.reference_time,
    )
