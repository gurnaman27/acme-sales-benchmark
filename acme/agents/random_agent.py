"""Random baseline agent for lower-bound comparison.

Picks a random tool from the registry and calls it with random
plausible arguments. Serves as a principled lower-bound:
any real agent should easily beat a random policy.
"""

from __future__ import annotations

import random

from acme.agents.agent_interface import ParticipantAgent
from acme.tasks.task_schema import BenchmarkTask
from acme.tools.tool_registry import ToolRegistry


# Pre-defined plausible argument pools for random selection
_CUSTOMER_IDS = ["C001", "C002", "C003", "C004", "C005", "C006"]
_PRODUCT_IDS = ["PROD-001", "PROD-002", "PROD-003", "PROD-004", "PROD-005"]
_REP_IDS = ["REP-001", "REP-002", "REP-003"]
_POLICY_CATS = ["discount", "scheduling", "assignment", "eligibility"]
_INDUSTRIES = ["Technology", "Healthcare", "Finance", "Retail"]
_STAGES = ["prospecting", "qualification", "proposal", "negotiation"]


def _random_args(tool_name: str, rng: random.Random) -> dict:
    """Generate plausible random arguments for a tool."""
    if tool_name == "get_customer":
        return {"customer_id": rng.choice(_CUSTOMER_IDS)}
    if tool_name == "search_customers":
        return {"industry": rng.choice(_INDUSTRIES)}
    if tool_name == "update_customer":
        return {"customer_id": rng.choice(_CUSTOMER_IDS), "updates": {"notes": "random"}}
    if tool_name == "get_product":
        return {"product_id": rng.choice(_PRODUCT_IDS)}
    if tool_name == "check_product_eligibility":
        return {"customer_id": rng.choice(_CUSTOMER_IDS), "product_id": rng.choice(_PRODUCT_IDS)}
    if tool_name == "search_opportunities":
        return {"customer_id": rng.choice(_CUSTOMER_IDS)}
    if tool_name == "create_opportunity":
        return {
            "customer_id": rng.choice(_CUSTOMER_IDS),
            "product_ids": [rng.choice(_PRODUCT_IDS)],
            "stage": rng.choice(_STAGES),
        }
    if tool_name == "update_opportunity":
        return {"opportunity_id": "OPP-0001", "updates": {"stage": rng.choice(_STAGES)}}
    if tool_name == "get_customer_history":
        return {"customer_id": rng.choice(_CUSTOMER_IDS)}
    if tool_name == "get_policies":
        return {"category": rng.choice(_POLICY_CATS)}
    if tool_name == "get_current_time":
        return {}
    if tool_name == "get_available_slots":
        return {"rep_id": rng.choice(_REP_IDS), "start_date": "2025-03-24", "end_date": "2025-03-24"}
    if tool_name == "schedule_meeting":
        return {
            "customer_id": rng.choice(_CUSTOMER_IDS),
            "rep_id": rng.choice(_REP_IDS),
            "start_time": "2025-03-25T18:00:00+00:00",
            "duration_minutes": 30,
        }
    if tool_name == "cancel_meeting":
        return {"meeting_id": "MTG-9001", "reason": "random"}
    if tool_name == "create_followup":
        return {
            "customer_id": rng.choice(_CUSTOMER_IDS),
            "action": "Random followup",
            "due_date": "2025-03-25",
        }
    if tool_name == "complete_followup":
        return {"followup_id": "FU-0001"}
    if tool_name == "get_followups":
        return {}
    return {}


class RandomAgent(ParticipantAgent):
    """Random baseline agent — picks tools at random.

    Useful as a lower-bound comparison: any structured agent
    should significantly outperform random tool selection.
    """

    def __init__(self, seed: int = 42, max_steps: int = 5):
        self.rng = random.Random(seed)
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "random_baseline"

    def run(self, task: BenchmarkTask, registry: ToolRegistry) -> str:
        available = registry.get_available_tools()
        if not available:
            return "No tools available."

        for _ in range(self.rng.randint(1, self.max_steps)):
            tool = self.rng.choice(available)
            args = _random_args(tool, self.rng)
            try:
                registry.call(tool, **args)
            except Exception:
                pass

        return "Random actions completed."


__all__ = ["RandomAgent"]
