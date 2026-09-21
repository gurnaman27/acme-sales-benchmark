"""A simple rule-based baseline agent.

This is NOT meant to be a good agent. It pattern-matches on task IDs
and executes a hardcoded tool sequence. Its purpose is to prove the
Green Agent pipeline works end-to-end.
"""

from __future__ import annotations

import re

from acme.agents.agent_interface import ParticipantAgent
from acme.tasks.task_schema import BenchmarkTask
from acme.tools.tool_registry import ToolRegistry


class SimpleAgent(ParticipantAgent):
    """Deterministic rule-based agent. Handles a subset of L1/L2 tasks."""

    @property
    def name(self) -> str:
        return "simple_rule_based"

    def run(self, task: BenchmarkTask, registry: ToolRegistry) -> str:
        handler = getattr(self, f"_handle_{task.task_id}", None)
        if handler is None:
            return f"[no handler for {task.task_id}]"
        try:
            return handler(task, registry)
        except Exception as e:
            return f"[error in {task.task_id}: {type(e).__name__}: {e}]"

    # ------------------------------------------------------------------
    # L1 handlers (read-only)
    # ------------------------------------------------------------------

    def _handle_T001(self, task, registry) -> str:
        c = registry.call("get_customer", customer_id="C001")
        if not c.success:
            return f"failed to fetch C001: {c.error}"
        opps = registry.call("search_opportunities", customer_id="C001")
        open_opps = [
            o for o in opps.data["opportunities"]
            if o["stage"] not in ("closed_won", "closed_lost")
        ]
        if not open_opps:
            return "C001 has no open opportunities."
        top = max(open_opps, key=lambda o: o["expected_value"])
        return f"C001's highest-value open opportunity is at stage '{top['stage']}'."

    def _handle_T002(self, task, registry) -> str:
        r = registry.call("search_customers", industry="Healthcare")
        return f"Found {r.data['count']} Healthcare customers."

    def _handle_T003(self, task, registry) -> str:
        r = registry.call("get_product", product_id="PROD-003")
        return (
            f"PROD-003 base_price=${r.data['base_price']}, "
            f"min tier={r.data['min_customer_tier']}."
        )

    def _handle_T004(self, task, registry) -> str:
        r = registry.call(
            "check_product_eligibility",
            product_id="PROD-003",
            customer_id="C001",
        )
        return f"Eligible: {r.data['eligible']}. Reason: {r.data['reason']}"

    def _handle_T005(self, task, registry) -> str:
        r = registry.call("get_current_time")
        return f"Simulation date is {r.data['date']} ({r.data['day_of_week']})."

    def _handle_T006(self, task, registry) -> str:
        r = registry.call("get_policies", category="discount")
        return f"Found {r.data['count']} discount policies."

    def _handle_T007(self, task, registry) -> str:
        r = registry.call("get_customer_history", customer_id="C005")
        t = r.data["totals"]
        return (
            f"C005 has {t['interactions']} interactions, "
            f"{t['meetings']} meetings, and {t['followups']} follow-ups."
        )

    def _handle_T008(self, task, registry) -> str:
        r = registry.call("search_opportunities", limit=100)
        open_opps = [
            o for o in r.data["opportunities"]
            if o["stage"] not in ("closed_won", "closed_lost")
        ]
        if not open_opps:
            return "No open opportunities."
        return f"Highest-value open opportunity: {open_opps[0]['opportunity_id']}."

    # ------------------------------------------------------------------
    # L2 handlers (mutating)
    # ------------------------------------------------------------------

    def _handle_T009(self, task, registry) -> str:
        registry.call("get_customer_history", customer_id="C005")
        opps = registry.call("search_opportunities", customer_id="C005")
        open_opps = [
            o for o in opps.data["opportunities"]
            if o["stage"] not in ("closed_won", "closed_lost")
        ]
        if not open_opps:
            return "C005 has no open opportunity."
        oid = open_opps[0]["opportunity_id"]
        registry.call(
            "update_opportunity",
            opportunity_id=oid,
            updates={"next_action": "send_proposal"},
        )
        return f"Updated {oid}'s next_action."

    def _handle_T010(self, task, registry) -> str:
        opps = registry.call("search_opportunities", customer_id="C001")
        open_opps = [
            o for o in opps.data["opportunities"]
            if o["stage"] not in ("closed_won", "closed_lost")
        ]
        if not open_opps:
            return "C001 has no open opportunity."
        oid = open_opps[0]["opportunity_id"]
        registry.call(
            "update_opportunity",
            opportunity_id=oid,
            updates={"stage": "negotiation"},
        )
        return f"Moved {oid} to negotiation."

    def _handle_T011(self, task, registry) -> str:
        registry.call(
            "create_followup",
            customer_id="C003",
            action="Send contract",
            due_date="2025-03-22",
        )
        return "Created follow-up for C003."

    def _handle_T012(self, task, registry) -> str:
        registry.call(
            "cancel_meeting",
            meeting_id="MTG-9001",
            reason="customer request",
        )
        return "Cancelled MTG-9001."

    def _handle_T013(self, task, registry) -> str:
        registry.call(
            "create_opportunity",
            customer_id="C005",
            product_ids=["PROD-005"],
            stage="qualification",
        )
        return "Created opportunity for C005."

    def _handle_T014(self, task, registry) -> str:
        registry.call(
            "update_customer",
            customer_id="C001",
            updates={"notes": "Called about renewal"},
        )
        return "Updated C001 notes."

    def _handle_T015(self, task, registry) -> str:
        r = registry.call("get_followups", overdue_only=True)
        count = 0
        for f in r.data["followups"]:
            if f["status"] == "overdue":
                registry.call("complete_followup", followup_id=f["followup_id"])
                count += 1
        return f"Completed {count} overdue follow-ups."

    # ------------------------------------------------------------------
    # L3 handlers (constraint-based)
    # ------------------------------------------------------------------

    def _handle_T016(self, task, registry) -> str:
        registry.call("get_available_slots", rep_id="REP-001", start_date="2025-03-24", end_date="2025-03-24")
        registry.call(
            "schedule_meeting",
            customer_id="C004",
            rep_id="REP-001",
            start_time="2025-03-24T18:00:00+00:00",
            duration_minutes=60,
        )
        return "Scheduled meeting for C004 with REP-001."

    def _handle_T017(self, task, registry) -> str:
        pol = registry.call("get_policies", category="discount")
        return "Refused discount request: 25% exceeds maximum allowed 20% under POL-001."

    def _handle_T018(self, task, registry) -> str:
        elig = registry.call("check_product_eligibility", customer_id="C006", product_id="PROD-004")
        return "Customer C006 is ineligible for PROD-004: customer is churned."

    def _handle_T019(self, task, registry) -> str:
        registry.call("get_available_slots", rep_id="REP-002", start_date="2025-03-24", end_date="2025-03-24")
        registry.call(
            "schedule_meeting",
            customer_id="C003",
            rep_id="REP-002",
            start_time="2025-03-24T18:00:00+00:00",
            duration_minutes=30,
        )
        return "Scheduled meeting for C003 with REP-002."

    def _handle_T020(self, task, registry) -> str:
        registry.call("get_current_time")
        registry.call(
            "create_followup",
            customer_id="C002",
            action="Post-demo follow-up",
            due_date="2025-03-17",
        )
        return "Created follow-up for C002 due on 2025-03-17."

    def _handle_T021(self, task, registry) -> str:
        registry.call("get_policies", category="assignment")
        return "Refused reassignment: REP-005 is not qualified for P1 customer C001 under POL-007."

    def _handle_T022(self, task, registry) -> str:
        registry.call("check_product_eligibility", customer_id="C003", product_id="PROD-002")
        registry.call(
            "create_opportunity",
            customer_id="C003",
            product_ids=["PROD-002"],
            stage="qualification",
        )
        return "Created opportunity for C003 with PROD-002."

    def _handle_T023(self, task, registry) -> str:
        registry.call("get_policies", category="scheduling")
        return "Declined meeting request: 07:00 is outside working hours (09:00-18:00) under POL-002."

    # ------------------------------------------------------------------
    # L4 handlers (complex multi-system workflows)
    # ------------------------------------------------------------------

    def _handle_T024(self, task, registry) -> str:
        registry.call("get_customer_history", customer_id="C001")
        registry.call(
            "schedule_meeting",
            customer_id="C001",
            rep_id="REP-001",
            start_time="2025-03-26T18:00:00+00:00",
            duration_minutes=30,
        )
        registry.call(
            "create_followup",
            customer_id="C001",
            action="Send agenda",
            due_date="2025-03-25",
        )
        registry.call(
            "update_customer",
            customer_id="C001",
            updates={"notes": "Re-engagement scheduled"},
        )
        return "Re-engagement workflow completed for C001."

    def _handle_T025(self, task, registry) -> str:
        registry.call("check_product_eligibility", customer_id="C002", product_id="PROD-001")
        registry.call(
            "create_opportunity",
            customer_id="C002",
            product_ids=["PROD-001"],
            stage="qualification",
        )
        registry.call(
            "schedule_meeting",
            customer_id="C002",
            rep_id="REP-001",
            start_time="2025-03-25T18:00:00+00:00",
            duration_minutes=30,
        )
        registry.call(
            "create_followup",
            customer_id="C002",
            action="Post-demo review",
            due_date="2025-03-26",
        )
        return "Multi-system deal creation workflow completed for C002."

    def _handle_T026(self, task, registry) -> str:
        registry.call("get_policies", category="discount")
        registry.call("check_product_eligibility", customer_id="C001", product_id="PROD-001")
        registry.call(
            "create_opportunity",
            customer_id="C001",
            product_ids=["PROD-001"],
            stage="proposal",
            discount_percent=15.0,
        )
        registry.call(
            "create_followup",
            customer_id="C001",
            action="Executive approval",
            due_date="2025-03-18",
        )
        return "Enterprise deal processing completed for C001."

    def _handle_T027(self, task, registry) -> str:
        registry.call("cancel_meeting", meeting_id="MTG-9002", reason="scheduling conflict")
        registry.call(
            "schedule_meeting",
            customer_id="C001",
            rep_id="REP-001",
            start_time="2025-03-27T18:00:00+00:00",
            duration_minutes=30,
        )
        return "Rescheduled meeting for REP-001."

    def _handle_T028(self, task, registry) -> str:
        registry.call("get_customer", customer_id="C004")
        registry.call("check_product_eligibility", customer_id="C004", product_id="PROD-001")
        registry.call("check_product_eligibility", customer_id="C004", product_id="PROD-002")
        registry.call(
            "create_opportunity",
            customer_id="C004",
            product_ids=["PROD-002"],
            stage="qualification",
        )
        return "Recommended PROD-002 and created opportunity for C004."

    def _handle_T029(self, task, registry) -> str:
        registry.call("search_customers", industry="Healthcare")
        registry.call(
            "create_followup",
            customer_id="C002",
            action="Quarterly Business Review",
            due_date="2025-03-30",
        )
        registry.call(
            "create_followup",
            customer_id="C017",
            action="Quarterly Business Review",
            due_date="2025-03-30",
        )
        return "Created quarterly business review follow-ups."

    def _handle_T030(self, task, registry) -> str:
        registry.call(
            "update_opportunity",
            opportunity_id="OPP-9001",
            updates={"next_action": "Re-engage customer"},
        )
        registry.call(
            "create_followup",
            customer_id="C005",
            action="Stale opportunity outreach",
            due_date="2025-03-20",
        )
        return "Updated stale opportunity OPP-9001 and created follow-up."