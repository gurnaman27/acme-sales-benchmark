"""First 15 benchmark tasks (8 L1 + 7 L2).

Every task is designed to be solvable with the 21 tools from Phase 2
and checkable by comparing the final state + action log to
`expected_outcome`. All tasks are anchored to the seed data (seed=42).
"""

from __future__ import annotations

from acme.tasks.task_schema import (
    BenchmarkTask,
    ExpectedOutcome,
    TaskCategory,
)


# ---------------------------------------------------------------------------
# L1 — Basic (1-2 tools, read-only)
# ---------------------------------------------------------------------------

T001 = BenchmarkTask(
    task_id="T001",
    category=TaskCategory.OPPORTUNITY_MGMT,
    difficulty=1,
    instruction=(
        "Find customer C001 and report the current stage of their "
        "highest-value open opportunity."
    ),
    expected_outcome=ExpectedOutcome(
        required_tool_calls=["get_customer", "search_opportunities"],
    ),
    notes="Read-only lookup. Checks that both tools were called.",
)


T002 = BenchmarkTask(
    task_id="T002",
    category=TaskCategory.FOLLOW_UP,
    difficulty=1,
    instruction=(
        "List all customers in the Healthcare industry. How many are there?"
    ),
    expected_outcome=ExpectedOutcome(
        required_tool_calls=["search_customers"],
    ),
    notes="Simple filter search.",
)


T003 = BenchmarkTask(
    task_id="T003",
    category=TaskCategory.PRODUCT_RECOMMENDATION,
    difficulty=1,
    instruction=(
        "Find product PROD-003 and report its base_price and "
        "minimum customer tier."
    ),
    expected_outcome=ExpectedOutcome(
        required_tool_calls=["get_product"],
    ),
    notes="Single product lookup.",
)


T004 = BenchmarkTask(
    task_id="T004",
    category=TaskCategory.PRODUCT_RECOMMENDATION,
    difficulty=1,
    instruction=(
        "Check whether customer C001 is eligible to purchase product "
        "PROD-003. Report the reason for the result."
    ),
    expected_outcome=ExpectedOutcome(
        required_tool_calls=["check_product_eligibility"],
    ),
    notes="Eligibility check runs POL-003.",
)


T005 = BenchmarkTask(
    task_id="T005",
    category=TaskCategory.FOLLOW_UP,
    difficulty=1,
    instruction=(
        "What is the current simulation date and day of the week?"
    ),
    expected_outcome=ExpectedOutcome(
        required_tool_calls=["get_current_time"],
    ),
    notes="Agents must use simulation time, not wall-clock.",
)


T006 = BenchmarkTask(
    task_id="T006",
    category=TaskCategory.POLICY_CONFLICT,
    difficulty=1,
    instruction=(
        "List all business policies of type 'discount'."
    ),
    expected_outcome=ExpectedOutcome(
        required_tool_calls=["get_policies"],
    ),
    notes="Policy discovery.",
)


T007 = BenchmarkTask(
    task_id="T007",
    category=TaskCategory.FOLLOW_UP,
    difficulty=1,
    instruction=(
        "Get the full interaction history for customer C005. "
        "Report how many interactions, meetings, and follow-ups they have."
    ),
    expected_outcome=ExpectedOutcome(
        required_tool_calls=["get_customer_history"],
    ),
    notes="Single history lookup.",
)


T008 = BenchmarkTask(
    task_id="T008",
    category=TaskCategory.OPPORTUNITY_MGMT,
    difficulty=1,
    instruction=(
        "Find the open opportunity with the highest expected value "
        "across the entire pipeline. Report its opportunity_id."
    ),
    expected_outcome=ExpectedOutcome(
        required_tool_calls=["search_opportunities"],
    ),
    notes="Agent must sort/scan results.",
)


# ---------------------------------------------------------------------------
# L2 — Multi-step (2-4 tools, one state change)
# ---------------------------------------------------------------------------

T009 = BenchmarkTask(
    task_id="T009",
    category=TaskCategory.OPPORTUNITY_MGMT,
    difficulty=2,
    instruction=(
        "Review customer C005's recent interaction history, then update "
        "the next_action field on their open opportunity to 'send_proposal'."
    ),
    initial_state_patch={
        "opportunities": {
            "OPP-9009": {
                "opportunity_id": "OPP-9009",
                "customer_id": "C005",
                "product_ids": ["PROD-005"],
                "stage": "proposal",
                "expected_value": 12000.0,
                "probability": 0.5,
                "assigned_rep": "REP-001",
                "created_at": "2025-02-15T10:00:00+00:00",
                "last_contact": "2025-03-01T10:00:00+00:00",
                "next_action": "schedule_demo",
                "notes": "Seed opportunity for T009",
                "close_date": "2025-04-15",
            },
        },
    },
    expected_outcome=ExpectedOutcome(
        required_state={
            "opportunities": [
                {"customer_id": "C005", "next_action": "send_proposal"},
            ],
        },
        required_tool_calls=["get_customer_history", "update_opportunity"],
    ),
    notes="Multi-step: read history then write. Patch guarantees C005 has an open opportunity.",
)


T010 = BenchmarkTask(
    task_id="T010",
    category=TaskCategory.OPPORTUNITY_MGMT,
    difficulty=2,
    instruction=(
        "Find customer C001's open opportunity and move it to the "
        "'negotiation' stage."
    ),
    initial_state_patch={
        "opportunities": {
            "OPP-9010": {
                "opportunity_id": "OPP-9010",
                "customer_id": "C001",
                "product_ids": ["PROD-003"],
                "stage": "proposal",
                "expected_value": 50000.0,
                "probability": 0.5,
                "assigned_rep": "REP-001",
                "created_at": "2025-02-15T10:00:00+00:00",
                "last_contact": "2025-03-01T10:00:00+00:00",
                "next_action": "send_proposal",
                "notes": "Seed opportunity for T010",
                "close_date": "2025-04-15",
            },
        },
    },
    expected_outcome=ExpectedOutcome(
        required_state={
            "opportunities": [
                {"customer_id": "C001", "stage": "negotiation"},
            ],
        },
        required_tool_calls=["update_opportunity"],
    ),
    notes="Simple stage transition. Patch guarantees C001 has an open opportunity.",
)


T011 = BenchmarkTask(
    task_id="T011",
    category=TaskCategory.FOLLOW_UP,
    difficulty=2,
    instruction=(
        "Create a follow-up task for customer C003 with action "
        "'Send contract' and due date 2025-03-22."
    ),
    expected_outcome=ExpectedOutcome(
        required_state={
            "followups": [
                {
                    "customer_id": "C003",
                    "action": "Send contract",
                    "status": "pending",
                },
            ],
        },
        required_tool_calls=["create_followup"],
    ),
    notes="Tests created_at uses reference_time.",
)


T012 = BenchmarkTask(
    task_id="T012",
    category=TaskCategory.SCHEDULING,
    difficulty=2,
    instruction=(
        "Cancel the scheduled meeting MTG-9001 with the reason "
        "'customer request'."
    ),
    initial_state_patch={
        "meetings": {
            "MTG-9001": {
                "meeting_id": "MTG-9001",
                "customer_id": "C001",
                "rep_id": "REP-001",
                "datetime_start": "2025-03-20T17:00:00+00:00",
                "duration_minutes": 30,
                "status": "scheduled",
                "meeting_type": "video",
                "purpose": "general",
                "notes": "",
                "location": "",
            },
        },
    },
    expected_outcome=ExpectedOutcome(
        required_state={
            "meetings": [
                {"meeting_id": "MTG-9001", "status": "cancelled"},
            ],
        },
        required_tool_calls=["cancel_meeting"],
    ),
    notes="Patch adds MTG-9001 to guarantee it exists.",
)


T013 = BenchmarkTask(
    task_id="T013",
    category=TaskCategory.OPPORTUNITY_MGMT,
    difficulty=2,
    instruction=(
        "Create a new opportunity for customer C005 with product "
        "PROD-005, at the 'qualification' stage."
    ),
    expected_outcome=ExpectedOutcome(
        required_state={
            "opportunities": [
                {
                    "customer_id": "C005",
                    "stage": "qualification",
                    "product_ids": ["PROD-005"],
                },
            ],
        },
        required_tool_calls=["create_opportunity"],
    ),
    notes="Tests create_opportunity, id generation, expected_value compute.",
)


T014 = BenchmarkTask(
    task_id="T014",
    category=TaskCategory.FOLLOW_UP,
    difficulty=2,
    instruction=(
        "Update customer C001's notes field to 'Called about renewal'."
    ),
    expected_outcome=ExpectedOutcome(
        required_state={
            "customers": [
                {"customer_id": "C001", "notes": "Called about renewal"},
            ],
        },
        required_tool_calls=["update_customer"],
    ),
    notes="Simple field update.",
)


T015 = BenchmarkTask(
    task_id="T015",
    category=TaskCategory.FOLLOW_UP,
    difficulty=2,
    instruction=(
        "Find every follow-up currently marked as 'overdue' and mark "
        "each one as completed."
    ),
    expected_outcome=ExpectedOutcome(
        forbidden_state={
            "followups": [{"status": "overdue"}],
        },
        required_tool_calls=["complete_followup"],
        max_tool_calls=5,   # ← 1 read + up to 3 completes + 1 slack
    ),
    notes=(
        "Requires fetching then updating in a loop. "
        "Budget = 5 allows the required 1+3 calls plus one slack."
    ),
)


# ---------------------------------------------------------------------------
# L3 — Constraint-Based (3+ tools, policies, business rules)
# ---------------------------------------------------------------------------

T016 = BenchmarkTask(
    task_id="T016",
    category=TaskCategory.SCHEDULING,
    difficulty=3,
    instruction=(
        "Check rep REP-001's calendar for 2025-03-24. Schedule a 60-minute "
        "meeting with customer C004 on 2025-03-24 at 18:00 UTC (respecting "
        "working hours and buffer time)."
    ),
    expected_outcome=ExpectedOutcome(
        required_state={
            "meetings": [
                {"customer_id": "C004", "rep_id": "REP-001", "status": "scheduled"},
            ],
        },
        required_tool_calls=["get_available_slots", "schedule_meeting"],
    ),
    notes="Constraint-based scheduling respecting business hours and buffer.",
)


T017 = BenchmarkTask(
    task_id="T017",
    category=TaskCategory.POLICY_CONFLICT,
    difficulty=3,
    instruction=(
        "Customer C002 requested a 25% discount on opportunity. Check "
        "discount policy POL-001. If 25% exceeds the maximum allowed "
        "discount (20%), refuse the request and do NOT update the opportunity discount."
    ),
    expected_outcome=ExpectedOutcome(
        required_tool_calls=["get_policies"],
        forbidden_state={
            "opportunities": [{"discount_percent": 25.0}],
        },
        max_tool_calls=5,
    ),
    notes="Policy constraint check: agent must refuse discount > 20%.",
)


T018 = BenchmarkTask(
    task_id="T018",
    category=TaskCategory.PRODUCT_RECOMMENDATION,
    difficulty=3,
    instruction=(
        "Customer C006 requested to buy product PROD-004. Check customer "
        "eligibility for PROD-004 using check_product_eligibility. If customer "
        "is churned or ineligible, report the reason and do NOT create an opportunity."
    ),
    expected_outcome=ExpectedOutcome(
        required_tool_calls=["check_product_eligibility"],
        forbidden_state={
            "opportunities": [{"customer_id": "C006", "product_ids": ["PROD-004"]}],
        },
        max_tool_calls=5,
    ),
    notes="Product eligibility constraint check for churned customer.",
)


T019 = BenchmarkTask(
    task_id="T019",
    category=TaskCategory.SCHEDULING,
    difficulty=3,
    instruction=(
        "Check calendar for rep REP-002 on 2025-03-24. Find an open 30-minute slot "
        "after 17:00 UTC and schedule a meeting with customer C003."
    ),
    expected_outcome=ExpectedOutcome(
        required_state={
            "meetings": [
                {"customer_id": "C003", "rep_id": "REP-002", "status": "scheduled"},
            ],
        },
        required_tool_calls=["get_available_slots", "schedule_meeting"],
    ),
    notes="Finds non-conflicting calendar slot.",
)


T020 = BenchmarkTask(
    task_id="T020",
    category=TaskCategory.FOLLOW_UP,
    difficulty=3,
    instruction=(
        "Create a post-demo follow-up task for customer C002 with due date set "
        "to 2 days after current simulation date (2025-03-17)."
    ),
    expected_outcome=ExpectedOutcome(
        required_state={
            "followups": [
                {"customer_id": "C002", "due_date": "2025-03-17"},
            ],
        },
        required_tool_calls=["get_current_time", "create_followup"],
    ),
    notes="Follow-up creation within required policy window.",
)


T021 = BenchmarkTask(
    task_id="T021",
    category=TaskCategory.POLICY_CONFLICT,
    difficulty=3,
    instruction=(
        "Request to reassign customer C001 (P1 tier) to rep REP-005. Check "
        "assignment policy POL-007 first. If rep REP-005 is not qualified "
        "for P1 priority customers, decline the reassignment."
    ),
    expected_outcome=ExpectedOutcome(
        required_tool_calls=["get_policies"],
        forbidden_state={
            "customers": [{"customer_id": "C001", "assigned_rep_id": "REP-005"}],
        },
        max_tool_calls=5,
    ),
    notes="Rep assignment policy constraint check.",
)


T022 = BenchmarkTask(
    task_id="T022",
    category=TaskCategory.OPPORTUNITY_MGMT,
    difficulty=3,
    instruction=(
        "Check if customer C003 is eligible for product PROD-002. If eligible, "
        "create a new opportunity for C003 with PROD-002 in stage 'qualification'."
    ),
    expected_outcome=ExpectedOutcome(
        required_state={
            "opportunities": [
                {"customer_id": "C003", "stage": "qualification"},
            ],
        },
        required_tool_calls=["check_product_eligibility", "create_opportunity"],
    ),
    notes="Product eligibility check followed by opportunity creation.",
)


T023 = BenchmarkTask(
    task_id="T023",
    category=TaskCategory.SCHEDULING,
    difficulty=3,
    instruction=(
        "Customer C005 requests a meeting at 07:00 UTC on 2025-03-25. Check "
        "scheduling policy POL-002 for working hours. Decline the request "
        "because 07:00 is outside standard working hours (09:00-18:00)."
    ),
    expected_outcome=ExpectedOutcome(
        required_tool_calls=["get_policies"],
        forbidden_state={
            "meetings": [{"customer_id": "C005", "start_time": "2025-03-25T07:00:00+00:00"}],
        },
        max_tool_calls=5,
    ),
    notes="Outside working hours scheduling refusal.",
)


# ---------------------------------------------------------------------------
# L4 — Complex Multi-System Workflows (4+ tools, state tracking)
# ---------------------------------------------------------------------------

T024 = BenchmarkTask(
    task_id="T024",
    category=TaskCategory.FOLLOW_UP,
    difficulty=4,
    instruction=(
        "Find customer C001, get customer history, schedule a 30-minute review "
        "meeting on 2025-03-26 at 18:00 UTC with rep REP-001, create a follow-up task "
        "'Send agenda', and update customer notes to 'Re-engagement scheduled'."
    ),
    expected_outcome=ExpectedOutcome(
        required_state={
            "meetings": [
                {"customer_id": "C001", "rep_id": "REP-001", "status": "scheduled"},
            ],
            "followups": [
                {"customer_id": "C001", "action": "Send agenda"},
            ],
            "customers": [
                {"customer_id": "C001", "notes": "Re-engagement scheduled"},
            ],
        },
        required_tool_calls=[
            "get_customer_history",
            "schedule_meeting",
            "create_followup",
            "update_customer",
        ],
    ),
    notes="Multi-step customer re-engagement workflow.",
)


T025 = BenchmarkTask(
    task_id="T025",
    category=TaskCategory.OPPORTUNITY_MGMT,
    difficulty=4,
    instruction=(
        "Check customer C002 eligibility for product PROD-001. If eligible, "
        "create an opportunity for C002 with PROD-001 in stage 'qualification', "
        "schedule a demo meeting on 2025-03-25 at 18:00 UTC with rep REP-001, and "
        "create a follow-up task 'Post-demo review'."
    ),
    expected_outcome=ExpectedOutcome(
        required_state={
            "opportunities": [
                {"customer_id": "C002", "stage": "qualification"},
            ],
            "meetings": [
                {"customer_id": "C002", "rep_id": "REP-001"},
            ],
            "followups": [
                {"customer_id": "C002", "action": "Post-demo review"},
            ],
        },
        required_tool_calls=[
            "check_product_eligibility",
            "create_opportunity",
            "schedule_meeting",
            "create_followup",
        ],
    ),
    notes="Multi-system deal creation workflow.",
)


T026 = BenchmarkTask(
    task_id="T026",
    category=TaskCategory.POLICY_CONFLICT,
    difficulty=4,
    instruction=(
        "Process deal for customer C001: check discount policy POL-001 for 15% discount, "
        "check eligibility for PROD-001, create an opportunity for C001 with PROD-001 "
        "and 15% discount in stage 'proposal', and create a high-priority follow-up 'Executive approval'."
    ),
    expected_outcome=ExpectedOutcome(
        required_state={
            "opportunities": [
                {"customer_id": "C001", "stage": "proposal", "discount_percent": 15.0},
            ],
            "followups": [
                {"customer_id": "C001", "action": "Executive approval"},
            ],
        },
        required_tool_calls=[
            "get_policies",
            "check_product_eligibility",
            "create_opportunity",
            "create_followup",
        ],
    ),
    notes="Enterprise deal processing with policy verification.",
)


T027 = BenchmarkTask(
    task_id="T027",
    category=TaskCategory.SCHEDULING,
    difficulty=4,
    instruction=(
        "Rep REP-001 has meeting MTG-9002 on 2025-03-20T18:00:00+00:00. Cancel MTG-9002 "
        "due to scheduling conflict and schedule a replacement meeting for customer "
        "C001 on 2025-03-27 at 18:00 UTC."
    ),
    initial_state_patch={
        "meetings": {
            "MTG-9002": {
                "meeting_id": "MTG-9002",
                "customer_id": "C001",
                "rep_id": "REP-001",
                "datetime_start": "2025-03-20T18:00:00+00:00",
                "duration_minutes": 30,
                "status": "scheduled",
                "meeting_type": "video",
                "purpose": "demo",
                "notes": "",
                "location": "",
            },
        },
    },
    expected_outcome=ExpectedOutcome(
        required_state={
            "meetings": [
                {"meeting_id": "MTG-9002", "status": "cancelled"},
                {"customer_id": "C001", "rep_id": "REP-001", "status": "scheduled"},
            ],
        },
        required_tool_calls=["cancel_meeting", "schedule_meeting"],
    ),
    notes="Reschedule rep calendar due to conflict.",
)


T028 = BenchmarkTask(
    task_id="T028",
    category=TaskCategory.PRODUCT_RECOMMENDATION,
    difficulty=4,
    instruction=(
        "Check customer C004 profile. Check eligibility for PROD-001 and PROD-002. "
        "Identify the eligible product (PROD-002) and create an opportunity for C004 "
        "with PROD-002 in stage 'qualification'."
    ),
    expected_outcome=ExpectedOutcome(
        required_state={
            "opportunities": [
                {"customer_id": "C004", "stage": "qualification"},
            ],
        },
        required_tool_calls=["get_customer", "check_product_eligibility", "create_opportunity"],
    ),
    notes="Best product recommendation and opportunity creation.",
)


T029 = BenchmarkTask(
    task_id="T029",
    category=TaskCategory.FOLLOW_UP,
    difficulty=4,
    instruction=(
        "Search for all customers in the Healthcare industry (C002, C017). Create "
        "a follow-up task titled 'Quarterly Business Review' for customer C002 and customer C017."
    ),
    expected_outcome=ExpectedOutcome(
        required_state={
            "followups": [
                {"customer_id": "C002", "action": "Quarterly Business Review"},
                {"customer_id": "C017", "action": "Quarterly Business Review"},
            ],
        },
        required_tool_calls=["search_customers", "create_followup"],
    ),
    notes="Batch quarterly account review follow-ups.",
)


T030 = BenchmarkTask(
    task_id="T030",
    category=TaskCategory.OPPORTUNITY_MGMT,
    difficulty=4,
    instruction=(
        "Find opportunity OPP-9001 for customer C005, update its next_action to "
        "'Re-engage customer', and create a follow-up task 'Stale opportunity outreach' for customer C005."
    ),
    initial_state_patch={
        "opportunities": {
            "OPP-9001": {
                "opportunity_id": "OPP-9001",
                "customer_id": "C005",
                "product_ids": ["PROD-001"],
                "stage": "prospecting",
                "expected_value": 10000.0,
                "probability": 0.2,
                "close_date": "2025-05-01",
                "assigned_rep": "REP-001",
                "created_at": "2025-03-01T10:00:00+00:00",
                "next_action": "initial_contact",
                "notes": "stale",
            },
        },
    },
    expected_outcome=ExpectedOutcome(
        required_state={
            "opportunities": [
                {"opportunity_id": "OPP-9001", "next_action": "Re-engage customer"},
            ],
            "followups": [
                {"customer_id": "C005", "action": "Stale opportunity outreach"},
            ],
        },
        required_tool_calls=["update_opportunity", "create_followup"],
    ),
    notes="Stale opportunity cleanup and follow-up creation.",
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_LIBRARY: list[BenchmarkTask] = [
    T001, T002, T003, T004, T005, T006, T007, T008,   # L1
    T009, T010, T011, T012, T013, T014, T015,         # L2
    T016, T017, T018, T019, T020, T021, T022, T023,   # L3
    T024, T025, T026, T027, T028, T029, T030,         # L4
]


def get_task_by_id(task_id: str) -> BenchmarkTask | None:
    for task in TASK_LIBRARY:
        if task.task_id == task_id:
            return task
    return None


def get_tasks_by_difficulty(level: int) -> list[BenchmarkTask]:
    return [t for t in TASK_LIBRARY if t.difficulty == level]


def get_tasks_by_category(category: TaskCategory) -> list[BenchmarkTask]:
    return [t for t in TASK_LIBRARY if t.category == category]


__all__ = [
    "TASK_LIBRARY",
    "get_task_by_id",
    "get_tasks_by_difficulty",
    "get_tasks_by_category",
]