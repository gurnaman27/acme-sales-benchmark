"""Tests for the evaluator, failure taxonomy, and result models."""

from acme.environment.data_generator import generate_seed_data
from acme.evaluation.evaluator import Evaluator
from acme.evaluation.failure_taxonomy import FailureType
from acme.evaluation.metrics import BenchmarkResult
from acme.environment.policies import PolicyEngine
from acme.tasks.task_library import get_task_by_id
from acme.tasks.task_schema import (
    BenchmarkTask,
    ExpectedOutcome,
    TaskCategory,
)
from acme.tools.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prepare(task_id: str):
    """Load task, generate state, apply patch, snapshot before, return all."""
    state = generate_seed_data(seed=42)
    task = get_task_by_id(task_id)
    state.apply_patch(task.initial_state_patch)
    before = state.snapshot()
    registry = ToolRegistry(state)
    return task, state, before, registry


# ---------------------------------------------------------------------------
# Perfect runs
# ---------------------------------------------------------------------------

def test_perfect_l1_task():
    """T004: check eligibility — agent calls the one required tool."""
    task, state, before, reg = prepare("T004")
    reg.call("check_product_eligibility", product_id="PROD-003", customer_id="C001")
    result = Evaluator().evaluate(task, before, state, reg.log)
    assert result.success is True
    assert result.score == 1.0
    assert result.failure_type is None


def test_perfect_l2_task():
    """T014: update C001 notes."""
    task, state, before, reg = prepare("T014")
    reg.call("update_customer", customer_id="C001", updates={"notes": "Called about renewal"})
    result = Evaluator().evaluate(task, before, state, reg.log)
    assert result.success is True
    assert result.score == 1.0


def test_perfect_l2_opportunity_stage():
    """T010: move C001's opportunity to negotiation."""
    task, state, before, reg = prepare("T010")
    opp = next(o for o in state.opportunities.values() if o.customer_id == "C001")
    reg.call("update_opportunity", opportunity_id=opp.opportunity_id, updates={"stage": "negotiation"})
    result = Evaluator().evaluate(task, before, state, reg.log)
    assert result.success is True


def test_perfect_followup_create():
    """T011: create a follow-up."""
    task, state, before, reg = prepare("T011")
    reg.call("create_followup", customer_id="C003", action="Send contract", due_date="2025-03-22")
    result = Evaluator().evaluate(task, before, state, reg.log)
    assert result.success is True


# ---------------------------------------------------------------------------
# Failures by category
# ---------------------------------------------------------------------------

def test_retrieval_failure():
    """T001 requires get_customer + search_opportunities. Agent calls wrong ID."""
    task, state, before, reg = prepare("T001")
    reg.call("get_customer", customer_id="C999")          # NOT_FOUND
    reg.call("search_opportunities", customer_id="C999")  # empty
    result = Evaluator().evaluate(task, before, state, reg.log)
    assert result.success is False
    assert result.failure_type == FailureType.RETRIEVAL
    assert "not_found_calls" in result.failure_details


def test_planning_failure_missing_tool():
    """T001 requires two tools; agent only calls one."""
    task, state, before, reg = prepare("T001")
    reg.call("get_customer", customer_id="C001")   # only this
    result = Evaluator().evaluate(task, before, state, reg.log)
    assert result.success is False
    assert result.failure_type == FailureType.PLANNING
    assert "search_opportunities" in result.failure_details["missing_tools"]


def test_execution_failure_wrong_state():
    """T010 requires stage=negotiation; agent leaves it unchanged."""
    task, state, before, reg = prepare("T010")
    opp = next(o for o in state.opportunities.values() if o.customer_id == "C001")
    # Call the required tool but with no actual state change
    reg.call("update_opportunity", opportunity_id=opp.opportunity_id, updates={"notes": "noop"})
    result = Evaluator().evaluate(task, before, state, reg.log)
    assert result.success is False
    assert result.failure_type == FailureType.EXECUTION
    assert result.state_correctness < 1.0


def test_policy_violation_failure():
    """Synthetic task: force a POL-007 violation by reassigning a P1/P2 customer.

    To have a clean delta, we first PATCH C005 onto a senior rep so
    the "before" state has zero violations for C005, then reassign to
    a non-senior rep — introducing exactly one violation.
    """
    task = BenchmarkTask(
        task_id="TEST-POLICY",
        category=TaskCategory.FOLLOW_UP,
        difficulty=2,
        instruction="Assign C005 to REP-003.",
        expected_outcome=ExpectedOutcome(
            required_state={
                "customers": [{"customer_id": "C005", "assigned_rep": "REP-003"}],
            },
            required_tool_calls=["update_customer"],
            max_policy_violations=0,
        ),
    )
    state = generate_seed_data(seed=42)

    # Establish a clean baseline: C005 (P2) assigned to a senior_ae rep
    state.apply_patch({
        "customers": {"C005": {"assigned_rep": "REP-001"}}
    })
    before = state.snapshot()

    # Sanity check: baseline has 0 violations *for C005*
    before_audit = PolicyEngine(before).audit_full_state()
    c005_violations_before = [
        r for r in before_audit.violations
        if r.rule_id == "POL-007" and "C005" in r.message
    ]
    assert len(c005_violations_before) == 0, "Baseline should have no C005 POL-007 violation"

    # Now introduce the violation
    reg = ToolRegistry(state)
    reg.call("update_customer", customer_id="C005", updates={"assigned_rep": "REP-003"})

    result = Evaluator().evaluate(task, before, state, reg.log)

    assert result.success is False
    assert result.failure_type == FailureType.POLICY_VIOLATION
    assert result.policy_violations_introduced >= 1


def test_forbidden_tool_failure():
    """Synthetic task: agent uses a tool that's forbidden."""
    task = BenchmarkTask(
        task_id="TEST-FORBIDDEN",
        category=TaskCategory.FOLLOW_UP,
        difficulty=1,
        instruction="Report C001's company name.",
        expected_outcome=ExpectedOutcome(
            required_tool_calls=["get_customer"],
            forbidden_tool_calls=["update_customer"],
        ),
    )
    state = generate_seed_data(seed=42)
    before = state.snapshot()
    reg = ToolRegistry(state)
    reg.call("get_customer", customer_id="C001")
    reg.call("update_customer", customer_id="C001", updates={"notes": "sneaky"})
    result = Evaluator().evaluate(task, before, state, reg.log)
    assert result.success is False
    assert result.failure_type == FailureType.TOOL_USE
    assert "update_customer" in result.failure_details["forbidden_tools_called"]


def test_forbidden_state_failure():
    """Synthetic task: forbidden state matcher is violated."""
    task = BenchmarkTask(
        task_id="TEST-FORBIDDEN-STATE",
        category=TaskCategory.FOLLOW_UP,
        difficulty=1,
        instruction="Do not create a follow-up for C001.",
        expected_outcome=ExpectedOutcome(
            forbidden_state={
                "followups": [{"customer_id": "C001"}],
            },
        ),
    )
    state = generate_seed_data(seed=42)
    # Pre-existing follow-ups for C001? Not guaranteed; agent creates one.
    before = state.snapshot()
    reg = ToolRegistry(state)
    reg.call("create_followup", customer_id="C001", action="Oops", due_date="2025-03-20")
    result = Evaluator().evaluate(task, before, state, reg.log)
    assert result.success is False
    assert result.no_forbidden_score == 0.0


# ---------------------------------------------------------------------------
# Scoring properties
# ---------------------------------------------------------------------------

def test_score_is_in_unit_interval():
    """Composite score must always be in [0, 1]."""
    task, state, before, reg = prepare("T010")
    result = Evaluator().evaluate(task, before, state, reg.log)
    assert 0.0 <= result.score <= 1.0


def test_efficiency_penalty_for_excessive_calls():
    """Agent calls dozens of unnecessary tools."""
    task, state, before, reg = prepare("T004")
    # Required: check_product_eligibility. Let's add many extras.
    for _ in range(20):
        reg.call("get_current_time")
    reg.call("check_product_eligibility", product_id="PROD-003", customer_id="C001")
    result = Evaluator().evaluate(task, before, state, reg.log)
    assert result.efficiency_score < 1.0
    assert result.tool_call_count == 21


def test_zero_tool_calls_no_mutation():
    """Agent does nothing — should fail planning."""
    task, state, before, reg = prepare("T014")
    result = Evaluator().evaluate(task, before, state, reg.log)
    assert result.success is False
    assert result.tool_call_count == 0
    # The state didn't change, so state_correctness should be 0
    assert result.state_correctness == 0.0


def test_partial_state_credit():
    """Task with 2 matchers, 1 satisfied → 0.5 state correctness."""
    task = BenchmarkTask(
        task_id="TEST-PARTIAL",
        category=TaskCategory.FOLLOW_UP,
        difficulty=2,
        instruction="Two updates.",
        expected_outcome=ExpectedOutcome(
            required_state={
                "customers": [
                    {"customer_id": "C001", "notes": "first"},
                    {"customer_id": "C002", "notes": "second"},
                ],
            },
            required_tool_calls=["update_customer"],
        ),
    )
    state = generate_seed_data(seed=42)
    before = state.snapshot()
    reg = ToolRegistry(state)
    # Only satisfy the first matcher
    reg.call("update_customer", customer_id="C001", updates={"notes": "first"})
    result = Evaluator().evaluate(task, before, state, reg.log)
    assert result.state_correctness == 0.5
    assert result.success is False


def test_policy_violation_delta_not_absolute():
    """Pre-existing violations should not penalize the agent.

    A task whose seed data already has POL-007 violations should not
    fail just because those violations exist in the final state.
    """
    task = BenchmarkTask(
        task_id="TEST-DELTA",
        category=TaskCategory.FOLLOW_UP,
        difficulty=1,
        instruction="Report C001's notes.",
        expected_outcome=ExpectedOutcome(
            required_tool_calls=["get_customer"],
        ),
    )
    state = generate_seed_data(seed=42)
    before = state.snapshot()
    reg = ToolRegistry(state)
    reg.call("get_customer", customer_id="C001")
    result = Evaluator().evaluate(task, before, state, reg.log)
    # Even if seed has pre-existing violations, introduced=0 → policy score 1.0
    assert result.policy_compliance_score == 1.0
    assert result.policy_violations_introduced == 0


# ---------------------------------------------------------------------------
# Result model properties
# ---------------------------------------------------------------------------

def test_task_result_has_all_fields():
    task, state, before, reg = prepare("T004")
    reg.call("check_product_eligibility", product_id="PROD-003", customer_id="C001")
    result = Evaluator().evaluate(task, before, state, reg.log)
    assert result.task_id == "T004"
    assert result.category == "product_recommendation"
    assert result.difficulty == 1
    assert result.tool_call_count == 1
    assert result.successful_tool_calls == 1
    assert result.failed_tool_calls == 0


def test_benchmark_result_aggregation():
    task = get_task_by_id("T004")

    # Build 3 results manually
    r1 = Evaluator().evaluate(*_run(task, "check_product_eligibility", product_id="PROD-003", customer_id="C001"))
    r2 = Evaluator().evaluate(*_run(task, "check_product_eligibility", product_id="PROD-999", customer_id="C001"))
    r3 = Evaluator().evaluate(*_run(task, "check_product_eligibility", product_id="PROD-003", customer_id="C001"))

    br = BenchmarkResult(results=[r1, r2, r3])
    assert br.total_tasks == 3
    assert br.successful_tasks == 2
    assert abs(br.success_rate - 2/3) < 1e-9
    assert br.mean_score > 0.0
    breakdown = br.by_category()
    assert "product_recommendation" in breakdown
    assert breakdown["product_recommendation"]["count"] == 3


def test_benchmark_result_failure_breakdown():
    task = get_task_by_id("T004")
    r1 = Evaluator().evaluate(*_run(task, "check_product_eligibility", product_id="PROD-999", customer_id="C001"))
    r2 = Evaluator().evaluate(*_run(task))  # no calls
    br = BenchmarkResult(results=[r1, r2])
    fb = br.failure_breakdown()
    # r1 → retrieval failure; r2 → planning failure (no tool calls)
    assert fb.get("retrieval", 0) >= 1
    assert fb.get("planning", 0) >= 1


def test_benchmark_result_empty():
    br = BenchmarkResult(results=[])
    assert br.total_tasks == 0
    assert br.success_rate == 0.0
    assert br.mean_score == 0.0


def test_by_difficulty_breakdown():
    t4 = get_task_by_id("T004")
    t10 = get_task_by_id("T010")

    r4 = Evaluator().evaluate(*_run(t4, "check_product_eligibility", product_id="PROD-003", customer_id="C001"))
    # For T010, find C001's opp and update it
    state = generate_seed_data(seed=42)
    state.apply_patch(t10.initial_state_patch)
    before = state.snapshot()
    reg = ToolRegistry(state)
    opp = next(o for o in state.opportunities.values() if o.customer_id == "C001")
    reg.call("update_opportunity", opportunity_id=opp.opportunity_id, updates={"stage": "negotiation"})
    r10 = Evaluator().evaluate(t10, before, state, reg.log)

    br = BenchmarkResult(results=[r4, r10])
    bd = br.by_difficulty()
    assert 1 in bd and 2 in bd
    assert bd[1]["count"] == 1
    assert bd[2]["count"] == 1


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _run(task, *tool_calls, **first_kwargs):
    """Run a task with a specified first call (or no calls) and return
    the evaluator inputs."""
    state = generate_seed_data(seed=42)
    state.apply_patch(task.initial_state_patch)
    before = state.snapshot()
    reg = ToolRegistry(state)
    if tool_calls:
        reg.call(tool_calls[0], **first_kwargs)
    return task, before, state, reg.log