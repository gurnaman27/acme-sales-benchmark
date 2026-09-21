"""Tests for the benchmark task schema and library."""

from acme.environment.data_generator import generate_seed_data
from acme.environment.state import EnvironmentState
from acme.tasks.task_library import (
    TASK_LIBRARY,
    get_task_by_id,
    get_tasks_by_category,
    get_tasks_by_difficulty,
)
from acme.tasks.task_schema import TaskCategory
from acme.tools.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------

def test_library_loads():
    assert len(TASK_LIBRARY) == 30


def test_all_task_ids_unique():
    ids = [t.task_id for t in TASK_LIBRARY]
    assert len(ids) == len(set(ids)), "Duplicate task IDs found"


def test_all_tasks_have_instruction():
    for t in TASK_LIBRARY:
        assert t.instruction.strip(), f"{t.task_id} has empty instruction"


def test_all_difficulties_in_range():
    for t in TASK_LIBRARY:
        assert 1 <= t.difficulty <= 4, f"{t.task_id} difficulty {t.difficulty} out of range"


def test_library_has_all_levels():
    l1 = get_tasks_by_difficulty(1)
    l2 = get_tasks_by_difficulty(2)
    l3 = get_tasks_by_difficulty(3)
    l4 = get_tasks_by_difficulty(4)
    assert len(l1) == 8, f"Expected 8 L1 tasks, got {len(l1)}"
    assert len(l2) == 7, f"Expected 7 L2 tasks, got {len(l2)}"
    assert len(l3) == 8, f"Expected 8 L3 tasks, got {len(l3)}"
    assert len(l4) == 7, f"Expected 7 L4 tasks, got {len(l4)}"


def test_all_categories_are_valid():
    for t in TASK_LIBRARY:
        assert isinstance(t.category, TaskCategory)


def test_task_lookup_helpers():
    t = get_task_by_id("T001")
    assert t is not None
    assert t.task_id == "T001"

    assert get_task_by_id("T999") is None

    opp_tasks = get_tasks_by_category(TaskCategory.OPPORTUNITY_MGMT)
    assert len(opp_tasks) > 0


# ---------------------------------------------------------------------------
# Tool reference validity
# ---------------------------------------------------------------------------

def test_required_tool_calls_are_registered():
    """Every tool name in required_tool_calls must exist in the registry."""
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    available = set(registry.get_available_tools())

    for t in TASK_LIBRARY:
        for tool in t.expected_outcome.required_tool_calls:
            assert tool in available, (
                f"{t.task_id}: required tool '{tool}' is not registered"
            )


def test_available_tools_are_registered():
    """Any restricted available_tools list must only mention real tools."""
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    available = set(registry.get_available_tools())

    for t in TASK_LIBRARY:
        for tool in t.available_tools:
            assert tool in available, (
                f"{t.task_id}: available tool '{tool}' is not registered"
            )


# ---------------------------------------------------------------------------
# Patch validity
# ---------------------------------------------------------------------------

def test_patches_apply_to_seed_data():
    """Every task's initial_state_patch must apply without error."""
    for t in TASK_LIBRARY:
        state = generate_seed_data(seed=42)
        try:
            state.apply_patch(t.initial_state_patch)
        except Exception as e:
            raise AssertionError(
                f"{t.task_id}: patch failed to apply: {type(e).__name__}: {e}"
            )


def test_patched_task_T012_has_mtg_9001():
    """T012 patches MTG-9001 into the state — verify it exists after patch."""
    t = get_task_by_id("T012")
    state = generate_seed_data(seed=42)
    state.apply_patch(t.initial_state_patch)
    assert "MTG-9001" in state.meetings
    assert state.meetings["MTG-9001"].status.value == "scheduled"


# ---------------------------------------------------------------------------
# Expected outcome well-formedness
# ---------------------------------------------------------------------------

def test_read_only_tasks_have_no_required_state():
    """L1 tasks are read-only — no required_state changes expected."""
    for t in get_tasks_by_difficulty(1):
        assert not t.expected_outcome.required_state, (
            f"{t.task_id} is L1 but declares required_state"
        )
        assert t.expected_outcome.required_tool_calls, (
            f"{t.task_id} is L1 but declares no required_tool_calls"
        )


def test_mutating_tasks_have_required_state():
    """L2 tasks mutate state — they must declare required_state OR forbidden_state."""
    for t in get_tasks_by_difficulty(2):
        has_state = bool(t.expected_outcome.required_state)
        has_forbidden = bool(t.expected_outcome.forbidden_state)
        assert has_state or has_forbidden, (
            f"{t.task_id} is L2 but declares no state expectations"
        )


def test_required_state_collection_names_are_known():
    """Every collection name in required_state must exist on EnvironmentState."""
    known = {
        "sales_reps", "customers", "products", "opportunities",
        "meetings", "interactions", "followups", "policy_rules",
    }
    for t in TASK_LIBRARY:
        for coll in t.expected_outcome.required_state:
            assert coll in known, f"{t.task_id}: unknown collection '{coll}'"
        for coll in t.expected_outcome.forbidden_state:
            assert coll in known, f"{t.task_id}: unknown collection '{coll}'"


def test_no_task_forbids_its_own_required_tools():
    """Sanity: no task should require and forbid the same tool."""
    for t in TASK_LIBRARY:
        required = set(t.expected_outcome.required_tool_calls)
        forbidden = set(t.expected_outcome.forbidden_tool_calls)
        overlap = required & forbidden
        assert not overlap, (
            f"{t.task_id}: tools are both required and forbidden: {overlap}"
        )

def test_L2_update_tasks_have_pre_existing_targets():
    """Every L2 UPDATE task must have its target entity present after
    applying its patch.

    We skip CREATE tasks because their required_state describes an
    entity the agent will create — it obviously doesn't exist yet.

    This test guards against tasks silently depending on seed data
    that happens not to be there (the bug that broke T009).
    """
    from acme.environment.data_generator import generate_seed_data

    for task in get_tasks_by_difficulty(2):
        required_tools = set(task.expected_outcome.required_tool_calls)

        # Skip creation tasks — they create the target themselves
        if "create_opportunity" in required_tools:
            continue

        required = task.expected_outcome.required_state.get("opportunities", [])
        if not required:
            continue

        state = generate_seed_data(seed=42)
        state.apply_patch(task.initial_state_patch)

        for matcher in required:
            customer_id = matcher.get("customer_id")
            if customer_id is None:
                continue

            matching = [
                opp for opp in state.opportunities.values()
                if opp.customer_id == customer_id
            ]
            assert matching, (
                f"{task.task_id}: requires updating an opportunity for "
                f"{customer_id}, but none exists after applying the patch"
            )
