"""Tests for the tools layer: registry, dispatch, logging, and customer tools."""

from acme.environment.data_generator import generate_seed_data
from acme.tools.tool_registry import ToolRegistry
from acme.tools.tool_types import (
    ERR_NOT_FOUND,
    ERR_INVALID_ARGS,
    ERR_UNKNOWN_TOOL,
    ERR_TOOL_NOT_ALLOWED,
)


# ---------------------------------------------------------------------------
# Registry behavior
# ---------------------------------------------------------------------------

def test_registry_lists_available_tools():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    tools = registry.get_available_tools()
    assert "get_customer" in tools
    assert "search_customers" in tools
    assert "get_customer_history" in tools
    assert "update_customer" in tools


def test_registry_unknown_tool_returns_error():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("this_tool_does_not_exist", foo="bar")
    assert result.success is False
    assert result.error_code == ERR_UNKNOWN_TOOL
    # Should still be logged
    assert len(registry.log) == 1
    assert registry.log[0].tool == "this_tool_does_not_exist"


def test_registry_access_control():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state, allowed_tools=["get_customer"])
    # Allowed tool works
    r1 = registry.call("get_customer", customer_id="C001")
    assert r1.success
    # Disallowed tool blocked
    r2 = registry.call("search_customers", query="Tech")
    assert r2.success is False
    assert r2.error_code == ERR_TOOL_NOT_ALLOWED


def test_registry_logs_every_call():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    registry.call("get_customer", customer_id="C001")
    registry.call("get_customer", customer_id="C002")
    registry.call("search_customers", query="Tech")

    assert len(registry.log) == 3
    assert registry.log[0].step == 1
    assert registry.log[0].tool == "get_customer"
    assert registry.log[1].step == 2
    assert registry.log[2].step == 3
    assert registry.log[2].tool == "search_customers"
    # Log entries use simulation time, not wall-clock
    for entry in registry.log:
        assert entry.timestamp == state.reference_time


def test_registry_handles_tool_exception():
    """A tool with bad args should produce a TOOL_EXCEPTION, not crash."""
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    # get_customer expects `customer_id` — passing the wrong kwarg raises TypeError
    result = registry.call("get_customer", wrong_arg="C001")
    assert result.success is False
    assert result.error_code == "TOOL_EXCEPTION"


# ---------------------------------------------------------------------------
# search_customers
# ---------------------------------------------------------------------------

def test_search_customers_by_query():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("search_customers", query="TechNova")
    assert result.success
    assert result.data["count"] == 1
    assert result.data["customers"][0]["company"] == "TechNova Inc."


def test_search_customers_by_industry():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("search_customers", industry="Healthcare")
    assert result.success
    assert result.data["count"] > 0
    for c in result.data["customers"]:
        assert c["industry"] == "Healthcare"


def test_search_customers_by_priority():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("search_customers", priority=1)
    assert result.success
    for c in result.data["customers"]:
        assert c["priority"] == 1  # P1


def test_search_customers_sorted_by_priority():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("search_customers", limit=20)
    priorities = [c["priority"] for c in result.data["customers"]]
    assert priorities == sorted(priorities), "Results should be sorted by priority"


# ---------------------------------------------------------------------------
# get_customer
# ---------------------------------------------------------------------------

def test_get_customer_success():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("get_customer", customer_id="C001")
    assert result.success
    assert result.data["customer_id"] == "C001"
    assert result.data["company"] == "TechNova Inc."
    # Datetime fields should be ISO strings (mode="json")
    assert isinstance(result.data["created_at"], str)
    assert "T" in result.data["created_at"]


def test_get_customer_not_found():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("get_customer", customer_id="C999")
    assert result.success is False
    assert result.error_code == ERR_NOT_FOUND
    assert "C999" in result.error


# ---------------------------------------------------------------------------
# get_customer_history
# ---------------------------------------------------------------------------

def test_get_customer_history_returns_all_categories():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    # Find a customer with interactions
    customer_with_ints = None
    for cid in state.customers:
        if any(i.customer_id == cid for i in state.interactions.values()):
            customer_with_ints = cid
            break
    assert customer_with_ints is not None, "seed data should have interactions"

    result = registry.call("get_customer_history", customer_id=customer_with_ints)
    assert result.success
    assert "interactions" in result.data
    assert "meetings" in result.data
    assert "followups" in result.data
    assert "totals" in result.data
    assert result.data["totals"]["interactions"] >= 1


def test_get_customer_history_sorted_newest_first():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    # Find a customer with at least 2 interactions
    cid = None
    for c in state.customers:
        count = sum(1 for i in state.interactions.values() if i.customer_id == c)
        if count >= 2:
            cid = c
            break
    if cid is None:
        return  # skip if seed data has no such customer

    result = registry.call("get_customer_history", customer_id=cid)
    timestamps = [i["datetime_occurred"] for i in result.data["interactions"]]
    assert timestamps == sorted(timestamps, reverse=True)


def test_get_customer_history_not_found():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("get_customer_history", customer_id="C999")
    assert result.success is False
    assert result.error_code == ERR_NOT_FOUND


# ---------------------------------------------------------------------------
# update_customer
# ---------------------------------------------------------------------------

def test_update_customer_success():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "update_customer",
        customer_id="C001",
        updates={"notes": "Updated by test"},
    )
    assert result.success
    assert "notes" in result.data["updated_fields"]
    # Verify state changed
    assert state.customers["C001"].notes == "Updated by test"


def test_update_customer_rejects_immutable_fields():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "update_customer",
        customer_id="C001",
        updates={"customer_id": "C999"},
    )
    assert result.success is False
    assert result.error_code == ERR_INVALID_ARGS
    assert "immutable" in result.error.lower()


def test_update_customer_rejects_empty_updates():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("update_customer", customer_id="C001", updates={})
    assert result.success is False
    assert result.error_code == ERR_INVALID_ARGS


def test_update_customer_not_found():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "update_customer",
        customer_id="C999",
        updates={"notes": "x"},
    )
    assert result.success is False
    assert result.error_code == ERR_NOT_FOUND
