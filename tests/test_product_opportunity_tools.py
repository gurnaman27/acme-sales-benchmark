"""Tests for product and opportunity tools."""

from acme.environment.data_generator import generate_seed_data
from acme.environment.models import OpportunityStage
from acme.tools.tool_registry import ToolRegistry
from acme.tools.tool_types import ERR_NOT_FOUND, ERR_INVALID_ARGS


# ---------------------------------------------------------------------------
# search_products
# ---------------------------------------------------------------------------

def test_search_products_all():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("search_products", limit=100)
    assert result.success
    assert result.data["count"] == 15


def test_search_products_by_query():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("search_products", query="CRM")
    assert result.success
    assert result.data["count"] > 0
    for p in result.data["products"]:
        assert "CRM" in p["name"]


def test_search_products_by_category():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("search_products", category="analytics")
    assert result.success
    for p in result.data["products"]:
        assert p["category"] == "analytics"


def test_search_products_price_range():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("search_products", min_price=5000, max_price=15000)
    assert result.success
    for p in result.data["products"]:
        assert 5000 <= p["base_price"] <= 15000


def test_search_products_available_only_excludes_prod010():
    """PROD-010 is unavailable in the seed data — it must not appear."""
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("search_products", available_only=True, limit=100)
    product_ids = [p["product_id"] for p in result.data["products"]]
    assert "PROD-010" not in product_ids


# ---------------------------------------------------------------------------
# get_product
# ---------------------------------------------------------------------------

def test_get_product_success():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("get_product", product_id="PROD-003")
    assert result.success
    assert result.data["product_id"] == "PROD-003"
    assert result.data["min_customer_tier"] == "enterprise"


def test_get_product_not_found():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("get_product", product_id="PROD-999")
    assert result.success is False
    assert result.error_code == ERR_NOT_FOUND


# ---------------------------------------------------------------------------
# check_product_eligibility
# ---------------------------------------------------------------------------

def test_check_eligibility_enterprise_customer_enterprise_product():
    """C001 is enterprise — should be eligible for PROD-003 (enterprise tier)."""
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "check_product_eligibility",
        product_id="PROD-003",
        customer_id="C001",
    )
    assert result.success
    assert result.data["eligible"] is True
    assert result.data["customer"]["tier"] == "enterprise"


def test_check_eligibility_prospect_customer_enterprise_product():
    """C008 is a prospect — must NOT be eligible for PROD-003."""
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "check_product_eligibility",
        product_id="PROD-003",
        customer_id="C008",
    )
    assert result.success
    assert result.data["eligible"] is False
    assert "does not meet" in result.data["reason"]


def test_check_eligibility_churned_customer():
    """C012 is churned — must NOT be eligible for any product."""
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "check_product_eligibility",
        product_id="PROD-001",
        customer_id="C012",
    )
    assert result.success
    assert result.data["eligible"] is False


def test_check_eligibility_unknown_product():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "check_product_eligibility",
        product_id="PROD-999",
        customer_id="C001",
    )
    assert result.success is False
    assert result.error_code == ERR_NOT_FOUND


# ---------------------------------------------------------------------------
# get_opportunity
# ---------------------------------------------------------------------------

def test_get_opportunity_success():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    first_id = next(iter(state.opportunities))
    result = registry.call("get_opportunity", opportunity_id=first_id)
    assert result.success
    assert result.data["opportunity_id"] == first_id


def test_get_opportunity_not_found():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("get_opportunity", opportunity_id="OPP-9999")
    assert result.success is False
    assert result.error_code == ERR_NOT_FOUND


# ---------------------------------------------------------------------------
# search_opportunities
# ---------------------------------------------------------------------------

def test_search_opportunities_by_customer():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    # Find a customer that has at least one opportunity
    cid = next(iter(state.opportunities.values())).customer_id
    result = registry.call("search_opportunities", customer_id=cid)
    assert result.success
    for o in result.data["opportunities"]:
        assert o["customer_id"] == cid


def test_search_opportunities_by_stage():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("search_opportunities", stage="proposal")
    assert result.success
    for o in result.data["opportunities"]:
        assert o["stage"] == "proposal"


def test_search_opportunities_sorted_by_value():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("search_opportunities", limit=100)
    values = [o["expected_value"] for o in result.data["opportunities"]]
    assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# update_opportunity
# ---------------------------------------------------------------------------

def test_update_opportunity_stage():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    oid = next(iter(state.opportunities))
    result = registry.call(
        "update_opportunity",
        opportunity_id=oid,
        updates={"stage": "negotiation", "next_action": "send_contract"},
    )
    assert result.success
    assert state.opportunities[oid].stage == OpportunityStage.NEGOTIATION
    assert state.opportunities[oid].next_action == "send_contract"


def test_update_opportunity_rejects_invalid_stage():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    oid = next(iter(state.opportunities))
    result = registry.call(
        "update_opportunity",
        opportunity_id=oid,
        updates={"stage": "not_a_stage"},
    )
    assert result.success is False
    assert result.error_code == ERR_INVALID_ARGS


def test_update_opportunity_rejects_immutable_fields():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    oid = next(iter(state.opportunities))
    result = registry.call(
        "update_opportunity",
        opportunity_id=oid,
        updates={"customer_id": "C999"},
    )
    assert result.success is False
    assert result.error_code == ERR_INVALID_ARGS


def test_update_opportunity_rejects_unknown_fields():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    oid = next(iter(state.opportunities))
    result = registry.call(
        "update_opportunity",
        opportunity_id=oid,
        updates={"made_up_field": "x"},
    )
    assert result.success is False
    assert result.error_code == ERR_INVALID_ARGS


def test_update_opportunity_not_found():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "update_opportunity",
        opportunity_id="OPP-9999",
        updates={"stage": "proposal"},
    )
    assert result.success is False
    assert result.error_code == ERR_NOT_FOUND


# ---------------------------------------------------------------------------
# create_opportunity
# ---------------------------------------------------------------------------

def test_create_opportunity_computes_expected_value():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    before_count = len(state.opportunities)

    result = registry.call(
        "create_opportunity",
        customer_id="C001",
        product_ids=["PROD-003"],   # base_price 50000
    )
    assert result.success
    assert result.data["expected_value"] == 50000.0
    assert result.data["customer_id"] == "C001"
    assert result.data["assigned_rep"] == state.customers["C001"].assigned_rep
    assert result.data["stage"] == "prospecting"
    assert len(state.opportunities) == before_count + 1


def test_create_opportunity_uses_reference_time_not_wall_clock():
    """The created_at field must equal simulation reference_time."""
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "create_opportunity",
        customer_id="C001",
        product_ids=["PROD-001"],
    )
    assert result.success
    oid = result.data["opportunity_id"]
    created = state.opportunities[oid].created_at
    assert created == state.reference_time


def test_create_opportunity_rejects_unknown_customer():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "create_opportunity",
        customer_id="C999",
        product_ids=["PROD-001"],
    )
    assert result.success is False
    assert result.error_code == ERR_NOT_FOUND


def test_create_opportunity_rejects_empty_products():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "create_opportunity",
        customer_id="C001",
        product_ids=[],
    )
    assert result.success is False
    assert result.error_code == ERR_INVALID_ARGS


def test_create_opportunity_rejects_unknown_product():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "create_opportunity",
        customer_id="C001",
        product_ids=["PROD-999"],
    )
    assert result.success is False
    assert result.error_code == ERR_NOT_FOUND


def test_create_opportunity_rejects_invalid_stage():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "create_opportunity",
        customer_id="C001",
        product_ids=["PROD-001"],
        stage="not_a_stage",
    )
    assert result.success is False
    assert result.error_code == ERR_INVALID_ARGS
