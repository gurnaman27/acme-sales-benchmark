"""Opportunity-related tools for the Acme Sales benchmark."""

from __future__ import annotations

from acme.environment.models import Opportunity, OpportunityStage
from acme.environment.state import EnvironmentState
from acme.tools.tool_types import (
    ToolResult,
    ERR_NOT_FOUND,
    ERR_INVALID_ARGS,
)


# Immutable fields — cannot be changed via update_opportunity
_IMMUTABLE_FIELDS = {"opportunity_id", "customer_id", "created_at"}

# Fields an agent is allowed to mutate
_MUTABLE_FIELDS = {
    "product_ids",
    "stage",
    "expected_value",
    "probability",
    "assigned_rep",
    "last_contact",
    "next_action",
    "notes",
    "close_date",
    "discount_percent",
}


def get_opportunity(
    state: EnvironmentState,
    opportunity_id: str,
) -> ToolResult:
    """Return the full opportunity record."""
    opp = state.opportunities.get(opportunity_id)
    if opp is None:
        return ToolResult.fail(
            error=f"Opportunity '{opportunity_id}' not found",
            error_code=ERR_NOT_FOUND,
            timestamp=state.reference_time,
        )
    return ToolResult.ok(
        data=opp.model_dump(mode="json"),
        timestamp=state.reference_time,
    )


def search_opportunities(
    state: EnvironmentState,
    customer_id: str | None = None,
    stage: str | None = None,
    rep_id: str | None = None,
    min_value: float | None = None,
    limit: int = 20,
) -> ToolResult:
    """Search opportunities by customer, stage, rep, or minimum value.

    Args:
        customer_id: Filter to a specific customer.
        stage: Exact match on `stage.value` (e.g., "proposal", "negotiation").
        rep_id: Filter by assigned rep.
        min_value: Minimum expected_value (inclusive).
        limit: Maximum results to return.

    Returns:
        ToolResult with `data = {"count": int, "opportunities": [dict, ...]}`.
    """
    results = []
    for opp in state.opportunities.values():
        if customer_id and opp.customer_id != customer_id:
            continue
        if stage and opp.stage.value != stage:
            continue
        if rep_id and opp.assigned_rep != rep_id:
            continue
        if min_value is not None and opp.expected_value < min_value:
            continue
        results.append(opp)

    # Sort by expected value descending (highest value first)
    results.sort(key=lambda o: o.expected_value, reverse=True)

    return ToolResult.ok(
        data={
            "count": len(results),
            "opportunities": [o.model_dump(mode="json") for o in results[:limit]],
        },
        timestamp=state.reference_time,
    )


def update_opportunity(
    state: EnvironmentState,
    opportunity_id: str,
    updates: dict,
) -> ToolResult:
    """Update mutable fields on an opportunity.

    Immutable fields: opportunity_id, customer_id, created_at.
    Mutable fields: product_ids, stage, expected_value, probability,
                    assigned_rep, last_contact, next_action, notes, close_date.
    """
    opp = state.opportunities.get(opportunity_id)
    if opp is None:
        return ToolResult.fail(
            error=f"Opportunity '{opportunity_id}' not found",
            error_code=ERR_NOT_FOUND,
            timestamp=state.reference_time,
        )

    if not updates:
        return ToolResult.fail(
            error="No fields provided to update",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )

    bad_immutable = set(updates.keys()) & _IMMUTABLE_FIELDS
    if bad_immutable:
        return ToolResult.fail(
            error=f"Cannot update immutable fields: {sorted(bad_immutable)}",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )

    bad_unknown = set(updates.keys()) - _MUTABLE_FIELDS
    if bad_unknown:
        return ToolResult.fail(
            error=f"Unknown update fields: {sorted(bad_unknown)}",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )

    # Validate stage if provided (model_copy does not re-validate)
    # Also cast to the enum so the stored value is a proper OpportunityStage,
    # not a raw string — avoids Pydantic serializer warnings later.
    updates = dict(updates)  # copy so we don't mutate the caller's dict
    if "stage" in updates:
        valid_stages = {s.value for s in OpportunityStage}
        if updates["stage"] not in valid_stages:
            return ToolResult.fail(
                error=f"Invalid stage '{updates['stage']}'. Valid: {sorted(valid_stages)}",
                error_code=ERR_INVALID_ARGS,
                timestamp=state.reference_time,
            )
        updates["stage"] = OpportunityStage(updates["stage"])

    updated = opp.model_copy(update=updates)
    state.opportunities[opportunity_id] = updated

    return ToolResult.ok(
        data={
            "opportunity_id": opportunity_id,
            "updated_fields": sorted(updates.keys()),
        },
        timestamp=state.reference_time,
    )


def create_opportunity(
    state: EnvironmentState,
    customer_id: str,
    product_ids: list[str],
    expected_value: float | None = None,
    stage: str = "prospecting",
    next_action: str = "",
    notes: str = "",
    discount_percent: float = 0.0,
) -> ToolResult:
    """Create a new opportunity for a customer.

    If `expected_value` is omitted, it is computed as the sum of the
    selected products' base prices.

    The assigned rep is inherited from the customer record.

    Returns:
        ToolResult with the newly created opportunity dict.
    """
    customer = state.customers.get(customer_id)
    if customer is None:
        return ToolResult.fail(
            error=f"Customer '{customer_id}' not found",
            error_code=ERR_NOT_FOUND,
            timestamp=state.reference_time,
        )

    if not product_ids:
        return ToolResult.fail(
            error="product_ids must be a non-empty list",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )

    for pid in product_ids:
        if pid not in state.products:
            return ToolResult.fail(
                error=f"Product '{pid}' not found",
                error_code=ERR_NOT_FOUND,
                timestamp=state.reference_time,
            )

    valid_stages = {s.value for s in OpportunityStage}
    if stage not in valid_stages:
        return ToolResult.fail(
            error=f"Invalid stage '{stage}'. Valid: {sorted(valid_stages)}",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )

    if expected_value is None:
        expected_value = sum(
            state.products[pid].base_price for pid in product_ids
        )

    oid = state.next_opportunity_id()
    opportunity = Opportunity(
        opportunity_id=oid,
        customer_id=customer_id,
        product_ids=list(product_ids),
        stage=OpportunityStage(stage),
        expected_value=expected_value,
        probability=0.1,
        assigned_rep=customer.assigned_rep,
        created_at=state.reference_time,   # required field — simulation time
        next_action=next_action,
        notes=notes,
        discount_percent=discount_percent,
    )
    state.opportunities[oid] = opportunity

    return ToolResult.ok(
        data=opportunity.model_dump(mode="json"),
        timestamp=state.reference_time,
    )
