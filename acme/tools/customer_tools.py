"""Customer-related tools for the Acme Sales benchmark."""

from __future__ import annotations

from acme.environment.state import EnvironmentState
from acme.tools.tool_types import (
    ToolResult,
    ERR_NOT_FOUND,
    ERR_INVALID_ARGS,
)


# Immutable fields — cannot be changed via update_customer
_IMMUTABLE_FIELDS = {"customer_id", "created_at"}


def search_customers(
    state: EnvironmentState,
    query: str | None = None,
    industry: str | None = None,
    status: str | None = None,
    priority: int | None = None,
    limit: int = 20,
) -> ToolResult:
    """Search for customers by company name substring, industry, status, or priority.

    Args:
        query: Substring to match against `company` (case-insensitive).
        industry: Exact match on `industry`.
        status: Exact match on `status.value` (e.g., "active").
        priority: Exact match on `priority.value` (1–5).
        limit: Maximum results to return.

    Returns:
        ToolResult with `data = {"count": int, "customers": [dict, ...]}`.
    """
    results = []
    for customer in state.customers.values():
        if query and query.lower() not in customer.company.lower():
            continue
        if industry and customer.industry != industry:
            continue
        if status and customer.status.value != status:
            continue
        if priority is not None and customer.priority.value != priority:
            continue
        results.append(customer)
        if len(results) >= limit:
            break

    # Sort by priority (P1 first) then by company for stable output
    results.sort(key=lambda c: (c.priority.value, c.company))

    return ToolResult.ok(
        data={
            "count": len(results),
            "customers": [c.model_dump(mode="json") for c in results],
        },
        timestamp=state.reference_time,
    )


def get_customer(state: EnvironmentState, customer_id: str) -> ToolResult:
    """Return the full customer record."""
    customer = state.customers.get(customer_id)
    if customer is None:
        return ToolResult.fail(
            error=f"Customer '{customer_id}' not found",
            error_code=ERR_NOT_FOUND,
            timestamp=state.reference_time,
        )
    return ToolResult.ok(
        data=customer.model_dump(mode="json"),
        timestamp=state.reference_time,
    )


def get_customer_history(
    state: EnvironmentState,
    customer_id: str,
    limit: int = 50,
) -> ToolResult:
    """Return the customer's interaction history, meetings, and follow-ups.

    Sorted newest-first within each category. `limit` applies per category.
    """
    customer = state.customers.get(customer_id)
    if customer is None:
        return ToolResult.fail(
            error=f"Customer '{customer_id}' not found",
            error_code=ERR_NOT_FOUND,
            timestamp=state.reference_time,
        )

    interactions = sorted(
        (i for i in state.interactions.values() if i.customer_id == customer_id),
        key=lambda x: x.datetime_occurred,
        reverse=True,
    )
    meetings = sorted(
        (m for m in state.meetings.values() if m.customer_id == customer_id),
        key=lambda x: x.datetime_start,
        reverse=True,
    )
    followups = sorted(
        (f for f in state.followups.values() if f.customer_id == customer_id),
        key=lambda x: x.due_date,
        reverse=True,
    )

    return ToolResult.ok(
        data={
            "customer_id": customer_id,
            "company": customer.company,
            "interactions": [i.model_dump(mode="json") for i in interactions[:limit]],
            "meetings": [m.model_dump(mode="json") for m in meetings[:limit]],
            "followups": [f.model_dump(mode="json") for f in followups[:limit]],
            "totals": {
                "interactions": len(interactions),
                "meetings": len(meetings),
                "followups": len(followups),
            },
        },
        timestamp=state.reference_time,
    )


def update_customer(
    state: EnvironmentState,
    customer_id: str,
    updates: dict,
) -> ToolResult:
    """Update mutable fields on a customer.

    Immutable fields: customer_id, created_at.
    """
    customer = state.customers.get(customer_id)
    if customer is None:
        return ToolResult.fail(
            error=f"Customer '{customer_id}' not found",
            error_code=ERR_NOT_FOUND,
            timestamp=state.reference_time,
        )

    bad_fields = set(updates.keys()) & _IMMUTABLE_FIELDS
    if bad_fields:
        return ToolResult.fail(
            error=f"Cannot update immutable fields: {sorted(bad_fields)}",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )

    if not updates:
        return ToolResult.fail(
            error="No fields provided to update",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )

    updated = customer.model_copy(update=updates)
    state.customers[customer_id] = updated

    return ToolResult.ok(
        data={
            "customer_id": customer_id,
            "updated_fields": sorted(updates.keys()),
        },
        timestamp=state.reference_time,
    )
