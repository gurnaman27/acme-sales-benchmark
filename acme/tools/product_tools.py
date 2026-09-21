"""Product-related tools for the Acme Sales benchmark."""

from __future__ import annotations

from acme.environment.policies import PolicyEngine
from acme.environment.state import EnvironmentState
from acme.tools.tool_types import (
    ToolResult,
    ERR_NOT_FOUND,
)


def search_products(
    state: EnvironmentState,
    query: str | None = None,
    category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    available_only: bool = False,
    limit: int = 20,
) -> ToolResult:
    """Search for products by name substring, category, or price range.

    Args:
        query: Substring to match against `name` (case-insensitive).
        category: Exact match on `category.value` (e.g., "crm", "analytics").
        min_price: Minimum base_price (inclusive).
        max_price: Maximum base_price (inclusive).
        available_only: If True, exclude products with `availability=False`.
        limit: Maximum results to return.

    Returns:
        ToolResult with `data = {"count": int, "products": [dict, ...]}`.
    """
    results = []
    for product in state.products.values():
        if query and query.lower() not in product.name.lower():
            continue
        if category and product.category.value != category:
            continue
        if min_price is not None and product.base_price < min_price:
            continue
        if max_price is not None and product.base_price > max_price:
            continue
        if available_only and not product.availability:
            continue
        results.append(product)

    # Sort by category then price for stable, predictable output
    results.sort(key=lambda p: (p.category.value, p.base_price))

    return ToolResult.ok(
        data={
            "count": len(results),
            "products": [p.model_dump(mode="json") for p in results[:limit]],
        },
        timestamp=state.reference_time,
    )


def get_product(state: EnvironmentState, product_id: str) -> ToolResult:
    """Return the full product record."""
    product = state.products.get(product_id)
    if product is None:
        return ToolResult.fail(
            error=f"Product '{product_id}' not found",
            error_code=ERR_NOT_FOUND,
            timestamp=state.reference_time,
        )
    return ToolResult.ok(
        data=product.model_dump(mode="json"),
        timestamp=state.reference_time,
    )


def check_product_eligibility(
    state: EnvironmentState,
    product_id: str,
    customer_id: str,
) -> ToolResult:
    """Check whether a customer is eligible to buy a product.

    This is a read-only query that runs the POL-003 eligibility rule.
    It does not change the environment.

    Returns:
        ToolResult with data containing:
        - eligible: bool
        - reason: str
        - product: {product_id, name, min_customer_tier}
        - customer: {customer_id, status, tier}
    """
    product = state.products.get(product_id)
    if product is None:
        return ToolResult.fail(
            error=f"Product '{product_id}' not found",
            error_code=ERR_NOT_FOUND,
            timestamp=state.reference_time,
        )

    customer = state.customers.get(customer_id)
    if customer is None:
        return ToolResult.fail(
            error=f"Customer '{customer_id}' not found",
            error_code=ERR_NOT_FOUND,
            timestamp=state.reference_time,
        )

    engine = PolicyEngine(state)
    report = engine.check_product_eligibility(product_id, customer_id)
    # check_product_eligibility always appends exactly one result
    first = report.results[0]

    return ToolResult.ok(
        data={
            "eligible": first.passed,
            "reason": first.message,
            "product": {
                "product_id": product.product_id,
                "name": product.name,
                "min_customer_tier": product.min_customer_tier.value,
                "availability": product.availability,
            },
            "customer": {
                "customer_id": customer.customer_id,
                "status": customer.status.value,
                "tier": customer.tier.value,
            },
        },
        timestamp=state.reference_time,
    )
