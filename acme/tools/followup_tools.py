"""Follow-up task tools for the Acme Sales benchmark.

Follow-ups have a required `created_at` field. Tools MUST pass
`state.reference_time` — never wall-clock time. This is what makes
POL-004 (post-demo follow-up) evaluation correct.
"""

from __future__ import annotations

from datetime import date

from acme.environment.models import FollowUp, FollowUpStatus
from acme.environment.state import EnvironmentState
from acme.tools.tool_types import (
    ToolResult,
    ERR_CONFLICT,
    ERR_INVALID_ARGS,
    ERR_NOT_FOUND,
)


def _parse_date(value) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def create_followup(
    state: EnvironmentState,
    customer_id: str,
    action: str,
    due_date: str | date,
    opportunity_id: str | None = None,
    notes: str = "",
) -> ToolResult:
    """Create a follow-up task for a customer.

    Args:
        customer_id: Which customer this follow-up concerns.
        action: Description (e.g., "Send proposal with updated pricing").
        due_date: ISO date string or date object.
        opportunity_id: Optional link to a specific opportunity.
        notes: Optional free-text notes.

    Returns:
        ToolResult with the created FollowUp dict.
    """
    customer = state.customers.get(customer_id)
    if customer is None:
        return ToolResult.fail(
            error=f"Customer '{customer_id}' not found",
            error_code=ERR_NOT_FOUND,
            timestamp=state.reference_time,
        )

    if opportunity_id is not None and opportunity_id not in state.opportunities:
        return ToolResult.fail(
            error=f"Opportunity '{opportunity_id}' not found",
            error_code=ERR_NOT_FOUND,
            timestamp=state.reference_time,
        )

    if not action or not action.strip():
        return ToolResult.fail(
            error="action must be a non-empty string",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )

    due = _parse_date(due_date)
    if due is None:
        return ToolResult.fail(
            error="due_date must be a date or ISO date string (YYYY-MM-DD)",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )

    fid = state.next_followup_id()
    followup = FollowUp(
        followup_id=fid,
        customer_id=customer_id,
        opportunity_id=opportunity_id,
        assigned_rep=customer.assigned_rep,
        action=action,
        due_date=due,
        status=FollowUpStatus.PENDING,
        notes=notes,
        created_at=state.reference_time,   # ← required; simulation time
    )
    state.followups[fid] = followup

    return ToolResult.ok(
        data=followup.model_dump(mode="json"),
        timestamp=state.reference_time,
    )


def get_followups(
    state: EnvironmentState,
    customer_id: str | None = None,
    status: str | None = None,
    assigned_rep: str | None = None,
    overdue_only: bool = False,
) -> ToolResult:
    """List follow-ups filtered by customer, status, rep, or overdue flag."""
    results = []
    for fu in state.followups.values():
        if customer_id and fu.customer_id != customer_id:
            continue
        if status and fu.status.value != status:
            continue
        if assigned_rep and fu.assigned_rep != assigned_rep:
            continue
        if overdue_only:
            if fu.status != FollowUpStatus.OVERDUE:
                # Also treat past-due PENDING as overdue
                if not (
                    fu.status == FollowUpStatus.PENDING
                    and fu.due_date < state.reference_time.date()
                ):
                    continue
        results.append(fu)

    results.sort(key=lambda f: (f.due_date, f.followup_id))

    return ToolResult.ok(
        data={
            "count": len(results),
            "followups": [f.model_dump(mode="json") for f in results],
        },
        timestamp=state.reference_time,
    )


def complete_followup(
    state: EnvironmentState,
    followup_id: str,
    notes: str = "",
) -> ToolResult:
    """Mark a follow-up task as completed."""
    fu = state.followups.get(followup_id)
    if fu is None:
        return ToolResult.fail(
            error=f"Follow-up '{followup_id}' not found",
            error_code=ERR_NOT_FOUND,
            timestamp=state.reference_time,
        )
    if fu.status == FollowUpStatus.COMPLETED:
        return ToolResult.fail(
            error=f"Follow-up '{followup_id}' is already completed",
            error_code=ERR_CONFLICT,
            timestamp=state.reference_time,
        )

    new_notes = fu.notes
    if notes:
        new_notes = (new_notes + f" | {notes}").strip(" |")

    updated = fu.model_copy(
        update={"status": FollowUpStatus.COMPLETED, "notes": new_notes}
    )
    state.followups[followup_id] = updated

    return ToolResult.ok(
        data={
            "followup_id": followup_id,
            "status": updated.status.value,
        },
        timestamp=state.reference_time,
    )
