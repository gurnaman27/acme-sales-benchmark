"""Calendar-related tools for the Acme Sales benchmark.

All datetimes in and out of these tools are timezone-aware UTC.
`get_available_slots` computes slots in the rep's local timezone and
converts to UTC for output.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from acme.environment.models import Meeting, MeetingStatus
from acme.environment.state import EnvironmentState
from acme.tools.tool_types import (
    ToolResult,
    ERR_CONFLICT,
    ERR_INVALID_ARGS,
    ERR_NOT_FOUND,
)

_UTC = ZoneInfo("UTC")
_MAX_SLOTS = 200  # cap output size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(value) -> date | None:
    """Parse a date from string or date object. Returns None on failure."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _parse_datetime(value) -> datetime | None:
    """Parse a datetime from string or datetime object. Returns None on failure."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _overlaps_or_violates_buffer(
    slot_start: datetime,
    slot_end: datetime,
    meeting_start: datetime,
    meeting_end: datetime,
    buffer_minutes: int,
) -> bool:
    """True if the slot overlaps or is within buffer_minutes of the meeting."""
    # Overlap (POL-005)
    if slot_start < meeting_end and slot_end > meeting_start:
        return True
    # Buffer (POL-006)
    if slot_start >= meeting_end:
        gap = (slot_start - meeting_end).total_seconds() / 60
        if gap < buffer_minutes:
            return True
    elif slot_end <= meeting_start:
        gap = (meeting_start - slot_end).total_seconds() / 60
        if gap < buffer_minutes:
            return True
    return False


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def get_current_time(state: EnvironmentState) -> ToolResult:
    """Return the simulation's current time.

    Agents should use this to know what "now" is in the simulation —
    do NOT rely on real wall-clock time.
    """
    ref = state.reference_time
    return ToolResult.ok(
        data={
            "utc": ref.isoformat(),
            "date": ref.date().isoformat(),
            "day_of_week": ref.strftime("%A"),
        },
        timestamp=ref,
    )


def get_available_slots(
    state: EnvironmentState,
    rep_id: str,
    start_date: str | date,
    end_date: str | date,
    duration_minutes: int = 30,
    step_minutes: int = 30,
) -> ToolResult:
    """Find free 30-minute (or `duration_minutes`) slots for a rep.

    Respects:
    - rep working hours (in rep's local timezone)
    - no double-booking (POL-005)
    - 30-minute buffer between meetings (POL-006)
    - skips weekends

    Args:
        rep_id: The rep whose calendar to inspect.
        start_date: First day of the range (inclusive). ISO date or date object.
        end_date: Last day of the range (inclusive).
        duration_minutes: Length of each candidate slot.
        step_minutes: Spacing between candidate slot start times.

    Returns:
        ToolResult with data containing a list of slots, each with
        `start_utc`, `end_utc`, `start_local`, and `rep_timezone`.
    """
    rep = state.sales_reps.get(rep_id)
    if rep is None:
        return ToolResult.fail(
            error=f"Sales rep '{rep_id}' not found",
            error_code=ERR_NOT_FOUND,
            timestamp=state.reference_time,
        )

    sd = _parse_date(start_date)
    ed = _parse_date(end_date)
    if sd is None or ed is None:
        return ToolResult.fail(
            error="start_date and end_date must be ISO date strings (YYYY-MM-DD)",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )
    if ed < sd:
        return ToolResult.fail(
            error="end_date must be on or after start_date",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )
    if (ed - sd).days > 60:
        return ToolResult.fail(
            error="Date range too large (max 60 days)",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )
    if duration_minutes <= 0 or step_minutes <= 0:
        return ToolResult.fail(
            error="duration_minutes and step_minutes must be positive",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )

    rep_tz = ZoneInfo(rep.timezone)
    buffer_minutes = state.policy_rules["POL-006"].parameters.get("min_gap_minutes", 30)

    existing = [
        m for m in state.meetings.values()
        if m.rep_id == rep_id and m.status == MeetingStatus.SCHEDULED
    ]

    slots: list[dict] = []
    cursor_date = sd
    while cursor_date <= ed:
        # Skip weekends
        if cursor_date.weekday() >= 5:
            cursor_date += timedelta(days=1)
            continue

        # Day window in rep local timezone
        day_start_local = datetime(
            cursor_date.year, cursor_date.month, cursor_date.day,
            rep.working_hours_start, 0, tzinfo=rep_tz,
        )
        day_end_local = datetime(
            cursor_date.year, cursor_date.month, cursor_date.day,
            rep.working_hours_end, 0, tzinfo=rep_tz,
        )

        cursor = day_start_local
        while cursor + timedelta(minutes=duration_minutes) <= day_end_local:
            slot_start = cursor.astimezone(_UTC)
            slot_end = (cursor + timedelta(minutes=duration_minutes)).astimezone(_UTC)

            conflict = False
            for m in existing:
                m_start = m.datetime_start
                m_end = m_start + timedelta(minutes=m.duration_minutes)
                if _overlaps_or_violates_buffer(
                    slot_start, slot_end, m_start, m_end, buffer_minutes
                ):
                    conflict = True
                    break

            if not conflict:
                slots.append({
                    "start_utc": slot_start.isoformat(),
                    "end_utc": slot_end.isoformat(),
                    "start_local": cursor.isoformat(),
                    "rep_timezone": rep.timezone,
                })
                if len(slots) >= _MAX_SLOTS:
                    break

            cursor += timedelta(minutes=step_minutes)

        if len(slots) >= _MAX_SLOTS:
            break
        cursor_date += timedelta(days=1)

    return ToolResult.ok(
        data={
            "rep_id": rep_id,
            "timezone": rep.timezone,
            "duration_minutes": duration_minutes,
            "range": {"start": sd.isoformat(), "end": ed.isoformat()},
            "count": len(slots),
            "slots": slots,
        },
        timestamp=state.reference_time,
    )


def schedule_meeting(
    state: EnvironmentState,
    customer_id: str,
    rep_id: str,
    start_time: str | datetime,
    duration_minutes: int = 30,
    meeting_type: str = "video",
    purpose: str = "general",
    notes: str = "",
) -> ToolResult:
    """Create a scheduled meeting.

    This tool does NOT enforce policy. If the slot conflicts with
    another meeting, violates working hours, or has insufficient
    buffer, the meeting will still be created — and the evaluator
    will flag the violation afterward. This is intentional: tools
    are executors, not gatekeepers.

    Args:
        start_time: Timezone-aware datetime or ISO string with offset.
        duration_minutes: Length of the meeting.
        meeting_type: "call" | "video" | "in_person".
        purpose: "general" | "demo" | "review" | "renewal" | "onboarding".
    """
    customer = state.customers.get(customer_id)
    if customer is None:
        return ToolResult.fail(
            error=f"Customer '{customer_id}' not found",
            error_code=ERR_NOT_FOUND,
            timestamp=state.reference_time,
        )
    rep = state.sales_reps.get(rep_id)
    if rep is None:
        return ToolResult.fail(
            error=f"Sales rep '{rep_id}' not found",
            error_code=ERR_NOT_FOUND,
            timestamp=state.reference_time,
        )

    start_dt = _parse_datetime(start_time)
    if start_dt is None:
        return ToolResult.fail(
            error="start_time must be a datetime or ISO datetime string",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )
    if start_dt.tzinfo is None:
        return ToolResult.fail(
            error="start_time must include timezone info (e.g., '2025-03-17T10:00:00+00:00')",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )
    if duration_minutes <= 0:
        return ToolResult.fail(
            error="duration_minutes must be positive",
            error_code=ERR_INVALID_ARGS,
            timestamp=state.reference_time,
        )

    mid = state.next_meeting_id()
    meeting = Meeting(
        meeting_id=mid,
        customer_id=customer_id,
        rep_id=rep_id,
        datetime_start=start_dt.astimezone(_UTC),
        duration_minutes=duration_minutes,
        status=MeetingStatus.SCHEDULED,
        meeting_type=meeting_type,
        purpose=purpose,
        notes=notes,
    )
    state.meetings[mid] = meeting

    return ToolResult.ok(
        data=meeting.model_dump(mode="json"),
        timestamp=state.reference_time,
    )


def cancel_meeting(
    state: EnvironmentState,
    meeting_id: str,
    reason: str = "",
) -> ToolResult:
    """Cancel a scheduled meeting. Already-cancelled meetings are rejected."""
    meeting = state.meetings.get(meeting_id)
    if meeting is None:
        return ToolResult.fail(
            error=f"Meeting '{meeting_id}' not found",
            error_code=ERR_NOT_FOUND,
            timestamp=state.reference_time,
        )
    if meeting.status == MeetingStatus.CANCELLED:
        return ToolResult.fail(
            error=f"Meeting '{meeting_id}' is already cancelled",
            error_code=ERR_CONFLICT,
            timestamp=state.reference_time,
        )

    new_notes = meeting.notes
    if reason:
        new_notes = (new_notes + f" | Cancelled: {reason}").strip(" |")

    updated = meeting.model_copy(
        update={"status": MeetingStatus.CANCELLED, "notes": new_notes}
    )
    state.meetings[meeting_id] = updated

    return ToolResult.ok(
        data={
            "meeting_id": meeting_id,
            "status": updated.status.value,
            "reason": reason,
        },
        timestamp=state.reference_time,
    )


def get_meetings(
    state: EnvironmentState,
    rep_id: str | None = None,
    customer_id: str | None = None,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    status: str | None = None,
) -> ToolResult:
    """List meetings filtered by rep, customer, date range, or status."""
    sd = _parse_date(start_date) if start_date is not None else None
    ed = _parse_date(end_date) if end_date is not None else None

    results = []
    for m in state.meetings.values():
        if rep_id and m.rep_id != rep_id:
            continue
        if customer_id and m.customer_id != customer_id:
            continue
        if status and m.status.value != status:
            continue
        if sd is not None and m.datetime_start.date() < sd:
            continue
        if ed is not None and m.datetime_start.date() > ed:
            continue
        results.append(m)

    results.sort(key=lambda m: m.datetime_start)

    return ToolResult.ok(
        data={
            "count": len(results),
            "meetings": [m.model_dump(mode="json") for m in results],
        },
        timestamp=state.reference_time,
    )
