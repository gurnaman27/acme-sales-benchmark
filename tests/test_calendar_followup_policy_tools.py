"""Tests for calendar, followup, and policy tools."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from acme.environment.data_generator import generate_seed_data
from acme.environment.models import FollowUpStatus, MeetingStatus
from acme.tools.tool_registry import ToolRegistry
from acme.tools.tool_types import (
    ERR_NOT_FOUND,
    ERR_INVALID_ARGS,
    ERR_CONFLICT,
)

_UTC = ZoneInfo("UTC")


# ---------------------------------------------------------------------------
# get_current_time
# ---------------------------------------------------------------------------

def test_get_current_time_returns_reference():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("get_current_time")
    assert result.success
    assert result.data["utc"] == state.reference_time.isoformat()
    assert result.data["date"] == "2025-03-15"
    assert result.data["day_of_week"] == "Saturday"


# ---------------------------------------------------------------------------
# get_available_slots
# ---------------------------------------------------------------------------

def test_get_available_slots_returns_slots():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "get_available_slots",
        rep_id="REP-001",
        start_date="2025-03-17",
        end_date="2025-03-21",
    )
    assert result.success
    assert result.data["count"] > 0
    # All slots should be within the range
    for slot in result.data["slots"]:
        start_local = datetime.fromisoformat(slot["start_local"])
        assert start_local.date() >= date(2025, 3, 17)
        assert start_local.date() <= date(2025, 3, 21)


def test_get_available_slots_respects_working_hours():
    """Slots must fall within rep's working hours (in rep local tz)."""
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "get_available_slots",
        rep_id="REP-001",   # 9–18, America/Los_Angeles
        start_date="2025-03-17",
        end_date="2025-03-17",
    )
    assert result.success
    la = ZoneInfo("America/Los_Angeles")
    for slot in result.data["slots"]:
        start_utc = datetime.fromisoformat(slot["start_utc"])
        local = start_utc.astimezone(la)
        assert 9 <= local.hour < 18


def test_get_available_slots_skips_weekends():
    """March 15/16 2025 are Sat/Sun — no slots should be returned."""
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "get_available_slots",
        rep_id="REP-001",
        start_date="2025-03-15",
        end_date="2025-03-16",
    )
    assert result.success
    assert result.data["count"] == 0


def test_get_available_slots_excludes_conflicting_meetings():
    """Get existing scheduled meeting for REP-001, then check its slot is excluded."""
    state = generate_seed_data(seed=42)
    # Find a future scheduled meeting for REP-001
    existing = [
        m for m in state.meetings.values()
        if m.rep_id == "REP-001" and m.status == MeetingStatus.SCHEDULED
    ]
    if not existing:
        return  # skip if no data
    meeting = existing[0]
    meeting_date = meeting.datetime_start.astimezone(
        ZoneInfo("America/Los_Angeles")
    ).date()

    registry = ToolRegistry(state)
    result = registry.call(
        "get_available_slots",
        rep_id="REP-001",
        start_date=meeting_date.isoformat(),
        end_date=meeting_date.isoformat(),
        duration_minutes=meeting.duration_minutes,
    )
    assert result.success
    # The exact conflicting slot should not appear
    for slot in result.data["slots"]:
        slot_start = datetime.fromisoformat(slot["start_utc"])
        slot_end = datetime.fromisoformat(slot["end_utc"])
        m_start = meeting.datetime_start
        m_end = m_start + timedelta(minutes=meeting.duration_minutes)
        # No full overlap
        assert not (slot_start < m_end and slot_end > m_start)


def test_get_available_slots_unknown_rep():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "get_available_slots",
        rep_id="REP-999",
        start_date="2025-03-17",
        end_date="2025-03-17",
    )
    assert result.success is False
    assert result.error_code == ERR_NOT_FOUND


def test_get_available_slots_bad_date():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "get_available_slots",
        rep_id="REP-001",
        start_date="not-a-date",
        end_date="2025-03-17",
    )
    assert result.success is False
    assert result.error_code == ERR_INVALID_ARGS


# ---------------------------------------------------------------------------
# schedule_meeting
# ---------------------------------------------------------------------------

def test_schedule_meeting_success():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    before = len(state.meetings)

    result = registry.call(
        "schedule_meeting",
        customer_id="C001",
        rep_id="REP-001",
        start_time="2025-03-17T17:00:00+00:00",   # 10am PT
        duration_minutes=30,
        purpose="demo",
    )
    assert result.success
    assert result.data["customer_id"] == "C001"
    assert result.data["rep_id"] == "REP-001"
    assert result.data["status"] == "scheduled"
    assert result.data["purpose"] == "demo"
    assert len(state.meetings) == before + 1


def test_schedule_meeting_requires_timezone_aware_datetime():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "schedule_meeting",
        customer_id="C001",
        rep_id="REP-001",
        start_time="2025-03-17T10:00:00",   # no timezone
    )
    assert result.success is False
    assert result.error_code == ERR_INVALID_ARGS


def test_schedule_meeting_unknown_customer():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "schedule_meeting",
        customer_id="C999",
        rep_id="REP-001",
        start_time="2025-03-17T17:00:00+00:00",
    )
    assert result.success is False
    assert result.error_code == ERR_NOT_FOUND


def test_schedule_meeting_unknown_rep():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "schedule_meeting",
        customer_id="C001",
        rep_id="REP-999",
        start_time="2025-03-17T17:00:00+00:00",
    )
    assert result.success is False
    assert result.error_code == ERR_NOT_FOUND


def test_schedule_meeting_does_not_enforce_policy():
    """Double-booking a slot should succeed — policy is checked later."""
    state = generate_seed_data(seed=42)
    # Find an existing scheduled meeting and try to book the same slot
    existing = next(
        (m for m in state.meetings.values() if m.status == MeetingStatus.SCHEDULED),
        None,
    )
    if existing is None:
        return
    registry = ToolRegistry(state)
    result = registry.call(
        "schedule_meeting",
        customer_id=existing.customer_id,
        rep_id=existing.rep_id,
        start_time=existing.datetime_start.isoformat(),
        duration_minutes=existing.duration_minutes,
    )
    # Should succeed — the evaluator will flag the double-booking
    assert result.success


# ---------------------------------------------------------------------------
# cancel_meeting
# ---------------------------------------------------------------------------

def test_cancel_meeting_success():
    state = generate_seed_data(seed=42)
    mid = next(
        m.meeting_id for m in state.meetings.values()
        if m.status == MeetingStatus.SCHEDULED
    )
    registry = ToolRegistry(state)
    result = registry.call("cancel_meeting", meeting_id=mid, reason="rescheduled")
    assert result.success
    assert state.meetings[mid].status == MeetingStatus.CANCELLED
    assert "rescheduled" in state.meetings[mid].notes


def test_cancel_meeting_already_cancelled():
    state = generate_seed_data(seed=42)
    mid = next(
        m.meeting_id for m in state.meetings.values()
        if m.status == MeetingStatus.SCHEDULED
    )
    registry = ToolRegistry(state)
    registry.call("cancel_meeting", meeting_id=mid)
    result = registry.call("cancel_meeting", meeting_id=mid)
    assert result.success is False
    assert result.error_code == ERR_CONFLICT


def test_cancel_meeting_not_found():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("cancel_meeting", meeting_id="MTG-9999")
    assert result.success is False
    assert result.error_code == ERR_NOT_FOUND


# ---------------------------------------------------------------------------
# get_meetings
# ---------------------------------------------------------------------------

def test_get_meetings_by_rep():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("get_meetings", rep_id="REP-001")
    assert result.success
    for m in result.data["meetings"]:
        assert m["rep_id"] == "REP-001"


def test_get_meetings_by_customer():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    cid = next(iter(state.customers))
    result = registry.call("get_meetings", customer_id=cid)
    assert result.success
    for m in result.data["meetings"]:
        assert m["customer_id"] == cid


def test_get_meetings_sorted_by_start():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("get_meetings")
    assert result.success
    starts = [m["datetime_start"] for m in result.data["meetings"]]
    assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# create_followup
# ---------------------------------------------------------------------------

def test_create_followup_success():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    before = len(state.followups)

    result = registry.call(
        "create_followup",
        customer_id="C001",
        action="Send updated pricing",
        due_date="2025-03-20",
    )
    assert result.success
    assert result.data["customer_id"] == "C001"
    assert result.data["status"] == "pending"
    assert len(state.followups) == before + 1


def test_create_followup_uses_reference_time():
    """created_at must equal state.reference_time, not wall-clock."""
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "create_followup",
        customer_id="C001",
        action="Follow up",
        due_date="2025-03-20",
    )
    fid = result.data["followup_id"]
    assert state.followups[fid].created_at == state.reference_time


def test_create_followup_inherits_assigned_rep():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "create_followup",
        customer_id="C001",
        action="Follow up",
        due_date="2025-03-20",
    )
    assert result.data["assigned_rep"] == state.customers["C001"].assigned_rep


def test_create_followup_unknown_customer():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "create_followup",
        customer_id="C999",
        action="Follow up",
        due_date="2025-03-20",
    )
    assert result.success is False
    assert result.error_code == ERR_NOT_FOUND


def test_create_followup_unknown_opportunity():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "create_followup",
        customer_id="C001",
        action="Follow up",
        due_date="2025-03-20",
        opportunity_id="OPP-9999",
    )
    assert result.success is False
    assert result.error_code == ERR_NOT_FOUND


def test_create_followup_empty_action():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "create_followup",
        customer_id="C001",
        action="   ",
        due_date="2025-03-20",
    )
    assert result.success is False
    assert result.error_code == ERR_INVALID_ARGS


# ---------------------------------------------------------------------------
# get_followups
# ---------------------------------------------------------------------------

def test_get_followups_by_customer():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    cid = next(iter(state.followups.values())).customer_id
    result = registry.call("get_followups", customer_id=cid)
    assert result.success
    for f in result.data["followups"]:
        assert f["customer_id"] == cid


def test_get_followups_by_status():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("get_followups", status="overdue")
    assert result.success
    for f in result.data["followups"]:
        assert f["status"] == "overdue"


def test_get_followups_overdue_only():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("get_followups", overdue_only=True)
    assert result.success
    # All returned should be OVERDUE or PENDING past reference date
    for f in result.data["followups"]:
        assert f["status"] in ("overdue", "pending")


# ---------------------------------------------------------------------------
# complete_followup
# ---------------------------------------------------------------------------

def test_complete_followup_success():
    state = generate_seed_data(seed=42)
    fid = next(
        f.followup_id for f in state.followups.values()
        if f.status == FollowUpStatus.PENDING
    )
    registry = ToolRegistry(state)
    result = registry.call("complete_followup", followup_id=fid, notes="Done")
    assert result.success
    assert state.followups[fid].status == FollowUpStatus.COMPLETED
    assert "Done" in state.followups[fid].notes


def test_complete_followup_already_completed():
    state = generate_seed_data(seed=42)
    fid = next(
        f.followup_id for f in state.followups.values()
        if f.status == FollowUpStatus.COMPLETED
    )
    registry = ToolRegistry(state)
    result = registry.call("complete_followup", followup_id=fid)
    assert result.success is False
    assert result.error_code == ERR_CONFLICT


def test_complete_followup_not_found():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("complete_followup", followup_id="FU-9999")
    assert result.success is False
    assert result.error_code == ERR_NOT_FOUND


# ---------------------------------------------------------------------------
# check_policy
# ---------------------------------------------------------------------------

def test_check_policy_discount_pass():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("check_policy", action_type="discount", discount_pct=10.0)
    assert result.success
    assert result.data["passed"] is True
    assert result.data["violation_count"] == 0


def test_check_policy_discount_fail():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("check_policy", action_type="discount", discount_pct=18.0)
    assert result.success
    assert result.data["passed"] is False
    assert result.data["violation_count"] == 1


def test_check_policy_discount_with_approval():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "check_policy",
        action_type="discount",
        discount_pct=18.0,
        approver_role="manager",
    )
    assert result.success
    assert result.data["passed"] is True


def test_check_policy_scheduling():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "check_policy",
        action_type="scheduling",
        rep_id="REP-001",
        proposed_start="2025-03-17T17:00:00+00:00",
        duration_minutes=30,
    )
    assert result.success
    assert "passed" in result.data


def test_check_policy_product_eligibility():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "check_policy",
        action_type="product_eligibility",
        product_id="PROD-003",
        customer_id="C001",
    )
    assert result.success
    assert result.data["passed"] is True


def test_check_policy_unknown_action():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("check_policy", action_type="not_a_real_action")
    assert result.success is False
    assert result.error_code == ERR_INVALID_ARGS


def test_check_policy_bad_parameters():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call(
        "check_policy",
        action_type="discount",
        wrong_param=10.0,
    )
    assert result.success is False
    assert result.error_code == ERR_INVALID_ARGS


# ---------------------------------------------------------------------------
# get_policies
# ---------------------------------------------------------------------------

def test_get_policies_returns_all():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("get_policies")
    assert result.success
    assert result.data["count"] == 8


def test_get_policies_by_category():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("get_policies", category="scheduling")
    assert result.success
    for p in result.data["policies"]:
        assert p["rule_type"] == "scheduling"


def test_get_policies_sorted():
    state = generate_seed_data(seed=42)
    registry = ToolRegistry(state)
    result = registry.call("get_policies")
    ids = [p["rule_id"] for p in result.data["policies"]]
    assert ids == sorted(ids)
