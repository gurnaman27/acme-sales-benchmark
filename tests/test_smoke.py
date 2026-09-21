from datetime import datetime
from datetime import date
from zoneinfo import ZoneInfo
from acme.environment.data_generator import generate_seed_data
from acme.environment.policies import PolicyEngine
from acme.environment.state import EnvironmentState, diff_states


def test_seed_and_counters():
    state = generate_seed_data(seed=42)
    assert len(state.sales_reps) == 5
    assert len(state.customers) == 20
    assert len(state.products) == 15
    assert len(state.opportunities) == 25
    assert len(state.interactions) == 30
    assert len(state.meetings) == 15
    assert len(state.followups) == 10
    assert len(state.policy_rules) == 8

    # Counter persistence test across snapshots
    snap = state.snapshot()
    m1 = state.next_meeting_id()
    state2 = EnvironmentState.from_snapshot(snap)
    m2 = state2.next_meeting_id()
    assert m1 == m2, f"Counter mismatch: {m1} != {m2}"


def test_snapshot_and_diff():
    state = generate_seed_data(seed=42)
    snapshot = state.snapshot()
    state.customers["C001"].notes = "Updated by smoke test"
    diff = diff_states(snapshot, state)
    assert "customers" in diff.modified
    assert "C001" in diff.modified["customers"]

    restored = EnvironmentState.from_snapshot(snapshot)
    assert restored.customers["C001"].notes != "Updated by smoke test"


def test_timezone_conversion():
    state = generate_seed_data(seed=42)
    engine = PolicyEngine(state)

    cust = state.customers["C001"]
    rep = state.sales_reps[cust.assigned_rep]

    rep_tz = ZoneInfo(rep.timezone)
    cust_tz = ZoneInfo(cust.timezone)

    aware_10am_rep = datetime(2025, 3, 17, 10, 0, tzinfo=rep_tz).astimezone(ZoneInfo("UTC"))
    expected_local = aware_10am_rep.astimezone(cust_tz)

    report = engine.check_contact_hours(aware_10am_rep, "C001")
    assert len(report.results) == 1

    expected_hhmm = expected_local.strftime("%H:%M")
    assert expected_hhmm in report.results[0].message, (
        f"Expected {expected_hhmm} in message, got: {report.results[0].message}"
    )

    expected_passed = 9 <= expected_local.hour < 18
    assert report.results[0].passed == expected_passed, (
        f"Expected passed={expected_passed} for local hour {expected_local.hour}, got {report.results[0].passed}"
    )


def test_all_datetimes_are_timezone_aware():
    state = generate_seed_data(seed=42)
    for customer in state.customers.values():
        if customer.created_at is not None:
            assert customer.created_at.tzinfo is not None, f"{customer.customer_id}.created_at is naive"
        if customer.last_contacted is not None:
            assert customer.last_contacted.tzinfo is not None, f"{customer.customer_id}.last_contacted is naive"
    for opp in state.opportunities.values():
        assert opp.created_at.tzinfo is not None, f"{opp.opportunity_id}.created_at is naive"
        if opp.last_contact is not None:
            assert opp.last_contact.tzinfo is not None, f"{opp.opportunity_id}.last_contact is naive"
    for meeting in state.meetings.values():
        assert meeting.datetime_start.tzinfo is not None, f"{meeting.meeting_id}.datetime_start is naive"
    for interaction in state.interactions.values():
        assert interaction.datetime_occurred.tzinfo is not None, f"{interaction.interaction_id}.datetime_occurred is naive"
    for fu in state.followups.values():
        assert fu.created_at.tzinfo is not None, f"{fu.followup_id}.created_at is naive"


def test_policy_engine():
    state = generate_seed_data(seed=42)
    engine = PolicyEngine(state)

    # POL-001: Discount check
    r1 = engine.check_discount(10.0)
    assert r1.passed, "10% discount should pass"
    r2 = engine.check_discount(18.0)
    assert not r2.passed, "18% discount without approval should fail"
    r3 = engine.check_discount(25.0)
    assert not r3.passed, "25% discount should exceed absolute max"
    r3_approved = engine.check_discount(18.0, approver_role="manager")
    assert r3_approved.passed, "18% discount with manager approval should pass"

    # POL-003: Product eligibility
    r4 = engine.check_product_eligibility("PROD-003", "C001")  # Enterprise product for Enterprise customer
    assert r4.passed, f"C001 (enterprise) should be eligible for PROD-003: {r4.results[0].message}"
    r5 = engine.check_product_eligibility("PROD-001", "C001")  # Basic product
    assert r5.passed, f"C001 should be eligible for PROD-001: {r5.results[0].message}"

    # POL-005 / POL-006: Scheduling
    ref = datetime(2025, 3, 17, 10, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles")).astimezone(ZoneInfo("UTC"))
    r6 = engine.check_scheduling("REP-001", ref, 30)
    assert r6.passed, f"Monday 10am should be valid for REP-001: {r6.results}"

    # POL-004: Post-demo follow-up check
    r7 = engine.check_post_demo_followup()
    assert len(r7.results) > 0, "POL-004 should check demo interactions"

    # POL-008: Quarterly review check
    r8 = engine.check_quarterly_review()
    assert len(r8.results) > 0, "POL-008 should check large accounts"

    # Full audit
    audit = engine.audit_full_state()
    assert len(audit.results) > 0


def test_state_patching():
    state2 = generate_seed_data(seed=42)
    state2.apply_patch({
        "customers": {
            "C001": {"notes": "Patched!"},
        }
    })
    assert state2.customers["C001"].notes == "Patched!"

def test_created_at_is_required():
    """Verify entities created without explicit created_at raise ValidationError.

    This prevents tools from accidentally using wall-clock time
    (datetime.now()) instead of simulation time (state.reference_time).
    """
    from pydantic import ValidationError
    from acme.environment.models import Customer, Opportunity, FollowUp

    # Customer without created_at should fail
    try:
        Customer(
            customer_id="C999",
            company="Test Co",
            industry="Tech",
            contact_name="Test",
            contact_email="t@t.com",
            assigned_rep="REP-001",
        )
        raise AssertionError("Customer created without created_at should fail")
    except ValidationError:
        pass  # expected

    # Opportunity without created_at should fail
    try:
        Opportunity(
            opportunity_id="OPP-9999",
            customer_id="C001",
            assigned_rep="REP-001",
        )
        raise AssertionError("Opportunity created without created_at should fail")
    except ValidationError:
        pass  # expected

    # FollowUp without created_at should fail
    try:
        FollowUp(
            followup_id="FU-9999",
            customer_id="C001",
            assigned_rep="REP-001",
            action="Test",
            due_date=date(2025, 3, 20),
        )
        raise AssertionError("FollowUp created without created_at should fail")
    except ValidationError:
        pass  # expected


def main():
    print("=" * 60)
    print("ACME SALES BENCHMARK — Smoke Test")
    print("=" * 60)

    test_seed_and_counters()
    print("\n✅ Seed data and counter persistence verified")

    test_snapshot_and_diff()
    print("\n✅ Snapshot & diff working")

    test_timezone_conversion()
    print("\n✅ Timezone conversion verified")

    test_created_at_is_required()          # ← add this
    print("\n✅ created_at is required on entities")

    test_policy_engine()
    print("\n✅ Policy engine checks verified")

    test_state_patching()
    print("\n✅ State patching works")

    print(f"\n{'=' * 60}")
    print("ALL SMOKE TESTS PASSED ✅")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

