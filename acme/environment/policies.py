"""Policy engine for validating business rules.

The PolicyEngine checks proposed actions against Acme's 8 business
policies. It is used both by agent tools (to provide policy-check
capabilities) and by the evaluator (to detect policy violations
after a task run).
"""

from __future__ import annotations

from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from acme.environment.models import (
    CustomerStatus,
    CustomerTier,
    InteractionType,
    MeetingStatus,
)
from acme.environment.state import EnvironmentState


@dataclass
class PolicyCheckResult:
    """Result of a single policy check."""
    rule_id: str
    rule_name: str
    passed: bool
    message: str
    severity: str = "error"  # error | warning


@dataclass
class PolicyReport:
    """Aggregate result of all policy checks for an action."""
    results: list[PolicyCheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity == "error")

    @property
    def violations(self) -> list[PolicyCheckResult]:
        return [r for r in self.results if not r.passed]

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "violation_count": self.violation_count,
            "results": [
                {
                    "rule_id": r.rule_id,
                    "rule_name": r.rule_name,
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity,
                }
                for r in self.results
            ],
        }


class PolicyEngine:
    """Validates actions against Acme's business policy rules."""

    def __init__(self, state: EnvironmentState):
        self.state = state

    # ------------------------------------------------------------------
    # Public API — high-level checks
    # ------------------------------------------------------------------

    def check_discount(
        self, discount_pct: float, *, approver_role: Optional[str] = None
    ) -> PolicyReport:
        """Check POL-001: discount limits."""
        report = PolicyReport()
        params = self.state.policy_rules["POL-001"].parameters
        max_no_approval = params["max_without_approval"]
        absolute_max = params["absolute_max"]
        expected_role = params.get("approval_role", "manager")

        if discount_pct > absolute_max:
            report.results.append(PolicyCheckResult(
                rule_id="POL-001",
                rule_name="Maximum Discount Limit",
                passed=False,
                message=f"Discount {discount_pct}% exceeds absolute maximum of {absolute_max}%.",
            ))
        elif discount_pct > max_no_approval:
            if approver_role and (
                approver_role == expected_role
                or (isinstance(expected_role, list) and approver_role in expected_role)
            ):
                report.results.append(PolicyCheckResult(
                    rule_id="POL-001",
                    rule_name="Maximum Discount Limit",
                    passed=True,
                    message=f"Discount {discount_pct}% approved by {approver_role}.",
                ))
            else:
                report.results.append(PolicyCheckResult(
                    rule_id="POL-001",
                    rule_name="Maximum Discount Limit",
                    passed=False,
                    message=f"Discount {discount_pct}% requires manager approval (>{max_no_approval}%).",
                ))
        else:
            report.results.append(PolicyCheckResult(
                rule_id="POL-001",
                rule_name="Maximum Discount Limit",
                passed=True,
                message=f"Discount {discount_pct}% is within allowed limit.",
            ))
        return report

    def check_contact_hours(
        self, contact_time: datetime, customer_id: str
    ) -> PolicyReport:
        """Check POL-002: business hours contact restriction in customer local timezone."""
        report = PolicyReport()
        params = self.state.policy_rules["POL-002"].parameters
        start_h = params["start_hour"]
        end_h = params["end_hour"]

        customer = self.state.customers.get(customer_id)
        if not customer:
            report.results.append(PolicyCheckResult(
                rule_id="POL-002",
                rule_name="Business Hours Contact",
                passed=False,
                message=f"Customer {customer_id} not found.",
            ))
            return report

        # Convert contact_time to customer's timezone
        try:
            tz = ZoneInfo(customer.timezone)
            if contact_time.tzinfo is None:
                local_dt = contact_time.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
            else:
                local_dt = contact_time.astimezone(tz)
        except Exception as e:
            report.results.append(PolicyCheckResult(
                rule_id="POL-002",
                rule_name="Business Hours Contact",
                passed=False,
                message=f"Could not resolve timezone '{customer.timezone}' for customer {customer_id}: {e}",
            ))
            return report

        hour = local_dt.hour

        if hour < start_h or hour >= end_h:
            report.results.append(PolicyCheckResult(
                rule_id="POL-002",
                rule_name="Business Hours Contact",
                passed=False,
                message=(
                    f"Contact at {local_dt.strftime('%H:%M')} ({customer.timezone}) is outside "
                    f"business hours ({start_h}:00-{end_h}:00)."
                ),
            ))
        else:
            report.results.append(PolicyCheckResult(
                rule_id="POL-002",
                rule_name="Business Hours Contact",
                passed=True,
                message=f"Contact at {local_dt.strftime('%H:%M')} ({customer.timezone}) is within business hours.",
            ))
        return report

    def check_product_eligibility(
        self, product_id: str, customer_id: str
    ) -> PolicyReport:
        """Check POL-003: product tier eligibility."""
        report = PolicyReport()

        product = self.state.products.get(product_id)
        customer = self.state.customers.get(customer_id)

        if not product:
            report.results.append(PolicyCheckResult(
                rule_id="POL-003",
                rule_name="Enterprise Product Eligibility",
                passed=False,
                message=f"Product {product_id} not found.",
            ))
            return report

        if not customer:
            report.results.append(PolicyCheckResult(
                rule_id="POL-003",
                rule_name="Enterprise Product Eligibility",
                passed=False,
                message=f"Customer {customer_id} not found.",
            ))
            return report

        if customer.status in (CustomerStatus.CHURNED, CustomerStatus.SUSPENDED):
            report.results.append(PolicyCheckResult(
                rule_id="POL-003",
                rule_name="Enterprise Product Eligibility",
                passed=False,
                message=f"Customer {customer_id} is {customer.status.value} and ineligible for products.",
            ))
            return report

        tier_order = {
            CustomerTier.PROSPECT: 0,
            CustomerTier.ACTIVE: 1,
            CustomerTier.ENTERPRISE: 2,
        }

        customer_tier_val = tier_order.get(customer.tier, 0)
        required_tier_val = tier_order.get(product.min_customer_tier, 0)

        if customer_tier_val < required_tier_val:
            report.results.append(PolicyCheckResult(
                rule_id="POL-003",
                rule_name="Enterprise Product Eligibility",
                passed=False,
                message=(
                    f"Customer {customer_id} ({customer.status.value}) does not meet "
                    f"minimum tier ({product.min_customer_tier.value}) for {product.name}."
                ),
            ))
        else:
            report.results.append(PolicyCheckResult(
                rule_id="POL-003",
                rule_name="Enterprise Product Eligibility",
                passed=True,
                message=f"Customer {customer_id} is eligible for {product.name}.",
            ))
        return report

    def check_post_demo_followup(self) -> PolicyReport:
        """Check POL-004: A follow-up task must be created within 48 hours after a product demo interaction."""
        report = PolicyReport()
        params = self.state.policy_rules["POL-004"].parameters
        max_hours = params.get("max_hours", 48)

        demo_interactions = [
            i for i in self.state.interactions.values()
            if i.interaction_type == InteractionType.DEMO
        ]

        for demo in demo_interactions:
            # Check if a follow-up exists for this customer within max_hours of demo
            has_followup = any(
                fu.customer_id == demo.customer_id
                and fu.created_at >= demo.datetime_occurred
                and (fu.created_at - demo.datetime_occurred).total_seconds() <= max_hours * 3600
                for fu in self.state.followups.values()
            )
            if not has_followup:
                report.results.append(PolicyCheckResult(
                    rule_id="POL-004",
                    rule_name="Post-Demo Follow-Up",
                    passed=False,
                    message=(
                        f"Product demo {demo.interaction_id} for customer {demo.customer_id} "
                        f"on {demo.datetime_occurred.strftime('%Y-%m-%d')} has no follow-up task within {max_hours} hours."
                    ),
                ))
            else:
                report.results.append(PolicyCheckResult(
                    rule_id="POL-004",
                    rule_name="Post-Demo Follow-Up",
                    passed=True,
                    message=f"Demo {demo.interaction_id} has valid follow-up task.",
                ))

        if not demo_interactions:
            report.results.append(PolicyCheckResult(
                rule_id="POL-004",
                rule_name="Post-Demo Follow-Up",
                passed=True,
                message="No demo interactions found.",
            ))

        return report

    def check_scheduling(
        self,
        rep_id: str,
        proposed_start: datetime,
        duration_minutes: int = 30,
        exclude_meeting_id: Optional[str] = None,
    ) -> PolicyReport:
        """Check POL-005 (no double-booking) and POL-006 (buffer time)."""
        report = PolicyReport()
        proposed_end = proposed_start + timedelta(minutes=duration_minutes)

        buffer_minutes = self.state.policy_rules["POL-006"].parameters.get("min_gap_minutes", 30)
        existing = [
            m for m in self.state.meetings.values()
            if m.rep_id == rep_id
            and m.status == MeetingStatus.SCHEDULED
            and m.datetime_start.date() == proposed_start.date()
            and (exclude_meeting_id is None or m.meeting_id != exclude_meeting_id)
        ]

        for meeting in existing:
            m_start = meeting.datetime_start
            m_end = m_start + timedelta(minutes=meeting.duration_minutes)

            # POL-005: overlap check
            if proposed_start < m_end and proposed_end > m_start:
                report.results.append(PolicyCheckResult(
                    rule_id="POL-005",
                    rule_name="No Double-Booking",
                    passed=False,
                    message=(
                        f"Proposed meeting overlaps with {meeting.meeting_id} "
                        f"({m_start.strftime('%H:%M')}-{m_end.strftime('%H:%M')})."
                    ),
                ))

            # POL-006: buffer check (only if no overlap)
            elif proposed_start < m_end + timedelta(minutes=buffer_minutes) and proposed_end > m_start - timedelta(minutes=buffer_minutes):
                if proposed_start >= m_end and (proposed_start - m_end).total_seconds() / 60 < buffer_minutes:
                    report.results.append(PolicyCheckResult(
                        rule_id="POL-006",
                        rule_name="Meeting Buffer Time",
                        passed=False,
                        message=(
                            f"Insufficient buffer before meeting. Only "
                            f"{int((proposed_start - m_end).total_seconds() / 60)} minutes "
                            f"gap after {meeting.meeting_id} (requires {buffer_minutes} min)."
                        ),
                        severity="warning",
                    ))
                elif proposed_end <= m_start and (m_start - proposed_end).total_seconds() / 60 < buffer_minutes:
                    report.results.append(PolicyCheckResult(
                        rule_id="POL-006",
                        rule_name="Meeting Buffer Time",
                        passed=False,
                        message=(
                            f"Insufficient buffer after meeting. Only "
                            f"{int((m_start - proposed_end).total_seconds() / 60)} minutes "
                            f"gap before {meeting.meeting_id} (requires {buffer_minutes} min)."
                        ),
                        severity="warning",
                    ))

        if not any(r.rule_id in ("POL-005", "POL-006") for r in report.results):
            report.results.append(PolicyCheckResult(
                rule_id="POL-005",
                rule_name="No Double-Booking",
                passed=True,
                message="No scheduling conflicts found.",
            ))

        return report

    def check_rep_assignment(
        self, customer_id: str, rep_id: str
    ) -> PolicyReport:
        """Check POL-007: high-priority customer rep assignment."""
        report = PolicyReport()
        customer = self.state.customers.get(customer_id)
        rep = self.state.sales_reps.get(rep_id)

        if not customer or not rep:
            report.results.append(PolicyCheckResult(
                rule_id="POL-007",
                rule_name="High-Priority Rep Assignment",
                passed=False,
                message="Customer or rep not found.",
            ))
            return report

        params = self.state.policy_rules["POL-007"].parameters
        threshold = params["priority_threshold"]
        required_roles = params["required_roles"]

        if customer.priority.value <= threshold and rep.role not in required_roles:
            report.results.append(PolicyCheckResult(
                rule_id="POL-007",
                rule_name="High-Priority Rep Assignment",
                passed=False,
                message=(
                    f"Customer {customer_id} is priority P{customer.priority.value} "
                    f"but rep {rep_id} has role '{rep.role}'. "
                    f"Requires one of: {required_roles}."
                ),
            ))
        else:
            report.results.append(PolicyCheckResult(
                rule_id="POL-007",
                rule_name="High-Priority Rep Assignment",
                passed=True,
                message="Rep assignment is valid for customer priority.",
            ))
        return report

    def check_quarterly_review(self) -> PolicyReport:
        """Check POL-008: Accounts with annual revenue above $100k must have a review meeting scheduled."""
        report = PolicyReport()
        params = self.state.policy_rules["POL-008"].parameters
        revenue_threshold = params.get("revenue_threshold", 100000)
        review_interval_days = params.get("review_interval_days", 90)

        large_accounts = [
            c for c in self.state.customers.values()
            if c.annual_revenue >= revenue_threshold and c.status == CustomerStatus.ACTIVE
        ]

        ref_time = self.state.reference_time

        for customer in large_accounts:
            has_review = any(
                m.customer_id == customer.customer_id
                and m.purpose == "review"
                and abs((m.datetime_start - ref_time).days) <= review_interval_days
                for m in self.state.meetings.values()
            )
            if not has_review:
                report.results.append(PolicyCheckResult(
                    rule_id="POL-008",
                    rule_name="Large Account Quarterly Review",
                    passed=False,
                    message=(
                        f"Large account {customer.customer_id} ({customer.company}, "
                        f"${customer.annual_revenue:,.0f}) has no quarterly review meeting scheduled "
                        f"within {review_interval_days} days of simulation reference time."
                    ),
                    severity="warning",
                ))
            else:
                report.results.append(PolicyCheckResult(
                    rule_id="POL-008",
                    rule_name="Large Account Quarterly Review",
                    passed=True,
                    message=f"Large account {customer.customer_id} has quarterly review meeting scheduled.",
                ))

        if not large_accounts:
            report.results.append(PolicyCheckResult(
                rule_id="POL-008",
                rule_name="Large Account Quarterly Review",
                passed=True,
                message="No large accounts found requiring quarterly review.",
            ))

        return report

    # ------------------------------------------------------------------
    # Full audit — run all applicable checks on final state
    # ------------------------------------------------------------------

    def audit_full_state(self) -> PolicyReport:
        """Run all policy checks against the current environment state."""
        report = PolicyReport()

        # 1. POL-002 & POL-005 & POL-006: Scheduled Meetings
        for meeting in self.state.meetings.values():
            if meeting.status != MeetingStatus.SCHEDULED:
                continue

            hours_report = self.check_contact_hours(
                meeting.datetime_start, meeting.customer_id
            )
            report.results.extend(hours_report.results)

            sched_report = self.check_scheduling(
                meeting.rep_id,
                meeting.datetime_start,
                meeting.duration_minutes,
                exclude_meeting_id=meeting.meeting_id,
            )
            for r in sched_report.results:
                if not r.passed:
                    report.results.append(r)

        # 2. POL-007: High-priority customer rep assignment
        for customer in self.state.customers.values():
            if customer.priority.value <= 2:
                rep_report = self.check_rep_assignment(
                    customer.customer_id, customer.assigned_rep
                )
                report.results.extend(rep_report.results)

        # 3. POL-003: Product eligibility for all opportunities
        for opp in self.state.opportunities.values():
            for pid in opp.product_ids:
                elig_report = self.check_product_eligibility(pid, opp.customer_id)
                for r in elig_report.results:
                    if not r.passed:
                        report.results.append(r)

        # 4. POL-004: Post-demo follow-up check
        demo_report = self.check_post_demo_followup()
        report.results.extend(demo_report.results)

        # 5. POL-008: Large account quarterly review check
        review_report = self.check_quarterly_review()
        report.results.extend(review_report.results)

        return report
