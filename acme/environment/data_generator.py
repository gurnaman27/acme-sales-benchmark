"""Seed data generator for the Acme Sales simulated environment.

Produces a realistic, deterministic dataset of customers, products,
opportunities, meetings, interactions, follow-ups, sales reps, and
policy rules. Uses a fixed random seed for reproducibility.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo
import random
from datetime import datetime, date, timedelta

from acme.environment.models import (
    Customer,
    CustomerStatus,
    CustomerTier,
    FollowUp,
    FollowUpStatus,
    Interaction,
    InteractionType,
    Meeting,
    MeetingStatus,
    Opportunity,
    OpportunityStage,
    PolicyRule,
    PolicyRuleType,
    Priority,
    Product,
    ProductCategory,
    SalesRep,
)
from acme.environment.state import EnvironmentState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INDUSTRIES = [
    "Technology", "Healthcare", "Finance", "Manufacturing", "Retail",
    "Education", "Energy", "Media", "Telecommunications", "Real Estate",
    "Automotive", "Aerospace", "Hospitality", "Logistics", "Agriculture",
]

_COMPANIES = [
    ("TechNova Inc.", "Technology"),
    ("MediCare Solutions", "Healthcare"),
    ("FinanceFirst Corp.", "Finance"),
    ("BuildWell Manufacturing", "Manufacturing"),
    ("ShopSmart Retail", "Retail"),
    ("EduLearn Academy", "Education"),
    ("GreenPower Energy", "Energy"),
    ("BrightMedia Group", "Media"),
    ("ConnectTel Systems", "Telecommunications"),
    ("UrbanSpace Properties", "Real Estate"),
    ("AutoDrive Motors", "Automotive"),
    ("SkyHigh Aerospace", "Aerospace"),
    ("TravelLux Hotels", "Hospitality"),
    ("SwiftShip Logistics", "Logistics"),
    ("AgriGrow Farms", "Agriculture"),
    ("DataStream Analytics", "Technology"),
    ("HealthBridge Clinics", "Healthcare"),
    ("WealthGuard Advisors", "Finance"),
    ("PrecisionParts Ltd.", "Manufacturing"),
    ("FreshMart Groceries", "Retail"),
]

_CONTACT_NAMES = [
    "Alice Johnson", "Bob Chen", "Carol Martinez", "David Kim",
    "Emma Williams", "Frank O'Brien", "Grace Lee", "Henry Patel",
    "Irene Nakamura", "James Wilson", "Karen Lopez", "Leo Schmidt",
    "Maria Gonzalez", "Nathan Brooks", "Olivia Taylor", "Peter Muller",
    "Quinn Davis", "Rachel Adams", "Samuel Rivera", "Tina Park",
]

_TIMEZONES = [
    "America/New_York", "America/Chicago", "America/Denver",
    "America/Los_Angeles", "Europe/London",
]

# Reference date: all generated dates are relative to this (in UTC)
_REFERENCE_DATE = datetime(2025, 3, 15, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
_REFERENCE_DATE_D = date(2025, 3, 15)


def generate_seed_data(seed: int = 42) -> EnvironmentState:
    """Generate the complete seed dataset for the Acme environment.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        A fully populated EnvironmentState.
    """
    rng = random.Random(seed)
    state = EnvironmentState(reference_time=_REFERENCE_DATE)

    # --- Sales Reps ---
    _generate_sales_reps(state)

    # --- Products ---
    _generate_products(state)

    # --- Policy Rules ---
    _generate_policy_rules(state)

    # --- Customers ---
    rep_ids = list(state.sales_reps.keys())
    _generate_customers(state, rng, rep_ids)

    # --- Opportunities ---
    _generate_opportunities(state, rng)

    # --- Interactions ---
    _generate_interactions(state, rng)

    # --- Meetings ---
    _generate_meetings(state, rng)

    # --- Follow-ups ---
    _generate_followups(state, rng)

    # Set the ID counters past the generated data
    state._next_meeting_id = len(state.meetings) + 1
    state._next_interaction_id = len(state.interactions) + 1
    state._next_followup_id = len(state.followups) + 1
    state._next_opportunity_id = len(state.opportunities) + 1

    return state


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _generate_sales_reps(state: EnvironmentState) -> None:
    reps = [
        SalesRep(
            rep_id="REP-001", name="Sarah Mitchell", email="sarah.mitchell@acme.com",
            role="senior_ae", region="US-West", timezone="America/Los_Angeles",
            working_hours_start=9, working_hours_end=18,
        ),
        SalesRep(
            rep_id="REP-002", name="Michael Chen", email="michael.chen@acme.com",
            role="account_executive", region="US-West", timezone="America/Los_Angeles",
            working_hours_start=9, working_hours_end=18,
        ),
        SalesRep(
            rep_id="REP-003", name="Jessica Taylor", email="jessica.taylor@acme.com",
            role="account_executive", region="US-East", timezone="America/New_York",
            working_hours_start=9, working_hours_end=17,
        ),
        SalesRep(
            rep_id="REP-004", name="Robert Garcia", email="robert.garcia@acme.com",
            role="senior_ae", region="US-East", timezone="America/New_York",
            working_hours_start=8, working_hours_end=17,
        ),
        SalesRep(
            rep_id="REP-005", name="Amanda Foster", email="amanda.foster@acme.com",
            role="manager", region="US-National", timezone="America/Chicago",
            working_hours_start=9, working_hours_end=18,
        ),
    ]
    for rep in reps:
        state.sales_reps[rep.rep_id] = rep


def _generate_products(state: EnvironmentState) -> None:
    products = [
        Product(
            product_id="PROD-001", name="Acme CRM Basic",
            category=ProductCategory.CRM, base_price=5000, monthly_price=500,
            description="Entry-level CRM with contact management and basic reporting.",
            features=["Contact Management", "Basic Reporting", "Email Integration"],
            min_customer_tier=CustomerTier.PROSPECT,
        ),
        Product(
            product_id="PROD-002", name="Acme CRM Professional",
            category=ProductCategory.CRM, base_price=15000, monthly_price=1500,
            description="Professional CRM with advanced pipeline, forecasting, and automation.",
            features=["Advanced Pipeline", "Forecasting", "Workflow Automation", "API Access"],
            min_customer_tier=CustomerTier.ACTIVE,
        ),
        Product(
            product_id="PROD-003", name="Acme CRM Enterprise",
            category=ProductCategory.CRM, base_price=50000, monthly_price=5000,
            description="Enterprise-grade CRM with dedicated support, custom integrations, and SLA.",
            features=["Dedicated Support", "Custom Integrations", "SLA", "Advanced Security"],
            min_customer_tier=CustomerTier.ENTERPRISE,
        ),
        Product(
            product_id="PROD-004", name="Acme Analytics Starter",
            category=ProductCategory.ANALYTICS, base_price=3000, monthly_price=300,
            description="Basic analytics dashboard with standard KPI tracking.",
            features=["KPI Dashboard", "Standard Reports", "Data Export"],
            min_customer_tier=CustomerTier.PROSPECT,
        ),
        Product(
            product_id="PROD-005", name="Acme Analytics Pro",
            category=ProductCategory.ANALYTICS, base_price=12000, monthly_price=1200,
            description="Advanced analytics with predictive models and custom dashboards.",
            features=["Predictive Analytics", "Custom Dashboards", "Real-time Data"],
            min_customer_tier=CustomerTier.ACTIVE,
        ),
        Product(
            product_id="PROD-006", name="Acme Automation Suite",
            category=ProductCategory.AUTOMATION, base_price=20000, monthly_price=2000,
            description="End-to-end workflow automation with AI-driven task routing.",
            features=["Workflow Builder", "AI Task Routing", "Integration Hub"],
            min_customer_tier=CustomerTier.ACTIVE,
        ),
        Product(
            product_id="PROD-007", name="Acme Security Shield",
            category=ProductCategory.SECURITY, base_price=25000, monthly_price=2500,
            description="Enterprise security suite with threat detection and compliance tools.",
            features=["Threat Detection", "Compliance Reporting", "SSO", "Audit Logs"],
            min_customer_tier=CustomerTier.ENTERPRISE,
        ),
        Product(
            product_id="PROD-008", name="Acme Integration Hub",
            category=ProductCategory.INTEGRATION, base_price=8000, monthly_price=800,
            description="Connect Acme products with 100+ third-party services.",
            features=["REST API", "Webhook Support", "Pre-built Connectors"],
            min_customer_tier=CustomerTier.ACTIVE,
        ),
        Product(
            product_id="PROD-009", name="Acme Support Desk",
            category=ProductCategory.SUPPORT, base_price=6000, monthly_price=600,
            description="Customer support ticketing with knowledge base and live chat.",
            features=["Ticket Management", "Knowledge Base", "Live Chat", "SLA Tracking"],
            min_customer_tier=CustomerTier.PROSPECT,
        ),
        Product(
            product_id="PROD-010", name="Acme Analytics Enterprise",
            category=ProductCategory.ANALYTICS, base_price=40000, monthly_price=4000,
            description="Enterprise analytics with data warehouse integration and ML pipelines.",
            features=["Data Warehouse", "ML Pipelines", "Custom Models", "Dedicated Analyst"],
            min_customer_tier=CustomerTier.ENTERPRISE,
            availability=False,  # Currently unavailable — edge case for tasks
        ),
        Product(
            product_id="PROD-011", name="Acme CRM Lite",
            category=ProductCategory.CRM, base_price=2000, monthly_price=200,
            description="Lightweight CRM for small teams. Limited to 5 users.",
            features=["Contact Management", "Basic Pipeline", "5 User Limit"],
            min_customer_tier=CustomerTier.PROSPECT,
        ),
        Product(
            product_id="PROD-012", name="Acme Automation Starter",
            category=ProductCategory.AUTOMATION, base_price=7000, monthly_price=700,
            description="Basic workflow automation with template-based flows.",
            features=["Template Workflows", "Email Automation", "Task Scheduling"],
            min_customer_tier=CustomerTier.PROSPECT,
        ),
        Product(
            product_id="PROD-013", name="Acme Security Essentials",
            category=ProductCategory.SECURITY, base_price=10000, monthly_price=1000,
            description="Core security features including MFA and basic threat monitoring.",
            features=["MFA", "Basic Threat Monitoring", "Password Policies"],
            min_customer_tier=CustomerTier.ACTIVE,
        ),
        Product(
            product_id="PROD-014", name="Acme Data Connector",
            category=ProductCategory.INTEGRATION, base_price=4000, monthly_price=400,
            description="Simple data sync between Acme and popular business tools.",
            features=["Bi-directional Sync", "Salesforce Connector", "Slack Integration"],
            min_customer_tier=CustomerTier.PROSPECT,
        ),
        Product(
            product_id="PROD-015", name="Acme Premium Support",
            category=ProductCategory.SUPPORT, base_price=15000, monthly_price=1500,
            description="Premium support with 24/7 availability and dedicated account manager.",
            features=["24/7 Support", "Dedicated Account Manager", "Priority Queue", "Quarterly Reviews"],
            min_customer_tier=CustomerTier.ENTERPRISE,
        ),
    ]
    for prod in products:
        state.products[prod.product_id] = prod


def _generate_customers(
    state: EnvironmentState, rng: random.Random, rep_ids: list[str]
) -> None:
    statuses = [
        CustomerStatus.ENTERPRISE, CustomerStatus.ENTERPRISE,
        CustomerStatus.ACTIVE, CustomerStatus.ACTIVE, CustomerStatus.ACTIVE,
        CustomerStatus.ACTIVE, CustomerStatus.ACTIVE,
        CustomerStatus.PROSPECT, CustomerStatus.PROSPECT, CustomerStatus.PROSPECT,
        CustomerStatus.PROSPECT,
        CustomerStatus.CHURNED, CustomerStatus.CHURNED,
        CustomerStatus.ACTIVE, CustomerStatus.ACTIVE,
        CustomerStatus.ENTERPRISE,
        CustomerStatus.PROSPECT, CustomerStatus.ACTIVE,
        CustomerStatus.ACTIVE, CustomerStatus.SUSPENDED,
    ]

    priorities = [
        Priority.P1, Priority.P1,
        Priority.P2, Priority.P2, Priority.P2,
        Priority.P3, Priority.P3, Priority.P3, Priority.P3,
        Priority.P4, Priority.P4, Priority.P4,
        Priority.P5, Priority.P5,
        Priority.P2, Priority.P3,
        Priority.P4, Priority.P3, Priority.P2, Priority.P3,
    ]

    revenues = [
        500000, 250000, 120000, 80000, 45000,
        200000, 60000, 15000, 30000, 10000,
        350000, 180000, 50000, 25000, 90000,
        150000, 20000, 70000, 110000, 40000,
    ]

    assert (
        len(_COMPANIES) == len(statuses) == len(priorities) == len(revenues) == 20
    ), f"Parallel lists out of sync: {len(_COMPANIES)}, {len(statuses)}, {len(priorities)}, {len(revenues)}"

    for i in range(20):
        company, industry = _COMPANIES[i]
        cid = f"C{i + 1:03d}"
        contact = _CONTACT_NAMES[i]
        email = contact.lower().replace(" ", ".") + "@" + company.lower().replace(" ", "").replace(".", "").replace(",", "") + ".com"

        # Determine last_contacted — some recent, some old, some never
        if i < 5:
            # Recently contacted (within 7 days)
            last_c = _REFERENCE_DATE - timedelta(days=rng.randint(1, 7))
        elif i < 12:
            # Contacted 8-30 days ago
            last_c = _REFERENCE_DATE - timedelta(days=rng.randint(8, 30))
        elif i < 16:
            # Old contact (30-90 days)
            last_c = _REFERENCE_DATE - timedelta(days=rng.randint(30, 90))
        else:
            # Never contacted
            last_c = None

        customer = Customer(
            customer_id=cid,
            company=company,
            industry=industry,
            status=statuses[i],
            contact_name=contact,
            contact_email=email,
            contact_phone=f"+1-555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
            priority=priorities[i],
            assigned_rep=rng.choice(rep_ids[:4]),  # Assign to AEs, not manager
            annual_revenue=revenues[i],
            tags=rng.sample(["key_account", "expansion", "at_risk", "upsell", "renewal", "new_logo"], k=rng.randint(0, 3)),
            created_at=_REFERENCE_DATE - timedelta(days=rng.randint(60, 365)),
            last_contacted=last_c,
            timezone=rng.choice(_TIMEZONES),
        )
        state.customers[cid] = customer


def _generate_opportunities(
    state: EnvironmentState, rng: random.Random
) -> None:
    """Generate 25 opportunities linked to customers."""
    stages = list(OpportunityStage)
    product_ids = list(state.products.keys())
    customer_ids = [c for c in state.customers.keys() if state.customers[c].status != CustomerStatus.CHURNED]

    for i in range(25):
        oid = f"OPP-{i + 1:04d}"
        cid = rng.choice(customer_ids)
        customer = state.customers[cid]
        stage = rng.choice(stages[:4])  # Mostly open stages
        if i >= 22:
            stage = rng.choice([OpportunityStage.CLOSED_WON, OpportunityStage.CLOSED_LOST])

        num_products = rng.randint(1, 3)
        selected_products = rng.sample(product_ids, k=min(num_products, len(product_ids)))
        expected_val = sum(
            state.products[pid].base_price for pid in selected_products
        ) * rng.uniform(0.8, 1.5)

        prob_map = {
            OpportunityStage.PROSPECTING: 0.1,
            OpportunityStage.QUALIFICATION: 0.3,
            OpportunityStage.PROPOSAL: 0.5,
            OpportunityStage.NEGOTIATION: 0.7,
            OpportunityStage.CLOSED_WON: 1.0,
            OpportunityStage.CLOSED_LOST: 0.0,
        }

        next_actions = [
            "Send proposal", "Schedule demo", "Follow up on pricing",
            "Arrange technical review", "Send contract", "Schedule meeting",
            "Prepare case study", "Request manager approval",
        ]

        last_contact_opp = _REFERENCE_DATE - timedelta(days=rng.randint(1, 45))

        opp = Opportunity(
            opportunity_id=oid,
            customer_id=cid,
            product_ids=selected_products,
            stage=stage,
            expected_value=round(expected_val, 2),
            probability=prob_map[stage],
            assigned_rep=customer.assigned_rep,
            created_at=_REFERENCE_DATE - timedelta(days=rng.randint(10, 180)),
            last_contact=last_contact_opp,
            next_action=rng.choice(next_actions) if stage not in (OpportunityStage.CLOSED_WON, OpportunityStage.CLOSED_LOST) else "",
            close_date=(_REFERENCE_DATE + timedelta(days=rng.randint(7, 90))).date() if stage not in (OpportunityStage.CLOSED_WON, OpportunityStage.CLOSED_LOST) else _REFERENCE_DATE_D,
        )
        state.opportunities[oid] = opp


def _generate_interactions(
    state: EnvironmentState, rng: random.Random
) -> None:
    """Generate 30 interaction history records."""
    interaction_types = list(InteractionType)
    customer_ids = list(state.customers.keys())
    sentiments = ["positive", "neutral", "negative", "positive", "neutral"]

    summaries = [
        "Discussed product features and pricing options",
        "Followed up on previous proposal — customer requested changes",
        "Initial discovery call — identified key pain points",
        "Product demo — customer impressed with automation features",
        "Addressed support ticket about integration issues",
        "Quarterly business review — discussed expansion plans",
        "Pricing negotiation — customer pushing for additional discount",
        "Technical deep-dive on security features",
        "Customer expressed concern about implementation timeline",
        "Shared case study from similar industry client",
        "Discussed renewal terms and potential upsell",
        "Customer requested custom integration support",
        "Escalation meeting regarding service quality concerns",
        "Introduction call with new stakeholder",
        "Contract review and final terms discussion",
    ]

    for i in range(30):
        iid = f"INT-{i + 1:04d}"
        cid = rng.choice(customer_ids)
        customer = state.customers[cid]
        days_ago = rng.randint(1, 60)

        itype = InteractionType.DEMO if i < 2 else rng.choice(interaction_types)

        interaction = Interaction(
            interaction_id=iid,
            customer_id=cid,
            rep_id=customer.assigned_rep,
            interaction_type=itype,
            datetime_occurred=_REFERENCE_DATE - timedelta(
                days=days_ago, hours=rng.randint(0, 8)
            ),
            summary="Product demo presentation" if i < 2 else rng.choice(summaries),
            sentiment="positive" if i < 2 else rng.choice(sentiments),
        )
        state.interactions[iid] = interaction


def _generate_meetings(
    state: EnvironmentState, rng: random.Random
) -> None:
    """Generate 15 meetings — some past (completed), some future (scheduled)."""
    customer_ids = list(state.customers.keys())

    for i in range(15):
        mid = f"MTG-{i + 1:04d}"
        cid = rng.choice(customer_ids[:15])  # Focus on non-churned
        customer = state.customers[cid]
        rep = state.sales_reps[customer.assigned_rep]
        rep_tz = ZoneInfo(rep.timezone)

        target_hour = rng.choice([9, 10, 11, 13, 14, 15, 16])
        if i < 8:
            days_offset = -rng.randint(1, 30)
            status = MeetingStatus.COMPLETED
        else:
            days_offset = rng.randint(1, 14)
            status = MeetingStatus.SCHEDULED

        # Construct meeting time in rep's local timezone, then convert to UTC
        ref_in_rep_tz = _REFERENCE_DATE.astimezone(rep_tz)
        local_date = (ref_in_rep_tz + timedelta(days=days_offset)).date()
        local_dt = datetime(local_date.year, local_date.month, local_date.day, target_hour, 0, tzinfo=rep_tz)
        dt_utc = local_dt.astimezone(ZoneInfo("UTC"))

        # Purpose: set review for large accounts or alternating meetings
        if customer.annual_revenue >= 100000 or i % 3 == 0:
            purpose = "review"
        else:
            purpose = "general"

        meeting = Meeting(
            meeting_id=mid,
            customer_id=cid,
            rep_id=customer.assigned_rep,
            datetime_start=dt_utc,
            duration_minutes=rng.choice([30, 30, 30, 60, 60]),
            status=status,
            meeting_type=rng.choice(["call", "video", "video", "in_person"]),
            purpose=purpose,
            notes=f"{'Quarterly Review' if purpose == 'review' else 'Discussion'} with {customer.company}",
        )
        state.meetings[mid] = meeting


def _generate_followups(
    state: EnvironmentState, rng: random.Random
) -> None:
    """Generate 10 follow-up tasks."""
    customer_ids = list(state.customers.keys())
    actions = [
        "Send proposal with updated pricing",
        "Schedule technical demo",
        "Follow up on contract review",
        "Send case study for their industry",
        "Check in on product satisfaction",
        "Prepare quarterly review materials",
        "Send onboarding documentation",
        "Request feedback on recent demo",
        "Discuss expansion opportunities",
        "Address outstanding support issues",
    ]

    for i in range(10):
        fid = f"FU-{i + 1:04d}"
        cid = rng.choice(customer_ids[:15])
        customer = state.customers[cid]

        if i < 3:
            status = FollowUpStatus.OVERDUE
            due = _REFERENCE_DATE_D - timedelta(days=rng.randint(1, 7))
        elif i < 7:
            status = FollowUpStatus.PENDING
            due = _REFERENCE_DATE_D + timedelta(days=rng.randint(1, 14))
        else:
            status = FollowUpStatus.COMPLETED
            due = _REFERENCE_DATE_D - timedelta(days=rng.randint(1, 14))

        followup = FollowUp(
            followup_id=fid,
            customer_id=cid,
            opportunity_id=rng.choice(list(state.opportunities.keys())) if rng.random() > 0.3 else None,
            assigned_rep=customer.assigned_rep,
            action=actions[i],
            due_date=due,
            status=status,
            created_at=_REFERENCE_DATE - timedelta(days=rng.randint(3, 21)),
        )
        state.followups[fid] = followup


def _generate_policy_rules(state: EnvironmentState) -> None:
    """Create the 8 core business policy rules."""
    rules = [
        PolicyRule(
            rule_id="POL-001",
            name="Maximum Discount Limit",
            description="Discounts above 15% require manager approval. Absolute maximum is 20%.",
            rule_type=PolicyRuleType.DISCOUNT,
            parameters={
                "max_without_approval": 15.0,
                "absolute_max": 20.0,
                "approval_role": "manager",
            },
        ),
        PolicyRule(
            rule_id="POL-002",
            name="Business Hours Contact",
            description="Customers must not be contacted outside 9:00-18:00 in their local timezone.",
            rule_type=PolicyRuleType.CONTACT,
            parameters={
                "start_hour": 9,
                "end_hour": 18,
            },
        ),
        PolicyRule(
            rule_id="POL-003",
            name="Enterprise Product Eligibility",
            description="Products with min_customer_tier='enterprise' can only be sold to customers with enterprise status.",
            rule_type=PolicyRuleType.ELIGIBILITY,
            parameters={
                "check_field": "min_customer_tier",
                "customer_field": "status",
            },
        ),
        PolicyRule(
            rule_id="POL-004",
            name="Post-Demo Follow-Up",
            description="A follow-up task must be created within 48 hours after a product demo interaction.",
            rule_type=PolicyRuleType.CONTACT,
            parameters={
                "trigger_interaction_type": "demo",
                "max_hours": 48,
            },
        ),
        PolicyRule(
            rule_id="POL-005",
            name="No Double-Booking",
            description="A sales rep cannot have two meetings overlapping at the same time.",
            rule_type=PolicyRuleType.SCHEDULING,
            parameters={
                "check": "no_overlap",
            },
        ),
        PolicyRule(
            rule_id="POL-006",
            name="Meeting Buffer Time",
            description="There must be at least a 30-minute gap between consecutive meetings for a rep.",
            rule_type=PolicyRuleType.SCHEDULING,
            parameters={
                "min_gap_minutes": 30,
            },
        ),
        PolicyRule(
            rule_id="POL-007",
            name="High-Priority Rep Assignment",
            description="P1 and P2 priority customers must be assigned to senior_ae or manager role reps.",
            rule_type=PolicyRuleType.ESCALATION,
            parameters={
                "priority_threshold": 2,
                "required_roles": ["senior_ae", "manager"],
            },
        ),
        PolicyRule(
            rule_id="POL-008",
            name="Large Account Quarterly Review",
            description="Customer accounts with annual revenue above $100,000 must have a quarterly review meeting scheduled.",
            rule_type=PolicyRuleType.CONTACT,
            parameters={
                "revenue_threshold": 100000,
                "review_interval_days": 90,
            },
        ),
    ]
    for rule in rules:
        state.policy_rules[rule.rule_id] = rule
