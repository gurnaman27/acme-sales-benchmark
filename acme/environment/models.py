"""Pydantic models for all Acme Sales Corporation enterprise entities.

Every entity in the simulated environment is represented as a Pydantic model
so we get automatic validation, serialization, and clean diffing between states.
"""

from __future__ import annotations

from datetime import datetime, date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CustomerStatus(str, Enum):
    """Account status for a customer."""
    PROSPECT = "prospect"
    ACTIVE = "active"
    ENTERPRISE = "enterprise"
    CHURNED = "churned"
    SUSPENDED = "suspended"


class CustomerTier(str, Enum):
    """Account tier for product eligibility."""
    PROSPECT = "prospect"
    ACTIVE = "active"
    ENTERPRISE = "enterprise"


class Priority(int, Enum):
    """Customer priority level (1 = highest)."""
    P1 = 1
    P2 = 2
    P3 = 3
    P4 = 4
    P5 = 5


class OpportunityStage(str, Enum):
    """Sales pipeline stages."""
    PROSPECTING = "prospecting"
    QUALIFICATION = "qualification"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


class MeetingStatus(str, Enum):
    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"


class InteractionType(str, Enum):
    CALL = "call"
    EMAIL = "email"
    MEETING = "meeting"
    DEMO = "demo"
    SUPPORT = "support"
    NOTE = "note"


class FollowUpStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class ProductCategory(str, Enum):
    CRM = "crm"
    ANALYTICS = "analytics"
    AUTOMATION = "automation"
    SECURITY = "security"
    INTEGRATION = "integration"
    SUPPORT = "support"


class PolicyRuleType(str, Enum):
    DISCOUNT = "discount"
    SCHEDULING = "scheduling"
    CONTACT = "contact"
    APPROVAL = "approval"
    ELIGIBILITY = "eligibility"
    ESCALATION = "escalation"


# ---------------------------------------------------------------------------
# Core Entities (All timestamps are timezone-aware UTC)
# ---------------------------------------------------------------------------

class SalesRep(BaseModel):
    """A sales representative at Acme."""
    rep_id: str
    name: str
    email: str
    role: str = "account_executive"  # account_executive | senior_ae | manager
    region: str = "US-West"
    timezone: str = "America/Los_Angeles"
    active: bool = True
    working_hours_start: int = 9   # 24h format
    working_hours_end: int = 18    # 24h format


class Customer(BaseModel):
    """An Acme customer or prospect."""
    customer_id: str
    company: str
    industry: str
    status: CustomerStatus = CustomerStatus.ACTIVE
    contact_name: str
    contact_email: str
    contact_phone: str = ""
    priority: Priority = Priority.P3
    assigned_rep: str  # rep_id
    annual_revenue: float = 0.0
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    created_at: datetime
    last_contacted: Optional[datetime] = None
    timezone: str = "America/Los_Angeles"

    @property
    def tier(self) -> CustomerTier:
        """Map customer status to eligibility tier."""
        return {
            CustomerStatus.PROSPECT: CustomerTier.PROSPECT,
            CustomerStatus.ACTIVE: CustomerTier.ACTIVE,
            CustomerStatus.ENTERPRISE: CustomerTier.ENTERPRISE,
        }.get(self.status, CustomerTier.PROSPECT)


class Product(BaseModel):
    """An Acme product available for sale."""
    product_id: str
    name: str
    category: ProductCategory
    base_price: float
    monthly_price: float = 0.0
    availability: bool = True
    min_customer_tier: CustomerTier = CustomerTier.PROSPECT
    description: str = ""
    features: list[str] = Field(default_factory=list)
    max_discount_pct: float = 15.0  # default max discount %
    requires_approval_above: float = 10000.0  # deal size needing manager approval


class Opportunity(BaseModel):
    """A sales opportunity in the pipeline."""
    opportunity_id: str
    customer_id: str
    product_ids: list[str] = Field(default_factory=list)
    stage: OpportunityStage = OpportunityStage.PROSPECTING
    expected_value: float = 0.0
    probability: float = 0.0  # 0.0 - 1.0
    assigned_rep: str  # rep_id
    created_at: datetime
    last_contact: Optional[datetime] = None
    next_action: str = ""
    notes: str = ""
    close_date: Optional[date] = None
    discount_percent: float = 0.0


class Meeting(BaseModel):
    """A scheduled meeting."""
    meeting_id: str
    customer_id: str
    rep_id: str
    datetime_start: datetime
    duration_minutes: int = 30
    status: MeetingStatus = MeetingStatus.SCHEDULED
    meeting_type: str = "call"  # call | video | in_person
    purpose: str = "general"   # general | demo | review | renewal | onboarding
    notes: str = ""
    location: str = ""


class Interaction(BaseModel):
    """A logged interaction with a customer."""
    interaction_id: str
    customer_id: str
    rep_id: str
    interaction_type: InteractionType
    datetime_occurred: datetime
    summary: str
    details: str = ""
    opportunity_id: Optional[str] = None
    sentiment: Optional[str] = None  # positive | neutral | negative


class FollowUp(BaseModel):
    """A follow-up task."""
    followup_id: str
    customer_id: str
    opportunity_id: Optional[str] = None
    assigned_rep: str  # rep_id
    action: str
    due_date: date
    status: FollowUpStatus = FollowUpStatus.PENDING
    notes: str = ""
    created_at: datetime


class PolicyRule(BaseModel):
    """A business policy rule."""
    rule_id: str
    name: str
    description: str
    rule_type: PolicyRuleType
    parameters: dict = Field(default_factory=dict)
    active: bool = True
