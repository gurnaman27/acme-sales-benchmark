"""Environment state management — snapshot, restore, and diff.

The EnvironmentState holds all enterprise data in-memory and supports
deep-copy snapshotting so each benchmark task starts from a clean state.
The diff() utility lets the evaluator compare before/after states to
determine what the participant agent actually changed.
"""

from __future__ import annotations

import copy
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, PrivateAttr

from acme.environment.models import (
    Customer,
    Product,
    Opportunity,
    Meeting,
    Interaction,
    FollowUp,
    PolicyRule,
    SalesRep,
)


class EnvironmentState(BaseModel):
    """Complete state of the Acme Sales simulated environment.

    All collections are keyed by their entity ID for O(1) lookup.
    """

    sales_reps: dict[str, SalesRep] = Field(default_factory=dict)
    customers: dict[str, Customer] = Field(default_factory=dict)
    products: dict[str, Product] = Field(default_factory=dict)
    opportunities: dict[str, Opportunity] = Field(default_factory=dict)
    meetings: dict[str, Meeting] = Field(default_factory=dict)
    interactions: dict[str, Interaction] = Field(default_factory=dict)
    followups: dict[str, FollowUp] = Field(default_factory=dict)
    policy_rules: dict[str, PolicyRule] = Field(default_factory=dict)

    # Simulation reference timestamp (all relative task logic compares against this)
    reference_time: datetime = Field(
        default_factory=lambda: datetime(2025, 3, 15, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
    )

    # Monotonic counters for generating unique IDs
    _next_meeting_id: int = PrivateAttr(default=1)
    _next_interaction_id: int = PrivateAttr(default=1)
    _next_followup_id: int = PrivateAttr(default=1)
    _next_opportunity_id: int = PrivateAttr(default=1)

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # ID generators
    # ------------------------------------------------------------------

    def next_meeting_id(self) -> str:
        mid = f"MTG-{self._next_meeting_id:04d}"
        self._next_meeting_id += 1
        return mid

    def next_interaction_id(self) -> str:
        iid = f"INT-{self._next_interaction_id:04d}"
        self._next_interaction_id += 1
        return iid

    def next_followup_id(self) -> str:
        fid = f"FU-{self._next_followup_id:04d}"
        self._next_followup_id += 1
        return fid

    def next_opportunity_id(self) -> str:
        oid = f"OPP-{self._next_opportunity_id:04d}"
        self._next_opportunity_id += 1
        return oid

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def snapshot(self) -> EnvironmentState:
        """Return a deep copy of the current state."""
        return copy.deepcopy(self)

    @classmethod
    def from_snapshot(cls, snapshot: EnvironmentState) -> EnvironmentState:
        """Create a new state from a snapshot."""
        return copy.deepcopy(snapshot)

    # ------------------------------------------------------------------
    # Apply task-specific patches
    # ------------------------------------------------------------------

    def apply_patch(self, patch: dict) -> None:
        """Apply a task-specific state patch.

        The patch dict maps collection names to dicts of entity data.
        Example::

            {
                "customers": {
                    "C001": {"last_contacted": None}  # override fields
                },
                "meetings": {
                    "MTG-0099": { ... full meeting dict ... }  # add new
                }
            }
        """
        collection_map = {
            "sales_reps": (self.sales_reps, SalesRep),
            "customers": (self.customers, Customer),
            "products": (self.products, Product),
            "opportunities": (self.opportunities, Opportunity),
            "meetings": (self.meetings, Meeting),
            "interactions": (self.interactions, Interaction),
            "followups": (self.followups, FollowUp),
            "policy_rules": (self.policy_rules, PolicyRule),
        }

        for collection_name, entities in patch.items():
            if collection_name not in collection_map:
                continue
            store, model_cls = collection_map[collection_name]
            for entity_id, entity_data in entities.items():
                if entity_id in store:
                    # Merge: update existing entity fields
                    existing = store[entity_id]
                    updated = existing.model_copy(update=entity_data)
                    store[entity_id] = updated
                else:
                    # Create new entity from full data
                    store[entity_id] = model_cls(**entity_data)


# ---------------------------------------------------------------------------
# State diffing for evaluation
# ---------------------------------------------------------------------------

class StateDiff(BaseModel):
    """Describes changes between two EnvironmentState snapshots."""

    added: dict[str, list[str]] = Field(default_factory=dict)
    removed: dict[str, list[str]] = Field(default_factory=dict)
    modified: dict[str, dict[str, dict]] = Field(default_factory=dict)


def diff_states(before: EnvironmentState, after: EnvironmentState) -> StateDiff:
    """Compute the difference between two environment states.

    Returns a StateDiff describing which entities were added, removed,
    or modified (with field-level changes).
    """
    result = StateDiff()
    collections = [
        "sales_reps", "customers", "products", "opportunities",
        "meetings", "interactions", "followups", "policy_rules",
    ]

    for coll_name in collections:
        before_store: dict = getattr(before, coll_name)
        after_store: dict = getattr(after, coll_name)

        before_keys = set(before_store.keys())
        after_keys = set(after_store.keys())

        # Added entities
        added = after_keys - before_keys
        if added:
            result.added[coll_name] = sorted(added)

        # Removed entities
        removed = before_keys - after_keys
        if removed:
            result.removed[coll_name] = sorted(removed)

        # Modified entities
        common = before_keys & after_keys
        for entity_id in sorted(common):
            before_dict = before_store[entity_id].model_dump()
            after_dict = after_store[entity_id].model_dump()
            changes = {}
            for key in before_dict:
                if before_dict[key] != after_dict.get(key):
                    changes[key] = {
                        "before": before_dict[key],
                        "after": after_dict.get(key),
                    }
            if changes:
                if coll_name not in result.modified:
                    result.modified[coll_name] = {}
                result.modified[coll_name][entity_id] = changes

    return result
