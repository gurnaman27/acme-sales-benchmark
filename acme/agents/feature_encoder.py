"""State and Action Featurizer for Neural Network (MLP) Agents.

Converts (task, step_number, history) into a fixed-size NumPy vector x,
and maps discrete tool action templates to integer indices y.
"""

from __future__ import annotations

import numpy as np

from acme.tasks.task_library import TASK_LIBRARY
from acme.tasks.task_schema import BenchmarkTask, TaskCategory


# Total number of tasks in benchmark
N_TASKS = len(TASK_LIBRARY)
TASK_ID_TO_INDEX = {t.task_id: i for i, t in enumerate(TASK_LIBRARY)}
CATEGORY_TO_INDEX = {c: i for i, c in enumerate(TaskCategory)}
N_CATEGORIES = len(TaskCategory)

# Feature dimension:
# 30 (task_id one-hot) + 5 (category one-hot) + 4 (difficulty) + 10 (step_num & history features) + 15 (state summary) = 64
FEATURE_DIM = N_TASKS + N_CATEGORIES + 4 + 10 + 15


class FeatureEncoder:
    """Encodes task state into a fixed-size numpy feature vector."""

    def __init__(self, feature_dim: int = FEATURE_DIM):
        self.feature_dim = feature_dim

    def encode_state(
        self,
        task: BenchmarkTask,
        step_number: int = 0,
        tool_history: list[str] | None = None,
    ) -> np.ndarray:
        """Encode the current state into a 1D numpy array of shape (feature_dim,)."""
        vec = np.zeros(self.feature_dim, dtype=np.float32)

        # 1. Task ID one-hot (bits 0..29)
        t_idx = TASK_ID_TO_INDEX.get(task.task_id, 0)
        vec[t_idx] = 1.0

        offset = N_TASKS
        # 2. Category one-hot (bits 30..34)
        c_idx = CATEGORY_TO_INDEX.get(task.category, 0)
        vec[offset + c_idx] = 1.0

        offset += N_CATEGORIES
        # 3. Difficulty one-hot (bits 35..38)
        diff_idx = max(0, min(3, task.difficulty - 1))
        vec[offset + diff_idx] = 1.0

        offset += 4
        # 4. Step features (bits 39..48)
        vec[offset] = float(step_number) / 10.0
        if tool_history:
            vec[offset + 1] = float(len(tool_history)) / 10.0
            # Frequency of tool calls
            for tool_name in tool_history[-5:]:
                vec[offset + 2] += 0.2

        offset += 10
        # 5. Task-specific instruction keywords (bits 49..63)
        instr = task.instruction.lower()
        keywords = [
            "customer", "opportunity", "meeting", "follow-up", "discount",
            "eligibility", "calendar", "reschedule", "reassign", "stale",
            "quarterly", "healthcare", "notes", "cancel", "product"
        ]
        for idx, kw in enumerate(keywords):
            if kw in instr:
                vec[offset + idx] = 1.0

        return vec


# Registered Action Templates for Imitation Learning
ACTION_TEMPLATES: list[dict] = [
    # L1 actions (0-7)
    {"type": "tool", "name": "get_customer", "args": {"customer_id": "C001"}},
    {"type": "tool", "name": "get_customer", "args": {"customer_id": "C004"}},
    {"type": "tool", "name": "get_customer", "args": {"customer_id": "C005"}},
    {"type": "tool", "name": "search_customers", "args": {"industry": "Healthcare"}},
    {"type": "tool", "name": "get_product", "args": {"product_id": "PROD-003"}},
    {"type": "tool", "name": "check_product_eligibility", "args": {"customer_id": "C001", "product_id": "PROD-003"}},
    {"type": "tool", "name": "get_current_time", "args": {}},
    {"type": "tool", "name": "get_policies", "args": {"category": "discount"}},
    # (8-14)
    {"type": "tool", "name": "get_customer_history", "args": {"customer_id": "C005"}},
    {"type": "tool", "name": "get_customer_history", "args": {"customer_id": "C001"}},
    {"type": "tool", "name": "search_opportunities", "args": {"limit": 100}},
    {"type": "tool", "name": "search_opportunities", "args": {"customer_id": "C005"}},
    {"type": "tool", "name": "search_opportunities", "args": {"customer_id": "C001"}},
    {"type": "tool", "name": "get_policies", "args": {"category": "assignment"}},
    {"type": "tool", "name": "get_policies", "args": {"category": "scheduling"}},

    # L2 mutating actions (15-21)
    {"type": "tool", "name": "update_opportunity", "args": {"opportunity_id": "OPP-0005", "updates": {"next_action": "send_proposal"}}},
    {"type": "tool", "name": "update_opportunity", "args": {"opportunity_id": "OPP-0001", "updates": {"stage": "negotiation"}}},
    {"type": "tool", "name": "create_followup", "args": {"customer_id": "C003", "action": "Send contract", "due_date": "2025-03-22"}},
    {"type": "tool", "name": "cancel_meeting", "args": {"meeting_id": "MTG-9001", "reason": "customer request"}},
    {"type": "tool", "name": "create_opportunity", "args": {"customer_id": "C005", "product_ids": ["PROD-005"], "stage": "qualification"}},
    {"type": "tool", "name": "update_customer", "args": {"customer_id": "C001", "updates": {"notes": "Called about renewal"}}},
    {"type": "tool", "name": "get_followups", "args": {"overdue_only": True}},

    # L3 actions (22-31)
    {"type": "tool", "name": "get_available_slots", "args": {"rep_id": "REP-001", "start_date": "2025-03-24", "end_date": "2025-03-24"}},
    {"type": "tool", "name": "get_available_slots", "args": {"rep_id": "REP-002", "start_date": "2025-03-24", "end_date": "2025-03-24"}},
    {"type": "tool", "name": "check_product_eligibility", "args": {"customer_id": "C006", "product_id": "PROD-004"}},
    {"type": "tool", "name": "check_product_eligibility", "args": {"customer_id": "C003", "product_id": "PROD-002"}},
    {"type": "tool", "name": "create_followup", "args": {"customer_id": "C002", "action": "Post-demo follow-up", "due_date": "2025-03-17"}},
    {"type": "tool", "name": "create_opportunity", "args": {"customer_id": "C003", "product_ids": ["PROD-002"], "stage": "qualification"}},
    {"type": "tool", "name": "schedule_meeting", "args": {"customer_id": "C004", "rep_id": "REP-001", "start_time": "2025-03-24T18:00:00+00:00", "duration_minutes": 60}},
    {"type": "tool", "name": "schedule_meeting", "args": {"customer_id": "C003", "rep_id": "REP-002", "start_time": "2025-03-24T18:00:00+00:00", "duration_minutes": 30}},

    # L4 actions (32-44)
    {"type": "tool", "name": "schedule_meeting", "args": {"customer_id": "C001", "rep_id": "REP-001", "start_time": "2025-03-26T18:00:00+00:00", "duration_minutes": 30}},
    {"type": "tool", "name": "schedule_meeting", "args": {"customer_id": "C002", "rep_id": "REP-001", "start_time": "2025-03-25T18:00:00+00:00", "duration_minutes": 30}},
    {"type": "tool", "name": "schedule_meeting", "args": {"customer_id": "C001", "rep_id": "REP-001", "start_time": "2025-03-27T18:00:00+00:00", "duration_minutes": 30}},
    {"type": "tool", "name": "create_followup", "args": {"customer_id": "C001", "action": "Send agenda", "due_date": "2025-03-25"}},
    {"type": "tool", "name": "create_followup", "args": {"customer_id": "C002", "action": "Post-demo review", "due_date": "2025-03-26"}},
    {"type": "tool", "name": "create_followup", "args": {"customer_id": "C001", "action": "Executive approval", "due_date": "2025-03-18"}},
    {"type": "tool", "name": "create_followup", "args": {"customer_id": "C002", "action": "Quarterly Business Review", "due_date": "2025-03-30"}},
    {"type": "tool", "name": "create_followup", "args": {"customer_id": "C017", "action": "Quarterly Business Review", "due_date": "2025-03-30"}},
    {"type": "tool", "name": "create_followup", "args": {"customer_id": "C005", "action": "Stale opportunity outreach", "due_date": "2025-03-20"}},
    {"type": "tool", "name": "cancel_meeting", "args": {"meeting_id": "MTG-9002", "reason": "scheduling conflict"}},
    {"type": "tool", "name": "check_product_eligibility", "args": {"customer_id": "C002", "product_id": "PROD-001"}},
    {"type": "tool", "name": "check_product_eligibility", "args": {"customer_id": "C001", "product_id": "PROD-001"}},
    {"type": "tool", "name": "check_product_eligibility", "args": {"customer_id": "C004", "product_id": "PROD-001"}},
    {"type": "tool", "name": "create_opportunity", "args": {"customer_id": "C002", "product_ids": ["PROD-001"], "stage": "qualification"}},
    {"type": "tool", "name": "create_opportunity", "args": {"customer_id": "C001", "product_ids": ["PROD-001"], "stage": "proposal", "discount_percent": 15.0}},
    {"type": "tool", "name": "create_opportunity", "args": {"customer_id": "C004", "product_ids": ["PROD-002"], "stage": "qualification"}},
    {"type": "tool", "name": "update_customer", "args": {"customer_id": "C001", "updates": {"notes": "Re-engagement scheduled"}}},
    {"type": "tool", "name": "update_opportunity", "args": {"opportunity_id": "OPP-9001", "updates": {"next_action": "Re-engage customer"}}},

    # Final action (finish task)
    {"type": "final", "message": "Task complete."},
]

N_ACTIONS = len(ACTION_TEMPLATES)
