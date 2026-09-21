"""Evaluator: scores a task run against expected outcomes.

The evaluator reads:
  - the BenchmarkTask (expected outcome)
  - the before/after EnvironmentState
  - the action log

It does NOT read the agent's text. This is the anti-gaming design.

Composite score:
    0.50 * state_correctness
  + 0.25 * required_tools_score
  + 0.15 * policy_compliance_score
  + 0.05 * efficiency_score
  + 0.05 * no_forbidden_score

Task "success" is a binary that requires all of:
  - all required matchers satisfied
  - all required tools called (successfully)
  - policy violations introduced <= budget
  - no forbidden tools or forbidden state matches
"""

from __future__ import annotations

from acme.environment.policies import PolicyEngine
from acme.environment.state import EnvironmentState
from acme.evaluation.failure_taxonomy import classify_failure
from acme.evaluation.metrics import TaskResult
from acme.tasks.task_schema import BenchmarkTask


class Evaluator:
    """Composite scorer for benchmark tasks."""

    WEIGHT_STATE = 0.50
    WEIGHT_TOOLS = 0.25
    WEIGHT_POLICY = 0.15
    WEIGHT_EFFICIENCY = 0.05
    WEIGHT_NO_FORBIDDEN = 0.05

    def evaluate(
        self,
        task: BenchmarkTask,
        before_state: EnvironmentState,
        after_state: EnvironmentState,
        action_log,
    ) -> TaskResult:
        # 1. State correctness
        state_score, state_details = self._score_state_correctness(task, after_state)

        # 2. Required tools
        tools_score, tools_details = self._score_required_tools(task, action_log)

        # 3. Policy compliance
        policy_score, policy_details = self._score_policy_compliance(
            task, before_state, after_state
        )

        # 4. Efficiency
        efficiency_score, efficiency_details = self._score_efficiency(task, action_log)

        # 5. No forbidden actions
        forbidden_score, forbidden_details = self._score_no_forbidden(
            task, after_state, action_log
        )

        # Composite
        composite = (
            self.WEIGHT_STATE * state_score
            + self.WEIGHT_TOOLS * tools_score
            + self.WEIGHT_POLICY * policy_score
            + self.WEIGHT_EFFICIENCY * efficiency_score
            + self.WEIGHT_NO_FORBIDDEN * forbidden_score
        )

        # Success criteria (all must be true)
        success = (
            state_score == 1.0
            and tools_score == 1.0
            and policy_details["introduced"] <= policy_details["budget"]
            and forbidden_score == 1.0
        )

        # Failure classification
        failure_type = None
        failure_details: dict = {}
        if not success:
            failure_type, failure_details = classify_failure(
                action_log=action_log,
                state_score=state_score,
                tools_score=tools_score,
                policy_score=policy_score,
                forbidden_score=forbidden_score,
                state_details=state_details,
                tools_details=tools_details,
                policy_details=policy_details,
                forbidden_details=forbidden_details,
            )

        successful_calls = sum(1 for e in action_log if e.result_success)
        failed_calls = len(action_log) - successful_calls

        return TaskResult(
            task_id=task.task_id,
            category=task.category.value,
            difficulty=task.difficulty,
            success=success,
            score=round(composite, 4),
            state_correctness=round(state_score, 4),
            required_tools_score=round(tools_score, 4),
            policy_compliance_score=round(policy_score, 4),
            efficiency_score=round(efficiency_score, 4),
            no_forbidden_score=round(forbidden_score, 4),
            policy_violations_before=policy_details["before"],
            policy_violations_after=policy_details["after"],
            policy_violations_introduced=policy_details["introduced"],
            tool_call_count=len(action_log),
            successful_tool_calls=successful_calls,
            failed_tool_calls=failed_calls,
            failure_type=failure_type,
            failure_details=failure_details,
        )

    # ------------------------------------------------------------------
    # Sub-scores
    # ------------------------------------------------------------------

    def _score_state_correctness(
        self, task: BenchmarkTask, after_state: EnvironmentState
    ) -> tuple[float, dict]:
        required = task.expected_outcome.required_state
        if not required:
            return 1.0, {"required_matchers": 0, "satisfied": 0, "details": []}

        total = 0
        satisfied = 0
        details: list[dict] = []
        for coll_name, matchers in required.items():
            for matcher in matchers:
                total += 1
                matched = self._matcher_matches(coll_name, matcher, after_state)
                details.append({
                    "collection": coll_name,
                    "matcher": matcher,
                    "matched": matched,
                })
                if matched:
                    satisfied += 1

        if total == 0:
            return 1.0, {"required_matchers": 0, "satisfied": 0, "details": details}

        return satisfied / total, {
            "required_matchers": total,
            "satisfied": satisfied,
            "details": details,
        }

    def _matcher_matches(
        self, coll_name: str, matcher: dict, state: EnvironmentState
    ) -> bool:
        store = getattr(state, coll_name, None)
        if store is None:
            return False
        for entity in store.values():
            entity_dict = entity.model_dump(mode="json")
            if all(entity_dict.get(k) == v for k, v in matcher.items()):
                return True
        return False

    def _score_required_tools(self, task: BenchmarkTask, action_log) -> tuple[float, dict]:
        required = task.expected_outcome.required_tool_calls
        if not required:
            return 1.0, {"required": [], "present": [], "missing": []}

        successful_calls = {e.tool for e in action_log if e.result_success}
        present = [t for t in required if t in successful_calls]
        missing = [t for t in required if t not in successful_calls]

        score = len(present) / len(required)
        return score, {"required": required, "present": present, "missing": missing}

    def _score_policy_compliance(
        self,
        task: BenchmarkTask,
        before_state: EnvironmentState,
        after_state: EnvironmentState,
    ) -> tuple[float, dict]:
        before_count = PolicyEngine(before_state).audit_full_state().violation_count
        after_count = PolicyEngine(after_state).audit_full_state().violation_count
        introduced = max(0, after_count - before_count)

        budget = task.expected_outcome.max_policy_violations
        if introduced <= budget:
            score = 1.0
        else:
            score = 0.0

        return score, {
            "before": before_count,
            "after": after_count,
            "introduced": introduced,
            "budget": budget,
        }

    def _score_efficiency(self, task: BenchmarkTask, action_log) -> tuple[float, dict]:
        required = task.expected_outcome.required_tool_calls
        optimal = task.expected_outcome.max_tool_calls
        if optimal is None:
            optimal = max(1, len(required)) + 2

        actual = len(action_log)
        if actual <= optimal:
            return 1.0, {"actual": actual, "optimal": optimal, "excess": 0}

        excess = actual - optimal
        excess_ratio = excess / max(1, optimal)
        score = 1.0 / (1.0 + excess_ratio)
        return score, {"actual": actual, "optimal": optimal, "excess": excess}

    def _score_no_forbidden(
        self,
        task: BenchmarkTask,
        after_state: EnvironmentState,
        action_log,
    ) -> tuple[float, dict]:
        forbidden_tools = set(task.expected_outcome.forbidden_tool_calls)
        called_forbidden = [e.tool for e in action_log if e.tool in forbidden_tools]

        forbidden_matches: list[dict] = []
        for coll_name, matchers in task.expected_outcome.forbidden_state.items():
            for matcher in matchers:
                if self._matcher_matches(coll_name, matcher, after_state):
                    forbidden_matches.append({
                        "collection": coll_name,
                        "matcher": matcher,
                    })

        if called_forbidden or forbidden_matches:
            return 0.0, {
                "forbidden_tools_called": called_forbidden,
                "forbidden_state_matches": forbidden_matches,
            }

        return 1.0, {
            "forbidden_tools_called": [],
            "forbidden_state_matches": [],
        }


__all__ = ["Evaluator"]