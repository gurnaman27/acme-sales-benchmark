"""Tool registry: dispatch, access control, and action logging.

The `ToolRegistry` is the single entry point for agent tool calls.
It wraps every tool with:
  - access control (per-task allowed_tools whitelist)
  - structured error handling (exceptions become ToolResult failures)
  - logging (every call is appended to `self.log`)
"""

from __future__ import annotations

from typing import Callable

from acme.environment.state import EnvironmentState
from acme.tools.tool_types import (
    ActionLogEntry,
    ToolResult,
    ERR_UNKNOWN_TOOL,
    ERR_TOOL_NOT_ALLOWED,
    ERR_TOOL_EXCEPTION,
)


class ToolRegistry:
    """Central registry for agent-callable tools.

    Example::

        registry = ToolRegistry(state)
        result = registry.call("get_customer", customer_id="C001")
        print(result.data["company"])

        # Access the log after a task
        for entry in registry.log:
            print(entry.step, entry.tool, entry.result_success)
    """

    def __init__(
        self,
        state: EnvironmentState,
        allowed_tools: list[str] | None = None,
    ):
        """
        Args:
            state: The environment the tools will operate on.
            allowed_tools: Optional whitelist. If provided, calling any
                tool not in this list returns a TOOL_NOT_ALLOWED error.
                If None, all registered tools are available.
        """
        self.state = state
        self.log: list[ActionLogEntry] = []
        self._allowed = set(allowed_tools) if allowed_tools is not None else None
        self._tools: dict[str, Callable] = {}
        self._register_default_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _register_default_tools(self) -> None:
        """Register all built-in tools."""
        from acme.tools.customer_tools import (
            search_customers,
            get_customer,
            get_customer_history,
            update_customer,
        )
        from acme.tools.product_tools import (
            search_products,
            get_product,
            check_product_eligibility,
        )
        from acme.tools.opportunity_tools import (
            get_opportunity,
            search_opportunities,
            update_opportunity,
            create_opportunity,
        )
        from acme.tools.calendar_tools import (
            get_current_time,
            get_available_slots,
            schedule_meeting,
            cancel_meeting,
            get_meetings,
        )
        from acme.tools.followup_tools import (
            create_followup,
            get_followups,
            complete_followup,
        )
        from acme.tools.policy_tools import (
            check_policy,
            get_policies,
        )

        self._tools = {
            # Customer tools
            "search_customers": search_customers,
            "get_customer": get_customer,
            "get_customer_history": get_customer_history,
            "update_customer": update_customer,
            # Product tools
            "search_products": search_products,
            "get_product": get_product,
            "check_product_eligibility": check_product_eligibility,
            # Opportunity tools
            "get_opportunity": get_opportunity,
            "search_opportunities": search_opportunities,
            "update_opportunity": update_opportunity,
            "create_opportunity": create_opportunity,
            # Calendar tools
            "get_current_time": get_current_time,
            "get_available_slots": get_available_slots,
            "schedule_meeting": schedule_meeting,
            "cancel_meeting": cancel_meeting,
            "get_meetings": get_meetings,
            # Followup tools
            "create_followup": create_followup,
            "get_followups": get_followups,
            "complete_followup": complete_followup,
            # Policy tools
            "check_policy": check_policy,
            "get_policies": get_policies,
        }

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def call(self, name: str, **kwargs) -> ToolResult:
        """Call a tool by name. Always returns a ToolResult (never raises)."""
        # 1. Unknown tool?
        if name not in self._tools:
            result = ToolResult.fail(
                error=f"Unknown tool: '{name}'",
                error_code=ERR_UNKNOWN_TOOL,
                timestamp=self.state.reference_time,
            )
            self._log(name, kwargs, result)
            return result

        # 2. Access control
        if self._allowed is not None and name not in self._allowed:
            result = ToolResult.fail(
                error=f"Tool '{name}' is not available for this task",
                error_code=ERR_TOOL_NOT_ALLOWED,
                timestamp=self.state.reference_time,
            )
            self._log(name, kwargs, result)
            return result

        # 3. Execute (catch any exception so a buggy tool doesn't crash a run)
        fn = self._tools[name]
        try:
            result = fn(self.state, **kwargs)
        except TypeError as e:
            # Wrong arguments — likely a bad agent call
            result = ToolResult.fail(
                error=f"Invalid arguments for '{name}': {e}",
                error_code=ERR_TOOL_EXCEPTION,
                timestamp=self.state.reference_time,
            )
        except Exception as e:
            result = ToolResult.fail(
                error=f"Tool '{name}' raised: {type(e).__name__}: {e}",
                error_code=ERR_TOOL_EXCEPTION,
                timestamp=self.state.reference_time,
            )

        # 4. Log
        self._log(name, kwargs, result)
        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_available_tools(self) -> list[str]:
        """Return the list of tool names available under current access control."""
        if self._allowed is not None:
            return sorted(self._allowed & set(self._tools.keys()))
        return sorted(self._tools.keys())

    def reset_log(self) -> None:
        """Clear the action log. Call before starting a new task."""
        self.log = []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log(self, name: str, args: dict, result: ToolResult) -> None:
        data_keys: list[str] | None = None
        if result.data is not None:
            if isinstance(result.data, dict):
                data_keys = sorted(result.data.keys())
            elif isinstance(result.data, list):
                data_keys = [f"list[{len(result.data)}]"]

        entry = ActionLogEntry(
            step=len(self.log) + 1,
            tool=name,
            args=args,
            result_success=result.success,
            result_error_code=result.error_code,
            result_data_keys=data_keys,
            timestamp=self.state.reference_time,
        )
        self.log.append(entry)
