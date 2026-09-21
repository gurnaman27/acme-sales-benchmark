"""LLM-based participant agent with pluggable providers.

The agent runs a ReAct-style loop:
  1. Send the task + tool descriptions + available tools to the LLM.
  2. Parse the response:
       - "TOOL_CALL: {...}" → execute tool, append result, loop
       - "FINAL: ..."       → return the message, exit
  3. Stop after max_steps iterations.

The provider abstraction means the same agent works with OpenAI
or DeepSeek — only the provider implementation differs.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from acme.agents.agent_interface import ParticipantAgent
from acme.tasks.task_schema import BenchmarkTask
from acme.tools.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class LLMProvider(Protocol):
    """Protocol for an LLM backend. Only one method is required."""

    name: str

    def generate(self, messages: list[dict]) -> str:
        """Send messages, return the model's text response.

        messages: list of {"role": "system" | "user" | "assistant",
                          "content": "..."}
        """
        ...


# ---------------------------------------------------------------------------
# Tool documentation for the LLM
# ---------------------------------------------------------------------------

TOOL_DOCS = {
    "get_customer": {
        "description": "Get a customer record by ID.",
        "args": {"customer_id": "str, e.g. 'C001'"},
    },
    "search_customers": {
        "description": "Search customers by industry, tier, or status.",
        "args": {"industry": "str (optional)", "tier": "str (optional)", "status": "str (optional)"},
    },
    "get_customer_history": {
        "description": "Get full interaction history for a customer.",
        "args": {"customer_id": "str"},
    },
    "update_customer": {
        "description": "Update a customer record.",
        "args": {"customer_id": "str", "updates": "dict of fields to update, e.g. {'notes': 'value'}"},
    },
    "get_product": {
        "description": "Get a product by ID.",
        "args": {"product_id": "str, e.g. 'PROD-001'"},
    },
    "search_products": {
        "description": "Search products.",
        "args": {},
    },
    "check_product_eligibility": {
        "description": "Check if a customer is eligible to buy a product.",
        "args": {"customer_id": "str", "product_id": "str"},
    },
    "search_opportunities": {
        "description": "Search sales opportunities. Filter by customer_id or limit.",
        "args": {"customer_id": "str (optional)", "limit": "int (optional)"},
    },
    "get_opportunity": {
        "description": "Get a single opportunity by ID.",
        "args": {"opportunity_id": "str"},
    },
    "create_opportunity": {
        "description": "Create a new sales opportunity.",
        "args": {
            "customer_id": "str",
            "product_ids": "list of str, e.g. ['PROD-001']",
            "stage": "str: 'prospecting'|'qualification'|'proposal'|'negotiation'",
        },
    },
    "update_opportunity": {
        "description": "Update an existing opportunity.",
        "args": {
            "opportunity_id": "str",
            "updates": "dict of fields to update, e.g. {'stage': 'negotiation', 'next_action': '...'}",
        },
    },
    "get_current_time": {
        "description": "Get the current simulation date and day of the week.",
        "args": {},
    },
    "get_available_slots": {
        "description": "Get available meeting slots for a rep on a date range.",
        "args": {"rep_id": "str", "start_date": "str YYYY-MM-DD", "end_date": "str YYYY-MM-DD"},
    },
    "schedule_meeting": {
        "description": "Schedule a meeting between a rep and a customer.",
        "args": {
            "customer_id": "str",
            "rep_id": "str",
            "start_time": "str ISO8601, e.g. '2025-03-25T18:00:00+00:00'",
            "duration_minutes": "int",
        },
    },
    "cancel_meeting": {
        "description": "Cancel a scheduled meeting.",
        "args": {"meeting_id": "str", "reason": "str"},
    },
    "get_meetings": {
        "description": "List meetings for a rep or customer.",
        "args": {"rep_id": "str (optional)", "customer_id": "str (optional)"},
    },
    "create_followup": {
        "description": "Create a follow-up task for a customer.",
        "args": {"customer_id": "str", "action": "str description", "due_date": "str YYYY-MM-DD"},
    },
    "get_followups": {
        "description": "List follow-up tasks. Optionally filter for overdue.",
        "args": {"customer_id": "str (optional)", "overdue_only": "bool (optional)"},
    },
    "complete_followup": {
        "description": "Mark a follow-up as completed.",
        "args": {"followup_id": "str"},
    },
    "get_policies": {
        "description": "Get business policies. Filter by category: 'discount', 'scheduling', 'assignment', 'eligibility'.",
        "args": {"category": "str (optional)"},
    },
    "check_policy": {
        "description": "Check if a specific action complies with a policy.",
        "args": {"policy_id": "str", "action": "dict describing the action"},
    },
}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an enterprise sales assistant for Acme Sales Corporation.

You interact with a simulated CRM environment exclusively through TOOLS.
At each step, you must output EXACTLY ONE of:

  TOOL_CALL: {{"name": "<tool_name>", "args": {{<arguments as JSON>}}}}
  FINAL: <your final answer>

Rules:
- Output a TOOL_CALL to invoke a tool. You will receive the result.
- Output FINAL when the task is complete.
- Do not explain your reasoning in the tool call line.
- Use the exact tool names listed below.
- Do not invent tools or entity IDs.
- When a task says to refuse/decline something, output FINAL with your refusal after checking the relevant policy.

Simulation time is 2025-03-15 10:00 UTC.

Available tools:
{tool_list}
"""


def _build_tool_list(registry: ToolRegistry) -> str:
    """Rich tool descriptions with parameter info for the LLM."""
    names = registry.get_available_tools()
    lines = []
    for name in names:
        doc = TOOL_DOCS.get(name, {})
        desc = doc.get("description", "No description.")
        args = doc.get("args", {})
        if args:
            arg_str = ", ".join(f"{k}: {v}" for k, v in args.items())
            lines.append(f"- {name}({arg_str})\n    {desc}")
        else:
            lines.append(f"- {name}()\n    {desc}")
    return "\n".join(lines)


def _build_user_prompt(task: BenchmarkTask) -> str:
    return f"TASK: {task.instruction}"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_FINAL_RE = re.compile(r"^\s*FINAL:\s*(.*)$", re.DOTALL | re.IGNORECASE)
_TOOL_RE = re.compile(r"^\s*TOOL_CALL:\s*(\{.*\})\s*$", re.DOTALL | re.IGNORECASE)


def _parse_response(text: str) -> tuple[str, str | dict | None]:
    """Return ("final", message) or ("tool", {name, args}) or ("error", msg)."""
    text = text.strip()

    # Handle multi-line responses — look for FINAL or TOOL_CALL on any line
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        m_final = _FINAL_RE.match(line)
        if m_final:
            return "final", m_final.group(1).strip()

        m_tool = _TOOL_RE.match(line)
        if m_tool:
            try:
                payload = json.loads(m_tool.group(1))
            except json.JSONDecodeError as e:
                return "error", f"Invalid JSON in TOOL_CALL: {e}"

            if "name" not in payload:
                return "error", "TOOL_CALL missing 'name'"
            return "tool", {"name": payload["name"], "args": payload.get("args", {})}

    return "error", (
        "Response did not match 'TOOL_CALL: {...}' or 'FINAL: ...'. "
        f"Raw text: {text[:200]!r}"
    )


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class LLMAgent(ParticipantAgent):
    """An LLM-backed agent using a provider (OpenAI/DeepSeek)."""

    def __init__(
        self,
        provider: LLMProvider,
        max_steps: int = 20,
        verbose: bool = False,
    ):
        self.provider = provider
        self.max_steps = max_steps
        self.verbose = verbose

    @property
    def name(self) -> str:
        return f"llm_{self.provider.name}"

    def run(self, task: BenchmarkTask, registry: ToolRegistry) -> str:
        messages: list[dict] = [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT.format(
                    tool_list=_build_tool_list(registry),
                ),
            },
            {"role": "user", "content": _build_user_prompt(task)},
        ]

        for step in range(self.max_steps):
            if self.verbose:
                print(f"[step {step + 1}] calling {self.provider.name}")

            try:
                raw = self.provider.generate(messages)
            except Exception as e:
                return f"[provider error: {type(e).__name__}: {e}]"

            if self.verbose:
                print(f"[response] {raw[:300]}")

            kind, payload = _parse_response(raw)

            if kind == "final":
                return payload  # type: ignore[return-value]

            if kind == "error":
                # Feed the error back so the model can self-correct
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Parse error: {payload}. "
                        "Respond with exactly one line: "
                        "'TOOL_CALL: {\"name\": \"...\", \"args\": {...}}' "
                        "or 'FINAL: ...'."
                    ),
                })
                continue

            # kind == "tool"
            tool_name = payload["name"]              # type: ignore[index]
            tool_args = payload["args"]              # type: ignore[index]

            tool_result = registry.call(tool_name, **tool_args)

            summary = _summarize_result(tool_result)
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"Tool result: {summary}",
            })

        return f"[max steps ({self.max_steps}) exceeded]"


# ---------------------------------------------------------------------------
# Result summarization
# ---------------------------------------------------------------------------

def _summarize_result(result) -> str:
    """Compact string version of a ToolResult for the LLM."""
    if not result.success:
        return f"ERROR [{result.error_code}]: {result.error}"

    if result.data is None:
        return "OK (no data)"

    try:
        return "OK: " + json.dumps(result.data, default=str)[:1500]
    except Exception:
        return "OK (unserializable)"


__all__ = ["LLMAgent", "LLMProvider"]