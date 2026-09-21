"""Tool result envelope and action log entry.

Every tool call returns a `ToolResult`. Every call is logged as an
`ActionLogEntry`. Together these form the evidence trail that the
evaluator and failure classifier read from.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------

ERR_NOT_FOUND = "NOT_FOUND"
ERR_INVALID_ARGS = "INVALID_ARGS"
ERR_CONFLICT = "CONFLICT"
ERR_UNAVAILABLE = "UNAVAILABLE"
ERR_UNKNOWN_TOOL = "UNKNOWN_TOOL"
ERR_TOOL_NOT_ALLOWED = "TOOL_NOT_ALLOWED"
ERR_TOOL_EXCEPTION = "TOOL_EXCEPTION"
ERR_POLICY_VIOLATION = "POLICY_VIOLATION"


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

class ToolResult(BaseModel):
    """Structured result returned by every tool call.

    Attributes:
        success: Whether the operation succeeded.
        data: The payload (dict or list). None on failure.
        error: Human-readable error message. None on success.
        error_code: Machine-readable error code (see ERR_* constants).
        timestamp: Simulation time when the tool was called.
    """
    success: bool
    data: dict | list | None = None
    error: str | None = None
    error_code: str | None = None
    timestamp: datetime

    @classmethod
    def ok(cls, data: dict | list, timestamp: datetime) -> "ToolResult":
        return cls(success=True, data=data, timestamp=timestamp)

    @classmethod
    def fail(cls, error: str, error_code: str, timestamp: datetime) -> "ToolResult":
        return cls(
            success=False,
            error=error,
            error_code=error_code,
            timestamp=timestamp,
        )


# ---------------------------------------------------------------------------
# ActionLogEntry
# ---------------------------------------------------------------------------

class ActionLogEntry(BaseModel):
    """A single tool call in an agent's trajectory.

    The list of these entries is what the evaluator iterates to compute
    tool-use metrics, unnecessary-action counts, and failure classes.
    """
    step: int                              # 1-indexed call sequence
    tool: str                              # tool name
    args: dict                             # arguments passed
    result_success: bool
    result_error_code: str | None = None
    result_data_keys: list[str] | None = None  # lightweight summary of data
    timestamp: datetime                    # simulation time


__all__ = [
    "ToolResult",
    "ActionLogEntry",
    "ERR_NOT_FOUND",
    "ERR_INVALID_ARGS",
    "ERR_CONFLICT",
    "ERR_UNAVAILABLE",
    "ERR_UNKNOWN_TOOL",
    "ERR_TOOL_NOT_ALLOWED",
    "ERR_TOOL_EXCEPTION",
    "ERR_POLICY_VIOLATION",
]
