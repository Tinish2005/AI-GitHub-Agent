"""
JSON-RPC 2.0 + MCP-shaped models.

Just enough of the protocol to expose tools to a client. Fully typed
so callers can rely on Pydantic to validate inbound payloads instead
of writing manual parsing.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PROTOCOL_VERSION: str = "2024-11-05"
JSONRPC_VERSION: Literal["2.0"] = "2.0"


# ---------------------------------------------------------------------------
# Request / response envelopes (JSON-RPC 2.0)
# ---------------------------------------------------------------------------


class JsonRpcRequest(BaseModel):
    """Inbound JSON-RPC 2.0 request."""

    jsonrpc: Literal["2.0"] = Field(default="2.0")
    id: int | str | None = Field(default=None, description="Optional client request id.")
    method: str = Field(min_length=1, description="MCP method name.")
    params: dict[str, Any] = Field(default_factory=dict)


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error object."""

    model_config = {"frozen": True}

    code: int
    message: str
    data: dict[str, Any] | None = None


class JsonRpcResponse(BaseModel):
    """Outbound JSON-RPC 2.0 response. Exactly one of `result` / `error` is set."""

    jsonrpc: Literal["2.0"] = Field(default="2.0")
    id: int | str | None = None
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None


# Standard JSON-RPC error codes (https://www.jsonrpc.org/specification#error_object)
class ErrorCode:
    PARSE_ERROR: int = -32700
    INVALID_REQUEST: int = -32600
    METHOD_NOT_FOUND: int = -32601
    INVALID_PARAMS: int = -32602
    INTERNAL_ERROR: int = -32603
    # MCP-specific (we pick our own range above -32000)
    TOOL_NOT_FOUND: int = -32001
    TOOL_EXECUTION_FAILED: int = -32002


# ---------------------------------------------------------------------------
# MCP-shaped payloads
# ---------------------------------------------------------------------------


class ToolDefinition(BaseModel):
    """Public description of a tool as returned by `tools/list`."""

    model_config = {"frozen": True}

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing accepted parameters.",
        alias="inputSchema",
    )

    model_config = {"frozen": True, "populate_by_name": True}


class InitializeResult(BaseModel):
    """Server-side handshake response."""

    model_config = {"frozen": True}

    protocol_version: str = Field(default=PROTOCOL_VERSION, alias="protocolVersion")
    server_name: str = Field(default="ai-github-agent", alias="serverName")
    server_version: str = Field(default="0.1.0", alias="serverVersion")
    capabilities: dict[str, Any] = Field(default_factory=lambda: {"tools": {}})

    model_config = {"frozen": True, "populate_by_name": True}


class ToolCallResult(BaseModel):
    """Server-side result for a `tools/call`."""

    model_config = {"frozen": True}

    content: list[dict[str, Any]] = Field(
        default_factory=list,
        description="MCP-style content blocks (e.g. [{type: 'text', text: '...'}]).",
    )
    is_error: bool = Field(default=False, alias="isError")

    model_config = {"frozen": True, "populate_by_name": True}