"""
MCP server orchestrator.

Routes a parsed `JsonRpcRequest` to the right MCP method handler and
returns a `JsonRpcResponse`. Handles errors uniformly so callers (e.g.
the FastAPI route) only deal with valid response objects.
"""

from __future__ import annotations

from typing import Any

from backend.mcp.models import (
    ErrorCode,
    InitializeResult,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    ToolCallResult,
    ToolDefinition,
)
from backend.mcp.tools import ToolRegistry


class MCPServer:
    """
    Single-process MCP server.

    Stateless across requests; safe to share between FastAPI requests.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        server_name: str = "ai-github-agent",
        server_version: str = "0.1.0",
    ) -> None:
        self.registry = registry
        self.server_name = server_name
        self.server_version = server_version

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def handle(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Dispatch a parsed JSON-RPC request to the right method."""
        method = request.method
        try:
            if method == "initialize":
                return self._ok(request.id, self._initialize())
            if method == "tools/list":
                return self._ok(request.id, self._tools_list())
            if method == "tools/call":
                return self._ok(request.id, self._tools_call(request.params))
            return self._err(
                request.id,
                ErrorCode.METHOD_NOT_FOUND,
                f"Unknown method: {method}",
            )
        except ValueError as exc:
            return self._err(request.id, ErrorCode.INVALID_PARAMS, str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            return self._err(
                request.id,
                ErrorCode.INTERNAL_ERROR,
                f"Unhandled server error: {exc}",
            )

    # ------------------------------------------------------------------
    # Method handlers
    # ------------------------------------------------------------------

    def _initialize(self) -> dict[str, Any]:
        result = InitializeResult(
            server_name=self.server_name,
            server_version=self.server_version,
        )
        return result.model_dump(by_alias=True)

    def _tools_list(self) -> dict[str, Any]:
        tools = [
            ToolDefinition(
                name=t.name,
                description=t.description,
                input_schema=t.input_schema,
            ).model_dump(by_alias=True)
            for t in self.registry.list_tools()
        ]
        return {"tools": tools}

    def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = str(params.get("name", "")).strip()
        if not name:
            raise ValueError("Param 'name' is required for tools/call.")
        arguments = params.get("arguments", {}) or {}
        if not isinstance(arguments, dict):
            raise ValueError("Param 'arguments' must be an object.")

        tool = self.registry.get(name)
        if tool is None:
            # Return a structured error wrapped in a ToolCallResult so the
            # client gets a normal response with is_error=true, matching
            # the MCP convention for tool-level failures.
            return ToolCallResult(
                content=[{"type": "text", "text": f"Unknown tool: {name}"}],
                is_error=True,
            ).model_dump(by_alias=True)

        try:
            text = tool.execute(arguments)
            return ToolCallResult(
                content=[{"type": "text", "text": text}],
                is_error=False,
            ).model_dump(by_alias=True)
        except ValueError as exc:
            return ToolCallResult(
                content=[{"type": "text", "text": str(exc)}],
                is_error=True,
            ).model_dump(by_alias=True)

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ok(request_id: int | str | None, result: dict[str, Any]) -> JsonRpcResponse:
        return JsonRpcResponse(id=request_id, result=result)

    @staticmethod
    def _err(
        request_id: int | str | None,
        code: int,
        message: str,
    ) -> JsonRpcResponse:
        return JsonRpcResponse(
            id=request_id,
            error=JsonRpcError(code=code, message=message),
        )