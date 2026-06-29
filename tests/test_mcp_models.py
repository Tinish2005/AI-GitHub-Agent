"""Unit tests for `backend.mcp.models`."""

from __future__ import annotations

import pytest

from backend.mcp.models import (
    JSONRPC_VERSION,
    PROTOCOL_VERSION,
    ErrorCode,
    InitializeResult,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    ToolCallResult,
    ToolDefinition,
)


def test_jsonrpc_version_constant() -> None:
    assert JSONRPC_VERSION == "2.0"


def test_protocol_version_is_set() -> None:
    assert PROTOCOL_VERSION


def test_request_defaults() -> None:
    req = JsonRpcRequest(method="ping")
    assert req.jsonrpc == "2.0"
    assert req.params == {}
    assert req.id is None


def test_request_rejects_empty_method() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        JsonRpcRequest(method="")


def test_response_can_carry_result_or_error() -> None:
    ok = JsonRpcResponse(id=1, result={"ok": True})
    assert ok.result == {"ok": True}
    assert ok.error is None

    bad = JsonRpcResponse(id=1, error=JsonRpcError(code=-32601, message="nope"))
    assert bad.error is not None
    assert bad.error.code == ErrorCode.METHOD_NOT_FOUND


def test_initialize_result_serializes_with_camel_case_aliases() -> None:
    result = InitializeResult()
    dumped = result.model_dump(by_alias=True)
    assert dumped["protocolVersion"] == PROTOCOL_VERSION
    assert dumped["serverName"] == "ai-github-agent"
    assert dumped["serverVersion"]


def test_tool_definition_serializes_with_input_schema_alias() -> None:
    td = ToolDefinition(
        name="t",
        description="d",
        input_schema={"type": "object"},
    )
    dumped = td.model_dump(by_alias=True)
    assert dumped["name"] == "t"
    assert "inputSchema" in dumped


def test_tool_call_result_default_is_not_error() -> None:
    r = ToolCallResult(content=[{"type": "text", "text": "ok"}])
    dumped = r.model_dump(by_alias=True)
    assert dumped["isError"] is False