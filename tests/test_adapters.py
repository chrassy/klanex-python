import json

import httpx
import pytest

from klanex import Klanex
from klanex.adapters import KlanexTool, adk_tool, crewai_tool, langchain_tool

TARGET = {"url": "https://api.example.com/x", "connection_id": "con_1"}
SCHEMA = {"type": "object", "required": ["charge_id"]}


def make_client(handler):
    return Klanex(
        api_key="klx_test",
        base_url="https://ingest.example.com",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def scripted(*responses):
    """A MockTransport handler returning the given responses in order,
    repeating the last one."""
    calls = {"n": 0, "seen": []}
    seq = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls["seen"].append(request)
        i = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        status, body = seq[i]
        return httpx.Response(status, json=body)

    return handler, calls


def test_tool_forwards_payload_and_returns_result():
    handler, calls = scripted(
        (202, {"execution_id": "exe_1", "status": "QUEUED"}),
        (200, {
            "execution_id": "exe_1", "status": "SUCCEEDED", "attempts": 1,
            "max_attempts": 5, "target": TARGET,
            "result": {"status_code": 200, "body": '{"refund_id":"re_1"}'},
            "created_at": "", "updated_at": "",
        }),
    )
    tool = KlanexTool(make_client(handler), name="refund", description="Refund",
                      target=TARGET, payload_schema=SCHEMA, poll_interval=0.001)
    out = tool.run({"charge_id": "ch_1"})
    assert out == '{"refund_id":"re_1"}'

    # The submit carried the target (incl. connection_id) and payload.
    submit = json.loads(calls["seen"][0].content)
    assert submit["target"]["connection_id"] == "con_1"
    assert submit["payload"] == {"charge_id": "ch_1"}
    assert submit["payload_schema"] == SCHEMA


def test_tool_returns_llm_hint_on_schema_error():
    def handler(request):
        return httpx.Response(422, json={"error": {
            "code": "SCHEMA_INVALID", "message": "bad",
            "problems": ["missing charge_id"],
            "llm_hint": "Add charge_id and resubmit.",
        }})
    tool = KlanexTool(make_client(handler), name="t", description="d", target=TARGET)
    # The schema hint comes straight back for the agent to self-correct.
    assert tool.run({}) == "Add charge_id and resubmit."


def test_tool_returns_hint_on_terminal_failure():
    handler, _ = scripted(
        (202, {"execution_id": "exe_2", "status": "QUEUED"}),
        (200, {
            "execution_id": "exe_2", "status": "FAILED", "attempts": 1, "max_attempts": 5,
            "target": TARGET,
            "error": {"code": "TARGET_REJECTED", "message": "400",
                      "llm_hint": "Fix the auth header."},
            "created_at": "", "updated_at": "",
        }),
    )
    tool = KlanexTool(make_client(handler), name="t", description="d",
                      target=TARGET, poll_interval=0.001)
    assert tool.run({"x": 1}) == "Fix the auth header."


def test_tool_reports_pending_approval_without_blocking():
    def handler(request):
        return httpx.Response(202, json={"execution_id": "exe_3", "status": "PENDING_APPROVAL"})
    tool = KlanexTool(make_client(handler), name="t", description="d",
                      target=TARGET, requires_approval=True)
    out = tool.run({"amount": 999999})
    assert "approval" in out.lower() and "exe_3" in out


def test_requires_approval_is_sent():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(202, json={"execution_id": "exe_4", "status": "PENDING_APPROVAL"})

    tool = KlanexTool(make_client(handler), name="t", description="d",
                      target=TARGET, requires_approval=True)
    tool.run({})
    assert seen["body"].get("requires_approval") is True


def test_description_embeds_schema():
    tool = KlanexTool(make_client(lambda r: httpx.Response(202, json={})),
                      name="t", description="Base desc", target=TARGET, payload_schema=SCHEMA)
    assert "Base desc" in tool.description
    assert "charge_id" in tool.description  # schema embedded for the model


def test_string_payload_is_parsed():
    seen = {}

    def handler(request):
        path = request.url.path
        if path == "/v1/executions":
            seen["body"] = json.loads(request.content)
            return httpx.Response(202, json={"execution_id": "exe_5", "status": "QUEUED"})
        return httpx.Response(200, json={
            "execution_id": "exe_5", "status": "SUCCEEDED", "attempts": 1, "max_attempts": 5,
            "target": TARGET, "result": {"status_code": 200, "body": "ok"},
            "created_at": "", "updated_at": "",
        })

    tool = KlanexTool(make_client(handler), name="t", description="d",
                      target=TARGET, poll_interval=0.001)
    tool.run('{"charge_id": "ch_9"}')
    assert seen["body"]["payload"] == {"charge_id": "ch_9"}  # parsed from JSON string


# ── framework adapters ──────────────────────────────────────────────────

def test_langchain_adapter_produces_structured_tool():
    langchain_core = pytest.importorskip("langchain_core")  # noqa: F841
    tool = langchain_tool(
        make_client(lambda r: httpx.Response(202, json={})),
        name="refund", description="Refund a charge", target=TARGET, payload_schema=SCHEMA,
    )
    from langchain_core.tools import StructuredTool

    assert isinstance(tool, StructuredTool)
    assert tool.name == "refund"
    assert "charge_id" in tool.description


def test_missing_framework_raises_clear_error(monkeypatch):
    # Simulate CrewAI/ADK not being installed.
    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("crewai") or name.startswith("google"):
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    client = make_client(lambda r: httpx.Response(202, json={}))
    with pytest.raises(ImportError, match="pip install crewai"):
        crewai_tool(client, name="t", description="d", target=TARGET)
    with pytest.raises(ImportError, match="pip install google-adk"):
        adk_tool(client, name="t", description="d", target=TARGET)
