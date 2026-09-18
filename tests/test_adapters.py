import json

import httpx
import pytest

from klanex import Klanex
from klanex.adapters import KlanexTool, adk_tool, crewai_tool, langchain_tool, openai_agents_tool

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


def test_schema_is_native_and_embedded_only_where_needed():
    tool = KlanexTool(make_client(lambda r: httpx.Response(202, json={})),
                      name="t", description="Base desc", target=TARGET, payload_schema=SCHEMA)
    assert tool.description == "Base desc"
    assert tool.input_schema == SCHEMA
    # CrewAI/ADK take a single `payload` argument, so they get it in the text.
    assert "charge_id" in tool.description_with_schema
    bare = KlanexTool(make_client(lambda r: httpx.Response(202, json={})),
                      name="t", description="d", target=TARGET)
    assert bare.input_schema == {"type": "object", "additionalProperties": True}


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


# ── core behavior shared by every adapter ───────────────────────────────

def succeeded(execution_id="exe_1", **result):
    return (200, {
        "execution_id": execution_id, "status": "SUCCEEDED", "attempts": 1, "max_attempts": 5,
        "target": TARGET, "result": {"status_code": 200, "body": "ok", **result},
        "created_at": "", "updated_at": "",
    })


QUEUED = (202, {"execution_id": "exe_1", "status": "QUEUED"})


def test_tool_call_id_becomes_the_idempotency_key():
    handler, calls = scripted(QUEUED, succeeded())
    tool = KlanexTool(make_client(handler), name="refund", description="d",
                      target=TARGET, poll_interval=0.001)
    tool.run({}, tool_call_id="call_7")
    assert json.loads(calls["seen"][0].content)["idempotency_key"] == "tool:refund:call_7"

    handler, calls = scripted(QUEUED, succeeded())
    off = KlanexTool(make_client(handler), name="refund", description="d",
                     target=TARGET, poll_interval=0.001, idempotency=False)
    off.run({}, tool_call_id="call_7")
    assert "idempotency_key" not in json.loads(calls["seen"][0].content)


def test_wait_timeout_tells_the_model_not_to_resubmit():
    handler, _ = scripted(QUEUED, (200, {
        "execution_id": "exe_1", "status": "RETRYING", "attempts": 1, "max_attempts": 5,
        "target": TARGET, "created_at": "", "updated_at": "",
    }))
    tool = KlanexTool(make_client(handler), name="t", description="d", target=TARGET,
                      poll_interval=0.001, wait_timeout=0.005)
    out = tool.run({})
    assert "still running (execution exe_1)" in out
    assert "Do not call this tool again" in out


def test_duplicate_success_note_is_shown():
    handler, _ = scripted(
        QUEUED, succeeded(status_code=409, body="exists", note="Attempt 1 did it.")
    )
    tool = KlanexTool(make_client(handler), name="t", description="d",
                      target=TARGET, poll_interval=0.001)
    assert tool.run({}) == "Attempt 1 did it.\n\nTarget response: exists"


def test_arun_with_async_client():
    import asyncio

    from klanex import AsyncKlanex

    handler, calls = scripted(QUEUED, succeeded(body="done"))
    client = AsyncKlanex(api_key="klx_test", base_url="https://ingest.example.com",
                         client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    tool = KlanexTool(client, name="t", description="d", target=TARGET, poll_interval=0.001)
    assert asyncio.run(tool.arun({"a": 1}, tool_call_id="c1")) == "done"
    with pytest.raises(TypeError, match="use arun"):
        tool.run({})


# ── LangGraph / LangChain ───────────────────────────────────────────────

def _graph_with(tool):
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode([tool]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return graph.compile()


def _tool_call(args):
    from langchain_core.messages import AIMessage

    return {"messages": [AIMessage(content="", tool_calls=[
        {"name": "refund", "args": args, "id": "call_lg", "type": "tool_call"},
    ])]}


def test_langgraph_tool_node_runs_the_tool():
    pytest.importorskip("langgraph")
    from langchain_core.utils.function_calling import convert_to_openai_tool

    handler, calls = scripted(QUEUED, succeeded(body='{"refund_id":"re_1"}'))
    tool = langchain_tool(make_client(handler), name="refund", description="Refund a charge",
                          target=TARGET, payload_schema=SCHEMA, poll_interval=0.001)
    # The model sees the payload schema as the tool's own parameters.
    assert convert_to_openai_tool(tool)["function"]["parameters"] == SCHEMA

    out = _graph_with(tool).invoke(_tool_call({"charge_id": "ch_1"}))
    message = out["messages"][-1]
    assert message.content == '{"refund_id":"re_1"}' and message.tool_call_id == "call_lg"
    submit = json.loads(calls["seen"][0].content)
    assert submit["payload"] == {"charge_id": "ch_1"}  # arguments are the payload
    assert submit["idempotency_key"] == "tool:refund:call_lg"


def test_langgraph_async_graph_with_async_client():
    pytest.importorskip("langgraph")
    import asyncio

    from klanex import AsyncKlanex

    handler, calls = scripted(QUEUED, succeeded(body="done"))
    client = AsyncKlanex(api_key="klx_test", base_url="https://ingest.example.com",
                         client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    tool = langchain_tool(client, name="refund", description="Refund", target=TARGET,
                          payload_schema=SCHEMA, poll_interval=0.001)
    out = asyncio.run(_graph_with(tool).ainvoke(_tool_call({"charge_id": "ch_2"})))
    assert out["messages"][-1].content == "done"
    assert json.loads(calls["seen"][0].content)["idempotency_key"] == "tool:refund:call_lg"


# ── OpenAI Agents SDK ───────────────────────────────────────────────────

def test_openai_agents_tool_invocation():
    pytest.importorskip("agents")
    import asyncio

    from agents import FunctionTool
    from agents.tool_context import ToolContext

    handler, calls = scripted(QUEUED, succeeded(body="refunded"))
    tool = openai_agents_tool(make_client(handler), name="refund", description="Refund",
                              target=TARGET, payload_schema=SCHEMA, poll_interval=0.001)
    assert isinstance(tool, FunctionTool)
    assert tool.params_json_schema == SCHEMA and tool.strict_json_schema is False

    ctx = ToolContext(context=None, tool_name="refund", tool_call_id="call_oa",
                      tool_arguments='{"charge_id": "ch_3"}')
    out = asyncio.run(tool.on_invoke_tool(ctx, '{"charge_id": "ch_3"}'))
    assert out == "refunded"
    submit = json.loads(calls["seen"][0].content)
    assert submit["payload"] == {"charge_id": "ch_3"}
    assert submit["payload_schema"] == SCHEMA
    assert submit["idempotency_key"] == "tool:refund:call_oa"
