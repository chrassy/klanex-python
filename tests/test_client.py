import asyncio
import json

import httpx
import pytest

from klanex import AsyncKlanex, Klanex, KlanexError, KlanexSchemaError

BASE = "https://ingest.example.com"


def make_client(handler):
    transport = httpx.MockTransport(handler)
    return Klanex(
        api_key="klx_test", base_url=BASE + "/", client=httpx.Client(transport=transport)
    )


def test_execute_sends_wire_format_and_returns_ids():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["api_key"] = request.headers["X-API-Key"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(202, json={"execution_id": "exe_1", "status": "QUEUED"})

    accepted = make_client(handler).execute(
        target={
            "method": "POST",
            "url": "https://api.example.com/refunds",
            "headers": {"Authorization": "Bearer sk"},
            "timeout_ms": 10000,
        },
        payload={"charge_id": "ch_1", "amount": 100},
        payload_schema={"type": "object"},
        callback_url="https://me.example.com/hook",
        max_attempts=5,
        idempotency_key="refund-ch_1",
    )

    assert accepted.execution_id == "exe_1"
    assert accepted.status == "QUEUED"
    assert accepted.idempotent_replay is False
    assert seen["url"] == f"{BASE}/v1/executions"
    assert seen["api_key"] == "klx_test"
    assert seen["body"]["target"]["timeout_ms"] == 10000
    assert seen["body"]["payload_schema"] == {"type": "object"}
    assert seen["body"]["idempotency_key"] == "refund-ch_1"
    assert "callback_url" in seen["body"] and "max_attempts" in seen["body"]


def test_execute_flags_idempotent_replay():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"execution_id": "exe_1", "status": "SUCCEEDED"},
            headers={"X-Klanex-Idempotent-Replay": "true"},
        )

    accepted = make_client(handler).execute(target={"url": "https://x.example.com"})
    assert accepted.idempotent_replay is True


def test_execute_raises_schema_error_with_llm_hint():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "error": {
                    "code": "SCHEMA_INVALID",
                    "message": "payload does not match payload_schema",
                    "problems": ["at \"/\": missing properties: 'charge_id'"],
                    "llm_hint": "Fix the following and resubmit:\n- missing charge_id",
                }
            },
        )

    with pytest.raises(KlanexSchemaError) as excinfo:
        make_client(handler).execute(target={"url": "https://x.example.com"}, payload={})
    assert "missing charge_id" in (excinfo.value.llm_hint or "")
    assert len(excinfo.value.problems or []) == 1


def test_execute_raises_klanex_error_with_status_and_code():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"error": {"code": "UNAUTHENTICATED", "message": "invalid API key"}}
        )

    with pytest.raises(KlanexError) as excinfo:
        make_client(handler).execute(target={"url": "https://x.example.com"})
    assert excinfo.value.status == 401
    assert excinfo.value.code == "UNAUTHENTICATED"


def test_get_parses_execution():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "execution_id": "exe_1",
                "status": "SUCCEEDED",
                "attempts": 2,
                "max_attempts": 5,
                "target": {"method": "POST", "url": "https://api.example.com/x"},
                "result": {"status_code": 200, "body": '{"ok":true}'},
                "replay_of": "exe_0",
                "created_at": "2026-07-03T12:00:00Z",
                "updated_at": "2026-07-03T12:00:05Z",
            },
        )

    execution = make_client(handler).get("exe_1")
    assert execution.terminal
    assert execution.result is not None and execution.result.status_code == 200
    assert execution.replay_of == "exe_0"


def test_replay_returns_clone_with_origin():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            202, json={"execution_id": "exe_2", "status": "QUEUED", "replay_of": "exe_1"}
        )

    replayed = make_client(handler).replay("exe_1")
    assert replayed.execution_id == "exe_2"
    assert replayed.replay_of == "exe_1"
    assert seen["url"] == f"{BASE}/v1/executions/exe_1/replay"


def test_wait_for_result_polls_until_terminal():
    calls = {"n": 0}
    pending = {
        "execution_id": "exe_1",
        "status": "RETRYING",
        "attempts": 1,
        "max_attempts": 5,
        "target": {"url": "https://x.example.com"},
    }

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(200, json=pending)
        return httpx.Response(200, json={**pending, "status": "SUCCEEDED", "attempts": 2})

    execution = make_client(handler).wait_for_result("exe_1", poll_interval=0.001)
    assert execution.status == "SUCCEEDED"
    assert calls["n"] == 2


def test_wait_for_result_times_out():
    pending = {
        "execution_id": "exe_1",
        "status": "QUEUED",
        "attempts": 0,
        "max_attempts": 5,
        "target": {"url": "https://x.example.com"},
    }

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=pending)

    with pytest.raises(KlanexError) as excinfo:
        make_client(handler).wait_for_result("exe_1", poll_interval=0.005, timeout=0.001)
    assert excinfo.value.code == "WAIT_TIMEOUT"


def test_async_client_round_trip():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/replay"):
            return httpx.Response(
                202, json={"execution_id": "exe_2", "status": "QUEUED", "replay_of": "exe_1"}
            )
        return httpx.Response(202, json={"execution_id": "exe_1", "status": "QUEUED"})

    async def run():
        transport = httpx.MockTransport(handler)
        async with AsyncKlanex(
            api_key="klx_test",
            base_url=BASE,
            client=httpx.AsyncClient(transport=transport),
        ) as klanex:
            accepted = await klanex.execute(target={"url": "https://x.example.com"})
            replayed = await klanex.replay(accepted.execution_id)
            return accepted, replayed

    accepted, replayed = asyncio.run(run())
    assert accepted.execution_id == "exe_1"
    assert replayed.replay_of == "exe_1"
