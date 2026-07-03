# klanex

Official Python SDK for [klanex](https://github.com/chrassy/klanex) — the tool
orchestration engine for AI agents. Fire a tool-use intent, get an
`execution_id` back in milliseconds, and let the engine own retries, backoff,
circuit breaking, credentials, and signed webhooks.

```bash
pip install klanex
```

Requires Python 3.10+. Single dependency (`httpx`); sync and async clients.
Field names match the wire format — everything is snake_case end to end.

## Submit a tool call

```python
from klanex import Klanex, KlanexSchemaError

klanex = Klanex(api_key=os.environ["KLANEX_API_KEY"],
                base_url="https://klanex-ingest-....run.app")

accepted = klanex.execute(
    target={
        "method": "POST",
        "url": "https://api.stripe.com/v1/refunds",
        "headers": {"Authorization": f"Bearer {STRIPE_KEY}"},  # encrypted at rest
    },
    payload=agent_generated_json,
    payload_schema=refund_schema,          # gate hallucinations before they queue
    callback_url="https://you.example.com/hooks/klanex",
    idempotency_key=f"refund-{charge_id}", # retries can never double-refund
)
print(accepted.execution_id, accepted.status)
```

Async is a mirror image:

```python
from klanex import AsyncKlanex

async with AsyncKlanex(api_key=..., base_url=...) as klanex:
    accepted = await klanex.execute(target=..., payload=...)
```

## The self-correction loop

When the agent hallucinates a payload, `execute` raises synchronously with a
hint written to be pasted straight back into the model's context:

```python
try:
    klanex.execute(target=target, payload=payload, payload_schema=schema)
except KlanexSchemaError as err:
    # e.g. "The JSON payload you generated does not match the required
    #       schema. Fix the following and resubmit: ..."
    messages.append({"role": "user", "content": err.llm_hint})
    return retry_with_llm(messages)
```

Failed executions carry the same shape: `execution.error.llm_hint` explains a
`TARGET_REJECTED` (4xx) so the agent can fix its payload, while retryable
failures (`TARGET_RATE_LIMITED`, `TARGET_UNAVAILABLE`, ...) never reach you —
the engine absorbs them.

## Receive results via webhook

```python
from klanex import verify_webhook, WEBHOOK_HEADERS, WebhookVerificationError

@app.post("/hooks/klanex")
async def hook(request: Request):
    try:
        event = verify_webhook(
            secret=os.environ["KLANEX_WEBHOOK_SECRET"],
            body=await request.body(),   # RAW bytes — never re-serialize
            signature=request.headers[WEBHOOK_HEADERS["signature"]],
            timestamp=request.headers[WEBHOOK_HEADERS["timestamp"]],
        )
    except WebhookVerificationError:
        return Response(status_code=400)
    # event.status is "SUCCEEDED" or "FAILED"; event.result.body holds the
    # target API's response.
    return Response(status_code=200)
```

Signature format: `sha256=` + hex HMAC-SHA256 of `"<timestamp>.<body>"` —
verified byte-for-byte compatible with the engine's Go implementation, with
replay protection via the timestamp (300s tolerance by default).

## Poll instead (scripts, tests)

```python
execution = klanex.wait_for_result(accepted.execution_id, timeout=60)
if execution.status == "FAILED":
    print(execution.error)
```

## Replay after an outage

```python
clone = klanex.replay(failed_execution_id)
```

Re-runs the byte-exact original payload with the same sealed credentials —
no re-prompting the LLM that generated it.

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```
