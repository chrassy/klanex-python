# klanex

Official Python SDK for [klanex](https://klanexai.com) — the tool
orchestration engine for AI agents. Fire a tool-use intent, get an
`execution_id` back in milliseconds, and let the engine own retries, backoff,
circuit breaking, credentials, and signed webhooks.

```bash
pip install klanex
```

Requires Python 3.10+. Single dependency (`httpx`); sync and async clients.
Field names match the wire format — everything is snake_case end to end.

> Building in TypeScript/Node? See the [TypeScript SDK](https://github.com/chrassy/klanex-js).

## Submit a tool call

```python
from klanex import Klanex, KlanexSchemaError

klanex = Klanex(api_key=os.environ["KLANEX_API_KEY"])  # https://api.klanexai.com

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

async with AsyncKlanex(api_key=...) as klanex:
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

## Rotate credentials

```python
# Old key stops working immediately; this client switches to the new one.
new = klanex.rotate_api_key()
print(new.api_key)

# Callbacks after this are signed with the new secret — update your verifier.
rotated = klanex.rotate_webhook_secret()
print(rotated.webhook_secret)
```

Each secret is returned only once. Both methods exist on `AsyncKlanex` too.
If other processes share the key, persist the value from `rotate_api_key()`.

## Agent framework adapters

Wrap an API call as a native tool for **LangGraph / LangChain**, the
**OpenAI Agents SDK**, CrewAI, or Google ADK. The model's tool arguments
become the request payload; klanex owns the call's reliability (schema gate,
retries with backoff, circuit breakers, approvals, credentials) and the tool
returns text the model can act on:

- **Success:** the target's response.
- **Rejected:** the `llm_hint`, which names the bad field when klanex can
  tell, so the model fixes one value and calls again.
- **Still running** after `wait_timeout` (default 120 s), or **waiting for
  approval:** a note telling the model the action is in progress and not to
  call the tool again, so a slow API never turns into a duplicate charge.
- **Exactly once per tool call:** the framework's tool call ID becomes the
  idempotency key, so a resumed or retried agent step never runs the action
  twice. Opt out with `idempotency=False`.

The model never sees the target URL or credentials. Neither LangChain nor
the Agents SDK validates a plain JSON Schema, so klanex's schema gate does,
and a failing input comes back to the model as a correction hint.

### LangGraph / LangChain

```bash
pip install "klanex[langgraph]" langchain   # langchain for create_agent
```

```python
from langchain.agents import create_agent
from klanex import AsyncKlanex
from klanex.adapters import langchain_tool

klanex = AsyncKlanex(api_key=...)   # a sync Klanex works too

refund = langchain_tool(
    klanex,
    name="create_refund",
    description="Refund a Stripe charge",
    target={"url": "https://api.stripe.com/v1/refunds",
            "connection_id": "con_..."},   # vault-managed credential
    payload_schema={
        "type": "object",
        "properties": {"charge": {"type": "string"}, "amount": {"type": "integer"}},
        "required": ["charge", "amount"],
    },
    requires_approval=True,                # pause for a human in Slack first
)

agent = create_agent(model, tools=[refund])
await agent.ainvoke({"messages": [("user", "Refund charge ch_123 in full")]})
```

`payload_schema` is the tool's own parameter schema, so the model fills in
`charge` and `amount` directly. The tool works in LangGraph's `ToolNode`,
`create_agent`, and custom graphs, sync or async.

### OpenAI Agents SDK

```bash
pip install "klanex[openai-agents]"
```

```python
from agents import Agent, Runner
from klanex import AsyncKlanex
from klanex.adapters import openai_agents_tool

refund = openai_agents_tool(
    AsyncKlanex(api_key=...),
    name="create_refund",
    description="Refund a Stripe charge",
    target={"url": "https://api.stripe.com/v1/refunds", "connection_id": "con_..."},
    payload_schema={...},   # same JSON Schema as above
)

agent = Agent(name="Support", instructions="...", tools=[refund])
result = await Runner.run(agent, "Refund charge ch_123 in full")
```

Pass `strict=True` for OpenAI strict mode if your schema meets its rules
(every property required, `additionalProperties: false`).

### CrewAI and Google ADK

The same arguments produce a CrewAI `BaseTool` (`crewai_tool`) or a Google
ADK `FunctionTool` (`adk_tool`). These take a single `payload` argument with
the schema described in the tool description. Frameworks are imported
lazily, so the base `klanex` install stays dependency-light.

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```
