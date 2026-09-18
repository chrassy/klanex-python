"""Framework-agnostic core. A KlanexTool turns a fixed klanex target into an
agent tool: calling it submits the model's tool input as the payload, waits
for the managed execution, and returns a string for the model: the target's
response on success, or a correction hint / progress note otherwise, so the
agent can react instead of crashing."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..client import AsyncKlanex, Klanex
from ..errors import KlanexError, KlanexSchemaError
from ..types import ExecuteResponse, Execution

#: Tool input schema when no payload_schema is given: any JSON object.
ANY_OBJECT: dict[str, Any] = {"type": "object", "additionalProperties": True}

_MAX_IDEMPOTENCY_KEY = 255


class KlanexTool:
    """A klanex-backed tool, independent of any agent framework.

    The model's tool input (a JSON object) is forwarded as the payload to a
    fixed ``target`` (URL + method, plus either a vault ``connection_id`` or
    ``headers``); the model never sees the URL or credentials. Reliability
    (schema gate, retries, backoff, approvals, credentials) is handled by
    klanex, not the agent.

    ``client`` may be a :class:`Klanex` or an :class:`AsyncKlanex`; async
    frameworks call :meth:`arun`, which runs a sync client in a thread.
    """

    def __init__(
        self,
        client: Klanex | AsyncKlanex,
        *,
        name: str,
        description: str,
        target: dict[str, Any],
        payload_schema: dict[str, Any] | None = None,
        requires_approval: bool = False,
        max_attempts: int | None = None,
        wait_timeout: float = 120.0,
        poll_interval: float = 2.0,
        idempotency: bool = True,
    ) -> None:
        self.client = client
        self.name = name
        self.description = description
        self.target = target
        self.payload_schema = payload_schema
        self.requires_approval = requires_approval
        self.max_attempts = max_attempts
        self.wait_timeout = wait_timeout
        self.poll_interval = poll_interval
        self.idempotency = idempotency

    @property
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema of the tool input, for frameworks that take one."""
        return self.payload_schema or ANY_OBJECT

    @property
    def description_with_schema(self) -> str:
        """The description plus the payload schema, for frameworks whose tool
        takes a single ``payload`` argument instead of a native schema."""
        if not self.payload_schema:
            return self.description
        return (
            f"{self.description}\n\nThe `payload` argument must be a JSON object "
            f"matching this schema:\n{json.dumps(self.payload_schema)}"
        )

    def run(self, payload: Any, *, tool_call_id: str | None = None) -> str:
        """Execute the tool with the model's input and return a string for the
        model's context. Requires a sync :class:`Klanex` client."""
        if not isinstance(self.client, Klanex):
            raise TypeError("KlanexTool.run needs a Klanex client; use arun with AsyncKlanex")
        try:
            accepted = self.client.execute(**self._submit(payload, tool_call_id))
        except KlanexSchemaError as err:
            return _schema_hint(err)
        if accepted.status == "PENDING_APPROVAL":
            return _pending(accepted)
        try:
            execution = self.client.wait_for_result(
                accepted.execution_id, poll_interval=self.poll_interval, timeout=self.wait_timeout
            )
        except KlanexError as err:
            if err.code == "WAIT_TIMEOUT":
                return _still_running(accepted)
            raise
        return _outcome(execution)

    async def arun(self, payload: Any, *, tool_call_id: str | None = None) -> str:
        """Async :meth:`run`. Works with either client type."""
        if isinstance(self.client, Klanex):
            return await asyncio.to_thread(self.run, payload, tool_call_id=tool_call_id)
        try:
            accepted = await self.client.execute(**self._submit(payload, tool_call_id))
        except KlanexSchemaError as err:
            return _schema_hint(err)
        if accepted.status == "PENDING_APPROVAL":
            return _pending(accepted)
        try:
            execution = await self.client.wait_for_result(
                accepted.execution_id, poll_interval=self.poll_interval, timeout=self.wait_timeout
            )
        except KlanexError as err:
            if err.code == "WAIT_TIMEOUT":
                return _still_running(accepted)
            raise
        return _outcome(execution)

    def _submit(self, payload: Any, tool_call_id: str | None) -> dict[str, Any]:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                pass  # forward as-is; the schema gate will explain if wrong
        key = None
        if self.idempotency and tool_call_id:
            # The framework's tool call ID makes a re-run of the same call
            # (a resumed or retried agent step) execute only once.
            key = f"tool:{self.name}:{tool_call_id}"[-_MAX_IDEMPOTENCY_KEY:]
        return {
            "target": self.target,
            "payload": payload,
            "payload_schema": self.payload_schema,
            "max_attempts": self.max_attempts,
            "requires_approval": self.requires_approval,
            "idempotency_key": key,
        }


def _schema_hint(err: KlanexSchemaError) -> str:
    # The self-correction loop: hand the hint straight back to the model.
    return err.llm_hint or f"Invalid input: {err}"


def _pending(accepted: ExecuteResponse) -> str:
    return (
        f"Submitted for human approval (execution {accepted.execution_id}). "
        "The action runs once a reviewer approves it; do not call this tool again for it."
    )


def _still_running(accepted: ExecuteResponse) -> str:
    return (
        f"The action was accepted and is still running (execution {accepted.execution_id}); "
        "klanex keeps retrying it in the background. Do not call this tool again for the "
        "same action, or it may be performed twice. Tell the user it is in progress."
    )


def _outcome(execution: Execution) -> str:
    if execution.status == "SUCCEEDED":
        result = execution.result
        if result is None:
            return "ok"
        body = result.body or f"(empty response, HTTP {result.status_code})"
        return f"{result.note}\n\nTarget response: {body}" if result.note else body
    error = execution.error
    if error and error.llm_hint:
        return error.llm_hint
    if error:
        return f"The tool call failed ({error.code}): {error.message}"
    return "The tool call failed: no details"
