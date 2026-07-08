"""Framework-agnostic core. A KlanexTool turns a fixed klanex target into an
agent tool: calling it submits the agent's payload, waits for the managed
execution, and returns a string for the model — the target's response on
success, or a correction hint / error description on failure (so the agent
can react instead of crashing)."""

from __future__ import annotations

import json
from typing import Any

from ..client import Klanex
from ..errors import KlanexSchemaError


class KlanexTool:
    """A klanex-backed tool, independent of any agent framework.

    The agent supplies a JSON ``payload`` that is forwarded to a fixed
    ``target`` (URL + method, plus either a vault ``connection_id`` or
    ``headers``). Reliability — retries, backoff, approvals, credentials —
    is handled by klanex, not the agent.
    """

    def __init__(
        self,
        client: Klanex,
        *,
        name: str,
        description: str,
        target: dict[str, Any],
        payload_schema: dict[str, Any] | None = None,
        requires_approval: bool = False,
        max_attempts: int | None = None,
        wait_timeout: float = 120.0,
        poll_interval: float = 2.0,
    ) -> None:
        self.client = client
        self.name = name
        self.target = target
        self.payload_schema = payload_schema
        self.requires_approval = requires_approval
        self.max_attempts = max_attempts
        self.wait_timeout = wait_timeout
        self.poll_interval = poll_interval
        self.description = self._describe(description, payload_schema)

    @staticmethod
    def _describe(description: str, schema: dict[str, Any] | None) -> str:
        if schema:
            return (
                f"{description}\n\nThe `payload` argument must be a JSON object "
                f"matching this schema:\n{json.dumps(schema)}"
            )
        return description

    def run(self, payload: Any) -> str:
        """Execute the tool with the agent-supplied payload and return a
        string suitable for the model's context."""
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                pass  # forward as-is; the schema gate will explain if wrong

        try:
            accepted = self.client.execute(
                target=self.target,
                payload=payload,
                payload_schema=self.payload_schema,
                max_attempts=self.max_attempts,
                requires_approval=self.requires_approval,
            )
        except KlanexSchemaError as err:
            # The self-correction loop: hand the hint straight back to the model.
            return err.llm_hint or f"Invalid payload: {err}"

        if accepted.status == "PENDING_APPROVAL":
            return (
                f"Submitted for human approval (execution {accepted.execution_id}). "
                "The action will run once a reviewer approves it; do not resubmit."
            )

        execution = self.client.wait_for_result(
            accepted.execution_id,
            poll_interval=self.poll_interval,
            timeout=self.wait_timeout,
        )
        if execution.status == "SUCCEEDED":
            return execution.result.body if execution.result else "ok"

        # Failed: surface the correction hint if there is one, else the message.
        if execution.error and execution.error.llm_hint:
            return execution.error.llm_hint
        msg = execution.error.message if execution.error else "unknown error"
        return f"The tool call failed: {msg}"
