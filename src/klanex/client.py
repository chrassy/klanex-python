"""Sync and async clients for the klanex ingest API."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from .errors import KlanexError, KlanexSchemaError
from .types import (
    ExecuteResponse,
    Execution,
    ReplayResponse,
    RotateApiKeyResponse,
    RotateWebhookSecretResponse,
    execution_from_wire,
)

_REPLAY_HEADER = "X-Klanex-Idempotent-Replay"


def _build_body(
    target: dict[str, Any],
    payload: Any,
    payload_schema: Any,
    callback_url: str | None,
    max_attempts: int | None,
    idempotency_key: str | None,
    requires_approval: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {"target": target}
    if requires_approval:
        body["requires_approval"] = True
    if payload is not None:
        body["payload"] = payload
    if payload_schema is not None:
        body["payload_schema"] = payload_schema
    if callback_url is not None:
        body["callback_url"] = callback_url
    if max_attempts is not None:
        body["max_attempts"] = max_attempts
    if idempotency_key is not None:
        body["idempotency_key"] = idempotency_key
    return body


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    code = "UNKNOWN"
    message = f"klanex API returned {response.status_code}"
    problems: list[str] | None = None
    llm_hint: str | None = None
    try:
        error = response.json().get("error", {})
        code = error.get("code", code)
        message = error.get("message", message)
        problems = error.get("problems")
        llm_hint = error.get("llm_hint")
    except ValueError:
        pass
    if response.status_code == 422 and code == "SCHEMA_INVALID":
        raise KlanexSchemaError(message, problems=problems, llm_hint=llm_hint)
    raise KlanexError(
        message,
        status=response.status_code,
        code=code,
        problems=problems,
        llm_hint=llm_hint,
    )


def _execute_response(response: httpx.Response) -> ExecuteResponse:
    data = response.json()
    return ExecuteResponse(
        execution_id=data["execution_id"],
        status=data["status"],
        idempotent_replay=response.headers.get(_REPLAY_HEADER) == "true",
    )


def _replay_response(response: httpx.Response) -> ReplayResponse:
    data = response.json()
    return ReplayResponse(
        execution_id=data["execution_id"],
        status=data["status"],
        replay_of=data["replay_of"],
    )


def _rotate_api_key_response(response: httpx.Response) -> RotateApiKeyResponse:
    data = response.json()
    return RotateApiKeyResponse(tenant_id=data["tenant_id"], api_key=data["api_key"])


def _rotate_webhook_secret_response(
    response: httpx.Response,
) -> RotateWebhookSecretResponse:
    data = response.json()
    return RotateWebhookSecretResponse(
        tenant_id=data["tenant_id"], webhook_secret=data["webhook_secret"]
    )


class Klanex:
    """Synchronous client.

    >>> klanex = Klanex(api_key="klx_...", base_url="https://klanex-ingest-....run.app")
    >>> accepted = klanex.execute(target={"url": "https://api.example.com/x"}, payload={...})
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("klanex: api_key is required")
        if not base_url:
            raise ValueError("klanex: base_url is required")
        self._client = client or httpx.Client(timeout=30)
        self._base = base_url.rstrip("/")
        self._headers = {"X-API-Key": api_key}

    def execute(
        self,
        *,
        target: dict[str, Any],
        payload: Any = None,
        payload_schema: Any = None,
        callback_url: str | None = None,
        max_attempts: int | None = None,
        idempotency_key: str | None = None,
        requires_approval: bool = False,
    ) -> ExecuteResponse:
        """Submit a tool-use intent. Returns as soon as the engine has
        validated and queued it. Raises KlanexSchemaError when the payload
        fails the schema gate — feed ``llm_hint`` back to your agent."""
        body = _build_body(
            target, payload, payload_schema, callback_url, max_attempts,
            idempotency_key, requires_approval
        )
        response = self._client.post(
            f"{self._base}/v1/executions", json=body, headers=self._headers
        )
        _raise_for_status(response)
        return _execute_response(response)

    def get(self, execution_id: str) -> Execution:
        """Fetch current state. Credentials are always redacted."""
        response = self._client.get(
            f"{self._base}/v1/executions/{execution_id}", headers=self._headers
        )
        _raise_for_status(response)
        return execution_from_wire(response.json())

    def replay(self, execution_id: str) -> ReplayResponse:
        """Re-run a terminal execution byte-exact. Not idempotent: every
        call creates a fresh execution."""
        response = self._client.post(
            f"{self._base}/v1/executions/{execution_id}/replay", headers=self._headers
        )
        _raise_for_status(response)
        return _replay_response(response)

    def rotate_api_key(self) -> RotateApiKeyResponse:
        """Rotate this tenant's API key. The old key stops working
        immediately; this client switches to the new key so further calls
        keep working. The raw key is returned only here."""
        response = self._client.post(
            f"{self._base}/v1/api-key/rotate", headers=self._headers
        )
        _raise_for_status(response)
        result = _rotate_api_key_response(response)
        self._headers["X-API-Key"] = result.api_key
        return result

    def rotate_webhook_secret(self) -> RotateWebhookSecretResponse:
        """Rotate this tenant's webhook signing secret. Callbacks sent after
        this are signed with the new secret, so update your verifier. The
        secret is returned only here."""
        response = self._client.post(
            f"{self._base}/v1/webhook-secret/rotate", headers=self._headers
        )
        _raise_for_status(response)
        return _rotate_webhook_secret_response(response)

    def wait_for_result(
        self,
        execution_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> Execution:
        """Poll until SUCCEEDED or FAILED. Prefer webhooks in production."""
        deadline = time.monotonic() + timeout
        while True:
            execution = self.get(execution_id)
            if execution.terminal:
                return execution
            if time.monotonic() + poll_interval > deadline:
                raise KlanexError(
                    f"execution {execution_id} still {execution.status} after {timeout}s",
                    status=0,
                    code="WAIT_TIMEOUT",
                )
            time.sleep(poll_interval)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Klanex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncKlanex:
    """Asynchronous client with the same surface as :class:`Klanex`."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("klanex: api_key is required")
        if not base_url:
            raise ValueError("klanex: base_url is required")
        self._client = client or httpx.AsyncClient(timeout=30)
        self._base = base_url.rstrip("/")
        self._headers = {"X-API-Key": api_key}

    async def execute(
        self,
        *,
        target: dict[str, Any],
        payload: Any = None,
        payload_schema: Any = None,
        callback_url: str | None = None,
        max_attempts: int | None = None,
        idempotency_key: str | None = None,
        requires_approval: bool = False,
    ) -> ExecuteResponse:
        body = _build_body(
            target, payload, payload_schema, callback_url, max_attempts,
            idempotency_key, requires_approval
        )
        response = await self._client.post(
            f"{self._base}/v1/executions", json=body, headers=self._headers
        )
        _raise_for_status(response)
        return _execute_response(response)

    async def get(self, execution_id: str) -> Execution:
        response = await self._client.get(
            f"{self._base}/v1/executions/{execution_id}", headers=self._headers
        )
        _raise_for_status(response)
        return execution_from_wire(response.json())

    async def replay(self, execution_id: str) -> ReplayResponse:
        response = await self._client.post(
            f"{self._base}/v1/executions/{execution_id}/replay", headers=self._headers
        )
        _raise_for_status(response)
        return _replay_response(response)

    async def rotate_api_key(self) -> RotateApiKeyResponse:
        """Rotate this tenant's API key; this client switches to the new key.
        The raw key is returned only here."""
        response = await self._client.post(
            f"{self._base}/v1/api-key/rotate", headers=self._headers
        )
        _raise_for_status(response)
        result = _rotate_api_key_response(response)
        self._headers["X-API-Key"] = result.api_key
        return result

    async def rotate_webhook_secret(self) -> RotateWebhookSecretResponse:
        """Rotate this tenant's webhook signing secret. The secret is returned
        only here; update your verifier."""
        response = await self._client.post(
            f"{self._base}/v1/webhook-secret/rotate", headers=self._headers
        )
        _raise_for_status(response)
        return _rotate_webhook_secret_response(response)

    async def wait_for_result(
        self,
        execution_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> Execution:
        deadline = time.monotonic() + timeout
        while True:
            execution = await self.get(execution_id)
            if execution.terminal:
                return execution
            if time.monotonic() + poll_interval > deadline:
                raise KlanexError(
                    f"execution {execution_id} still {execution.status} after {timeout}s",
                    status=0,
                    code="WAIT_TIMEOUT",
                )
            await asyncio.sleep(poll_interval)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncKlanex:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
