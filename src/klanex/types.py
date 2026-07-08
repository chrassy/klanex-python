"""Response types. Field names match the wire format directly — the klanex
API is snake_case, so nothing is renamed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TERMINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED"})


@dataclass(frozen=True)
class ExecutionResult:
    status_code: int
    body: str


@dataclass(frozen=True)
class ExecutionError:
    code: str
    message: str
    llm_hint: str | None = None


@dataclass(frozen=True)
class ExecuteResponse:
    execution_id: str
    status: str
    idempotent_replay: bool


@dataclass(frozen=True)
class ReplayResponse:
    execution_id: str
    status: str
    replay_of: str


@dataclass(frozen=True)
class RotateApiKeyResponse:
    tenant_id: str
    #: The new API key. Returned only once — persist it if needed.
    api_key: str


@dataclass(frozen=True)
class RotateWebhookSecretResponse:
    tenant_id: str
    #: The new webhook signing secret. Returned only once.
    webhook_secret: str


@dataclass(frozen=True)
class Execution:
    execution_id: str
    status: str
    attempts: int
    max_attempts: int
    target: dict[str, Any]
    callback_url: str | None
    result: ExecutionResult | None
    error: ExecutionError | None
    replay_of: str | None
    created_at: str
    updated_at: str

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


@dataclass(frozen=True)
class WebhookEvent:
    event: str
    execution_id: str
    status: str
    attempts: int
    result: ExecutionResult | None
    error: ExecutionError | None


def result_from_wire(raw: dict[str, Any] | None) -> ExecutionResult | None:
    if not raw:
        return None
    return ExecutionResult(status_code=raw["status_code"], body=raw["body"])


def error_from_wire(raw: dict[str, Any] | None) -> ExecutionError | None:
    if not raw:
        return None
    return ExecutionError(
        code=raw["code"], message=raw["message"], llm_hint=raw.get("llm_hint")
    )


def execution_from_wire(raw: dict[str, Any]) -> Execution:
    return Execution(
        execution_id=raw["execution_id"],
        status=raw["status"],
        attempts=raw.get("attempts", 0),
        max_attempts=raw.get("max_attempts", 0),
        target=raw.get("target", {}),
        callback_url=raw.get("callback_url"),
        result=result_from_wire(raw.get("result")),
        error=error_from_wire(raw.get("error")),
        replay_of=raw.get("replay_of"),
        created_at=raw.get("created_at", ""),
        updated_at=raw.get("updated_at", ""),
    )
