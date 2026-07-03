"""Typed errors. Schema failures carry an ``llm_hint`` written to be fed
back into the agent's context for self-correction."""

from __future__ import annotations


class KlanexError(Exception):
    """Non-2xx response from the klanex API."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        code: str,
        problems: list[str] | None = None,
        llm_hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.problems = problems
        self.llm_hint = llm_hint


class KlanexSchemaError(KlanexError):
    """The payload failed the JSON Schema gate (HTTP 422).

    ``llm_hint`` can be injected verbatim into the agent's context so it
    regenerates a corrected payload.
    """

    def __init__(
        self,
        message: str,
        *,
        problems: list[str] | None = None,
        llm_hint: str | None = None,
    ) -> None:
        super().__init__(
            message, status=422, code="SCHEMA_INVALID", problems=problems, llm_hint=llm_hint
        )


class WebhookVerificationError(Exception):
    """A webhook delivery failed signature or timestamp verification."""
