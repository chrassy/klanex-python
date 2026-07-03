"""Webhook signature verification, byte-compatible with the engine's Go
implementation: "sha256=" + hex(HMAC-SHA256(secret, f"{timestamp}.{body}"))."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Final

from .errors import WebhookVerificationError
from .types import WebhookEvent, error_from_wire, result_from_wire

#: Header names used on klanex webhook deliveries.
WEBHOOK_HEADERS: Final = {
    "signature": "X-Klanex-Signature",
    "timestamp": "X-Klanex-Timestamp",
    "event": "X-Klanex-Event",
    "execution_id": "X-Klanex-Execution-Id",
}


def sign_webhook(secret: str, timestamp: int, body: bytes | str) -> str:
    """Compute the signature for a body at a unix timestamp."""
    if isinstance(body, str):
        body = body.encode("utf-8")
    mac = hmac.new(secret.encode("utf-8"), f"{timestamp}.".encode(), hashlib.sha256)
    mac.update(body)
    return "sha256=" + mac.hexdigest()


def verify_webhook(
    *,
    secret: str,
    body: bytes | str,
    signature: str,
    timestamp: int | str,
    tolerance_seconds: int = 300,
    now: int | None = None,
) -> WebhookEvent:
    """Verify a delivery and return the parsed event.

    Pass the raw request body exactly as received — re-serializing parsed
    JSON breaks the signature. Raises WebhookVerificationError on a bad
    signature, a stale timestamp (replay protection; set
    ``tolerance_seconds=0`` to disable), or an unparseable body.
    """
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        raise WebhookVerificationError("invalid timestamp header") from None

    if tolerance_seconds > 0:
        current = now if now is not None else int(time.time())
        drift = abs(current - ts)
        if drift > tolerance_seconds:
            raise WebhookVerificationError(
                f"timestamp outside tolerance ({drift}s > {tolerance_seconds}s); possible replay"
            )

    expected = sign_webhook(secret, ts, body)
    if not hmac.compare_digest(expected, signature):
        raise WebhookVerificationError("signature mismatch")

    text = body.decode("utf-8") if isinstance(body, bytes) else body
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raise WebhookVerificationError("body is not valid JSON") from None

    return WebhookEvent(
        event=raw["event"],
        execution_id=raw["execution_id"],
        status=raw["status"],
        attempts=raw.get("attempts", 0),
        result=result_from_wire(raw.get("result")),
        error=error_from_wire(raw.get("error")),
    )
