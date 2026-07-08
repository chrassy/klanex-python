"""Adapters that wrap a :class:`KlanexTool` as a native tool for popular
agent frameworks. Frameworks are imported lazily, so ``import klanex`` never
requires any of them; a missing framework raises a clear install hint."""

from __future__ import annotations

from typing import Any

from ..client import Klanex
from .core import KlanexTool


def _tool(client: Klanex, kwargs: dict[str, Any]) -> KlanexTool:
    return KlanexTool(client, **kwargs)


def _missing(framework: str, package: str) -> ImportError:
    return ImportError(
        f"{framework} is not installed. Install it to use this adapter: pip install {package}"
    )


def langchain_tool(client: Klanex, **kwargs: Any):
    """Return a LangChain ``StructuredTool`` (works with LangGraph too).

    >>> tool = langchain_tool(klanex, name="create_refund",
    ...     description="Refund a Stripe charge",
    ...     target={"url": "https://api.stripe.com/v1/refunds",
    ...             "connection_id": "con_..."},
    ...     payload_schema={"type": "object", "required": ["charge_id"]})
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise _missing("LangChain", "langchain-core") from exc

    kt = _tool(client, kwargs)

    def _run(payload: dict[str, Any]) -> str:
        return kt.run(payload)

    return StructuredTool.from_function(
        func=_run,
        name=kt.name,
        description=kt.description,
    )


def crewai_tool(client: Klanex, **kwargs: Any):
    """Return a CrewAI ``BaseTool`` instance."""
    try:
        from crewai.tools import BaseTool
    except ImportError as exc:  # pragma: no cover
        raise _missing("CrewAI", "crewai") from exc

    kt = _tool(client, kwargs)

    class _KlanexCrewTool(BaseTool):
        name: str = kt.name
        description: str = kt.description

        def _run(self, payload: dict[str, Any]) -> str:
            return kt.run(payload)

    return _KlanexCrewTool()


def adk_tool(client: Klanex, **kwargs: Any):
    """Return a Google ADK ``FunctionTool``."""
    try:
        from google.adk.tools import FunctionTool
    except ImportError as exc:  # pragma: no cover
        raise _missing("Google ADK", "google-adk") from exc

    kt = _tool(client, kwargs)

    def _run(payload: dict[str, Any]) -> str:
        return kt.run(payload)

    _run.__name__ = kt.name
    _run.__doc__ = kt.description
    return FunctionTool(_run)
