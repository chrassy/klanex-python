"""Adapters that wrap a :class:`KlanexTool` as a native tool for popular
agent frameworks. Frameworks are imported lazily, so ``import klanex`` never
requires any of them; a missing framework raises a clear install hint."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from ..client import AsyncKlanex, Klanex
from .core import KlanexTool


def _tool(client: Klanex | AsyncKlanex, kwargs: dict[str, Any]) -> KlanexTool:
    return KlanexTool(client, **kwargs)


def _missing(framework: str, package: str) -> ImportError:
    return ImportError(
        f"{framework} is not installed. Install it to use this adapter: pip install {package}"
    )


# The tool call ID LangChain hands to invoke/ainvoke, read back in _run/_arun
# for the idempotency key. A context variable keeps concurrent calls apart.
_lc_tool_call_id: ContextVar[str | None] = ContextVar("klanex_tool_call_id", default=None)


def langchain_tool(client: Klanex | AsyncKlanex, **kwargs: Any):
    """Return a LangChain tool; works in LangGraph (``ToolNode``,
    ``create_react_agent``) and LangChain agents.

    The model's tool arguments are the payload, described to the model by
    ``payload_schema``. Pass an :class:`AsyncKlanex` for async graphs; a sync
    :class:`Klanex` also works there (it runs in a thread).

    >>> tool = langchain_tool(klanex, name="create_refund",
    ...     description="Refund a Stripe charge",
    ...     target={"url": "https://api.stripe.com/v1/refunds",
    ...             "connection_id": "con_..."},
    ...     payload_schema={"type": "object", "required": ["charge_id"],
    ...                     "properties": {"charge_id": {"type": "string"}}})
    """
    try:
        from langchain_core.tools import BaseTool
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise _missing("LangChain", "langchain-core") from exc

    kt = _tool(client, kwargs)

    class KlanexLangChainTool(BaseTool):
        name: str = kt.name
        description: str = kt.description
        # A JSON Schema dict: LangChain passes the arguments through as kwargs.
        args_schema: Any = kt.input_schema

        def invoke(self, input: Any, config: Any = None, **kw: Any) -> Any:
            token = _lc_tool_call_id.set(_call_id(input))
            try:
                return super().invoke(input, config, **kw)
            finally:
                _lc_tool_call_id.reset(token)

        async def ainvoke(self, input: Any, config: Any = None, **kw: Any) -> Any:
            token = _lc_tool_call_id.set(_call_id(input))
            try:
                return await super().ainvoke(input, config, **kw)
            finally:
                _lc_tool_call_id.reset(token)

        def _run(self, *args: Any, run_manager: Any = None, **payload: Any) -> str:
            return kt.run(payload, tool_call_id=_lc_tool_call_id.get())

        async def _arun(self, *args: Any, run_manager: Any = None, **payload: Any) -> str:
            return await kt.arun(payload, tool_call_id=_lc_tool_call_id.get())

    return KlanexLangChainTool()


def _call_id(input: Any) -> str | None:
    """The ID of a LangChain ToolCall input ({"type": "tool_call", ...})."""
    if isinstance(input, dict) and input.get("type") == "tool_call":
        return input.get("id")
    return None


def openai_agents_tool(client: Klanex | AsyncKlanex, *, strict: bool = False, **kwargs: Any):
    """Return an OpenAI Agents SDK ``FunctionTool``.

    The model's tool arguments are the payload, described by
    ``payload_schema``. The Agents SDK does not validate a plain JSON Schema,
    so klanex's schema gate does, and a failing input comes back to the
    model as a correction hint. ``strict`` enables OpenAI strict mode, which
    requires a strict-compatible schema (every property required, no
    additional properties).

    >>> refund = openai_agents_tool(klanex, name="create_refund",
    ...     description="Refund a Stripe charge",
    ...     target={"url": "https://api.stripe.com/v1/refunds",
    ...             "connection_id": "con_..."},
    ...     payload_schema={...})
    >>> agent = Agent(name="Support", tools=[refund])
    """
    try:
        from agents import FunctionTool
    except ImportError as exc:  # pragma: no cover
        raise _missing("OpenAI Agents SDK", "openai-agents") from exc

    kt = _tool(client, kwargs)

    async def _invoke(ctx: Any, arguments: str) -> str:
        return await kt.arun(arguments or "{}", tool_call_id=getattr(ctx, "tool_call_id", None))

    return FunctionTool(
        name=kt.name,
        description=kt.description,
        params_json_schema=kt.input_schema,
        on_invoke_tool=_invoke,
        strict_json_schema=strict,
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
        description: str = kt.description_with_schema

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
    _run.__doc__ = kt.description_with_schema
    return FunctionTool(_run)
