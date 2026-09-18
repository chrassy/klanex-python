"""Agent-framework adapters for klanex.

Wrap a klanex-managed target as a native tool for LangChain/LangGraph, the
OpenAI Agents SDK, CrewAI, or Google ADK. The agent calls it like any tool,
and klanex owns the reliability (schema gate, retries, approvals,
credentials).
"""

from .core import KlanexTool
from .frameworks import adk_tool, crewai_tool, langchain_tool, openai_agents_tool

__all__ = ["KlanexTool", "adk_tool", "crewai_tool", "langchain_tool", "openai_agents_tool"]
