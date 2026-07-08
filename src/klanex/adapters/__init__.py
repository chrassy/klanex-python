"""Agent-framework adapters for klanex.

Wrap a klanex-managed target as a native tool for LangChain/LangGraph,
CrewAI, or Google ADK — the agent calls it like any tool, and klanex owns
the reliability (schema gate, retries, approvals, credentials).
"""

from .core import KlanexTool
from .frameworks import adk_tool, crewai_tool, langchain_tool

__all__ = ["KlanexTool", "adk_tool", "crewai_tool", "langchain_tool"]
