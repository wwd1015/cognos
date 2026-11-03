"""Agent ids. An id is the agent's display name in snake case (``data_agent`` is the Data
agent), so a raw id on a page or in a file reads the same as the name (ADR-0023).

Runs recorded before the rename carry the old ids (``data_analyst``, ``validator``, ...) in
their results, audit files, challenges and replay recordings. Those files are history and are
never rewritten; everything that reads an id from disk goes through :func:`canonical`.
"""

from __future__ import annotations

LEGACY = {"intake_analyst": "intake_agent", "data_analyst": "data_agent",
          "data_scout": "data_agent_requests", "data_linker": "data_agent_joins",
          "design_lead": "design_agent", "modeler": "modeling_agent",
          "experiment": "modeling_agent_search", "outcomes_analyst": "backtest_agent",
          "validator": "validation_agent", "risk_analyst": "compliance_agent",
          "writer": "writing_agent"}
STEP_SUFFIXES = ("_tools", "_ask")


def canonical(agent: str) -> str:
    """The current id for an id read from a record: an old id (or its tool or question step)
    becomes the new one; anything else is returned as it is."""
    for suffix in STEP_SUFFIXES:
        if agent.endswith(suffix) and agent[:-len(suffix)] in LEGACY:
            return LEGACY[agent[:-len(suffix)]] + suffix
    return LEGACY.get(agent, agent)


def legacy_names(agent: str) -> list[str]:
    """The old ids an agent's recordings may be filed under (replay)."""
    base, suffix = agent, ""
    for s in STEP_SUFFIXES:
        if agent.endswith(s):
            base, suffix = agent[:-len(s)], s
    return [old + suffix for old, new in LEGACY.items() if new == base]
