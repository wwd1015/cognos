"""The agent layer: agents *recommend*; the engine checks and disposes; humans decide at gates.

- ``contracts``  typed output per agent (Pydantic, extra="forbid")
- ``prompts/``   role prompts (the domain playbooks) + shared rules
- ``slices``     what each agent may see (independence rules)
- ``facts``      the only numbers an agent may cite; ``{{fact:id}}`` rendering
- ``checks``     engine re-checks on every answer (reject → retry with errors)
- ``heuristic``  deterministic agents — the offline/demo path
- ``providers``  heuristic | replay | claude_cli | anthropic | openai_compat
- ``runner``     validation, retries, audit, raw I/O, events, time and spend limits
"""
