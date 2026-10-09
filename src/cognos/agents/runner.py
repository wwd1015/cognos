"""AgentRunner — obtain one validated recommendation from an agent.

``recommend`` builds the agent's slice, calls the backend, validates the answer against the
contract *and* the engine's checks, and retries with the errors fed back (up to
``agents.max_retries``). Every attempt is audited (``runs/<id>/agents/audit.jsonl``), its raw input
and output are kept (``runs/<id>/agents/<call_id>.{input,output}.json``), and progress is published
to the run's activity feed. A wall-clock limit per call and a spend budget per run apply.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..engine import events
from . import checks, heuristic, providers, slices
from .contracts import AGENT_STAGE, CONTRACTS, FRIENDLY, Contract

PROMPTS_DIR = Path(__file__).with_name("prompts")
MIN_ATTEMPT_S = 20.0  # don't start a retry with less time than this left

TASKS = {
    "data_analyst": "Review the data profile. Decide keep or exclude for every leakage suspect and any "
                    "other column that should not be a model input, note data-quality issues, and list "
                    "questions the model sponsor must answer.",
    "design_lead": "Design the model: decide the role of every econometric framework, rank a slate of "
                   "engine-fittable specifications, optionally propose feature transforms, and list "
                   "design questions the sponsor must answer.",
    "modeler": "Choose the champion from the admissible set and check its coefficient signs against "
               "economic priors. You choose on cross-validation evidence only.",
    "experiment": "Propose ONE next experiment that could beat the current champion on the frozen "
                  "metric, or set stop=true if none is worth running.",
    "outcomes_analyst": "Interpret the outcomes analysis (discrimination, calibration, stability) of "
                        "the champion on the evaluation sample and raise findings.",
    "validator": "Independently challenge this model (SR 11-7 effective challenge). Raise findings the "
                 "decision-maker must see, each with evidence, the stage that must change, and a remedy.",
    "risk_analyst": "Assess model-risk readiness from the evidence and list the priority actions "
                    "humans must take before use.",
    "writer": "Write the narrative sections of the white paper. Use {{fact:<id>}} placeholders for "
              "every number.",
}


class AgentRunError(RuntimeError):
    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or []


class BudgetExceeded(AgentRunError):
    pass


def system_prompt(agent: str) -> str:
    name = "modeler" if agent == "experiment" else agent  # guided search is the modeler's role
    body = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    common = (PROMPTS_DIR / "_common.md").read_text(encoding="utf-8")
    return f"{body}\n\n{common}"


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def output_schema(contract: type[Contract]) -> dict[str, Any]:
    """The contract's JSON schema, cleaned for strict structured-output grammars: unsupported
    keywords removed (Pydantic re-checks them) and every object closed."""
    unsupported = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
                   "minLength", "maxLength", "minItems", "maxItems", "pattern", "title", "default"}

    def clean(o: Any) -> Any:
        if isinstance(o, dict):
            d = {k: clean(v) for k, v in o.items() if k not in unsupported}
            if d.get("type") == "object" and "properties" in d:
                d["additionalProperties"] = False
            return d
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    return clean(contract.model_json_schema())


def build_prompt(agent: str, sl: dict[str, Any], feedback: str | None) -> str:
    parts = [TASKS[agent],
             "## Context\nThe JSON below is your complete context. `facts` holds the only numbers "
             "you may cite (by id).\n```json\n" + json.dumps(sl, indent=1, default=str) + "\n```"]
    if sl.get("challenges"):
        parts.append("## Challenges\n`context.challenges` contains pushback on this stage from a "
                     "human reviewer or the independent validator. Address every one in "
                     "`responses_to_challenges`, and change your recommendation where it is right.")
    if feedback:
        parts.append("## Your previous answer was rejected\nFix every problem below and answer "
                     "again:\n" + feedback)
    parts.append("Reply with a single JSON object that matches the required schema.")
    return "\n\n".join(parts)


class AgentRunner:
    def __init__(self, provider: dict[str, Any], run_dir: Path, *, max_retries: int = 2,
                 time_limit_s: float = 300.0, budget_usd: float = 0.0,
                 replay_dir: str | None = None, backend=None):
        self.provider = provider
        self.run_dir = Path(run_dir)
        self.max_retries = max_retries
        self.time_limit_s = time_limit_s
        self.budget_usd = budget_usd
        self.replay_dir = replay_dir
        self._backend = backend  # injected in tests
        self._calls: dict[str, int] = {}
        self.last: dict[str, dict[str, Any]] = {}

    @classmethod
    def for_config(cls, cfg, run_dir: Path, provider: str | None = None) -> AgentRunner:
        a = cfg.agents
        return cls(providers.resolve(provider or a.provider, a.model), run_dir,
                   max_retries=a.max_retries, time_limit_s=a.time_limit_s,
                   budget_usd=a.budget_usd, replay_dir=a.replay_dir)

    @property
    def kind(self) -> str:
        return self.provider["kind"]

    @property
    def dir(self) -> Path:
        d = self.run_dir / "agents"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- public --------------------------------------------------------------------------
    def recommend(self, ctx, agent: str, data: dict[str, Any], *, fresh: dict | None = None,
                  check=None) -> Contract:
        contract = CONTRACTS[agent]
        sl = slices.build(ctx, agent, data, fresh)
        stage = AGENT_STAGE[agent]
        deadline = time.time() + self.time_limit_s
        feedback: str | None = None
        errors: list[str] = []
        name = FRIENDLY[agent]
        for attempt in range(1, self.max_retries + 2):
            if attempt > 1 and deadline - time.time() < MIN_ATTEMPT_S:
                self._emit("agent_error", f"{name} is out of time; not retrying.", agent, stage)
                break
            self._check_budget()
            call_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{agent}_a{attempt}_{uuid.uuid4().hex[:4]}"
            prompt = build_prompt(agent, sl, feedback)
            system = system_prompt(agent)
            self._write(f"{call_id}.input.json", {
                "agent": agent, "stage": stage, "attempt": attempt, "provider": self.provider["id"],
                "model": self.provider.get("model"), "system": system, "prompt": prompt,
                "context": sl})
            self._emit("agent_start", f"{name} is working" + (" (retry)" if attempt > 1 else "")
                       + "…", agent, stage)
            t0 = time.time()
            meta: dict[str, Any] = {}
            try:
                raw, meta = self._call(agent, system, prompt, contract, sl, call_id, deadline)
            except BudgetExceeded:
                raise
            except Exception as exc:  # backend/infra failure: audited, then retried
                errors = [f"{type(exc).__name__}: {exc}"]
                self._audit(call_id, agent, stage, attempt, sl, t0, meta, "error", errors[0])
                self._emit("agent_error", f"{name} hit an error: {exc}", agent, stage)
                feedback = None
                continue
            self._write(f"{call_id}.output.json", {"raw": raw, "meta": meta})
            out: Contract | None = None
            try:
                out = contract.model_validate(raw)
                errors = checks.run_checks(agent, out, sl, check)
            except ValidationError as e:
                errors = [f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:20]]
            status = "ok" if not errors else "invalid"
            self._audit(call_id, agent, stage, attempt, sl, t0, meta, status,
                        "; ".join(errors) or None)
            if out is not None and not errors:
                self.last[agent] = {"call_id": call_id, "attempts": attempt,
                                    "provider": self.provider["id"],
                                    "model": meta.get("model") or self.provider.get("model"),
                                    "kind": self.kind}
                self._emit("agent_done", f"{name} finished.", agent, stage)
                return out
            feedback = "\n".join(f"- {e}" for e in errors)
            self._emit("agent_invalid", f"{name}'s answer failed the engine's checks; asking it to "
                       "correct.", agent, stage, errors=errors)
        raise AgentRunError(f"{name} produced no valid recommendation: "
                            + ("; ".join(errors[:3]) or "no attempt completed"), errors)

    # --- backends --------------------------------------------------------------------------
    def _call(self, agent, system, prompt, contract, sl, call_id, deadline):
        if self._backend is not None:
            return self._backend(agent=agent, system=system, prompt=prompt, contract=contract,
                                 context=sl, provider=self.provider, deadline=deadline)
        kind = self.kind
        if kind == "heuristic":
            return heuristic.recommend(agent, sl), {"model": "heuristic", "turns": 0}
        if kind == "replay":  # recorded output where one exists, the heuristic agent otherwise
            recorded = self._replay(agent)
            if recorded is None:
                return heuristic.recommend(agent, sl), {"model": "heuristic", "turns": 0}
            return recorded, {"model": "replay", "turns": 0}
        schema = output_schema(contract)
        log_path = self.dir / f"{call_id}.transcript.jsonl"
        if kind == "claude_cli":
            from .backends import claude_cli
            return claude_cli.call(system, prompt, schema, self.provider, deadline, log_path)
        if kind == "anthropic":
            from .backends import anthropic_api
            return anthropic_api.call(system, prompt, schema, self.provider, deadline, log_path)
        if kind == "openai_compat":
            from .backends import openai_compat
            return openai_compat.call(system, prompt, schema, contract, self.provider, deadline,
                                      log_path)
        raise AgentRunError(f"unknown provider kind {kind!r}")

    def _replay(self, agent: str) -> dict[str, Any] | None:
        if not self.replay_dir:
            raise AgentRunError("provider 'replay' needs agents.replay_dir")
        path = Path(self.replay_dir) / f"{agent}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        n = self._calls.get(agent, 0)
        self._calls[agent] = n + 1
        if isinstance(data, list):
            return data[min(n, len(data) - 1)]
        return data

    # --- bookkeeping -----------------------------------------------------------------------
    def spent_usd(self) -> float:
        path = self.dir / "audit.jsonl"
        if not path.exists():
            return 0.0
        total = 0.0
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                total += float(json.loads(line).get("cost_usd") or 0.0)
            except (ValueError, TypeError):
                continue
        return total

    def _check_budget(self) -> None:
        if self.budget_usd and self.kind not in ("heuristic", "replay"):
            spent = self.spent_usd()
            if spent >= self.budget_usd:
                raise BudgetExceeded(f"the run's agent budget (${self.budget_usd:.2f}) is used up "
                                     f"(${spent:.2f} spent); raise agents.budget_usd to continue")

    def _write(self, name: str, obj: Any) -> None:
        (self.dir / name).write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")

    def _audit(self, call_id, agent, stage, attempt, sl, t0, meta, status, error) -> None:
        entry = {
            "call_id": call_id, "agent": agent, "stage": stage, "attempt": attempt,
            "provider": self.provider["id"], "model": meta.get("model") or self.provider.get("model"),
            "prompt_hash": _hash(system_prompt(agent)), "context_hash": _hash(sl),
            "status": status, "error": error, "duration_s": round(time.time() - t0, 2),
            "cost_usd": meta.get("cost_usd"), "turns": meta.get("turns"),
            "input_tokens": meta.get("input_tokens"), "output_tokens": meta.get("output_tokens"),
            "ts": time.time(),
        }
        with open(self.dir / "audit.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    def _emit(self, type_: str, message: str, agent: str, stage: str, **extra) -> None:
        events.publish(self.run_dir, type_, message, agent=agent, stage=stage, **extra)


def read_audit(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "agents" / "audit.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
