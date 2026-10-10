"""RunContext — the shared, checkpointed state object threaded through every stage.

Heavy artifacts (datasets, fitted models, result tables, OKF bundles) live on disk under the run
directory and are passed *by reference*; only small structured payloads live in memory. This is
what lets a stage run in a fresh process (``cognos run-stage ...``) reconstruct everything the
previous stages produced — the key to COGNOS's stage-by-stage / human-in-the-loop mode.

v1: the context also carries the run's human **overrides** (from ``state.json``) — design answers
shape the *effective* config, column exclusions shape :meth:`RunContext.profile` — and the single
seam through which a stage obtains judgment: :meth:`RunContext.recommend` (the agent layer).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import pandas as pd

from .artifacts import ArtifactRef, StageResult
from .fsutil import atomic_write

if TYPE_CHECKING:
    from .agents.contracts import Contract
    from .agents.runner import AgentRunner
    from .config import CognosConfig
    from .engine.state import Challenge, Overrides


def new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:7]}"


class RunContext:
    """Owns the run directory, the artifact store, checkpoints, overrides, and the agent runner."""

    def __init__(
        self,
        config: CognosConfig,
        run_id: str | None = None,
        runs_root: str | Path | None = None,
        runner: AgentRunner | None = None,
    ) -> None:
        self.base_config = config
        self.run_id = run_id or new_run_id()
        root = Path(runs_root or config.runs_dir)
        self.run_dir = root / self.run_id
        self._runner = runner
        self.overrides = self._load_overrides()
        self.config = self._effective_config(config, self.overrides).with_target(
            *self._decided_target(config))
        self._results: dict[str, StageResult] = {}
        # Tool runs requested by a stage's agent in this process (analysis/consult.py); the
        # stage payload is the durable copy.
        self.tool_runs: dict[str, list[dict[str, Any]]] = {}
        self.tools_unavailable: dict[str, list[dict[str, str]]] = {}
        self.logger = logging.getLogger(f"cognos.run.{self.run_id}")

        self._ensure_dirs()
        if self.manifest_path.exists():
            self._load_existing()
        else:
            self._write_manifest()

    # --- directory layout --------------------------------------------------------
    def _ensure_dirs(self) -> None:
        for d in (self.run_dir, self.data_dir, self.models_dir, self.docs_dir, self.stages_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def data_dir(self) -> Path:
        return self.run_dir / "data"

    @property
    def models_dir(self) -> Path:
        return self.run_dir / "models"

    @property
    def docs_dir(self) -> Path:
        return self.run_dir / "docs"  # OKF bundle root

    @property
    def stages_dir(self) -> Path:
        return self.run_dir / "stages"

    @property
    def agents_dir(self) -> Path:
        return self.run_dir / "agents"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def stage_dir(self, stage: str) -> Path:
        d = self.stages_dir / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def rel(self, path: Path) -> str:
        """Path relative to the run dir (how artifacts are referenced in JSON)."""
        try:
            return str(path.relative_to(self.run_dir))
        except ValueError:
            return str(path)

    def resolve(self, relpath: str) -> Path:
        return self.run_dir / relpath

    # --- artifact store ----------------------------------------------------------
    def save_json(self, relpath: str, obj: Any) -> ArtifactRef:
        path = self.run_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, json.dumps(obj, indent=2, default=str))
        return ArtifactRef(name=Path(relpath).stem, kind="json", path=relpath)

    def load_json(self, relpath: str) -> Any:
        with open(self.run_dir / relpath, encoding="utf-8") as fh:
            return json.load(fh)

    def save_text(self, relpath: str, text: str, kind: str = "text") -> ArtifactRef:
        path = self.run_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return ArtifactRef(name=Path(relpath).stem, kind=kind, path=relpath)

    def save_df(self, relpath: str, df: pd.DataFrame, kind: str = "table") -> ArtifactRef:
        path = self.run_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        if relpath.endswith(".parquet"):
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        return ArtifactRef(name=Path(relpath).stem, kind=kind, path=relpath)

    def load_df(self, relpath: str) -> pd.DataFrame:
        path = self.run_dir / relpath
        if str(path).endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def save_model(self, name: str, obj: Any) -> ArtifactRef:
        relpath = f"models/{name}.joblib"
        joblib.dump(obj, self.run_dir / relpath)
        return ArtifactRef(name=name, kind="model", path=relpath)

    def load_model(self, name: str) -> Any:
        return joblib.load(self.run_dir / f"models/{name}.joblib")

    # --- checkpointing -----------------------------------------------------------
    def record(self, result: StageResult) -> StageResult:
        self._results[result.stage] = result
        path = self.stage_dir(result.stage) / "result.json"
        atomic_write(path, result.model_dump_json(indent=2))  # readers never see a torn file
        self._write_manifest()
        self.logger.info(result.token_line())
        return result

    def get(self, stage: str) -> StageResult | None:
        if stage in self._results:
            return self._results[stage]
        path = self.stages_dir / stage / "result.json"
        if path.exists():
            res = StageResult.model_validate_json(path.read_text(encoding="utf-8"))
            self._results[stage] = res
            return res
        return None

    def require(self, stage: str) -> StageResult:
        res = self.get(stage)
        if res is None:
            raise RuntimeError(
                f"Stage '{stage}' has not run yet for run {self.run_id}. "
                f"Run it first (cognos run-stage {stage} --run {self.run_id})."
            )
        return res

    def has(self, stage: str) -> bool:
        return self.get(stage) is not None

    def completed_stages(self) -> list[str]:
        return [s for s in self.config.stages.enabled if self.has(s)]

    # --- dataset access (loaded fresh; cached in data/ on first explore) ----------
    def load_dataset(self) -> pd.DataFrame:
        cached = self.data_dir / "dataset.parquet"
        if cached.exists():
            return pd.read_parquet(cached)
        dc = self.config.data
        if len(dc.sources) >= 2:
            raise RuntimeError("This run joins several sources; the joined dataset is built by "
                               "the explore stage. Run explore first.")
        if not dc.path and dc.source is None and not dc.sources:
            raise RuntimeError(
                "No dataset on disk and config.data.path is unset. Pass a DataFrame via "
                "RunContext.attach_dataset() before running stages."
            )
        from . import datasources

        df, provenance = datasources.load(self.config)
        df.to_parquet(cached, index=False)
        # Where the snapshot came from, and its hash: every later stage reads this copy.
        import hashlib

        provenance["snapshot_sha256"] = hashlib.sha256(cached.read_bytes()).hexdigest()
        atomic_write(self.data_dir / "source.json", json.dumps(provenance, indent=1, default=str))
        return df

    def data_source(self) -> dict[str, Any]:
        """Provenance of the dataset snapshot (empty when a DataFrame was attached in code)."""
        try:
            return json.loads((self.data_dir / "source.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {}

    # --- v1: overrides, effective config, agent seam ------------------------------
    def _load_overrides(self) -> Overrides:
        from .engine.state import Overrides, RunState

        if RunState.exists(self.run_dir):
            try:
                return RunState.load(self.run_dir).overrides
            except Exception:  # a corrupt/partial state file must not break a stage
                pass
        return Overrides()

    def _decided_target(self, config: CognosConfig) -> tuple[str | None, str | None]:
        """The target and task in force when the profile leaves them open: the data-gate
        decision, else what explore analysed (a run whose data gate is switched off)."""
        if config.data.target and config.task is not None:
            return None, None
        if self.overrides.target:
            return self.overrides.target, self.overrides.task
        try:
            payload = json.loads((self.run_dir / "stages" / "explore" / "result.json").read_text(
                encoding="utf-8")).get("payload") or {}
        except (FileNotFoundError, ValueError):
            return None, None
        return payload.get("target"), payload.get("task")

    @staticmethod
    def _effective_config(config: CognosConfig, overrides: Overrides) -> CognosConfig:
        """The profile plus human design answers (the YAML itself is never edited)."""
        if not overrides.design and not overrides.compliance:
            return config
        cfg = config.model_copy(deep=True)
        for section, values in ((cfg.design, overrides.design),
                                (cfg.compliance, overrides.compliance)):
            for field_name, value in values.items():
                if hasattr(section, field_name):
                    setattr(section, field_name, value)
        return cfg

    def profile(self) -> dict[str, Any]:
        """The explore profile as downstream stages must see it: human-excluded columns removed."""
        base = dict(self.require("explore").payload)
        excluded = set(self.overrides.exclude_columns)
        if not excluded:
            return base
        for key in ("features", "numeric_features", "categorical_features"):
            base[key] = [c for c in base.get(key, []) if c not in excluded]
        base["top_correlations"] = [c for c in base.get("top_correlations", [])
                                    if c["feature"] not in excluded]
        base["excluded_columns"] = sorted(excluded)
        return base

    def challenges_for(self, stage: str) -> list[Challenge]:
        from .engine.state import RunState

        if not RunState.exists(self.run_dir):
            return []
        return RunState.load(self.run_dir).open_challenges(stage)

    def sponsor_answers(self) -> list[dict[str, str]]:
        """Questions the sponsor has answered (or accepted as assumptions) — every agent sees them."""
        from .engine.state import RunState

        if not RunState.exists(self.run_dir):
            return []
        return [{"id": g.id, "question": g.question, "answer": g.answer or "", "status": g.status}
                for g in RunState.load(self.run_dir).gaps if g.status != "open"]

    @property
    def runner(self) -> AgentRunner:
        if self._runner is None:
            from .agents.runner import AgentRunner

            self._runner = AgentRunner.for_config(self.config, self.run_dir)
        return self._runner

    def recommend(self, agent: str, data: dict[str, Any], *,
                  fresh: dict[str, StageResult] | None = None,
                  check=None) -> Contract:
        """Ask ``agent`` for its recommendation. The engine validates it (``check`` adds
        stage-specific rules) and retries with the errors; the result is a validated contract."""
        return self.runner.recommend(self, agent, data, fresh=fresh, check=check)

    def consult_tools(self, agent: str, brief: dict[str, Any] | None = None, *,
                      res: StageResult | None = None,
                      inputs: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Let ``agent`` request the tools registered for its stage (built in or from a plugin)
        before it recommends; the engine runs them. Returns the runs to pass on as
        ``tool_runs``. A stage with no available tool makes no agent call."""
        from .analysis import consult

        return consult.consult(self, agent, brief, res=res, inputs=inputs)

    def attach_dataset(self, df: pd.DataFrame) -> ArtifactRef:
        """Persist an in-memory DataFrame as the canonical dataset for this run."""
        path = self.data_dir / "dataset.parquet"
        df.to_parquet(path, index=False)
        return ArtifactRef(name="dataset", kind="table", path=self.rel(path))

    # --- manifest ----------------------------------------------------------------
    def _write_manifest(self) -> None:
        manifest = {
            "run_id": self.run_id,
            "project": self.config.name,
            "mode": self.config.mode.value,
            "task": self.config.task.value if self.config.task else None,
            "created_at": datetime.now(UTC).isoformat(),
            "stages": {
                s: (self._results[s].verdict.value if s in self._results else None)
                for s in self.config.stages.enabled
            },
        }
        with open(self.manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    def _load_existing(self) -> None:
        for stage in self.config.stages.enabled:
            self.get(stage)  # populates cache from disk

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        done = ",".join(self.completed_stages()) or "-"
        return f"<RunContext {self.run_id} project={self.config.name!r} done=[{done}]>"
