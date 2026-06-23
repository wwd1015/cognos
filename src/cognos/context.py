"""RunContext — the shared, checkpointed state object threaded through every stage.

Heavy artifacts (datasets, fitted models, result tables, OKF bundles) live on disk under the run
directory and are passed *by reference*; only small structured payloads live in memory. This is
what lets a stage run in a fresh process (``cognos run-stage ...``) reconstruct everything the
previous stages produced — the key to COGNOS's stage-by-stage / human-in-the-loop mode.
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

if TYPE_CHECKING:
    from .brains.base import Brain
    from .config import CognosConfig


def new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:7]}"


class RunContext:
    """Owns the run directory, the artifact store, checkpoints, and the (optional) LLM brain."""

    def __init__(
        self,
        config: CognosConfig,
        run_id: str | None = None,
        runs_root: str | Path | None = None,
        brain: Brain | None = None,
    ) -> None:
        from .brains import make_brain  # local import to avoid cycles

        self.config = config
        self.run_id = run_id or new_run_id()
        root = Path(runs_root or config.runs_dir)
        self.run_dir = root / self.run_id
        self.brain = brain or make_brain(config.brain)
        self._results: dict[str, StageResult] = {}
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
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
        return ArtifactRef(name=Path(relpath).stem, kind="json", path=relpath)

    def load_json(self, relpath: str) -> Any:
        with open(self.run_dir / relpath) as fh:
            return json.load(fh)

    def save_text(self, relpath: str, text: str, kind: str = "text") -> ArtifactRef:
        path = self.run_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
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
        with open(path, "w") as fh:
            fh.write(result.model_dump_json(indent=2))
        self._write_manifest()
        self.logger.info(result.token_line())
        return result

    def get(self, stage: str) -> StageResult | None:
        if stage in self._results:
            return self._results[stage]
        path = self.stages_dir / stage / "result.json"
        if path.exists():
            res = StageResult.model_validate_json(path.read_text())
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
        if not dc.path:
            raise RuntimeError(
                "No dataset on disk and config.data.path is unset. Pass a DataFrame via "
                "RunContext.attach_dataset() before running stages."
            )
        df = pd.read_parquet(dc.path) if dc.format == "parquet" else pd.read_csv(dc.path)
        df.to_parquet(cached, index=False)
        return df

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
            "task": self.config.task.value,
            "created_at": datetime.now(UTC).isoformat(),
            "stages": {
                s: (self._results[s].verdict.value if s in self._results else None)
                for s in self.config.stages.enabled
            },
        }
        with open(self.manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

    def _load_existing(self) -> None:
        for stage in self.config.stages.enabled:
            self.get(stage)  # populates cache from disk

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        done = ",".join(self.completed_stages()) or "-"
        return f"<RunContext {self.run_id} project={self.config.name!r} done=[{done}]>"
