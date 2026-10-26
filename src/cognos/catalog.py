"""The data catalog (ADR-0020): the tables a team keeps, what each is for, and how they join.

COGNOS ships a default catalog (``default_catalog/catalog.yaml``): the tables a commercial-credit
team typically keeps, with synthetic data for every table (written to
``<runs>/_catalog/default_data/`` on first use), so every built-in template runs out of the box.
It is in force while the runs folder has no catalog of its own. A team writes
``<runs>/_catalog/catalog.yaml`` (``init`` starts it from the default) and it then replaces the
default entirely. Mechanical, like the rest of the plumbing: it reads the file, measures a table
when asked (rows, columns, the date range, whether the keys identify a row) and turns a
selection plus any added data into a run's ``data`` mapping. Which tables to join, and on what,
is still decided at explore: the catalog's links are offered and measured, never imposed.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import JoinStep, SourceConfig
from .fsutil import atomic_write

DEFAULT = Path(__file__).parent / "default_catalog" / "catalog.yaml"
DEFAULT_DATA = "default_data"  # the default catalog's synthetic tables, under <runs>/_catalog/


class CatalogTable(BaseModel):
    id: str
    title: str = ""
    description: str = ""  # what the table is and what it is used for
    grain: str = ""  # what one row is
    keys: list[str] = Field(default_factory=list)  # the columns that identify a row
    datetime_col: str | None = None
    outcomes: list[str] = Field(default_factory=list)  # outcome columns: never features
    owner: str = ""
    refresh: str = ""
    source: SourceConfig
    columns: dict[str, str] = Field(default_factory=dict)  # column -> what it means


class Catalog(BaseModel):
    title: str = ""
    defaults: list[str] = Field(default_factory=list)  # preselected for a run from scratch
    tables: list[CatalogTable] = Field(default_factory=list)
    links: list[JoinStep] = Field(default_factory=list)

    def table(self, tid: str) -> CatalogTable:
        for t in self.tables:
            if t.id == tid:
                return t
        raise KeyError(f"the data catalog has no table {tid!r}")


def folder(root: str | Path | None = None) -> Path:
    from .service import runs_root

    return runs_root(root) / "_catalog"


def team_file(root: str | Path | None = None) -> Path:
    """Where a runs folder's own catalog lives. While it is absent the default is in force."""
    return folder(root) / "catalog.yaml"


def in_force(root: str | Path | None = None) -> tuple[Path, str]:
    """The catalog file a run reads and where it comes from: ``team`` (the runs folder's own)
    or ``default`` (shipped with COGNOS, its synthetic data made on first use)."""
    team = team_file(root)
    if team.exists():
        return team, "team"
    _default_data(root)
    return DEFAULT, "default"


def _default_data(root: str | Path | None = None) -> Path:
    """The default catalog's synthetic tables, made once per runs folder."""
    data = folder(root) / DEFAULT_DATA
    done = data / ".made"
    if done.exists():
        return data
    from . import synth

    data.mkdir(parents=True, exist_ok=True)
    for name, df in synth.make_catalog_tables().items():
        tmp = data / f".{name}.{os.getpid()}.parquet"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, data / f"{name}.parquet")
    atomic_write(done, "")  # written last: the marker means every table is there
    return data


def init(root: str | Path | None = None) -> Path:
    """Start the runs folder's own catalog from the default, to edit. Refuses to overwrite."""
    team = team_file(root)
    if team.exists():
        raise FileExistsError(f"{team} already exists")
    _default_data(root)
    tmp = team.with_suffix(f".{os.getpid()}.tmp")
    shutil.copyfile(DEFAULT, tmp)
    os.replace(tmp, team)
    return team


def load(root: str | Path | None = None) -> Catalog:
    import yaml

    path, _ = in_force(root)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cat = Catalog.model_validate(raw)
    ids = [t.id for t in cat.tables]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: every catalog table needs its own id")
    for t in cat.tables:  # a relative path is read from <runs>/_catalog/
        if t.source.path and not Path(t.source.path).is_absolute():
            t.source.path = str(folder(root) / t.source.path)
    return cat


# --- what a table holds (measured, cached) ------------------------------------------------------
def _stamp(src: SourceConfig) -> str | None:
    """What a cached summary is valid for: a file's size and time. A database table has none:
    it is measured only when asked (``measure(..., fetch=True)``)."""
    if src.kind != "file" or not src.path:
        return None
    try:
        st = Path(src.path).stat()
    except OSError:
        return None
    return f"{src.path}:{st.st_size}:{st.st_mtime_ns}"


def measure(table: CatalogTable, root: str | Path | None = None, *,
            fetch: bool = False) -> dict[str, Any]:
    """Rows, columns, the date range and whether the keys identify a row. Read from the cache
    while the file is unchanged; a database table is read only with ``fetch``."""
    from . import datasources

    cache_path = folder(root) / "summaries.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        cache = {}
    stamp = _stamp(table.source)
    known = cache.get(table.id)
    if known and (known.get("stamp") == stamp if stamp else not fetch):
        return known
    if stamp is None and not fetch:
        return {"measured": False, "note": "measured when a run reads it"}
    try:
        df, _ = datasources.load_source(table.source)
    except (datasources.SourceError, datasources.SourceUnavailable) as exc:
        return {"measured": False, "error": str(exc)}
    out: dict[str, Any] = {"measured": True, "stamp": stamp, "n_rows": int(len(df)),
                           "n_cols": int(df.shape[1])}
    if table.datetime_col and table.datetime_col in df.columns:
        dates = df[table.datetime_col].dropna().astype(str)
        if len(dates):
            out["from"], out["to"] = str(dates.min())[:10], str(dates.max())[:10]
    keys = [k for k in table.keys if k in df.columns]
    if keys and len(keys) == len(table.keys):
        out["keys_unique"] = bool(not df.duplicated(keys).any())
    # The catalog and the data disagree: described columns that are not there, and columns
    # nobody described.
    out["missing_columns"] = [c for c in table.columns if c not in df.columns]
    out["undescribed"] = [str(c) for c in df.columns if c not in table.columns]
    cache[table.id] = out
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(cache_path, json.dumps(cache, indent=1))
    return out


def listing(root: str | Path | None = None) -> dict[str, Any]:
    """The catalog as a page shows it: every table with its description and what it holds."""
    cat = load(root)
    path, origin = in_force(root)
    return {"title": cat.title, "origin": origin, "file": str(path),
            "team_file": str(team_file(root)), "defaults": list(cat.defaults),
            "tables": [{**t.model_dump(mode="json", exclude={"source"}),
                        "location": t.source.table or t.source.query or (
                            Path(t.source.path).name if t.source.path else ""),
                        "kind": t.source.kind, "summary": measure(t, root)}
                       for t in cat.tables],
            "links": [lk.model_dump(mode="json") for lk in cat.links]}


def preselected(config: Any, root: str | Path | None = None) -> list[str]:
    """What gate_source starts with checked: the template's tables, else (a run from scratch)
    the catalog's defaults. Ids the catalog no longer has are left out."""
    cat = load(root)
    known = {t.id for t in cat.tables}
    ids = config.data.catalog if config.template is not None else cat.defaults
    return [i for i in ids if i in known]


def summary_line(t: dict[str, Any]) -> str:
    """One line on what a catalog table holds: its grain, size, dates and keys."""
    m = t.get("summary") or {}
    parts = [t.get("grain") or ""]
    if m.get("measured"):
        parts.append(f"{m['n_rows']:,} rows × {m['n_cols']} columns")
        if m.get("from"):
            parts.append(f"{m['from']} to {m['to']}")
    elif m.get("error"):
        parts.append(f"cannot be read: {m['error']}")
    elif m.get("note"):
        parts.append(m["note"])
    if t.get("keys"):
        parts.append("keys " + ", ".join(t["keys"]) + (
            "" if m.get("keys_unique", True) else " (repeat: not one row per key)"))
    return " · ".join(p for p in parts if p)


# --- a selection as a run's data ---------------------------------------------------------------
def data_spec(ids: list[str], added: list[dict[str, Any]] = (), *,
              datetime_col: str | None = None,
              root: str | Path | None = None) -> dict[str, Any]:
    """The ``data`` mapping for the catalog tables ``ids`` (in that order; the first is the
    proposed base table) and any ``added`` sources (``{"name", "kind", "path" | "table" |
    "query"}``). Each catalog table carries its description and column meanings into the run;
    the links the catalog documents between the chosen tables come with them."""
    cat = load(root)
    ids = list(dict.fromkeys(ids))
    chosen = [cat.table(i) for i in ids]
    sources = [{"name": t.id, **t.source.model_dump(mode="json", exclude_defaults=True),
                "catalog": t.id, "description": t.description, "columns": dict(t.columns),
                "keys": list(t.keys), "outcomes": list(t.outcomes)}
               for t in chosen]
    taken = set(ids)
    for a in added:
        name, n = a["name"], 2
        while name in taken:
            name, n = f"{a['name']}_{n}", n + 1
        taken.add(name)
        sources.append({**a, "name": name})
    if not sources:
        raise ValueError("choose a table from the data catalog or add data")
    date = datetime_col or (chosen[0].datetime_col if chosen else None)
    if len(sources) == 1:
        one = {k: v for k, v in sources[0].items() if k != "name"}
        return {"source": one, **({"datetime_col": date} if date else {})}
    links = [lk.model_dump(mode="json", exclude_defaults=True) for lk in cat.links
             if lk.left in taken and lk.right in taken]
    return {"sources": sources, **({"base": chosen[0].id} if chosen else {}),
            **({"links": links} if links else {}), **({"datetime_col": date} if date else {})}
