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

**Catalog sources** (ADR-0022). Everything a page or a run asks of the catalog goes through a
:class:`CatalogSource`: ``search`` (tables for some words and the run's intent), ``describe``
(one table: schema, column meanings, keys, outcomes), ``links`` (documented joins among some
tables), ``defaults`` and, optionally, ``assess`` (does each table fit the intent, with
reasons). The YAML catalog above is the built-in source. A plugin adds another
(``registry.add_catalog``), for instance one that reads an internal metadata system, and
``COGNOS_CATALOG`` names the one in force. A source informs, it never decides: its assessment
is advice shown with its name, and without one the engine assesses mechanically (:func:`fit`).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Literal

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
    """The catalog as the CLI shows it: every table with its description and what it holds.
    A source that cannot list everything gives what an empty search returns."""
    src = source(root)
    tables = src.search("", None, limit=10_000)
    meta = src.info()
    return {"title": src.title, "origin": meta.get("kind") or src.name,
            "file": meta.get("file", ""), "team_file": meta.get("team_file", ""),
            "source": src.name, "defaults": src.defaults(),
            "tables": [card(t, root) for t in tables],
            "links": [lk.model_dump(mode="json") for lk in src.links([t.id for t in tables])]}


def preselected(config: Any, root: str | Path | None = None) -> list[str]:
    """What gate_source starts with checked: the template's tables, else (a run from scratch)
    the catalog's defaults. Ids the catalog no longer has are left out."""
    src = source(root)
    ids = config.data.catalog if config.template is not None else src.defaults()
    return [i for i in ids if _known(src, i)]


def _known(src: CatalogSource, table_id: str) -> bool:
    try:
        src.describe(table_id)
    except KeyError:
        return False
    return True


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
    src = source(root)
    ids = list(dict.fromkeys(ids))
    chosen = [src.describe(i) for i in ids]
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
    links = [lk.model_dump(mode="json", exclude_defaults=True) for lk in src.links(ids)
             if lk.left in taken and lk.right in taken]
    return {"sources": sources, **({"base": chosen[0].id} if chosen else {}),
            **({"links": links} if links else {}), **({"datetime_col": date} if date else {})}


# --- catalog sources (ADR-0022) -------------------------------------------------------------
class Fit(BaseModel):
    """Whether a table suits the run's intent, and why: advice, never a decision."""

    table: str
    level: Literal["fits", "partly", "unclear"]
    reasons: list[str] = Field(default_factory=list)
    origin: str = "engine"  # who said so: "engine" (the mechanical check) or a source's name


class CatalogSource:
    """Where the catalog's tables come from. Subclass it in a plugin and register it with
    ``registry.add_catalog``; ``COGNOS_CATALOG`` names the source in force. ``search`` must
    work with a few hundred tables (it is called as a person types); ``describe`` must raise
    ``KeyError`` for an unknown id. ``assess`` may return ``None``: the engine then assesses
    mechanically. Whatever a source returns is configuration and advice, never a secret."""

    name = "catalog"
    title = ""
    origin = "cognos"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def search(self, text: str, intent: dict[str, Any] | None = None,
               limit: int = 20) -> list[CatalogTable]:
        raise NotImplementedError

    def describe(self, table_id: str) -> CatalogTable:
        raise NotImplementedError

    def links(self, ids: list[str]) -> list[JoinStep]:
        return []

    def defaults(self) -> list[str]:
        return []

    def assess(self, ids: list[str], intent: dict[str, Any] | None = None
               ) -> list[Fit] | None:
        return None

    def info(self) -> dict[str, Any]:
        """What a page says about the source: its title and where it comes from."""
        return {"name": self.name, "title": self.title, "origin": self.origin}


class YamlCatalog(CatalogSource):
    """The built-in source: the runs folder's ``catalog.yaml``, else the default."""

    name = "yaml"

    def __init__(self, root: str | Path | None = None):
        self.root = root
        self.cat = load(root)
        self.path, self.kind = in_force(root)
        self.title = self.cat.title

    def search(self, text: str, intent: dict[str, Any] | None = None,
               limit: int = 20) -> list[CatalogTable]:
        return rank(self.cat.tables, text, intent)[:limit]

    def describe(self, table_id: str) -> CatalogTable:
        return self.cat.table(table_id)

    def links(self, ids: list[str]) -> list[JoinStep]:
        keep = set(ids)
        return [lk for lk in self.cat.links if lk.left in keep and lk.right in keep]

    def defaults(self) -> list[str]:
        return list(self.cat.defaults)

    def info(self) -> dict[str, Any]:
        return {**super().info(), "kind": self.kind, "file": str(self.path),
                "team_file": str(team_file(self.root)), "n_tables": len(self.cat.tables)}


def source(root: str | Path | None = None) -> CatalogSource:
    """The catalog source in force: the plugin source ``COGNOS_CATALOG`` names when it is
    registered and available, else the YAML catalog. A named source that is missing or
    unavailable is an error, never a silent switch to other tables."""
    want = os.environ.get("COGNOS_CATALOG", "").strip()
    if not want or want == YamlCatalog.name:
        return YamlCatalog(root)
    from . import plugins

    found = plugins.registry().catalogs.get(want)
    if found is None:
        raise ValueError(f"COGNOS_CATALOG names {want!r}, which no plugin registers "
                         "(see cognos plugins)")
    ok, why = found.available()
    if not ok:
        raise ValueError(f"the catalog source {want!r} is not available: {why}")
    return found


_WORD = re.compile(r"[a-z0-9]+")
_COMMON = {"the", "a", "an", "of", "and", "or", "to", "for", "in", "on", "at", "by", "is",
           "with", "per", "one", "row", "rows", "each", "every", "its", "it", "be", "as", "from",
           "model", "data", "table"}


def words(*texts: Any) -> set[str]:
    """The words of some text that say something (lower case, no filler)."""
    out: set[str] = set()
    for t in texts:
        out |= {w for w in _WORD.findall(str(t or "").lower().replace("_", " "))
                if w not in _COMMON and len(w) > 1}
    return out


def intent_words(intent: dict[str, Any] | None) -> set[str]:
    intent = intent or {}
    return words(intent.get("objective"), *(intent.get("fields") or {}).values())


def rank(tables: list[CatalogTable], text: str, intent: dict[str, Any] | None = None
         ) -> list[CatalogTable]:
    """Tables matching ``text`` (every word must appear in the id, title, description, a
    column or its meaning), best first; ties go to the tables the intent's words mention."""
    asked = words(text)
    goal = intent_words(intent)
    scored = []
    for i, t in enumerate(tables):
        name = words(t.id, t.title)
        body = words(t.description, t.grain, *t.outcomes)
        cols = words(*t.columns, *t.columns.values())
        if asked and not all(any(w in s or any(x.startswith(w) for x in s)
                                 for s in (name, body, cols)) for w in asked):
            continue
        score = sum(3 if w in name else 2 if w in body else 1 for w in asked)
        scored.append((-score, -len(goal & (name | body)), i, t))
    return [t for *_, t in sorted(scored, key=lambda x: x[:3])]


def card(t: CatalogTable, root: str | Path | None = None) -> dict[str, Any]:
    """A table as a page shows it: what it is, what it holds (measured when it can be)."""
    return {**t.model_dump(mode="json", exclude={"source"}),
            "location": t.source.table or t.source.query or (
                Path(t.source.path).name if t.source.path else ""),
            "kind": t.source.kind, "summary": measure(t, root)}


def fit(src: CatalogSource, ids: list[str], intent: dict[str, Any] | None = None,
        root: str | Path | None = None) -> list[Fit]:
    """Does each chosen table suit the intent? The source's own assessment when it gives one
    (labelled with its name); else the engine's mechanical one: the first table is the base,
    so it should carry an outcome; every other table should be linked to one already chosen;
    the dates should overlap the base table's. Advice for the developer, never a decision."""
    ids = list(dict.fromkeys(ids))
    given = src.assess(ids, intent)
    if given is not None:
        return [f if f.origin != "engine" else f.model_copy(update={"origin": src.name})
                for f in given]
    tables: dict[str, CatalogTable] = {}
    for i in ids:
        try:
            tables[i] = src.describe(i)
        except KeyError:
            continue
    links = src.links(list(tables))
    goal = intent_words(intent)
    sizes = {i: measure(t, root) for i, t in tables.items()}
    out = []
    for n, (i, t) in enumerate(tables.items()):
        reasons: list[str] = []
        level = "fits"
        if n == 0:
            if t.outcomes:
                reasons.append(f"Carries the outcome {', '.join(t.outcomes)}, so it can be "
                               "the base table.")
            else:
                level = "partly"
                other = next((o for o in tables.values() if o.outcomes), None)
                reasons.append("Has no outcome column, so the dependent variable must come "
                               "from another table" + (f"; {other.id} has one and may be the "
                                                       "better base table." if other else "."))
        else:
            linked = [lk for lk in links if i in (lk.left, lk.right)]
            shared = [o.id for o in list(tables.values())[:n]
                      if set(t.keys) & set(o.keys) or set(t.keys) & set(o.columns)]
            if linked:
                lk = linked[0]
                other = lk.right if lk.left == i else lk.left
                on = lk.right_on if lk.left == i else lk.left_on
                reasons.append(f"The catalog documents how it joins {other} "
                               f"(on {', '.join(on)}).")
            elif shared:
                reasons.append(f"Shares a key with {shared[0]}, but the catalog documents no "
                               "join: the engine measures one at explore.")
            else:
                level = "partly"
                reasons.append("No documented join to the tables chosen before it, and no "
                               "shared key: it may not connect.")
            base = sizes.get(ids[0]) or {}
            mine = sizes.get(i) or {}
            if base.get("from") and mine.get("from") and (
                    mine["to"] < base["from"] or mine["from"] > base["to"]):
                level = "partly"
                reasons.append(f"Its dates ({mine['from'][:4]}–{mine['to'][:4]}) do not "
                               f"overlap the base table's ({base['from'][:4]}–"
                               f"{base['to'][:4]}).")
        said = goal & words(t.title, t.description, t.grain)
        if said:
            reasons.append("Its description shares words with the intent: "
                           + ", ".join(sorted(said)[:5]) + ".")
        if not goal and n == 0 and not t.outcomes:
            level = "unclear"
        out.append(Fit(table=i, level=level, reasons=reasons))
    return out
