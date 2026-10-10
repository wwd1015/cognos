"""Data sources: where a run's dataset comes from.

A source is a small connector with one job: return a DataFrame for a ``SourceConfig``. Three ship
with COGNOS (``file`` for a local or uploaded file, ``sqlite``, ``snowflake``); a plugin may
register more (plugins.py). The engine fetches once, when explore first needs the data, keeps the
snapshot in ``runs/<id>/data/dataset.parquet`` and records its provenance beside it
(``data/source.json``: the connector, the table or query, the row count, when, and the snapshot's
hash). Every later stage reads the snapshot, so a run never depends on the database staying still.

Secrets never enter a profile or a run directory: a connector reads passwords and keys from the
environment only, and an option written ``env:NAME`` is resolved from the environment at fetch time.
A missing driver or credential makes the source *unavailable* with a message, never a crash at
import.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


class SourceUnavailable(RuntimeError):
    """The connector cannot run here (driver not installed, credentials not set)."""


class SourceError(ValueError):
    """The source is misconfigured (no table, a query that is not a single read-only SELECT…)."""


@dataclass
class DataSource:
    kind: str
    label: str
    load: Callable[[Any], pd.DataFrame]  # SourceConfig -> DataFrame
    available: Callable[[], tuple[bool, str]] = lambda: (True, "")
    describe: Callable[[Any], str] = lambda source: source.table or source.path or "query"
    origin: str = "cognos"


_IDENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_"][A-Za-z0-9_$"]*){0,2}$')
_WRITE_RE = re.compile(r"\b(insert|update|delete|merge|drop|alter|create|truncate|grant|revoke|"
                       r"call|copy|put|remove|use|set|unset|execute|begin|commit)\b", re.I)


def statement(source: Any) -> str:
    """The one read-only statement a database source runs. A table is selected whole; a query
    must be a single SELECT (or WITH … SELECT). The row cap wraps either."""
    if source.query:
        sql = source.query.strip().rstrip(";").strip()
        if ";" in sql:
            raise SourceError("the query must be a single statement")
        if not re.match(r"(?is)^\s*(select|with)\b", sql) or _WRITE_RE.search(sql):
            raise SourceError("the query must be one read-only SELECT")
    elif source.table:
        if not _IDENT_RE.match(source.table):
            raise SourceError(f"not a table name: {source.table!r}")
        sql = f"SELECT * FROM {source.table}"
    else:
        raise SourceError(f"a {source.kind} source needs a table or a query")
    if source.limit:
        sql = f"SELECT * FROM ({sql}) AS cognos_src LIMIT {int(source.limit)}"
    return sql


def option(source: Any, name: str, env: str | None = None) -> str | None:
    """A connector setting: the profile's option (``env:NAME`` is read from the environment),
    else the conventional environment variable."""
    value = source.options.get(name)
    if value and value.startswith("env:"):
        return os.environ.get(value[4:]) or None
    return value or (os.environ.get(env) if env else None)


# --- file ---------------------------------------------------------------------------------------
def read_file(path: str | Path, fmt: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise SourceError(f"data file not found: {path}")
    kind = (fmt or path.suffix.lstrip(".")).lower()
    if kind == "parquet":
        return pd.read_parquet(path)
    if kind in ("xlsx", "xls", "xlsm"):
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise SourceUnavailable("reading Excel needs the optional openpyxl package; save "
                                    "the sheet as .csv or install openpyxl") from exc
    if kind in ("json", "jsonl", "ndjson"):
        return pd.read_json(path, lines=kind != "json")
    if kind in ("tsv", "tab"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _load_file(source: Any) -> pd.DataFrame:
    if not source.path:
        raise SourceError("a file source needs a path")
    df = read_file(source.path, source.options.get("format"))
    return df.head(source.limit) if source.limit else df


# --- sqlite -------------------------------------------------------------------------------------
def _load_sqlite(source: Any) -> pd.DataFrame:
    import sqlite3

    if not source.path or not Path(source.path).is_file():
        raise SourceError(f"SQLite database not found: {source.path}")
    con = sqlite3.connect(f"file:{Path(source.path).resolve()}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(statement(source), con)
    finally:
        con.close()


# --- snowflake ----------------------------------------------------------------------------------
_SNOWFLAKE_ENV = {"account": "SNOWFLAKE_ACCOUNT", "user": "SNOWFLAKE_USER",
                  "warehouse": "SNOWFLAKE_WAREHOUSE", "database": "SNOWFLAKE_DATABASE",
                  "schema": "SNOWFLAKE_SCHEMA", "role": "SNOWFLAKE_ROLE",
                  "authenticator": "SNOWFLAKE_AUTHENTICATOR"}


def _snowflake_connector():
    try:
        import snowflake.connector as connector
    except ImportError as exc:
        raise SourceUnavailable("Snowflake needs the optional snowflake-connector-python package "
                                "(pip install 'cognos[snowflake]')") from exc
    return connector


def _snowflake_available() -> tuple[bool, str]:
    try:
        _snowflake_connector()
    except SourceUnavailable as exc:
        return False, str(exc)
    missing = [env for env in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER") if not os.environ.get(env)]
    return (not missing, f"set {', '.join(missing)} (or name them in data.source.options)"
            if missing else "")


def _load_snowflake(source: Any) -> pd.DataFrame:
    connector = _snowflake_connector()
    sql = statement(source)
    args = {k: v for k in _SNOWFLAKE_ENV if (v := option(source, k, _SNOWFLAKE_ENV[k]))}
    for need in ("account", "user"):
        if need not in args:
            raise SourceUnavailable(f"Snowflake {need} is not set: set {_SNOWFLAKE_ENV[need]} or "
                                    f"data.source.options.{need}")
    # Secrets come from the environment only; the profile may name the variable, never the value.
    password = os.environ.get(source.options.get("password_env", "SNOWFLAKE_PASSWORD"))
    key_file = os.environ.get(source.options.get("private_key_file_env",
                                                 "SNOWFLAKE_PRIVATE_KEY_FILE"))
    if key_file:
        args["private_key_file"] = key_file
    elif password:
        args["password"] = password
    elif "authenticator" not in args:
        raise SourceUnavailable("no Snowflake credential: set SNOWFLAKE_PASSWORD or "
                                "SNOWFLAKE_PRIVATE_KEY_FILE, or an authenticator such as "
                                "externalbrowser")
    con = connector.connect(**args, application="COGNOS")
    try:
        cur = con.cursor()
        cur.execute(sql)
        if hasattr(cur, "fetch_pandas_all"):
            return cur.fetch_pandas_all()
        return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    finally:
        con.close()


BUILTIN: tuple[DataSource, ...] = (
    DataSource("file", "A local or uploaded file (.csv, .parquet, .xlsx, .json)", _load_file,
               describe=lambda s: Path(s.path or "").name),
    DataSource("sqlite", "SQLite database", _load_sqlite,
               describe=lambda s: f"{Path(s.path or '').name}:{s.table or 'query'}"),
    DataSource("snowflake", "Snowflake", _load_snowflake, _snowflake_available),
)


def effective_source(config: Any):
    """The source a profile means: ``data.source``, the older ``data.path`` + ``data.format``,
    or the only entry of ``data.sources``. (Several sources are joined first: linking.py.)"""
    from .config import SourceConfig

    dc = config.data
    if dc.source is not None:
        return dc.source
    if dc.path:
        return SourceConfig(kind="file", path=dc.path, options={"format": dc.format})
    if len(dc.sources) == 1:
        return dc.sources[0]
    return None


def load_source(source: Any, plugin_modules: Any = ()) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch one source and say where it came from. No secret is part of the provenance."""
    from . import plugins

    connector = plugins.registry(plugin_modules).sources.get(source.kind)
    if connector is None:
        raise SourceError(f"unknown data source {source.kind!r}; known: "
                          f"{sorted(plugins.registry(plugin_modules).sources)}")
    df = connector.load(source)
    if df.empty:
        raise SourceError(f"the {source.kind} source returned no rows")
    df.columns = [str(c) for c in df.columns]
    return df, {
        "kind": source.kind, "connector": connector.label, "origin": connector.origin,
        "location": connector.describe(source), "table": source.table, "query": source.query,
        "limit": source.limit, "options": {k: v for k, v in source.options.items()
                                           if "password" not in k.lower() and "key" not in k.lower()},
        "n_rows": int(len(df)), "n_cols": int(df.shape[1]),
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def load(config: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch the dataset and say where it came from."""
    source = effective_source(config)
    if source is None:
        raise SourceError("no data source: set data.source or data.path")
    return load_source(source, config.plugins)
