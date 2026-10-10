"""Linked sources: several inputs joined into the one table a model is built on.

A profile may list several sources of any kind (``data.sources``: a Snowflake table and a CSV,
two files, ...). Everything here is mechanical:

- ``snapshot`` fetches each source once and keeps it under ``data/sources/``.
- ``profile`` measures how the tables connect: which columns could be keys, how many rows of
  one table find a match in the other, and whether a match is one row or many. These are the
  only candidate links there are; an agent chooses among them, it cannot invent one.
- ``resolve`` turns a choice (a base table and an ordered list of links) into join steps and
  says what is wrong with it.
- ``apply`` runs the steps. The base table's rows are the modelling rows: a join never adds
  or repeats one. A key that matches several rows is aggregated or reduced to one row first,
  or the join is refused. A base row with no match keeps missing values, or is dropped when
  the plan says so (counted and reported; nothing is dropped silently).

Who decides the plan is not decided here: the profile may state it (``data.join``), the Data
Analyst proposes it otherwise (``data_linker``), and the developer confirms or changes it at
the data gate (``overrides.join``). See stages/explore.py.
"""

from __future__ import annotations

import hashlib
import json
import re
from itertools import combinations
from typing import Any

import pandas as pd

from .fsutil import atomic_write

MAX_LINKS_PER_PAIR = 4
MIN_OVERLAP = 0.5  # share of one table's rows that must find a match for a link to be offered
MANY = ("aggregate", "first", "refuse")
UNMATCHED = ("keep", "drop")
# What makes dropping unmatched rows risky (unmatched_advice): too many of them, or an outcome
# that differs from the matched rows by more than chance.
DROP_MAX_SHARE = 0.10
OUTCOME_MIN_ROWS = 20
OUTCOME_Z = 2.0


class LinkError(ValueError):
    """The sources cannot be joined as asked."""


def is_linked(config: Any) -> bool:
    return len(config.data.sources) >= 2


def named_sources(files: list[str] = (), snowflake_lines: str = "") -> dict[str, Any]:
    """``{"sources": [...]}`` from what a person gives a front end: data files (each named
    after its file) and Snowflake lines reading ``name = TABLE`` or ``name = SELECT ...``."""
    from pathlib import Path

    sources: list[dict[str, Any]] = []

    def unique(name: str) -> str:
        name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "source"
        name = name if name[0].isalpha() else f"t_{name}"
        taken = {s["name"] for s in sources}
        return name if name not in taken else next(
            f"{name}_{i}" for i in range(2, 999) if f"{name}_{i}" not in taken)

    for path in files:
        sources.append({"name": unique(Path(path).stem), "kind": "file", "path": str(path)})
    for line in (snowflake_lines or "").splitlines():
        if not line.strip():
            continue
        name, sep, what = line.partition("=")
        if not sep or not name.strip() or not what.strip():
            raise LinkError("Write each Snowflake source as  name = TABLE  or  "
                            f"name = SELECT ...  ({line.strip()!r})")
        what = what.strip()
        is_query = bool(re.match(r"(?i)(select|with)\b", what))
        sources.append({"name": unique(name.strip()), "kind": "snowflake",
                        **({"query": what} if is_query else {"table": what})})
    if len(sources) < 2:
        raise LinkError("Give two or more sources: data files and/or Snowflake tables.")
    return {"sources": sources}


# --- fetching -----------------------------------------------------------------------------------
def snapshot(ctx: Any) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Every source as a DataFrame, fetched once and kept in the run (``data/sources/``), with
    the provenance of each. A re-run reads the copies, never the sources."""
    from . import datasources

    cfg = ctx.base_config
    folder = ctx.data_dir / "sources"
    record = ctx.data_dir / "sources.json"
    try:
        known = {p["name"]: p for p in json.loads(record.read_text(encoding="utf-8"))}
    except (FileNotFoundError, ValueError):
        known = {}
    tables: dict[str, pd.DataFrame] = {}
    provenance: list[dict[str, Any]] = []
    for src in cfg.data.sources:
        path = folder / f"{src.name}.parquet"
        if path.exists() and src.name in known:
            tables[src.name] = pd.read_parquet(path)
            provenance.append(known[src.name])
            continue
        try:
            df, prov = datasources.load_source(src, cfg.plugins)
        except datasources.SourceError as exc:
            raise LinkError(f"source {src.name!r}: {exc}") from exc
        folder.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        prov = {"name": src.name, **prov,
                "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        tables[src.name] = df
        provenance.append(prov)
    atomic_write(record, json.dumps(provenance, indent=1, default=str))
    return tables, provenance


# --- measuring how tables connect -------------------------------------------------------------
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _words(name: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", str(name).lower()) if w}


def _key_like(s: pd.Series) -> bool:
    """Could this column identify something: text or whole numbers, not a measure or a flag."""
    if pd.api.types.is_bool_dtype(s):
        return False
    if pd.api.types.is_float_dtype(s):  # whole numbers stored as floats (a CSV with gaps)
        whole = s.dropna()
        return bool(len(whole)) and bool((whole % 1 == 0).all())
    return (pd.api.types.is_integer_dtype(s) or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_object_dtype(s))


def _keys(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """The rows' key over ``cols`` as text, comparable across sources whatever each stored it
    as (``7`` in a database, ``"7"`` or ``7.0`` in a CSV). A row with any part missing has no
    key."""
    parts = []
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            num = pd.to_numeric(s, errors="coerce")
            whole = num.dropna()
            text = (num.astype("Int64").astype("string")
                    if len(whole) and (whole % 1 == 0).all() else num.astype("string"))
        else:
            text = s.astype("string").str.strip()
            as_num = pd.to_numeric(text, errors="coerce")
            if (text.notna().any() and as_num.notna().sum() == text.notna().sum()
                    and (as_num.dropna() % 1 == 0).all()):
                text = as_num.astype("Int64").astype("string")  # "7.0" and "007" read as 7
        parts.append(text)
    key = parts[0]
    for part in parts[1:]:
        key = key.str.cat(part, sep="\x1f")  # NA in any part makes the whole key NA
    return key


def _measure(a: pd.DataFrame, a_on: list[str], b: pd.DataFrame, b_on: list[str]) -> dict[str, Any]:
    ka, kb = _keys(a, a_on).dropna(), _keys(b, b_on).dropna()
    sa, sb = set(ka.unique()), set(kb.unique())
    return {
        "a_in_b": round(float(ka.isin(sb).mean()), 4) if len(ka) else 0.0,
        "b_in_a": round(float(kb.isin(sa).mean()), 4) if len(kb) else 0.0,
        "a_unique": bool(len(ka) and ka.is_unique), "b_unique": bool(len(kb) and kb.is_unique),
        "a_distinct": len(sa), "b_distinct": len(sb),
    }


def _outcome(df: pd.DataFrame, on: list[str], other: pd.DataFrame, other_on: list[str],
             target: str) -> dict[str, Any] | None:
    """The target among the rows of ``df`` that find a match in ``other`` and among those that
    do not: are the rows a join would lose like the ones it keeps? None when ``df`` does not
    hold the target or every row matches."""
    if not target or target not in df.columns:
        return None
    y = pd.to_numeric(df[target], errors="coerce")
    key = _keys(df, on)
    matched = (key.isin(set(_keys(other, other_on).dropna().unique())) & key.notna()).to_numpy()
    y1, y0 = y[matched].dropna(), y[~matched].dropna()
    if not len(y0) or not len(y1):
        return None
    out = {"target": target, "matched_rows": int(matched.sum()),
           "unmatched_rows": int((~matched).sum()),
           "matched_mean": round(float(y1.mean()), 4), "unmatched_mean": round(float(y0.mean()), 4),
           "z": None}
    se = (y1.var(ddof=0) / len(y1) + y0.var(ddof=0) / len(y0)) ** 0.5
    if len(y0) >= OUTCOME_MIN_ROWS and se > 0:
        out["z"] = round(float((y0.mean() - y1.mean()) / se), 2)
    return out


def unmatched_advice(step: dict[str, Any]) -> tuple[str, str]:
    """Keep or drop the base rows this join step leaves without a match, and why. A rule over
    the engine's measurements: drop only when the rows are few and, where the outcome can be
    compared, look like the rest."""
    share = 1 - step["expected_match"]
    if share <= 0:
        return "keep", "Every row finds a match."
    o = step.get("outcome")
    compared = ""
    if o:
        compared = (f" Their mean {o['target']} is {o['unmatched_mean']:.4g}, against "
                    f"{o['matched_mean']:.4g} for the matched rows")
        if o["z"] is not None and abs(o["z"]) >= OUTCOME_Z:
            return "keep", (f"{share:.1%} of the rows have no match and their outcome differs."
                            f"{compared} (z = {o['z']}). Dropping them would bias the "
                            "population the model is built on.")
        compared += (f" (z = {o['z']}, within chance)." if o["z"] is not None
                     else " (too few rows to test the difference).")
    if share > DROP_MAX_SHARE:
        return "keep", (f"{share:.1%} of the rows have no match: too many to drop without "
                        f"changing the population.{compared}")
    return "drop", (f"{share:.1%} of the rows have no match."
                    + (compared or " The outcome could not be compared (the target is not in "
                                   "this table or is not settled yet).")
                    + " Few enough to drop; the model families do not take missing values.")


def _relation(a_unique: bool, b_unique: bool) -> str:
    return {(True, True): "one-to-one", (False, True): "many-to-one",
            (True, False): "one-to-many", (False, False): "many-to-many"}[(a_unique, b_unique)]


def profile(tables: dict[str, pd.DataFrame], stated: list[Any] = (),
            target: str = "") -> dict[str, Any]:
    """The tables and every link worth offering between two of them, best first per pair.
    ``stated`` (the profile's ``data.join``) is always included, measured the same way. With a
    ``target``, each link also says how the outcome differs between the rows it matches and
    the rows it does not."""
    meta = []
    for name, df in tables.items():
        keyish = [c for c in df.columns if _key_like(df[c])]
        unique = [c for c in keyish if df[c].notna().all() and df[c].is_unique]
        meta.append({"name": name, "n_rows": int(len(df)), "n_cols": int(df.shape[1]),
                     "columns": [str(c) for c in df.columns], "unique_columns": unique})
    found: list[dict[str, Any]] = []

    def add(a: str, a_on: list[str], b: str, b_on: list[str], *, is_stated: bool = False) -> None:
        if any(lk["a"] == a and lk["b"] == b and lk["a_on"] == a_on and lk["b_on"] == b_on
               for lk in found):
            return
        m = _measure(tables[a], a_on, tables[b], b_on)
        same = [_norm(x) for x in a_on] == [_norm(y) for y in b_on]
        m["a_outcome"] = _outcome(tables[a], a_on, tables[b], b_on, target)
        m["b_outcome"] = _outcome(tables[b], b_on, tables[a], a_on, target)
        found.append({"a": a, "a_on": a_on, "b": b, "b_on": b_on, **m, "same_name": same,
                      "relation": _relation(m["a_unique"], m["b_unique"]), "stated": is_stated})

    for j in stated:
        for t, cols in ((j.left, j.left_on), (j.right, j.right_on)):
            missing = [c for c in cols if c not in tables[t].columns]
            if missing:
                raise LinkError(f"data.join: {t} has no column(s) {missing}")
        add(j.left, list(j.left_on), j.right, list(j.right_on), is_stated=True)

    for a, b in combinations(tables, 2):
        ta, tb = tables[a], tables[b]
        pair: list[dict[str, Any]] = []
        shared: list[tuple[str, str]] = []
        for ca in ta.columns:
            if not _key_like(ta[ca]):
                continue
            for cb in tb.columns:
                if not _key_like(tb[cb]):
                    continue
                same = _norm(ca) == _norm(cb)
                m = _measure(ta, [ca], tb, [cb])
                overlap = max(m["a_in_b"], m["b_in_a"])
                if overlap < MIN_OVERLAP:
                    continue
                if same:
                    shared.append((ca, cb))
                # A link needs more than overlapping values: the same name, or one side that
                # identifies its rows. Two unrelated small-integer columns overlap by chance.
                if not (same or m["a_unique"] or m["b_unique"]):
                    continue
                # Under different names: nearly every row matches, and the names share a
                # word (borrower_id / obligor_id) or the match holds in both directions.
                if not same and (overlap < 0.9 or not (
                        _words(ca) & _words(cb) or min(m["a_in_b"], m["b_in_a"]) >= 0.9)):
                    continue
                score = overlap + (0.5 if same else 0.0) + (
                    0.25 if m["a_unique"] or m["b_unique"] else 0.0)
                pair.append({"a": a, "a_on": [ca], "b": b, "b_on": [cb], "score": score})
        # Several shared columns that only together identify a row (obligor + period).
        if len(shared) >= 2:
            a_on, b_on = [x for x, _ in shared], [y for _, y in shared]
            m = _measure(ta, a_on, tb, b_on)
            single_unique = any(tables[x["a"]][x["a_on"][0]].is_unique
                                or tables[x["b"]][x["b_on"][0]].is_unique for x in pair)
            if max(m["a_in_b"], m["b_in_a"]) >= MIN_OVERLAP and (
                    (m["a_unique"] or m["b_unique"]) and not single_unique):
                pair.append({"a": a, "a_on": a_on, "b": b, "b_on": b_on,
                             "score": max(m["a_in_b"], m["b_in_a"]) + 1.0})
        pair.sort(key=lambda x: -x["score"])
        for x in pair[:MAX_LINKS_PER_PAIR]:
            add(x["a"], x["a_on"], x["b"], x["b_on"])
    for i, lk in enumerate(found, 1):
        lk["id"] = f"L{i}"
    return {"tables": meta, "links": found}


# --- a plan ---------------------------------------------------------------------------------------
def _oriented(link: dict[str, Any], left: str) -> dict[str, Any]:
    """The link as a join step that adds its other table to ``left``."""
    if link["a"] == left:
        l_on, right, r_on = link["a_on"], link["b"], link["b_on"]
        coverage, right_unique, outcome = link["a_in_b"], link["b_unique"], link.get("a_outcome")
    else:
        l_on, right, r_on = link["b_on"], link["a"], link["a_on"]
        coverage, right_unique, outcome = link["b_in_a"], link["a_unique"], link.get("b_outcome")
    return {"link": link["id"], "left": left, "left_on": list(l_on), "right": right,
            "right_on": list(r_on), "expected_match": coverage, "right_unique": right_unique,
            "outcome": outcome}


def resolve(base: str, choices: list[dict[str, str]], prof: dict[str, Any]
            ) -> tuple[list[dict[str, Any]], list[str]]:
    """Join steps for a base table and chosen links (``[{"link": id, "many": ...}]``), in an
    order that works, and everything that is wrong with the choice."""
    names = [t["name"] for t in prof["tables"]]
    links = {lk["id"]: lk for lk in prof["links"]}
    problems: list[str] = []
    if base not in names:
        return [], [f"base table {base!r} is not a source; choose from {names}"]
    pending = []
    for ch in choices:
        if ch.get("link") not in links:
            problems.append(f"unknown link {ch.get('link')!r}; choose from {sorted(links)}")
        elif ch["link"] in [p["link"] for p in pending]:
            problems.append(f"link {ch['link']} is listed twice")
        else:
            pending.append(ch)
    joined, steps = {base}, []
    while pending:
        nxt = next((ch for ch in pending
                    if len({links[ch["link"]]["a"], links[ch["link"]]["b"]} & joined) == 1), None)
        if nxt is None:
            for ch in pending:
                lk = links[ch["link"]]
                problems.append(
                    f"link {lk['id']} joins {lk['a']} and {lk['b']}, which are both already "
                    "joined (one link per added table)" if {lk["a"], lk["b"]} <= joined else
                    f"link {lk['id']} ({lk['a']} – {lk['b']}) does not reach the base table "
                    f"{base!r} through the other links")
            break
        pending.remove(nxt)
        lk = links[nxt["link"]]
        step = _oriented(lk, lk["a"] if lk["a"] in joined else lk["b"])
        many = nxt.get("many") or "refuse"
        if step["right_unique"]:
            many = "none"
        elif many not in ("aggregate", "first"):
            problems.append(
                f"link {lk['id']}: a key can match several rows of {step['right']}; say how to "
                "keep one row per base row (many: aggregate or first)")
        step["many"] = many
        unmatched = nxt.get("unmatched") or "keep"
        if unmatched not in UNMATCHED:
            problems.append(f"link {lk['id']}: unmatched is keep or drop, not {unmatched!r}")
        step["unmatched"] = unmatched
        steps.append(step)
        joined.add(step["right"])
    return steps, problems


def suggest(prof: dict[str, Any], base: str | None = None, target: str = "") -> dict[str, Any]:
    """A plan built from the measurements alone (what the deterministic Data Analyst proposes).
    Base: the stated one, else the table holding the target, else the table with most rows.
    Then, for each other table, the link to something already joined that matches most rows."""
    tables = prof["tables"]
    if base is None:
        holding = [t["name"] for t in tables if target and target in t["columns"]]
        base = holding[0] if holding else max(tables, key=lambda t: t["n_rows"])["name"]
    joined, chosen = {base}, []
    while True:
        best = None
        for lk in prof["links"]:
            inside = {lk["a"], lk["b"]} & joined
            if len(inside) != 1:
                continue
            step = _oriented(lk, next(iter(inside)))
            rank = (lk["stated"], lk["same_name"], step["right_unique"], step["expected_match"])
            if step["expected_match"] >= MIN_OVERLAP and (best is None or rank > best[0]):
                best = (rank, lk, step)
        if best is None:
            break
        _, lk, step = best
        unmatched, why = unmatched_advice(step)
        chosen.append({"link": lk["id"], "many": "none" if step["right_unique"] else "aggregate",
                       "unmatched": unmatched, "unmatched_reason": why})
        joined.add(step["right"])
    return {"base": base, "joins": chosen,
            "left_out": [t["name"] for t in tables if t["name"] not in joined]}


# --- running a plan -----------------------------------------------------------------------------
def _one_row_per_key(right: pd.DataFrame, key: str, name: str, many: str) -> pd.DataFrame:
    if many == "first":
        return right.drop_duplicates(key, keep="first")
    if many != "aggregate":
        raise LinkError(f"a key matches several rows of {name}; the plan must aggregate them or "
                        "keep the first")
    values = [c for c in right.columns if c != key]
    numeric = [c for c in values if pd.api.types.is_numeric_dtype(right[c])
               and not pd.api.types.is_bool_dtype(right[c])]
    how = {c: ("mean" if c in numeric else "first") for c in values}
    grouped = right.groupby(key, sort=False)
    out = grouped.agg(how).reset_index() if how else grouped.size().reset_index()[[key]]
    out[f"{name}__rows"] = grouped.size().to_numpy()
    return out


def apply(tables: dict[str, pd.DataFrame], base: str, steps: list[dict[str, Any]]
          ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the join steps on the base table. Returns the modelling table and what happened at
    each step. Rows keep the base table's order; none is added, and one is dropped only by a
    step that says ``unmatched: drop``."""
    if base not in tables:
        raise LinkError(f"base table {base!r} is not a source")
    out = tables[base].reset_index(drop=True).copy()
    n = base_rows = len(out)
    origin = {c: base for c in out.columns}
    # where each source column lives in the joined table (a later step may join on it)
    place = {(base, c): c for c in out.columns}
    report = []
    for step in steps:
        left, right_name = step["left"], step["right"]
        if right_name not in tables:
            raise LinkError(f"{right_name!r} is not a source")
        right = tables[right_name]
        try:
            left_cols = [place[(left, c)] for c in step["left_on"]]
        except KeyError as exc:
            raise LinkError(f"{left} has no column {exc.args[0][1]!r} in the joined table") from exc
        missing = [c for c in step["right_on"] if c not in right.columns]
        if missing:
            raise LinkError(f"{right_name} has no column(s) {missing}")
        key = "__cognos_key"
        r = right.drop(columns=list(step["right_on"])).copy()
        r[key] = _keys(right, list(step["right_on"])).to_numpy()
        r = r[r[key].notna()]
        duplicated = bool(r[key].duplicated().any())
        if duplicated:
            r = _one_row_per_key(r, key, right_name, step.get("many", "refuse"))
        renamed = {}
        for c in r.columns:
            if c == key:
                continue
            new = c if c not in out.columns else f"{right_name}__{c}"
            renamed[c] = new
            origin[new] = right_name
            place[(right_name, c)] = new
        for lc, rc in zip(step["left_on"], step["right_on"], strict=True):
            place[(right_name, rc)] = place[(left, lc)]  # the key itself lives in the left column
        r = r.rename(columns=renamed)
        out[key] = _keys(out, left_cols).to_numpy()
        matched = out[key].isin(set(r[key])) & out[key].notna()
        out = out.merge(r, on=key, how="left", sort=False).drop(columns=[key])
        if len(out) != n:  # cannot happen with one row per key; never let it pass silently
            raise LinkError(f"joining {right_name} changed the number of rows ({n} to {len(out)})")
        dropped = 0
        if step.get("unmatched") == "drop" and not matched.all():
            dropped = int((~matched).sum())
            out = out[matched.to_numpy()].reset_index(drop=True)
            n = len(out)
            if not n:
                raise LinkError(f"no base row finds a match in {right_name}; nothing is left")
        report.append({
            "link": step.get("link", ""), "right": right_name, "left": left,
            "on": [f"{left}.{lc} = {right_name}.{rc}"
                   for lc, rc in zip(step["left_on"], step["right_on"], strict=True)],
            "matched_rows": int(matched.sum()), "match_rate": round(float(matched.mean()), 4),
            "dropped_rows": dropped,
            "right_rows": int(len(right)),
            "reduced": (step.get("many") if duplicated else "none"),
            "columns_added": list(renamed.values()),
        })
    return out, {"base": base, "base_rows": base_rows, "n_rows": n,
                 "n_cols": int(out.shape[1]), "steps": report,
                 "column_origin": origin}


def plan_digest(base: str, steps: list[dict[str, Any]]) -> str:
    body = {"base": base, "steps": [
        {**{k: s[k] for k in ("left", "left_on", "right", "right_on", "many")},
         "unmatched": s.get("unmatched", "keep")} for s in steps]}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
