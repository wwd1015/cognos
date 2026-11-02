"""An example catalog source (ADR-0022): the data catalog read from an internal metadata system.

A bank's metadata system knows every table: its schema, what each field means, its keys, how
tables relate and who owns them. This plugin stands in for one. It reads an export of that
system (a JSON file named by ``METADATA_EXPORT``) and answers the four questions COGNOS asks a
catalog source: ``search``, ``describe``, ``links`` and ``defaults``. A real one would call the
system's API in the same places, or a Claude skill that does; nothing else in COGNOS changes.

Load it and put it in force::

    export PYTHONPATH=examples/plugins
    export COGNOS_PLUGINS=metadata_catalog COGNOS_CATALOG=metadata
    export METADATA_EXPORT=/path/to/export.json
    cognos plugins     # lists the source and whether it is available
    cognos catalog     # the tables it offers

The export, in the metadata system's own words::

    {"system": "Acme metadata",
     "datasets": [{"name": "obligor_quarters", "label": "Obligor financials by quarter",
                   "summary": "...", "grain": "one row per obligor and quarter end",
                   "primary_key": ["obligor_id", "as_of_date"], "event_time": "as_of_date",
                   "targets": ["default_12m"], "steward": "Credit Risk Analytics",
                   "location": {"kind": "file", "path": "/data/obligor_quarters.parquet"},
                   "fields": [{"name": "leverage", "definition": "Total debt / total assets"}]}],
     "relationships": [{"from": "facilities", "to": "obligor_quarters",
                        "on": [["obligor_id", "obligor_id"], ["as_of_date", "as_of_date"]],
                        "comment": "..."}],
     "recommended": ["obligor_quarters"]}

``assess`` is left to the engine here (it returns ``None``). A source that can judge fit
better, for instance a skill that reads the model's intent against the metadata, returns
``catalog.Fit`` items: they are shown as advice with the source's name, and the developer
still decides.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from cognos.catalog import CatalogSource, CatalogTable, rank
from cognos.config import JoinStep, SourceConfig


class MetadataCatalog(CatalogSource):
    name = "metadata"

    def _export(self) -> dict[str, Any]:
        path = os.environ.get("METADATA_EXPORT", "")
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def available(self) -> tuple[bool, str]:
        path = os.environ.get("METADATA_EXPORT", "")
        if not path:
            return False, "set METADATA_EXPORT to the metadata system's export"
        if not Path(path).exists():
            return False, f"{path} does not exist"
        return True, ""

    def _tables(self) -> list[CatalogTable]:
        out = []
        for d in self._export().get("datasets", []):
            out.append(CatalogTable(
                id=d["name"], title=d.get("label", ""), description=d.get("summary", ""),
                grain=d.get("grain", ""), keys=list(d.get("primary_key") or []),
                datetime_col=d.get("event_time"), outcomes=list(d.get("targets") or []),
                owner=d.get("steward", ""), source=SourceConfig(**d["location"]),
                columns={f["name"]: f.get("definition", "") for f in d.get("fields", [])}))
        return out

    @property
    def title(self) -> str:  # type: ignore[override]
        return self._export().get("system", "Metadata system") if self.available()[0] else ""

    def search(self, text: str, intent: dict[str, Any] | None = None,
               limit: int = 20) -> list[CatalogTable]:
        return rank(self._tables(), text, intent)[:limit]

    def describe(self, table_id: str) -> CatalogTable:
        for t in self._tables():
            if t.id == table_id:
                return t
        raise KeyError(f"the metadata system has no dataset {table_id!r}")

    def links(self, ids: list[str]) -> list[JoinStep]:
        keep = set(ids)
        return [JoinStep(left=r["from"], right=r["to"],
                         left_on=[a for a, _ in r["on"]], right_on=[b for _, b in r["on"]],
                         note=r.get("comment", ""))
                for r in self._export().get("relationships", [])
                if r["from"] in keep and r["to"] in keep]

    def defaults(self) -> list[str]:
        return list(self._export().get("recommended") or [])


def register(registry) -> None:
    registry.add_catalog(MetadataCatalog())
