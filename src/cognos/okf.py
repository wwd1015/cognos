"""Open Knowledge Format (OKF v0.1) reader/writer.

OKF is Google Cloud's vendor-neutral markdown spec (June 2026) for giving AI agents curated
context. COGNOS emits each model-development run as an OKF *bundle*: a directory of markdown
*concepts* (one per artifact — dataset, hypothesis, model, backtest, finding) each carrying YAML
frontmatter, plus a reserved ``index.md`` (progressive-disclosure listing, declares ``okf_version``)
and ``log.md`` (newest-first change history). The only required frontmatter field is ``type``.

Concepts cross-link with ordinary markdown links — bundle-relative (begin with ``/``) or relative
(``./``). Consumers treat links as untyped directed edges, so the consistency-review stage can walk
the graph to verify every documented claim resolves to a real artifact or code symbol.

Spec: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

OKF_VERSION = "0.1"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# COGNOS docs<->code traceability anchor, e.g. {@code:src/cognos/runtime/score.py#score_row}
_CODE_ANCHOR_RE = re.compile(r"\{@code:([^}#]+)(?:#([^}]+))?\}")


@dataclass
class OKFConcept:
    """One ``.md`` concept = one node in the knowledge graph."""

    name: str  # filename stem (no .md)
    type: str  # REQUIRED frontmatter field
    title: str = ""
    description: str = ""
    resource: str = ""  # URI of the underlying asset (code/data/model path)
    tags: list[str] = field(default_factory=list)
    timestamp: str = ""
    body: str = ""
    extra: dict[str, Any] = field(default_factory=dict)  # arbitrary preserved keys

    def frontmatter(self) -> dict[str, Any]:
        fm: dict[str, Any] = {"type": self.type}
        if self.title:
            fm["title"] = self.title
        if self.description:
            fm["description"] = self.description
        if self.resource:
            fm["resource"] = self.resource
        if self.tags:
            fm["tags"] = self.tags
        fm["timestamp"] = self.timestamp or datetime.now(UTC).isoformat()
        fm.update(self.extra)
        return fm

    def render(self) -> str:
        fm = yaml.safe_dump(self.frontmatter(), sort_keys=False).strip()
        return f"---\n{fm}\n---\n\n{self.body.strip()}\n"

    def links(self) -> list[str]:
        """All markdown link targets in the body (untyped directed edges)."""
        return [m.group(2) for m in _LINK_RE.finditer(self.body)]

    def code_anchors(self) -> list[tuple[str, str | None]]:
        """All ``{@code:path#symbol}`` traceability anchors (path, symbol)."""
        return [(m.group(1), m.group(2)) for m in _CODE_ANCHOR_RE.finditer(self.body)]


class OKFBundle:
    """A directory of OKF concepts with reserved ``index.md`` and ``log.md``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._concepts: dict[str, OKFConcept] = {}
        self._log: list[str] = []

    # --- authoring ---------------------------------------------------------------
    def add(self, concept: OKFConcept) -> OKFConcept:
        self._concepts[concept.name] = concept
        (self.root / f"{concept.name}.md").write_text(concept.render())
        return concept

    def log_event(self, kind: str, message: str, when: str | None = None) -> None:
        ts = when or datetime.now(UTC).date().isoformat()
        self._log.append(f"## {ts}\n\n**{kind}:** {message}")

    def write_index(self, title: str, description: str = "") -> None:
        lines = [f"# {title}", ""]
        if description:
            lines += [description, ""]
        lines += [f"`okf_version: {OKF_VERSION}`", "", "## Concepts", ""]
        for name, c in sorted(self._concepts.items()):
            label = c.title or name
            desc = f" — {c.description}" if c.description else ""
            lines.append(f"- [{label}](./{name}.md){desc}")
        (self.root / "index.md").write_text("\n".join(lines) + "\n")

    def write_log(self) -> None:
        body = "# Change Log\n\n" + "\n\n".join(reversed(self._log)) + "\n"
        (self.root / "log.md").write_text(body)

    def finalize(self, title: str, description: str = "") -> None:
        self.write_index(title, description)
        self.write_log()

    # --- reading -----------------------------------------------------------------
    def concepts(self) -> list[OKFConcept]:
        if not self._concepts:
            self.load()
        return list(self._concepts.values())

    def load(self) -> OKFBundle:
        self._concepts.clear()
        for md in sorted(self.root.glob("*.md")):
            if md.name in ("index.md", "log.md"):
                continue
            self._concepts[md.stem] = parse_concept(md.read_text(), name=md.stem)
        return self


def parse_concept(text: str, name: str = "") -> OKFConcept:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        # Permissive: a body with no frontmatter still parses (type defaults to "note").
        return OKFConcept(name=name, type="note", body=text.strip())
    fm = yaml.safe_load(m.group(1)) or {}
    body = m.group(2)
    known = {"type", "title", "description", "resource", "tags", "timestamp"}
    extra = {k: v for k, v in fm.items() if k not in known}
    return OKFConcept(
        name=name,
        type=str(fm.get("type", "note")),
        title=str(fm.get("title", "")),
        description=str(fm.get("description", "")),
        resource=str(fm.get("resource", "")),
        tags=list(fm.get("tags", []) or []),
        timestamp=str(fm.get("timestamp", "")),
        body=body.strip(),
        extra=extra,
    )


def is_conformant(text: str) -> bool:
    """A concept is OKF-conformant iff its frontmatter parses and ``type`` is non-empty."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return False
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return False
    return bool(str(fm.get("type", "")).strip())
