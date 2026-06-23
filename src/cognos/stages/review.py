"""Stage 8 — Docs<->code consistency / drift review (gate).

The OKF bundle the document stage emits is only trustworthy if every claim it makes still resolves
to a real artifact. This gate walks the bundle's knowledge graph and verifies the load-bearing
edges: ``{@code:path#symbol}`` traceability anchors must point at code that actually exists (AST-
checked), internal markdown links must reach real concepts, and declared resources must be present
on disk. Stale code references are the highest-confidence drift signal — documentation that promises
code which no longer exists is actively misleading — so they BLOCK; softer drift (broken internal
links, missing resources) only WARNs, mirroring OKF's own tolerance for dangling links.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ..artifacts import ArtifactRef, Finding, Severity, StageResult, Verdict
from ..context import RunContext
from ..okf import OKFBundle
from .base import Stage, register_stage


def _symbols(py_source: str) -> set[str]:
    """All addressable top-level symbols in a module: functions, classes, ``Class.method``,
    bare method names, and module-level assignment targets — the set an anchor symbol may name."""
    out: set[str] = set()
    try:
        tree = ast.parse(py_source)
    except SyntaxError:
        return out
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
        elif isinstance(node, ast.ClassDef):
            out.add(node.name)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.add(f"{node.name}.{sub.name}")
                    out.add(sub.name)  # bare method name (anchors often omit the class)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
    return out


def _link_concept(target: str) -> str | None:
    """Resolve a bundle-relative/relative ``.md`` link to a concept name, or None if external."""
    t = target.strip()
    if "://" in t or t.startswith(("http:", "https:", "mailto:")):
        return None
    t = t.split("#", 1)[0].split("?", 1)[0]  # drop anchor/query fragments
    if not t.endswith(".md"):
        return None
    return Path(t).stem or None


@register_stage
class ReviewStage(Stage):
    name = "review"
    requires = ("document",)
    is_gate = True
    description = "Verify the OKF docs bundle stays consistent with the code and artifacts it claims."

    def run(self, ctx: RunContext) -> StageResult:
        repo_root = Path(__import__("cognos").__file__).resolve().parents[2]
        bundle = OKFBundle(ctx.docs_dir).load()
        concepts = bundle.concepts()
        concept_names = {c.name for c in concepts}
        res = StageResult(stage=self.name, verdict=Verdict.PASS)

        missing_code_paths: list[str] = []
        missing_symbols: list[str] = []
        broken_links: list[str] = []
        n_code_anchors = 0
        n_links = 0
        total_checks = 0
        conformant = True

        for c in concepts:
            # (1) OKF conformance: parser guarantees a non-empty type, but verify the invariant.
            total_checks += 1
            if not str(c.type).strip():
                conformant = False
                res.add_finding(Finding(
                    id=f"nonconformant-{c.name}", severity=Severity.MEDIUM, category="okf",
                    message=f"Concept '{c.name}' has no OKF type.", location=c.name, confidence=1.0,
                    reviewed=c.extra.get("reviewed", False) if isinstance(c.extra, dict) else False,
                ))

            # (2) Code anchors: AST-verify each {@code:path#symbol} points at real code.
            for path, symbol in c.code_anchors():
                n_code_anchors += 1
                total_checks += 1
                fpath = repo_root / path
                anchor = f"{path}#{symbol}" if symbol else path
                if not fpath.exists():
                    missing_code_paths.append(path)
                    res.add_finding(Finding(
                        id=f"stale-path-{c.name}-{len(missing_code_paths)}", severity=Severity.HIGH,
                        category="drift", confidence=1.0, location=f"{c.name} -> {anchor}",
                        message=f"Concept '{c.name}' references code path '{path}' that does not exist.",
                        suggestion="Update the {@code:...} anchor or restore the file.",
                    ))
                    continue
                if symbol:
                    syms = _symbols(fpath.read_text())
                    if symbol not in syms:
                        missing_symbols.append(anchor)
                        res.add_finding(Finding(
                            id=f"stale-symbol-{c.name}-{len(missing_symbols)}", severity=Severity.HIGH,
                            category="drift", confidence=1.0, location=f"{c.name} -> {anchor}",
                            message=f"Concept '{c.name}' references symbol '{symbol}' absent from '{path}'.",
                            suggestion="Rename the anchor symbol to a current function/class/method.",
                        ))

            # (3) Markdown links: internal .md targets must reach a real concept (LOW — OKF tolerates dangling).
            for target in c.links():
                ref = _link_concept(target)
                if ref is None:
                    continue  # external / non-concept link
                n_links += 1
                total_checks += 1
                if ref not in concept_names:
                    broken_links.append(target)
                    res.add_finding(Finding(
                        id=f"broken-link-{c.name}-{len(broken_links)}", severity=Severity.LOW,
                        category="drift", confidence=0.8, location=f"{c.name} -> {target}",
                        message=f"Concept '{c.name}' links to missing internal concept '{ref}'.",
                        suggestion="Fix the link target or add the referenced concept.",
                    ))

            # (4) Resource existence: a local-looking resource path must resolve under the run dir or repo.
            resource = str(c.resource).strip()
            if resource and "://" not in resource and not resource.startswith(("http:", "https:")):
                total_checks += 1
                candidates = [ctx.resolve(resource), repo_root / resource, Path(resource)]
                if not any(p.exists() for p in candidates):
                    res.add_finding(Finding(
                        id=f"missing-resource-{c.name}", severity=Severity.LOW, category="drift",
                        confidence=0.8, location=f"{c.name} -> {resource}",
                        message=f"Concept '{c.name}' declares resource '{resource}' not found on disk.",
                        suggestion="Point the resource at an existing artifact or drop it.",
                    ))

        n_high = sum(1 for f in res.findings if f.severity is Severity.HIGH)
        drift_score = n_high / max(1, total_checks)
        has_high_drift = bool(missing_code_paths or missing_symbols)

        payload = {
            "n_concepts": len(concepts),
            "n_code_anchors": n_code_anchors,
            "n_links": n_links,
            "conformant": conformant,
            "missing_code_paths": missing_code_paths,
            "missing_symbols": missing_symbols,
            "broken_links": broken_links,
            "drift_score": drift_score,
            "repo_root": str(repo_root),
        }
        ref: ArtifactRef = ctx.save_json("stages/review/consistency.json", payload)
        res.add_artifact(ref)
        res.payload = payload
        res.metrics = {"n_concepts": len(concepts), "drift_score": drift_score, "n_high": n_high}

        if has_high_drift:
            res.verdict = Verdict.BLOCK
        elif res.findings:
            res.verdict = Verdict.WARN
        else:
            res.verdict = Verdict.PASS
        res.summary = (
            f"Reviewed {len(concepts)} concept(s): {n_code_anchors} code anchor(s), {n_links} internal "
            f"link(s); drift_score={drift_score:.3f}, {n_high} high-severity drift finding(s) "
            f"[{res.verdict.value}]."
        )
        return res
