"""Engagements: what kind of development a run is, and the documents it starts from.

Two development modes (``engagement.kind``; distinct from the autonomous / interactive run mode):

- ``new``: a complete new model development. It starts from the sponsor's **business intent
  document** and any background material.
- ``update``: a change to an existing model. It starts from the existing model's artifacts (white
  paper, development or deployment code, validation and monitoring reports, or an earlier COGNOS
  run) and a change request written on the same template.

This module is the mechanical half of intake: the intent template (render and parse), reading
uploaded documents as text, copying them into the run (``runs/<id>/inputs/``, so a fresh process
finds them), and a keyword scan of the prior model's artifacts. Judgment (is the intent clear,
what must the sponsor be asked) belongs to the Intake Analyst agent; see ``stages/intake.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, NamedTuple

KINDS = ("new", "update")
KIND_LABEL = {"new": "New model development", "update": "Model update"}

# The four sponsor decisions that shape the build (engine/process.py::CORE_DESIGN).
CORE = ("use_case", "horizon", "default_definition", "segment")
# Brief fields an accepted intent document writes into the effective config.
DESIGN_FIELDS = (*CORE, "interpretability")
COMPLIANCE_FIELDS = ("intended_use", "out_of_scope_use")
INTERPRETABILITY = ("required", "preferred", "flexible")


class BriefField(NamedTuple):
    id: str
    label: str  # the template heading
    hint: str  # the prompt under the heading
    required: bool  # an unanswered required field is a blocking interview question
    update_only: bool = False
    aliases: tuple[str, ...] = ()


FIELDS: tuple[BriefField, ...] = (
    BriefField("objective", "Business objective",
               "What business problem does the model solve, and what happens today without it?",
               True, aliases=("objective", "purpose", "business purpose", "business problem",
                              "business goal", "goal", "goals", "background and objective")),
    BriefField("use_case", "Decision the model supports",
               "e.g. origination underwriting, portfolio surveillance, CECL / IFRS 9, IRB, "
               "stress testing.", True,
               aliases=("use case", "model use", "decision supported", "decision")),
    BriefField("segment", "Portfolio segment",
               "Which obligors, products or exposures the model covers.", True,
               aliases=("segment", "portfolio", "population", "target population",
                        "model scope", "scope")),
    BriefField("default_definition", "Outcome definition",
               "The event the model predicts, e.g. 90+ days past due or nonaccrual.", True,
               aliases=("default definition", "target definition", "definition of default",
                        "event definition", "outcome", "target")),
    BriefField("horizon", "Outcome horizon",
               "The window the outcome is observed over, e.g. 12 months, lifetime.", True,
               aliases=("horizon", "outcome window", "prediction horizon", "time horizon")),
    BriefField("intended_use", "Intended use and users",
               "Who uses the output, for which decisions, with what human oversight.", False,
               aliases=("intended use", "intended users", "users")),
    BriefField("out_of_scope_use", "Out-of-scope uses",
               "Uses the model must not be relied on for.", False,
               aliases=("out of scope uses", "out of scope use", "out of scope",
                        "prohibited uses", "exclusions")),
    BriefField("interpretability", "Interpretability requirement",
               "required, preferred or flexible, and why.", False,
               aliases=("interpretability", "explainability", "explainability requirement")),
    BriefField("success_criteria", "Success criteria",
               "How the sponsor will judge the result: performance, stability, adoption.", False,
               aliases=("success measures", "acceptance criteria", "definition of success")),
    BriefField("constraints", "Constraints",
               "Regulatory, policy, operational or timing limits the build must respect.", False,
               aliases=("constraint", "regulatory constraints", "limitations and constraints")),
    BriefField("data_sources", "Data sources",
               "Where the development data comes from, its period, and known gaps.", False,
               aliases=("data", "data source", "development data")),
    BriefField("stakeholders", "Stakeholders",
               "Sponsor, model owner, users, validation, approver.", False,
               aliases=("stakeholder", "owners", "roles")),
    BriefField("prior_model", "Existing model",
               "Name, version, in use since, and where it is used.", True, True,
               aliases=("current model", "prior model", "incumbent model", "model being updated")),
    BriefField("update_reason", "Reason for the update",
               "e.g. performance deterioration, a data change, new regulation, a validation "
               "finding, a scope expansion.", True, True,
               aliases=("reason for update", "update reason", "trigger", "update trigger",
                        "why now")),
    BriefField("requested_changes", "Requested changes",
               "One change per bullet.", True, True,
               aliases=("requested change", "changes requested", "change request", "changes",
                        "scope of the update", "scope of update")),
    BriefField("must_not_change", "What must not change",
               "Segments, inputs, definitions or interfaces the update has to keep.", False, True,
               aliases=("must not change", "unchanged", "out of scope for the update",
                        "what stays the same")),
    BriefField("known_issues", "Known issues and findings",
               "Validation or audit findings and monitoring breaches the update should close.",
               False, True,
               aliases=("known issues", "open findings", "findings", "issues",
                        "known issues and open findings")),
)
FIELD = {f.id: f for f in FIELDS}

# What the interview asks when a required field is unanswered (the deterministic agent's wording).
QUESTIONS = {
    "objective": "What business problem should this model solve, and what would be done "
                 "differently once it exists?",
    "use_case": "What decision will the model support (origination underwriting, portfolio "
                "surveillance, CECL/IFRS 9, IRB, stress testing)? The use case fixes the target "
                "definition, horizon, and documentation depth.",
    "segment": "What portfolio segment does the model cover (C&I, CRE, small business…)? Pooling "
               "heterogeneous segments biases coefficients.",
    "default_definition": "What event should the model predict (e.g. 90+ DPD, nonaccrual, "
                          "bankruptcy)? Validation re-derives risk from this definition.",
    "horizon": "Over what outcome window is the event observed (e.g. 12 months)? The window must "
               "match how the target was labelled.",
    "success_criteria": "How will the sponsor judge the result (a performance threshold, "
                        "stability, a benchmark to beat)?",
    "prior_model": "Which existing model is being updated (name, version, where it is used)?",
    "update_reason": "What triggered this update (performance deterioration, a data change, new "
                     "regulation, a validation finding, a scope change)?",
    "requested_changes": "What exactly should change in the model? List each requested change.",
}

_BLANK = {"", "tbd", "tba", "tbc", "n/a", "na", "none", "unknown", "?", "to be decided",
          "to be determined", "to be confirmed", "undecided", "not decided", "-", "--", "…", "..."}


def fields_for(kind: str) -> list[BriefField]:
    return [f for f in FIELDS if kind == "update" or not f.update_only]


# --- the template -------------------------------------------------------------------------------
def render_template(kind: str = "new", values: dict[str, str] | None = None,
                    name: str = "") -> str:
    """The business intent document as Markdown: blank (hints under each heading) or filled with
    ``values``. The update template is the same document plus the change-request sections."""
    if kind not in KINDS:
        raise ValueError(f"unknown development mode {kind!r}; choose one of {list(KINDS)}")
    values = values or {}
    title = "Model update request" if kind == "update" else "Business intent"
    lines = [
        f"# {title}: {name or '<model name>'}",
        "",
        f"<!-- COGNOS intent template v1 (kind: {kind}). Write under each heading and keep the "
        "headings as they are. Leave a section empty when it is undecided: the Intake Analyst "
        "asks about it instead of assuming. -->",
        "",
    ]
    for f in fields_for(kind):
        value = str(values.get(f.id) or "").strip()
        lines += [f"## {f.label}", "", value or f"_{f.hint}_", ""]
    return "\n".join(lines)


def _norm_heading(text: str) -> str:
    text = re.sub(r"^\s*\d+(\.\d+)*[.)]?\s*", "", text.lower())
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


_HEADINGS: dict[str, str] = {}
for _f in FIELDS:
    for _name in (_f.label, *_f.aliases):
        _HEADINGS.setdefault(_norm_heading(_name), _f.id)

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_BOLD_HEADING_RE = re.compile(r"^\s*\*\*(.+?)\*\*\s*:?\s*$")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_HINT_RE = re.compile(r"^\s*_[^_].*_\s*$")


def parse_intent(text: str) -> dict[str, str]:
    """Template sections by field id: the text under each recognised heading. Hints the author
    left in place and placeholders such as ``TBD`` count as empty. Mechanical: an unrecognised
    heading is skipped here (a free-form document is the Intake Analyst's to read)."""
    out: dict[str, list[str]] = {}
    current: str | None = None
    for line in _COMMENT_RE.sub("", text or "").splitlines():
        m = _HEADING_RE.match(line) or _BOLD_HEADING_RE.match(line)
        if m:
            current = _HEADINGS.get(_norm_heading(m.group(1)))
            continue
        if current is None or _HINT_RE.match(line):
            continue
        out.setdefault(current, []).append(line.rstrip())
    parsed = {}
    for field, lines in out.items():
        value = "\n".join(lines).strip()
        if value.lower().strip(" .") not in _BLANK:
            parsed[field] = value
    return parsed


def bullets(text: str) -> list[str]:
    """The items of a bulleted or numbered section (one paragraph is one item)."""
    items: list[str] = []
    for line in (text or "").splitlines():
        m = re.match(r"^\s*(?:[-*+•]|\d+[.)])\s+(.*)$", line)
        if m:
            items.append(m.group(1).strip())
        elif line.strip() and items and line.startswith((" ", "\t")):
            items[-1] += " " + line.strip()
        elif line.strip():
            items.append(line.strip())
    return [i for i in items if i]


def interpretability_of(text: str) -> str | None:
    """The design setting an interpretability statement names, if it names exactly one first."""
    hits = [(m.start(), word) for word in INTERPRETABILITY
            if (m := re.search(rf"\b{word}\b", (text or "").lower()))]
    return min(hits)[1] if hits else None


def squash(text: str) -> str:
    """Whitespace- and case-insensitive form used to check that a quote is in a document."""
    return " ".join(re.sub(r"[*_`>#|]+", " ", (text or "").lower()).split())


# --- reading documents ----------------------------------------------------------------------------
CODE_SUFFIXES = {".py", ".r", ".sql", ".sas", ".scala", ".java", ".jl", ".m", ".ipynb", ".do"}
_TEXT_SUFFIXES = CODE_SUFFIXES | {".md", ".markdown", ".txt", ".rst", ".yaml", ".yml", ".json",
                                  ".csv", ".tsv", ".toml", ".cfg", ".ini", ".html", ".htm", ".tex"}
MAX_BYTES = 4_000_000


def _docx_text(path: Path) -> str:
    """Paragraph text of a Word document, headings as Markdown headings (stdlib only)."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    out = []
    for para in re.split(r"</w:p>", xml):
        text = "".join(re.findall(r"<w:t(?:\s[^>]*)?>([^<]*)</w:t>", para))
        text = (text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                .replace("&quot;", '"').replace("&apos;", "'")).strip()
        if not text:
            continue
        level = re.search(r'<w:pStyle w:val="(?:Heading|heading)\s?(\d)"', para)
        if level:
            text = "#" * max(1, min(6, int(level.group(1)))) + " " + text
        elif re.search(r'<w:pStyle w:val="Title"', para):
            text = "# " + text
        elif re.search(r"<w:numPr>", para):
            text = "- " + text
        out.append(text)
    return "\n\n".join(out)


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader  # optional: unavailable makes the document unreadable, not a crash

    return "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)


def _notebook_text(raw: str) -> str:
    cells = json.loads(raw).get("cells", [])
    return "\n\n".join("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                       else str(c.get("source", "")) for c in cells)


def read_text(path: str | Path) -> tuple[str | None, str]:
    """A document as text, or ``(None, why)`` when it cannot be read. Never raises."""
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if path.stat().st_size > MAX_BYTES:
            return None, f"larger than {MAX_BYTES // 1_000_000} MB; supply an extract"
        if suffix == ".docx":
            return _docx_text(path), ""
        if suffix == ".pdf":
            try:
                text = _pdf_text(path)
            except ImportError:
                return None, "reading a PDF needs the optional pypdf package; supply .md, .txt or .docx"
            return (text, "") if text.strip() else (None, "no extractable text (a scanned PDF?)")
        data = path.read_bytes()
        if suffix not in _TEXT_SUFFIXES and b"\x00" in data[:4096]:
            return None, "a binary file; supply a text format (.md, .txt, .docx, source code)"
        text = data.decode("utf-8", errors="replace")
        return (_notebook_text(text) if suffix == ".ipynb" else text), ""
    except Exception as exc:  # an unreadable upload is a finding, never a crash
        return None, f"{type(exc).__name__}: {exc}"


def infer_role(name: str) -> str:
    """What a prior-model artifact is, from its name. The author may state the role instead."""
    stem, suffix = Path(name).stem.lower(), Path(name).suffix.lower()
    if suffix in CODE_SUFFIXES:
        return "code"
    if "valid" in stem or "audit" in stem or "finding" in stem:
        return "validation"
    if "monitor" in stem or "performance" in stem or "tracking" in stem:
        return "monitoring"
    if suffix in (".md", ".markdown", ".docx", ".pdf", ".txt", ".rst", ".tex", ".html", ".htm"):
        return "whitepaper"
    return "other"


# --- copying the engagement into the run -----------------------------------------------------------
def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._") or "document"


def _source(path: str, run_dir: Path) -> Path | None:
    for candidate in (run_dir / path, Path(path)):
        if candidate.is_file():
            return candidate
    return None


def resolve_run(ref: str, runs_root: Path) -> Path | None:
    """A prior COGNOS run named by id (under the runs root) or by directory."""
    for candidate in (Path(runs_root) / ref, Path(ref)):
        if (candidate / "state.json").is_file():
            return candidate
    return None


def summarize_run(run_dir: Path) -> dict[str, Any]:
    """What an earlier COGNOS run recorded about its model. A mechanical read of its results."""
    def load(rel: str) -> dict:
        try:
            return json.loads((run_dir / rel).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {}

    state = load("state.json")
    model = load("stages/model/result.json")
    payload = model.get("payload") or {}
    champion = payload.get("champion") or {}
    package = state.get("package") or {}
    return {
        "run_id": run_dir.name,
        "project": state.get("project", ""),
        "status": state.get("status", ""),
        "champion_family": champion.get("family"),
        "champion_label": payload.get("champion_label"),
        "features": list(champion.get("features") or []),
        "metric": payload.get("metric"),
        "cv_mean": payload.get("cv_mean"),
        "cv_std": payload.get("cv_std"),
        "holdout_metric": payload.get("holdout_metric"),
        "n_train": payload.get("n_train"),
        "validation_verdict": load("stages/validate/result.json").get("verdict"),
        "package_version": package.get("version"),
        "package_status": package.get("status"),
    }


def ingest(config: Any, run_dir: Path, runs_root: Path | None = None) -> Any:
    """Copy the engagement's documents into ``run_dir/inputs/`` and return a config whose
    engagement paths are run-relative. A missing document refuses the run before it is created.
    The profile the user wrote is not edited; the copy in the run is the run's own record."""
    eng = config.engagement
    plan: list[tuple[Path, str]] = []  # (source, run-relative destination)
    taken: set[str] = set()

    def place(path: str, folder: str, what: str) -> str:
        if path.startswith("inputs/") and (run_dir / path).is_file():
            return path  # already in the run (a reloaded run)
        src = _source(path, run_dir)
        if src is None:
            raise FileNotFoundError(f"{what} not found: {path}")
        name, n = _safe_name(src.name), 1
        while f"{folder}/{name}" in taken:
            n += 1
            name = f"{Path(_safe_name(src.name)).stem}_{n}{Path(src.name).suffix}"
        taken.add(f"{folder}/{name}")
        plan.append((src, f"inputs/{folder}/{name}"))
        return f"inputs/{folder}/{name}"

    intent = place(eng.intent, "intent", "business intent document") if eng.intent else None
    supporting = [place(p, "supporting", "supporting document") for p in eng.supporting]
    prior = [a.model_copy(update={"path": place(a.path, "prior", "prior-model artifact")})
             for a in eng.prior_artifacts]
    prior_summary = None
    if eng.prior_run and not (run_dir / "inputs" / "prior_run.json").is_file():
        prior_dir = resolve_run(eng.prior_run, runs_root or run_dir.parent)
        if prior_dir is None:
            raise FileNotFoundError(f"prior run not found: {eng.prior_run}")
        prior_summary = summarize_run(prior_dir)
        paper = prior_dir / "stages" / "document" / "whitepaper.md"
        if paper.is_file() and not any(
                (a.role if a.role != "auto" else infer_role(a.path)) == "whitepaper" for a in prior):
            from .config import PriorArtifact

            taken.add("prior/prior_run_whitepaper.md")
            plan.append((paper, "inputs/prior/prior_run_whitepaper.md"))
            prior.append(PriorArtifact(path="inputs/prior/prior_run_whitepaper.md",
                                       role="whitepaper"))
    if not plan and prior_summary is None:
        return config
    for src, rel in plan:
        dest = run_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    if prior_summary is not None:
        dest = run_dir / "inputs" / "prior_run.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(prior_summary, indent=1, default=str), encoding="utf-8")
    return config.model_copy(update={"engagement": eng.model_copy(update={
        "intent": intent, "supporting": supporting, "prior_artifacts": prior})})


def load_documents(config: Any, run_dir: Path) -> list[dict[str, Any]]:
    """Every document of the engagement with its text (``text`` is None when unreadable)."""
    eng = config.engagement
    entries = ([(eng.intent, "intent")] if eng.intent else []) \
        + [(p, "supporting") for p in eng.supporting] \
        + [(a.path, a.role if a.role != "auto" else infer_role(a.path)) for a in eng.prior_artifacts]
    docs = []
    for path, role in entries:
        src = _source(path, Path(run_dir))
        doc: dict[str, Any] = {"name": Path(path).name, "role": role, "path": path,
                               "prior": role not in ("intent", "supporting"),
                               "readable": False, "note": "", "chars": 0, "sha256": "",
                               "text": None}
        if src is None:
            doc["note"] = "file not found"
        else:
            doc["sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()
            text, note = read_text(src)
            doc.update(text=text, note=note, readable=text is not None,
                       chars=len(text or ""))
        docs.append(doc)
    return docs


def prior_run_summary(run_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((Path(run_dir) / "inputs" / "prior_run.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


# --- the prior model, read mechanically ------------------------------------------------------------
# Phrases that name a model family the engine can fit. Counted, never interpreted: a mention is a
# hint for the Intake Analyst and the Design Lead, not a statement of what the prior model is.
_FAMILY_PHRASES: dict[str, tuple[str, ...]] = {
    "logit": (r"logistic regression", r"\blogit\b", r"\bscorecard\b", r"LogisticRegression"),
    "probit": (r"\bprobit\b",),
    "cloglog": (r"\bcloglog\b", r"complementary log-log"),
    "hazard_logit": (r"discrete-time hazard", r"\bhazard model\b", r"survival model"),
    "lasso_logit": (r"\blasso\b", r"\bl1[- ]penal", r"penalty=['\"]l1"),
    "ridge_logit": (r"\bridge\b", r"\bl2[- ]penal"),
    "ols": (r"\bols\b", r"ordinary least squares", r"linear regression", r"LinearRegression"),
    "elasticnet": (r"elastic[- ]?net",),
    "random_forest": (r"random forest", r"RandomForest"),
    "gradient_boosting": (r"gradient boost", r"\bxgboost\b", r"\blightgbm\b", r"\bgbm\b",
                          r"GradientBoosting", r"XGB", r"LGBM"),
}


def family_mentions(texts: list[str]) -> list[dict[str, Any]]:
    """How often each engine-fittable family is named in the prior model's artifacts."""
    corpus = "\n".join(t for t in texts if t)
    counts = []
    for family, patterns in _FAMILY_PHRASES.items():
        n = sum(len(re.findall(p, corpus, flags=re.I)) for p in patterns)
        if n:
            counts.append({"family": family, "mentions": n})
    return sorted(counts, key=lambda c: (-c["mentions"], c["family"]))


def columns_mentioned(columns: list[str], texts: list[str]) -> list[str]:
    """Dataset columns named (as whole identifiers) in the prior model's artifacts."""
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", "\n".join(t for t in texts if t)))
    lowered = {t.lower() for t in tokens}
    return [c for c in columns if c in tokens or c.lower() in lowered]
