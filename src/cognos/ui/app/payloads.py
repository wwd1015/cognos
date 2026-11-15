"""Gate forms to the engine's gate payloads."""

from __future__ import annotations

import json
from typing import Any

from ... import service
from .. import design_steps, transform_review


def source_payload(fields: dict, run_id: str | None = None) -> dict:
    """The data the "Provide the data" form describes: the catalog tables checked, the files
    dropped and the Snowflake lines. An incomplete form is refused in the words of the form."""
    try:
        files = list(json.loads(fields.get("add_files") or "[]"))
    except ValueError:
        files = []
    chosen = fields.get("catalog") or []
    if isinstance(chosen, str):  # the picker keeps its list as JSON in a hidden field
        try:
            chosen = list(json.loads(chosen or "[]"))
        except ValueError:
            chosen = []
    return service.data_payload(
        list(chosen), files, fields.get("add_sql") or "",
        datetime_col=(fields.get("datetime_col") or "").strip() or None,
        target=(fields.get("target") or "").strip() or None, run_id=run_id)


def gate_payload(gate: str, action: str, fields: dict, run_id: str) -> tuple[dict, str]:
    """Translate form fields into the engine's gate payload."""
    reason = (fields.get("reason") or "").strip()
    payload: dict = {}
    if action == "send_back":
        payload["message"] = (fields.get("message") or reason or "").strip()
        if fields.get("target"):
            payload["target"] = fields["target"]
        return payload, reason or payload["message"]
    if gate == "gate_source":
        return source_payload(fields, run_id), reason
    if gate == "gate_data" and action == "edit":
        payload["exclude_columns"] = list(fields.get("exclude") or [])
        kinds = transform_review.decisions(fields.get("transforms"))
        if "transforms" in fields:  # the picker was on the page: every decision it holds
            payload["transforms"] = kinds
        if fields.get("target"):
            payload["target"] = fields["target"]
        if fields.get("join_spec"):  # several sources: the join as the canvas now states it
            try:
                spec = json.loads(fields["join_spec"])
            except ValueError:
                spec = {}
            if spec.get("base"):
                payload["join"] = {"base": spec["base"], "links": list(spec.get("links") or []),
                                   **({"rules": spec["rules"]} if spec.get("rules") else {}),
                                   **({"drawn": spec["drawn"]} if spec.get("drawn") else {})}
    if gate in ("gate_intent", "gate_design"):
        answers = {k.split("::", 1)[1]: v for k, v in fields.items()
                   if k.startswith("answer::") and v and str(v).strip()}
        st = service.state(run_id)
        answers = {k: v for k, v in answers.items() if (g := st.gap(k)) and g.status == "open"}
        if answers:
            payload["answers"] = answers
    if gate == "gate_design":
        ideate = service.results(run_id).get("ideate").payload or {}
        hyps = ideate.get("hypotheses", [])
        families = fields.get("families")
        if (action == "edit" and families is not None
                and set(families) != set(design_steps.slate_families(ideate))):
            payload["slate"] = [next(({k: h[k] for k in ("family", "feature_strategy", "role",
                                                         "priority", "rationale")}
                                      for h in hyps if h["family"] == fam),
                                     {"family": fam, "feature_strategy": "all",
                                      "role": "candidate", "priority": 0.5,
                                      "rationale": "Added by the developer."})
                                for fam in families]
        roles = {k.split("::", 1)[1]: v for k, v in fields.items()
                 if k.startswith("framework::") and v}
        in_force = {k: f["decision"] for k, f in design_steps.frameworks_in_force(ideate).items()}
        if action == "edit" and any(in_force.get(k) != v for k, v in roles.items()):
            payload["framework_roles"] = roles
            payload["model_form_reason"] = fields.get("model_form_reason") or ""
        forms: dict[str, dict] = {}
        for k, v in fields.items():
            if k.startswith("proposed::"):
                _, i, part = k.split("::")
                forms.setdefault(i, {})[part] = " ".join(str(v or "").split())
        forms_asked = [{"name": f.get("name", ""), "reason": f.get("reason", "")}
                       for _, f in sorted(forms.items(), key=lambda kv: int(kv[0]))
                       if f.get("name")]
        forms_now = [{"name": f["name"], "reason": f["reason"]}
                     for f in service.state(run_id).overrides.proposed_forms]
        if action == "edit" and forms and forms_asked != forms_now:
            payload["proposed_forms"] = forms_asked
        params = transform_review.decisions(fields.get("transform_params"))
        if action == "edit" and params:
            payload["transform_params"] = params
        plan = (service.results(run_id).get("ideate").payload or {}).get("plan")
        if action == "edit" and plan:
            patch = plan_patch(plan, fields)
            if patch:
                payload["plan"] = patch
        if action == "edit" and "screen_exceptions" in fields:
            asked = design_steps.exceptions_of(fields.get("screen_exceptions"))
            in_force = {n: {"keep": e["keep"], "reason": e["reason"]}
                        for n, e in service.state(run_id).overrides.screen_exceptions.items()}
            if asked != in_force:
                payload["screen_exceptions"] = asked
    if gate == "gate_candidates" and action == "edit":
        extra = fields.get("extra") or []
        if isinstance(extra, str):  # ids typed by hand, comma separated
            extra = [x.strip() for x in extra.replace(";", ",").split(",")]
        payload["pool"] = list(dict.fromkeys([*(fields.get("pool") or []), *(x for x in extra if x)]))
        model = (service.results(run_id).get("model").payload or {})
        shown = {a["id"]: a.get("name") for a in model.get("leaders") or []}
        names = {k.split("::", 1)[1]: " ".join(str(v).split()) for k, v in fields.items()
                 if k.startswith("name::") and v and str(v).strip()}
        names = {k: v for k, v in names.items() if shown.get(k) != v}
        if names:
            payload["names"] = names
    if gate == "gate_champion" and action == "override":
        payload["champion"] = fields.get("champion")
    return payload, reason


def plan_patch(plan: dict, fields: dict) -> dict:
    """What the plan form changed: only the fields that differ from the plan on the page."""
    if any(k.startswith("must::") for k in fields):  # one cell per variable in the table
        fields = {**fields, "plan::selection.must_include": [
            v for k, v in fields.items() if k.startswith("must::") and v]}
    def blank(v: Any) -> bool:
        return v is None or v == ""

    patch: dict = {}
    for key, value in fields.items():
        if not key.startswith("plan::") or key.startswith("plan::rank."):
            continue
        section_name, _, name = key[len("plan::"):].partition(".")
        if section_name not in plan or name not in plan[section_name]:
            continue
        current = plan[section_name][name]
        value = None if blank(value) else value
        if isinstance(current, bool):
            value = value == "yes" if value in ("yes", "no") else bool(value)
        if isinstance(current, list) and value is None:
            continue  # an untouched multi-select
        if isinstance(current, int) and not isinstance(current, bool) and isinstance(value, float):
            value = int(value)
        if value != current and not (blank(value) and blank(current)):
            patch.setdefault(section_name, {})[name] = value
    rank = []
    for i in range(3):  # the ranking lines on the page, each with its weight for a blend
        metric = fields.get(f"plan::rank.{i}.metric")
        if not blank(metric):
            line = {"metric": metric,
                    "sample": fields.get(f"plan::rank.{i}.sample") or "validation"}
            weight = fields.get(f"plan::rank.{i}.weight")
            if not blank(weight):
                line["weight"] = float(weight)
            rank.append(line)
    current = [{k: (float(v) if k == "weight" else v) for k, v in r.items()}
               for r in plan["evaluation"].get("rank_by") or []]
    blended = (patch.get("evaluation") or {}).get(
        "rank_mode", plan["evaluation"].get("rank_mode")) == "weights"

    def same(a: list[dict], b: list[dict]) -> bool:  # weights count only in a blend
        def strip(r: dict) -> dict:
            return r if blended else {k: v for k, v in r.items() if k != "weight"}
        return [strip(r) for r in a] == [strip(r) for r in b]

    if rank and not same(rank, current):
        patch.setdefault("evaluation", {})["rank_by"] = rank
    return patch
