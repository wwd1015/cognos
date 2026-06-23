"""Open Knowledge Format reader/writer + traceability anchors."""

from __future__ import annotations

from cognos.okf import OKF_VERSION, OKFBundle, OKFConcept, is_conformant, parse_concept


def test_concept_render_roundtrip():
    c = OKFConcept(name="model", type="model", title="Champion", description="d",
                   resource="models/x.joblib", tags=["a", "b"],
                   body="# Schema\nbody text\n[see model](./other.md)")
    text = c.render()
    assert text.startswith("---")
    assert "type: model" in text
    back = parse_concept(text, name="model")
    assert back.type == "model" and back.title == "Champion"
    assert back.resource == "models/x.joblib" and back.tags == ["a", "b"]


def test_code_anchor_and_link_extraction():
    c = OKFConcept(name="m", type="methodology",
                   body="See {@code:src/cognos/runtime/score.py#score_row} and "
                        "{@code:src/cognos/stages/model.py}. Link [d](./dataset.md).")
    anchors = c.code_anchors()
    assert ("src/cognos/runtime/score.py", "score_row") in anchors
    assert ("src/cognos/stages/model.py", None) in anchors
    assert "./dataset.md" in c.links()


def test_bundle_finalize_writes_index_and_log(tmp_path):
    b = OKFBundle(tmp_path / "docs")
    b.add(OKFConcept(name="overview", type="model_overview", title="Overview", body="hi"))
    b.add(OKFConcept(name="model", type="model", title="Model", body="m"))
    b.log_event("Creation", "bundle created")
    b.finalize("COGNOS White Paper", "desc")
    index = (tmp_path / "docs" / "index.md").read_text()
    assert f"okf_version: {OKF_VERSION}" in index
    assert "[Overview](./overview.md)" in index
    assert (tmp_path / "docs" / "log.md").exists()
    # reload
    loaded = OKFBundle(tmp_path / "docs").load()
    names = {c.name for c in loaded.concepts()}
    assert names == {"overview", "model"}  # index/log excluded


def test_is_conformant():
    assert is_conformant("---\ntype: model\n---\nbody")
    assert not is_conformant("---\ntitle: no type\n---\nbody")
    assert not is_conformant("no frontmatter")
