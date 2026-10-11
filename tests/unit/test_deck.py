"""The presentation deck (docs/deck/index.html) is documentation held to the code: its version,
its workflow and its figures must describe the system as it is. A failure here means the system
changed and the deck did not: update the deck (and its edition log) in the same commit."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cognos
from cognos import plugins
from cognos.agents.contracts import FRIENDLY
from cognos.engine import graph
from cognos.engine.process import SEAT_LABEL, SEAT_OF_GATE, seat_label

DECK = Path(__file__).resolve().parents[2] / "docs" / "deck" / "index.html"


def _deck() -> str:
    return DECK.read_text(encoding="utf-8")


def test_the_deck_describes_this_version_and_logs_its_edition():
    html = _deck()
    edition = json.loads(re.search(r'<script type="application/json" id="edition">(.*?)</script>',
                                   html, re.S).group(1))
    assert edition["system_version"] == cognos.__version__
    log = re.search(r'<table class="log.*?</table>', html, re.S).group(0)
    rows = re.findall(r'<tr><td class="mono">(\d+)</td><td class="mono">v([\d.]+)</td>'
                      r'<td class="mono">([\d-]+)</td>', log)
    # the newest edition is the one the deck claims to be, and every edition has a row
    assert [int(r[0]) for r in rows] == list(range(1, edition["edition"] + 1))
    assert rows[-1][1:] == (edition["system_version"], edition["date"])


def test_the_decks_workflow_is_the_engines_workflow():
    block = re.search(r"/\* FLOW:BEGIN \*/(.*?)/\* FLOW:END \*/", _deck(), re.S).group(1)
    steps = re.findall(r'\{ id: "(\w+)", label: "([^"]+)"', block)
    assert [s[0] for s in steps] == list(graph.STEPS)
    assert dict(steps) == dict(graph.LABELS)
    seats = dict(re.findall(r'\{ id: "(gate_\w+)", label: "[^"]+", seat: "([^"]+)"', block))
    assert seats == {g: seat_label(s).capitalize() for g, s in SEAT_OF_GATE.items()}
    # a stage with an agent names it as the workbench does
    agents = set(re.findall(r'agentName: "([^"]+)"', block)) - {"No agent"}
    assert agents == {name for name in FRIENDLY.values() if "(" not in name}


def test_the_decks_figures_are_the_systems():
    html = _deck()
    shown: dict[str, set[str]] = {}
    for name, value in re.findall(r'data-fact="(\w+)">([^<]+)<', html):
        shown.setdefault(name, set()).add(value.strip())
    builtin = [t for t in plugins.registry().tools_for("explore") if t.origin == "cognos"]
    assert shown == {
        "n_stages": {str(len(graph.STAGES))},
        "n_gates": {str(len(graph.GATES))},
        "n_agents": {str(len({n for n in FRIENDLY.values() if "(" not in n}))},
        "n_seats": {str(len([s for s in SEAT_LABEL if s != "express"]))},
        "n_explore_tools": {str(len(builtin))},
    }


def test_the_deck_is_one_self_contained_file_of_full_screen_slides():
    html = _deck()
    assert len(re.findall(r'<section class="slide', html)) >= 10
    assert "overflow: hidden" in html and "prefers-reduced-motion" in html
    # nothing but the web fonts is fetched: no script or image from elsewhere
    assert not re.findall(r'<script[^>]+src=', html) and "<img" not in html
