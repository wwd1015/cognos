"""Design tokens for the workbench: one palette, two selected modes (not an automatic flip).

Series colors follow the validated reference palette (categorical slots in fixed order); status
colors are reserved for verdicts and always ship with an icon and a label, never color alone.
"""

from __future__ import annotations

SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
SEQ_LIGHT_STEP = {"light": "#86b6ef", "dark": "#184f95"}  # ordinal-safe light step of blue

CHROME = {
    "light": {"surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff", "ink2": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

# Verdict -> (Mantine color, icon, plain label). Color is never the only carrier.
VERDICT = {
    "PASS": ("green", "tabler:circle-check", "Pass"),
    "WARN": ("yellow", "tabler:alert-triangle", "Warn"),
    "FAIL": ("orange", "tabler:alert-octagon", "Fail"),
    "BLOCK": ("red", "tabler:ban", "Block"),
    "ERROR": ("red", "tabler:bug", "Error"),
    "SKIP": ("gray", "tabler:player-skip-forward", "Skipped"),
    "OPEN_QUESTIONS": ("yellow", "tabler:help", "Open questions"),
}

# Step status -> (Mantine color, icon, label)
STEP = {
    "pending": ("gray", "tabler:circle-dashed", "Not started"),
    "running": ("blue", "tabler:loader-2", "Working"),
    "done": ("green", "tabler:circle-check", "Done"),
    "awaiting": ("violet", "tabler:hand-stop", "Awaiting your review"),
    "stale": ("yellow", "tabler:refresh-alert", "Out of date — will re-run"),
    "failed": ("red", "tabler:alert-circle", "Failed"),
    "blocked": ("red", "tabler:ban", "Blocked"),
    "skipped": ("gray", "tabler:player-skip-forward", "Skipped"),
}

RUN = {
    "created": ("gray", "Created"), "running": ("blue", "Running"),
    "awaiting": ("violet", "Awaiting review"), "blocked": ("red", "Blocked"),
    "failed": ("red", "Failed"), "completed": ("teal", "Completed"),
    "approved": ("green", "Approved"), "rejected": ("red", "Rejected"), "legacy": ("gray", "Legacy"),
}

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
MONO = 'ui-monospace, "Cascadia Code", "SF Mono", Consolas, monospace'

MANTINE_THEME = {
    "primaryColor": "indigo",
    "fontFamily": FONT,
    "fontFamilyMonospace": MONO,
    "defaultRadius": "md",
    "headings": {"fontFamily": FONT, "fontWeight": "650"},
    "components": {
        "Card": {"defaultProps": {"withBorder": True, "radius": "md", "padding": "lg"}},
        "Badge": {"defaultProps": {"radius": "sm"}},
    },
}


def plot_layout(scheme: str, **overrides) -> dict:
    """Base Plotly layout: recessive grid and axes, system sans, the mode's own surface."""
    c = CHROME[scheme]
    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": FONT, "size": 12, "color": c["ink2"]},
        "margin": {"l": 56, "r": 16, "t": 16, "b": 44},
        "hoverlabel": {"bgcolor": c["surface"], "bordercolor": c["axis"],
                       "font": {"family": FONT, "color": c["ink"], "size": 12}},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0,
                   "font": {"color": c["ink2"]}},
        "xaxis": {"gridcolor": c["grid"], "linecolor": c["axis"], "zerolinecolor": c["axis"],
                  "tickfont": {"color": c["muted"]}, "title": {"font": {"color": c["ink2"]}}},
        "yaxis": {"gridcolor": c["grid"], "linecolor": c["axis"], "zerolinecolor": c["axis"],
                  "tickfont": {"color": c["muted"]}, "title": {"font": {"color": c["ink2"]}}},
        "bargap": 0.35,
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(layout.get(key), dict):
            layout[key] = {**layout[key], **value}
        else:
            layout[key] = value
    return layout
