"""Design tokens for the workbench: the "Ledger" system shared with IRIS-D.

An editorial, print-like look: warm paper, ink, one oxblood accent, serif display type, mono
numerals, hairline rules instead of boxed cards. Light (paper) and dark (warm black) are two
selected modes. The same tokens exist as CSS variables in ``assets/cognos.css``; this module
holds what Python needs (the Mantine theme and the chart palette).

Series colors are mid-tones chosen to read on paper and on warm black alike; status colors are
reserved for verdicts and always ship with an icon and a label, never color alone.
"""

from __future__ import annotations

_SERIES = ["#9d3a4a", "#41719a", "#b08415", "#2e8063", "#b0673f", "#5b4a8a", "#2a6a62", "#8d8775"]
SERIES = {"light": _SERIES, "dark": _SERIES}
SEQ_LIGHT_STEP = {"light": "#dcbcc2", "dark": "#5a2a33"}  # ordinal-safe light step of the accent

CHROME = {
    "light": {"surface": "#fffefb", "page": "#faf9f6", "ink": "#16140f", "ink2": "#6b6557",
              "muted": "#8d8775", "grid": "#e9e6dc", "axis": "#cfcbbd"},
    "dark": {"surface": "#201d16", "page": "#171510", "ink": "#f0ede4", "ink2": "#9a937f",
             "muted": "#7d7767", "grid": "#2b271e", "axis": "#3d382c"},
}

STATUS = {"good": "#2e8063", "warning": "#b08415", "serious": "#b0673f", "critical": "#a8232f"}

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
    "awaiting": ("oxblood", "tabler:hand-stop", "Awaiting your review"),
    "stale": ("yellow", "tabler:refresh-alert", "Out of date — will re-run"),
    "failed": ("red", "tabler:alert-circle", "Failed"),
    "blocked": ("red", "tabler:ban", "Blocked"),
    "skipped": ("gray", "tabler:player-skip-forward", "Skipped"),
}

RUN = {
    "created": ("gray", "Created"), "running": ("blue", "Running"),
    "awaiting": ("oxblood", "Awaiting review"), "blocked": ("red", "Blocked"),
    "failed": ("red", "Failed"), "completed": ("teal", "Completed"),
    "approved": ("green", "Approved"), "rejected": ("red", "Rejected"), "legacy": ("gray", "Legacy"),
}

FONT = '"Instrument Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
DISPLAY = '"Source Serif 4", Georgia, "Times New Roman", serif'
MONO = '"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace'
# Loaded when the network allows; every stack above falls back to a system face offline.
FONT_CSS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&"
            "family=Instrument+Sans:wght@400;500;600;700&"
            "family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap")


def _mix(a: str, b: str, t: float) -> str:
    pa, pb = (int(a[i:i + 2], 16) for i in (1, 3, 5)), (int(b[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(pa, pb, strict=True))


def ramp(base: str) -> list[str]:
    """A ten-step Mantine palette around ``base`` (step 6): paper tints below, ink shades above."""
    return ([_mix(base, "#faf9f6", t) for t in (0.92, 0.84, 0.70, 0.52, 0.34, 0.16)] + [base]
            + [_mix(base, "#16140f", t) for t in (0.14, 0.28, 0.44)])


# Mantine's named colors, re-pitched to the Ledger palette so every badge, alert and button in
# the app is in key without each call site choosing a hex.
COLORS = {
    "oxblood": ramp("#7d2230"), "red": ramp("#a8232f"), "orange": ramp("#b0673f"),
    "yellow": ramp("#9a6b00"), "green": ramp("#1a5e45"), "teal": ramp("#2a6a62"),
    "blue": ramp("#27506e"), "grape": ramp("#5b4a8a"), "violet": ramp("#5b4a8a"),
    "indigo": ramp("#7d2230"),
    "gray": ["#faf9f6", "#f5f3ee", "#eceae2", "#dedbd0", "#c9c5b7", "#a8a393", "#8d8775",
             "#6b6557", "#3d392f", "#16140f"],
    "dark": ["#f0ede4", "#cfcabb", "#9a937f", "#6f6957", "#3d382c", "#2b271e", "#201d16",
             "#171510", "#121009", "#0c0a06"],
}

MANTINE_THEME = {
    "primaryColor": "oxblood",
    "primaryShade": {"light": 6, "dark": 3},
    "autoContrast": True,
    "luminanceThreshold": 0.36,
    "colors": COLORS,
    "white": "#fffefb",
    "black": "#16140f",
    "fontFamily": FONT,
    "fontFamilyMonospace": MONO,
    "defaultRadius": "sm",
    "radius": {"xs": "2px", "sm": "3px", "md": "4px", "lg": "6px", "xl": "6px"},
    "headings": {"fontFamily": DISPLAY, "fontWeight": "600"},
    "components": {
        "Card": {"defaultProps": {"withBorder": False, "radius": 0, "padding": 0}},
        "Badge": {"defaultProps": {"radius": "xs"}},
        "Button": {"defaultProps": {"radius": "sm"}},
        "Table": {"defaultProps": {"highlightOnHover": True}},
    },
}


def plot_layout(scheme: str, **overrides) -> dict:
    """Base Plotly layout: recessive grid and axes, mono numerals, a transparent ground."""
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
        "colorway": SERIES[scheme],
        "xaxis": {"gridcolor": c["grid"], "linecolor": c["axis"], "zerolinecolor": c["axis"],
                  "tickfont": {"color": c["muted"], "family": MONO, "size": 11},
                  "title": {"font": {"color": c["ink2"]}}},
        "yaxis": {"gridcolor": c["grid"], "linecolor": c["axis"], "zerolinecolor": c["axis"],
                  "tickfont": {"color": c["muted"], "family": MONO, "size": 11},
                  "title": {"font": {"color": c["ink2"]}}},
        "bargap": 0.35,
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(layout.get(key), dict):
            layout[key] = {**layout[key], **value}
        else:
            layout[key] = value
    return layout
