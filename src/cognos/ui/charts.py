"""Plotly figures for the workbench. Every figure is theme-aware (``scheme`` = light | dark), has a
single y-axis, thin marks, a hover layer, and a legend whenever it shows two or more series."""

from __future__ import annotations

import math

import plotly.graph_objects as go

from .theme import CHROME, SEQ_LIGHT_STEP, SERIES, STATUS, plot_layout


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def ledger(records: list[dict], metric: str, champion_label: str | None, scheme: str) -> go.Figure:
    """Every experiment the ratchet tried: kept (new best) vs discarded, champion ringed."""
    s, c = SERIES[scheme], CHROME[scheme]
    ok = [r for r in records if r.get("status") != "crash"]
    fig = go.Figure()
    for status, name, color, size in (("keep", "Kept (new best)", s[0], 10),
                                      ("discard", "Discarded", c["muted"], 8)):
        rows = [r for r in ok if r["status"] == status]
        fig.add_trace(go.Scatter(
            x=[r["idx"] for r in rows], y=[r["metric_value"] for r in rows], mode="markers",
            name=name, marker={"color": color, "size": size, "line": {"width": 2, "color": c["surface"]}},
            customdata=[[r["label"], r["n_features"], r["cv_std"]] for r in rows],
            hovertemplate="<b>%{customdata[0]}</b><br>" + metric
                          + " %{y:.4f} ± %{customdata[2]:.4f}<br>%{customdata[1]} features<extra></extra>"))
    champ = [r for r in ok if r["label"] == champion_label]
    if champ:
        r = champ[-1]
        fig.add_trace(go.Scatter(
            x=[r["idx"]], y=[r["metric_value"]], mode="markers+text", name="Champion",
            marker={"size": 18, "color": "rgba(0,0,0,0)", "line": {"width": 2, "color": c["ink"]}},
            text=["champion"], textposition="top center", textfont={"color": c["ink2"], "size": 11},
            hoverinfo="skip"))
    fig.update_layout(**plot_layout(scheme, xaxis={"title": {"text": "experiment"}},
                                    yaxis={"title": {"text": f"cross-validated {metric}"}}),
                      height=300)
    return fig


def coefficients(coefs: dict | None, pvalues: dict | None, importances: dict | None,
                 scheme: str) -> go.Figure | None:
    """Coefficient (or importance) bars, sorted by magnitude; significance as fill strength."""
    s, c = SERIES[scheme], CHROME[scheme]
    fig = go.Figure()
    if coefs:
        items = [(k, v) for k, v in coefs.items() if k.lower() not in ("const", "intercept")
                 and _num(v) is not None]
        items.sort(key=lambda kv: abs(kv[1]))
        items = items[-20:]
        pv = pvalues or {}
        sig = [k for k, _ in items if (_num(pv.get(k)) is not None and pv[k] < 0.05)]
        for label, keys, color in (("significant (p < 0.05)", sig, s[0]),
                                   ("not significant", [k for k, _ in items if k not in sig],
                                    SEQ_LIGHT_STEP[scheme])):
            rows = [(k, v) for k, v in items if k in keys]
            if not rows:
                continue
            fig.add_trace(go.Bar(
                y=[k for k, _ in rows], x=[v for _, v in rows], orientation="h", name=label,
                marker={"color": color, "line": {"width": 0}, "cornerradius": 4},
                customdata=[_num(pv.get(k)) for k, _ in rows],
                hovertemplate="<b>%{y}</b><br>coefficient %{x:.4f}<br>p-value %{customdata:.4f}"
                              "<extra></extra>"))
        fig.update_layout(barmode="overlay")
        title = "coefficient (log-odds / units of the target)"
    elif importances:
        items = sorted(importances.items(), key=lambda kv: kv[1])[-20:]
        fig.add_trace(go.Bar(y=[k for k, _ in items], x=[v for _, v in items], orientation="h",
                             marker={"color": s[0], "cornerradius": 4}, name="importance",
                             hovertemplate="<b>%{y}</b><br>importance %{x:.4f}<extra></extra>"))
        title = "feature importance"
    else:
        return None
    n = len(fig.data[0].y) + (len(fig.data[1].y) if len(fig.data) > 1 else 0)
    fig.update_layout(**plot_layout(scheme, xaxis={"title": {"text": title}},
                                    margin={"l": 150}), height=max(220, 26 * n + 80),
                      showlegend=len(fig.data) > 1)
    fig.update_yaxes(categoryorder="array",
                     categoryarray=[k for tr in fig.data for k in tr.y])
    fig.add_vline(x=0, line={"color": c["axis"], "width": 1})
    return fig


def calibration(table: list[dict], scheme: str) -> go.Figure | None:
    """Predicted vs observed default rate per score band, against the perfect-calibration line."""
    if not table:
        return None
    s, c = SERIES[scheme], CHROME[scheme]
    pred = [r["predicted"] for r in table]
    obs = [r["observed"] for r in table]
    hi = max(max(pred), max(obs)) * 1.08 or 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", name="perfect calibration",
                             line={"color": c["axis"], "width": 1, "dash": "dot"}, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=pred, y=obs, mode="lines+markers", name="score bands",
        line={"color": s[0], "width": 2}, marker={"size": 9, "color": s[0],
                                                   "line": {"width": 2, "color": c["surface"]}},
        customdata=[[r["band"] + 1, r["n"]] for r in table],
        hovertemplate="band %{customdata[0]} (n=%{customdata[1]})<br>predicted %{x:.3%}"
                      "<br>observed %{y:.3%}<extra></extra>"))
    fig.update_layout(**plot_layout(scheme, xaxis={"title": {"text": "mean predicted PD"},
                                                   "tickformat": ".0%"},
                                    yaxis={"title": {"text": "observed default rate"},
                                           "tickformat": ".0%"}), height=320)
    return fig


def rubric(scores: dict[str, float], scheme: str) -> go.Figure:
    """Validation rubric (0-1 per axis). Status color + the value label, never color alone."""
    c = CHROME[scheme]
    axes = list(scores)[::-1]
    vals = [scores[a] for a in axes]

    def status(v: float) -> str:
        return STATUS["good"] if v >= 0.8 else (STATUS["warning"] if v >= 0.5 else STATUS["critical"])

    fig = go.Figure(go.Bar(
        y=axes, x=vals, orientation="h", marker={"color": [status(v) for v in vals],
                                                 "cornerradius": 4},
        text=[f"{v:.2f}" for v in vals], textposition="outside", textfont={"color": c["ink2"]},
        hovertemplate="<b>%{y}</b><br>score %{x:.2f}<extra></extra>", showlegend=False))
    fig.update_layout(**plot_layout(scheme, xaxis={"range": [0, 1.12], "title": {"text": "score (1 = no concern)"}},
                                    margin={"l": 110}), height=60 + 38 * len(axes))
    return fig


def term_structure(ts: dict | None, scheme: str) -> go.Figure | None:
    """PD term structure of a hazard champion (mean cumulative PD per period)."""
    if not ts:
        return None
    periods = ts.get("periods") or list(range(1, len(ts.get("mean_cumulative_pd", [])) + 1))
    cum = ts.get("mean_cumulative_pd") or []
    if not cum:
        return None
    s = SERIES[scheme]
    fig = go.Figure(go.Scatter(x=periods, y=cum, mode="lines+markers", name="cumulative PD",
                               line={"color": s[0], "width": 2}, marker={"size": 8},
                               hovertemplate="period %{x}<br>cumulative PD %{y:.3%}<extra></extra>"))
    fig.update_layout(**plot_layout(scheme, xaxis={"title": {"text": "period"}, "dtick": 1},
                                    yaxis={"title": {"text": "mean cumulative PD"},
                                           "tickformat": ".1%"}), height=260)
    return fig
