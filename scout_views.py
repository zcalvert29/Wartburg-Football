"""
Rendering for the opposing-offense scout (one scouted team at a time). Every render_* function takes the filtered plays
(and, where it needs whole drives or the previous snap, the unfiltered games) and draws one section.

Conventions (same as the Dowling app):
  - Rates are shares of run/pass snaps. Penalty snaps (PASS RESULT F) count toward what was called but not
    toward success, EPA, yards, or conversions.
  - A "tell" is a split 70%+ one way, graded by the Wilson lower bound (90%): tell = lower bound >= 70%,
    lean = >= 55%, otherwise small sample.
"""
from __future__ import annotations

import html as _html

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import bv_data as bd

RUN_COLOR = "#BA7517"
PASS_COLOR = "#534AB7"
MIN_TELL_SHARE = 0.70
TELL_MIN_N = 5
TENDENCY_GRADES = [(0.70, "Solid"), (0.55, "Likely")]
BADGE = {"Solid": ("tell", "#FCEBEB", "#791F1F"), "Likely": ("lean", "#FAEEDA", "#854F0B"),
         "Small sample": ("small sample", "rgba(128,128,128,.15)", "inherit")}
CARD = "border:0.5px solid rgba(128,128,128,.35);border-radius:8px;padding:8px 10px"
DN_LABEL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
RZ_BANDS = [("Red zone (11–20)", 11, 20), ("Inside the 10 (6–10)", 6, 10), ("Goal line (1–5)", 1, 5)]
RZ_RESULTS = [("Touchdown", "#1D9E75"), ("FG Made", "#E8B923"), ("FG Missed", "#E24B4A"),
              ("Turnover on Downs", "#E24B4A"), ("Turnover", "#A32D2D"), ("Punt", "#B4B2A9"),
              ("End of Half", "#888780"), ("End of Game", "#888780")]


# ==============================================================================
# Helpers
# ==============================================================================
def wilson_lower(k: float, n: float, z: float = 1.645) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((center - margin) / denom)


def tendency_grade(k: float, n: float) -> tuple[str, float]:
    lb = wilson_lower(k, n)
    for cut, name in TENDENCY_GRADES:
        if lb >= cut:
            return name, lb
    return "Small sample", lb


def grade_word(k: int, n: int) -> str:
    if n < TELL_MIN_N or k / n < MIN_TELL_SHARE:
        return ""
    return BADGE[tendency_grade(k, n)[0]][0]


def badge(k: int, n: int) -> str:
    word = grade_word(k, n)
    if not word:
        return ""
    _, bg, fg = next(v for v in BADGE.values() if v[0] == word)
    return (f'<span style="font-size:11px;padding:1px 7px;border-radius:999px;background:{bg};color:{fg};'
            f'margin-left:6px;white-space:nowrap">{word}</span>')


def split_bar(run_share: float, height: int = 6) -> str:
    r = 0 if pd.isna(run_share) else run_share * 100
    return (f'<div style="display:flex;height:{height}px;border-radius:3px;overflow:hidden;margin:3px 0">'
            f'<span style="width:{r:.1f}%;background:{RUN_COLOR}"></span>'
            f'<span style="width:{100 - r:.1f}%;background:{PASS_COLOR}"></span></div>')


LEGEND = (f'<div style="display:flex;gap:14px;font-size:12px;margin:2px 0 8px;flex-wrap:wrap">'
          f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{RUN_COLOR};'
          f'margin-right:4px"></span>Run</span><span><span style="display:inline-block;width:10px;height:10px;'
          f'border-radius:2px;background:{PASS_COLOR};margin-right:4px"></span>Pass</span>'
          f'<span style="opacity:.7">tell = 70%+ one way and still 70%+ at the low end of a 90% interval · '
          f'lean = low end 55%+</span></div>')


def team_of(d: pd.DataFrame) -> str:
    """Name of the scouted team in these plays (the app always passes one team's plays)."""
    if "team" in d and d["team"].notna().any():
        return str(d["team"].dropna().iat[0])
    return bd.SCOUTED_TEAM


def esc(x) -> str:
    return _html.escape("" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x))


def snaps(d: pd.DataFrame) -> pd.DataFrame:
    """Run/pass snaps (calls), including penalty snaps."""
    return d[d["snap"]]


def top_share(s: pd.Series) -> tuple[str | None, float, int]:
    s = s.dropna()
    if s.empty:
        return None, np.nan, 0
    vc = s.value_counts()
    return vc.index[0], vc.iloc[0] / len(s), int(vc.iloc[0])


def summarize(d: pd.DataFrame, by, top_cols: tuple[str, ...] = ("CALL GROUP",), min_n: int = 1,
              sort_by_count: bool = True) -> pd.DataFrame:
    """Run/pass split, efficiency, and most-common values, grouped by `by`."""
    d = snaps(d)
    by = [by] if isinstance(by, str) else list(by)
    d = d.dropna(subset=by)
    if d.empty:
        return pd.DataFrame()
    rows = []
    for key, g in d.groupby(by, sort=False, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        n = len(g)
        live = g[g["live"]]
        pass_rate = g["PASS"].mean()
        k = round(max(pass_rate, 1 - pass_rate) * n)
        row = dict(zip(by, key))
        row.update({"Plays": n, "Run": (1 - pass_rate) * 100, "Pass": pass_rate * 100,
                    "Read": grade_word(k, n),
                    "Success": live["success"].mean() * 100, "EPA/play": live["epa"].mean(),
                    "Yds/play": live["GN/LS"].mean(), "Explosive": live["explosive_play"].mean() * 100})
        for c in top_cols:
            v, sh, _ = top_share(g[c])
            row[f"Top {c.lower()}"] = f"{v} ({sh:.0%})" if v is not None else ""
        rows.append(row)
    t = pd.DataFrame(rows)
    t = t[t["Plays"] >= min_n]
    if sort_by_count:
        t = t.sort_values("Plays", ascending=False)
    return t.reset_index(drop=True)


def show(t: pd.DataFrame, title: str | None = None, caption: str | None = None, height: int | None = None):
    if title:
        st.markdown(f"**{title}**")
    if t is None or t.empty:
        st.info("No plays match the current filters.")
        return
    cfg = {}
    for c in t.columns:
        if c in ("Run", "Pass"):
            cfg[c] = st.column_config.NumberColumn(c, format="%.0f%%")
        elif c in ("Success", "Explosive", "Share", "Conv rate", "TD rate", "Motion", "To field", "To strength",
                   "Sack rate", "Comp rate") or c.endswith(" %"):
            cfg[c] = st.column_config.NumberColumn(c, format="%.0f%%")
        elif c == "EPA/play" or c == "EPA":
            cfg[c] = st.column_config.NumberColumn(c, format="%+.2f")
        elif c in ("Yds/play", "Avg gain", "Points / drive"):
            cfg[c] = st.column_config.NumberColumn(c, format="%.1f")
    kw = {"height": height} if height else {}
    st.dataframe(t, hide_index=True, column_config=cfg, width="stretch", **kw)
    if caption:
        st.caption(caption)


PLAY_COLS = {"game_label": "Game", "SERIES": "Ser", "SNAP": "#", "DN": "Dn", "DIST": "Dist", "YARD LN": "Yd ln",
             "HASH": "Hash", "PERSONNEL": "Pers", "OFF FORM": "Formation", "OFF STR": "Str", "BACKFIELD": "Bkfd",
             "MOTION": "Motion", "OFF PLAY": "Play", "CONCEPT": "Concept", "PROTECTION": "Prot",
             "PASS RESULT": "Res", "GN/LS": "Gain", "epa": "EPA"}


def plays_table(d: pd.DataFrame) -> pd.DataFrame:
    t = d[list(PLAY_COLS)].rename(columns=PLAY_COLS)
    t["Game"] = t["Game"].str.replace(r"^\d{4} · ", "", regex=True)
    return t


def show_plays(d: pd.DataFrame, title: str, caption: str | None = None, extra: dict | None = None):
    st.markdown(f"**{title}**")
    if d.empty:
        st.info("No plays match the current filters.")
        return
    t = plays_table(d)
    for name, values in (extra or {}).items():
        t.insert(0, name, values)
    st.dataframe(t, hide_index=True, width="stretch",
                 column_config={"EPA": st.column_config.NumberColumn("EPA", format="%+.2f"),
                                "Gain": st.column_config.NumberColumn("Gain", format="%d")})
    if caption:
        st.caption(caption)


def kpis(d: pd.DataFrame, drives: pd.DataFrame):
    s = snaps(d)
    live = s[s["live"]]
    third = live[live["DN"] == 3]
    c = st.columns(7)
    c[0].metric("Snaps", len(s))
    c[1].metric("Run rate", f"{1 - s['PASS'].mean():.0%}" if len(s) else "–")
    c[2].metric("Success rate", f"{live['success'].mean():.0%}" if len(live) else "–")
    c[3].metric("EPA / play", f"{live['epa'].mean():+.2f}" if len(live) else "–")
    c[4].metric("Explosive rate", f"{live['explosive_play'].mean():.0%}" if len(live) else "–",
                help="Runs (incl. QB runs) of 10+ yards, passes of 20+.")
    c[5].metric("3rd down conv.", f"{third['converted'].mean():.0%}" if len(third) else "–")
    c[6].metric("Points / drive", f"{drives['points'].mean():.1f}" if len(drives) else "–",
                help="Touchdowns counted as 7. Uses whole drives in the selected games.")


# ==============================================================================
# Visual building blocks (HTML, so they follow the light/dark theme like the cards above)
# ==============================================================================
HEAT_RGB = (83, 74, 183)  # PASS_COLOR; one hue, light -> dark = small -> large
GRID_ROW = "border-top:0.5px solid rgba(128,128,128,.25)"


def _two_prop_z(k1, n1, k2, n2) -> float:
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else np.nan
    return (k1 / n1 - k2 / n2) / se if se and se > 0 else 0.0


def _fmt(val, kind: str) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "–"
    return {"pct": f"{val:.0%}", "epa": f"{val:+.2f}", "num": f"{val:.1f}", "int": f"{int(val)}"}.get(kind, esc(val))


def split_rows_html(items: list[tuple[str, pd.DataFrame]], first_header: str, base_pass: float | None = None,
                    extra: list[tuple[str, callable, str]] | None = None, top_col: str | None = "CALL GROUP",
                    dim_under: int = TELL_MIN_N) -> str:
    """
    One row per (label, snaps): plays, a run/pass split bar with the tell badge, how far the pass rate sits from
    `base_pass`, any `extra` columns (header, fn(g) -> value, kind), the top call, success and EPA.
    Same layout as the Dowling sequencing tab.
    """
    extra = extra or []
    cols = ["minmax(150px,1.3fr)", "52px", "minmax(150px,2fr)"]
    heads = [first_header, '<div style="text-align:right">Plays</div>', "Run / pass"]
    if base_pass is not None:
        cols.append("minmax(120px,1fr)")
        heads.append("vs. usual pass rate")
    for h, _, _ in extra:
        cols.append("72px")
        heads.append(f'<div style="text-align:right">{h}</div>')
    if top_col:
        cols.append("minmax(120px,1.2fr)")
        heads.append({"CALL GROUP": "Top call", "OFF FORM": "Top formation", "MOTION TYPE": "Top motion",
                      "CONCEPT": "Top concept"}.get(top_col, "Top " + top_col.lower()))
    cols += ["64px", "64px"]
    heads += ['<div style="text-align:right">Success</div>', '<div style="text-align:right">EPA</div>']
    grid = f"display:grid;grid-template-columns:{' '.join(cols)};gap:10px;align-items:center"
    head = f'<div style="{grid};font-size:11px;opacity:.6;padding-bottom:4px">' + "".join(
        h if h.startswith("<div") else f"<div>{h}</div>" for h in heads) + "</div>"
    rows = []
    for label, g in items:
        n = len(g)
        if n == 0:
            continue
        live = g[g["live"]]
        run = 1 - g["PASS"].mean()
        k = round(max(run, 1 - run) * n)
        cells = [f"<div>{esc(label)}</div>", f'<div style="text-align:right;opacity:.7">{n}</div>',
                 f'<div>{split_bar(run, 8)}<div style="font-size:11px;opacity:.75">Run {run:.0%} · Pass {1 - run:.0%}'
                 f'{badge(k, n)}</div></div>']
        if base_pass is not None:
            diff = (1 - run) - base_pass
            cells.append(f'<div style="font-size:12px;opacity:.7">{diff * 100:+.0f} pts pass</div>' if abs(diff) >= .05
                         else '<div style="font-size:12px;opacity:.4">about usual</div>')
        for _, fn, kind in extra:
            cells.append(f'<div style="text-align:right">{_fmt(fn(g), kind)}</div>')
        if top_col:
            v, sh, _ = top_share(g[top_col])
            cells.append(f'<div style="font-size:12px">{esc(v)} <span style="opacity:.6">{sh:.0%}</span></div>'
                         if v is not None else "<div></div>")
        cells += [f'<div style="text-align:right">{_fmt(live["success"].mean(), "pct")}</div>',
                  f'<div style="text-align:right">{_fmt(live["epa"].mean(), "epa")}</div>']
        rows.append(f'<div style="{grid};padding:6px 0;{GRID_ROW};font-size:13px'
                    f'{";opacity:.55" if n < dim_under else ""}">{"".join(cells)}</div>')
    if not rows:
        return ""
    width = 560 + 130 * (base_pass is not None) + 82 * len(extra) + 130 * bool(top_col)
    return LEGEND + f'<div style="overflow-x:auto"><div style="min-width:{width}px">{head}{"".join(rows)}</div></div>'


def show_split_rows(items, first_header: str, title: str | None = None, caption: str | None = None, **kw):
    if title:
        st.markdown(f"**{title}**")
    h = split_rows_html(items, first_header, **kw)
    st.html(h) if h else st.info("No plays match the current filters.")
    if caption:
        st.caption(caption)


def groups(d: pd.DataFrame, col, min_n: int = 1, limit: int | None = None) -> list[tuple[str, pd.DataFrame]]:
    """(label, snaps) per value of col (or list of cols), biggest first."""
    cols = [col] if isinstance(col, str) else list(col)
    x = snaps(d).dropna(subset=cols)
    out = [(" · ".join(map(str, k if isinstance(k, tuple) else (k,))), g)
           for k, g in x.groupby(cols if len(cols) > 1 else cols[0], sort=False)]
    out = sorted([o for o in out if len(o[1]) >= min_n], key=lambda o: -len(o[1]))
    return out[:limit] if limit else out


def heat_html(t: pd.DataFrame, kind: str = "pct", row_header: str = "", total_col: pd.Series | None = None,
              col_width: int = 64) -> str:
    """A table where each cell is shaded by its value (one hue). kind: 'pct' (0-1 shares) or 'int' (counts)."""
    if t.empty:
        return ""
    vmax = float(np.nanmax(t.to_numpy(dtype=float))) or 1.0
    ncol = len(t.columns) + (total_col is not None)
    grid = f"display:grid;grid-template-columns:minmax(110px,1.4fr) repeat({ncol},minmax({col_width}px,1fr));gap:3px"
    cells = [f'<div style="font-size:11px;opacity:.6;align-self:end">{esc(row_header)}</div>']
    cells += [f'<div style="font-size:11px;font-weight:600;text-align:center;align-self:end">{esc(c)}</div>'
              for c in t.columns]
    if total_col is not None:
        cells.append('<div style="font-size:11px;opacity:.6;text-align:center;align-self:end">Plays</div>')
    r, g_, b = HEAT_RGB
    for idx, row in t.iterrows():
        cells.append(f'<div style="font-size:12px;font-weight:600;align-self:center">{esc(idx)}</div>')
        for c, val in row.items():
            if pd.isna(val) or val == 0:
                cells.append('<div style="border-radius:4px;background:rgba(128,128,128,.06);height:26px"></div>')
                continue
            a = 0.12 + 0.78 * (float(val) / vmax)
            ink = "#fff" if a > 0.5 else "inherit"
            txt = f"{val:.0%}" if kind == "pct" else f"{int(val)}"
            cells.append(f'<div title="{esc(idx)} · {esc(c)}: {txt}" style="border-radius:4px;background:rgba({r},{g_},{b},{a:.2f});'
                         f'color:{ink};font-size:12px;text-align:center;line-height:26px;height:26px">{txt}</div>')
        if total_col is not None:
            cells.append(f'<div style="font-size:12px;opacity:.6;text-align:center;line-height:26px">'
                         f'{int(total_col.get(idx, 0))}</div>')
    width = 120 + ncol * (col_width + 3)
    return f'<div style="overflow-x:auto"><div style="{grid};min-width:{width}px">{"".join(cells)}</div></div>'


def show_heat(t: pd.DataFrame, title: str, caption: str | None = None, **kw):
    st.markdown(f"**{title}**")
    h = heat_html(t, **kw)
    st.html(h) if h else st.info("No plays match the current filters.")
    if caption:
        st.caption(caption)


def hbars_html(items: list[tuple[str, float, str, str]], kind: str = "pct", vmax: float | None = None) -> str:
    """Horizontal bars: (label, value, note, color). The value sits in its own column so a full bar never wraps."""
    if not items:
        return ""
    vmax = vmax or max(v for _, v, _, _ in items) or 1
    rows = []
    for label, val, note, color in items:
        w = max(0.0, min(1.0, val / vmax)) * 100
        txt = f"{val:.0%}" if kind == "pct" else f"{val:.1f}"
        rows.append(f'<div title="{esc(label)}: {txt} {esc(note)}" style="display:grid;grid-template-columns:'
                    f'minmax(90px,1.2fr) minmax(100px,3fr) minmax(86px,auto);gap:10px;align-items:center;'
                    f'padding:3px 0;font-size:12px">'
                    f'<div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{esc(label)}</div>'
                    f'<div style="background:rgba(128,128,128,.10);border-radius:0 4px 4px 0;height:14px">'
                    f'<div style="height:14px;width:{w:.1f}%;min-width:2px;background:{color};'
                    f'border-radius:0 4px 4px 0"></div></div>'
                    f'<div style="white-space:nowrap"><b>{txt}</b> <span style="opacity:.55">{esc(note)}</span></div>'
                    f'</div>')
    return "".join(rows)


def show_hbars(items, title: str, caption: str | None = None, **kw):
    st.markdown(f"**{title}**")
    h = hbars_html(items, **kw)
    st.html(h) if h else st.info("No plays match the current filters.")
    if caption:
        st.caption(caption)


def type_color(g: pd.DataFrame) -> str:
    """Run color if the group is mostly runs, pass color otherwise."""
    return RUN_COLOR if g["PASS"].mean() < 0.5 else PASS_COLOR


# ==============================================================================
# 1. Tendencies
# ==============================================================================
def dd_grid_html(d: pd.DataFrame) -> str:
    d = snaps(d).dropna(subset=["Distance"])
    if d.empty:
        return ""
    cells = ['<div></div>'] + [f'<div style="font-size:12px;font-weight:600;text-align:center">{c}</div>'
                               for c in bd.DISTANCE_ORDER]
    for dn in [1, 2, 3, 4]:
        cells.append(f'<div style="font-size:13px;font-weight:600;align-self:center">{["1st", "2nd", "3rd", "4th"][dn - 1]}</div>')
        for dist in bd.DISTANCE_ORDER:
            c = d[(d["DN"] == dn) & (d["Distance"] == dist)]
            n = len(c)
            if n == 0:
                cells.append(f'<div style="{CARD};opacity:.4;font-size:12px">no plays</div>')
                continue
            run = 1 - c["PASS"].mean()
            k = round(max(run, 1 - run) * n)
            lead = f"Run {run:.0%}" if run >= 0.5 else f"Pass {1 - run:.0%}"
            pers, psh, _ = top_share(c["PERSONNEL"])
            form, fsh, _ = top_share(c["OFF FORM"])
            grp, gsh, _ = top_share(c["CONCEPT"])
            faded = "opacity:.55;" if n < TELL_MIN_N else ""
            cells.append(
                f'<div style="{CARD};{faded}font-size:12px;line-height:1.45">'
                f'<div><b style="font-size:14px">{lead}</b>{badge(k, n)}'
                f'<span style="float:right;opacity:.6">{n}</span></div>{split_bar(run)}'
                f'<div style="opacity:.8">{esc(grp)} {gsh:.0%}</div>'
                f'<div style="opacity:.65">{esc(pers)} pers {psh:.0%} · {esc(form)} {fsh:.0%}</div></div>')
    return (LEGEND + '<div style="overflow-x:auto"><div style="display:grid;grid-template-columns:44px '
            'repeat(4,minmax(150px,1fr));gap:6px;min-width:660px">' + "".join(cells) + "</div></div>")


def render_tendencies(d_any_down: pd.DataFrame, d: pd.DataFrame, drives: pd.DataFrame):
    kpis(d, drives)
    st.markdown("**Down & distance tendencies**")
    h = dd_grid_html(d_any_down)
    st.html(h) if h else st.info("No plays match the current filters.")
    st.caption("Each box: the run/pass lean, the most common concept, and the most-used personnel and "
               "formation. Faded boxes have fewer than 5 plays. Ignores the down and distance filters.")
    with st.expander("Down & distance as a table"):
        t = summarize(d_any_down, ["DN", "Distance"], ("CALL GROUP", "OFF FORM"), sort_by_count=False)
        if not t.empty:
            t["_o"] = t["Distance"].map({v: i for i, v in enumerate(bd.DISTANCE_ORDER)})
            t = t.sort_values(["DN", "_o"]).drop(columns="_o")
        show(t, f"{team_of(d_any_down)} offensive tendencies by down & distance")

    st.markdown("**What they call from each look**")
    c1, c2 = st.columns([4, 1])
    look = c1.radio("Look", ["Personnel", "Formation", "Backfield", "Personnel + formation", "Formation family"],
                    horizontal=True, key="look_col", label_visibility="collapsed")
    show_all = c2.toggle("Show all", key="look_all", help="Off = the 15 most-used looks.")
    cols = {"Personnel": ["PERSONNEL"], "Formation": ["OFF FORM"], "Backfield": ["BACKFIELD"],
            "Personnel + formation": ["PERSONNEL", "OFF FORM"], "Formation family": ["FORM BASE"]}[look]
    show_split_rows(groups(d, cols, limit=None if show_all else 15), look, top_col="CONCEPT",
                    extra=[("Yds/play", lambda g: g.loc[g["live"], "GN/LS"].mean(), "num"),
                           ("Explosive", lambda g: g.loc[g["live"], "explosive_play"].mean(), "pct")],
                    caption="Success, EPA, yards and explosives exclude penalty snaps. Faded rows have under 5 plays.")

    for dn in (3, 4):
        x = snaps(d_any_down[d_any_down["DN"] == dn])
        items = [(DIST_LABEL[dist], x[x["Distance"] == dist]) for dist in bd.DISTANCE_ORDER]
        show_split_rows(items, f"{['3rd', '4th'][dn - 3]} &", f"{['3rd', '4th'][dn - 3]} downs",
                        extra=[("Conv.", lambda g: g.loc[g["live"], "converted"].mean(), "pct")])
    for dn in (3, 4):
        with st.expander(f"Every {['3rd', '4th'][dn - 3]} down call", expanded=False):
            x = snaps(d_any_down[d_any_down["DN"] == dn]).sort_values(["date", "PLAY #"])
            conv = x["converted"].map({1.0: "Yes", 0.0: "No"}).fillna("Penalty")
            show_plays(x, f"{len(x)} plays, oldest first", extra={"Converted": conv.values})

    live = d[d["live"]]
    show_plays(live.nlargest(10, "epa"), "Best 10 plays by EPA (most likely to come back)")


# ==============================================================================
# 2. Tendency tree
# ==============================================================================
TREE_LEVELS = {"Personnel": "PERSONNEL", "Formation": "OFF FORM", "Formation family": "FORM BASE",
               "Backfield": "BACKFIELD", "Motion": "MOTION", "Run/pass": "PLAY TYPE", "Call group": "CALL GROUP",
               "Concept": "CONCEPT"}


def render_tree(d: pd.DataFrame):
    c1, c2 = st.columns([3, 1])
    levels = c1.multiselect("Tree levels (in order)", list(TREE_LEVELS),
                            default=["Personnel", "Formation", "Run/pass", "Concept"], key="tree_levels")
    min_n = c2.number_input("Hide branches under", 1, 20, 3, key="tree_min")
    if not levels:
        st.info("Pick at least one level.")
        return
    cols = [TREE_LEVELS[l] for l in levels]
    x = snaps(d).copy()
    for c in cols:
        x[c] = x[c].fillna("(none)" if c == "MOTION" else "(untagged)").astype(str)
    if x.empty:
        st.info("No plays match the current filters.")
        return
    counts = x.groupby(cols).agg(Plays=("snap", "size"), Pass=("PASS", "mean"),
                                 Success=("success", "mean")).reset_index()
    keep = x.groupby(cols[0])["snap"].transform("size") >= min_n
    x = x[keep]
    counts = counts[counts[cols[0]].isin(x[cols[0]].unique())]
    fig = px.icicle(counts, path=[px.Constant("All snaps")] + cols, values="Plays", color="Pass",
                    color_continuous_scale=[[0, RUN_COLOR], [0.5, "#D3D1C7"], [1, PASS_COLOR]],
                    range_color=[0, 1], hover_data={"Success": ":.0%", "Pass": ":.0%"})
    fig.update_traces(textinfo="label+value+percent parent", root_color="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=620,
                      coloraxis_colorbar=dict(title="Pass %", tickformat=".0%"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Width = share of the parent's snaps; color = pass rate (orange run, purple pass). Click a "
               "branch to zoom in, click the top bar to zoom out.")

    # Takeaways: the strongest one-way branches at depth 2+.
    out = []
    for i in range(1, len(cols)):
        parent, child = cols[:i], cols[i]
        for key, g in x.groupby(parent):
            n = len(g)
            if n < max(min_n, TELL_MIN_N):
                continue
            v, sh, k = top_share(g[child])
            if v is None or sh < MIN_TELL_SHARE:
                continue
            grade, lb = tendency_grade(k, n)
            key = key if isinstance(key, tuple) else (key,)
            out.append((lb, f"From **{' / '.join(map(str, key))}** → **{v}** {sh:.0%} ({k} of {n}) — {BADGE[grade][0]}"))
    if out:
        st.markdown("**Strongest branches**")
        for _, line in sorted(out, reverse=True)[:8]:
            st.markdown(f"- {line}")


# ==============================================================================
# 3. Field & strength
# ==============================================================================
HASHES = [("L", "Left hash"), ("M", "Middle"), ("R", "Right hash")]
STRENGTHS = [("L", "Strength left"), ("R", "Strength right")]


def _fs_flags(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    dir_ = d["PLAY DIR"].fillna("")
    field = d["HASH"].map({"L": "R", "R": "L"})
    d["to_field"] = np.where(dir_.isin(["L", "R"]) & field.notna(), (dir_ == field).astype(float), np.nan)
    d["to_strength"] = np.where(dir_.isin(["L", "R"]) & d["OFF STR"].isin(["L", "R"]),
                                (dir_ == d["OFF STR"]).astype(float), np.nan)
    d["str_to_field"] = np.where(field.notna() & d["OFF STR"].isin(["L", "R"]),
                                 (d["OFF STR"] == field).astype(float), np.nan)
    return d


def _fs_cell(c: pd.DataFrame) -> str:
    n = len(c)
    if n == 0:
        return f'<div style="{CARD};opacity:.4;font-size:12px">no plays</div>'
    run = 1 - c["PASS"].mean()
    k = round(max(run, 1 - run) * n)
    lead = f"Run {run:.0%}" if run >= 0.5 else f"Pass {1 - run:.0%}"
    bits = []
    for col, yes, no in (("to_field", "to field", "to boundary"), ("to_strength", "to strength", "away from strength")):
        s = c[col].dropna()
        if len(s):
            sh = s.mean()
            kk = round(max(sh, 1 - sh) * len(s))
            bits.append(f"{max(sh, 1 - sh):.0%} {yes if sh >= .5 else no} ({kk}/{len(s)}){badge(kk, len(s))}")
    grp, gsh, _ = top_share(c["CALL GROUP"])
    bits.append(f'<span style="opacity:.7">{esc(grp)} {gsh:.0%}</span>')
    faded = "opacity:.55;" if n < TELL_MIN_N else ""
    return (f'<div style="{CARD};{faded}font-size:12px;line-height:1.5"><div><b style="font-size:14px">{lead}</b>'
            f'{badge(k, n)}<span style="float:right;opacity:.6">{n} plays</span></div>{split_bar(run)}'
            + "".join(f"<div>{b}</div>" for b in bits) + "</div>")


def render_field_strength(d: pd.DataFrame, direction_on: bool):
    x = _fs_flags(snaps(d))
    if x.empty:
        st.info("No plays match the current filters.")
        return
    sided = x["str_to_field"].dropna()
    if len(sided) >= 10:
        k = int(sided.sum()); n = len(sided)
        word, kk = ("the field", k) if k >= n - k else ("the boundary", n - k)
        st.html(f'<div style="font-size:14px;margin-bottom:6px">On a hash, the formation\'s strength is set to '
                f'<b>{word} {kk / n:.0%}</b> of the time ({kk} of {n}){badge(kk, n)}.</div>')
    head = '<div></div>' + "".join(f'<div style="font-size:12px;font-weight:600;text-align:center">{l}</div>'
                                   for _, l in STRENGTHS + [("", "All")])
    body = []
    for h, hl in HASHES:
        body.append(f'<div style="font-size:12px;font-weight:600;align-self:center">{hl}</div>')
        body += [_fs_cell(x[(x["HASH"] == h) & (x["OFF STR"] == s)]) for s, _ in STRENGTHS]
        body.append(_fs_cell(x[x["HASH"] == h]))
    body.append('<div style="font-size:12px;font-weight:600;align-self:center">All</div>')
    body += [_fs_cell(x[x["OFF STR"] == s]) for s, _ in STRENGTHS] + [_fs_cell(x)]
    st.html(LEGEND + '<div style="overflow-x:auto"><div style="display:grid;grid-template-columns:90px '
            'repeat(3,minmax(170px,1fr));gap:6px;min-width:620px">' + head + "".join(body) + "</div></div>")
    note = ("Rows = where the ball was spotted; columns = the formation's strength. ")
    note += ("\"To field / to strength\" use the direction read from the play number (even = right, odd = left) on "
             "runs and run-action passes only." if direction_on else
             "Direction inference is off, so only the run/pass split shows.")
    st.caption(note)

    if direction_on:
        runs = x[(x["PLAY TYPE"] == "Run") & x["PLAY DIR"].notna()]
        if len(runs):
            rows = []
            for name, g in [("All runs", runs)] + [(c, g) for c, g in runs.groupby("CONCEPT")]:
                if len(g) < 3:
                    continue
                ts, tf = g["to_strength"].dropna(), g["to_field"].dropna()
                rows.append({"Run concept": name, "Plays": len(g),
                             "To strength": ts.mean() * 100 if len(ts) else np.nan,
                             "To field": tf.mean() * 100 if len(tf) else np.nan,
                             "Success": g[g["live"]]["success"].mean() * 100,
                             "EPA/play": g[g["live"]]["epa"].mean()})
            show(pd.DataFrame(rows), "Run direction by concept (3+ plays)",
                 "To field counts only plays from a hash.")


# ==============================================================================
# 4. Sequencing
# ==============================================================================
def sequence_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Every snap with the previous snap in the same series (from whole games, before filters)."""
    s = snaps(games).sort_values(["game_id", "SERIES", "SNAP"]).copy()
    g = s.groupby(["game_id", "SERIES"], sort=False)
    s["prev_type"] = g["PLAY TYPE"].shift()
    s["prev_gain"] = g["GN/LS"].shift()
    s["prev_res"] = g["PASS RESULT"].shift()
    s["prev_concept"] = g["CONCEPT"].shift()
    s["prev_explosive"] = g["explosive_play"].shift()
    s["prev_motion"] = g["has_motion"].shift()
    s["first_of_drive"] = g.cumcount() == 0
    return s


SEQUENCE_ROWS = [
    ("First snap of a drive", lambda d: d["first_of_drive"]),
    ("After a run", lambda d: d["prev_type"] == "Run"),
    ("After a pass", lambda d: d["prev_type"] == "Pass"),
    ("After a completion", lambda d: d["prev_res"] == "C"),
    ("After an incompletion", lambda d: d["prev_res"] == "I"),
    ("After a sack", lambda d: d["prev_res"] == "S"),
    ("After a penalty on a pass snap", lambda d: d["prev_res"] == "F"),
    ("After an explosive", lambda d: d["prev_explosive"] == 1),
    ("After a gain of 10+", lambda d: d["prev_gain"] >= 10),
    ("After no gain or a loss", lambda d: d["prev_type"].notna() & (d["prev_gain"] <= 0)),
    ("After a motion snap", lambda d: d["prev_motion"] == 1),
    ("New set of downs (not drive start)", lambda d: (d["DN"] == 1) & ~d["first_of_drive"]),
]


def render_sequencing(d: pd.DataFrame, games: pd.DataFrame):
    """Dowling-style sequencing: what comes next after each kind of snap, then repeats and the opening script."""
    team = team_of(games)
    st.markdown(f"**What {team} calls after...**")
    seq = sequence_frame(games)
    seq = seq[seq.index.isin(d.index)]  # apply the sidebar filters after the previous snap is known
    if len(seq) < 10:
        st.info("Not enough plays match the current filters.")
        return
    base = seq["PASS"].mean()

    # Headline: does the previous play type move the next call?
    a, b = seq[seq["prev_type"] == "Run"], seq[seq["prev_type"] == "Pass"]
    if len(a) >= 10 and len(b) >= 10:
        pa, pb = a["PASS"].mean(), b["PASS"].mean()
        z = _two_prop_z(b["PASS"].sum(), len(b), a["PASS"].sum(), len(a))
        verdict = "a real pattern at this sample size" if abs(z) >= 1.645 else "could still be noise at this sample size"
        if abs(pb - pa) >= 0.05:
            st.html(f'<div style="font-size:14px;margin-bottom:6px">Throws <b>{pb:.0%}</b> after a pass vs '
                    f'<b>{pa:.0%}</b> after a run ({len(b)} and {len(a)} plays): {verdict}.</div>')
        else:
            st.html(f'<div style="font-size:14px;margin-bottom:6px">The previous call barely moves the next one: '
                    f'<b>{pb:.0%}</b> pass after a pass, <b>{pa:.0%}</b> after a run.</div>')

    items = [(name, seq[rule(seq).fillna(False).astype(bool)]) for name, rule in SEQUENCE_ROWS]
    show_split_rows(items, "Previous snap", base_pass=base, top_col="CONCEPT", caption=(
        f'"Usual" = {base:.0%} pass across all these plays. Previous snap = the run or pass right before, in the same '
        f"drive, even if the filters hide it. Faded rows have under {TELL_MIN_N} plays."))

    # Same-concept repeats
    rep = seq[seq["prev_concept"].notna()]
    if len(rep):
        same = (rep["CONCEPT"] == rep["prev_concept"]).mean()
        after_good = rep[rep["prev_gain"] >= 5]
        same_good = (after_good["CONCEPT"] == after_good["prev_concept"]).mean() if len(after_good) else np.nan
        after_bad = rep[rep["prev_gain"] <= 0]
        same_bad = (after_bad["CONCEPT"] == after_bad["prev_concept"]).mean() if len(after_bad) else np.nan
        c = st.columns(3)
        c[0].metric("Back-to-back same concept", f"{same:.0%}", help="Share of snaps that repeat the previous concept.")
        c[1].metric("…after it gained 5+", "–" if pd.isna(same_good) else f"{same_good:.0%}",
                    help=f"{len(after_good)} snaps. Do they go back to what just worked?")
        c[2].metric("…after it gained 0 or less", "–" if pd.isna(same_bad) else f"{same_bad:.0%}",
                    help=f"{len(after_bad)} snaps.")

    render_opening_script(games)


def opening_script_html(games: pd.DataFrame, n_open: int) -> str:
    """Snap-by-snap grid of each game's opening calls: one column per game, each cell colored run/pass."""
    op = snaps(games).sort_values(["date", "PLAY #"])
    op = op.assign(n=op.groupby("game_id").cumcount() + 1)
    op = op[op["n"] <= n_open]
    order = games.drop_duplicates("game_id").sort_values("date")
    labels = list(order["game_label"])
    grid = f"display:grid;grid-template-columns:28px repeat({len(labels)},minmax(150px,1fr));gap:4px"
    cells = ["<div></div>"] + [f'<div style="font-size:11px;font-weight:600;text-align:center">'
                               f'{esc(l.split(" · ", 1)[-1])}</div>' for l in labels]
    for i in range(1, n_open + 1):
        cells.append(f'<div style="font-size:11px;opacity:.6;align-self:center;text-align:right">{i}</div>')
        for gid in order["game_id"]:
            r = op[(op["game_id"] == gid) & (op["n"] == i)]
            if r.empty:
                cells.append('<div></div>')
                continue
            r = r.iloc[0]
            col = RUN_COLOR if r["PLAY TYPE"] == "Run" else PASS_COLOR
            gain = "" if pd.isna(r["GN/LS"]) else f'{int(r["GN/LS"]):+d}'
            if r["interception"]:
                gain = "INT"
            elif r["PASS RESULT"] == "I":
                gain = "Inc"
            elif r["PASS RESULT"] == "F":
                gain = "Penalty"
            elif r["SACK"]:
                gain = f"Sack {gain}"
            sit = f'{DN_LABEL.get(r["DN"], "")}&amp;{"" if pd.isna(r["DIST"]) else int(r["DIST"])}'
            cells.append(
                f'<div title="{esc(r["PERSONNEL"])} {esc(r["OFF FORM"])} · {esc(r["OFF PLAY"])} · {esc(r["CALL GROUP"])}" '
                f'style="border-radius:6px;padding:4px 7px;font-size:11px;line-height:1.3;color:#fff;background:{col}">'
                f'<div style="font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
                f'{esc(r["OFF PLAY"] or r["PLAY TYPE"])}</div>'
                f'<div style="opacity:.85;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{sit} · '
                f'{esc(r["PERSONNEL"])} {esc(r["OFF FORM"])} · {gain}</div></div>')
    return (LEGEND.replace("tell = 70%+ one way and still 70%+ at the low end of a 90% interval · lean = low end 55%+",
                           "hover a box for the full call")
            + f'<div style="overflow-x:auto"><div style="{grid};min-width:{40 + 154 * len(labels)}px">'
            + "".join(cells) + "</div></div>")


def render_opening_script(games: pd.DataFrame):
    st.markdown("**Opening script**")
    n_open = st.slider("First N snaps of each game", 5, 20, 10, key="script_n")
    op = snaps(games).sort_values(["date", "PLAY #"]).groupby("game_id").head(n_open)
    if op.empty:
        return
    p = op["PASS"].mean()
    grp = op["CALL GROUP"].value_counts(normalize=True)
    st.caption(f"Across {op['game_id'].nunique()} games, the first {n_open} snaps were {p:.0%} pass. "
               + " · ".join(f"{k} {v:.0%}" for k, v in grp.items()))
    st.html(opening_script_html(games, n_open))
    st.caption("Uses whole games, so the sidebar filters (other than game) don't apply here.")


# ==============================================================================
# 5. Field position
# ==============================================================================
DRIVE_RESULTS = [("Touchdown", ("Touchdown",), "#1D9E75"), ("Field goal", ("FG Made",), "#BA7517"),
                 ("Punt", ("Punt",), "#888780"), ("Turnover", ("Turnover", "Turnover on Downs"), "#A32D2D"),
                 ("Other", ("FG Missed", "End of Half", "End of Game", "Unknown"), "#C9C7BF")]


def drive_results_html(drives: pd.DataFrame) -> str:
    """One stacked bar per starting zone: how the drives ended, with points per drive at the end."""
    leg = "".join(f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{c};'
                  f'margin-right:4px"></span>{n}</span>' for n, _, c in DRIVE_RESULTS)
    rows = []
    for zone in bd.ZONE_ORDER:
        g = drives[drives["start_zone"] == zone]
        n = len(g)
        if n == 0:
            continue
        segs = []
        for name, res, col in DRIVE_RESULTS:
            k = int(g["result"].isin(res).sum())
            if k:
                segs.append(f'<span title="{name}: {k} of {n}" style="width:{k / n * 100:.1f}%;background:{col};'
                            f'color:#fff;font-size:11px;text-align:center;line-height:22px;overflow:hidden">'
                            f'{k / n:.0%}</span>' if k / n >= .1 else
                            f'<span title="{name}: {k} of {n}" style="width:{k / n * 100:.1f}%;background:{col}"></span>')
        rows.append(f'<div style="display:grid;grid-template-columns:minmax(120px,1fr) minmax(200px,4fr) 150px;'
                    f'gap:10px;align-items:center;padding:5px 0;{GRID_ROW};font-size:13px">'
                    f'<div><b>{zone}</b><div style="font-size:11px;opacity:.6">{n} drives</div></div>'
                    f'<div style="display:flex;gap:2px;height:22px;border-radius:4px;overflow:hidden">{"".join(segs)}</div>'
                    f'<div style="font-size:12px"><b>{g["points"].mean():.1f}</b> pts/drive · '
                    f'{g["yards"].mean():.0f} yds · {g["plays"].mean():.1f} plays</div></div>')
    if not rows:
        return ""
    return (f'<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:12px;margin:2px 0 6px">{leg}</div>'
            f'<div style="overflow-x:auto"><div style="min-width:560px">{"".join(rows)}</div></div>')


def render_field_position(d: pd.DataFrame, drives: pd.DataFrame):
    x = snaps(d)
    if x.empty:
        st.info("No plays match the current filters.")
        return
    cols = st.columns(4)
    for col, (name, label, lo, hi) in zip(cols, bd.FP_ZONES):
        z = x[x["ZONE"] == name]
        with col:
            n = len(z)
            if n == 0:
                st.html(f'<div style="{CARD};opacity:.45"><b>{name}</b><div style="font-size:12px">{label}</div>'
                        f'<div>no plays</div></div>')
                continue
            run = 1 - z["PASS"].mean()
            k = round(max(run, 1 - run) * n)
            live = z[z["live"]]
            items = [("Personnel", "PERSONNEL"), ("Formation", "OFF FORM"), ("Call", "CALL GROUP"),
                     ("Concept", "CONCEPT")]
            tops = "".join(f'<div><span style="opacity:.6">{lab}:</span> {esc(top_share(z[c])[0])} '
                           f'{top_share(z[c])[1]:.0%}</div>' for lab, c in items)
            st.html(f'<div style="{CARD};font-size:12px;line-height:1.5{";opacity:.6" if n < 10 else ""}">'
                    f'<div style="font-size:14px"><b>{name}</b> <span style="opacity:.6">{label}</span></div>'
                    f'<div><b>{"Run" if run >= .5 else "Pass"} {max(run, 1 - run):.0%}</b>{badge(k, n)}'
                    f'<span style="float:right;opacity:.6">{n} plays</span></div>{split_bar(run)}'
                    f'<div>Success {live["success"].mean():.0%} · EPA {live["epa"].mean():+.2f} · '
                    f'Explosive {live["explosive_play"].mean():.0%}</div>{tops}</div>')
    st.caption("Zones by where the ball was snapped. Faded = under 10 plays.")

    if not drives.empty:
        st.markdown("**How drives end, by starting field position**")
        h = drive_results_html(drives)
        if h:
            st.html(h)
        st.caption("Start = first snap of the series (these files don't include the kick that set it up). "
                   "Whole drives in the selected games. Hover a segment for the count.")


# ==============================================================================
# 6. Red zone
# ==============================================================================
def render_red_zone(d: pd.DataFrame, drives: pd.DataFrame):
    trips = drives[drives["red_zone_trip"]] if not drives.empty else drives
    st.markdown("**Red zone trips**")
    if trips.empty:
        st.info("No drives reached the red zone in the selected games.")
    else:
        n = len(trips)
        c = st.columns(5)
        c[0].metric("Trips", n, help="Drives with a snap at or inside the 20.")
        c[1].metric("TD rate", f"{(trips['result'] == 'Touchdown').mean():.0%}")
        c[2].metric("Scoring rate", f"{(trips['points'] > 0).mean():.0%}")
        c[3].metric("Points / trip", f"{trips['points'].mean():.1f}")
        c[4].metric("Trips / game", f"{n / drives['game_id'].nunique():.1f}")
        res = trips["result"].value_counts()
        seg = "".join(f'<span title="{r}: {res[r]}" style="width:{res[r] / n * 100:.1f}%;background:{col}"></span>'
                      for r, col in RZ_RESULTS if r in res)
        leg = "".join(f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
                      f'background:{col};margin-right:4px"></span>{r} {res[r]}</span>' for r, col in RZ_RESULTS if r in res)
        st.html(f'<div style="display:flex;gap:2px;height:14px;border-radius:4px;overflow:hidden;margin:4px 0 6px">{seg}</div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:12px">{leg}</div>')

    rz = snaps(d)
    rz = rz[rz["YARDLINE_100"] <= 20]
    if rz.empty:
        return
    show_split_rows([(name, rz[rz["YARDLINE_100"].between(lo, hi)]) for name, lo, hi in RZ_BANDS], "Zone",
                    "How it's called inside the 20", top_col="CONCEPT",
                    extra=[("TD rate", lambda g: g.loc[g["live"], "touchdown"].mean(), "pct")])
    view = st.radio("Inside the 20 by", ["Personnel", "Formation", "Concept"], horizontal=True, key="rz_by")
    col, top = {"Personnel": ("PERSONNEL", "OFF FORM"), "Formation": ("OFF FORM", "CONCEPT"),
                "Concept": ("CONCEPT", "OFF FORM")}[view]
    show_split_rows(groups(rz, col, limit=12), view, top_col=top,
                    extra=[("TD rate", lambda g: g.loc[g["live"], "touchdown"].mean(), "pct")])
    gl = rz[(rz["goal_to_go"] == 1)].sort_values(["date", "PLAY #"])
    with st.expander(f"Every goal-to-go snap ({len(gl)})"):
        show_plays(gl, "Goal-to-go, oldest first", extra={"TD": gl["touchdown"].map({1: "TD", 0: ""}).values})


# ==============================================================================
# 7. Personnel & motion (new)
# ==============================================================================
def render_personnel_motion(d: pd.DataFrame):
    x = snaps(d)
    if x.empty:
        st.info("No plays match the current filters.")
        return
    pers_order = list(x["PERSONNEL"].fillna("untagged").value_counts().index)
    dn = x["DN"].map(DN_LABEL)
    mix = pd.crosstab(dn, x["PERSONNEL"].fillna("untagged"), normalize="index")[pers_order]
    show_heat(mix, "Personnel usage by down", row_header="Down", total_col=x.groupby(dn).size(),
              caption="Share of each down's snaps in each personnel group (row adds to 100%).")
    show_split_rows(groups(x, "PERSONNEL"), "Personnel", "Personnel groupings", top_col="OFF FORM",
                    extra=[("Motion", lambda g: g["has_motion"].mean(), "pct")])

    pf = pd.crosstab(x["OFF FORM"].fillna("untagged"), x["PERSONNEL"].fillna("untagged"))[pers_order]
    pf = pf.loc[pf.sum(axis=1).sort_values(ascending=False).index]
    show_heat(pf.head(20), "Formations by personnel", kind="int", row_header="Formation",
              total_col=pf.sum(axis=1), caption="Counts (top 20 formations). A formation that only shows up from "
                                                "one personnel group is a substitution tell.")

    st.divider()
    m = x.copy()
    rate = m["has_motion"].mean()
    c = st.columns(3)
    c[0].metric("Motion rate", f"{rate:.0%}")
    c[1].metric("Run rate with motion", f"{1 - m.loc[m['has_motion'] == 1, 'PASS'].mean():.0%}"
                if (m["has_motion"] == 1).any() else "–")
    c[2].metric("Run rate without", f"{1 - m.loc[m['has_motion'] == 0, 'PASS'].mean():.0%}")
    m["MOTION"] = m["MOTION"].fillna("No motion")
    show_split_rows(groups(m, "MOTION", limit=15), "Motion", "What follows each motion", top_col="CONCEPT")
    show_split_rows(groups(m[m["has_motion"] == 1], "MOTION PLAYER"), "Player", "Who motions", top_col="MOTION TYPE")
    c1, c2 = st.columns(2)
    with c1:
        items = [(p, g["has_motion"].mean(), f"of {len(g)}", PASS_COLOR) for p, g in groups(m, "PERSONNEL")]
        show_hbars(items, "Motion rate by personnel", vmax=1)
    with c2:
        items = [(f, g["has_motion"].mean(), f"of {len(g)}", PASS_COLOR) for f, g in groups(m, "OFF FORM", min_n=3)]
        items = sorted(items, key=lambda i: -i[1])[:15]
        show_hbars(items, "Motion rate by formation (3+ plays)", vmax=1)


# ==============================================================================
# 8. Pre-snap tells (new)
# ==============================================================================
LOOK_COMBOS = [("Personnel + formation", ["PERSONNEL", "OFF FORM"]),
               ("Formation + backfield", ["OFF FORM", "BACKFIELD"]),
               ("Personnel + formation + backfield", ["PERSONNEL", "OFF FORM", "BACKFIELD"]),
               ("Formation + motion", ["OFF FORM", "MOTION"]),
               ("Motion", ["MOTION"]),
               ("Backfield", ["BACKFIELD"]),
               ("Personnel + down", ["PERSONNEL", "DN"]),
               ("Formation + hash", ["OFF FORM", "HASH"]),
               ("Formation + down & distance", ["OFF FORM", "DN", "Distance"])]
TELL_OUTCOMES = [("Run/pass", "PLAY TYPE"), ("Call group", "CALL GROUP"), ("Concept", "CONCEPT"),
                 ("Direction", "DIR_VS_STR")]


def tells_table(d: pd.DataFrame, min_n: int, outcomes: list[str], keep_parts: bool = False) -> pd.DataFrame:
    x = snaps(d).copy()
    has = x["PLAY DIR"].isin(["L", "R"]) & x["OFF STR"].isin(["L", "R"])
    x["DIR_VS_STR"] = np.where(has, np.where(x["PLAY DIR"] == x["OFF STR"], "Toward strength", "Away from strength"),
                               None)
    rows = []
    for look_name, cols in LOOK_COMBOS:
        for key, g in x.dropna(subset=cols).groupby(cols):
            n = len(g)
            if n < min_n:
                continue
            key = key if isinstance(key, tuple) else (key,)
            key = tuple(int(k) if isinstance(k, float) and k.is_integer() else k for k in key)
            label = " · ".join(f"{'Dn ' if c == 'DN' else ''}{'Hash ' if c == 'HASH' else ''}{k}"
                               for c, k in zip(cols, key))
            for out_name, oc in TELL_OUTCOMES:
                if out_name not in outcomes:
                    continue
                v, sh, k = top_share(g[oc])
                nn = int(g[oc].notna().sum())
                if v is None or nn < min_n or sh < MIN_TELL_SHARE:
                    continue
                grade, lb = tendency_grade(k, nn)
                if out_name == "Call group" and v == "Run":
                    continue  # same thing the run/pass row says
                rows.append({"Look": label, "Look type": look_name, "Tells you": out_name, "Expect": str(v),
                             "Share": sh * 100, "Of": f"{k}/{nn}", "Grade": BADGE[grade][0], "Floor": lb * 100,
                             "Success": g[g["live"]]["success"].mean() * 100,
                             "_parts": frozenset(zip(cols, key)), "_n": nn})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    # Drop a tell that's just a narrower slice of a broader look with the same answer and nearly the same plays
    # (e.g. "12 · TRIPS BUN SQ · GTB" when "TRIPS BUN SQ · GTB" already says it).
    keep = []
    for i, r in t.iterrows():
        broader = t[(t["Tells you"] == r["Tells you"]) & (t["Expect"] == r["Expect"])
                    & t["_parts"].apply(lambda p: p < r["_parts"]) & (t["_n"] * 0.9 <= r["_n"])]
        keep.append(broader.empty)
    t = t[keep].sort_values(["Floor", "Share"], ascending=False)
    if not keep_parts:
        t = t.drop(columns=["_parts", "_n"])
    return t.reset_index(drop=True)


def render_tells(d: pd.DataFrame, direction_on: bool):
    c1, c2 = st.columns([1, 3])
    min_n = c1.number_input("Minimum plays", 3, 30, 5, key="tell_min")
    opts = [o for o, _ in TELL_OUTCOMES if direction_on or o != "Direction"]
    outs = c2.multiselect("Tells about", opts, default=opts[:2], key="tell_outs")
    t = tells_table(d, int(min_n), outs)
    show(t, "Pre-snap tells, strongest first", height=min(38 * (len(t) + 1) + 4, 700) if len(t) else None,
         caption=("Every look (personnel, formation, backfield, motion, hash, down) where one outcome happens 70%+ "
                  "of the time. Floor = the lowest share consistent with what we've seen (90% confidence); "
                  "sorted by it, so 9-of-10 ranks above 5-of-5."))


# ==============================================================================
# 9. Protection (new)
# ==============================================================================
def render_protection(d: pd.DataFrame):
    p = snaps(d)
    p = p[(p["PLAY TYPE"] == "Pass")]
    tagged = p[p["PROTECTION"].notna()]
    if p.empty:
        st.info("No passes match the current filters.")
        return
    cov = len(tagged) / len(p)
    by_game = p.groupby("game_label")["PROTECTION"].apply(lambda s: s.notna().any())
    untagged = [g.split(" · ", 1)[-1] for g, has in by_game.items() if not has]
    st.caption(f"Protection is tagged on {cov:.0%} of passes in these filters"
               + (f" (no protection tags in: {', '.join(untagged)})." if untagged else "."))
    if tagged.empty:
        return
    # Protections that aren't really protections
    real = tagged[~tagged["PROTECTION"].isin(["RUN", "SCREEN"])]
    rows = []
    for prot, g in real.groupby("PROTECTION"):
        live = g[g["live"]]
        v, sh, _ = top_share(g["CALL GROUP"])
        rows.append({"Protection": prot, "Plays": len(g), "Share": len(g) / len(real) * 100,
                     "Top call": f"{v} ({sh:.0%})", "Sack rate": live["SACK"].mean() * 100,
                     "Comp rate": live["completion"].mean() * 100, "EPA/play": live["epa"].mean(),
                     "3rd down share": (g["DN"] == 3).mean() * 100})
    t = pd.DataFrame(rows).sort_values("Plays", ascending=False)
    show(t.rename(columns={"3rd down share": "On 3rd %"}), "Pass protections (RPO and screen blocking left out)")

    real = real.assign(Situation=real["DN"].map(DN_LABEL) + " & "
                       + real["Distance"].fillna("?").str.replace(r" \(.*\)", "", regex=True))
    prot_order = list(real["PROTECTION"].value_counts().index)
    sit_order = [f"{DN_LABEL[dn]} & {dist.split(' (')[0]}" for dn in (1, 2, 3, 4) for dist in bd.DISTANCE_ORDER]
    ct = pd.crosstab(real["Situation"], real["PROTECTION"], normalize="index")[prot_order]
    ct = ct.reindex([s_ for s_ in sit_order if s_ in ct.index])
    show_heat(ct, "Protection by down & distance", row_header="Situation",
              total_col=real.groupby("Situation").size(), caption="Share of each situation's passes (row adds to 100%).")
    ct2 = pd.crosstab(real["CALL GROUP"], real["PROTECTION"])[prot_order]
    show_heat(ct2, "Protection by call group", kind="int", row_header="Call group",
              total_col=ct2.sum(axis=1),
              caption="Counts. Use with the pressure plan: e.g. which protection shows up on 3rd & long, and whether "
                      "max protection only comes with play-action shots.")


# ==============================================================================
# 10. Call predictor (new)
# ==============================================================================
MATCH_MIN = 8


def match_situation(x: pd.DataFrame, dn=None, bucket=None, zone=None, hash_="Any", pers="Any", form="Any",
                    need: int = MATCH_MIN):
    """
    Snaps like this situation. Conditions are listed most to least important; the least important ones are
    dropped one at a time until at least `need` snaps match. Returns (matches, used names, dropped names).
    """
    filters = [("down", x["DN"] == dn if dn else None),
               ("distance", x["Distance"] == bucket if bucket else None),
               ("personnel", x["PERSONNEL"] == pers if pers not in (None, "Any") else None),
               ("formation", x["OFF FORM"] == form if form not in (None, "Any") else None),
               ("hash", x["HASH"] == hash_ if hash_ not in (None, "Any") else None),
               ("field zone", x["ZONE"] == zone if zone else None)]
    filters = [(n, m) for n, m in filters if m is not None]
    used = list(filters)
    match = x
    while used:
        match = x[np.logical_and.reduce([m.to_numpy() for _, m in used])]
        if len(match) >= need:
            break
        used = used[:-1]
    if not used:
        match = x
    used_names = [n for n, _ in used]
    return match, used_names, [n for n, _ in filters if n not in used_names]
def render_predictor(d: pd.DataFrame):
    x = snaps(d)
    if x.empty:
        st.info("No plays match the current filters.")
        return
    c = st.columns(6)
    dn = c[0].selectbox("Down", [1, 2, 3, 4], key="pr_dn")
    dist = c[1].number_input("Distance", 1, 30, 10, key="pr_dist")
    yl = c[2].slider("Yards to goal", 1, 99, 65, key="pr_yl")
    hash_ = c[3].selectbox("Hash", ["Any", "L", "M", "R"], key="pr_hash")
    pers = c[4].selectbox("Personnel", ["Any"] + sorted(x["PERSONNEL"].dropna().unique()), key="pr_pers")
    form = c[5].selectbox("Formation", ["Any"] + sorted(x["OFF FORM"].dropna().unique()), key="pr_form")

    bucket = bd.DISTANCE_ORDER[np.digitize(dist, [3.5, 6.5, 10.5])]
    zone = bd.zone_of(pd.Series([yl])).iat[0]
    match, used, dropped = match_situation(x, dn, bucket, zone, hash_, pers, form)
    need = MATCH_MIN
    st.markdown(f"**{len(match)} similar snaps** — matched on {', '.join(used) or 'nothing'}"
                + (f"; relaxed {', '.join(dropped)} to get at least {need}" if dropped else ""))
    if match.empty:
        return
    run = 1 - match["PASS"].mean()
    k = round(max(run, 1 - run) * len(match))
    st.html(f'<div style="font-size:15px"><b>{"Run" if run >= .5 else "Pass"} {max(run, 1 - run):.0%}</b>'
            f'{badge(k, len(match))}</div>{split_bar(run, 10)}')
    c1, c2 = st.columns(2)
    for col, oc, title in ((c1, "CALL GROUP", "Call group"), (c2, "CONCEPT", "Concept")):
        tot = match[oc].notna().sum()
        items = [(val, len(g) / tot, f"{len(g)} plays", type_color(g))
                 for val, g in sorted(match.dropna(subset=[oc]).groupby(oc), key=lambda kv: -len(kv[1]))[:8]]
        with col:
            show_hbars(items, title)
    show_plays(match.sort_values("date", ascending=False).head(25), "The matching snaps (newest first)")


# ==============================================================================
# 11. Last meeting vs Wartburg (new)
# ==============================================================================
def profile(g: pd.DataFrame) -> dict:
    """Headline offense numbers for a set of whole games."""
    s = snaps(g)
    live = s[s["live"]]
    return {"Snaps / game": len(s) / max(g["game_id"].nunique(), 1), "Run %": (1 - s["PASS"].mean()) * 100,
            "Success %": live["success"].mean() * 100, "EPA/play": live["epa"].mean(),
            "Explosive %": live["explosive_play"].mean() * 100, "Motion %": s["has_motion"].mean() * 100,
            "3rd down conv %": live.loc[live["DN"] == 3, "converted"].mean() * 100,
            "Points / drive": bd.drive_table(g)["points"].mean()}


THEN_COLOR = "#888780"


def dumbbell_html(then: pd.Series, now: pd.Series, then_label: str, now_label: str, n: int = 8) -> str:
    """Share of each value then vs now: a gray dot, a purple dot, and the line between them."""
    a = then.value_counts(normalize=True)
    b = now.value_counts(normalize=True)
    m = pd.DataFrame({"a": a, "b": b}).fillna(0)
    m = m.loc[m.max(axis=1).sort_values(ascending=False).index].head(n)
    if m.empty:
        return ""
    vmax = max(m.max().max(), 0.05)
    dot = "position:absolute;top:50%;width:10px;height:10px;border-radius:50%;transform:translate(-50%,-50%);" \
          "box-shadow:0 0 0 2px var(--background-color, #fff)"
    rows = []
    for k, r in m.iterrows():
        x1, x2 = r["a"] / vmax * 100, r["b"] / vmax * 100
        lo, hi = min(x1, x2), max(x1, x2)
        chg = (r["b"] - r["a"]) * 100
        rows.append(f'<div title="{esc(k)}: {r["a"]:.0%} {esc(then_label)} → {r["b"]:.0%} {esc(now_label)}" '
                    f'style="display:grid;grid-template-columns:minmax(80px,1fr) minmax(110px,2fr) 44px;gap:8px;'
                    f'align-items:center;padding:4px 0;font-size:12px">'
                    f'<div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{esc(k)}</div>'
                    f'<div style="position:relative;height:14px;margin:0 6px">'
                    f'<span style="position:absolute;top:50%;left:{lo:.1f}%;width:{hi - lo:.1f}%;height:2px;'
                    f'background:rgba(128,128,128,.5);transform:translateY(-50%)"></span>'
                    f'<span style="{dot};left:{x1:.1f}%;background:{THEN_COLOR}"></span>'
                    f'<span style="{dot};left:{x2:.1f}%;background:{PASS_COLOR}"></span></div>'
                    f'<div style="text-align:right;opacity:{1 if abs(chg) >= 5 else .5}">{chg:+.0f}</div></div>')
    leg = (f'<div style="display:flex;gap:12px;font-size:11px;opacity:.8;margin-bottom:4px">'
           f'<span><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:{THEN_COLOR};'
           f'margin-right:4px"></span>{esc(then_label)}</span><span><span style="display:inline-block;width:9px;'
           f'height:9px;border-radius:50%;background:{PASS_COLOR};margin-right:4px"></span>{esc(now_label)}</span>'
           f'<span style="margin-left:auto">pts</span></div>')
    return leg + "".join(rows)


def render_last_meeting(games_all: pd.DataFrame, our_team: str = bd.OUR_TEAM):
    vs = games_all[games_all["opponent"].str.lower() == our_team.lower()]
    if vs.empty:
        st.info(f"No {team_of(games_all)} game against {our_team} in the loaded files. Add last season's "
                f"breakdown to the data folder to compare how they attacked {our_team} with what they do now.")
        return
    last = vs["date"].max()
    meet = vs[vs["date"] == last]
    other = games_all[(games_all["opponent"].str.lower() != our_team.lower()) & (games_all["date"] > last)]
    if other.empty:
        other = games_all[games_all["opponent"].str.lower() != our_team.lower()]
    label_m = meet["game_label"].iat[0]
    label_o = f"Since then ({other['game_id'].nunique()} games)"
    st.markdown(f"**How {team_of(games_all)} attacked {our_team} ({label_m}) vs. what they've done since**")

    pm, po = profile(meet), profile(other)
    fmt = {"Snaps / game": "{:.0f}", "EPA/play": "{:+.2f}", "Points / drive": "{:.1f}"}
    cols = st.columns(4)
    for i, (k, now) in enumerate(po.items()):
        then = pm[k]
        f = fmt.get(k, "{:.0f}%")
        unit = "" if k in fmt else " pts"
        delta = None if pd.isna(now) or pd.isna(then) else f"{(now - then):+.{2 if k == 'EPA/play' else 1 if k == 'Points / drive' else 0}f}{unit} since"
        cols[i % 4].metric(k.replace(" %", ""), "–" if pd.isna(then) else f.format(then), delta, delta_color="off",
                           help=f"vs {our_team}: {'–' if pd.isna(then) else f.format(then)} · "
                                f"since: {'–' if pd.isna(now) else f.format(now)}")
    st.caption(f"Big number = the {our_team} game; the arrow = how much it has moved in the {other['game_id'].nunique()} "
               "games since.")

    c = st.columns(3)
    for col, (title, field) in zip(c, [("Personnel", "PERSONNEL"), ("Call group", "CALL GROUP"),
                                       ("Formation", "OFF FORM")]):
        with col:
            st.markdown(f"**{title} mix**")
            st.html(dumbbell_html(snaps(meet)[field], snaps(other)[field], f"vs {our_team}", "Since"))
    new = set(snaps(other)["CONCEPT"].dropna()) - set(snaps(meet)["CONCEPT"].dropna())
    newf = set(snaps(other)["OFF FORM"].dropna()) - set(snaps(meet)["OFF FORM"].dropna())
    if new or newf:
        st.markdown(f"**New since {our_team}:** concepts {', '.join(sorted(new)) or '—'}; "
                    f"formations {', '.join(sorted(newf)) or '—'}")
        st.caption("Some of this is tagging, not scheme: older files may be broken down with a slightly different "
                   "vocabulary (e.g. bare '34' tagged BOB). Check the play log before calling something new.")
    live = snaps(meet)[snaps(meet)["live"]]
    c1, c2 = st.columns(2)
    with c1:
        show_plays(live.nlargest(8, "epa"), f"What hurt {our_team} most")
    with c2:
        show_plays(live.nsmallest(8, "epa"), f"Where {our_team} won")


# ==============================================================================
# 12. Play log
# ==============================================================================
def render_play_log(d: pd.DataFrame):
    x = d.sort_values(["date", "PLAY #"])
    q = st.text_input("Search (formation, play, concept, motion…)", key="log_q").strip().upper()
    if q:
        hay = x[["OFF FORM", "OFF PLAY", "CONCEPT", "MOTION", "BACKFIELD", "PROTECTION", "ROUTE"]].fillna("").astype(str)
        x = x[hay.apply(lambda r: q in " ".join(r), axis=1)]
    show_plays(x, f"{len(x)} rows (includes no-play rows)")
    st.download_button("Download filtered plays (CSV)", x.to_csv(index=False).encode(), f"{team_of(d).lower().replace(' ', '_')}_plays.csv",
                       "text/csv")


def render_coverage(games: pd.DataFrame):
    s = snaps(games)
    cols = ["PERSONNEL", "OFF FORM", "OFF STR", "BACKFIELD", "MOTION", "OFF PLAY", "CONCEPT", "ROUTE", "PASS RESULT",
            "PROTECTION", "PLAY DIR", "GN/LS"]
    t = s.groupby("game_label")[cols].apply(lambda g: g.notna().mean().mul(100).round(0))
    t.insert(0, "Snaps", s.groupby("game_label").size())
    st.dataframe(t, width="stretch")
    st.caption("% of run/pass snaps with each tag. MOTION blank = no motion; ROUTE, PASS RESULT and PROTECTION only "
               "apply to passes; PLAY DIR only exists for runs and run-action passes.")


# ==============================================================================
# Sideline mode (new): one situation at a time, big text, phone-friendly
# ==============================================================================
SIDELINE_CSS = """
<style>
.sl-banner{border-radius:14px;padding:14px 18px;color:#fff;margin:6px 0 10px}
.sl-big{font-size:clamp(40px,11vw,72px);font-weight:800;line-height:1}
.sl-sub{font-size:15px;opacity:.92;margin-top:6px}
.sl-chip{display:inline-block;border-radius:10px;padding:8px 12px;margin:0 8px 8px 0;font-size:18px;
         border:1px solid rgba(128,128,128,.35)}
.sl-chip b{font-size:22px}
.sl-h{font-size:13px;text-transform:uppercase;letter-spacing:.06em;opacity:.65;margin:12px 0 6px}
.sl-tell{font-size:17px;padding:8px 0;border-top:1px solid rgba(128,128,128,.25)}
.sl-play{font-size:15px;padding:6px 0;border-top:1px solid rgba(128,128,128,.2)}
</style>
"""
PRE_SNAP_LATE = {"BACKFIELD", "MOTION"}  # seen at the line, after personnel and formation are known
DIST_LABEL = {"Short (1-3)": "1-3", "Medium (4-6)": "4-6", "Long (7-10)": "7-10", "Extra Long (11+)": "11+"}


def _sideline_tells(x: pd.DataFrame, sel: dict) -> list[str]:
    """Tells whose look fits what's been picked; backfield/motion parts read as 'if you see…'."""
    t = tells_table(x, TELL_MIN_N, ["Run/pass", "Call group", "Concept", "Direction"], keep_parts=True)
    out = []
    for _, r in t[t["Grade"] != "small sample"].iterrows():
        cond, ok = [], True
        for col, val in r["_parts"]:
            if col in PRE_SNAP_LATE:
                cond.append(f"{'backfield' if col == 'BACKFIELD' else 'motion'} <b>{esc(val)}</b>")
            elif sel.get(col) is None or sel.get(col) != val:
                ok = False
                break
        if not ok:
            continue
        lead = "If you see " + " + ".join(cond) + ": " if cond else "Here: "
        out.append(f"{lead}<b>{esc(r['Expect'])}</b> {r['Share']:.0f}% ({r['Of']}, {r['Grade']})")
        if len(out) >= 5:
            break
    return out


def render_sideline(plays: pd.DataFrame):
    st.html(SIDELINE_CSS)
    latest = int(plays["season"].max())
    this_season = st.toggle(f"{latest} games only", value=True, key="sl_season",
                            help="Off = every loaded game, including earlier seasons.")
    x = snaps(plays[plays["season"] == latest] if this_season else plays)
    if x.empty:
        st.info("No snaps loaded.")
        return

    dn = st.segmented_control("Down", [1, 2, 3, 4], format_func=DN_LABEL.get, key="sl_dn", default=1)
    dist = st.segmented_control("Distance", bd.DISTANCE_ORDER, format_func=DIST_LABEL.get, key="sl_dist",
                                default="Long (7-10)")
    zone = st.segmented_control("Field", bd.ZONE_ORDER, key="sl_zone")
    hash_ = st.segmented_control("Hash", ["L", "M", "R"], key="sl_hash",
                                 format_func={"L": "Left", "M": "Middle", "R": "Right"}.get)
    pers_opts = [p for p in x["PERSONNEL"].value_counts().index]
    pers = st.segmented_control("Personnel", pers_opts, key="sl_pers")
    form_src = x[x["PERSONNEL"] == pers] if pers else x
    form_opts = list(form_src["OFF FORM"].value_counts().index)
    form = st.selectbox("Formation", ["Any"] + form_opts, key="sl_form")
    form = None if form == "Any" else form

    match, used, dropped = match_situation(x, dn, dist, zone, hash_ or "Any", pers or "Any", form or "Any")
    if match.empty:
        st.info("Nothing similar yet.")
        return
    n = len(match)
    run = 1 - match["PASS"].mean()
    k = round(max(run, 1 - run) * n)
    word = grade_word(k, n) or "no clear lean"
    # Gray when it's close to a coin flip, so a 52% "pass" doesn't read like a call.
    color = "#888780" if max(run, 1 - run) < 0.6 else (RUN_COLOR if run >= 0.5 else PASS_COLOR)
    st.html(f'<div class="sl-banner" style="background:{color}">'
            f'<div class="sl-big">{"RUN" if run >= .5 else "PASS"} {max(run, 1 - run):.0%}</div>'
            f'<div class="sl-sub">{k} of {n} similar snaps · {word}</div>'
            f'<div class="sl-sub" style="font-size:13px">Matched on {", ".join(used) or "all snaps"}'
            + (f" · relaxed {', '.join(dropped)}" if dropped else "") + "</div></div>")

    conc = match["CONCEPT"].value_counts().head(4)
    grp = match["CALL GROUP"].value_counts().head(3)
    st.html('<div class="sl-h">Most likely concepts</div>'
            + "".join(f'<span class="sl-chip"><b>{esc(c)}</b> {v / n:.0%}</span>' for c, v in conc.items())
            + '<div class="sl-h">Call group</div>'
            + "".join(f'<span class="sl-chip">{esc(c)} <b>{v / n:.0%}</b></span>' for c, v in grp.items()))

    sel = {"PERSONNEL": pers, "OFF FORM": form, "DN": dn, "Distance": dist, "HASH": hash_}
    tells = _sideline_tells(x, sel)
    if tells:
        st.html('<div class="sl-h">Watch for</div>' + "".join(f'<div class="sl-tell">{t}</div>' for t in tells))

    last = match.sort_values(["date", "PLAY #"], ascending=False).head(5)

    def outcome(r) -> str:
        if r["PASS RESULT"] == "F":
            return "penalty"
        if r["interception"]:
            return "INT"
        return "" if pd.isna(r["GN/LS"]) else f"{int(r['GN/LS']):+d}"

    rows = "".join(
        f'<div class="sl-play">{DN_LABEL.get(r["DN"], "")} &amp; {"" if pd.isna(r["DIST"]) else int(r["DIST"])} · '
        f'{esc(r["PERSONNEL"])} {esc(r["OFF FORM"])} · <b>{esc(r["OFF PLAY"])}</b> · {outcome(r)} '
        f'<span style="opacity:.55">({esc(r["opponent"])})</span></div>' for _, r in last.iterrows())
    st.html(f'<div class="sl-h">Last {len(last)} times</div>{rows}')
