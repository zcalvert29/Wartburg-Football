"""
One-page printable scouting report (landscape letter PDF) for the Buena Vista offense.

build_report(games) takes the plays for the games to include (whole games: no down/distance filters) and returns
PDF bytes. Everything is computed from the same helpers the app uses, so the numbers match the screens.
"""
from __future__ import annotations

import io
from datetime import date
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import bv_data as bd
import scout_views as v

RUN = colors.HexColor(v.RUN_COLOR)
PASS = colors.HexColor(v.PASS_COLOR)
INK = colors.HexColor("#222222")
MUTED = colors.HexColor("#6b6b6b")
RULE = colors.HexColor("#cfcfcf")
HEAD_BG = colors.HexColor("#f0efeb")

BASE = ParagraphStyle("base", fontName="Helvetica", fontSize=7.2, leading=8.6, textColor=INK)
SMALL = ParagraphStyle("small", parent=BASE, fontSize=6.4, leading=7.6, textColor=MUTED)
H1 = ParagraphStyle("h1", parent=BASE, fontName="Helvetica-Bold", fontSize=15, leading=17)
H2 = ParagraphStyle("h2", parent=BASE, fontName="Helvetica-Bold", fontSize=8.6, leading=10, spaceBefore=5,
                    spaceAfter=2, textColor=INK)
CELL = ParagraphStyle("cell", parent=BASE, fontSize=6.9, leading=8.1)
CELL_C = ParagraphStyle("cellc", parent=CELL, alignment=TA_CENTER)
KPI_N = ParagraphStyle("kpin", parent=BASE, fontName="Helvetica-Bold", fontSize=12, leading=13, alignment=TA_CENTER)
KPI_L = ParagraphStyle("kpil", parent=SMALL, alignment=TA_CENTER)

PAGE_W, PAGE_H = landscape(letter)
MARGIN = 0.38 * inch
COL_GAP = 0.22 * inch
COL_W = (PAGE_W - 2 * MARGIN - COL_GAP) / 2


def P(text, style=CELL) -> Paragraph:
    return Paragraph(text, style)


def e(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else escape(str(x))


def pct(x) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:.0%}"


def tint(c: colors.Color, strength: float) -> colors.Color:
    """Blend c with white; strength 0 = white, 1 = c."""
    s = float(np.clip(strength, 0, 1))
    return colors.Color(1 - (1 - c.red) * s, 1 - (1 - c.green) * s, 1 - (1 - c.blue) * s)


def grid(rows, widths, header=True, zebra=False, extra=None) -> Table:
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 1.6),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6), ("LEFTPADDING", (0, 0), (-1, -1), 3),
             ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("LINEBELOW", (0, 0), (-1, 0), 0.6, MUTED)]
    t.setStyle(TableStyle(style + (extra or [])))
    return t


def head(*cols) -> list:
    return [P(f"<b>{c}</b>") for c in cols]


# ------------------------------------------------------------------------------
# Sections
# ------------------------------------------------------------------------------
def kpi_strip(s: pd.DataFrame, drives: pd.DataFrame) -> Table:
    live = s[s["live"]]
    third = live[live["DN"] == 3]
    trips = drives[drives["red_zone_trip"]]
    items = [("Snaps / game", f"{len(s) / max(s['game_id'].nunique(), 1):.0f}"),
             ("Run rate", pct(1 - s["PASS"].mean())), ("Success", pct(live["success"].mean())),
             ("EPA / play", f"{live['epa'].mean():+.2f}"), ("Explosive", pct(live["explosive_play"].mean())),
             ("Motion", pct(s["has_motion"].mean())), ("3rd down conv.", pct(third["converted"].mean())),
             ("Points / drive", f"{drives['points'].mean():.1f}" if len(drives) else "–"),
             ("Red zone TD", pct((trips["result"] == "Touchdown").mean()) if len(trips) else "–")]
    w = (PAGE_W - 2 * MARGIN) / len(items)
    t = Table([[P(n, KPI_N) for _, n in items], [P(l, KPI_L) for l, _ in items]], colWidths=[w] * len(items))
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, RULE), ("BACKGROUND", (0, 0), (-1, -1), HEAD_BG),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def dd_grid(s: pd.DataFrame) -> Table:
    widths = [0.42 * inch] + [(COL_W - 0.42 * inch) / 4] * 4
    rows = [[P("")] + [P(f"<b>{l}</b>", CELL_C) for l in ("1-3", "4-6", "7-10", "11+")]]
    bg = []
    for i, dn in enumerate([1, 2, 3, 4], start=1):
        row = [P(f"<b>{v.DN_LABEL[dn]}</b>")]
        for j, dist in enumerate(bd.DISTANCE_ORDER, start=1):
            c = s[(s["DN"] == dn) & (s["Distance"] == dist)]
            n = len(c)
            if n == 0:
                row.append(P("–", CELL_C))
                continue
            run = 1 - c["PASS"].mean()
            share = max(run, 1 - run)
            k = round(share * n)
            word = v.grade_word(k, n)
            grp, gsh, _ = v.top_share(c["CONCEPT"])
            form, fsh, _ = v.top_share(c["OFF FORM"])
            row.append(P(f"<b>{'Run' if run >= .5 else 'Pass'} {share:.0%}</b> ({n}){' · ' + word if word else ''}"
                         f"<br/><font size=6.2 color='#555555'>{e(grp)} {gsh:.0%} · {e(form)}</font>", CELL_C))
            strength = 0.12 + 0.55 * (share - 0.5) * 2 if n >= v.TELL_MIN_N else 0.06
            bg.append(("BACKGROUND", (j, i), (j, i), tint(RUN if run >= .5 else PASS, strength)))
        rows.append(row)
    return grid(rows, widths, extra=bg + [("GRID", (0, 0), (-1, -1), 0.3, RULE)])


def personnel_table(s: pd.DataFrame) -> Table:
    rows = [head("Pers", "Use", "Run", "Top formations", "Top concept")]
    n = len(s)
    for p, g in sorted(s.groupby("PERSONNEL"), key=lambda kv: -len(kv[1])):
        forms = g["OFF FORM"].value_counts(normalize=True).head(2)
        conc, csh, _ = v.top_share(g["CONCEPT"])
        rows.append([P(f"<b>{e(p)}</b>"), P(pct(len(g) / n)), P(pct(1 - g["PASS"].mean())),
                     P(", ".join(f"{e(f)} {sh:.0%}" for f, sh in forms.items())), P(f"{e(conc)} {csh:.0%}")])
    return grid(rows, [0.38 * inch, 0.38 * inch, 0.38 * inch, COL_W - 2.34 * inch, 1.2 * inch])


def formation_table(s: pd.DataFrame, n: int = 7) -> Table:
    rows = [head("Formation", "Plays", "Run", "Pers", "Top concepts")]
    for f, g in sorted(s.groupby("OFF FORM"), key=lambda kv: -len(kv[1]))[:n]:
        pers, psh, _ = v.top_share(g["PERSONNEL"])
        top = g["CONCEPT"].value_counts(normalize=True).head(2)
        rows.append([P(f"<b>{e(f)}</b>"), P(str(len(g))), P(pct(1 - g["PASS"].mean())), P(f"{e(pers)} {psh:.0%}"),
                     P(", ".join(f"{e(c)} {sh:.0%}" for c, sh in top.items()))])
    return grid(rows, [1.0 * inch, 0.4 * inch, 0.38 * inch, 0.6 * inch, COL_W - 2.38 * inch])


def third_down_table(s: pd.DataFrame) -> Table:
    x = s[s["DN"] == 3]
    rows = [head("3rd &", "Plays", "Pass", "Conv.", "Top calls")]
    for dist in bd.DISTANCE_ORDER:
        g = x[x["Distance"] == dist]
        if g.empty:
            continue
        live = g[g["live"]]
        top = g["CALL GROUP"].value_counts(normalize=True).head(2)
        rows.append([P(f"<b>{v.DIST_LABEL[dist]}</b>"), P(str(len(g))), P(pct(g["PASS"].mean())),
                     P(pct(live["converted"].mean())), P(", ".join(f"{e(c)} {sh:.0%}" for c, sh in top.items()))])
    return grid(rows, [0.48 * inch, 0.42 * inch, 0.42 * inch, 0.42 * inch, COL_W - 1.74 * inch])


def tells_list(s: pd.DataFrame, n: int = 9) -> Table:
    t = v.tells_table(s, v.TELL_MIN_N, ["Run/pass", "Call group", "Concept", "Direction"])
    rows = [head("Look", "Expect", "Share", "Grade")]
    for _, r in t.head(n).iterrows():
        rows.append([P(e(r["Look"])), P(f"<b>{e(r['Expect'])}</b>"), P(f"{r['Share']:.0f}% ({r['Of']})"),
                     P(e(r["Grade"]))])
    if len(rows) == 1:
        rows.append([P("No tells at 5+ plays yet."), P(""), P(""), P("")])
    return grid(rows, [COL_W - 2.75 * inch, 1.25 * inch, 0.85 * inch, 0.65 * inch])


def red_zone_block(s: pd.DataFrame, drives: pd.DataFrame) -> list:
    trips = drives[drives["red_zone_trip"]]
    out = []
    if len(trips):
        out.append(P(f"<b>{len(trips)}</b> trips · TD {pct((trips['result'] == 'Touchdown').mean())} · "
                     f"scored {pct((trips['points'] > 0).mean())} · {trips['points'].mean():.1f} pts/trip"))
    rz = s[s["YARDLINE_100"] <= 20]
    rows = [head("Zone", "Plays", "Run", "Top concepts", "Top formation")]
    for name, lo, hi in v.RZ_BANDS:
        b = rz[rz["YARDLINE_100"].between(lo, hi)]
        if b.empty:
            continue
        top = b["CONCEPT"].value_counts(normalize=True).head(2)
        form, fsh, _ = v.top_share(b["OFF FORM"])
        rows.append([P(e(name.split(" (")[0])), P(str(len(b))), P(pct(1 - b["PASS"].mean())),
                     P(", ".join(f"{e(c)} {sh:.0%}" for c, sh in top.items())), P(f"{e(form)} {fsh:.0%}")])
    if len(rows) > 1:
        out.append(grid(rows, [0.85 * inch, 0.4 * inch, 0.38 * inch, COL_W - 2.93 * inch, 1.3 * inch]))
    return out


def protection_block(s: pd.DataFrame) -> list:
    p = s[(s["PLAY TYPE"] == "Pass") & s["PROTECTION"].notna() & ~s["PROTECTION"].isin(["RUN", "SCREEN"])]
    if p.empty:
        return [P("No protection tags in these games.", SMALL)]
    overall = p["PROTECTION"].value_counts(normalize=True).head(4)
    third = p[p["DN"] == 3]["PROTECTION"].value_counts(normalize=True).head(3)
    sacks = p[p["SACK"] == 1]["PROTECTION"].value_counts()
    lines = [f"<b>All passes:</b> " + ", ".join(f"{e(k)} {sh:.0%}" for k, sh in overall.items()),
             f"<b>3rd down:</b> " + (", ".join(f"{e(k)} {sh:.0%}" for k, sh in third.items()) or "–")]
    by_grp = p.groupby("CALL GROUP")["PROTECTION"].agg(lambda x: f"{x.mode().iat[0]} {(x == x.mode().iat[0]).mean():.0%}")
    lines.append("<b>By call:</b> " + "; ".join(f"{e(g)} - {e(val)}" for g, val in by_grp.items()))
    if len(sacks):
        lines.append("<b>Sacks:</b> " + ", ".join(f"{e(k)} {n}" for k, n in sacks.items()))
    return [P(l) for l in lines]


def motion_block(s: pd.DataFrame) -> list:
    m = s[s["has_motion"] == 1]
    if m.empty:
        return [P("No motion in these games.", SMALL)]
    rows = [head("Motion", "Plays", "Run", "Then", "Top formation")]
    for mo, g in sorted(m.groupby("MOTION"), key=lambda kv: -len(kv[1]))[:5]:
        conc, csh, _ = v.top_share(g["CONCEPT"])
        form, fsh, _ = v.top_share(g["OFF FORM"])
        rows.append([P(f"<b>{e(mo)}</b>"), P(str(len(g))), P(pct(1 - g["PASS"].mean())), P(f"{e(conc)} {csh:.0%}"),
                     P(f"{e(form)} {fsh:.0%}")])
    return [P(f"Motion on {pct(s['has_motion'].mean())} of snaps; run {pct(1 - m['PASS'].mean())} with motion vs "
              f"{pct(1 - s.loc[s['has_motion'] == 0, 'PASS'].mean())} without.", SMALL),
            grid(rows, [0.85 * inch, 0.4 * inch, 0.38 * inch, 1.15 * inch, COL_W - 2.78 * inch])]


def explosive_block(s: pd.DataFrame, n: int = 5) -> Table:
    live = s[s["live"]].nlargest(n, "epa")
    rows = [head("Situation", "Look", "Play", "Gain")]
    for _, r in live.iterrows():
        dist = "" if pd.isna(r["DIST"]) else int(r["DIST"])
        gain = "INT" if r["interception"] else ("" if pd.isna(r["GN/LS"]) else f"{int(r['GN/LS']):+d}")
        rows.append([P(f"{v.DN_LABEL.get(r['DN'], '')} &amp; {dist}"), P(f"{e(r['PERSONNEL'])} {e(r['OFF FORM'])}"),
                     P(f"<b>{e(r['OFF PLAY'])}</b>"), P(gain)])
    return grid(rows, [0.6 * inch, 1.45 * inch, COL_W - 2.5 * inch, 0.45 * inch])


def opening_block(games: pd.DataFrame, n: int = 10) -> list:
    s = v.snaps(games).sort_values(["date", "PLAY #"])
    first = s.groupby("game_id").head(1)
    op = s.groupby("game_id").head(n)
    grp = op["CALL GROUP"].value_counts(normalize=True).head(3)
    return [P(f"<b>First {n} snaps:</b> pass {pct(op['PASS'].mean())} · " +
              ", ".join(f"{e(g)} {sh:.0%}" for g, sh in grp.items())),
            P("<b>First snap:</b> " + "; ".join(f"{e(r['opponent'])} - {e(r['PERSONNEL'])} {e(r['OFF FORM'])} "
                                                  f"{e(r['OFF PLAY'])}" for _, r in first.iterrows()))]


# ------------------------------------------------------------------------------
# Assemble
# ------------------------------------------------------------------------------
def build_report(games: pd.DataFrame, our_team: str = bd.OUR_TEAM, scouted: str = bd.SCOUTED_TEAM) -> bytes:
    s = v.snaps(games)
    drives = bd.drive_table(games)
    order = games.drop_duplicates("game_id").sort_values("date")
    game_line = ", ".join(f"{'vs' if r.home_away == 'Home' else '@'} {r.opponent} {r.date:%m/%d/%y}"
                          for r in order.itertuples())

    left = [P("Down &amp; distance (run/pass lean, top concept, top formation)", H2), dd_grid(s),
            P("Personnel", H2), personnel_table(s),
            P("Formations", H2), formation_table(s),
            P("3rd down", H2), third_down_table(s),
            P("Red zone", H2), *red_zone_block(s, drives)]
    right = [P("Pre-snap tells (strongest first)", H2), tells_list(s),
             P("Motion", H2), *motion_block(s),
             P("Protection", H2), *protection_block(s),
             P("Opening script", H2), *opening_block(games),
             P("Plays that hurt most (by EPA)", H2), explosive_block(s)]

    body = Table([[left, right]], colWidths=[COL_W + COL_GAP / 2, COL_W + COL_GAP / 2])
    body.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (0, 0), COL_GAP / 2), ("LEFTPADDING", (1, 0), (1, 0), COL_GAP / 2),
                              ("TOPPADDING", (0, 0), (-1, -1), 0)]))

    story = [Table([[P(f"{e(scouted)} Offense - Scouting Report", H1),
                     P(f"{e(our_team)} Football · prepared {date.today():%m/%d/%Y}",
                       ParagraphStyle("r", parent=SMALL, alignment=2))]],
                   colWidths=[PAGE_W * 0.6 - MARGIN, PAGE_W * 0.4 - MARGIN],
                   style=[("VALIGN", (0, 0), (-1, -1), "BOTTOM"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                          ("RIGHTPADDING", (0, 0), (-1, -1), 0)]),
             P(f"{len(order)} games: {e(game_line)} · {len(s)} snaps", SMALL), Spacer(1, 4),
             kpi_strip(s, drives), Spacer(1, 2), body, Spacer(1, 3),
             P("Tell = 70%+ one way and still 70%+ at the low end of a 90% interval; lean = low end 55%+. "
               "Success: 40% of the distance on 1st, 70% on 2nd, all of it on 3rd/4th. Explosive: runs 10+, passes 20+. "
               "Direction is read from the hole number (even right, odd left).", SMALL)]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter), leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN,
                            title=f"{scouted} Offense Scouting Report", author=f"{our_team} Football")
    doc.build(story)
    return buf.getvalue()
