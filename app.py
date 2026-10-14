"""
Wartburg football — Scout Opposing Offenses (Buena Vista, Loras, ...).

Run with:  streamlit run app.py
Game files go in ./data (one offense breakdown .xlsx per game, any team), or upload them in the sidebar.
The team comes from each file's title row; pick the team at the top of the sidebar (or ?team=Loras in the URL).
Pages: Scout (every tab), Sideline (one situation at a time, phone-friendly), Scouting report (one-page PDF),
Compare teams (every loaded offense side by side, plus common opponents).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import bv_data as bd
import scout_views as v

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title=f"{bd.OUR_TEAM} · Opponent Scout", page_icon="🏈", layout="wide")


@st.cache_data(show_spinner="Reading game files…")
def load(files: tuple[tuple[str, bytes], ...], infer_direction: bool):
    return bd.curate_files(list(files), infer_direction)


def gather_files(uploads) -> tuple[tuple[str, bytes], ...]:
    files = {p.name: p.read_bytes() for p in sorted(DATA_DIR.glob("*.xlsx")) if not p.name.startswith("~$")}
    for u in uploads or []:
        files[u.name] = u.getvalue()  # an upload with the same name replaces the folder copy
    return tuple(sorted(files.items()))


# ------------------------------------------------------------------------------
# Data (tucked away at the bottom of the sidebar; rarely touched)
# ------------------------------------------------------------------------------
team_box = st.sidebar.container()    # team picker, top of the sidebar on every page
filter_box = st.sidebar.container()  # page filters are drawn here, above the data settings

with st.sidebar:
    st.divider()
    with st.expander("Data & settings"):
        uploads = st.file_uploader("Add game files (.xlsx)", type="xlsx", accept_multiple_files=True,
                                   help="Same layout as the files in /data. Uploads last until the page is "
                                        "closed; drop files in /data to keep them.")
        infer_dir = st.toggle("Read run direction from play number", value=True,
                              help="Even hole = right, odd = left (34 → right, 35 → left). "
                                   "Turn off if their numbering is different.")

files = gather_files(uploads)
if not files:
    st.title("Scout Opposing Offenses")
    st.info(f"No game files yet. Put offense breakdowns in `{DATA_DIR}` or upload them in the sidebar.")
    st.stop()

plays_all, errors = load(files, infer_dir)
for e in errors:
    st.sidebar.warning(e)
if plays_all.empty:
    st.error("None of the files could be read.")
    st.stop()

# ------------------------------------------------------------------------------
# Team (every page works on one scouted team; Compare teams uses them all)
# ------------------------------------------------------------------------------
TEAMS = bd.teams_by_recency(plays_all)  # most recent game first, so the newest scout is the default
# Widgets whose options depend on the team; cleared when the team changes so stale picks don't linger.
TEAM_KEYS = ["f_games", "f_down", "f_dist", "f_zone", "rp_games", "pr_pers", "pr_form", "sl_pers", "sl_form",
             "log_q"]


def on_team_change():
    for k in TEAM_KEYS:
        st.session_state.pop(k, None)
    st.query_params["team"] = st.session_state["team"]


if st.session_state.get("team") not in TEAMS:
    qp = st.query_params.get("team", "")
    st.session_state["team"] = next((t for t in TEAMS if t.lower() == qp.lower()), TEAMS[0])
with team_box:
    counts = plays_all.groupby("team")["game_id"].nunique()
    if len(TEAMS) > 1:
        st.radio("Scouting", TEAMS, key="team", on_change=on_team_change,
                 format_func=lambda t: f"{t} ({counts[t]} games)")
    TEAM = st.session_state["team"]
    if len(TEAMS) == 1:
        st.markdown(f"**Scouting {TEAM}** · {counts[TEAM]} games")
st.query_params["team"] = TEAM
plays = plays_all[plays_all["team"] == TEAM]

# ------------------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------------------
game_order = plays.drop_duplicates("game_id").sort_values("date")
GAME_SHORT = {lab: lab.replace(" · ", " ") for lab in game_order["game_label"]}
DIST_SHORT = {"Short (1-3)": "1-3", "Medium (4-6)": "4-6", "Long (7-10)": "7-10", "Extra Long (11+)": "11+"}
DOWN_SHORT = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
FILTER_KEYS = ["f_games", "f_down", "f_dist", "f_zone"]


def reset_filters():
    for k in FILTER_KEYS:
        st.session_state[k] = []


def last_n_games(n: int = 3):
    st.session_state["f_games"] = list(game_order["game_label"].tail(n))


def scout_page():
    # Filters: games, down, distance, field zone. Nothing picked = everything.
    with filter_box:
        st.header("Filters")
        st.caption("Leave a filter blank to include everything.")
        sel_games = st.pills("Games", list(game_order["game_label"]), selection_mode="multi", key="f_games",
                             format_func=GAME_SHORT.get) or list(game_order["game_label"])
        sel_down = st.pills("Down", [1, 2, 3, 4], selection_mode="multi", key="f_down",
                            format_func=DOWN_SHORT.get) or [1, 2, 3, 4]
        sel_dist = st.pills("Distance", bd.DISTANCE_ORDER, selection_mode="multi", key="f_dist",
                            format_func=DIST_SHORT.get) or bd.DISTANCE_ORDER
        sel_zone = st.pills("Field zone", bd.ZONE_ORDER, selection_mode="multi", key="f_zone") or bd.ZONE_ORDER
        c1, c2 = st.columns(2)
        c1.button("Last 3 games", on_click=last_n_games, width="stretch",
                  help="Weight the scout toward what they're doing now.",
                  disabled=len(game_order) <= 3)
        c2.button("Clear filters", on_click=reset_filters, width="stretch")

    # games = whole selected games (for drives, previous-snap context, scripts)
    games = plays[plays["game_label"].isin(sel_games)]

    def apply(d: pd.DataFrame, use_down: bool = True) -> pd.DataFrame:
        def keep(col, sel, opts):  # untagged rows stay unless the filter is narrowed
            return d[col].isin(sel) | (d[col].isna() if len(sel) == len(opts) else False)
        m = keep("ZONE", sel_zone, bd.ZONE_ORDER)
        if use_down:
            m &= d["DN"].isin(sel_down) & keep("Distance", sel_dist, bd.DISTANCE_ORDER)
        return d[m]

    df = apply(games)
    df_any_down = apply(games, use_down=False)
    drives = bd.drive_table(games)
    filters_on = len(sel_down) < 4 or len(sel_dist) < 4 or len(sel_zone) < 4

    st.title(f"Scout {TEAM} Offense")
    st.caption(f"{bd.OUR_TEAM} football · {len(sel_games)} of {plays['game_id'].nunique()} games selected"
               + (" · situation filters on" if filters_on else ""))

    tabs = st.tabs(["Tendencies", "Tendency tree", "Field & strength", "Sequencing", "Field position", "Red zone",
                    "Personnel & motion", "Pre-snap tells", "Protection", "Call predictor",
                    f"Last meeting vs {bd.OUR_TEAM}", "Play log"])

    with tabs[0]:
        v.render_tendencies(df_any_down, df, drives)
    with tabs[1]:
        v.render_tree(df)
    with tabs[2]:
        v.render_field_strength(df, infer_dir)
    with tabs[3]:
        v.render_sequencing(df, games)
    with tabs[4]:
        v.render_field_position(df, drives)
    with tabs[5]:
        v.render_red_zone(df, drives)
    with tabs[6]:
        v.render_personnel_motion(df)
    with tabs[7]:
        v.render_tells(df, infer_dir)
    with tabs[8]:
        v.render_protection(df)
    with tabs[9]:
        v.render_predictor(df)
    with tabs[10]:
        v.render_last_meeting(plays)  # always every loaded game, so the comparison doesn't vanish with filters
    with tabs[11]:
        v.render_play_log(df)
        with st.expander("Tagging coverage by game"):
            v.render_coverage(games)

    with st.expander("Definitions"):
        st.markdown(
            "- **Snap**: a run or pass call. Pass snaps wiped out by a penalty (PASS RESULT F) count toward what was "
            "called, but not toward success, EPA, yards, or conversions. No-play rows (N) are left out.\n"
            "- **Success**: 40% of the distance on 1st down, 70% on 2nd, all of it on 3rd/4th.\n"
            "- **Explosive**: runs (including QB runs off pass calls) of 10+ yards, passes of 20+.\n"
            "- **EPA**: change in expected points from the snap to the next snap of the drive (same EP curve as the "
            "Dowling app). A touchdown is worth 6.9; an interception or lost fumble hands the opponent the ball at the "
            "spot. Punts and field goals aren't rows in these files, so the last snap before a kick is graded on the "
            "4th down it left.\n"
            "- **Call group**: run, or how the pass is delivered (quick 3-step, 5-step dropback, play-action, naked/boot, "
            "sprint-out, screen, RPO), from PLAY FAMILY.\n"
            "- **Personnel**: 00 = empty; files store it as 0.\n"
            "- **Direction**: read from the hole number (even right, odd left) when the toggle is on.")

@st.cache_data(show_spinner="Building the report…")
def report_pdf(game_ids: tuple[str, ...], infer: bool, file_key: int) -> bytes:
    import report  # reportlab is only needed for this page
    return report.build_report(plays_all[plays_all["game_id"].isin(game_ids)])


def report_page():
    st.title(f"{TEAM} scouting report")
    st.caption("One landscape page for printing. Uses whole games (no down/distance filters).")
    latest = int(game_order["season"].max())
    default = list(game_order.loc[game_order["season"] == latest, "game_label"])
    picked = st.pills("Games in the report", list(game_order["game_label"]), selection_mode="multi",
                      default=default, key="rp_games", format_func=GAME_SHORT.get)
    if not picked:
        st.info("Pick at least one game.")
        return
    ids = tuple(game_order.loc[game_order["game_label"].isin(picked), "game_id"])
    pdf = report_pdf(ids, infer_dir, hash(files))
    st.download_button("Download PDF", pdf, f"{TEAM.replace(' ', '_')}_offense_scouting_report.pdf",
                       "application/pdf", type="primary", icon=":material/download:")
    try:  # preview, if pypdfium2 is installed
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(pdf)
        page = doc[0]
        img = page.render(scale=2).to_pil()
        page.close()
        doc.close()
        st.image(img, width="stretch")
    except ImportError:
        st.caption("Install pypdfium2 to see a preview here.")


def sideline_page():
    st.subheader(f"{TEAM} offense · sideline")
    v.render_sideline(plays)


def compare_page():
    st.title("Compare teams")
    st.caption("Every loaded offense side by side. Ignores the team picker.")
    v.render_compare(plays_all)


pg = st.navigation([st.Page(scout_page, title="Scout", icon=":material/search:", default=True),
                    st.Page(sideline_page, title="Sideline", icon=":material/sports_football:", url_path="sideline"),
                    st.Page(report_page, title="Scouting report", icon=":material/description:", url_path="report"),
                    st.Page(compare_page, title="Compare teams", icon=":material/compare_arrows:",
                            url_path="compare")])
pg.run()
