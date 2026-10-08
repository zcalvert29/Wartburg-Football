"""
Load and curate Buena Vista offensive breakdown files.

Each file is one game: row 1 is a title like "02 BUENA VISTA O vs BENEDICTINE 090526"
(game number, home "vs" / away "@", opponent, MMDDYY), row 2 is the header, and every
row after that is one offensive play (or a no-play row).

curate_folder() / curate_files() return a play-level DataFrame with the same core column
names the Dowling app uses (DN, DIST, YARD LN, HASH, OFF FORM, OFF STR, OFF PLAY, GN/LS,
PLAY TYPE, PLAY DIR, success, explosive_play, ep, epa, YARDLINE_100, ...) plus the extra
tags these files carry (PERSONNEL, CONCEPT, CALL GROUP, MOTION, PROTECTION, ...).
drive_table() rolls plays up to one row per series.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

SCOUTED_TEAM = "Buena Vista"
OUR_TEAM = "Wartburg"

RAW_COLUMNS = ["SER", "S#", "DN", "DIST", "FPOS", "HASH - D", "OFF PERS", "FORMATION", "FORM STR",
               "BACKFIELD", "MOTION", "R/P", "PLAY", "PLAY FAMILY", "RPO", "ROUTE TYPE", "PASS RESULT",
               "PROTECTION", "GAIN", "SER END"]

RENAME = {"SER": "SERIES", "S#": "SNAP", "FPOS": "YARD LN", "HASH - D": "HASH", "OFF PERS": "PERSONNEL",
          "FORMATION": "OFF FORM", "FORM STR": "OFF STR", "PLAY": "OFF PLAY", "PLAY FAMILY": "CONCEPT",
          "ROUTE TYPE": "ROUTE", "GAIN": "GN/LS"}

TITLE_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s+O\s+(vs|@)\s+(.+?)\s+(\d{6})\s*$", re.IGNORECASE)

DISTANCE_ORDER = ["Short (1-3)", "Medium (4-6)", "Long (7-10)", "Extra Long (11+)"]

# Field zones by YARDLINE_100 (yards from the opponent's goal), same as the Dowling app.
FP_ZONES = [("Backed up", "Goal to own 20", 80, 99), ("Neutral", "Own 20 to opp 40", 41, 79),
            ("FG range", "Opp 40 to 20", 21, 40), ("Red zone", "Opp 20 to goal", 1, 20)]
ZONE_ORDER = [z[0] for z in FP_ZONES]

# PLAY FAMILY spelling fixes.
CONCEPT_FIXES = {"SPRINT OUT": "SPRINTOUT"}

# Pass families rolled up into how the ball is delivered.
PASS_GROUPS = {"3 STEP": "Quick (3-step)", "5 STEP": "Dropback (5-step)", "PAD": "Play-action",
               "PAP": "Play-action", "NAKED": "Naked / boot", "SPRINTOUT": "Sprint-out", "SCREEN": "Screen",
               "RPO": "RPO"}
PASS_GROUP_ORDER = ["Quick (3-step)", "Dropback (5-step)", "Play-action", "Naked / boot", "Sprint-out",
                    "Screen", "RPO"]

RESULT_NAMES = {"TD": "Touchdown", "FG": "FG Made", "FG MISSED": "FG Missed", "PUNT": "Punt",
                "DOWNS": "Turnover on Downs", "INT": "Turnover", "FUMBLE": "Turnover", "HALF": "End of Half",
                "GAME": "End of Game"}
RESULT_POINTS = {"Touchdown": 7, "FG Made": 3}

XP_EP = 0.9
TD_POINTS = 6.0


# ---------------------------------------------------------------------------
# Expected points (same formula as the Dowling curate_pbp.calculate_ep_epa)
# ---------------------------------------------------------------------------
def _ep_field(yl100):
    return -2 + 9 / (1 + np.exp(0.037 * (yl100 - 50)))


def _down_penalty(down, dist, yl100):
    if down == 1:
        base = 0.0
    elif down == 2:
        base = 0.62
    elif down == 3:
        base = 1.12 + 0.0184 * yl100 - 0.00016 * yl100 ** 2
    else:
        base = 1.62 + 0.0735 * yl100 - 0.00075 * yl100 ** 2
    return base + 0.06 * (dist - 10)


def expected_points(down, dist, yl100) -> float:
    """EP for the offense at down/dist with yl100 yards to the opponent's goal."""
    if pd.isna(down) or pd.isna(dist) or pd.isna(yl100):
        return np.nan
    yl100 = float(np.clip(yl100, 1, 99))
    dist = float(min(max(dist, 1), yl100))
    return float(np.clip(_ep_field(yl100) - _down_penalty(int(down), dist, yl100), -2, 7))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _clean_str(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.replace("\xa0", " ", regex=False).str.replace(r"\s+", " ", regex=True)
    out = out.str.strip().str.upper()
    out = out.mask(out.isin(["", "NAN", "NONE"]))
    return _obj(out)


def _obj(s: pd.Series) -> pd.Series:
    """Plain object dtype with None for missing, so == / isin comparisons give False instead of NA."""
    return s.astype(object).where(s.notna(), None)


def parse_title(title: str) -> dict:
    m = TITLE_RE.match(str(title).replace("\xa0", " "))
    if not m:
        raise ValueError(f"Can't read the game title {title!r} (expected e.g. '02 BUENA VISTA O vs BENEDICTINE 090526')")
    num, _team, loc, opp, mmddyy = m.groups()
    date = pd.to_datetime(mmddyy, format="%m%d%y")
    season = date.year if date.month >= 3 else date.year - 1  # January bowl/playoff games count to the prior fall
    opp = opp.title().replace("Mn", "MN")
    home_away = "Home" if loc.lower() == "vs" else "Away"
    label = f"{season} · {'vs' if home_away == 'Home' else '@'} {opp} ({date.month}/{date.day})"
    return {"game_no": int(num), "opponent": opp, "home_away": home_away, "date": date, "season": season,
            "game_id": f"{date:%Y-%m-%d} {opp}", "game_label": label}


def read_game(source, name: str | None = None) -> pd.DataFrame:
    """Read one breakdown file (path, bytes, or file-like) into raw rows plus game info columns."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    raw = pd.read_excel(source, header=None)
    info = parse_title(raw.iloc[0, 0])
    header = [str(h).replace("\xa0", " ").strip().upper() for h in raw.iloc[1]]
    missing = [c for c in RAW_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{name or 'file'}: missing columns {missing}")
    d = raw.iloc[2:].copy()
    d.columns = header
    d = d[RAW_COLUMNS].dropna(how="all")
    d = d[pd.to_numeric(d["SER"], errors="coerce").notna()]
    for k, v in info.items():
        d[k] = v
    d["source_file"] = name or ""
    return d.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------
def _distance_bucket(dist: pd.Series) -> pd.Series:
    return _obj(pd.cut(dist, [0, 3, 6, 10, 100], labels=DISTANCE_ORDER).astype(object))


def zone_of(yl100: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=yl100.index, dtype="object")
    for name, _, lo, hi in FP_ZONES:
        out[yl100.between(lo, hi)] = name
    return out


def _call_number(play: pd.Series) -> pd.Series:
    """Leading number of the call: '335 POSTS OVER' -> '335', '34 ZONE' -> '34', '100 A SWING' -> '100'."""
    return _obj(play.astype("string").str.extract(r"^(\d+)", expand=False))


def _infer_direction(num: pd.Series, play_type: pd.Series, concept: pd.Series) -> pd.Series:
    """
    Run direction from the hole number: the last digit of a 2-digit run call (34 -> 4) or of a 3-digit
    run-action pass (335 -> 5, 434 -> 4, 135 RPO -> 5, 39 NAKED -> 9). Even = right, odd = left.
    Dropback/quick/screen passes (100/200/300/400/600 series) have no direction.
    """
    num = num.fillna("").astype(str)
    last = num.str[-1:]
    two_digit_run = (num.str.len() == 2) & ((play_type == "Run") | concept.isin(["NAKED", "PAD", "PAP"]))
    run_action = (num.str.len() == 3) & ~num.str[-2:].eq("00") & concept.isin(["PAD", "PAP", "RPO", "NAKED"])
    has = two_digit_run | run_action
    out = pd.Series(None, index=num.index, dtype=object)
    out[has & last.isin(list("02468"))] = "R"
    out[has & last.isin(list("13579"))] = "L"
    return out


def _motion_parts(motion: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'Z IN' -> ('Z', 'IN'); 'HOP RETURN' -> (None, 'HOP RETURN'); 'H JET' -> ('H', 'JET')."""
    m = motion.fillna("").astype(str)
    first = m.str.split(" ").str[0]
    is_player = (first.str.len() == 1) & m.str.contains(" ")
    player = first.where(is_player & m.ne(""), None)
    mtype = m.where(~is_player, m.str.split(" ", n=1).str[1]).where(m.ne(""), None)
    return _obj(player), _obj(mtype)


def curate(raw: pd.DataFrame, infer_direction: bool = True) -> pd.DataFrame:
    d = raw.rename(columns=RENAME).copy()

    num_cols = ["SERIES", "SNAP", "DN", "DIST", "YARD LN", "GN/LS"]
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["HASH", "OFF FORM", "OFF STR", "BACKFIELD", "MOTION", "R/P", "OFF PLAY", "CONCEPT", "RPO", "ROUTE",
              "PASS RESULT", "PROTECTION", "SER END"]:
        d[c] = _clean_str(d[c])

    # Personnel: Excel turns 00 into 0.
    pers = pd.to_numeric(d["PERSONNEL"], errors="coerce")
    d["PERSONNEL"] = _obj(pers.map(lambda x: None if pd.isna(x) else f"{int(x):02d}"))

    d = d.sort_values(["date", "SERIES", "SNAP"], kind="stable").reset_index(drop=True)
    d["PLAY #"] = d.groupby("game_id").cumcount() + 1
    d["WEEK"] = d["date"].rank(method="dense").astype(int)

    d["PLAY TYPE"] = _obj(d["R/P"].map({"R": "Run", "P": "Pass", "N": "No Play"}))
    d["snap"] = d["PLAY TYPE"].isin(["Run", "Pass"])
    d["PASS"] = (d["PLAY TYPE"] == "Pass").astype(float).where(d["snap"])

    # Concept clean-up. 'STEP' is a partial tag: take the family most often used with the same call number.
    d["CONCEPT"] = d["CONCEPT"].replace(CONCEPT_FIXES)
    # Naked boots were tagged SPRINT OUT in the 2025 file; the call name is the reliable signal.
    is_naked = d["OFF PLAY"].fillna("").astype(str).str.contains(r"\bNAKED\b") & (d["R/P"] == "P")
    d.loc[is_naked, "CONCEPT"] = "NAKED"
    d["CALL #"] = _call_number(d["OFF PLAY"])
    passes = d[(d["PLAY TYPE"] == "Pass") & d["CONCEPT"].notna() & (d["CONCEPT"] != "STEP")]
    modal = passes.groupby("CALL #")["CONCEPT"].agg(lambda s: s.mode().iat[0])
    step = d["CONCEPT"] == "STEP"
    d.loc[step, "CONCEPT"] = d.loc[step, "CALL #"].map(modal).fillna("STEP")
    d["CALL GROUP"] = np.where(d["PLAY TYPE"] == "Pass", d["CONCEPT"].map(PASS_GROUPS),
                               np.where(d["PLAY TYPE"] == "Run", "Run", pd.NA))
    d["CALL GROUP"] = _obj(pd.Series(d["CALL GROUP"], index=d.index).where(pd.notna(d["CALL GROUP"]), None))
    d.loc[(d["PLAY TYPE"] == "Pass") & d["CALL GROUP"].isna(), "CALL GROUP"] = "Other pass"
    # A short readable call: concept for runs, concept + route for passes.
    d["CALL"] = np.where(d["PLAY TYPE"] == "Run", d["CONCEPT"],
                         np.where(d["ROUTE"].notna() & (d["ROUTE"] != "SCREEN"),
                                  d["ROUTE"], d["OFF PLAY"].str.replace(r"^\d+\s*", "", regex=True)))
    d["CALL"] = _obj(pd.Series(d["CALL"], index=d.index).where(pd.notna(d["CALL"]), d["OFF PLAY"]))

    d["FORM BASE"] = _obj(d["OFF FORM"].astype("string").str.split(" ").str[0])
    d["MOTION PLAYER"], d["MOTION TYPE"] = _motion_parts(d["MOTION"])
    d["has_motion"] = d["MOTION"].notna().astype(float).where(d["snap"])

    d["PLAY DIR"] = (_infer_direction(d["CALL #"], d["PLAY TYPE"], d["CONCEPT"]) if infer_direction
                     else pd.Series(None, index=d.index, dtype=object))

    # Results
    pr = d["PASS RESULT"]
    d["penalty"] = ((pr == "F") | ((d["PLAY TYPE"] == "No Play") & d["GN/LS"].fillna(0).ne(0))).astype(int)
    d["live"] = d["snap"] & (pr != "F")  # a real snap whose result counts
    d["completion"] = np.select([pr == "C", pr.isin(["I", "X"])], [1.0, 0.0], np.nan)
    d.loc[~d["live"] | (d["PLAY TYPE"] != "Pass"), "completion"] = np.nan
    d["SACK"] = (pr == "S").astype(int)
    d["interception"] = (pr == "X").astype(int)
    d["QB_RUN"] = (pr == "R").astype(int)

    # Series-ending results land on the last live snap of the series.
    d["_last_live"] = False
    last_idx = d[d["live"]].groupby(["game_id", "SERIES"]).tail(1).index
    d.loc[last_idx, "_last_live"] = True
    ser_end = d.groupby(["game_id", "SERIES"])["SER END"].transform(lambda s: s.dropna().iat[-1] if s.notna().any() else pd.NA)
    d["SERIES_END"] = ser_end
    d["touchdown"] = (d["_last_live"] & (ser_end == "TD")).astype(int)
    d["fumble"] = (d["_last_live"] & (ser_end == "FUMBLE")).astype(int)
    d["turnover"] = ((d["interception"] == 1) | (d["fumble"] == 1)).astype(int)

    d["YARDLINE_100"] = np.where(d["YARD LN"] > 0, d["YARD LN"], 100 + d["YARD LN"])
    d["Distance"] = _distance_bucket(d["DIST"])
    d["ZONE"] = zone_of(pd.Series(d["YARDLINE_100"], index=d.index))
    d["goal_to_go"] = (d["DIST"] >= d["YARDLINE_100"]).astype(int)

    gain = d["GN/LS"]
    made_line = (gain >= d["DIST"]) | (d["touchdown"] == 1)
    d["success"] = np.select(
        [d["DN"] == 1, d["DN"] == 2, d["DN"].isin([3, 4])],
        [gain >= 0.4 * d["DIST"], gain >= 0.7 * d["DIST"], made_line], np.nan).astype(float)
    d.loc[d["touchdown"] == 1, "success"] = 1.0
    d.loc[d["turnover"] == 1, "success"] = 0.0
    d.loc[~d["live"] | (gain.isna() & (d["turnover"] == 0)), "success"] = np.nan

    explosive = ((d["PLAY TYPE"] == "Run") | (d["QB_RUN"] == 1)) & (gain >= 10) | \
                ((d["PLAY TYPE"] == "Pass") & (d["QB_RUN"] == 0) & (gain >= 20))
    d["explosive_play"] = explosive.astype(float).where(d["live"])

    d["converted"] = made_line.astype(float).where(d["live"] & d["DN"].isin([3, 4]))
    d["THIRD_DOWN_CONVERTED"] = d["converted"].where(d["DN"] == 3)
    d["FOURTH_DOWN_CONVERTED"] = d["converted"].where(d["DN"] == 4)

    d = _add_ep_epa(d)
    d = d.drop(columns=["_last_live"])
    return d


def _add_ep_epa(d: pd.DataFrame) -> pd.DataFrame:
    """
    EP from down/distance/yard line. EPA within each series, offense-only:
      - touchdown: 6.9 (TD + extra point) - ep
      - interception: opponent 1st & 10 at the line of scrimmage (negated) - ep
      - fumble: opponent 1st & 10 at the spot after the gain (negated) - ep
      - otherwise the next row of the series (no-play rows hold the state after the previous snap), or for the
        last snap the state it left: a new set of downs, the next down, or a turnover on downs. A punt or field
        goal isn't a row in these files, so the last snap before a kick is graded on the 4th down it left.
      - end of half / game: NaN.
    Penalty snaps (PASS RESULT F) get no EPA.
    """
    d = d.copy()
    d["ep"] = [expected_points(a, b, c) for a, b, c in zip(d["DN"], d["DIST"], d["YARDLINE_100"])]
    epa = np.full(len(d), np.nan)
    for _, g in d.groupby(["game_id", "SERIES"], sort=False):
        idx = list(g.index)
        for j, i in enumerate(idx):
            r = d.loc[i]
            if not r["live"] or pd.isna(r["ep"]):
                continue
            ep0 = r["ep"]
            yl, gain = r["YARDLINE_100"], r["GN/LS"]
            if r["touchdown"] == 1:
                epa[i] = TD_POINTS + XP_EP - ep0
            elif r["interception"] == 1:
                epa[i] = -expected_points(1, 10, 100 - yl) - ep0
            elif r["fumble"] == 1:
                spot = yl - (0 if pd.isna(gain) else gain)
                epa[i] = -expected_points(1, 10, 100 - spot) - ep0
            elif j + 1 < len(idx):
                epa[i] = d.loc[idx[j + 1], "ep"] - ep0
            elif r["SERIES_END"] in ("HALF", "GAME") or pd.isna(gain):
                continue
            else:
                new_yl = float(np.clip(yl - gain, 1, 99))
                if gain >= r["DIST"]:
                    epa[i] = expected_points(1, min(10, new_yl), new_yl) - ep0
                elif r["DN"] < 4:
                    epa[i] = expected_points(r["DN"] + 1, r["DIST"] - gain, new_yl) - ep0
                else:
                    epa[i] = -expected_points(1, 10, 100 - new_yl) - ep0
    d["epa"] = epa
    return d


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def curate_files(files: list[tuple[str, object]], infer_direction: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """files = [(name, path-or-bytes), ...]. Returns (plays, errors)."""
    frames, errors = [], []
    for name, src in files:
        try:
            frames.append(read_game(src, name))
        except Exception as e:  # keep going; show the problem in the app
            errors.append(f"{name}: {e}")
    if not frames:
        return pd.DataFrame(), errors
    raw = pd.concat(frames, ignore_index=True)
    if raw.groupby("game_id")["source_file"].nunique().gt(1).any():
        errors.append("The same game appears in more than one file; only the first copy is kept.")
        first = raw.groupby("game_id")["source_file"].first()
        raw = raw[raw["source_file"] == raw["game_id"].map(first)]
    return curate(raw, infer_direction), errors


def curate_folder(folder: str | Path, infer_direction: bool = True) -> tuple[pd.DataFrame, list[str]]:
    paths = sorted(Path(folder).glob("*.xlsx"))
    return curate_files([(p.name, p) for p in paths if not p.name.startswith("~$")], infer_direction)


def drive_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per series."""
    if d.empty:
        return pd.DataFrame()
    g = d.sort_values(["game_id", "SERIES", "SNAP"]).groupby(["game_id", "SERIES"], sort=False)
    out = g.agg(game_label=("game_label", "first"), season=("season", "first"), opponent=("opponent", "first"),
                start=("YARDLINE_100", "first"), deepest=("YARDLINE_100", "min"), plays=("live", "sum"),
                passes=("PASS", "sum"), end=("SERIES_END", "first")).reset_index()
    last_live = d[d["live"]].sort_values(["game_id", "SERIES", "SNAP"]).groupby(["game_id", "SERIES"]).tail(1)
    end_spot = (last_live.set_index(["game_id", "SERIES"])["YARDLINE_100"]
                - last_live.set_index(["game_id", "SERIES"])["GN/LS"].fillna(0))
    out = out.merge(end_spot.rename("end_spot").reset_index(), on=["game_id", "SERIES"], how="left")
    out.loc[out["end"] == "TD", "end_spot"] = 0
    out["yards"] = out["start"] - out["end_spot"]
    out["result"] = out["end"].map(RESULT_NAMES).fillna("Unknown")
    out["points"] = out["result"].map(RESULT_POINTS).fillna(0).astype(int)
    out["start_zone"] = zone_of(out["start"])
    out["red_zone_trip"] = out["deepest"] <= 20
    return out
