# Wartburg — Scout Opposing Offenses

Streamlit app that replicates the Dowling "Scout Opposing Offense" page (minus Run Game / Pass Game)
for opponents' offensive breakdown files (Buena Vista, Loras, Dubuque, Simpson, ...), plus new views built on the extra tags
these files carry.

## Run
    pip install -r requirements.txt
    streamlit run app.py

Put one breakdown `.xlsx` or `.xls` per game in `data/` (or upload in the sidebar). Files are read by their
title row (`02 BUENA VISTA O vs BENEDICTINE 090526`, `05 LORAS O @ COE 100326`), not the file name, so a new
opponent needs no code changes: drop its files in and it shows up in the team picker. Files with no title
row (the Dubuque `.xls` exports) take the title from the sheet name (`04 DUBUQUE O vs MCMURRY 092626`).

Pick the team at the top of the sidebar. The choice is kept in the URL (`?team=Loras`), so a link opens
straight to that team's scout.

## Files
- `bv_data.py`   — reading, cleaning, derived columns (success, explosive, EP/EPA, call groups,
                   run direction, drives). `OUR_TEAM` / `SCOUTED_TEAM` live here.
- `scout_views.py` — one render function per tab.
- `app.py`       — sidebar (team, data, filters), pages and tab layout.

## Tabs
Replicated: Tendencies, Tendency tree, Field & strength, Sequencing, Field position, Red zone.
New: Personnel & motion, Pre-snap tells, Protection, Call predictor, Last meeting vs Wartburg, Play log.

Pages: Scout, Sideline, Scouting report (PDF).

## Assumptions (confirm with the client)
- PASS RESULT: C complete, I incomplete, S sack, X interception, R run (scramble / RPO give),
  F penalty on a pass snap. R/P = N is a no-play row.
- Run direction from the hole number: even = right, odd = left (toggle in the sidebar).
- OFF PERS 0 = 00 (empty). Naked boots tagged "SPRINT OUT" in 2025 are re-labeled NAKED;
  a bare "STEP" family takes the most common family for the same call number (within the same team).
- A blank R/P is filled from the row: PLAY "N" = no-play, a pass family or pass result = pass, a run
  family = run, nothing tagged = no-play. PLAY FAMILY "OW" is read as POW, "SPINT OUT" as SPRINTOUT. Dubuque's DASH calls count as Sprint-out (by the call name, since one file tags them PAR).
- Excel cuts sheet names at 31 characters; titles that lost their date are completed in `TITLE_FIXES`
  (`bv_data.py`). Opponent display names that differ from the file (Stout → UW-Stout, Platteville → UW-Platteville) are in `TEAM_NAMES`.
- EPA is offense-only: a TD is worth 6.9; INT/fumble give the opponent the ball at the spot;
  kicks aren't rows, so the last snap before a punt/FG is graded on the 4th down it left.
