# Wartburg — Scout Buena Vista Offense

Streamlit app that replicates the Dowling "Scout Opposing Offense" page (minus Run Game / Pass Game)
for Buena Vista's offensive breakdown files, plus new views built on the extra tags these files carry.

## Run
    pip install -r requirements.txt
    streamlit run app.py

Put one breakdown `.xlsx` per game in `data/` (or upload in the sidebar). Files are read by their
title row (`02 BUENA VISTA O vs BENEDICTINE 090526`), not the file name.

## Files
- `bv_data.py`   — reading, cleaning, derived columns (success, explosive, EP/EPA, call groups,
                   run direction, drives). `OUR_TEAM` / `SCOUTED_TEAM` live here.
- `scout_views.py` — one render function per tab.
- `app.py`       — sidebar (data, filters) and tab layout.

## Tabs
Replicated: Tendencies, Tendency tree, Field & strength, Sequencing, Field position, Red zone.
New: Personnel & motion, Pre-snap tells, Protection, Call predictor, Last meeting vs Wartburg, Play log.

## Assumptions (confirm with the client)
- PASS RESULT: C complete, I incomplete, S sack, X interception, R run (scramble / RPO give),
  F penalty on a pass snap. R/P = N is a no-play row.
- Run direction from the hole number: even = right, odd = left (toggle in the sidebar).
- OFF PERS 0 = 00 (empty). Naked boots tagged "SPRINT OUT" in 2025 are re-labeled NAKED;
  a bare "STEP" family takes the most common family for the same call number.
- EPA is offense-only: a TD is worth 6.9; INT/fumble give the opponent the ball at the spot;
  kicks aren't rows, so the last snap before a punt/FG is graded on the 4th down it left.
