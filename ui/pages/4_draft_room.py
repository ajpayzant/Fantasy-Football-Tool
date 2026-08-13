"""Draft Room: run a mock draft, pick by pick, with the engine's reasoning exposed.

The layout follows the order a drafter actually reads, which is why the board comes
first and the advice last:

* **The board** — what has gone, in three views: the pick list, the wall-chart grid
  where the snake is visible as a shape, and every team's roster construction.
* **Best players left** — the shortlist, so a pick can be made without consulting the
  model at all.
* **Who picks between now and my next turn**, and what they are likely to take.
* **Who will still be there when I pick again** — survival odds, simulated against
  those specific managers rather than against ADP.
* **What to take now**, through several lenses, each with its own justification.

Every recommendation is advisory. You can draft anyone at any time, undo, or hand
the pick to the engine.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from core.enums import Position
from core.validation import ConfigurationError
from engine.draft_state import DraftState
from engine.recommender import recommend_for
from engine.simulator import (
    DraftSimulator,
    likely_next_picks,
    upcoming_position_pressure,
)
from models.database import session_scope
from models.draft import MockDraftResult
from services import draft_session
from services.repository import save_mock_draft
from ui import board_views, components, state

LOGGER = logging.getLogger("fantasy_mock_draft.ui.draft_room")

_K_RESUME_IDS = "resume_draft_ids"
"""Session key holding ``(league_id, source_id)`` for the autosave.

Kept in session rather than looked up per render: they are written once when the
draft starts, and re-deriving them would mean a database query on every rerun.
"""

components.page_header(
    "🎯 Draft Room",
    "A live mock draft against the modelled opponents.",
)

# ─────────────────────────────────────────────────────────────────────────────
# Recovering from a refresh
#
# This runs before ``require()`` on purpose. Session state is gone after a
# refresh, so the gate would otherwise send the user to Setup to fetch data again
# — and by the time they had, the saved draft would be stranded in the database
# with nothing able to reach it. Reloading the league and board the snapshot names
# is what turns "recoverable" into "the refresh did not happen".
# ─────────────────────────────────────────────────────────────────────────────
_snapshot = draft_session.load_snapshot()
if (
    _snapshot is not None
    and not _snapshot.is_empty
    and state.draft() is None
    and (state.league() is None or state.pool() is None)
):
    _revived = draft_session.rehydrate(_snapshot)
    if _revived.ok:
        state.set_league(_revived.league, source="restored with the saved draft")
        state.set_pool(_revived.pool, source="restored with the saved draft")
        LOGGER.info("Reloaded the league and board for a saved draft")
    elif _revived.reason:
        st.warning(
            f"There is a saved draft ({_snapshot.label()}) but it cannot be "
            f"reopened: {_revived.reason}. Load a league and board on **Setup** "
            "first.",
            icon="🗂️",
        )

components.require()

league = state.league()
pool = state.pool()
profiles = state.profiles()
settings = state.settings()

if not profiles:
    st.warning(
        "No manager profiles are built, so the opponents will be simulated on the "
        "league-average prior. Build them on **Manager Profiles** for opponents that "
        "reflect your league.",
        icon="⚠️",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Start / restart
# ─────────────────────────────────────────────────────────────────────────────
draft = state.draft()

if draft is None and draft_session.resumable(_snapshot, league, pool):
    # Offered rather than restored automatically. Replaying picks changes what the
    # page shows completely, and a user who deliberately abandoned a draft and came
    # back to start a fresh one should not find the old one waiting for them.
    with st.container(border=True):
        st.subheader("🗂️ Pick up where you left off")
        st.write(
            f"**{_snapshot.label()}** — {draft_session.describe_age(_snapshot)}."
        )
        st.caption(
            "Saved automatically after every pick. Restoring replays those picks "
            "through the same code that made them, so the rosters, the clock and "
            "undo all come back — it is the draft, not a summary of it."
        )
        resume_columns = st.columns([1, 1, 2])
        if resume_columns[0].button(
            "Resume draft", type="primary", key="resume_draft", width="stretch"
        ):
            outcome = draft_session.restore(_snapshot, league, pool, settings)
            state.set_draft(outcome.draft)
            # ``set_draft`` runs after the restore, and the league/pool writes above
            # already cleared the snapshot in the rehydrate case, so it is written
            # back rather than assumed to still be there.
            draft_session.save_snapshot(
                draft_session.with_sources(
                    draft_session.snapshot_from_draft(outcome.draft),
                    league_id=_snapshot.league_id, source_id=_snapshot.source_id,
                )
            )
            for message in outcome.warnings:
                components.flash(message, "warning")
            if outcome.is_exact:
                components.flash(
                    f"Restored {outcome.replayed} pick(s). You are back on the clock "
                    "where you left off.",
                )
            st.rerun()
        if resume_columns[1].button(
            "Discard it", key="discard_draft", width="stretch"
        ):
            draft_session.clear_snapshot()
            components.flash("Saved draft discarded.", "info")
            st.rerun()
        with st.expander(f"The {_snapshot.pick_count} saved pick(s)"):
            st.dataframe(
                pd.DataFrame(draft_session.snapshot_frame_rows(_snapshot)),
                width="stretch", hide_index=True, height=260,
            )
    st.divider()

if draft is None:
    st.subheader("Start a draft")
    start_columns = st.columns([1, 1, 1, 2])
    user_slot = start_columns[0].number_input(
        "Your draft slot", min_value=1, max_value=league.config.team_count,
        value=int(league.config.user_draft_slot or 1),
    )
    seed = start_columns[1].number_input(
        "Random seed", min_value=0, max_value=2**31 - 1, value=20260801,
        help="Same seed, same opponent behaviour — so you can retry a decision "
             "against an identical room.",
    )
    if start_columns[2].button("Start draft", type="primary", width="stretch"):
        league.set_user_slot(int(user_slot))
        new_draft = DraftState(league, pool, settings=settings, seed=int(seed))
        state.set_draft(new_draft)
        # Saved here, once, so a resume has a league and a board to come back to.
        # Doing it per pick would re-save several hundred player rows every click.
        league_id, source_id = draft_session.persist_inputs(
            league, pool, is_sample=state.is_sample_data()
        )
        st.session_state[_K_RESUME_IDS] = (league_id, source_id)
        LOGGER.info(
            "Draft started: slot %d, seed %d, %d picks",
            user_slot, seed, league.config.team_count * league.config.rounds,
        )
        st.rerun()
    st.caption(
        f"{league.config.team_count} teams × {league.config.rounds} rounds = "
        f"{league.config.team_count * league.config.rounds} picks, "
        f"{league.config.draft_type} order."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Autosave
#
# Once per render, not at each of the six places a pick can be made. Every one of
# those ends in ``st.rerun()``, so the render that follows sees the mutation and
# saves it: one write per interaction, and a pick path added later cannot forget.
# ─────────────────────────────────────────────────────────────────────────────
_autosave_enabled = not state.is_sample_data()
"""Sample data is never written to the database.

Resuming needs the board stored, and storing a fictional board would put it in the
"reload a saved league" list on Setup as though it were real — the one thing the
sample-data rules exist to prevent. A synthetic draft is still fully playable; it
just does not survive a refresh.
"""

if _autosave_enabled:
    _league_id, _source_id = st.session_state.get(_K_RESUME_IDS) or (
        league.config.league_id,
        _snapshot.source_id if _snapshot is not None else None,
    )
    if _source_id is None:
        # A draft resumed in a fresh session, or one started before this feature
        # existed, has no stored board id yet. Store one now so the *next* refresh
        # can reload it — otherwise the autosave keeps writing picks that point at
        # nothing.
        _league_id, _source_id = draft_session.persist_inputs(
            league, pool, is_sample=False
        )
        st.session_state[_K_RESUME_IDS] = (_league_id, _source_id)
    draft_session.autosave(draft, league_id=_league_id, source_id=_source_id)

# ─────────────────────────────────────────────────────────────────────────────
# Status bar
# ─────────────────────────────────────────────────────────────────────────────
# The league's user slots, not the slot on the clock: this page is written from the
# perspective of the seat the user is drafting from, which stays fixed while the
# clock moves around the room.
user_slot = min(league.user_slots) if league.user_slots else 1
on_clock_slot = draft.on_the_clock_slot
_on_clock = draft.manager_on_clock()
on_clock_manager = _on_clock.name if _on_clock is not None else "—"

status_columns = st.columns([3, 1, 1, 1])
with status_columns[0]:
    if draft.is_complete:
        st.success(
            f"**Draft complete** — {draft.pick_index} picks made. "
            "Review it on **Analysis**."
        )
    elif draft.is_user_on_clock:
        st.success(f"**You are on the clock** — pick {draft.summary_line()}")
    else:
        st.info(f"**{on_clock_manager}** is on the clock — {draft.summary_line()}")

until_turn = draft.picks_until_turn(user_slot)
status_columns[1].metric(
    "Picks until your turn", "now" if not until_turn else until_turn
)
status_columns[2].metric("Available", draft.available_count())
status_columns[3].metric(
    "Your picks", len(draft.picks_by_slot(user_slot)),
    help=f"Upcoming: {', '.join(str(n) for n in draft.next_pick_numbers(user_slot, 3))}",
)

control_columns = st.columns([1, 1, 1, 1, 2])
if control_columns[0].button(
    "Advance one pick", disabled=draft.is_complete or draft.is_user_on_clock,
    width="stretch",
):
    # The simulator mutates the DraftState it is given, which is the object in session
    # state — so the advance is persisted by the rerun without anything else to do.
    simulated = DraftSimulator(draft, profiles).simulate_pick()
    if simulated is None:
        st.error("The simulator could not make a pick. The board may be empty.")
    else:
        LOGGER.info(
            "Simulated pick %s: %s", simulated.pick.overall_pick, simulated.pick.player_name
        )
        st.rerun()

if control_columns[1].button(
    "Advance to my turn", disabled=draft.is_complete or draft.is_user_on_clock,
    width="stretch",
):
    made = DraftSimulator(draft, profiles).simulate_until_user()
    LOGGER.info("Advanced %d simulated picks", len(made))
    st.rerun()

if control_columns[2].button(
    "Undo", disabled=not draft.can_undo, width="stretch"
):
    draft.undo()
    st.rerun()

if control_columns[3].button(
    "Redo", disabled=not draft.can_redo, width="stretch"
):
    draft.redo()
    st.rerun()

if control_columns[4].button("Abandon this draft", width="stretch"):
    state.set_draft(None)
    draft_session.clear_snapshot()
    st.session_state.pop(_K_RESUME_IDS, None)
    components.flash("Draft abandoned.", "info")
    st.rerun()

if _autosave_enabled:
    st.caption(
        "💾 Saved automatically after every pick — a refresh, a closed tab or a "
        "sleeping laptop will not lose this draft. *Abandon* is the only thing that "
        "deletes it."
    )
else:
    st.caption(
        "⚠️ This draft is **not** being saved, because the loaded board is flagged as "
        "sample data and nothing fictional is written to your database. A refresh "
        "will lose it."
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# The board
#
# First on the page, above the advice. A drafter on the clock looks at the board and
# then at the suggestions, in that order, and the suggestions only mean anything
# against what is already gone. Three views because they answer different questions:
# the list is a log, the grid is where-are-we, the rosters are what-is-everyone-building.
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("The board")
order_tab, grid_tab, rosters_tab, runs_tab = st.tabs(
    ["📋 Draft order", "🗓️ Board grid", "👥 Team rosters", "📈 Runs and scarcity"]
)

with order_tab:
    board_views.render_draft_order(draft)

with grid_tab:
    board_views.render_snake_grid(draft, league, user_slot=user_slot)

with rosters_tab:
    board_views.render_team_rosters(draft, league, pool, user_slot=user_slot)

with runs_tab:
    st.caption(
        "A run is several managers taking the same position in quick succession. The "
        "engine's opponents chase runs at their own estimated rate, so a run in "
        "progress genuinely raises the chance the next pick is that position too."
    )
    snapshot = draft.run_snapshot()
    for window, counts in snapshot.counts_by_window.items():
        if counts:
            components.position_bar_chart(
                {p: n for p, n in counts.items()},
                f"Positions taken in the last {window} picks",
            )
    gaps = {
        str(position): ("never" if gap is None else gap)
        for position, gap in snapshot.picks_since_position.items()
    }
    st.markdown("**Picks since each position last went**")
    st.dataframe(
        pd.DataFrame(sorted(gaps.items()), columns=["Position", "Picks ago"]),
        width="stretch", hide_index=True,
    )
    st.markdown("**Drafted so far, by position**")
    components.position_bar_chart(
        dict(draft.position_counts_drafted()), "Total drafted"
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# The shortlist
#
# Above the recommendations on purpose. The lenses answer "what should I take"; this
# answers "what is even there", which is the question a drafter asks first and the one
# that lets them disagree with the model from an informed position rather than a blind
# one. A pick can be made straight from here without consulting any lens.
# ─────────────────────────────────────────────────────────────────────────────
if not draft.is_complete:
    _board = state.user_board()
    st.subheader("Best players left")
    shortlist_columns = st.columns([1, 1, 1, 2])
    top_count = shortlist_columns[0].number_input(
        "How many", min_value=5, max_value=100, value=25, step=5,
        key="top_remaining_count",
    )
    top_position = shortlist_columns[1].selectbox(
        "Position", ["All"] + [str(p) for p in Position], key="top_remaining_position",
    )
    top_order = shortlist_columns[2].selectbox(
        "Order by",
        ["Board order", "My board", "Projection", "Value over replacement"],
        key="top_remaining_order",
        help="`Board order` is the blended consensus ranking. `My board` applies your "
             "target list and your own rankings from **Player Pool** — the gap between "
             "the two is where you are deliberately off consensus.",
    )
    top_frame = board_views.top_remaining_frame(
        draft, _board,
        count=int(top_count),
        position=None if top_position == "All" else Position.coerce(top_position),
        order=top_order,
    )
    if top_frame.empty:
        st.caption("Nobody left at that position.")
    else:
        st.dataframe(
            top_frame.drop(columns=["player_id"]),
            width="stretch", hide_index=True,
            height=min(640, 60 + 35 * len(top_frame)),
            column_config={
                "Mine": st.column_config.TextColumn(
                    "Mine",
                    help="🎯 N = your target list, in your order. ⛔ = never draft.",
                ),
                "VOR": st.column_config.NumberColumn(
                    "VOR",
                    help="Value over replacement: projected points above the last "
                         "startable player at this position in your league. It is what "
                         "makes 250 points at tight end worth more than 250 at running "
                         "back.",
                ),
            },
        )
        pick_columns = st.columns([3, 1])
        shortlist_ids = list(top_frame["player_id"])
        shortlist_names = dict(zip(top_frame["player_id"], top_frame["Player"]))
        quick_id = pick_columns[0].selectbox(
            "Draft from this list", shortlist_ids,
            format_func=lambda pid: shortlist_names.get(pid, pid),
            key="top_remaining_pick",
        )
        if pick_columns[1].button(
            "Draft", key="top_remaining_draft", width="stretch",
            disabled=not draft.is_user_on_clock,
        ):
            try:
                made = draft.make_pick(
                    quick_id, explanation="Taken from the best-players-left list."
                )
            except ConfigurationError as error:
                st.error(f"That pick was rejected: {error}")
            else:
                LOGGER.info(
                    "Shortlist pick %s: %s", made.overall_pick, made.player_name
                )
                st.rerun()
        if not draft.is_user_on_clock:
            st.caption(
                "It is not your turn, so this button is disabled — use *Draft anyone* "
                "below to pick for the manager on the clock."
            )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# The clock: who picks next, and what they will probably take
# ─────────────────────────────────────────────────────────────────────────────
if not draft.is_complete:
    clock_left, clock_right = st.columns([3, 2])

    with clock_left:
        st.subheader("Between now and your next pick")
        upcoming = likely_next_picks(draft, profiles, count=min(12, max(1, until_turn or 1)))
        if not upcoming:
            st.caption("Nobody picks before you.")
        else:
            st.caption(
                "The single most likely pick for each manager, with its probability. "
                "These are guesses about individuals, not the field — a 12% likeliest "
                "pick means that manager's choice is genuinely open."
            )
            st.dataframe(
                pd.DataFrame([
                    {
                        "Slot": slot,
                        "Manager": name,
                        "Most likely pick": player.name,
                        "Pos": str(player.position),
                        "Probability": f"{probability:.0%}",
                    }
                    for slot, name, player, probability in upcoming
                ]),
                width="stretch", hide_index=True,
            )

    with clock_right:
        st.subheader("Positional pressure")
        st.caption(
            "Expected number of each position to be taken before your next turn. "
            "Above ~2 means the position is thinning while you wait — that is what "
            "makes waiting expensive, rather than the raw count remaining."
        )
        pressure = upcoming_position_pressure(draft, profiles, draft_slot=user_slot)
        components.position_bar_chart(
            {p: round(v, 2) for p, v in pressure.items() if v >= 0.005},
            "Expected picks before your turn",
        )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Recommendations
# ─────────────────────────────────────────────────────────────────────────────
if not draft.is_complete:
    st.subheader("What to do with this pick")

    rec_columns = st.columns([1, 1, 2])
    simulations = rec_columns[0].number_input(
        "Availability simulations", min_value=20, max_value=1000,
        value=int(settings.availability_simulations), step=20,
        help="More runs, tighter survival estimates, slower page. 120 is usually "
             "enough to separate 'safe' from 'coin flip'.",
    )
    shortlist_size = rec_columns[1].number_input(
        "Shortlist size", min_value=4, max_value=40, value=12,
        help="How many candidates each lens scores over.",
    )
    auto = rec_columns[2].checkbox(
        "Recompute automatically at every pick", value=True,
        help="Off, recommendations are computed only when you press the button — "
             "useful while advancing quickly through opponent picks.",
    )

    run_now = st.button("Compute recommendations", type="primary")

    recommendation_set = None
    if auto or run_now:
        # Stamped on the pick index so a recommendation computed two picks ago is
        # never shown as if it applied to the current board.
        # The board is part of the stamp because it is part of the answer: editing a
        # target list must recompute, and the cache cannot see inside the object.
        stamp = (
            draft.pick_index, int(simulations), int(shortlist_size),
            state.user_board().fingerprint,
        )
        with st.spinner(f"Simulating {simulations} rollouts to your next pick…"):
            recommendation_set = state.cached(
                "recommendations",
                lambda: recommend_for(
                    draft, profiles,
                    board=state.user_board(),
                    simulations=int(simulations),
                    shortlist_size=int(shortlist_size),
                    seed=draft.seed,
                ),
                stamp=stamp,
            )

    if recommendation_set is not None:
        for warning in recommendation_set.warnings:
            st.warning(warning, icon="⚠️")

        st.caption(
            f"{recommendation_set.picks_until_next} pick(s) until your next turn · "
            f"computed in {recommendation_set.elapsed_seconds:.2f}s · "
            f"{recommendation_set.roster_summary}"
        )

        consensus = recommendation_set.consensus_players
        if consensus:
            st.success(
                "**Several lenses agree on:** "
                + ", ".join(sorted(player.name for player in consensus))
                + " — agreement across lenses that optimise for different things is "
                "the strongest signal this app produces."
            )

        # One card per lens. The headline and detail come from the engine, not from
        # this page, so what is displayed is the reasoning the engine actually used.
        lens_columns = st.columns(2)
        for index, recommendation in enumerate(recommendation_set.recommendations):
            with lens_columns[index % 2].container(border=True):
                player = recommendation.player
                st.markdown(
                    f"**{recommendation.lens_label}**"
                    + ("  ⭐ consensus" if recommendation.is_consensus else "")
                    + ("  🎯 on your target list" if recommendation.is_target else "")
                    + (
                        f"  📋 your #{recommendation.board_rank}"
                        if recommendation.board_rank else ""
                    )
                )
                st.markdown(
                    f"### {player.name}  \n"
                    f"{player.position} · {player.nfl_team or 'FA'}"
                    + (f" · bye {player.bye_week}" if player.bye_week else "")
                )
                st.write(recommendation.headline)
                st.caption(recommendation.detail)
                metric_columns = st.columns(3)
                metric_columns[0].metric(
                    "Survives to your next pick",
                    f"{recommendation.survival:.0%}" if recommendation.survival is not None else "—",
                )
                metric_columns[1].metric("Utility", f"{recommendation.utility:.1f}")
                metric_columns[2].metric(
                    "Risk", components.risk_label(recommendation.risk_band)
                )
                if recommendation.components:
                    with st.expander("Score breakdown"):
                        st.caption(
                            "The weighted terms behind this lens's score. These weights "
                            "are editable on **Settings**."
                        )
                        st.dataframe(
                            pd.DataFrame(
                                sorted(
                                    recommendation.components.items(),
                                    key=lambda kv: -abs(kv[1]),
                                ),
                                columns=["Term", "Contribution"],
                            ),
                            width="stretch", hide_index=True,
                        )
                if st.button(
                    f"Draft {player.name}",
                    key=f"draft_{recommendation.lens}_{player.player_id}",
                    disabled=not draft.is_user_on_clock,
                    width="stretch",
                ):
                    draft.make_pick(
                        player,
                        pick_probability=recommendation.survival,
                        explanation=f"{recommendation.lens_label}: {recommendation.headline}",
                    )
                    st.rerun()

        if not draft.is_user_on_clock:
            st.caption(
                "The draft buttons are disabled because it is not your turn. Advance "
                "to your pick using the controls above."
            )

        # -- survival table ---------------------------------------------------
        availability = recommendation_set.availability
        if availability is not None:
            with st.expander(
                f"Will they last? — {len(availability.players)} players, "
                f"{availability.simulations} simulations"
            ):
                st.caption(
                    "Probability each player is still on the board at your next pick "
                    f"(overall {availability.target_pick}), and who tends to take them "
                    "when they go. This is simulated against your league's modelled "
                    "managers, so it differs from what raw ADP would suggest."
                )
                rows = []
                for entry in sorted(
                    availability.players.values(), key=lambda e: -e.survival
                ):
                    taken_by = ", ".join(
                        f"{name} {count / max(1, entry.simulations):.0%}"
                        for name, count in sorted(
                            entry.taken_by.items(), key=lambda kv: -kv[1]
                        )[:3]
                    )
                    rows.append({
                        "Player": entry.player.name,
                        "Pos": str(entry.player.position),
                        "ADP": entry.player.overall_adp,
                        "Survives": entry.survival,
                        "": components.survival_bar(entry.survival),
                        "Usually gone by": (
                            round(entry.mean_pick_taken, 1)
                            if entry.mean_pick_taken is not None else None
                        ),
                        "Most likely taken by": taken_by or "—",
                    })
                survival_frame = pd.DataFrame(rows)
                st.dataframe(
                    survival_frame, width="stretch", hide_index=True,
                    height=420,
                    column_config={
                        "Survives": st.column_config.ProgressColumn(
                            "Survives", min_value=0.0, max_value=1.0, format="%.0f%%",
                        ),
                    },
                )
                components.download_frame(
                    survival_frame, "Download survival table (CSV)",
                    f"survival_pick_{draft.pick_index + 1}.csv",
                )
                if availability.position_gone:
                    st.caption(
                        "Probability the *entire* startable supply at a position is "
                        "gone before your turn: "
                        + " · ".join(
                            f"{position} {probability:.0%}"
                            for position, probability in availability.position_gone.items()
                            if probability > 0.01
                        )
                    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Draft anyone
# ─────────────────────────────────────────────────────────────────────────────
if not draft.is_complete:
    st.subheader("Draft anyone")
    st.caption(
        "The recommendations are advice. Take whoever you want — including for the "
        "manager on the clock, if you are replaying a real draft pick by pick."
    )
    manual_columns = st.columns([2, 1, 1])
    position_filter = manual_columns[1].selectbox(
        "Position", ["All"] + [str(p) for p in Position],
    )
    candidates = (
        draft.available_players(limit=400) if position_filter == "All"
        else draft.available_at_position(Position.coerce(position_filter), limit=400)
    )
    chosen_id = manual_columns[0].selectbox(
        f"Player for {on_clock_manager} (slot {on_clock_slot})",
        [player.player_id for player in candidates],
        format_func=lambda pid: next(
            f"{p.name} — {p.position} {p.nfl_team or 'FA'} (ADP {p.overall_adp})"
            for p in candidates if p.player_id == pid
        ),
    )
    if manual_columns[2].button("Make this pick", width="stretch"):
        try:
            pick = draft.make_pick(
                chosen_id,
                was_manual_override=not draft.is_user_on_clock,
                explanation="Manually selected.",
            )
        except ConfigurationError as error:
            # Surfaced rather than swallowed: the reasons are all things the user
            # needs to know (draft over, player already gone, reserved keeper).
            st.error(f"That pick was rejected: {error}")
        else:
            LOGGER.info("Manual pick %s: %s", pick.overall_pick, pick.player_name)
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Keeping it
#
# The autosave above survives a refresh but is overwritten by the next draft. This is
# the deliberate keep: a named record that stays put and can be opened on Analysis.
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("💾 Save this draft for later", expanded=draft.is_complete):
    st.caption(
        "Saves the picks and every team's final roster to the local database under a "
        "name, so the draft can be reviewed on **Analysis** after this session ends. "
        "This is separate from the automatic save, which only ever holds the *current* "
        "draft."
    )
    save_name = st.text_input(
        "Name", value=f"{league.config.name} mock (pick {draft.pick_index})"
    )
    if st.button("Save this draft", type="primary"):
        result = MockDraftResult(
            name=save_name.strip() or "Mock draft",
            league_name=league.config.name,
            season=league.config.season,
            picks=list(draft.picks),
            user_slots=sorted(league.user_slots),
            random_seed=draft.seed,
            mode="interactive",
            notes=(
                "SAMPLE DATA — fictional players." if state.is_sample_data() else ""
            ),
            settings_snapshot=settings.to_dict(),
        )
        rosters = [draft.roster_copy(slot) for slot in league.slots_in_order()]
        with session_scope() as session:
            mock_id = save_mock_draft(
                session, result, league_id=league.config.league_id, rosters=rosters,
                user_slots=league.user_slots,
            )
        st.success(f"Saved as mock draft #{mock_id}. Open it on **Analysis**.")
