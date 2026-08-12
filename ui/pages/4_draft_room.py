"""Draft Room: run a mock draft, pick by pick, with the engine's reasoning exposed.

The layout follows the actual decision a drafter makes on the clock:

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
from services.repository import save_mock_draft
from ui import components, state

LOGGER = logging.getLogger("fantasy_mock_draft.ui.draft_room")

components.page_header(
    "🎯 Draft Room",
    "A live mock draft against the modelled opponents.",
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
    components.flash("Draft abandoned.", "info")
    st.rerun()

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
        stamp = (draft.pick_index, int(simulations), int(shortlist_size))
        with st.spinner(f"Simulating {simulations} rollouts to your next pick…"):
            recommendation_set = state.cached(
                "recommendations",
                lambda: recommend_for(
                    draft, profiles,
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
# Board and rosters
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("The board")
board_tab, roster_tab, runs_tab, save_tab = st.tabs(
    ["Picks so far", "Rosters", "Runs and scarcity", "Save"]
)

with board_tab:
    picks = draft.picks
    if not picks:
        st.caption("No picks yet.")
    else:
        board_frame = pd.DataFrame([
            {
                "Pick": f"{pick.round_number}.{pick.pick_in_round:02d}",
                "Overall": pick.overall_pick,
                "Manager": pick.manager_name,
                "You": "★" if pick.is_user_pick else "",
                "Player": pick.player_name,
                "Pos": str(pick.position) if pick.position else "",
                "Team": pick.nfl_team or "",
                "ADP": pick.adp_at_pick,
                "Reach": (
                    round(pick.overall_pick - pick.adp_at_pick, 1)
                    if pick.adp_at_pick else None
                ),
                "Slot filled": str(pick.assigned_slot or ""),
                "Why": pick.explanation,
            }
            for pick in reversed(picks)
        ])
        st.dataframe(board_frame, width="stretch", hide_index=True, height=420)
        components.download_frame(board_frame, "Download board (CSV)", "draft_board.csv")
        st.caption(
            "`Reach` is picks earlier than ADP — negative means the player fell. "
            "Newest pick first."
        )

with roster_tab:
    slot_choice = st.selectbox(
        "Team", league.slots_in_order(),
        index=max(0, league.slots_in_order().index(user_slot)),
        format_func=lambda s: (
            f"Slot {s} — {league.require_manager_by_slot(s).name}"
            + (" (you)" if s == user_slot else "")
        ),
    )
    roster = draft.roster_copy(slot_choice)
    roster_left, roster_right = st.columns([2, 1])
    with roster_left:
        st.markdown("**Starting lineup**")
        lineup_rows = []
        for slot, player_ids in roster.lineup.items():
            for player_id in player_ids:
                player = pool.get(player_id)
                lineup_rows.append({
                    "Slot": str(slot),
                    "Player": player.name if player else player_id,
                    "Pos": str(player.position) if player else "",
                    "Proj": player.projection if player else None,
                    "Bye": player.bye_week if player else None,
                })
        if lineup_rows:
            st.dataframe(
                pd.DataFrame(lineup_rows), width="stretch", hide_index=True
            )
        else:
            st.caption("No starters yet.")

        if roster.bench:
            st.markdown("**Bench**")
            st.dataframe(
                pd.DataFrame([
                    {
                        "Player": (pool.get(pid).name if pool.get(pid) else pid),
                        "Pos": str(pool.get(pid).position) if pool.get(pid) else "",
                        "Proj": pool.get(pid).projection if pool.get(pid) else None,
                    }
                    for pid in roster.bench
                ]),
                width="stretch", hide_index=True,
            )

    with roster_right:
        st.markdown("**Still to fill**")
        open_slots = roster.open_starting_slots()
        if open_slots:
            st.dataframe(
                pd.DataFrame(
                    [{"Slot": str(s), "Open": n} for s, n in open_slots.items()]
                ),
                width="stretch", hide_index=True,
            )
        else:
            st.success("Every starting slot is filled.")
        st.metric("Roster size", f"{len(roster)} / {league.config.roster.roster_size}")
        components.position_bar_chart(
            {p: n for p, n in roster.position_counts().items()}, "Positions rostered"
        )

        notes = draft.strategy_notes(slot_choice)
        note = st.text_input("Add a note about this team", key=f"note_{slot_choice}")
        if st.button("Save note", key=f"save_note_{slot_choice}") and note.strip():
            draft.note_strategy(slot_choice, note.strip())
            st.rerun()
        for existing in notes:
            st.caption(f"• {existing}")

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

with save_tab:
    st.caption(
        "Saves the picks and every team's final roster to the local database, so the "
        "draft can be reviewed on **Analysis** after this session ends."
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
