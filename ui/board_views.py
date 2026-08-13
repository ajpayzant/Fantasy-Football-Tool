"""The three ways to look at a draft in progress, and the shortlist above them.

The Draft Room used to bury the board at the bottom of the page, under the
recommendations. That is the wrong way round: the board is the thing a drafter
actually looks at between picks, and the suggestions only make sense against it. So
this module renders it, and the page puts it first.

Three views, because they answer different questions:

* **Draft order** — "what has happened", newest first. A log.
* **The board** — "where are we", the wall-chart grid every real draft room has, one
  column per team and one row per round, snaking left-to-right then right-to-left so
  the shape of the order is visible rather than described.
* **Team rosters** — "what is everyone building", position by position.

Rendering lives here rather than in the page so the page reads as a sequence of
sections instead of three hundred lines of table construction. Everything here takes
engine objects and returns frames or draws widgets; nothing decides anything.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from core.enums import Position
from engine.draft_state import DraftState
from models.league import League
from models.player import PlayerPool
from services.user_board import UserBoard

POSITION_COLOURS: dict[str, str] = {
    "QB": "#f2d0d9",
    "RB": "#cfe8d4",
    "WR": "#cfe0f2",
    "TE": "#f6e3c5",
    "K": "#e4dcf0",
    "DST": "#dcdcdc",
}
"""Background per position on the grid.

Deliberately pale: the grid is read as text, and saturated colours make a wall of
twelve columns unreadable. They are also the only cue that survives shrinking a cell
to fit, which is why position is coloured rather than, say, reach.
"""

_ON_CLOCK_COLOUR = "#fff3bf"
_EMPTY_COLOUR = "#fafafa"


# ─────────────────────────────────────────────────────────────────────────────
# View 1 — the draft order as a list
# ─────────────────────────────────────────────────────────────────────────────
def draft_order_frame(draft: DraftState, *, newest_first: bool = True) -> pd.DataFrame:
    """Every pick made, in order, with what it cost against ADP."""
    picks = list(reversed(draft.picks)) if newest_first else list(draft.picks)
    return pd.DataFrame([
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
            "Keeper": "K" if pick.is_keeper else "",
            "Why": pick.explanation,
        }
        for pick in picks
    ])


def render_draft_order(draft: DraftState) -> None:
    if not draft.picks:
        st.caption("No picks yet. Advance the clock or make one.")
        return
    frame = draft_order_frame(draft)
    st.dataframe(frame, width="stretch", hide_index=True, height=420)
    st.caption(
        "`Reach` is picks earlier than ADP — negative means the player fell to them. "
        "Newest pick first."
    )
    from ui import components

    components.download_frame(frame, "Download the pick list (CSV)", "draft_order.csv")


# ─────────────────────────────────────────────────────────────────────────────
# View 2 — the wall chart
# ─────────────────────────────────────────────────────────────────────────────
def _cell_text(draft: DraftState, slot_info: Any, pick: Any) -> str:
    """What goes in one grid cell: the player, or the pick number if unmade."""
    if pick is not None:
        position = str(pick.position) if pick.position else ""
        keeper = " (K)" if pick.is_keeper else ""
        return f"{pick.player_name}{keeper}\n{position} · {pick.nfl_team or 'FA'}"
    if slot_info.overall_pick == draft.pick_index + 1:
        return "▶ ON THE CLOCK"
    if slot_info.is_keeper_pick and slot_info.keeper_player_name:
        return f"{slot_info.keeper_player_name}\nkeeper"
    return f"#{slot_info.overall_pick}"


def snake_grid(draft: DraftState, league: League) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The wall chart, plus a parallel frame of position codes for colouring.

    Rows are rounds and columns are draft slots, which is what makes the snake
    visible: round 1 fills left to right, round 2 right to left, and a user in slot 1
    can see at a glance that their next pick is twenty-two picks away. Returned as two
    frames rather than one styled object so the caller can also show it unstyled — a
    Styler cannot be filtered or downloaded.
    """
    by_overall = {pick.overall_pick: pick for pick in draft.picks}
    slots = league.slots_in_order()
    columns = [
        f"{slot} · {league.require_manager_by_slot(slot).name}"[:28] for slot in slots
    ]
    rows: dict[str, dict[str, str]] = {}
    marks: dict[str, dict[str, str]] = {}
    for slot_info in draft.order:
        row = f"R{slot_info.round_number}"
        try:
            column = columns[slots.index(slot_info.draft_slot)]
        except ValueError:  # pragma: no cover - a slot outside the league
            continue
        pick = by_overall.get(slot_info.overall_pick)
        rows.setdefault(row, {})[column] = _cell_text(draft, slot_info, pick)
        marks.setdefault(row, {})[column] = (
            str(pick.position) if pick is not None and pick.position
            else ("__clock__" if slot_info.overall_pick == draft.pick_index + 1 else "")
        )
    order = [f"R{n}" for n in range(1, league.config.rounds + 1) if f"R{n}" in rows]
    grid = pd.DataFrame.from_dict(rows, orient="index").reindex(order)[columns]
    mark = pd.DataFrame.from_dict(marks, orient="index").reindex(order)[columns]
    return grid.fillna(""), mark.fillna("")


def render_snake_grid(draft: DraftState, league: League, *, user_slot: int) -> None:
    """The wall chart, coloured by position with the current pick highlighted."""
    grid, marks = snake_grid(draft, league)
    if grid.empty:
        st.caption("The draft order is empty.")
        return

    def colour(_: pd.DataFrame) -> pd.DataFrame:
        styles = marks.copy()
        for row in marks.index:
            for column in marks.columns:
                token = marks.at[row, column]
                if token == "__clock__":
                    styles.at[row, column] = (
                        f"background-color: {_ON_CLOCK_COLOUR}; font-weight: 700"
                    )
                elif token:
                    styles.at[row, column] = (
                        f"background-color: {POSITION_COLOURS.get(token, _EMPTY_COLOUR)}"
                    )
                else:
                    styles.at[row, column] = f"background-color: {_EMPTY_COLOUR}"
        return styles

    st.caption(
        "One column per team, one row per round — so the snake reads as a shape: "
        "round 1 goes left to right, round 2 comes back. The highlighted cell is the "
        "pick on the clock, and colour is position. Your column is marked below."
    )
    user_column = [
        column for slot, column in zip(league.slots_in_order(), grid.columns)
        if slot == user_slot
    ]
    if user_column:
        st.caption(f"You are **{user_column[0]}**.")
    st.dataframe(
        grid.style.apply(colour, axis=None),
        width="stretch", height=min(720, 60 + 46 * len(grid)),
    )
    legend = " · ".join(
        f"{position}" for position in POSITION_COLOURS if position in set(
            marks.to_numpy().ravel()
        )
    )
    if legend:
        st.caption(f"Positions on the board: {legend}")


# ─────────────────────────────────────────────────────────────────────────────
# View 3 — roster construction
# ─────────────────────────────────────────────────────────────────────────────
def roster_shape_frame(draft: DraftState, league: League) -> pd.DataFrame:
    """Every team's roster shape side by side: who has what, and what is missing.

    The comparison is the point. A single team's roster answers "what do I have"; the
    whole table answers "who else needs a quarterback", which is the question that
    decides whether waiting a round is safe.
    """
    slots = league.slots_in_order()
    rosters = {slot: draft.roster_copy(slot) for slot in slots}
    counts = {slot: roster.position_counts() for slot, roster in rosters.items()}
    # Only show position columns somebody has actually drafted, so an empty board is
    # four columns wide instead of ten mostly-zero ones.
    drafted = [
        position for position in Position
        if any(counts[slot].get(position, 0) for slot in slots)
    ]
    rows: list[dict[str, Any]] = []
    for slot in slots:
        roster = rosters[slot]
        open_slots = roster.open_starting_slots()
        rows.append({
            "Slot": slot,
            "Manager": league.require_manager_by_slot(slot).name,
            "You": "★" if slot in league.user_slots else "",
            "Picks": len(roster),
            **{
                str(position): int(counts[slot].get(position, 0))
                for position in drafted
            },
            "Starters unfilled": sum(open_slots.values()),
            "Still needs": ", ".join(
                f"{count}×{slot_name}" for slot_name, count in open_slots.items()
            ) or "—",
        })
    return pd.DataFrame(rows)


def render_team_rosters(
    draft: DraftState, league: League, pool: PlayerPool, *, user_slot: int
) -> None:
    """One team's lineup and bench, with every team's shape underneath."""
    slots = league.slots_in_order()
    slot_choice = st.selectbox(
        "Team", slots,
        index=max(0, slots.index(user_slot)) if user_slot in slots else 0,
        format_func=lambda s: (
            f"Slot {s} — {league.require_manager_by_slot(s).name}"
            + (" (you)" if s == user_slot else "")
        ),
        key="roster_view_team",
    )
    roster = draft.roster_copy(slot_choice)
    left, right = st.columns([2, 1])
    with left:
        st.markdown("**Starting lineup**")
        lineup_rows = [
            {
                "Slot": str(slot),
                "Player": (pool.get(pid).name if pool.get(pid) else pid),
                "Pos": str(pool.get(pid).position) if pool.get(pid) else "",
                "Proj": pool.get(pid).projection if pool.get(pid) else None,
                "Bye": pool.get(pid).bye_week if pool.get(pid) else None,
            }
            for slot, player_ids in roster.lineup.items()
            for pid in player_ids
        ]
        if lineup_rows:
            st.dataframe(pd.DataFrame(lineup_rows), width="stretch", hide_index=True)
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
    with right:
        from ui import components

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
            dict(roster.position_counts()), "Positions rostered"
        )

        note = st.text_input(
            "Add a note about this team", key=f"note_{slot_choice}"
        )
        if st.button("Save note", key=f"save_note_{slot_choice}") and note.strip():
            draft.note_strategy(slot_choice, note.strip())
            st.rerun()
        for existing in draft.strategy_notes(slot_choice):
            st.caption(f"• {existing}")

    st.markdown("**Every team's shape**")
    st.caption(
        "Who else still needs what. This is the table that tells you whether waiting a "
        "round on a position is safe or suicidal."
    )
    st.dataframe(roster_shape_frame(draft, league), width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# The shortlist above the recommendations
# ─────────────────────────────────────────────────────────────────────────────
def top_remaining_frame(
    draft: DraftState,
    board: UserBoard,
    *,
    count: int = 25,
    position: Position | None = None,
    order: str = "Board order",
) -> pd.DataFrame:
    """The best players left, in whichever order the user asked for.

    ``Board order`` is the pool's own ordering — the blended consensus. ``My board``
    applies the user's target list and personal rankings and drops the players they
    have sworn off. The two are offered side by side rather than merged because the
    gap between them is information: it is where the user is deliberately off
    consensus.
    """
    limit = max(1, int(count))
    available = (
        draft.available_players(limit=400) if position is None
        else draft.available_at_position(position, limit=400)
    )
    if order == "My board":
        available = board.sorted_players(available)
    elif order == "Projection":
        available = sorted(
            available, key=lambda p: -(p.projection or 0.0)
        )
    elif order == "Value over replacement":
        available = sorted(
            available, key=lambda p: -(p.value_over_replacement or 0.0)
        )
    rows: list[dict[str, Any]] = []
    for index, player in enumerate(available[:limit], start=1):
        priority = board.target_priority(player)
        rows.append({
            "#": index,
            "Mine": (
                f"🎯 {priority}" if priority is not None
                else ("⛔" if board.is_avoided(player) else "")
            ),
            "Player": player.name,
            "Pos": str(player.position),
            "Team": player.nfl_team or "FA",
            "Bye": player.bye_week,
            "ADP": player.overall_adp,
            "My rank": board.custom_rank(player),
            "Tier": player.tier,
            "Proj": player.projection,
            "VOR": player.value_over_replacement,
            "Injury": (
                "" if str(player.injury_status).lower() in ("healthy", "")
                else str(player.injury_status)
            ),
            "player_id": player.player_id,
        })
    return pd.DataFrame(rows)


__all__ = [
    "POSITION_COLOURS", "draft_order_frame", "render_draft_order", "snake_grid",
    "render_snake_grid", "roster_shape_frame", "render_team_rosters",
    "top_remaining_frame",
]
