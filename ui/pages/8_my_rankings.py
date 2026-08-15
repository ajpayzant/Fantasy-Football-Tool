"""My Board: the user's own rankings, targets and never-draft list.

The one page in the app that holds the *user's* opinion rather than the model's.
Everything here changes what gets recommended to the person reading it and nothing
else — the eleven simulated opponents never see any of it, because a personal ranking
is not a prediction about how the room will behave. Making it one would corrupt every
availability percentage on the Draft Room page.

Three ways in, because ranking players is done three different ways in practice:

* **Upload** a file you already have, from anywhere you are entitled to use one.
* **Paste** a list off a site or out of a chat.
* **Edit** the table directly, which is the only one of the three that lets you build
  a ranking against the live board rather than in a vacuum.

All three write the same thing: ``UserBoard.custom_ranks``, a sparse name → rank map.
Sparse matters — a top-40 is a complete statement about your first forty picks and
says nothing about the rest, and the app treats it that way rather than assuming
everyone you left out is worthless.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from core.enums import Position
from services.adapters import read_tabular
from services.user_board import (
    MAX_NAMES,
    RANK_COLUMN_CANDIDATES,
    UserBoard,
    parse_names,
    parse_rankings,
    rankings_from_frame,
    rankings_from_order,
)
from ui import components, state

LOGGER = logging.getLogger("fantasy_mock_draft.ui.my_rankings")

components.page_header(
    "📋 My Board",
    "Your rankings, your targets, and the players you will not draft.",
)

board = state.user_board()
pool = state.pool()

st.info(
    "Everything on this page is **yours**. It changes what this app recommends to you "
    "and nothing else: the other managers keep drafting the players you have sworn "
    "off, because that is what they would really do, and pretending otherwise would "
    "make every availability number in the Draft Room wrong.",
    icon="🔒",
)


def _store(new_board: UserBoard, message: str) -> None:
    """Save and re-render. Every write on this page goes through here."""
    state.set_user_board(new_board)
    components.flash(message)
    st.rerun()


def _with_ranks(ranks: dict[str, int]) -> UserBoard:
    """The current board with its rankings replaced, the two lists untouched."""
    return UserBoard(
        targets=list(board.targets), avoid=list(board.avoid), custom_ranks=ranks
    )


# ─────────────────────────────────────────────────────────────────────────────
# Where the board stands
# ─────────────────────────────────────────────────────────────────────────────
components.metric_row([
    ("Ranked", len(board.custom_ranks), "players you have placed yourself"),
    ("Targets", len(board.targets), "wanted, in priority order"),
    ("Never draft", len(board.avoid), "excluded from every suggestion"),
])

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# My rankings
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("My rankings")
st.caption(
    "A partial list is fine and is the normal case. Rank the players you have a real "
    "opinion about; everyone else keeps the board's own ordering and sorts after the "
    f"ones you ranked. Up to {MAX_NAMES} names."
)

upload_tab, paste_tab, edit_tab = st.tabs(
    ["Upload a file", "Paste a list", "Build it here"]
)

with upload_tab:
    st.caption(
        "CSV, TSV or Excel. Only two things are read: which column holds the names, "
        "and what order the players are in — a rank column if the file has one, row "
        "order if it does not. Nothing else in the file is imported, because a "
        "ranking is an ordering and the projections in someone else's file are theirs."
    )
    st.caption(
        "Rank columns recognised, best first: "
        + ", ".join(f"`{c}`" for c in RANK_COLUMN_CANDIDATES)
        + ". Name columns: `player_name`, `player`, `name`."
    )
    ranking_file = st.file_uploader(
        "Ranking file", type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls"],
        key="ranks_upload",
    )
    replace_upload = st.checkbox(
        "Replace my current rankings", value=True, key="ranks_upload_replace",
        help="Off merges the file into what you already have. Where both give a "
             "player a rank, the file wins — you just uploaded it.",
    )
    if st.button("Import rankings", type="primary", key="ranks_upload_go"):
        if ranking_file is None:
            st.warning("Choose a file first.")
        else:
            frame, read_report = read_tabular(ranking_file, file_name=ranking_file.name)
            for problem in read_report.errors:
                st.error(problem.message)
            parsed, notes = rankings_from_frame(frame)
            for note in notes:
                st.caption(f"• {note}")
            if not parsed:
                st.error("Nothing usable in that file, so the board is unchanged.")
            else:
                merged = dict(parsed) if replace_upload else {
                    **board.custom_ranks, **parsed
                }
                _store(
                    _with_ranks(merged),
                    f"Imported {len(parsed)} ranked players from {ranking_file.name}.",
                )

with paste_tab:
    st.caption(
        "One player per line. `1. Player` numbering is honoured where you give it, and "
        "a line without a number takes its place in the list — so you can paste ranks "
        "1-10, skip to 25, and get what you meant."
    )
    pasted = st.text_area(
        "Pasted rankings",
        value="\n".join(
            f"{rank}. {name}"
            for name, rank in sorted(board.custom_ranks.items(), key=lambda kv: kv[1])
        ),
        height=260, key="ranks_paste", label_visibility="collapsed",
        placeholder="1. Ja'Marr Chase\n2. Bijan Robinson\n3. Justin Jefferson",
    )
    paste_left, paste_right = st.columns([1, 3])
    if paste_left.button("Save this list", type="primary", key="ranks_paste_go"):
        parsed = parse_rankings(pasted)
        if not parsed and pasted.strip():
            st.error("No player names found in that text.")
        else:
            _store(_with_ranks(parsed), f"Saved {len(parsed)} ranked players.")
    paste_right.caption(
        "This box replaces your rankings wholesale, and it starts out showing what "
        "you already have — so editing it in place is safe."
    )

with edit_tab:
    if pool is None or not len(pool):
        st.info(
            "Load a player pool on **Setup** first. This tab ranks players against "
            "the live board, so it needs one.",
            icon="ℹ️",
        )
    else:
        st.caption(
            "Type a number in **My rank** to place a player. Leave a row blank to "
            "leave that player unranked. Save re-bases whatever you typed to a dense "
            "1, 2, 3… so gaps and ties sort themselves out."
        )
        filter_left, filter_right = st.columns([1, 1])
        position_filter = filter_left.multiselect(
            "Positions", [str(p) for p in Position], key="ranks_edit_positions"
        )
        depth = filter_right.number_input(
            "Players to show, by consensus rank", min_value=25, max_value=MAX_NAMES,
            value=min(150, MAX_NAMES), step=25, key="ranks_edit_depth",
        )

        candidates = sorted(
            pool.players,
            key=lambda p: (
                board.custom_rank(p) is None,
                float(board.custom_rank(p) or 0),
                float(p.overall_adp if p.overall_adp is not None else 9e9),
            ),
        )
        if position_filter:
            wanted = set(position_filter)
            candidates = [p for p in candidates if str(p.position) in wanted]
        candidates = candidates[: int(depth)]

        editable = pd.DataFrame([
            {
                "My rank": board.custom_rank(player),
                "Player": player.name,
                "Pos": str(player.position),
                "Team": player.nfl_team or "FA",
                "ADP": player.overall_adp,
                "Proj": player.projection,
                "VOR": player.value_over_replacement,
            }
            for player in candidates
        ])
        edited = st.data_editor(
            editable,
            width="stretch", hide_index=True, height=520, key="ranks_editor",
            column_config={
                "My rank": st.column_config.NumberColumn(
                    "My rank", min_value=1, max_value=MAX_NAMES, step=1,
                    help="Your own overall rank for this player. Blank = unranked.",
                ),
            },
            disabled=["Player", "Pos", "Team", "ADP", "Proj", "VOR"],
        )
        edit_left, edit_right = st.columns([1, 3])
        if edit_left.button("Save this order", type="primary", key="ranks_edit_go"):
            typed = edited.dropna(subset=["My rank"])
            ordered = typed.sort_values("My rank", kind="stable")["Player"].tolist()
            # Re-based rather than saved as typed: the editor is where ties and gaps
            # come from, and two players sharing rank 4 is a statement about their
            # order that the app would otherwise have to break arbitrarily.
            parsed = rankings_from_order(ordered)
            # Anyone ranked earlier who is not on screen — filtered out by position, or
            # below the depth cut — keeps their rank rather than being silently dropped.
            offscreen = {
                name: rank for name, rank in board.custom_ranks.items()
                if name not in set(editable["Player"].tolist())
            }
            merged = {**offscreen, **parsed} if offscreen else parsed
            _store(
                _with_ranks(merged),
                f"Saved {len(parsed)} ranked players from the table."
                + (f" {len(offscreen)} off-screen ranking(s) kept." if offscreen else ""),
            )
        edit_right.caption(
            "Only the rows on screen are re-ordered. Players you ranked earlier that "
            "this filter hides keep the rank they had."
        )

if board.custom_ranks:
    clear_left, clear_right = st.columns([1, 3])
    if clear_left.button("Clear my rankings", key="ranks_clear"):
        _store(_with_ranks({}), "Rankings cleared. Targets and never-draft kept.")
    clear_right.caption("Leaves your targets and never-draft list alone.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Targets and never-draft
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Targets and never-draft")
st.caption(
    "Two lists that do a different job from a ranking. A ranking says how good you "
    "think a player is; a target says you want him **regardless**, and a never-draft "
    "says no price is low enough."
)
targets_column, avoid_column = st.columns(2)
with targets_column:
    st.markdown("**Targets**")
    st.caption("One per line, best first. The order is the priority.")
    targets_text = st.text_area(
        "Targets", value="\n".join(board.targets), height=220,
        key="my_targets", label_visibility="collapsed",
        placeholder="Ja'Marr Chase\nBijan Robinson",
    )
with avoid_column:
    st.markdown("**Never draft**")
    st.caption("Kept out of every suggestion, whatever the model thinks.")
    avoid_text = st.text_area(
        "Never draft", value="\n".join(board.avoid), height=220,
        key="my_avoid", label_visibility="collapsed",
        placeholder="A player you will not take",
    )

lists_left, lists_middle, _ = st.columns([1, 1, 3])
if lists_left.button("Save both lists", type="primary", key="my_lists_save"):
    _store(
        UserBoard(
            targets=parse_names(targets_text),
            avoid=parse_names(avoid_text),
            custom_ranks=dict(board.custom_ranks),
        ),
        "Lists saved. They apply to every draft from here on.",
    )
if lists_middle.button("Clear everything", key="my_board_clear"):
    _store(UserBoard(), "Board cleared — rankings, targets and never-draft.")

if board.conflicts:
    st.warning(
        "On both lists, so treated as never-draft: "
        + ", ".join(board.conflicts)
        + ". Refusing to draft someone is the safer reading of a contradiction than "
        "recommending them.",
        icon="⚠️",
    )

# ─────────────────────────────────────────────────────────────────────────────
# What the board actually resolves to
# ─────────────────────────────────────────────────────────────────────────────
if pool is not None and len(pool):
    unmatched = board.unmatched(pool)
    if unmatched:
        st.warning(
            "These names match nobody in the current player pool, so they do nothing: "
            + "; ".join(
                f"**{label.replace('_', ' ')}** — {', '.join(names)}"
                for label, names in unmatched.items()
            )
            + ". Check the spelling, or the player may not be in this file at all.",
            icon="⚠️",
        )

    if board.custom_ranks:
        st.divider()
        st.subheader("Where you disagree with the board")
        st.caption(
            "Your rank against the consensus one, biggest disagreement first. This is "
            "the useful readout of a personal ranking: the players you are higher on "
            "than the room are the ones you can wait on, and the ones you are lower on "
            "are where the room will hand you value."
        )
        rows = []
        for player in pool.players:
            mine = board.custom_rank(player)
            if mine is None:
                continue
            theirs = player.rank_for()
            rows.append({
                "My rank": mine,
                "Player": player.name,
                "Pos": str(player.position),
                "Board rank": theirs,
                "ADP": player.overall_adp,
                "Gap": None if theirs is None else round(float(theirs) - mine, 1),
                "Target": "★" if board.is_target(player) else "",
            })
        if rows:
            frame = pd.DataFrame(rows)
            frame = frame.reindex(
                frame["Gap"].abs().sort_values(ascending=False, na_position="last").index
            )
            st.dataframe(frame, width="stretch", hide_index=True, height=420)
            st.caption(
                "**Gap** is board rank minus yours. Positive means you are higher on "
                "him than the consensus is."
            )
            components.download_frame(frame, "Download my rankings (CSV)", "my_rankings.csv")
        else:
            st.info(
                "None of your ranked names matched a player in this pool.", icon="ℹ️"
            )
else:
    st.caption(
        "Load a player pool on **Setup** to see how your board resolves against it."
    )
