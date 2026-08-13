"""Player Pool: the board the whole app draws from, and where its numbers came from.

Two jobs. First, let the user find and sort players. Second — and the reason this page
is more than a table — show what the importer *derived*: value over replacement is
computed from your scoring settings, tiers and ranks may have been imputed, and a
recommendation that surprises you usually traces back to one of those.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd
import streamlit as st

from core.enums import Position
from ui import components, state

components.page_header(
    "📋 Player Pool",
    "Every player the engine can draft, with the values it derived from your settings.",
)
components.require()

pool = state.pool()
league = state.league()
frame = pool.to_frame()

# ─────────────────────────────────────────────────────────────────────────────
# The platforms this pool carries
#
# Every source keeps its own column all the way to the table, and those columns are on
# the default view rather than hidden behind a toggle. The engine drafts against a
# single *weighted* blend (`overall_adp`, weights editable on Settings), which is the
# right number to draft against and the wrong one to answer "what does ESPN actually
# think" — the user cannot see the weights from the table. So each platform's own number
# sits beside the blend, and the plain unweighted mean sits beside both: where the mean
# and the blend disagree, the difference *is* the weighting, which is the thing worth
# noticing. Which platforms count is the user's call, so it is a filter.
# ─────────────────────────────────────────────────────────────────────────────
class Platform(NamedTuple):
    """One data source and the two kinds of column it can contribute.

    ``None`` where a source does not publish that kind: Fantasy Football Calculator
    reports draft position and never a ranking, Sleeper reports popularity and never a
    draft position. Pretending either had both would put an empty column on the table.
    """

    name: str
    adp_column: str | None
    rank_column: str | None
    what_it_is: str


PLATFORMS: tuple[Platform, ...] = (
    Platform(
        "FFC", "ffc_adp", None,
        "Fantasy Football Calculator — average pick across real public mock drafts. "
        "The only source here that reports what drafters did rather than what a site "
        "recommends.",
    ),
    Platform(
        "ESPN", "espn_adp", "espn_rank",
        "ESPN — average pick across ESPN leagues, plus ESPN's own published ranking.",
    ),
    Platform(
        "Yahoo", "yahoo_adp", "yahoo_rank",
        "Yahoo — average pick across Yahoo leagues, plus Yahoo's ranking.",
    ),
    Platform(
        "Sleeper", None, "sleeper_rank",
        "Sleeper — search popularity rather than a ranking. It leads ADP rather than "
        "reporting it, so treat it as attention, not opinion.",
    ),
)

PLATFORM_COLUMN_LABELS: dict[str, str] = {
    "ffc_adp": "FFC ADP", "espn_adp": "ESPN ADP", "yahoo_adp": "Yahoo ADP",
    "espn_rank": "ESPN rank", "yahoo_rank": "Yahoo rank", "sleeper_rank": "Sleeper",
}
"""Column name to header. One dict, used for both the rename and the column sets, so a
relabelled header cannot silently drop out of a view."""


def _coverage(column: str | None) -> int:
    """How many players this source actually has. Zero means it did not load."""
    if not column or column not in frame.columns:
        return 0
    return int(frame[column].notna().sum())


def _row_mean(source_frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Mean of whichever of ``columns`` this row actually has.

    ``skipna=True`` is the whole point: a player only Yahoo has gets Yahoo's number
    rather than a mean dragged toward nothing, and the Sources count beside it says how
    many opinions that average rests on.
    """
    present = [column for column in columns if column in source_frame.columns]
    if not present:
        return pd.Series(dtype=float, index=source_frame.index)
    return source_frame[present].mean(axis=1, skipna=True)


# Offered only where the source has at least one number in this pool. A platform that
# failed to load, or that Setup switched off, is absent from the picker rather than
# present and empty — a checkbox that does nothing is worse than no checkbox.
AVAILABLE_PLATFORMS = [
    platform for platform in PLATFORMS
    if _coverage(platform.adp_column) or _coverage(platform.rank_column)
]

# ─────────────────────────────────────────────────────────────────────────────
# Provenance — where these numbers came from
# ─────────────────────────────────────────────────────────────────────────────
metadata = pool.metadata
components.metric_row([
    ("Players", len(pool), "Total rows accepted by the importer."),
    ("Source", metadata.source or "unknown", "The file, paste or adapter this came from."),
    ("Season", metadata.season or "—", ""),
    ("Platform", metadata.platform or "—", "Used to interpret platform-specific ranks."),
])

with st.expander("Where every number on this page comes from"):
    st.caption(metadata.describe())

    # Each column the engine relies on, and how this pool got it. Named per column
    # rather than as one blanket disclaimer, because "supplied by ESPN" and "inferred
    # from draft position" deserve very different amounts of trust and the user cannot
    # tell them apart from the table alone.
    st.markdown("**Column by column**")
    supplied = int(len(frame)) - int(metadata.imputed_fields.get("projection", 0))
    provenance = [
        (
            "Projection",
            f"{supplied} of {len(frame)} players carry a real projected stat line — "
            "ESPN's projected season, re-scored under your own scoring rules rather "
            "than ESPN's. The rest are estimated from draft position, and the "
            "Source column on each row says which is which.",
        ),
        (
            "ADP",
            "A weighted blend of every source that had the player: Fantasy Football "
            "Calculator (real mock drafts), ESPN and Yahoo (their own leagues' "
            "averages). Each source's own number is in its own column below, and "
            "Sources counts how many had him.",
        ),
        (
            "ADP σ",
            "The standard deviation of the player's pick across real mock drafts, "
            "from Fantasy Football Calculator. Estimated at a fifth of ADP where FFC "
            "has no data on him — flagged per row.",
        ),
        (
            "Tier",
            "Not supplied by anyone: derived here. Within a position, players are "
            "ordered by projection and a new tier starts wherever the drop to the "
            "next player is bigger than the average drop plus one standard "
            "deviation. So a tier break is a gap that is unusual for that position, "
            "which is why QB and WR tiers come out different sizes.",
        ),
        (
            "Ceiling / Floor",
            "The range of value implied by how much drafters disagree about the "
            "player, mapped onto his position's projection curve. **Not a forecast "
            "of his season** — a player everyone slots at the same pick gets a "
            "narrow band here even though his real season could go anywhere.",
        ),
        (
            "Risk",
            "How much less predictable this player is than his positional peers: "
            "mostly draft-pick disagreement relative to his own ADP, plus injury "
            "status and whether he has an NFL season on record.",
        ),
        (
            "VOR",
            (
                f"Projected points above the last startable player at the position, "
                f"for {league.config.team_count} teams and this exact lineup "
                f"({league.config.roster.starters_total} starters, "
                f"{league.config.roster.bench_total} bench). Change the lineup or "
                f"scoring on **Setup** and every VOR here moves."
                if league is not None else
                "Projected points above the last startable player at the position. "
                "Needs a league, so it is blank until one is set up."
            ),
        ),
    ]
    st.dataframe(
        pd.DataFrame(provenance, columns=["Column", "How it was produced"]),
        width="stretch", hide_index=True,
    )

    if metadata.missing_fields:
        st.markdown("**Nothing supplied this, and nothing could derive it**")
        st.caption(
            "These cells are genuinely blank. Nothing downstream depends on them, "
            "which is why they were left alone rather than guessed at."
        )
        st.dataframe(
            pd.DataFrame(
                sorted(metadata.missing_fields.items()), columns=["Field", "Rows missing"]
            ),
            width="stretch", hide_index=True,
        )
    if metadata.imputed_fields:
        st.markdown("**Derived rather than supplied**")
        st.caption(
            "The engine needs an ordering and a projection for every player, so a "
            "missing value is derived rather than left blank. Derived values are "
            "worse than real ones — if a column here matters to you, supply it."
        )
        st.dataframe(
            pd.DataFrame(
                sorted(metadata.imputed_fields.items()), columns=["Field", "Rows derived"]
            ),
            width="stretch", hide_index=True,
        )
    if metadata.notes:
        st.caption(metadata.notes)

# ─────────────────────────────────────────────────────────────────────────────
# VOR, explained
#
# Its own section rather than a tooltip, because VOR is the number most likely to be
# quoted without being understood — and the honest version includes what it cannot do.
# The replacement ranks are read out of the live league config, so the numbers here are
# this league's, not a textbook's.
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📐 What VOR is, where it comes from, and how much to trust it"):
    st.markdown(
        "**VOR = this player's projected points − the projected points of the last "
        "player at his position anyone would start.**"
    )
    st.caption(
        "That subtraction is the whole idea. Raw projected points cannot be compared "
        "across positions: a quarterback out-scores every running back in almost any "
        "format, and drafting on projection alone would have you take five of them. "
        "What matters is not how many points a player scores but how many *more* he "
        "scores than the player you would otherwise have had in that slot. VOR is that "
        "difference, and it is why a tight end projected for 190 can be worth more than "
        "a running back projected for 210."
    )
    st.markdown("**How it is computed here**")
    if league is not None:
        replacement_rows = []
        for position in Position:
            group = pool.by_position(position)
            if not group:
                continue
            cutoff = int(round(league.config.replacement_rank(position)))
            ranked = sorted(group, key=lambda p: -(p.projection or 0.0))
            index = min(max(0, cutoff - 1), len(ranked) - 1)
            replacement = ranked[index]
            replacement_rows.append({
                "Position": str(position),
                "Replacement is": f"{position}{index + 1}",
                "Who that is": replacement.name,
                "Replacement points": (
                    round(float(replacement.projection or 0.0), 1)
                ),
                "Players in pool": len(group),
            })
        st.caption(
            "Within each position, players are sorted by projection and the "
            f"replacement level is set at the rank your league's demand implies for "
            f"{league.config.team_count} teams and this exact lineup "
            f"({league.config.roster.starters_total} starters, "
            f"{league.config.roster.bench_total} bench). Every player at that position "
            "is then measured against that one player's projection:"
        )
        st.dataframe(
            pd.DataFrame(replacement_rows), width="stretch", hide_index=True,
        )
        st.caption(
            "This is why VOR moves when you change the lineup. Add a second starting "
            "quarterback or a superflex and the QB replacement level drops much deeper "
            "into the position, every quarterback's VOR jumps, and the model starts "
            "taking them earlier — without anyone editing a ranking."
        )
    else:
        st.caption(
            "Needs a league to know how many starters each position demands, so VOR is "
            "blank until one is set up on **Setup**."
        )
    st.markdown("**How useful it actually is**")
    st.markdown(
        "- **Good at**: comparing across positions, and telling you when a position is "
        "genuinely scarce in *your* format rather than in general. It is the reason "
        "this app will tell you a mid-tier tight end is a better pick than a better-"
        "projected receiver.\n"
        "- **Good at**: exposing the flat middle of a position. When twenty running "
        "backs have VOR within ten points of each other, the position is a commodity "
        "and reaching inside that block costs you nothing but flexibility.\n"
        "- **Bad at**: anything that depends on *when* a player is available. VOR says "
        "a player is worth 60 points more than replacement; it does not say he will "
        "still be there in two rounds. That is what the survival simulation in the "
        "**Draft Room** is for, and it is why VOR is one term in the pick model rather "
        "than the model itself.\n"
        "- **Bad at**: bench value and streaming. Replacement level assumes you start "
        "the same lineup all year, so a handcuff running back or a bye-week fill-in "
        "scores badly on VOR while being a perfectly sensible late pick.\n"
        "- **Only as good as the projection behind it.** A projection this app derived "
        "from draft position produces a VOR derived from draft position — circular, and "
        "flagged as such in the `Projection from` column."
    )
    st.caption(
        "Its weight in the model is `value_over_replacement` on **Settings**. Set it to "
        "zero and the room drafts on raw points and ADP instead — a quick way to see how "
        "much of any given recommendation is VOR's doing."
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# The user's own board
#
# Kept on this page rather than in the Draft Room because it is preparation, not a
# live decision: this is where the user reads the pool and forms the opinions the
# lists record. The Draft Room then obeys them.
# ─────────────────────────────────────────────────────────────────────────────
board = state.user_board()
with st.expander(
    f"🎯 Your board — targets, do-not-draft and your own rankings ({board.describe()})",
    expanded=board.is_empty,
):
    st.caption(
        "These three lists are **yours**, and they change only what this app "
        "recommends to you. The eleven opponents never see them — they keep drafting "
        "the players you have sworn off, because that is what they would really do, "
        "and pretending otherwise would make every availability percentage wrong."
    )
    board_left, board_middle, board_right = st.columns(3)
    with board_left:
        st.markdown("**Targets**")
        st.caption("One per line, best first. The order is the priority.")
        targets_text = st.text_area(
            "Targets", value="\n".join(board.targets), height=200,
            key="board_targets", label_visibility="collapsed",
            placeholder="Ja'Marr Chase\nBijan Robinson",
        )
    with board_middle:
        st.markdown("**Never draft**")
        st.caption("Kept out of every suggestion, whatever the model thinks.")
        avoid_text = st.text_area(
            "Never draft", value="\n".join(board.avoid), height=200,
            key="board_avoid", label_visibility="collapsed",
            placeholder="A player you will not take",
        )
    with board_right:
        st.markdown("**Your own rankings**")
        st.caption(
            "Optional and partial is fine — `1. Player` numbering is honoured, plain "
            "lines take their place in the list."
        )
        ranks_text = st.text_area(
            "Your rankings",
            value="\n".join(
                f"{rank}. {name}"
                for name, rank in sorted(board.custom_ranks.items(), key=lambda kv: kv[1])
            ),
            height=200, key="board_ranks", label_visibility="collapsed",
            placeholder="1. Ja'Marr Chase\n2. Bijan Robinson",
        )

    save_column, clear_column, _ = st.columns([1, 1, 3])
    if save_column.button("Save my board", type="primary", key="board_save"):
        from services.user_board import UserBoard, parse_names, parse_rankings

        state.set_user_board(UserBoard(
            targets=parse_names(targets_text),
            avoid=parse_names(avoid_text),
            custom_ranks=parse_rankings(ranks_text),
        ))
        components.flash("Board saved. It applies to every draft from here on.")
        st.rerun()
    if clear_column.button("Clear it", key="board_clear"):
        from services.user_board import UserBoard

        state.set_user_board(UserBoard())
        components.flash("Board cleared.")
        st.rerun()

    if board.conflicts:
        st.warning(
            "On both lists, so treated as never-draft: "
            + ", ".join(board.conflicts)
            + ". Refusing to draft someone is the safer reading of a contradiction "
            "than recommending them.",
            icon="⚠️",
        )
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

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Filters
# ─────────────────────────────────────────────────────────────────────────────
filter_columns = st.columns([2, 1, 1, 1])
positions = filter_columns[0].multiselect(
    "Positions",
    [p for p in Position if pool.by_position(p)],
    format_func=lambda p: f"{p} ({len(pool.by_position(p))})",
)
search = filter_columns[1].text_input("Name contains", "")
rookies_only = filter_columns[2].checkbox("Rookies only")
hide_injured = filter_columns[3].checkbox(
    "Hide injured", help="Drops anyone whose injury status is not Healthy."
)

# ─────────────────────────────────────────────────────────────────────────────
# Which platforms count
#
# This drives three things at once, deliberately: which per-platform columns appear on
# the table, which sources the plain average is taken over, and which per-platform sort
# options are offered. One control, because a user who says "I only care about ESPN and
# FFC" means all three.
# ─────────────────────────────────────────────────────────────────────────────
complete_only = False
if not AVAILABLE_PLATFORMS:
    # A pool imported from a single file, or a synthetic one. No picker, because there is
    # nothing to pick between, and no empty columns on the table either.
    selected_platforms: list[Platform] = []
    st.caption(
        "This pool carries no per-platform numbers, so there is nothing to compare: "
        "every ADP and rank came from one source. Load a pool from **Setup** to get "
        "each platform's own column."
    )
else:
    platform_columns = st.columns([3, 1])
    platform_choice = platform_columns[0].multiselect(
        "Platforms",
        [platform.name for platform in AVAILABLE_PLATFORMS],
        default=[platform.name for platform in AVAILABLE_PLATFORMS],
        format_func=lambda name: next(
            f"{p.name} ({_coverage(p.adp_column) or _coverage(p.rank_column)} players)"
            for p in AVAILABLE_PLATFORMS if p.name == name
        ),
        key="pool_platforms",
        help="Which sources to show and average. Deselecting one removes its column, "
             "drops it out of the Avg ADP and Avg rank calculations and takes it off "
             "the sort list. It does **not** change what the engine drafts against — "
             "that blend and its weights live on Settings.",
    )
    selected_platforms = [
        platform for platform in AVAILABLE_PLATFORMS if platform.name in platform_choice
    ]
    if not selected_platforms:
        # Falling back to everything rather than showing an averageless table, and saying
        # so — a silent fallback would read as "these are your selected platforms".
        st.info(
            "No platform selected, so all of them are shown. Pick at least one to "
            "narrow the table."
        )
        selected_platforms = list(AVAILABLE_PLATFORMS)
    complete_only = platform_columns[1].checkbox(
        "Only fully covered", key="pool_complete_only",
        help="Keep only players that *every* selected platform has an ADP for. Useful "
             "when comparing columns — a blank cell makes two platforms look like they "
             "disagree when one of them simply never listed the player.",
    )

selected_adp_columns = [
    platform.adp_column for platform in selected_platforms
    if _coverage(platform.adp_column)
]
selected_rank_columns = [
    platform.rank_column for platform in selected_platforms
    if _coverage(platform.rank_column)
]

if not frame.empty:
    frame["avg_source_adp"] = _row_mean(frame, selected_adp_columns).round(1)
    frame["avg_source_rank"] = _row_mean(frame, selected_rank_columns).round(1)
    frame["adp_vs_blend"] = (frame["avg_source_adp"] - frame["overall_adp"]).round(1)
    frame["selected_adp_count"] = (
        frame[selected_adp_columns].notna().sum(axis=1) if selected_adp_columns
        else pd.Series(0, index=frame.index)
    )

if AVAILABLE_PLATFORMS:
    with st.expander("What each platform is, and how much of the pool it covers"):
        st.dataframe(
            pd.DataFrame([
                {
                    "Platform": platform.name,
                    "Has ADP for": _coverage(platform.adp_column),
                    "Has a rank for": _coverage(platform.rank_column),
                    "Selected": "✓" if platform in selected_platforms else "",
                    "What it is": platform.what_it_is,
                }
                for platform in AVAILABLE_PLATFORMS
            ]),
            width="stretch", hide_index=True,
        )
        st.caption(
            f"Out of {len(frame)} players. A source with a low count is not broken — it "
            "simply publishes a shorter list, which is why the average is taken over "
            "whichever sources have the player rather than over all of them. A zero in "
            "**Has a rank for** means that platform publishes draft position but no "
            "ranking, so it has no column in the Ranks view."
        )

sort_columns = st.columns([2, 1, 1, 1])
sort_options = {
    "ADP (earliest first)": ("overall_adp", True),
    "Projection (highest first)": ("projection", False),
    "Value over replacement": ("value_over_replacement", False),
    "Ceiling": ("ceiling", False),
    "Floor": ("floor", False),
    "Risk (safest first)": ("risk_score", True),
    "Risk (riskiest first)": ("risk_score", False),
    "Source disagreement": ("adp_disagreement", False),
    "Sources (most first)": ("adp_source_count", False),
    "Average ADP across sources": ("avg_source_adp", True),
    "Average rank across sources": ("avg_source_rank", True),
    "Consensus rank": ("overall_rank", True),
}
# Per-platform sorts for the selected platforms only, so deselecting a source removes it
# from here too rather than leaving an option that sorts by a column no longer shown.
for _column in selected_adp_columns:
    sort_options[f"{PLATFORM_COLUMN_LABELS[_column]} (earliest first)"] = (_column, True)
for _column in selected_rank_columns:
    sort_options[f"{PLATFORM_COLUMN_LABELS[_column]} (best first)"] = (_column, True)
if board.custom_ranks:
    # Offered only when there is something to sort by, so the option never appears
    # and then silently falls back to ADP.
    sort_options = {"My own ranking": ("my_rank", True), **sort_options}
sort_choice = sort_columns[0].selectbox("Sort by", list(sort_options))
row_limit = sort_columns[1].number_input(
    "Rows to show", min_value=10, max_value=1000, value=100, step=10
)
tiers = sorted({int(t) for t in frame["tier"].dropna().tolist()})
tier_choice = sort_columns[2].multiselect("Tiers", tiers)
column_set = sort_columns[3].radio(
    "Columns",
    ["Value", "ADP by platform", "Ranks by platform"],
    help="All three carry each selected platform's own number. **Value** is the board "
         "with projections, VOR and risk beside it. **ADP by platform** drops those to "
         "make room for the spread between sources. **Ranks by platform** shows what "
         "each site *says* rather than where players actually go.",
)

view = frame.copy()
# The board's own columns, added before filtering so they can be filtered on. Keyed by
# player id via the pool, because the board matches on name and the frame does not
# carry the match key.
if not board.is_empty:
    marks: dict[str, str] = {}
    my_ranks: dict[str, int] = {}
    for player in pool:
        priority = board.target_priority(player)
        if priority is not None:
            marks[player.player_id] = f"🎯 {priority}"
        elif board.is_avoided(player):
            marks[player.player_id] = "⛔"
        own = board.custom_rank(player)
        if own is not None:
            my_ranks[player.player_id] = own
    view["board_mark"] = view["player_id"].map(marks).fillna("")
    view["my_rank"] = view["player_id"].map(my_ranks)
    board_columns = st.columns([1, 1, 3])
    only_mine = board_columns[0].checkbox(
        "Only my targets", help="Show just the players on your target list."
    )
    hide_avoided = board_columns[1].checkbox(
        "Hide never-draft", value=True,
        help="Hide the players on your do-not-draft list. They are still on the real "
             "board — your opponents can and will take them.",
    )
    if only_mine:
        view = view[view["board_mark"].str.startswith("🎯")]
    if hide_avoided:
        view = view[view["board_mark"] != "⛔"]
if positions:
    wanted = {str(p) for p in positions}
    view = view[view["position"].astype(str).isin(wanted)]
if search.strip():
    view = view[view["player_name"].str.contains(search.strip(), case=False, na=False)]
if rookies_only:
    view = view[view["is_rookie"].fillna(False).astype(bool)]
if hide_injured:
    view = view[view["injury_status"].astype(str).str.lower().isin({"healthy", "", "nan"})]
if tier_choice:
    view = view[view["tier"].isin(tier_choice)]
if complete_only and selected_adp_columns and "selected_adp_count" in view.columns:
    view = view[view["selected_adp_count"] >= len(selected_adp_columns)]

sort_field, ascending = sort_options[sort_choice]
if sort_field not in view.columns:
    # A per-source column is absent when that source failed or was switched off. Say
    # so rather than falling back silently to a different ordering.
    st.info(
        f"Nothing in this pool carries **{sort_choice}**, so it is sorted by ADP "
        "instead. That source either failed to load or was not enabled on Setup."
    )
    sort_field, ascending = "overall_adp", True
# na_position="last" matters: a player with no ADP sorted to the top would look like
# the consensus first pick when in fact nothing is known about them.
view = view.sort_values(sort_field, ascending=ascending, na_position="last")

drafted: set[str] = set()
draft = state.draft()
if draft is not None:
    drafted = set(draft.drafted_ids)
    if st.checkbox(f"Hide the {len(drafted)} players already drafted", value=True):
        view = view[~view["player_id"].isin(drafted)]

st.caption(f"{len(view)} of {len(frame)} players match.")

display = view.head(int(row_limit)).rename(columns={
    "player_name": "Player", "position": "Pos", "nfl_team": "Team",
    "bye_week": "Bye", "tier": "Tier", "projection": "Proj",
    "overall_rank": "Rank", "position_rank": "PosRank", "platform_rank": "PlatRank",
    "overall_adp": "ADP", "adp_stdev": "ADP σ",
    "value_over_replacement": "VOR", "ceiling": "Ceiling", "floor": "Floor",
    "risk_score": "Risk", "is_rookie": "Rookie", "injury_status": "Injury",
    **PLATFORM_COLUMN_LABELS,
    "adp_source_count": "Sources", "adp_disagreement": "Disagreement",
    "avg_source_adp": "Avg ADP", "avg_source_rank": "Avg rank",
    "adp_vs_blend": "Avg − blend",
    "projection_source": "Projection from",
    "board_mark": "My list", "my_rank": "My rank",
})

# Headers for the selected platforms, in registry order so the columns do not reshuffle
# when the user changes the selection.
selected_adp_labels = [PLATFORM_COLUMN_LABELS[c] for c in selected_adp_columns]
selected_rank_labels = [PLATFORM_COLUMN_LABELS[c] for c in selected_rank_columns]

# Every view carries the per-platform columns, including the default one. They used to
# live only in the two "by platform" views, which meant the answer to "what does each
# site say about this player" was behind a radio button in the corner of the filter row —
# findable, but only if you already knew it was there.
#
# `Avg ADP` is deliberately *not* here, though it is two columns away in the ADP view.
# For any player only one source lists, a weighted blend and a plain mean are both that
# source's number, so the two columns are identical — and in this pool that is three
# quarters of the rows, because FFC and Yahoo publish roughly 215 players deep while
# ESPN goes 1000. A column that duplicates its neighbour on most rows reads as
# redundant, which is a fair reading. The comparison is real but it is a comparison, so
# it belongs in the view whose job is comparing.
VALUE_COLUMNS = [
    "Player", "Pos", "Team", "Bye", "Tier",
    "ADP", *selected_adp_labels, "Sources",
    "Proj", "VOR", "Ceiling", "Floor", "Risk", "Rookie", "Injury",
]
ADP_PLATFORM_COLUMNS = [
    "Player", "Pos", "Team", "Tier", "ADP", "Avg ADP", "Avg − blend",
    *selected_adp_labels, "Sources", "Disagreement", "ADP σ",
]
RANK_PLATFORM_COLUMNS = [
    "Player", "Pos", "Team", "Tier", "Rank", "Avg rank",
    *selected_rank_labels, "PosRank", "PlatRank", "Proj", "VOR",
]
wanted = {
    "Value": VALUE_COLUMNS,
    "ADP by platform": ADP_PLATFORM_COLUMNS,
    "Ranks by platform": RANK_PLATFORM_COLUMNS,
}[column_set]
if not board.is_empty:
    # Beside the name, where a marker is read rather than hunted for.
    wanted = [wanted[0], "My list", "My rank", *wanted[1:]]

if column_set == "Value" and selected_adp_labels:
    st.caption(
        "`ADP` is the blend the engine drafts against — a *weighted* one, so it is not "
        "the average of the columns beside it; "
        + ", ".join(f"`{label}`" for label in selected_adp_labels)
        + " are each platform's own number, unmodified. A blank cell means that "
        "platform never listed the player, not that it rates him badly. Switch to "
        "**ADP by platform** to see the blend against a plain average and the spread "
        "between sources."
    )
elif column_set == "ADP by platform":
    st.caption(
        "**ADP is where a player actually goes; rank is where a site says he should "
        "go.** They differ, and the gap is often the interesting part. `ADP` is the "
        "weighted blend the engine drafts against, `Avg ADP` is the plain unweighted "
        "mean of the columns to its right, and `Avg − blend` is the difference: "
        "positive means the blend has him going *earlier* than a straight average "
        "would, because it trusts the source that likes him more."
    )
    st.caption(
        "`Avg − blend` is **0.0 for any player only one source lists** — with one "
        "opinion a weighted blend and a plain mean are the same number. It is only "
        "meaningful where `Sources` is 2 or 3, which in practice means the top couple "
        "of hundred players, because FFC and Yahoo publish much shorter boards than "
        "ESPN. Sort by `Sources` to see those rows together."
    )
elif column_set == "Ranks by platform":
    st.caption(
        "Rankings, not draft position — a site's stated opinion rather than what its "
        "drafters do. `Rank` is the consensus ordering the engine uses, `Avg rank` is "
        "the plain mean of the per-site columns, `PosRank` is rank within the position "
        "and `PlatRank` is your league's own platform. A player whose rank is far "
        "better than his ADP is one the sites like more than the drafters do, which is "
        "exactly the kind of player who lasts a round longer than he should."
    )

st.dataframe(
    display[[c for c in wanted if c in display.columns]],
    width="stretch", hide_index=True, height=460,
    column_config={
        "My list": st.column_config.TextColumn(
            "My list",
            help="🎯 with its priority for a target, ⛔ for a player you have said "
                 "you will never draft.",
        ),
        "My rank": st.column_config.NumberColumn(
            "My rank", format="%d",
            help="Your own ranking, from the board above. Blank where you have not "
                 "ranked the player and the consensus order is used instead.",
        ),
        "Risk": st.column_config.ProgressColumn(
            "Risk", min_value=0.0, max_value=1.0, format="%.2f",
            help="0 = the room agrees and he is healthy, 1 = nobody can place him. "
                 "Mostly draft-pick disagreement, plus injury status and rookie status.",
        ),
        "ADP σ": st.column_config.NumberColumn(
            "ADP σ", format="%.1f",
            help="Spread of this player's draft position across real mock drafts. Wide "
                 "means the room disagrees, so his availability is less predictable — "
                 "and it is what sets his ceiling and floor.",
        ),
        "VOR": st.column_config.NumberColumn(
            "VOR", format="%.1f",
            help="Projected points above the last startable player at this position, "
                 "for your league's lineup.",
        ),
        "Sources": st.column_config.NumberColumn(
            "Sources", format="%d",
            help="How many ADP sources had this player. A blended ADP from one source "
                 "is a single opinion, not a consensus.",
        ),
        "ADP": st.column_config.NumberColumn(
            "ADP", format="%.1f",
            help="The blended average draft position the engine drafts against. A "
                 "*weighted* blend — Fantasy Football Calculator counts double, being "
                 "the only source built from real drafts — so it is deliberately not "
                 "the plain average of the columns beside it. The weights are editable "
                 "on **Settings**.",
        ),
        "FFC ADP": st.column_config.NumberColumn(
            "FFC ADP", format="%.1f",
            help="Fantasy Football Calculator's average draft position — where this "
                 "player actually went in real public mock drafts. The only column here "
                 "that reports behaviour rather than opinion.",
        ),
        "ESPN ADP": st.column_config.NumberColumn(
            "ESPN ADP", format="%.1f",
            help="Average pick across ESPN leagues. Blank means ESPN's list does not go "
                 "deep enough to include him.",
        ),
        "Yahoo ADP": st.column_config.NumberColumn(
            "Yahoo ADP", format="%.1f",
            help="Average pick across Yahoo leagues. Blank means Yahoo's list does not "
                 "include him.",
        ),
        "ESPN rank": st.column_config.NumberColumn(
            "ESPN rank", format="%.0f",
            help="ESPN's own published overall ranking — its stated opinion, which can "
                 "differ sharply from where ESPN's own drafters take him.",
        ),
        "Yahoo rank": st.column_config.NumberColumn(
            "Yahoo rank", format="%.0f",
            help="Yahoo's published overall ranking.",
        ),
        "Avg ADP": st.column_config.NumberColumn(
            "Avg ADP", format="%.1f",
            help="Plain unweighted mean of the per-source ADP columns — every source "
                 "that had him counted once. The `ADP` column is the *weighted* blend "
                 "the engine actually drafts against.",
        ),
        "Avg − blend": st.column_config.NumberColumn(
            "Avg − blend", format="%+.1f",
            help="Average ADP minus the blended ADP. Near zero means the sources agree "
                 "and the weighting is irrelevant. Large either way means the blend is "
                 "leaning on one source, and it is worth looking at which.",
        ),
        "Avg rank": st.column_config.NumberColumn(
            "Avg rank", format="%.1f",
            help="Plain mean of the per-site rankings. Sleeper's number is popularity "
                 "rather than a ranking, so it pulls this toward whoever is being "
                 "searched for right now.",
        ),
        "Rank": st.column_config.NumberColumn(
            "Rank", format="%.0f",
            help="The consensus overall ranking the engine uses for ordering.",
        ),
        "PosRank": st.column_config.NumberColumn(
            "PosRank", format="%d",
            help="Rank within the position — RB7, WR14. Often more useful than overall "
                 "rank, because you draft against a position's supply, not the pool's.",
        ),
        "PlatRank": st.column_config.NumberColumn(
            "PlatRank", format="%.0f",
            help="Your own league platform's ranking. The simulated managers estimated "
                 "to follow their platform's list are pulled toward this one.",
        ),
        "Disagreement": st.column_config.NumberColumn(
            "Disagreement", format="%.0f",
            help="Picks between the earliest and latest source ADP for this player. "
                 "Large means the platforms are telling genuinely different stories.",
        ),
        "Sleeper": st.column_config.NumberColumn(
            "Sleeper", format="%d",
            help="Sleeper's own search-popularity rank. Not an ADP — it is how much "
                 "attention the player is getting, which leads ADP rather than "
                 "reporting it.",
        ),
        "Ceiling": st.column_config.NumberColumn(
            "Ceiling", format="%.1f",
            help="What he would be worth if the room's optimists are right about where "
                 "he belongs. Not a forecast of his best possible season.",
        ),
        "Floor": st.column_config.NumberColumn(
            "Floor", format="%.1f",
            help="What he would be worth if the room's pessimists are right. Not a "
                 "forecast of his worst possible season.",
        ),
    },
)

# The stat line is the direct answer to "where did this projection come from", but it
# is far too wide for the table, so it lives in a per-player lookup instead.
detail_pool = view[view.get("projection_detail", pd.Series(dtype=str)).astype(str).str.len() > 0] \
    if "projection_detail" in view.columns else view.iloc[0:0]
if len(detail_pool):
    with st.expander("Break a projection down into its stat line"):
        choice = st.selectbox(
            "Player",
            detail_pool["player_name"].tolist()[: int(row_limit)],
            key="pool_projection_detail",
        )
        row = detail_pool[detail_pool["player_name"] == choice].iloc[0]
        st.metric(f"{choice} — projected points", f"{float(row['projection']):.1f}")
        st.caption(str(row.get("projection_source") or ""))
        st.markdown(f"**Projected stat line:** {row['projection_detail']}")
        st.caption(
            "These are ESPN's projected season totals. The points beside them are what "
            "those totals are worth under *your* scoring rules, not ESPN's — so a "
            "TE-premium or 6-point-passing-TD league gets a different number here than "
            "ESPN's own site would show."
        )
        band = str(row.get("outcome_band_source") or "")
        if band:
            st.markdown(
                f"**Ceiling {float(row['ceiling']):.1f} / floor "
                f"{float(row['floor']):.1f}** — {band}"
            )
        tier_note = str(row.get("tier_source") or "")
        if tier_note:
            st.markdown(f"**Tier {int(row['tier'])}** — {tier_note}")

components.download_frame(view, "Download filtered pool (CSV)", "player_pool.csv")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Shape of the pool
# ─────────────────────────────────────────────────────────────────────────────
shape_left, shape_right = st.columns(2)
with shape_left:
    components.position_bar_chart(
        {p: len(pool.by_position(p)) for p in Position if pool.by_position(p)},
        "Players by position",
    )
with shape_right:
    st.markdown("**Value over replacement by position**")
    st.caption(
        "How much the top of each position is worth over its own replacement level. "
        "The positions with the tallest bars are where an early pick buys the most, "
        "and this is the quantity the value lens ranks on."
    )
    top_vor = {}
    for position in Position:
        players = pool.by_position(position)
        if players:
            top_vor[position] = round(
                max(p.value_over_replacement or 0.0 for p in players), 1
            )
    components.position_bar_chart(top_vor, "Best VOR available at each position")

with st.expander("Tier structure, and where tiers come from"):
    st.markdown(
        "No source publishes tiers, so these are derived here. Within each position, "
        "players are sorted by projection and the gap to the next player is measured. "
        "A new tier starts wherever that gap is larger than the average gap **plus one "
        "standard deviation** of all the gaps at that position — that is, wherever the "
        "drop-off is unusual for that position rather than routine."
    )
    st.caption(
        "Two consequences worth knowing. Positions get different numbers of tiers, "
        "because the threshold is computed per position rather than fixed. And tiers "
        "move when your scoring does: change the scoring preset on **Setup** and the "
        "projections shift, so the gaps — and the tier breaks — shift with them. "
        "Tiers group players the engine treats as near-interchangeable, and a tier "
        "about to empty out is what drives the 'last chance' recommendation lens."
    )
    breakpoints = sorted({
        str(p.tier_source) for p in pool if p.tier_source
    })
    if breakpoints:
        st.markdown("**The threshold actually used, per position**")
        for line in breakpoints:
            st.caption(f"• {line}")
    tier_table = (
        frame.dropna(subset=["tier"])
        .assign(tier=lambda f: f["tier"].astype(int))
        .groupby(["tier", "position"])
        .size().unstack(fill_value=0).sort_index()
    )
    if tier_table.empty:
        st.warning("No tiers in this pool.")
    else:
        st.dataframe(tier_table, width="stretch")
