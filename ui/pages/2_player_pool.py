"""Player Pool: the board the whole app draws from, and where its numbers came from.

Two jobs. First, let the user find and sort players. Second — and the reason this page
is more than a table — show what the importer *derived*: value over replacement is
computed from your scoring settings, tiers and ranks may have been imputed, and a
recommendation that surprises you usually traces back to one of those.
"""

from __future__ import annotations

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
    "ESPN rank": ("espn_rank", True),
    "ESPN ADP": ("espn_adp", True),
    "Yahoo ADP": ("yahoo_adp", True),
    "Fantasy Football Calculator ADP": ("ffc_adp", True),
    "Sleeper popularity": ("sleeper_rank", True),
}
sort_choice = sort_columns[0].selectbox("Sort by", list(sort_options))
row_limit = sort_columns[1].number_input(
    "Rows to show", min_value=10, max_value=1000, value=100, step=10
)
tiers = sorted({int(t) for t in frame["tier"].dropna().tolist()})
tier_choice = sort_columns[2].multiselect("Tiers", tiers)
column_set = sort_columns[3].radio(
    "Columns",
    ["Value", "Every platform"],
    help="Value shows the blended board. Every platform shows what each source "
         "thinks, side by side, so you can see where they disagree.",
)

view = frame.copy()
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
    "ffc_adp": "FFC ADP", "espn_adp": "ESPN ADP", "espn_rank": "ESPN rank",
    "yahoo_adp": "Yahoo ADP", "sleeper_rank": "Sleeper",
    "adp_source_count": "Sources", "adp_disagreement": "Disagreement",
    "projection_source": "Projection from",
})

VALUE_COLUMNS = [
    "Player", "Pos", "Team", "Bye", "Tier", "ADP", "ADP σ", "Sources",
    "Proj", "VOR", "Ceiling", "Floor", "Risk", "Rookie", "Injury",
]
PLATFORM_COLUMNS = [
    "Player", "Pos", "Team", "Tier", "ADP", "Sources", "Disagreement",
    "FFC ADP", "ESPN ADP", "ESPN rank", "Yahoo ADP", "Sleeper", "Proj", "VOR",
]
wanted = VALUE_COLUMNS if column_set == "Value" else PLATFORM_COLUMNS
st.dataframe(
    display[[c for c in wanted if c in display.columns]],
    width="stretch", hide_index=True, height=460,
    column_config={
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
