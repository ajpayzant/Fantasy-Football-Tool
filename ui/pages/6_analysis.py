"""Analysis: review a draft after the fact.

Reviews either the draft currently in the Draft Room or any saved one. The useful
question after a draft is not "what is my projected total" — every tool shows that —
but "which picks did I get wrong, and what was the alternative". So the emphasis here
is on the reach/fall of each pick, where the value went relative to the room, and the
holes the roster finished with.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from models.database import session_scope
from services.repository import (
    delete_mock_draft,
    list_mock_drafts,
    load_mock_draft,
    load_mock_roster_payloads,
)
from ui import components, state

components.page_header(
    "📊 Analysis",
    "Review a finished draft pick by pick — reaches, falls, and what the rosters became.",
)
components.require()

league = state.league()
pool = state.pool()
draft = state.draft()

# ─────────────────────────────────────────────────────────────────────────────
# Choose what to review
# ─────────────────────────────────────────────────────────────────────────────
with session_scope() as session:
    saved = list_mock_drafts(session)

sources: dict[str, object] = {}
if draft is not None and draft.pick_index:
    sources[f"Current draft in the Draft Room ({draft.pick_index} picks)"] = "live"
for row in saved:
    sources[
        f"#{row['mock_id']} — {row['name']} ({row['pick_count']} picks, "
        f"saved {row['created_at']})"
    ] = row["mock_id"]

if not sources:
    components.blocked(
        "Nothing to review yet. Run a draft on **Draft Room** — you can review it "
        "part-finished, or save it there and come back."
    )

choice_label = st.selectbox("Draft to review", list(sources))
choice = sources[choice_label]

if choice == "live":
    picks = list(draft.picks)
    user_slots = set(league.user_slots)
    rosters = {slot: draft.roster_copy(slot) for slot in league.slots_in_order()}
    roster_payloads = []
    seed = draft.seed
    notes = ""
else:
    with session_scope() as session:
        result = load_mock_draft(session, int(choice))
        roster_payloads = load_mock_roster_payloads(session, int(choice))
    if result is None:
        components.blocked(f"Mock draft #{choice} could not be loaded.")
    picks = list(result.picks)
    user_slots = set(result.user_slots)
    rosters = {}
    seed = result.random_seed
    notes = result.notes
    if notes:
        st.caption(notes)

    delete_columns = st.columns([1, 4])
    if delete_columns[0].button("Delete this saved draft"):
        with session_scope() as session:
            delete_mock_draft(session, int(choice))
        components.flash(f"Deleted mock draft #{choice}.", "info")
        st.rerun()

if not picks:
    components.blocked("That draft has no picks in it.")

# ─────────────────────────────────────────────────────────────────────────────
# Headline
# ─────────────────────────────────────────────────────────────────────────────
frame = pd.DataFrame([
    {
        "overall": pick.overall_pick,
        "round": pick.round_number,
        "in_round": pick.pick_in_round,
        "slot": pick.draft_slot,
        "manager": pick.manager_name,
        "is_user": pick.draft_slot in user_slots,
        "player": pick.player_name,
        "position": str(pick.position) if pick.position else "",
        "team": pick.nfl_team or "",
        "adp": pick.adp_at_pick,
        "projection": pick.projection,
        "tier": pick.tier,
        "assigned_slot": str(pick.assigned_slot or ""),
        "explanation": pick.explanation,
    }
    for pick in picks
])
# Positive = taken earlier than the market expected. Computed here rather than read
# from the pick so it is consistent for saved drafts, whose ADP snapshot is the one
# taken at pick time.
frame["reach"] = frame["adp"] - frame["overall"]

user_frame = frame[frame["is_user"]]
components.metric_row([
    ("Picks in this draft", len(frame), ""),
    ("Rounds completed", int(frame["round"].max()), ""),
    ("Your picks", len(user_frame), ""),
    ("Seed", seed if seed is not None else "—",
     "Re-using this seed reproduces the same opponent behaviour."),
])

st.divider()

board_tab, your_tab, value_tab, rosters_tab = st.tabs(
    ["Full board", "Your draft", "Where the value went", "Every roster"]
)

# ─────────────────────────────────────────────────────────────────────────────
with board_tab:
    display = frame.assign(
        Pick=frame["round"].astype(str) + "." + frame["in_round"].map("{:02d}".format),
        You=frame["is_user"].map({True: "★", False: ""}),
    )[[
        "Pick", "overall", "manager", "You", "player", "position", "team",
        "adp", "reach", "tier", "assigned_slot", "explanation",
    ]].rename(columns={
        "overall": "Overall", "manager": "Manager", "player": "Player",
        "position": "Pos", "team": "Team", "adp": "ADP", "reach": "Reach",
        "tier": "Tier", "assigned_slot": "Filled", "explanation": "Why",
    })
    st.dataframe(display, width="stretch", hide_index=True, height=460)
    components.download_frame(display, "Download board (CSV)", "draft_analysis.csv")
    st.caption(
        "`Reach` is how many picks ahead of ADP the player went. Positive = the "
        "manager reached; negative = the player fell to them."
    )

# ─────────────────────────────────────────────────────────────────────────────
with your_tab:
    if user_frame.empty:
        st.info("No picks belong to your slot in this draft.")
    else:
        st.markdown("**Your picks, in order**")
        for row in user_frame.sort_values("overall").itertuples():
            reach = row.reach
            if pd.isna(reach):
                verdict = "no ADP recorded for this player"
            elif reach > 8:
                verdict = f"a reach of {reach:.0f} picks"
            elif reach < -8:
                verdict = f"fell {abs(reach):.0f} picks to you"
            else:
                verdict = "about where the market had them"
            with st.container(border=True):
                st.markdown(
                    f"**{row.round}.{row.in_round:02d}** (overall {row.overall}) — "
                    f"**{row.player}** · {row.position} "
                    f"{row.team} — {verdict}"
                )
                caption_bits = []
                if row.assigned_slot:
                    caption_bits.append(f"filled {row.assigned_slot}")
                if not pd.isna(row.projection):
                    caption_bits.append(f"projected {row.projection:.0f}")
                if not pd.isna(row.tier):
                    caption_bits.append(f"tier {int(row.tier)}")
                if row.explanation:
                    caption_bits.append(row.explanation)
                if caption_bits:
                    st.caption(" · ".join(caption_bits))

        st.divider()
        st.markdown("**Your positional order**")
        st.caption(
            "The shape of your own draft. Compare it against the archetypes on "
            "**Manager Profiles** — this is exactly what the estimator would read if "
            "this draft became part of your history."
        )
        components.position_bar_chart(
            user_frame["position"].value_counts().to_dict(), "Your picks by position"
        )
        mean_reach = user_frame["reach"].mean()
        if not pd.isna(mean_reach):
            st.caption(
                f"You reached an average of {mean_reach:+.1f} picks versus ADP across "
                f"{len(user_frame)} picks. The room averaged "
                f"{frame['reach'].mean():+.1f}."
            )

# ─────────────────────────────────────────────────────────────────────────────
with value_tab:
    st.caption(
        "Projected points drafted per team. This is a crude scoreboard — it counts "
        "every player equally, including the bench, and takes the projections at face "
        "value. It answers 'who accumulated the most projected production', not 'who "
        "will win'."
    )
    by_manager = (
        frame.groupby(["slot", "manager"], as_index=False)
        .agg(
            picks=("player", "count"),
            projection=("projection", "sum"),
            mean_reach=("reach", "mean"),
        )
        .sort_values("projection", ascending=False)
    )
    by_manager["You"] = by_manager["slot"].isin(user_slots).map({True: "★", False: ""})
    st.dataframe(
        by_manager.rename(columns={
            "slot": "Slot", "manager": "Manager", "picks": "Picks",
            "projection": "Total projection", "mean_reach": "Mean reach",
        })[["Slot", "Manager", "You", "Picks", "Total projection", "Mean reach"]],
        width="stretch", hide_index=True,
        column_config={
            "Total projection": st.column_config.NumberColumn(format="%.0f"),
            "Mean reach": st.column_config.NumberColumn(
                format="%+.1f", help="Positive = this manager tended to reach."
            ),
        },
    )

    st.markdown("**Biggest reaches and biggest falls**")
    reach_columns = st.columns(2)
    ranked = frame.dropna(subset=["reach"]).sort_values("reach", ascending=False)
    with reach_columns[0]:
        st.caption("Taken furthest ahead of ADP")
        st.dataframe(
            ranked.head(10)[["player", "position", "manager", "overall", "adp", "reach"]]
            .rename(columns={
                "player": "Player", "position": "Pos", "manager": "Manager",
                "overall": "Pick", "adp": "ADP", "reach": "Reach",
            }),
            width="stretch", hide_index=True,
        )
    with reach_columns[1]:
        st.caption("Fell furthest past ADP")
        st.dataframe(
            ranked.tail(10).iloc[::-1][
                ["player", "position", "manager", "overall", "adp", "reach"]
            ].rename(columns={
                "player": "Player", "position": "Pos", "manager": "Manager",
                "overall": "Pick", "adp": "ADP", "reach": "Reach",
            }),
            width="stretch", hide_index=True,
        )

    st.markdown("**When each position went**")
    st.caption(
        "Every pick plotted by round. Clusters are positional runs; a long empty "
        "stretch is where a position was there for the taking."
    )
    import plotly.express as express

    scatter = express.scatter(
        frame, x="overall", y="position", color="position",
        color_discrete_map=components.POSITION_COLOURS,
        hover_data=["player", "manager", "adp", "reach"],
        labels={"overall": "Overall pick", "position": ""},
    )
    scatter.update_layout(
        height=320, showlegend=False, margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(scatter, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
with rosters_tab:
    if roster_payloads:
        st.caption(
            "Final rosters as they were saved, with each team's projected starting "
            "lineup total."
        )
        summary = pd.DataFrame([
            {
                "Slot": payload["draft_slot"],
                "Manager": payload["manager_name"],
                "You": "★" if payload["is_user"] else "",
                "Starter projection": payload["starter_projection"],
                "Total projection": payload["total_projection"],
                "ADP value": payload["adp_value"],
            }
            for payload in roster_payloads
        ]).sort_values("Starter projection", ascending=False)
        st.dataframe(
            summary, width="stretch", hide_index=True,
            column_config={
                "Starter projection": st.column_config.NumberColumn(
                    format="%.0f",
                    help="Projected points from the starting lineup only — the number "
                         "that actually matters in a weekly-lineup league.",
                ),
                "ADP value": st.column_config.NumberColumn(
                    format="%+.0f",
                    help="Total picks of ADP surplus captured across the draft.",
                ),
            },
        )
        chosen = st.selectbox(
            "Show roster for",
            [payload["draft_slot"] for payload in roster_payloads],
            format_func=lambda slot: next(
                p["manager_name"] for p in roster_payloads if p["draft_slot"] == slot
            ),
        )
        payload = next(p for p in roster_payloads if p["draft_slot"] == chosen)
        st.json(payload["roster"], expanded=True)
    elif rosters:
        st.caption("Rosters as they currently stand in the Draft Room.")
        for slot in sorted(rosters):
            roster = rosters[slot]
            with st.expander(
                f"Slot {slot} — {roster.manager_name} "
                f"({len(roster)}/{league.config.roster.roster_size})"
                + (" ★ you" if slot in user_slots else "")
            ):
                lineup_rows = []
                for lineup_slot, player_ids in roster.lineup.items():
                    for player_id in player_ids:
                        player = pool.get(player_id) if pool else None
                        lineup_rows.append({
                            "Slot": str(lineup_slot),
                            "Player": player.name if player else player_id,
                            "Pos": str(player.position) if player else "",
                            "Proj": player.projection if player else None,
                            "Bye": player.bye_week if player else None,
                        })
                for player_id in roster.bench:
                    player = pool.get(player_id) if pool else None
                    lineup_rows.append({
                        "Slot": "BN",
                        "Player": player.name if player else player_id,
                        "Pos": str(player.position) if player else "",
                        "Proj": player.projection if player else None,
                        "Bye": player.bye_week if player else None,
                    })
                st.dataframe(
                    pd.DataFrame(lineup_rows), width="stretch", hide_index=True
                )
                open_slots = roster.open_starting_slots()
                if open_slots:
                    st.warning(
                        "Unfilled starting slots: "
                        + ", ".join(f"{slot_name} ×{n}" for slot_name, n in open_slots.items())
                    )
                byes = [
                    (pool.get(pid).bye_week if pool and pool.get(pid) else None)
                    for pid in roster.player_ids
                ]
                clashes = {
                    week: byes.count(week) for week in set(byes)
                    if week and byes.count(week) >= 4
                }
                if clashes:
                    st.caption(
                        "Bye-week concentration: "
                        + ", ".join(f"week {w}: {n} players" for w, n in sorted(clashes.items()))
                    )
    else:
        st.info("No roster detail was stored with this draft.")
