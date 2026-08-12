"""Simulations: run the draft to completion many times and study the spread.

The Draft Room answers "what should I do with this pick". This page answers a
different question: "given how this room drafts, what kind of roster do I end up
with?" A single mock draft cannot answer that, because a single mock is one draw from
a distribution — so everything here is reported as a range, and the mean is marked
rather than presented on its own.

Runs from wherever the current draft has reached, so it can be used both before pick
one and mid-draft to test a decision you have already made.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from engine.simulator import monte_carlo_draft
from models.database import session_scope
from services.repository import list_simulation_runs, save_simulation_run
from ui import components, state

LOGGER = logging.getLogger("fantasy_mock_draft.ui.simulations")

components.page_header(
    "🎲 Simulations",
    "Run the rest of the draft hundreds of times and look at the distribution.",
)
components.require(needs_draft=True)

league = state.league()
draft = state.draft()
profiles = state.profiles()
settings = state.settings()

user_slot = min(league.user_slots) if league.user_slots else 1

if not profiles:
    st.warning(
        "No manager profiles are built, so every opponent draws on the league-average "
        "prior and the spread below understates how differently your league's managers "
        "actually behave. Build them on **Manager Profiles**.",
        icon="⚠️",
    )

st.caption(
    f"Simulating from pick {draft.pick_index + 1} of "
    f"{league.config.team_count * league.config.rounds}. "
    f"{draft.pick_index} pick(s) already made are held fixed in every run."
)

control_columns = st.columns([1, 1, 1, 2])
runs = control_columns[0].number_input(
    "Simulations", min_value=20, max_value=2000,
    value=int(settings.monte_carlo_default_runs), step=20,
    help="Each one drafts the remainder of the board to completion. 200 gives a "
         "readable distribution; 1000 tightens the tails and takes noticeably longer.",
)
seed = control_columns[1].number_input(
    "Seed", min_value=0, max_value=2**31 - 1, value=int(draft.seed or 0),
    help="Fixing the seed makes the whole run reproducible.",
)
strategy = control_columns[2].selectbox(
    "Your picks are made by",
    ["The same model as the opponents", "Best available by ranking"],
    help=(
        "How your own seat drafts inside the simulation. The first uses your own "
        "estimated profile, so it reflects how you have actually drafted; the second "
        "is a neutral baseline to compare against."
    ),
)

if control_columns[3].button("Run simulations", type="primary"):
    progress_bar = st.progress(0.0, text="Starting…")

    def report_progress(done: int, total: int) -> None:
        progress_bar.progress(
            min(1.0, done / max(1, total)), text=f"{done} / {total} drafts simulated"
        )

    user_strategy = None
    if strategy == "Best available by ranking":
        # A deliberate baseline rather than a model: takes the top of the ranking
        # list every time, so the difference against the modelled strategy is
        # attributable to the strategy and nothing else.
        def user_strategy(simulated_state):  # noqa: ANN001 — engine-supplied type
            return simulated_state.best_available()

    with st.spinner(f"Running {runs} drafts…"):
        report = monte_carlo_draft(
            draft, profiles, simulations=int(runs), draft_slot=user_slot,
            user_strategy=user_strategy, seed=int(seed), progress=report_progress,
        )
    progress_bar.empty()
    st.session_state[state.K_MC_REPORT] = report
    LOGGER.info(
        "Monte Carlo: %d runs in %.1fs, mean starter points %.1f",
        report.simulations, report.elapsed_seconds, report.mean_starter_points,
    )

report = st.session_state.get(state.K_MC_REPORT)
if report is None:
    components.blocked("No simulations run yet. Press **Run simulations** above.")

if report.from_pick != draft.pick_index:
    st.warning(
        f"These results were computed from pick {report.from_pick + 1}, but the draft "
        f"has since moved to pick {draft.pick_index + 1}. Re-run to match the current "
        "board.",
        icon="⚠️",
    )

st.divider()
components.metric_row([
    ("Drafts simulated", report.simulations, ""),
    ("Mean starter points", f"{report.mean_starter_points:.1f}",
     "Projected points from your starting lineup only — bench depth is not counted."),
    ("Time taken", f"{report.elapsed_seconds:.1f}s", ""),
    ("Your slot", report.user_slot, ""),
])

# ─────────────────────────────────────────────────────────────────────────────
# Outcome distribution
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("How good does your roster end up?")
st.caption(
    "Projected starting-lineup points across every simulated draft. The width is the "
    "finding: a narrow distribution means your slot's outcome is largely settled by "
    "the room, a wide one means the picks you make from here matter a great deal."
)
components.histogram(
    report.starter_points, "Projected starter points", "Starter points"
)

points = sorted(report.starter_points)
if points:
    def percentile(fraction: float) -> float:
        return points[min(len(points) - 1, int(fraction * len(points)))]

    components.metric_row([
        ("Worst case (5th %ile)", f"{percentile(0.05):.0f}", ""),
        ("Typical (median)", f"{percentile(0.50):.0f}", ""),
        ("Good case (95th %ile)", f"{percentile(0.95):.0f}", ""),
        ("Spread", f"{percentile(0.95) - percentile(0.05):.0f}",
         "The range you should expect, not a margin of error."),
    ])

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Who you end up with
# ─────────────────────────────────────────────────────────────────────────────
frequency_tab, shape_tab, holes_tab, save_tab = st.tabs(
    ["Players you end up with", "Roster shape", "Unfilled slots", "Save this run"]
)

with frequency_tab:
    st.caption(
        "How often each player finishes on your roster. A player at 100% is one you "
        "will essentially always get at this slot; one at 40% is a genuine coin flip "
        "you should have a plan for."
    )
    if not report.player_frequency:
        st.info("No player frequencies recorded — the draft may already be complete.")
    else:
        frequency_rows = [
            {
                "Player": report.names.get(player_id, player_id),
                "On your roster": count / max(1, report.simulations),
                "Drafts": count,
            }
            for player_id, count in report.player_frequency.items()
        ]
        frequency_frame = pd.DataFrame(frequency_rows).sort_values(
            "On your roster", ascending=False
        )
        st.dataframe(
            frequency_frame.head(60), width="stretch", hide_index=True,
            height=440,
            column_config={
                "On your roster": st.column_config.ProgressColumn(
                    "On your roster", min_value=0.0, max_value=1.0, format="%.0f%%",
                ),
            },
        )
        components.download_frame(
            frequency_frame, "Download frequencies (CSV)", "player_frequency.csv"
        )

with shape_tab:
    st.caption(
        "Average number of each position on your finished roster. Compare against your "
        f"starting requirements ({league.config.roster.starters_total} starters) — a "
        "position sitting at exactly its starter count across every simulation means "
        "you never get depth there."
    )
    if report.position_shape:
        components.position_bar_chart(
            {
                position: round(total / max(1, report.simulations), 2)
                for position, total in report.position_shape.items()
            },
            "Mean players rostered per position",
        )
        demand = league.config.roster.starting_demand()
        st.dataframe(
            pd.DataFrame([
                {
                    "Position": str(position),
                    "Mean rostered": round(total / max(1, report.simulations), 2),
                    "Starters required": demand.get(position, 0),
                }
                for position, total in sorted(
                    report.position_shape.items(), key=lambda kv: str(kv[0])
                )
            ]),
            width="stretch", hide_index=True,
        )
    else:
        st.info("No positional shape recorded.")

with holes_tab:
    st.caption(
        "How often each starting slot finishes the draft empty. Anything above zero is "
        "a structural problem with the plan, not bad luck — it means the simulated "
        "draft ran out of rounds before that slot was filled."
    )
    if not report.open_starter_counts:
        st.success("Every starting slot was filled in every simulation.")
    else:
        st.dataframe(
            pd.DataFrame([
                {
                    "Slot": str(slot),
                    "Left empty": f"{count / max(1, report.simulations):.0%}",
                    "Simulations": count,
                }
                for slot, count in sorted(
                    report.open_starter_counts.items(), key=lambda kv: -kv[1]
                )
            ]),
            width="stretch", hide_index=True,
        )

with save_tab:
    st.caption(
        "Stores the summary metrics against this league so runs can be compared later."
    )
    run_name = st.text_input(
        "Name", value=f"{league.config.name} — {report.simulations} runs from pick "
                      f"{report.from_pick + 1}",
    )
    if st.button("Save run", type="primary"):
        results = [
            {
                "metric_kind": "starter_points",
                "subject": "user",
                "context": f"pick_{report.from_pick + 1}",
                "value": float(value),
                "payload": None,
            }
            for value in report.starter_points
        ] + [
            {
                "metric_kind": "player_frequency",
                "subject": report.names.get(player_id, player_id),
                "context": "user_roster",
                "value": count / max(1, report.simulations),
                "payload": None,
            }
            for player_id, count in report.player_frequency.items()
        ]
        with session_scope() as session:
            run_id = save_simulation_run(
                session, name=run_name.strip() or "Monte Carlo run",
                run_kind="monte_carlo", iterations=report.simulations,
                results=results, league_id=league.config.league_id,
                random_seed=int(seed), user_slot=report.user_slot,
                settings_snapshot=settings.to_dict(),
                duration_seconds=report.elapsed_seconds,
                notes="SAMPLE DATA — fictional players." if state.is_sample_data() else "",
            )
        st.success(f"Saved as run #{run_id}.")

    st.divider()
    st.markdown("**Previous runs**")
    with session_scope() as session:
        previous = list_simulation_runs(session, league.config.league_id)
    if previous:
        st.dataframe(pd.DataFrame(previous), width="stretch", hide_index=True)
    else:
        st.caption("Nothing saved yet.")
