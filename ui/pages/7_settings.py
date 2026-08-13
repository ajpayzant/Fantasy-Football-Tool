"""Settings: every weight and constant the model uses, in one editable place.

The point of exposing these is not tuning for its own sake — it is that a
recommendation you disagree with should be traceable to a number you can see and
change. Each control says what it does to behaviour, not just what it is called.

Changing anything here invalidates cached recommendations and rollouts, because they
were computed under the previous weights and there is no way for you to tell by
looking at them.
"""

from __future__ import annotations

import dataclasses
import json
import logging

import pandas as pd
import streamlit as st

from core.config import (
    ModelWeights,
    ProfileEstimationConfig,
    ShrinkageConfig,
    SimulationConfig,
)
from core.enums import RankingSource
from models.database import database_path
from ui import components, state

LOGGER = logging.getLogger("fantasy_mock_draft.ui.settings")

components.page_header(
    "🔧 Settings",
    "The model's weights and constants. Change one and re-run to see what it does.",
)

settings = state.settings()

WEIGHT_HELP: dict[str, str] = {
    "adp": "How much a simulated manager follows consensus draft position. Raise it "
           "and the whole room drafts closer to the ranking list.",
    "projection": "Pull toward raw projected points, independent of where the market "
                  "has the player.",
    "tier": "Preference for the top of a tier over the bottom of the one above.",
    "value_over_replacement": "Preference for positional scarcity as measured against "
                              "replacement level rather than raw points.",
    "roster_need": "How strongly an unfilled starting slot pulls a manager toward that "
                   "position. Zero makes everyone a pure best-player-available drafter.",
    "positional_scarcity": "Sensitivity to a position thinning out league-wide.",
    "manager_position_preference": "How much each manager's estimated positional lean "
                                   "matters. Zero erases the archetypes entirely.",
    "round_specific_preference": "How much their round-by-round habits matter — the "
                                 "'always takes a QB in round 2' effect.",
    "platform_rank_dependence": "Extra pull toward the platform's own ranking for "
                                "managers estimated to follow it.",
    "rookie_preference": "Strength of a manager's estimated rookie bias.",
    "favorite_team_preference": "Strength of a manager's estimated NFL-team bias.",
    "named_player_preference": "How hard a manager chases the specific players you "
                               "named for them on **Manager Profiles**, and avoids the "
                               "ones you said they will not touch. Does nothing until "
                               "you name someone.",
    "repeat_player_affinity": "How hard a manager chases a player they have drafted in "
                              "*previous seasons*. Measured in seasons, not picks, so "
                              "two mocks of one draft do not read as loyalty. Does "
                              "nothing until a draft history is imported.",
    "stack": "Preference for pairing a quarterback with their own receivers.",
    "handcuff": "Preference for backing up a running back they already own.",
    "positional_run": "How strongly a run in progress drags the next pick toward the "
                      "same position.",
    "expected_availability": "How much a manager takes scarcity-now over value-later.",
    "strategy": "Weight on the manager's overall archetype plan.",
    "randomness": "Irreducible noise. Zero makes every simulated draft identical given "
                  "the seed; high values wash out the profiles.",
    "injury_penalty": "How far an injury designation pushes a player down.",
    "roster_imbalance_penalty": "Penalty for stacking one position while others are "
                                "empty.",
    "positional_limit_penalty": "Penalty for exceeding a sensible cap at a position.",
}

ESTIMATION_HELP: dict[str, str] = {
    "reach_clip_picks": "Reaches larger than this are treated as data errors rather "
                        "than habits, so one absurd pick cannot define a manager.",
    "rank_delta_scale_picks": "The scale that converts 'picks away from rank' into a "
                              "0-1 rank-dependence score.",
    "predictability_scale_picks": "The scale that converts reach spread into a "
                                  "predictability score. Smaller = harsher.",
    "fill_rate_anchor": "The league-typical rate of filling a starting slot, used as "
                        "the prior for managers with thin history.",
    "run_continue_anchor": "League-typical rate of joining a positional run.",
    "tier_cliff_anchor": "League-typical rate of taking the last player in a tier.",
    "upside_anchor": "League-typical risk appetite. Above 0.5 = ceiling-chasing.",
    "position_bias_timing_scale": "How much *when* a manager takes a position counts "
                                  "toward their estimated lean.",
    "position_bias_share_scale": "How much *how often* they take it counts.",
    "position_bias_clip": "Cap on any positional lean, so one extreme season cannot "
                          "make a manager deterministic.",
    "early_rounds": "How many rounds count as 'early' for every early-round statistic "
                    "and archetype test.",
    "favorite_team_min_share": "Share of picks from one NFL team before it counts as a "
                               "bias rather than coincidence.",
    "user_preference_weight": "How much your own stated preferences override what your "
                              "history implies.",
    "run_window_picks": "How many recent picks count as 'the current window' when "
                        "detecting a run.",
    "run_threshold_picks": "How many same-position picks inside that window make it a run.",
}

SHRINKAGE_HELP: dict[str, str] = {
    "prior_strength": "Effective number of prior picks a manager's own data has to "
                      "outweigh. Higher = more conservative, everyone looks average "
                      "for longer.",
    "season_prior_strength": "Same idea, applied per season.",
    "league_share": "Of the prior, how much comes from this league's own average.",
    "platform_share": "How much comes from platform-wide behaviour.",
    "baseline_share": "How much comes from the general prior.",
    "recency_half_life_seasons": "Seasons after which a pick counts half as much. "
                                 "Lower = the model forgets faster.",
    "min_picks_for_observed": "Picks required before a statistic is labelled 'observed' "
                              "rather than a fallback.",
}

st.info(
    "These are the defaults from `core/config.py`. Nothing here is stored in the "
    "database — it applies to this session, so a reload returns to the defaults.",
    icon="ℹ️",
)

weights_tab, sim_tab, estimation_tab, shrinkage_tab, system_tab = st.tabs(
    ["Model weights", "Simulation", "Profile estimation", "Shrinkage", "System"]
)

# ─────────────────────────────────────────────────────────────────────────────
with weights_tab:
    st.caption(
        "These weights decide how a simulated manager scores each available player. "
        "They apply to the opponents *and* to the recommendation lenses, so raising "
        "`roster_need` makes the room draft for need and makes the roster-fit lens "
        "more insistent at the same time."
    )
    weight_values: dict[str, float] = {}
    columns = st.columns(2)
    for index, field in enumerate(dataclasses.fields(ModelWeights)):
        weight_values[field.name] = columns[index % 2].slider(
            field.name.replace("_", " "),
            min_value=0.0, max_value=3.0,
            value=float(getattr(settings.weights, field.name)),
            step=0.05, help=WEIGHT_HELP.get(field.name, ""),
            key=f"weight_{field.name}",
        )

# ─────────────────────────────────────────────────────────────────────────────
with sim_tab:
    st.caption(
        "How the simulator turns scores into picks. Temperature is the important one: "
        "it converts a manager's predictability into how sharply their pick "
        "probabilities concentrate."
    )
    sim_columns = st.columns(2)
    base_temperature = sim_columns[0].slider(
        "Base temperature", 0.05, 2.0, float(settings.base_temperature), 0.01,
        help="Low = simulated managers almost always take their top-scoring player. "
             "High = they spread across candidates. This is the single biggest lever "
             "on how varied the simulated drafts are.",
    )
    temperature_low, temperature_high = sim_columns[1].slider(
        "Temperature range", 0.05, 3.0,
        (float(settings.temperature_range[0]), float(settings.temperature_range[1])),
        0.01,
        help="The band the per-manager temperature is clamped into. The most "
             "predictable manager gets the low end, the most erratic the high end.",
    )
    candidate_pool_size = sim_columns[0].number_input(
        "Candidate pool size", min_value=5, max_value=200,
        value=int(settings.candidate_pool_size),
        help="How many players deep a simulated manager considers. Small values make "
             "the room mechanical; large values are slower and let implausible picks in.",
    )
    reach_scale_picks = sim_columns[1].number_input(
        "Reach scale (picks)", min_value=1.0, max_value=60.0,
        value=float(settings.reach_scale_picks), step=1.0,
        help="Converts a manager's estimated reach into a pick-position shift.",
    )
    adp_sigma_floor = sim_columns[0].number_input(
        "ADP spread floor", min_value=0.5, max_value=40.0,
        value=float(settings.adp_sigma_floor), step=0.5,
        help="Minimum uncertainty on any player's draft position. Prevents the "
             "simulator treating a consensus ADP as a certainty.",
    )
    adp_sigma_round_growth = sim_columns[1].number_input(
        "ADP spread growth per round", min_value=0.0, max_value=10.0,
        value=float(settings.adp_sigma_round_growth), step=0.1,
        help="How much less predictable later rounds are. Late-round ADP is genuinely "
             "much noisier than early-round ADP.",
    )
    availability_simulations = sim_columns[0].number_input(
        "Default availability simulations", min_value=20, max_value=2000,
        value=int(settings.availability_simulations), step=20,
        help="Rollouts used for survival odds in the Draft Room.",
    )
    monte_carlo_default_runs = sim_columns[1].number_input(
        "Default Monte Carlo runs", min_value=20, max_value=5000,
        value=int(settings.monte_carlo_default_runs), step=20,
    )

    st.markdown("**Which ranking the model treats as the market**")
    ranking_source = st.selectbox(
        "Ranking source", list(RankingSource),
        index=list(RankingSource).index(
            RankingSource.coerce(settings.ranking_source, RankingSource.BLEND)
        ),
        format_func=lambda source: str(source).replace("_", " ").title(),
        help="`Blend` mixes the sources below, which is the most robust choice when "
             "your import has some columns missing.",
    )
    blend_weights = dict(settings.blend_weights)
    if ranking_source is RankingSource.BLEND:
        blend_columns = st.columns(len(blend_weights))
        for column, key in zip(blend_columns, sorted(blend_weights)):
            blend_weights[key] = column.slider(
                key.replace("_", " "), 0.0, 1.0, float(blend_weights[key]), 0.05,
                key=f"blend_{key}",
            )
        total = sum(blend_weights.values())
        if abs(total - 1.0) > 0.01:
            st.caption(
                f"These sum to {total:.2f}. They are normalised when used, so the "
                "ratios are what matter, not the total."
            )

    st.markdown("**Reproducibility**")
    use_seed = st.checkbox(
        "Fix a global random seed", value=settings.random_seed is not None,
        help="Off, each simulation draws its own seed. On, everything is reproducible.",
    )
    random_seed = None
    if use_seed:
        random_seed = st.number_input(
            "Seed", min_value=0, max_value=2**31 - 1,
            value=int(settings.random_seed or 20260801),
        )

    st.markdown("**Temperature curve**")
    st.caption(
        "The mapping from a manager's estimated predictability to their pick "
        "temperature, under the values above. This is how a profile becomes behaviour."
    )
    preview = SimulationConfig(
        base_temperature=float(base_temperature),
        temperature_range=(float(temperature_low), float(temperature_high)),
    )
    st.line_chart(
        pd.DataFrame({
            "predictability": [i / 20 for i in range(21)],
            "temperature": [preview.temperature_for(i / 20) for i in range(21)],
        }).set_index("predictability"),
        height=220,
    )

# ─────────────────────────────────────────────────────────────────────────────
with estimation_tab:
    st.caption(
        "How raw picks become the numbers on **Manager Profiles**. These change what "
        "the estimator concludes, so after editing them the profiles must be rebuilt "
        "before anything else reflects the change."
    )
    estimation_values: dict[str, float] = {}
    estimation_columns = st.columns(2)
    for index, field in enumerate(dataclasses.fields(ProfileEstimationConfig)):
        current = getattr(settings.estimation, field.name)
        column = estimation_columns[index % 2]
        if isinstance(current, int) and not isinstance(current, bool):
            estimation_values[field.name] = column.number_input(
                field.name.replace("_", " "), min_value=0, max_value=200,
                value=int(current), help=ESTIMATION_HELP.get(field.name, ""),
                key=f"est_{field.name}",
            )
        else:
            estimation_values[field.name] = column.number_input(
                field.name.replace("_", " "), min_value=0.0, max_value=200.0,
                value=float(current), step=0.05,
                help=ESTIMATION_HELP.get(field.name, ""), key=f"est_{field.name}",
            )

# ─────────────────────────────────────────────────────────────────────────────
with shrinkage_tab:
    st.caption(
        "How much a manager's own history is trusted against the league average. This "
        "is what stops a manager with four observed picks from being modelled as a "
        "fanatic because two of them happened to be tight ends."
    )
    shrinkage_values: dict[str, float] = {}
    shrinkage_columns = st.columns(2)
    for index, field in enumerate(dataclasses.fields(ShrinkageConfig)):
        current = getattr(settings.shrinkage, field.name)
        column = shrinkage_columns[index % 2]
        if isinstance(current, int) and not isinstance(current, bool):
            shrinkage_values[field.name] = column.number_input(
                field.name.replace("_", " "), min_value=0, max_value=200,
                value=int(current), help=SHRINKAGE_HELP.get(field.name, ""),
                key=f"shr_{field.name}",
            )
        else:
            shrinkage_values[field.name] = column.number_input(
                field.name.replace("_", " "), min_value=0.0, max_value=200.0,
                value=float(current), step=0.05,
                help=SHRINKAGE_HELP.get(field.name, ""), key=f"shr_{field.name}",
            )
    share_total = (
        shrinkage_values["league_share"] + shrinkage_values["platform_share"]
        + shrinkage_values["baseline_share"]
    )
    if abs(share_total - 1.0) > 0.01:
        st.caption(
            f"The three prior shares sum to {share_total:.2f}. They are normalised in "
            "use, so only the ratio matters."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Apply
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
apply_columns = st.columns([1, 1, 3])
if apply_columns[0].button("Apply settings", type="primary", width="stretch"):
    updated = SimulationConfig(
        weights=ModelWeights(**weight_values),
        shrinkage=ShrinkageConfig(**shrinkage_values),
        estimation=ProfileEstimationConfig(**estimation_values),
        ranking_source=ranking_source,
        blend_weights=blend_weights,
        base_temperature=float(base_temperature),
        temperature_range=(float(temperature_low), float(temperature_high)),
        candidate_pool_size=int(candidate_pool_size),
        adp_sigma_floor=float(adp_sigma_floor),
        adp_sigma_round_growth=float(adp_sigma_round_growth),
        run_windows=settings.run_windows,
        availability_simulations=int(availability_simulations),
        monte_carlo_default_runs=int(monte_carlo_default_runs),
        reach_scale_picks=float(reach_scale_picks),
        random_seed=int(random_seed) if random_seed is not None else None,
    )
    state.set_settings(updated)
    LOGGER.info("Simulation settings updated from the UI")
    components.flash(
        "Settings applied. Cached recommendations and simulations were discarded — "
        "rebuild the profiles on **Manager Profiles** for the estimation and "
        "shrinkage changes to take effect."
    )
    st.rerun()

if apply_columns[1].button("Reset to defaults", width="stretch"):
    state.set_settings(SimulationConfig())
    components.flash("Settings reset to the defaults in `core/config.py`.", "info")
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
with system_tab:
    st.markdown("**Current settings as JSON**")
    st.caption(
        "The exact object the engine is using. This is what gets stored alongside a "
        "saved draft or simulation run, so a result can be traced back to the "
        "configuration that produced it."
    )
    st.code(json.dumps(settings.to_dict(), indent=2, default=str), language="json")

    st.markdown("**Storage**")
    st.caption(f"Database: `{database_path()}`")
    st.caption(
        "Leagues, draft history, player pools, saved mocks and simulation runs live "
        "here. It is a local SQLite file — deleting it resets the app to empty."
    )

    st.markdown("**Session provenance**")
    provenance = state.provenance()
    st.dataframe(
        pd.DataFrame([
            {"What": "League", "From": provenance.league_source},
            {"What": "Player pool", "From": provenance.pool_source},
            {"What": "Draft history", "From": provenance.history_source},
            {
                "What": "Sample data",
                "From": "yes — data is fictional" if state.is_sample_data() else "no",
            },
        ]),
        width="stretch", hide_index=True,
    )
