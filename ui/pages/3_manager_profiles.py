"""Manager Profiles: what the engine believes about each opponent, and why.

This is the page that makes the app more than a ranking list, so it is built to be
argued with rather than trusted. Every modelled number is shown next to its
provenance — observed in that manager's own drafts, shrunk toward the league, or a
bare prior — because a parameter estimated from four picks and one estimated from
forty behave identically in the simulator and should not look identical here.

Archetype labels are shown as inferred summaries, never as ground truth: a real
league has no answer key. The estimator's accuracy is demonstrated against the
synthetic league in the test suite (``scripts/check_sample_archetypes.py``), which
is the only place ground truth exists.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import pandas as pd
import streamlit as st

from core.constants import NFL_TEAMS
from core.enums import Archetype, Position, ProvenanceKind
from engine.features import annotate_history
from engine.opponent_model import build_profiles, observe_manager
from models.league import League
from models.manager import Manager, ManagerPreferences
from services.normalize import player_key
from ui import components, state

LOGGER = logging.getLogger("fantasy_mock_draft.ui.profiles")


_NO_OPINION = "no opinion"


def _tendency_stop_label(stop: object) -> str:
    """Label a tendency stop: the no-opinion end reads as words, the rest as a %."""
    return _NO_OPINION if stop == _NO_OPINION else f"{float(stop):.0%}"


def _unmatched_names(names: list[str], player_pool) -> list[str]:
    """Names the user typed that match nobody on the board.

    Reported rather than dropped: a typo in a name silently does nothing, and the
    user would keep believing they had told the model something.
    """
    if not names or player_pool is None:
        return list(names)
    known = {player_key(player.name) for player in player_pool}
    return [name for name in names if player_key(name) not in known]

components.page_header(
    "🕵️ Manager Profiles",
    "What the engine infers about each opponent from your league's draft history.",
)
components.require()

league = state.league()
pool = state.pool()
history = state.history()

PROVENANCE_LABELS = {
    ProvenanceKind.OBSERVED: "🟢 observed in their own drafts",
    ProvenanceKind.MODEL_INFERRED: "🔵 model estimate",
    ProvenanceKind.USER_ENTERED: "✏️ you entered this",
    ProvenanceKind.LEAGUE_FALLBACK: "🟡 league average (too little history)",
    ProvenanceKind.PLATFORM_FALLBACK: "🟠 platform average",
    ProvenanceKind.BASELINE: "⚪ general prior (no history at all)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────
if not history.drafts:
    st.warning(
        "No draft history is loaded, so nothing has been *observed* about anybody. "
        "There are two ways to get a room worth drafting against, and they combine: "
        "**describe the managers yourself** below, or **import past drafts** on "
        "**Setup** (paste a draft recap or upload a CSV) and let the model infer them. "
        "With neither, all twelve opponents are the same league-average drafter.",
        icon="⚠️",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Build the room by hand
#
# The estimator can only describe what is in a history file, and plenty of leagues have
# no history file — a new league, a public draft, a first year on this app. Without this
# section those users get twelve identical opponents, which is worse than useless
# because it looks like a modelled room and behaves like one drafter copied twelve
# times. A declared archetype per seat is thin evidence, but it is *evidence*, and it
# goes through exactly the same preference path as everything else on this page: stored
# on ``Manager.preferences``, read by ``build_profiles``, and overwritten by real picks
# the moment any are imported.
# ─────────────────────────────────────────────────────────────────────────────
_NO_ARCHETYPE = "(let the model infer it)"
_NO_TEAM = "(none)"
_EXPERIENCE_LEVELS = ["new", "average", "veteran", "expert"]

with st.expander(
    "👥 Build the room by hand — name every manager and say how they draft",
    expanded=not history.drafts,
):
    st.caption(
        "One row per seat. Names are what historical picks are joined on, so spelling "
        "them the way your league spells them is what lets an import later attach to "
        "the right person. Everything else here is optional: leave *how they draft* "
        "alone and that manager stays whatever the model infers."
    )
    st.caption(
        "This is the from-scratch route. If you have real drafts, the other route is "
        "better — **Setup → Import past drafts** reads a pasted recap or a CSV and the "
        "model builds each profile from the picks themselves, which you can then adjust "
        "here exactly like anything you typed."
    )

    _slots = list(range(1, int(league.config.team_count) + 1))
    _seated = {int(m.draft_slot): m for m in league.managers}
    room_frame = st.data_editor(
        pd.DataFrame({
            "Slot": _slots,
            "Manager": [
                (_seated[s].name if s in _seated else f"Slot {s}") for s in _slots
            ],
            "You": [
                bool(_seated[s].is_user) if s in _seated else False for s in _slots
            ],
            "How they draft": [
                (
                    str(_seated[s].preferences.typical_strategy)
                    if s in _seated and _seated[s].preferences.typical_strategy
                    else _NO_ARCHETYPE
                )
                for s in _slots
            ],
            "Experience": [
                (
                    _seated[s].preferences.experience_level
                    if s in _seated
                    and _seated[s].preferences.experience_level in _EXPERIENCE_LEVELS
                    else "average"
                )
                for s in _slots
            ],
            "Fan of": [
                (
                    (_seated[s].preferences.favorite_nfl_team or _NO_TEAM).upper()
                    if s in _seated else _NO_TEAM
                )
                for s in _slots
            ],
        }),
        width="stretch", hide_index=True, disabled=["Slot"], key="room_editor",
        column_config={
            "You": st.column_config.CheckboxColumn(
                "You", help="Which seat you are drafting from. Exactly one.",
            ),
            "How they draft": st.column_config.SelectboxColumn(
                "How they draft",
                options=[_NO_ARCHETYPE] + [str(a) for a in Archetype],
                help="A declared archetype. It sets their positional leans and timing "
                     "directly, which is the only thing that makes a history-less room "
                     "contain twelve different drafters instead of twelve copies.",
            ),
            "Experience": st.column_config.SelectboxColumn(
                "Experience", options=_EXPERIENCE_LEVELS,
                help="Used only where nothing was observed about how predictable they "
                     "are: an expert drafts closer to the list, a new manager wanders.",
            ),
            "Fan of": st.column_config.SelectboxColumn(
                "Fan of", options=[_NO_TEAM] + list(NFL_TEAMS),
                help="Homer bias — raises their chance of taking that team's players.",
            ),
        },
    )

    with st.expander("What each archetype does when you pick it"):
        st.dataframe(
            pd.DataFrame([
                {
                    "Archetype": str(archetype).replace("_", " ").title(),
                    "How they behave": components.archetype_caption(archetype),
                }
                for archetype in Archetype
            ]),
            width="stretch", hide_index=True,
        )

    if st.button("Save the room and rebuild", type="primary", key="save_room"):
        rows = room_frame.to_dict("records")
        user_rows = [row for row in rows if bool(row["You"])]
        if len(user_rows) != 1:
            # Refused rather than guessed. Which seat is yours decides every survival
            # number on the Draft Room page, and silently picking one would make all of
            # them quietly wrong.
            st.error(
                "Tick exactly one seat as **You** — "
                + (
                    "no seat is ticked." if not user_rows
                    else f"{len(user_rows)} are ticked."
                )
                + " Which seat you draft from decides every 'will he last' number in "
                "the Draft Room."
            )
        else:
            new_managers = []
            for row in rows:
                slot = int(row["Slot"])
                existing = _seated.get(slot)
                # ``replace`` rather than a fresh object: this editor covers three of
                # the fourteen preference fields, and rebuilding from scratch would
                # silently discard the named players and positional notes entered in
                # the detail section below.
                base = existing.preferences if existing else ManagerPreferences()
                strategy = row["How they draft"]
                new_managers.append(Manager(
                    name=str(row["Manager"]).strip() or f"Slot {slot}",
                    draft_slot=slot,
                    is_user=bool(row["You"]),
                    manager_id=existing.manager_id if existing else None,
                    preferences=replace(
                        base,
                        typical_strategy=(
                            None if strategy == _NO_ARCHETYPE else strategy
                        ),
                        experience_level=str(row["Experience"]),
                        favorite_nfl_team=(
                            None if row["Fan of"] == _NO_TEAM else str(row["Fan of"])
                        ),
                    ),
                ))
            rebuilt = League(config=league.config, managers=new_managers)
            rebuilt.set_user_slot(int(user_rows[0]["Slot"]))
            report = rebuilt.validate()
            for issue in report.warnings:
                st.caption(f"Room: {issue.message}")
            if not report.ok:
                for issue in report.errors:
                    st.error(issue.message)
            else:
                state.set_league(rebuilt, source="managers entered by hand")
                if history.drafts:
                    annotate_history(history, pool=pool, roster=league.config.roster)
                state.set_profiles(
                    build_profiles(
                        rebuilt, history if history.drafts else None,
                        settings=state.settings(), pool=pool, annotate=False,
                    )
                )
                LOGGER.info("Room rebuilt by hand: %d managers", len(new_managers))
                named = sum(
                    1 for row in rows if row["How they draft"] != _NO_ARCHETYPE
                )
                components.flash(
                    f"Saved {len(new_managers)} managers and rebuilt their profiles"
                    + (
                        f" — {named} of them with a strategy you declared."
                        if named else
                        ". Nobody has a declared strategy yet, so they are still "
                        "twelve versions of the same drafter — set *how they draft* "
                        "above, or import real drafts on **Setup**."
                    )
                )
                st.rerun()

build_columns = st.columns([1, 1, 2])
if build_columns[0].button("Build profiles", type="primary", width="stretch"):
    with st.spinner("Estimating manager profiles…"):
        # Annotating stamps each historical pick with its context — rank inversions,
        # roster state, run position — which is what the estimators read. Doing it
        # here rather than inside build_profiles keeps the annotated history in
        # session state for the per-manager pick tables further down.
        if history.drafts:
            annotate_history(history, pool=pool, roster=league.config.roster)
        profiles = build_profiles(
            league, history if history.drafts else None,
            settings=state.settings(), pool=pool, annotate=False,
        )
    state.set_profiles(profiles)
    LOGGER.info("Built %d manager profiles", len(profiles))
    components.flash(f"Built {len(profiles)} profiles.")
    st.rerun()

profiles = state.profiles()
if not profiles:
    components.blocked(
        "Profiles have not been built yet. Press **Build profiles** above — with the "
        "sample league loaded this takes a second or two."
    )

if build_columns[1].button("Clear", width="stretch"):
    state.set_profiles({})
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# League-wide view
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("The room")

overview_rows = []
for slot in sorted(profiles):
    profile = profiles[slot]
    manager = league.manager_by_slot(slot)
    overview_rows.append({
        "Slot": slot,
        "Manager": profile.manager_name,
        "You": "★" if (manager is not None and manager.is_user) else "",
        "Inferred archetype": str(profile.archetype) if profile.archetype else "—",
        "Reach (picks vs ADP)": round(profile.reach_mean, 1),
        "Reach spread": round(profile.reach_stdev, 1),
        "Predictability": round(profile.get("predictability"), 2),
        "Rookie rate": round(profile.get("rookie_rate"), 2),
        "Run-chasing": round(profile.get("run_chase"), 2),
        "Rank dependence": round(profile.rank_dependence, 2),
        "Picks seen": int(profile.sample_picks),
        "Seasons": int(profile.sample_drafts),
    })
overview = pd.DataFrame(overview_rows)

st.dataframe(
    overview, width="stretch", hide_index=True,
    column_config={
        "Predictability": st.column_config.ProgressColumn(
            "Predictability", min_value=0.0, max_value=1.0, format="%.2f",
            help="1 = you can call their pick. 0 = anything could happen. Drives how "
                 "sharply the simulator concentrates their pick probabilities.",
        ),
        "Reach (picks vs ADP)": st.column_config.NumberColumn(
            "Reach", format="%.1f",
            help="Positive means they draft players before ADP says they should go.",
        ),
        "Rookie rate": st.column_config.NumberColumn(
            "Rookie rate", format="%.2f",
            help="Share of their picks spent on rookies.",
        ),
        "Run-chasing": st.column_config.NumberColumn(
            "Run-chasing", format="%.2f",
            help="How often they join a positional run already in progress.",
        ),
    },
)
components.download_frame(overview, "Download profiles (CSV)", "manager_profiles.csv")

thin = overview[overview["Picks seen"] < 8]
if len(thin):
    st.caption(
        "⚠️ "
        + ", ".join(thin["Manager"])
        + " have fewer than eight observed picks. Their parameters are mostly the "
        "league average regardless of what the table shows — shrinkage is doing the "
        "work, not evidence."
    )

single = overview[overview["Seasons"] == 1]
if len(single) and len(single) == len(overview):
    st.caption(
        "Every manager here has one draft on record. Sixteen picks from a single "
        "August are one afternoon's mood as much as a personality, so they are "
        "counted at **half** strength — a second season raises that to two thirds, "
        "a third to three quarters. The table still shows what those picks say; the "
        "simulator just leans on the league average more than the numbers suggest."
    )
elif len(single):
    st.caption(
        ", ".join(single["Manager"])
        + " have one draft on record, so their picks count at half strength — "
        "correlated picks from a single August are weaker evidence than the same "
        "count spread over three seasons."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Confidence in these profiles
# ─────────────────────────────────────────────────────────────────────────────
# The designed-vs-inferred answer key that used to live here needed the synthetic
# league, which the app no longer loads — that comparison now runs in the test
# suite (``scripts/check_sample_archetypes.py`` and ``tests/test_sample_data.py``),
# where it belongs. Against a real league there is no ground truth to show, so
# showing a table implying one would be dishonest.
if any(p.archetype for p in profiles.values()):
    st.caption(
        "Archetype labels are inferred from picks, not declared by the manager. "
        "They are a summary of observed behaviour, and the parameters above are what "
        "the simulator actually uses — a label with few picks behind it is mostly the "
        "league average wearing a name."
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# One manager in detail
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("One manager in detail")

chosen_slot = st.selectbox(
    "Manager", sorted(profiles),
    format_func=lambda s: f"Slot {s} — {profiles[s].manager_name}",
)
profile = profiles[chosen_slot]

if profile.archetype:
    st.markdown(
        f"### {profile.manager_name} — *{profile.archetype}*  \n"
        f"{components.archetype_caption(profile.archetype)}"
    )
else:
    st.markdown(
        f"### {profile.manager_name}  \n"
        "*No archetype inferred* — either too few picks to test against, or no "
        "threshold was crossed. They will be simulated on the league-average "
        "tendencies shown below."
    )

components.metric_row([
    ("Picks observed", int(profile.sample_picks),
     "Weighted toward recent seasons, so this can be fractional in the model."),
    ("Seasons", int(profile.sample_drafts), ""),
    ("Reach vs ADP", f"{profile.reach_mean:+.1f}",
     "Positive = drafts players earlier than the market."),
    ("Predictability", f"{profile.get('predictability'):.2f}",
     "How tightly the simulator concentrates their pick probability."),
])

detail_left, detail_right = st.columns(2)

with detail_left:
    st.markdown("**Modelled parameters and where each came from**")
    st.caption(
        "'Observed' means their own picks drove it. 'League average' means there was "
        "not enough of their history to say, so the room's average was used — the "
        "simulator cannot tell the difference, but you should be able to."
    )
    parameter_rows = []
    for key in sorted(profile.values):
        value = profile.values[key]
        parameter_rows.append({
            "Parameter": key.replace("_", " "),
            "Used": round(float(value.value), 3),
            "Their own data said": (
                round(value.observed_value, 3)
                if value.observed_value is not None else None
            ),
            "Personalised": value.manager_weight,
            "Based on": PROVENANCE_LABELS.get(value.provenance, str(value.provenance)),
        })
    st.dataframe(
        pd.DataFrame(parameter_rows), width="stretch", hide_index=True,
        height=380,
        column_config={
            "Personalised": st.column_config.ProgressColumn(
                "Personalised", min_value=0.0, max_value=1.0, format="%.0f%%",
                help="Weight given to this manager's own history versus the league "
                     "average. Low means the number shown is mostly the league's.",
            ),
            "Their own data said": st.column_config.NumberColumn(
                "Raw estimate", format="%.3f",
                help="Before shrinkage. A big gap from 'Used' means thin evidence.",
            ),
        },
    )

with detail_right:
    st.markdown("**Positional lean**")
    st.caption(
        "Not a share — a signed lean. Positive means the simulator raises this "
        "manager's probability of taking the position relative to the room, negative "
        "means it lowers it. Zero means they behave like everyone else there."
    )
    if profile.early_round_position_bias:
        components.position_bar_chart(
            {p: round(v, 3) for p, v in profile.early_round_position_bias.items()},
            "Early rounds (1-3)",
        )
    if profile.position_bias:
        components.position_bar_chart(
            {p: round(v, 3) for p, v in profile.position_bias.items()},
            "Whole draft",
        )
    else:
        st.caption("No positional lean estimated — not enough history.")

    # Only the teams they actually favour. The model stores a share for every team a
    # manager has drafted from, and printing all 32 would bury the one that matters:
    # an even spread is 1/32 ≈ 0.03, so anything at double that is a real lean.
    notable_teams = {
        team: share for team, share in profile.favorite_teams.items() if share >= 0.10
    }
    if notable_teams:
        st.markdown("**NFL teams they over-draft**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Team": team, "Share of their picks": f"{share:.0%}"}
                    for team, share in sorted(notable_teams.items(), key=lambda kv: -kv[1])
                ]
            ),
            width="stretch", hide_index=True,
        )
    if profile.repeat_players:
        st.markdown("**Players they draft again and again**")
        st.caption(
            ", ".join(
                f"{name} ({seasons} seasons)"
                for name, seasons in sorted(
                    profile.repeat_players.items(), key=lambda kv: (-kv[1], kv[0])
                )
            )
        )
        st.caption(
            "Counted in separate seasons, not picks, so this is a habit rather than one "
            "memorable draft. The simulator now pulls this manager toward these players "
            "when they are on the board — the strength is the *repeat player affinity* "
            "weight on **Settings**."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Tell the model what you know about them
# ─────────────────────────────────────────────────────────────────────────────
# The whole point of this section: the estimator can only ever describe what is in
# the history file, and a real user knows things it does not — that Dave took a kicker
# in round 9 last year because he forgot, or that Jen will not draft a Cowboy. Those
# statements live on ``Manager.preferences``, which ``build_profiles`` already reads
# on every rebuild, so they are not overwritten the next time profiles are built.
chosen_manager = league.manager_by_slot(chosen_slot)
if chosen_manager is None:
    st.caption(
        f"Slot {chosen_slot} has a profile but no manager in the league, so there is "
        "nothing to attach your own notes to."
    )
else:
    stated = chosen_manager.preferences
    with st.expander(
        f"Tell the model what you know about {profile.manager_name}"
        + (" — you have already entered something" if stated.has_any else ""),
        expanded=False,
    ):
        st.caption(
            "Anything you set here is treated as strong evidence, not as gospel: it is "
            f"blended with their observed picks at "
            f"{state.settings().estimation.user_preference_weight:.0%} weight, so a "
            "manager with a lot of history still keeps most of what their drafts show. "
            "Leave a slider at *no opinion* and the model is left entirely alone."
        )

        with st.form(f"preferences_{chosen_slot}"):
            identity = st.columns(3)
            strategy_options = ["(let the model infer it)"] + [str(a) for a in Archetype]
            current_strategy = (
                str(stated.typical_strategy) if stated.typical_strategy else strategy_options[0]
            )
            strategy_choice = identity[0].selectbox(
                "Their usual strategy",
                strategy_options,
                index=strategy_options.index(current_strategy),
                format_func=lambda s: s.replace("_", " ").title() if s in {
                    str(a) for a in Archetype
                } else s,
                help="Overrides the inferred archetype outright. Use it when you know "
                     "how they draft better than four rounds of history does.",
            )
            experience_options = ["new", "average", "veteran", "expert"]
            experience = identity[1].selectbox(
                "How experienced are they",
                experience_options,
                index=experience_options.index(
                    stated.experience_level if stated.experience_level in experience_options
                    else "average"
                ),
                help="Only used when nothing was observed about how predictable they "
                     "are — an expert drafts closer to the list, a new manager wanders.",
            )
            team_options = ["(none)"] + list(NFL_TEAMS)
            current_team = (stated.favorite_nfl_team or "").upper() or "(none)"
            favourite_team = identity[2].selectbox(
                "NFL team they are a fan of",
                team_options,
                index=team_options.index(current_team) if current_team in team_options else 0,
                help="Homer bias. Raises their chance of taking that team's players.",
            )

            st.markdown("**Tendencies**")
            st.caption(
                "Each slider starts at *no opinion*, the leftmost stop. Drag one off "
                "it and it becomes evidence about that manager; leave it there and the "
                "estimate stands."
            )
            slider_specs = [
                ("risk_tolerance", "Risk appetite",
                 "0 = takes the safe floor, 1 = swings for the ceiling."),
                ("rookie_preference", "Appetite for rookies",
                 "Share of picks you would expect them to spend on rookies."),
                ("rank_reliance", "Follows the rankings",
                 "0 = drafts on their own read, 1 = takes whoever is at the top."),
                ("predictability", "Predictability",
                 "1 = you can call their pick before they make it."),
                ("stack_preference", "Stacks a QB with his receivers", ""),
                ("handcuff_preference", "Backs up his own running backs", ""),
            ]
            slider_values: dict[str, float | None] = {}
            slider_columns = st.columns(3)
            for index, (attribute, label, help_text) in enumerate(slider_specs):
                column = slider_columns[index % 3]
                saved_value = getattr(stated, attribute)
                # One widget, with *no opinion* as its own stop, because 0.0 is a
                # meaningful answer ("never takes a rookie") and must stay distinct
                # from "I have nothing to say" — which is the ``None`` the engine
                # reads as leave-the-estimate-alone. This was previously a checkbox
                # gating a disabled slider, which cannot work inside a form: nothing
                # inside a form reruns until it is submitted, so ticking the box left
                # the slider frozen until after a save.
                stops: list[object] = [_NO_OPINION] + [
                    round(step * 0.05, 2) for step in range(21)
                ]
                if saved_value is not None and float(saved_value) not in stops:
                    # A value from elsewhere (an import, a finer-grained edit) keeps
                    # its own stop rather than being rounded off behind the user.
                    stops = [_NO_OPINION] + sorted(
                        [float(saved_value)] + [s for s in stops[1:]]
                    )
                value = column.select_slider(
                    label, options=stops,
                    value=_NO_OPINION if saved_value is None else float(saved_value),
                    format_func=_tendency_stop_label,
                    key=f"pref_val_{chosen_slot}_{attribute}",
                    help=help_text or None,
                )
                slider_values[attribute] = (
                    None if value == _NO_OPINION else float(value)
                )

            st.markdown("**Positions**")
            position_columns = st.columns(2)
            preferred = position_columns[0].multiselect(
                "Positions they load up on", list(Position),
                default=[p for p in stated.preferred_positions if p in list(Position)],
            )
            avoided = position_columns[1].multiselect(
                "Positions they avoid", list(Position),
                default=[p for p in stated.avoided_positions if p in list(Position)],
            )

            st.markdown("**Specific players**")
            st.caption(
                "One name per line. Matched the same forgiving way as an uploaded "
                "file, so \"AJ Brown\" finds \"A.J. Brown\". A name that matches "
                "nobody on the board is reported back to you rather than ignored."
            )
            player_columns = st.columns(2)
            favourites_text = player_columns[0].text_area(
                "Players they always want", value="\n".join(stated.favorite_players),
                height=110,
            )
            disliked_text = player_columns[1].text_area(
                "Players they will not touch", value="\n".join(stated.disliked_players),
                height=110,
            )
            notes = st.text_area(
                "Notes (for you — the model does not read these)",
                value=stated.notes, height=68,
            )

            saved_preferences = st.form_submit_button(
                "Save what I know", type="primary"
            )

        if saved_preferences:
            favourite_names = [n.strip() for n in favourites_text.splitlines() if n.strip()]
            disliked_names = [n.strip() for n in disliked_text.splitlines() if n.strip()]
            updated = ManagerPreferences(
                favorite_nfl_team=None if favourite_team == "(none)" else favourite_team,
                favorite_players=favourite_names,
                disliked_players=disliked_names,
                preferred_positions=list(preferred),
                avoided_positions=list(avoided),
                typical_strategy=(
                    None if strategy_choice == strategy_options[0] else strategy_choice
                ),
                experience_level=experience,
                risk_tolerance=slider_values["risk_tolerance"],
                rookie_preference=slider_values["rookie_preference"],
                stack_preference=slider_values["stack_preference"],
                handcuff_preference=slider_values["handcuff_preference"],
                rank_reliance=slider_values["rank_reliance"],
                predictability=slider_values["predictability"],
                notes=notes,
            )
            state.set_league(
                League(
                    config=league.config,
                    managers=[
                        replace(m, preferences=updated)
                        if m.draft_slot == chosen_slot else m
                        for m in league.managers
                    ],
                ),
                source="manager preferences",
            )
            # Rebuilt right here rather than left to the user to press the button
            # again: preferences that are saved but not folded in yet would show the
            # old numbers on this very page, which reads as "it did not work".
            if history.drafts:
                annotate_history(history, pool=pool, roster=league.config.roster)
            state.set_profiles(
                build_profiles(
                    state.league(), history if history.drafts else None,
                    settings=state.settings(), pool=pool, annotate=False,
                )
            )

            unmatched = _unmatched_names(favourite_names + disliked_names, pool)
            message = f"Saved what you know about {profile.manager_name}, and rebuilt "\
                      "their profile."
            if unmatched:
                components.flash(
                    message + " These names matched nobody on the board, so they will "
                    "have no effect: " + ", ".join(unmatched) + ".",
                    kind="warning",
                )
            else:
                components.flash(message)
            st.rerun()

        if stated.has_any:
            if st.button(f"Forget what I told you about {profile.manager_name}"):
                state.set_league(
                    League(
                        config=league.config,
                        managers=[
                            replace(m, preferences=ManagerPreferences())
                            if m.draft_slot == chosen_slot else m
                            for m in league.managers
                        ],
                    ),
                    source="manager preferences",
                )
                if history.drafts:
                    annotate_history(history, pool=pool, roster=league.config.roster)
                state.set_profiles(
                    build_profiles(
                        state.league(), history if history.drafts else None,
                        settings=state.settings(), pool=pool, annotate=False,
                    )
                )
                components.flash(
                    f"{profile.manager_name} is back to what the model infers on its own."
                )
                st.rerun()

if profile.position_rate_by_round:
    with st.expander("When they take each position"):
        st.caption(
            "Probability of spending a pick on each position in each round, from their "
            "history. This is what the simulator samples from when it is their turn, "
            "so it is the most direct view of how they will be played."
        )
        # Keyed round -> position -> rate. Rows are rounds so the table reads down the
        # draft in order, which is how the habit itself is described ("QB in round 2").
        rate_frame = pd.DataFrame(
            {
                f"R{round_number:02d}": {
                    str(position): round(rate, 3) for position, rate in by_position.items()
                }
                for round_number, by_position in sorted(
                    profile.position_rate_by_round.items()
                )
            }
        ).fillna(0.0).T
        st.dataframe(rate_frame, width="stretch")

if profile.notes:
    st.info(profile.notes)

# ─────────────────────────────────────────────────────────────────────────────
# The evidence itself
# ─────────────────────────────────────────────────────────────────────────────
if history.drafts:
    with st.expander(f"Every pick {profile.manager_name} has made"):
        st.caption(
            "The raw evidence. `Rank inversions` is the count of better-ranked players "
            "they passed over on that pick — the statistic that separates a "
            "list-follower from someone with a plan, and the one the archetype chain "
            "leans on hardest."
        )
        picks = [
            pick for pick in history.all_picks
            if pick.manager_key == profile.manager_key
        ]
        if not picks:
            st.warning(
                "No picks matched this manager's name in the loaded history. If they "
                "should have some, the spelling in your history file differs from the "
                "spelling in the league roster."
            )
        else:
            pick_frame = pd.DataFrame([
                {
                    "Season": pick.season,
                    "Round": pick.round_number,
                    "Overall": pick.overall_pick,
                    "Player": pick.player_name,
                    "Pos": str(pick.position) if pick.position else "",
                    "Team": pick.nfl_team or "",
                    "ADP": pick.adp,
                    "Reach": (
                        round(pick.adp_delta, 1) if pick.adp_delta is not None else None
                    ),
                    "Rank inversions": pick.rank_inversions,
                    "Rookie": bool(pick.is_rookie),
                    "Continued a run": bool(pick.continued_run),
                    "Filled": str(pick.filled_starting_slot or ""),
                }
                for pick in sorted(picks, key=lambda p: (p.season, p.overall_pick))
            ])
            st.dataframe(pick_frame, width="stretch", hide_index=True, height=400)
            components.download_frame(
                pick_frame, "Download these picks (CSV)",
                f"{profile.manager_name.replace(' ', '_')}_picks.csv",
            )

    with st.expander("Raw observed statistics (before shrinkage)"):
        st.caption(
            "What this manager's history alone says, with no pull toward the league "
            "average. Compare against the 'Used' column above to see how much the "
            "model is trusting them."
        )
        observations = observe_manager(
            profile.manager_name, history, estimation=state.settings().estimation
        )
        stat_rows = []
        for field in (
            "reach", "rank_gap", "rank_inversions", "fill_rate", "run_continue_rate",
            "rookie_rate", "stack_rate", "handcuff_rate",
        ):
            stat = getattr(observations, field)
            stat_rows.append({
                "Statistic": field.replace("_", " "),
                "Mean": None if stat.mean is None else round(stat.mean, 3),
                "Spread": None if stat.stdev is None else round(stat.stdev, 3),
                "Picks": stat.count,
                "Effective picks": round(stat.n, 1),
            })
        st.dataframe(pd.DataFrame(stat_rows), width="stretch", hide_index=True)
