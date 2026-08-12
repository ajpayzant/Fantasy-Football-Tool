"""Setup: get current player data, connect a league, import past drafts.

Everything the rest of the app needs enters here. Routes in, in order of how much
work they ask of the user:

1. **Get current data** — one button. Fetches live rankings and ADP from Sleeper,
   Fantasy Football Calculator, ESPN and Yahoo, and seats generic opponents so a
   draft can start immediately.
2. **Connect your league** — a Sleeper league ID pulls your real managers and every
   past draft, which is what makes the opponent model about *your* league.
3. **Paste your draft board** — for ESPN, Yahoo and everyone else, who all gate
   league reads behind cookies or OAuth. Copying the draft recap yields the same
   picks with nothing to set up; :mod:`services.draft_paste` reads whichever shape
   it arrives in.
4. **Import your own files** — a rankings export and/or past draft recaps.
5. **Reload a saved league** — anything previously persisted to the local database.

No fictional players or managers are reachable from this page. Opponents with no
history are labelled by tendency ("Slot 4 · Zero-RB tendency"), never given
invented human names.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import replace

import pandas as pd
import streamlit as st

from core.config import LeagueConfig, RosterSettings, ScoringRules
from core.constants import HISTORICAL_IMPORT_COLUMNS, PLAYER_IMPORT_COLUMNS
from core.enums import DraftType, LeagueFormat, Platform, ScoringPreset, Slot
from core.validation import ValidationReport
from engine.draft_order import round_slot_order, validate_custom_order
from models.database import session_scope
from models.league import League
from models.manager import Manager
from services.adapters import platform_hint, read_pasted_text, read_tabular
from services.draft_paste import LAYOUT_LABELS, parse_draft_board
from services.importers import import_historical_drafts, import_player_pool
from services.live import build_live_board, current_season, quick_league
from services.providers.base import cache_entries, clear_cache
from services.providers.leagues import (
    espn_league_instructions,
    fetch_sleeper_league,
    yahoo_league_instructions,
)
from services.repository import (
    list_leagues,
    load_history,
    load_league,
    save_history,
    save_league,
    save_player_pool,
)
from ui import components, state

LOGGER = logging.getLogger("fantasy_mock_draft.ui.setup")

components.page_header(
    "⚙️ Setup",
    "Get current player data, connect your league, and import any past drafts.",
)


def _report_messages(report: ValidationReport, *, context: str) -> None:
    """Show every validation message. Nothing is swallowed.

    Severities are surfaced differently because they mean different things: an error
    means the import did not happen, a warning means it happened with something worth
    knowing about, and info is provenance.
    """
    for issue in report.errors:
        st.error(f"{context}: {issue.message}")
    for issue in report.warnings:
        st.warning(f"{context}: {issue.message}")
    for issue in report.infos:
        st.caption(f"{context}: {issue.message}")


def _manager_label(name: str, *, slot: int, is_user: bool) -> str:
    """Keep a real name, but move the generated placeholders to the right slot.

    ``quick_league`` names the user's seat literally "You" and every other seat
    "Slot N · <tendency> tendency". Those are generated labels, not names the user
    typed, so when the user moves their draft slot the labels have to move with
    them — otherwise the old seat still reads "You" and the new one reads like an
    opponent. A name the user actually entered is never rewritten.
    """
    generic = not name or name == "You" or name.startswith(f"Slot {slot}")
    if is_user:
        return "You" if generic else name
    if name == "You":
        # The user vacated this seat; it is an opponent now and must not keep the
        # label that means "me".
        return f"Slot {slot}"
    return name or f"Team {slot}"


# Every per-event field, grouped the way a league's settings page groups them, with
# the label and help text the user sees. Driven off a table rather than written out as
# 40 hand-placed widgets so a new field in ``ScoringRules`` needs one line here.
SCORING_GROUPS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "Passing": (
        ("pass_yards_per_point", "Yards per point", "25 means 1 point per 25 passing yards."),
        ("pass_td", "Passing TD", "6 in a six-point-passing-TD league."),
        ("interception", "Interception", ""),
        ("pass_2pt", "2-point conversion (pass)", ""),
    ),
    "Rushing & receiving": (
        ("rush_yards_per_point", "Rush yards per point", ""),
        ("rush_td", "Rushing TD", ""),
        ("rec_yards_per_point", "Rec yards per point", ""),
        ("rec_td", "Receiving TD", ""),
        ("reception", "Per reception", "0 standard, 0.5 half-PPR, 1 full PPR."),
        ("te_premium_reception_bonus", "TE reception bonus",
         "Extra points per catch for tight ends only, on top of the value above."),
        ("fumble_lost", "Fumble lost", ""),
        ("rush_rec_2pt", "2-point conversion (rush/rec)", ""),
    ),
    "Yardage bonuses": (
        ("bonus_pass_300_yards", "300 passing yards", ""),
        ("bonus_pass_400_yards", "400 passing yards", ""),
        ("bonus_rush_100_yards", "100 rushing yards", ""),
        ("bonus_rush_200_yards", "200 rushing yards", ""),
        ("bonus_rec_100_yards", "100 receiving yards", ""),
        ("bonus_rec_200_yards", "200 receiving yards", ""),
        ("bonus_long_td_40_plus", "40+ yard TD", ""),
    ),
    "Kicking": (
        ("kick_fg_made", "Field goal made", "A flat value; distance tiers are not modelled."),
        ("kick_xp_made", "Extra point made", ""),
    ),
    "Defence / special teams": (
        ("dst_sack", "Sack", ""),
        ("dst_interception", "Interception", ""),
        ("dst_fumble_recovery", "Fumble recovery", ""),
        ("dst_touchdown", "Defensive/return TD", ""),
        ("dst_safety", "Safety", ""),
    ),
    "Defence — points allowed": (
        ("dst_points_allowed_0", "Shutout (0)", ""),
        ("dst_points_allowed_1_6", "1–6 allowed", ""),
        ("dst_points_allowed_7_13", "7–13 allowed", ""),
        ("dst_points_allowed_14_17", "14–17 allowed", ""),
        ("dst_points_allowed_18_21", "18–21 allowed", ""),
        ("dst_points_allowed_22_27", "22–27 allowed", ""),
        ("dst_points_allowed_28_34", "28–34 allowed", ""),
        ("dst_points_allowed_35_plus", "35+ allowed", ""),
    ),
}


def _scoring_editor(current: ScoringRules, preset: ScoringPreset) -> dict[str, float]:
    """Render every per-event scoring value and return only the ones that changed.

    Returns overrides relative to ``preset``, not the full field set, so a user who
    opens this expander and touches nothing keeps a clean preset — and a user who
    changes one number keeps preset behaviour for everything else. The comparison
    baseline is the preset rather than ``current`` because the preset selectbox may
    have just been changed in the same submit, and the new preset's values should win
    over the old league's unless the user actually edited that field.
    """
    baseline = ScoringRules.from_preset(preset)
    # Pre-fill from the saved league when it is already on this preset, so reopening
    # the form shows what was saved rather than resetting the user's own numbers.
    seed = current if ScoringPreset.coerce(current.preset, None) is preset else baseline
    edited: dict[str, float] = {}
    for group, fields in SCORING_GROUPS.items():
        st.markdown(f"*{group}*")
        columns = st.columns(4)
        for index, (attr, label, help_text) in enumerate(fields):
            value = columns[index % 4].number_input(
                label, value=float(getattr(seed, attr)), step=0.5, format="%.2f",
                key=f"scoring_{attr}", help=help_text or None,
            )
            if abs(float(value) - float(getattr(baseline, attr))) > 1e-9:
                edited[attr] = float(value)
    return edited


_GENERATED_SLOT_RE = re.compile(r"^Slot (\d+)(.*)$")


def _reslot_label(name: str, *, slot: int, is_user: bool) -> str:
    """Renumber a generated placeholder so it names the seat it now occupies.

    ``_manager_label`` only recognises a placeholder that already carries the right
    number, which is exactly what reordering breaks: "Slot 4 · Zero-RB tendency" moved
    to seat 7 would otherwise keep advertising seat 4. The tendency half is kept —
    that came from the opponent model, not from the seat.
    """
    match = _GENERATED_SLOT_RE.match(name.strip())
    if match:
        name = f"Slot {slot}{match.group(2)}"
    return _manager_label(name, slot=slot, is_user=is_user)


def _order_fits(order: dict[int, list[int]], team_count: int, rounds: int) -> bool:
    """Whether a saved custom order still describes this league's shape."""
    if not order:
        return False
    # Every round must be present, not just the ones the user edited: that is what
    # ``core.validation.validate_league`` demands of a custom order.
    if set(order) != set(range(1, rounds + 1)):
        return False
    expected = set(range(1, team_count + 1))
    return all(
        set(slots) == expected and len(slots) == team_count
        for slots in order.values()
    )


def _show_rejected(result, *, label: str) -> None:
    """Show the rows an importer refused, with the reason attached.

    Rejections are counted on the result and detailed on the report, so silently
    reporting only the count would leave the user unable to fix their file.
    """
    if not result.rejected_rows:
        return
    st.warning(f"{result.rejected_rows} {label} row(s) were rejected.")
    rejected = result.report.rejected
    if rejected is not None and len(rejected):
        with st.expander(f"Rejected {label} rows"):
            st.dataframe(rejected, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# Route 1 — live data
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Get current player data")
st.caption(
    "Pulls live rankings and average draft position from Sleeper, Fantasy Football "
    "Calculator, ESPN and Yahoo, joins them into one board, and seats generic "
    "opponents so you can draft straight away. Connect your league below to replace "
    "those with your real managers."
)

fetch_row = st.columns([1.1, 1, 1, 1.2])
with fetch_row[0]:
    live_scoring = st.selectbox(
        "Scoring format",
        list(ScoringPreset),
        index=list(ScoringPreset).index(ScoringPreset.HALF_PPR),
        format_func=lambda p: str(p).replace("_", " ").title(),
        help=(
            "ADP is format-specific — a receiving back goes a round earlier in full "
            "PPR than in standard, so this changes the board, not just the scoring."
        ),
        key="live_scoring",
    )
with fetch_row[1]:
    live_teams = st.number_input(
        "Teams", min_value=4, max_value=20, value=12, key="live_teams",
        help="ADP is fetched for this league size — a 10-team board differs from a 14.",
    )
with fetch_row[2]:
    live_rounds = st.number_input(
        "Rounds", min_value=1, max_value=30, value=15, key="live_rounds",
    )
with fetch_row[3]:
    live_slot = st.number_input(
        "Your draft slot", min_value=1, max_value=int(live_teams), value=1,
        key="live_slot",
    )

source_row = st.columns([1, 1, 2])
with source_row[0]:
    use_espn = st.checkbox(
        "Include ESPN", value=True, key="live_use_espn",
        help=(
            "ESPN's endpoint cannot be filtered and returns its entire player "
            "database (~39 MB), so it is slow and memory-hungry. Untick it if the "
            "fetch struggles — the board is complete without it."
        ),
    )
with source_row[1]:
    use_yahoo = st.checkbox(
        "Include Yahoo", value=True, key="live_use_yahoo",
        help="Yahoo pages 25 players at a time, so this adds about twelve requests.",
    )
with source_row[2]:
    force_refresh = st.checkbox(
        "Ignore cache and re-fetch", value=False, key="live_force",
        help=(
            "Fetched data is cached for 12 hours. ADP moves over days, so the cache "
            "is normally what you want — and re-fetching hits someone else's server."
        ),
    )

if st.button("Fetch current player data", type="primary", key="fetch_live"):
    with st.spinner("Fetching from Sleeper, Fantasy Football Calculator, ESPN and Yahoo…"):
        existing = state.league()
        result = build_live_board(
            scoring=live_scoring,
            team_count=int(live_teams),
            league=existing.config if existing else None,
            use_espn=bool(use_espn),
            use_yahoo=bool(use_yahoo),
            force_refresh=bool(force_refresh),
        )
    _report_messages(result.report, context="Live data")

    if result.ok:
        state.set_pool(result.pool, source=result.pool.metadata.source)
        state.mark_sample_data(False)
        # A league is only created if there is not one already: a user who has
        # connected their real league must not have it replaced by generic
        # opponents just because they refreshed the player data.
        if existing is None:
            league = quick_league(
                team_count=int(live_teams),
                rounds=int(live_rounds),
                user_slot=int(live_slot),
                scoring=live_scoring,
            )
            state.set_league(league, source="generic opponents (no league connected)")
            # The board was built before this league existed, so it was imported
            # with league=None and carries no VOR. Deriving it here is what makes
            # the Player Pool's VOR column and the value lens work on a first
            # fetch; without it every VOR stays blank until the user saves league
            # settings by hand.
            result.pool.apply_league(league.config)
            # set_league clears the draft and derived values, so the pool is set
            # again afterwards to survive that invalidation.
            state.set_pool(result.pool, source=result.pool.metadata.source)
            components.flash(
                f"Loaded **{len(result.pool)} real players** for {result.season}. "
                f"Opponents are generic tendencies for now — connect your league or "
                f"import past drafts to model your actual managers. "
                f"Next: **Draft Room**."
            )
        else:
            components.flash(
                f"Updated the board to **{len(result.pool)} real players** for "
                f"{result.season}, keeping **{existing.config.name}**."
            )
        st.rerun()

if state.pool() is not None and not state.is_sample_data():
    metadata = state.pool().metadata
    st.success(
        f"Loaded: {len(state.pool())} players — {metadata.source}"
        + (f", fetched {metadata.imported_at}" if metadata.imported_at else ""),
        icon="✅",
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Route 2 — your own league
# ─────────────────────────────────────────────────────────────────────────────
connect_tab, league_tab, players_tab, history_tab, saved_tab = st.tabs(
    [
        "Connect a league",
        "League settings",
        "Player pool",
        "Draft history",
        "Saved leagues",
    ]
)

# -- Connect a league --------------------------------------------------------
with connect_tab:
    st.markdown("**Sleeper** — needs only your league ID, no login")
    st.caption(
        "This pulls your real managers, your league's scoring and roster settings, "
        "and every completed draft Sleeper has for the league — walking back through "
        "past seasons automatically. Past drafts are what let the model learn each "
        "manager's actual habits instead of assuming an average one."
    )
    sleeper_row = st.columns([2, 1])
    with sleeper_row[0]:
        sleeper_id = st.text_input(
            "Sleeper league ID",
            placeholder="1048291234567890123",
            help=(
                "The long number in your league's URL: "
                "sleeper.com/leagues/**1048291234567890123**/team"
            ),
            key="sleeper_league_id",
        )
    with sleeper_row[1]:
        include_history = st.checkbox(
            "Also import past drafts", value=True, key="sleeper_history",
        )

    if st.button("Connect Sleeper league", key="connect_sleeper", type="primary"):
        if not sleeper_id.strip():
            st.warning("Enter your Sleeper league ID first.")
        else:
            with st.spinner("Reading your league from Sleeper…"):
                connected = fetch_sleeper_league(
                    sleeper_id.strip(), include_history=bool(include_history)
                )
            _report_messages(connected.report, context="Sleeper league")
            if connected.ok:
                state.set_league(connected.league, source=connected.source)
                if connected.history.drafts:
                    state.set_history(connected.history, source=connected.source)
                pool = state.pool()
                if pool is not None:
                    # Scoring drives projections and value-over-replacement, so the
                    # board has to be recomputed against the real league's rules.
                    pool.apply_league(connected.league.config)
                    state.set_pool(pool, source=pool.metadata.source)
                state.mark_sample_data(False)
                names = ", ".join(m.name for m in connected.league.managers[:4])
                components.flash(
                    f"Connected **{connected.league.config.name}** — "
                    f"{len(connected.league.managers)} real managers ({names}…) and "
                    f"{len(connected.history.all_picks)} historical picks across "
                    f"{len(connected.history.drafts)} draft(s). "
                    + (
                        "Next: build the profiles on **Manager Profiles**."
                        if connected.history.drafts
                        else "With no past drafts, opponents use archetype priors."
                    )
                )
                st.rerun()

    st.divider()
    st.markdown("**ESPN, Yahoo, NFL.com, CBS** — paste your draft recap, no login")
    st.caption(
        "None of these can be connected without either browser cookies or an OAuth "
        "application, so none of them are. What replaces it works on all of them and "
        "needs nothing installed: copy your draft recap and paste it on the **Draft "
        "history** tab. That gives the opponent model the same picks a connection "
        "would — who took whom, in what order. Fill in the league's settings on "
        "**League settings**; there are only a handful."
    )
    with st.expander("Why ESPN is not a one-click connect"):
        st.markdown(espn_league_instructions())
    with st.expander("Why Yahoo is not a one-click connect"):
        st.markdown(yahoo_league_instructions())

# -- League settings ---------------------------------------------------------
with league_tab:
    existing = state.league()
    current = existing.config if existing else LeagueConfig()

    with st.form("league_form"):
        st.markdown("**Basics**")
        row1 = st.columns(3)
        name = row1[0].text_input("League name", value=current.name or "My League")
        season = row1[1].number_input(
            "Season", min_value=2000, max_value=2100, value=int(current.season),
            help="The season you are drafting for.",
        )
        platform = row1[2].selectbox(
            "Platform", list(Platform),
            index=list(Platform).index(Platform.coerce(current.platform, Platform.CUSTOM)),
            format_func=lambda p: str(p).upper(),
        )

        row2 = st.columns(4)
        team_count = row2[0].number_input(
            "Teams", min_value=4, max_value=20, value=int(current.team_count)
        )
        rounds = row2[1].number_input(
            "Rounds", min_value=1, max_value=30, value=int(current.rounds)
        )
        user_slot = row2[2].number_input(
            "Your draft slot", min_value=1, max_value=int(team_count),
            value=min(int(current.user_draft_slot or 1), int(team_count)),
        )
        draft_type = row2[3].selectbox(
            "Draft type", list(DraftType),
            index=list(DraftType).index(
                DraftType.coerce(current.draft_type, DraftType.SNAKE)
            ),
            format_func=lambda d: str(d).replace("_", " ").title(),
        )

        st.markdown("**Scoring**")
        score_row = st.columns(2)
        preset = score_row[0].selectbox(
            "Preset", list(ScoringPreset),
            index=list(ScoringPreset).index(
                ScoringPreset.coerce(current.scoring.preset, ScoringPreset.HALF_PPR)
            ),
            format_func=lambda p: str(p).replace("_", " ").title(),
        )
        league_format = score_row[1].selectbox(
            "Format", list(LeagueFormat),
            index=list(LeagueFormat).index(
                LeagueFormat.coerce(current.league_format, LeagueFormat.REDRAFT)
            ),
            format_func=lambda f: str(f).replace("_", " ").title(),
            help=(
                "Redraft is fully supported. Keeper and dynasty affect how the "
                "engine treats consumed picks, but are less thoroughly exercised."
            ),
        )

        with st.expander("Advanced scoring — every per-event value"):
            st.caption(
                "The preset above fills these in. Change any of them and your league "
                "is saved as **custom** scoring, keeping the preset's values for "
                "everything you did not touch. Value over replacement recomputes "
                "immediately. Projections are ESPN's projected stat lines scored "
                "under these numbers *at the moment the board was fetched*, so to "
                "rescore them press **Get current data** again after saving."
            )
            scoring_overrides = _scoring_editor(current.scoring, preset)

        st.markdown("**Starting lineup**")
        st.caption(
            "Roster size is the sum of these. It must equal teams × rounds worth of "
            "seats for a draft to fill every team exactly."
        )
        slot_defaults = {
            Slot.QB: 1, Slot.RB: 2, Slot.WR: 2, Slot.TE: 1,
            Slot.FLEX: 1, Slot.K: 1, Slot.DST: 1, Slot.BENCH: 7,
        }
        slot_values: dict[Slot, int] = {}
        slot_columns = st.columns(len(slot_defaults))
        for column, (slot, default) in zip(slot_columns, slot_defaults.items()):
            slot_values[slot] = column.number_input(
                str(slot).upper(), min_value=0, max_value=12,
                value=int(current.roster.count(slot) or default),
                key=f"slot_{slot}",
            )

        st.markdown("**Managers**")
        st.caption(
            "One row per team, in draft-slot order. Names must match the spellings in "
            "your draft history, or the opponent model cannot join them up — close "
            "spellings are matched automatically, but exact is safer."
        )
        default_names = (
            [m.name for m in existing.managers] if existing
            else [f"Team {i}" for i in range(1, int(team_count) + 1)]
        )
        default_names = (default_names + [""] * int(team_count))[: int(team_count)]
        manager_frame = st.data_editor(
            pd.DataFrame({
                "draft_slot": list(range(1, int(team_count) + 1)),
                "manager_name": default_names,
            }),
            hide_index=True, width="stretch", num_rows="fixed",
            disabled=["draft_slot"], key="manager_editor",
        )

        submitted = st.form_submit_button("Save league settings", type="primary")

    if submitted:
        roster = RosterSettings(slots={s: int(v) for s, v in slot_values.items() if v})
        scoring = ScoringRules.from_preset(preset, **scoring_overrides)
        if scoring_overrides:
            # An edited value means the league no longer matches any named preset, so
            # relabel it rather than showing "Half PPR" while scoring something else.
            # ``preset`` cannot go through ``from_preset``'s overrides — it is that
            # method's own first parameter.
            scoring = replace(scoring, preset=ScoringPreset.CUSTOM)
        config = LeagueConfig(
            name=name.strip() or "My League",
            season=int(season),
            platform=platform,
            team_count=int(team_count),
            rounds=int(rounds),
            draft_type=draft_type,
            league_format=league_format,
            scoring=scoring,
            roster=roster,
            user_draft_slot=int(user_slot),
            # Carried across the save rather than defaulted, or editing the league name
            # would silently throw away a hand-built pick order. Dropped when the shape
            # no longer fits, because an order for a different number of teams cannot
            # be repaired — only rebuilt.
            custom_round_order=(
                current.custom_round_order
                if _order_fits(current.custom_round_order, int(team_count), int(rounds))
                else {}
            ),
            reversal_round=min(int(current.reversal_round), max(2, int(rounds))),
        )
        if draft_type is DraftType.CUSTOM and not config.custom_round_order:
            # ``validate_league`` requires an explicit order for every round, so
            # selecting Custom with nothing to seat would make the league unsavable.
            # Snake is the order the engine falls back to anyway; the user edits it
            # in "Draft order" below, starting from something valid.
            config = config.with_(custom_round_order={
                rnd: round_slot_order(config, rnd)
                for rnd in range(1, config.rounds + 1)
            })
        managers = [
            Manager(
                name=_manager_label(
                    str(row.manager_name).strip(),
                    slot=int(row.draft_slot),
                    is_user=(int(row.draft_slot) == int(user_slot)),
                ),
                draft_slot=int(row.draft_slot),
                is_user=(int(row.draft_slot) == int(user_slot)),
            )
            for row in manager_frame.itertuples()
        ]
        # Validate the assembled league rather than the config alone, so duplicate
        # slots and missing names are caught here instead of at the first pick.
        report = League(config=config, managers=managers).validate()
        _report_messages(report, context="League settings")
        if report.ok:
            league = League(config=config, managers=managers)
            state.set_league(league, source="manual configuration")
            pool = state.pool()
            if pool is not None:
                # Scoring drives projections and value-over-replacement, so a pool
                # loaded under the previous settings has to be recomputed.
                pool.apply_league(config)
            st.success(
                f"Saved **{config.name}**. "
                f"{config.team_count} teams × {config.rounds} rounds = "
                f"{config.team_count * config.rounds} picks; "
                f"roster size {roster.roster_size}."
                + (
                    f" Scoring: custom, {len(scoring_overrides)} value(s) changed from "
                    f"{str(preset).replace('_', ' ').title()}."
                    if scoring_overrides else ""
                )
            )
            # Real projections were scored when the board was fetched, and the raw
            # stat lines are not kept in a re-scorable form, so changing scoring
            # cannot retroactively move them. Say so instead of letting the user
            # believe the board now reflects the rules they just entered.
            scoring_changed = (
                existing is not None
                and existing.config.scoring.to_dict() != scoring.to_dict()
            )
            if scoring_changed and pool is not None and any(
                p.projection_detail for p in pool
            ):
                st.warning(
                    "You changed scoring, and this board's projections were scored "
                    "under the previous rules. Everything derived from them — tiers, "
                    "value over replacement, ceiling and floor — has been recomputed, "
                    "but the projections feeding it have not moved. Press **Get "
                    "current data** above to refetch and rescore them."
                )
            if roster.roster_size != config.rounds:
                st.warning(
                    f"Roster size ({roster.roster_size}) does not equal the number of "
                    f"rounds ({config.rounds}). The draft will still run, but teams "
                    "will finish with unfilled seats or undraftable surplus."
                )

    # -- Draft order ---------------------------------------------------------
    # Read the league back from state rather than reusing ``existing``: a save above
    # in this same run has already replaced it, and editing a stale copy would undo it.
    st.divider()
    st.markdown("**Draft order**")
    saved = state.league()
    if saved is None:
        st.caption(
            "Save league settings above first — the draft order needs to know how "
            "many teams and rounds there are."
        )
    else:
        saved_config = saved.config
        seated = sorted(saved.managers, key=lambda m: m.draft_slot)
        st.caption(
            f"Draft type is **{str(saved_config.draft_type).replace('_', ' ').title()}**, "
            f"set in the form above. Slot 1 picks first in round 1."
        )

        st.markdown("*Who sits in which seat*")
        st.caption(
            "Type new slot numbers to reorder — each slot from 1 to "
            f"{saved_config.team_count} exactly once — or draw them at random the way "
            "most leagues do. Your own seat follows your name, so you keep the same "
            "opponents either side of you."
        )
        seat_frame = st.data_editor(
            pd.DataFrame({
                "manager_name": [m.name for m in seated],
                "draft_slot": [int(m.draft_slot) for m in seated],
                "is_you": [bool(m.is_user) for m in seated],
            }),
            column_config={
                "manager_name": st.column_config.TextColumn("Manager", disabled=True),
                "draft_slot": st.column_config.NumberColumn(
                    "Slot", min_value=1, max_value=int(saved_config.team_count), step=1,
                ),
                "is_you": st.column_config.CheckboxColumn("You", disabled=True),
            },
            hide_index=True, width="stretch", num_rows="fixed", key="seat_editor",
        )
        seat_buttons = st.columns([1, 1, 2])
        apply_seats = seat_buttons[0].button("Apply this order")
        draw_seats = seat_buttons[1].button("Draw at random")

        def _reseat(assignment: list[int]) -> list[Manager]:
            """Move managers to new slots and follow the user's seat with them.

            ``assignment[i]`` is the new slot for ``seated[i]`` — positional rather
            than keyed by name, because two teams are allowed to share a name and
            keying on it would collapse them onto one seat.
            """
            moved = [
                replace(
                    manager,
                    draft_slot=int(slot),
                    name=_reslot_label(
                        manager.name, slot=int(slot), is_user=manager.is_user
                    ),
                )
                for manager, slot in zip(seated, assignment)
            ]
            user_slot_now = next(
                (m.draft_slot for m in moved if m.is_user), saved_config.user_draft_slot
            )
            # A hand-built custom order refers to slot numbers, and the managers sitting
            # in those slots have just changed. The order itself is still a valid
            # permutation, so it is kept — it describes seats, not people.
            moved = sorted(moved, key=lambda m: m.draft_slot)
            state.set_league(
                League(
                    config=saved_config.with_(user_draft_slot=int(user_slot_now)),
                    managers=moved,
                ),
                source="manual draft order",
            )
            return moved

        if apply_seats or draw_seats:
            if draw_seats:
                assignment = list(range(1, saved_config.team_count + 1))
                random.shuffle(assignment)
            else:
                assignment = [int(row.draft_slot) for row in seat_frame.itertuples()]
            wanted = set(range(1, saved_config.team_count + 1))
            if set(assignment) != wanted or len(assignment) != saved_config.team_count:
                duplicated = sorted({s for s in assignment if assignment.count(s) > 1})
                st.error(
                    "Every slot from 1 to "
                    f"{saved_config.team_count} must be used exactly once. "
                    + (f"Slot(s) {duplicated} are used twice. " if duplicated else "")
                    + (
                        f"Slot(s) {sorted(wanted - set(assignment))} are unused."
                        if wanted - set(assignment) else ""
                    )
                )
            else:
                moved = _reseat(assignment)
                components.flash(
                    "Draft order set. "
                    + ", ".join(f"{m.draft_slot}. {m.name}" for m in moved)
                )
                st.rerun()

        if saved_config.draft_type is DraftType.THIRD_ROUND_REVERSAL:
            st.markdown("*Reversal round*")
            reversal = st.number_input(
                "The round where the order flips a second time",
                min_value=2, max_value=int(saved_config.rounds),
                value=int(saved_config.reversal_round),
                help="3 is the usual choice, which is what makes it a third-round "
                     "reversal. Round 3 then repeats round 2's order instead of "
                     "switching back.",
                key="reversal_round_input",
            )
            if int(reversal) != int(saved_config.reversal_round):
                if st.button("Apply reversal round"):
                    state.set_league(
                        League(
                            config=saved_config.with_(reversal_round=int(reversal)),
                            managers=list(saved.managers),
                        ),
                        source="manual draft order",
                    )
                    components.flash(f"The order now flips at round {int(reversal)}.")
                    st.rerun()

        if saved_config.draft_type is DraftType.CUSTOM:
            st.markdown("*Pick order, round by round*")
            st.caption(
                "One row per round, listing draft slots in the order they pick. "
                "Seeded with the snake order so it is valid before you touch it. Each "
                f"row must use every slot from 1 to {saved_config.team_count} exactly "
                "once — this is the escape hatch for leagues whose order follows no "
                "formula at all."
            )
            existing_order = {
                rnd: list(saved_config.custom_round_order.get(rnd)
                          or round_slot_order(saved_config, rnd))
                for rnd in range(1, saved_config.rounds + 1)
            }
            grid = pd.DataFrame(
                {
                    f"Pick {position}": [
                        existing_order[rnd][position - 1]
                        for rnd in range(1, saved_config.rounds + 1)
                    ]
                    for position in range(1, saved_config.team_count + 1)
                },
                index=[f"Round {rnd}" for rnd in range(1, saved_config.rounds + 1)],
            )
            edited_grid = st.data_editor(
                grid,
                column_config={
                    column: st.column_config.NumberColumn(
                        column, min_value=1, max_value=int(saved_config.team_count),
                        step=1,
                    )
                    for column in grid.columns
                },
                width="stretch", num_rows="fixed", key="custom_order_editor",
            )
            if st.button("Apply pick order"):
                proposed = {
                    rnd: [int(v) for v in edited_grid.iloc[rnd - 1].tolist()]
                    for rnd in range(1, saved_config.rounds + 1)
                }
                candidate = saved_config.with_(custom_round_order=proposed)
                problems = validate_custom_order(candidate)
                if problems:
                    for problem in problems:
                        st.error(problem)
                else:
                    state.set_league(
                        League(config=candidate, managers=list(saved.managers)),
                        source="manual draft order",
                    )
                    components.flash("Custom pick order saved.")
                    st.rerun()

        with st.expander("What this produces — the first few rounds"):
            preview_rounds = min(4, int(saved_config.rounds))
            by_slot = {int(m.draft_slot): m.name for m in saved.managers}
            st.dataframe(
                pd.DataFrame(
                    [
                        [
                            by_slot.get(slot, f"Slot {slot}")
                            for slot in round_slot_order(saved_config, rnd)
                        ]
                        for rnd in range(1, preview_rounds + 1)
                    ],
                    columns=[
                        f"Pick {i}" for i in range(1, saved_config.team_count + 1)
                    ],
                    index=[f"Round {rnd}" for rnd in range(1, preview_rounds + 1)],
                ),
                width="stretch",
            )
            st.caption(
                "Straight from the same function the draft itself uses, so what you "
                "see here is what you will get."
            )

# -- Player pool ------------------------------------------------------------
with players_tab:
    st.markdown("**Load a player pool**")
    st.caption(
        "A ranking or projection export from anywhere you are entitled to use one. "
        "Only `player_name` and `position` are required — ADP, ranks, projections, "
        "tiers and bye weeks are all used when present and imputed when absent."
    )
    with st.expander("Accepted columns"):
        st.code(", ".join(PLAYER_IMPORT_COLUMNS), language="text")
        st.caption(
            "Header spellings are matched loosely, so `Player`, `player name` and "
            "`PLAYER_NAME` all work."
        )
        st.caption(platform_hint(
            state.league().config.platform if state.league() else Platform.CUSTOM
        ))

    upload_column, paste_column = st.columns(2)
    with upload_column:
        uploaded = st.file_uploader(
            "Upload CSV / TSV / Excel", type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls"],
            key="pool_upload",
        )
    with paste_column:
        pasted = st.text_area(
            "…or paste a table", height=150, key="pool_paste",
            placeholder="player_name,position,overall_adp\nJoe Example,RB,1.2",
        )

    if st.button("Import player pool", key="import_pool"):
        frame = None
        source_name = ""
        if uploaded is not None:
            frame, read_report = read_tabular(uploaded, file_name=uploaded.name)
            source_name = uploaded.name
            _report_messages(read_report, context="Reading the file")
        elif pasted.strip():
            frame, read_report = read_pasted_text(pasted)
            source_name = "pasted table"
            _report_messages(read_report, context="Reading the pasted table")
        else:
            st.warning("Upload a file or paste a table first.")

        if frame is not None and not frame.empty:
            league = state.league()
            result = import_player_pool(
                frame,
                league=league.config if league else None,
                source=source_name or "upload",
                season=league.config.season if league else None,
            )
            _report_messages(result.report, context="Importing players")
            if result.pool is not None and len(result.pool):
                state.set_pool(result.pool, source=source_name)
                # A user-supplied pool means the board is no longer fictional, even
                # if the sample league's managers and history are still loaded.
                state.mark_sample_data(False)
                st.success(f"Imported {len(result.pool)} players from {source_name}.")
                _show_rejected(result, label="player")
                st.rerun()

    pool = state.pool()
    if pool is not None:
        st.divider()
        st.markdown(f"**Currently loaded:** {pool.metadata.describe()}")
        if pool.metadata.imputed_fields:
            st.caption(
                "Imputed (missing in your file, filled in so the engine has an "
                "ordering): "
                + ", ".join(f"{k} ×{v}" for k, v in pool.metadata.imputed_fields.items())
            )
        st.dataframe(
            components.player_frame(pool.players[:25], pool=pool),
            width="stretch", hide_index=True,
        )
        st.caption("First 25 rows. The full board is on the **Player Pool** page.")

# -- Draft history ----------------------------------------------------------
with history_tab:
    # ── Paste a draft board ──────────────────────────────────────────────────
    # First, because it is the route that works for every platform. The tabular
    # import below it requires a file with the right column headers, which is a
    # thing almost nobody has; a draft recap is a thing everybody has.
    st.markdown("**Paste your draft board — ESPN, Yahoo, NFL.com, CBS, anywhere**")
    st.caption(
        "Open your league's draft recap, select the results, copy, paste. No login, "
        "no league ID, no export. Whatever shape it arrives in is read directly — a "
        "list by round, a block per team, or the draft-board grid — and what it "
        "understood is shown below before anything is imported."
    )

    board_text = st.text_area(
        "Paste the draft recap",
        height=180,
        key="board_paste",
        placeholder=(
            "ROUND 1\n"
            "1. Team Alpha — Ja'Marr Chase, WR CIN\n"
            "2. Beta Ballers — Bijan Robinson, RB ATL"
        ),
    )

    board_league = state.league()
    board_row = st.columns([1, 1, 1, 1])
    board_season = board_row[0].number_input(
        "Season of this draft",
        min_value=2000, max_value=2100,
        value=int(board_league.config.season) - 1 if board_league else current_season() - 1,
        key="board_season",
        help="Past drafts are what the profiles are built from, so this is normally a "
             "previous season rather than the one you are drafting.",
    )
    board_layout_choice = board_row[1].selectbox(
        "Layout",
        ["Detect automatically", *LAYOUT_LABELS],
        format_func=lambda value: (
            value if value == "Detect automatically" else LAYOUT_LABELS[value]
        ),
        key="board_layout",
        help="Only needed if the automatic reading below is wrong. Misreading the "
             "layout is the one mistake that produces picks that look fine and are "
             "attributed to the wrong managers.",
    )
    board_snake = board_row[2].checkbox(
        "Snake draft", value=True, key="board_snake",
        help="Used to reconstruct pick order for layouts that do not state it — a "
             "board grid or a by-team list. Uncheck for a linear draft.",
    )
    board_replace = board_row[3].checkbox(
        "Replace loaded history", value=False, key="board_replace",
        help="Off, this season is added to any history already loaded, which is what "
             "you want when pasting several seasons one at a time.",
    )

    if board_text.strip():
        # Parsed on every rerun rather than behind a button: the whole risk of this
        # feature is a misread layout, so the reading has to be visible *before* the
        # user commits to it, and re-parsing costs nothing.
        board = parse_draft_board(
            board_text,
            season=int(board_season),
            team_count=board_league.config.team_count if board_league else None,
            manager_names=[m.name for m in board_league.managers] if board_league else None,
            layout=None if board_layout_choice == "Detect automatically" else board_layout_choice,
            league_name=board_league.config.name if board_league else "",
            platform=board_league.config.platform if board_league else None,
            snake=bool(board_snake),
        )
        _report_messages(board.report, context="Reading the board")
        for note in board.notes:
            st.caption(f"ℹ️ {note}")

        if board.ok:
            st.success(board.describe(), icon="✅")
            with st.expander(
                f"What it read — {board.pick_count} pick(s), check before importing",
                expanded=True,
            ):
                st.dataframe(
                    board.frame[[
                        "overall_pick", "round", "pick_in_round", "manager_name",
                        "player_name", "position", "nfl_team",
                    ]].rename(columns={
                        "overall_pick": "Pick", "round": "Rd",
                        "pick_in_round": "In rd", "manager_name": "Manager",
                        "player_name": "Player", "position": "Pos", "nfl_team": "Team",
                    }),
                    width="stretch", hide_index=True, height=280,
                )
            if board.unparsed:
                with st.expander(f"{len(board.unparsed)} line(s) could not be read"):
                    st.caption(
                        "These were left out. Nothing is dropped silently — fix them "
                        "in the paste above, or add them by hand further down."
                    )
                    st.dataframe(
                        board.unparsed_frame(), width="stretch", hide_index=True
                    )

            if st.button(
                f"Import these {board.pick_count} picks", key="import_board",
                type="primary",
            ):
                result = import_historical_drafts(
                    board.frame,
                    default_season=int(board_season),
                    default_league_name=board_league.config.name if board_league else "",
                    default_platform=board_league.config.platform if board_league else None,
                    source_file=f"pasted draft board ({board.layout})",
                )
                _report_messages(result.report, context="Importing the board")
                if result.history is not None and result.history.drafts:
                    merged = result.history
                    if not board_replace:
                        merged = state.history()
                        for draft in result.history.drafts:
                            merged.add(draft)
                    state.set_history(merged, source=f"pasted draft board ({board.layout})")
                    components.flash(
                        f"Imported {len(result.history.all_picks)} picks from the "
                        f"pasted board for {int(board_season)}. "
                        "Next: build the profiles on **Manager Profiles**."
                    )
                    _show_rejected(result, label="board")
                    st.rerun()

    st.divider()
    st.markdown("**Import past draft results**")
    st.caption(
        "This is what makes the opponents specific to *your* league rather than "
        "generic. Two or three past seasons is enough for the model to tell a "
        "manager's habit from one unusual draft; with none, every opponent falls "
        "back to the league average."
    )
    with st.expander("Accepted columns"):
        st.code(", ".join(HISTORICAL_IMPORT_COLUMNS), language="text")
        st.caption(
            "Required: `season`, `manager_name`, `overall_pick`, `player_name`. "
            "Several seasons can share one file — they are split by the `season` column."
        )

    hist_upload, hist_paste = st.columns(2)
    with hist_upload:
        history_file = st.file_uploader(
            "Upload draft history", type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls"],
            key="history_upload",
        )
    with hist_paste:
        history_text = st.text_area(
            "…or paste draft results", height=150, key="history_paste",
            placeholder="season,manager_name,overall_pick,player_name\n2025,Alex,1,Joe Example",
        )

    replace_existing = st.checkbox(
        "Replace the currently loaded history", value=True,
        help="Unchecked, the imported seasons are added to what is already loaded.",
    )

    if st.button("Import draft history", key="import_history"):
        frame = None
        source_name = ""
        if history_file is not None:
            frame, read_report = read_tabular(history_file, file_name=history_file.name)
            source_name = history_file.name
            _report_messages(read_report, context="Reading the file")
        elif history_text.strip():
            frame, read_report = read_pasted_text(history_text)
            source_name = "pasted table"
            _report_messages(read_report, context="Reading the pasted table")
        else:
            st.warning("Upload a file or paste a table first.")

        if frame is not None and not frame.empty:
            league = state.league()
            result = import_historical_drafts(
                frame,
                default_league_name=league.config.name if league else "",
                default_platform=league.config.platform if league else None,
                source_file=source_name,
            )
            _report_messages(result.report, context="Importing history")
            if result.history.drafts:
                merged = result.history
                if not replace_existing:
                    merged = state.history()
                    for draft in result.history.drafts:
                        merged.add(draft)
                state.set_history(merged, source=source_name)
                components.flash(
                    f"Imported {len(result.history.all_picks)} picks across "
                    f"{len(result.history.drafts)} season(s). "
                    "Next: build the profiles on **Manager Profiles**."
                )
                _show_rejected(result, label="history")
                st.rerun()

    history = state.history()
    if history.drafts:
        st.divider()
        summary = pd.DataFrame([
            {
                "Season": draft.season,
                "Picks": len(draft.picks),
                "Managers": len({p.manager_key for p in draft.picks}),
                "Rounds": max((p.round_number or 0) for p in draft.picks),
            }
            for draft in sorted(history.drafts, key=lambda d: d.season)
        ])
        st.dataframe(summary, width="stretch", hide_index=True)

        league = state.league()
        if league is not None:
            # A name in the history that matches no manager is the single most
            # common reason a profile comes out empty, and it is silent otherwise.
            known = {m.key for m in league.managers}
            seen = {p.manager_key for p in history.all_picks}
            unmatched = seen - known
            if unmatched:
                st.warning(
                    "These names appear in the history but match no manager in the "
                    "league, so their picks will not inform any profile: "
                    + ", ".join(sorted(unmatched))
                )
            missing = known - seen
            if missing:
                st.info(
                    "These managers have no history and will fall back to the league "
                    "average: "
                    + ", ".join(sorted(missing))
                )

# -- Saved leagues ----------------------------------------------------------
with saved_tab:
    st.markdown("**Save the current session**")
    save_columns = st.columns([1, 1, 2])
    if save_columns[0].button("Save to database", disabled=state.league() is None):
        league = state.league()
        with session_scope() as session:
            league_id = save_league(session, league)
            if state.history().drafts:
                save_history(session, state.history(), league_id)
            if state.pool() is not None:
                save_player_pool(
                    session, state.pool(),
                    source_kind="sample" if state.is_sample_data() else "upload",
                )
        st.success(f"Saved as league #{league_id}.")

    st.divider()
    st.markdown("**Reload a saved league**")
    with session_scope() as session:
        saved = list_leagues(session)
    if not saved:
        st.caption("Nothing saved yet.")
    else:
        st.dataframe(pd.DataFrame(saved), width="stretch", hide_index=True)
        choice = st.selectbox(
            "League to load",
            [row["league_id"] for row in saved],
            format_func=lambda lid: next(
                f"#{r['league_id']} — {r['name']} ({r['season']})"
                for r in saved if r["league_id"] == lid
            ),
        )
        if st.button("Load selected league"):
            with session_scope() as session:
                league = load_league(session, int(choice))
                history = load_history(session, int(choice))
            if league is None:
                st.error(f"League #{choice} could not be loaded.")
            else:
                state.set_league(league, source=f"database #{choice}")
                state.set_history(history, source=f"database #{choice}")
                pool = state.pool()
                if pool is not None:
                    pool.apply_league(league.config)
                components.flash(
                    f"Loaded **{league.config.name}** with "
                    f"{len(history.all_picks)} historical picks. "
                    "The player pool is loaded separately, on the Player pool tab."
                )
                st.rerun()

    st.divider()
    st.markdown("**Fetched-data cache**")
    st.caption(
        "Live payloads are kept on disk for 12 hours so a rerun does not re-request "
        "them. Clearing the cache is the way to force genuinely fresh ADP without "
        "waiting for it to expire — the fetch will be slower afterwards."
    )
    entries = cache_entries()
    if not entries:
        st.caption("Nothing cached — the next fetch will go to the network.")
    else:
        st.dataframe(
            pd.DataFrame(entries).rename(columns={
                "key": "Source", "size_kb": "Size (KB)",
                "age_hours": "Age (hours)", "fetched_at": "Fetched",
            }),
            width="stretch", hide_index=True,
        )
        if st.button("Clear cached data", key="clear_provider_cache"):
            removed = clear_cache()
            components.flash(
                f"Cleared {removed} cached payload(s). The next fetch will be live.",
                kind="info",
            )
            st.rerun()
