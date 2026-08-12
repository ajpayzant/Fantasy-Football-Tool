"""Connect a real league: its managers, its settings, and its past drafts.

This is what makes the opponent model about *your* league rather than a generic
one. Everything else in :mod:`services.providers` describes players; this module
describes people.

**Sleeper needs nothing but a league ID.** Its read endpoints are public, so the
full path — league settings, the twelve real managers, and every pick of every
past draft — works with no login. Sleeper also chains seasons through
``previous_league_id``, so one ID walks back through the league's whole history.
That is the strongest source here and the one to prefer.

**ESPN and Yahoo league import are deliberately not built here.** ESPN's league
endpoints want session cookies copied out of the user's browser; Yahoo's want a
registered OAuth application and an hourly-expiring token. Both are fragile
dependencies for a local tool, and neither is worth asking a user to set up when
their draft recap is one Ctrl-C away. :mod:`services.draft_paste` reads that recap
instead and yields the same picks. Both instruction functions here point at it.

The ADP and ranking providers for ESPN and Yahoo are unaffected and do work (see
:mod:`services.providers.espn` and :mod:`services.providers.yahoo`) — it is only
*league* import that these platforms gate.

Nothing here raises. A wrong league ID, a private league without cookies, or a
season that predates the platform all come back as a result with errors on it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from core.config import LeagueConfig, RosterSettings, ScoringRules
from core.enums import (
    Archetype,
    DraftType,
    LeagueFormat,
    Platform,
    Position,
    ScoringPreset,
    Slot,
)
from core.validation import ValidationReport
from models.draft import DraftHistory, HistoricalDraft, HistoricalPick
from models.league import League
from models.manager import Manager
from services.normalize import clean_text
from services.providers.base import DEFAULT_CACHE_TTL_SECONDS, fetch_json

LOGGER = logging.getLogger("fantasy_mock_draft.providers.leagues")

SLEEPER_BASE = "https://api.sleeper.app/v1"

# League settings change during a season (trades, roster moves), so league
# metadata is cached far more briefly than player data. Fifteen minutes.
LEAGUE_CACHE_TTL_SECONDS = 15 * 60

# How far back to walk Sleeper's previous_league_id chain. Five seasons is well
# past the point of diminishing returns for tendency estimation, and it bounds the
# request count on a league that has run for a decade.
MAX_HISTORY_SEASONS = 5

# Sleeper's roster_positions strings mapped onto the app's lineup slots. Sleeper
# writes several names for a flex depending on league age.
SLOT_MAP: dict[str, Slot] = {
    "QB": Slot.QB,
    "RB": Slot.RB,
    "WR": Slot.WR,
    "TE": Slot.TE,
    "K": Slot.K,
    "DEF": Slot.DST,
    "DST": Slot.DST,
    "FLEX": Slot.FLEX,
    "WRRB_FLEX": Slot.FLEX,
    "WRRB_WRT": Slot.FLEX,
    "REC_FLEX": Slot.FLEX,
    "SUPER_FLEX": Slot.SUPERFLEX,
    "QB_FLEX": Slot.SUPERFLEX,
    "IDP_FLEX": Slot.BENCH,   # the engine does not model IDP; seat it on the bench
    "BN": Slot.BENCH,
}

# Slots that exist in Sleeper but hold no drafted player. They consume no pick, so
# counting them as lineup seats would make roster size disagree with the rounds.
NON_DRAFT_SLOTS = frozenset({"IR", "TAXI"})


@dataclass(slots=True)
class LeagueImportResult:
    """A connected league, its history, and everything worth reporting."""

    league: League | None = None
    history: DraftHistory = field(default_factory=DraftHistory)
    report: ValidationReport = field(default_factory=ValidationReport)
    source: str = ""
    seasons_found: tuple[int, ...] = ()

    @property
    def ok(self) -> bool:
        return self.league is not None


def _get(path: str, *, cache_key: str, ttl: float = LEAGUE_CACHE_TTL_SECONDS) -> Any:
    """GET one Sleeper endpoint, returning ``None`` on any failure."""
    payload, outcome = fetch_json(
        f"{SLEEPER_BASE}/{path}", cache_key=cache_key, ttl_seconds=ttl
    )
    if not outcome.ok:
        LOGGER.warning("Sleeper %s failed: %s", path, outcome.error)
        return None
    return payload


def fetch_sleeper_league(
    league_id: str,
    *,
    include_history: bool = True,
    max_seasons: int = MAX_HISTORY_SEASONS,
) -> LeagueImportResult:
    """Build a :class:`League` (and its draft history) from a Sleeper league ID.

    The ID is the number in the league's Sleeper URL. No login is required — these
    endpoints are public reads.
    """
    result = LeagueImportResult(source=f"Sleeper league {league_id}")
    report = result.report
    league_id = clean_text(league_id)
    if not league_id or not league_id.isdigit():
        report.error(
            "sleeper_league_id",
            "A Sleeper league ID is the long number in your league's URL — for "
            "example the '1048291234567890123' in "
            "sleeper.com/leagues/1048291234567890123/team.",
        )
        return result

    payload = _get(f"league/{league_id}", cache_key=f"sleeper_league_{league_id}")
    if not isinstance(payload, dict) or not payload.get("league_id"):
        report.error(
            "sleeper_league_missing",
            f"Sleeper has no league with ID {league_id}, or it could not be reached. "
            "Check the ID against your league's URL.",
        )
        return result

    settings = payload.get("settings") or {}
    scoring_settings = payload.get("scoring_settings") or {}
    roster_positions = [str(p).upper() for p in (payload.get("roster_positions") or [])]

    season = _as_int(payload.get("season")) or 0
    team_count = (
        _as_int(settings.get("num_teams"))
        or _as_int(payload.get("total_rosters"))
        or 12
    )

    # ── Roster settings ─────────────────────────────────────────────────────
    slots: dict[Slot, int] = {}
    unmapped: set[str] = set()
    for position in roster_positions:
        if position in NON_DRAFT_SLOTS:
            continue
        slot = SLOT_MAP.get(position)
        if slot is None:
            unmapped.add(position)
            continue
        slots[slot] = slots.get(slot, 0) + 1
    if not slots:
        report.warn(
            "sleeper_roster_default",
            "Sleeper reported no roster positions, so a standard lineup was assumed. "
            "Check the starting lineup on the League settings tab.",
        )
        slots = {
            Slot.QB: 1, Slot.RB: 2, Slot.WR: 2, Slot.TE: 1,
            Slot.FLEX: 1, Slot.K: 1, Slot.DST: 1, Slot.BENCH: 6,
        }
    if unmapped:
        report.warn(
            "sleeper_unmapped_slots",
            f"These Sleeper roster slots have no equivalent here and were ignored: "
            f"{', '.join(sorted(unmapped))}. Usually individual defensive players, "
            "which this app does not model.",
        )
    roster = RosterSettings(slots=slots)

    # ── Scoring ─────────────────────────────────────────────────────────────
    preset, ppr_note = _scoring_preset(scoring_settings)
    if ppr_note:
        report.info("sleeper_scoring", ppr_note)

    league_format = LeagueFormat.REDRAFT
    kind = str(payload.get("settings", {}).get("type", "")) or ""
    # Sleeper's settings.type: 0 redraft, 1 keeper, 2 dynasty.
    if str(settings.get("type")) == "1":
        league_format = LeagueFormat.KEEPER
    elif str(settings.get("type")) == "2":
        league_format = LeagueFormat.DYNASTY
    if league_format is not LeagueFormat.REDRAFT:
        report.warn(
            "sleeper_format",
            f"This is a {league_format} league. The draft itself is simulated the "
            "same way, but this app is built and tested around redraft — keeper "
            "and dynasty valuations are not modelled.",
        )

    # ── Managers ────────────────────────────────────────────────────────────
    users = _get(f"league/{league_id}/users", cache_key=f"sleeper_users_{league_id}")
    rosters = _get(f"league/{league_id}/rosters", cache_key=f"sleeper_rosters_{league_id}")
    draft_slots = _draft_slots(league_id, report)

    managers, manager_note = _build_managers(
        users if isinstance(users, list) else [],
        rosters if isinstance(rosters, list) else [],
        draft_slots,
        team_count,
    )
    if manager_note:
        report.warn("sleeper_managers", manager_note)
    if not managers:
        report.error(
            "sleeper_no_managers",
            "Sleeper returned no members for this league, so there are no opponents "
            "to model.",
        )
        return result

    rounds = len([p for p in roster_positions if p not in NON_DRAFT_SLOTS]) or roster.roster_size

    config = LeagueConfig(
        name=clean_text(payload.get("name")) or f"Sleeper league {league_id}",
        season=season or None,
        platform=Platform.SLEEPER,
        team_count=len(managers),
        rounds=rounds,
        draft_type=DraftType.SNAKE,
        league_format=league_format,
        scoring=ScoringRules.from_preset(preset),
        roster=roster,
        user_draft_slot=1,
    )
    league = League(config=config, managers=managers)
    league_report = league.validate()
    report.extend(league_report)
    if not league_report.ok:
        # The league failed its own validation, so returning it would push the
        # failure into the draft room. Report and stop here instead.
        return result
    result.league = league

    report.info(
        "sleeper_league_loaded",
        f"Loaded '{config.name}': {len(managers)} managers, {rounds} rounds, "
        f"{preset} scoring, {season} season.",
    )
    report.warn(
        "sleeper_user_slot",
        "Your draft slot defaults to 1 because Sleeper does not publish draft order "
        "until the draft is set up. Set it on the League settings tab before drafting.",
    )

    # ── History ─────────────────────────────────────────────────────────────
    if include_history:
        history, seasons = _fetch_history(league_id, managers, report, max_seasons)
        result.history = history
        result.seasons_found = seasons

    return result


def _draft_slots(league_id: str, report: ValidationReport) -> dict[str, int]:
    """Map Sleeper roster_id → draft slot from the league's most recent draft."""
    drafts = _get(f"league/{league_id}/drafts", cache_key=f"sleeper_drafts_{league_id}")
    if not isinstance(drafts, list) or not drafts:
        return {}
    for draft in drafts:
        if not isinstance(draft, dict):
            continue
        order = draft.get("slot_to_roster_id")
        if isinstance(order, dict) and order:
            # Invert to roster_id → slot. Sleeper keys slots as strings.
            return {
                str(roster_id): int(slot)
                for slot, roster_id in order.items()
                if roster_id is not None and str(slot).isdigit()
            }
    return {}


def _build_managers(
    users: list[Any],
    rosters: list[Any],
    draft_slots: dict[str, int],
    team_count: int,
) -> tuple[list[Manager], str]:
    """Turn Sleeper's users and rosters into managers with draft slots.

    Sleeper's user list has display names; the roster list ties a user to a
    roster_id; the draft's ``slot_to_roster_id`` ties a roster to a draft slot.
    All three are needed to seat a real person in the right slot.
    """
    owner_to_roster: dict[str, str] = {}
    for roster in rosters:
        if isinstance(roster, dict) and roster.get("owner_id") is not None:
            owner_to_roster[str(roster["owner_id"])] = str(roster.get("roster_id"))

    entries: list[tuple[int | None, str, str]] = []
    for user in users:
        if not isinstance(user, dict):
            continue
        user_id = str(user.get("user_id") or "")
        display = clean_text(user.get("display_name")) or f"Manager {len(entries) + 1}"
        team_name = clean_text((user.get("metadata") or {}).get("team_name"))
        roster_id = owner_to_roster.get(user_id, "")
        slot = draft_slots.get(roster_id)
        entries.append((slot, display, team_name))

    note = ""
    if any(slot is None for slot, _, _ in entries):
        note = (
            "Sleeper has not published a draft order for this league yet, so managers "
            "were seated in the order Sleeper lists them. Reorder them on the League "
            "settings tab if your real draft order differs."
        )

    # Managers with a known slot keep it; the rest fill the gaps in list order, so
    # every manager gets exactly one slot and no slot is doubled.
    taken = {slot for slot, _, _ in entries if slot is not None}
    free = [s for s in range(1, max(team_count, len(entries)) + 1) if s not in taken]
    managers: list[Manager] = []
    for slot, display, team_name in entries:
        if slot is None:
            slot = free.pop(0) if free else len(managers) + 1
        managers.append(
            Manager(
                name=display,
                draft_slot=int(slot),
                team_name=team_name,
                # Nothing is assumed about how a real manager drafts. BALANCED is the
                # neutral prior; their actual tendencies come from their history.
                archetype=Archetype.BALANCED,
            )
        )
    managers.sort(key=lambda m: m.draft_slot)
    return managers, note


def _drafter_names(league_id: str, managers: list[Manager]) -> dict[str, str]:
    """Map Sleeper user ids and roster ids to display names for one season.

    Names are taken from that season's own member list. Where a name matches a
    manager in the *current* league, the current spelling wins — that is what makes
    a past pick join onto a present manager's profile.
    """
    current_by_key = {
        clean_text(manager.name).lower(): manager.name for manager in managers
    }
    users = _get(f"league/{league_id}/users", cache_key=f"sleeper_users_{league_id}")
    rosters = _get(f"league/{league_id}/rosters", cache_key=f"sleeper_rosters_{league_id}")

    names: dict[str, str] = {}
    user_names: dict[str, str] = {}
    for user in users if isinstance(users, list) else []:
        if not isinstance(user, dict):
            continue
        user_id = str(user.get("user_id") or "")
        display = clean_text(user.get("display_name"))
        if not user_id or not display:
            continue
        resolved = current_by_key.get(display.lower(), display)
        user_names[user_id] = resolved
        names[user_id] = resolved

    for roster in rosters if isinstance(rosters, list) else []:
        if not isinstance(roster, dict):
            continue
        roster_id = str(roster.get("roster_id") or "")
        owner = str(roster.get("owner_id") or "")
        if roster_id and owner in user_names:
            names[roster_id] = user_names[owner]
    return names


def _fetch_history(
    league_id: str,
    managers: list[Manager],
    report: ValidationReport,
    max_seasons: int,
) -> tuple[DraftHistory, tuple[int, ...]]:
    """Walk Sleeper's previous_league_id chain and collect every past draft."""
    history = DraftHistory()
    seasons: list[int] = []
    current_id: str | None = league_id
    visited: set[str] = set()

    for _ in range(max(1, int(max_seasons))):
        if not current_id or current_id in visited:
            break
        visited.add(current_id)

        league_payload = _get(
            f"league/{current_id}", cache_key=f"sleeper_league_{current_id}"
        )
        if not isinstance(league_payload, dict):
            break
        season = _as_int(league_payload.get("season"))
        league_name = clean_text(league_payload.get("name"))

        # Each season is its own Sleeper league with its own membership, so the
        # id → name map is rebuilt per season. A manager who left two years ago
        # still gets their real name on their old picks.
        drafter_names = _drafter_names(current_id, managers)

        drafts = _get(
            f"league/{current_id}/drafts", cache_key=f"sleeper_drafts_{current_id}"
        )
        if isinstance(drafts, list):
            for draft_meta in drafts:
                if not isinstance(draft_meta, dict):
                    continue
                if str(draft_meta.get("status")) != "complete":
                    continue  # an unstarted or in-progress draft has nothing to learn from
                draft_id = str(draft_meta.get("draft_id") or "")
                if not draft_id:
                    continue
                picks = _get(
                    f"draft/{draft_id}/picks",
                    cache_key=f"sleeper_picks_{draft_id}",
                    # Completed drafts never change, so they cache for the full term.
                    ttl=DEFAULT_CACHE_TTL_SECONDS,
                )
                parsed = _parse_picks(
                    picks if isinstance(picks, list) else [],
                    season=season or 0,
                    league_name=league_name,
                    draft_id=draft_id,
                    drafter_names=drafter_names,
                )
                if parsed.picks:
                    history.add(parsed)
                    if season:
                        seasons.append(season)

        previous = league_payload.get("previous_league_id")
        current_id = str(previous) if previous else None

    if not history.drafts:
        report.warn(
            "sleeper_no_history",
            "No completed drafts were found for this league, so opponents will be "
            "modelled from archetype priors rather than their own history. If this "
            "league ran in past seasons on Sleeper, the current league is not linked "
            "to them.",
        )
    else:
        report.info(
            "sleeper_history",
            f"Found {len(history.all_picks)} picks across {len(history.drafts)} "
            f"completed draft(s): season(s) {', '.join(str(s) for s in sorted(set(seasons)))}.",
        )
    return history, tuple(sorted(set(seasons)))


def _parse_picks(
    picks: list[Any],
    *,
    season: int,
    league_name: str,
    draft_id: str,
    drafter_names: dict[str, str],
) -> HistoricalDraft:
    """Convert Sleeper draft picks into the app's historical-pick model.

    ``drafter_names`` maps Sleeper user ids *and* roster ids to display names, so a
    pick can be attributed whichever id it carries.
    """
    draft = HistoricalDraft(
        season=season,
        league_name=league_name,
        platform=Platform.SLEEPER,
        source_file=f"sleeper draft {draft_id}",
    )
    for pick in picks:
        if not isinstance(pick, dict):
            continue
        metadata = pick.get("metadata") or {}
        first = clean_text(metadata.get("first_name"))
        last = clean_text(metadata.get("last_name"))
        name = f"{first} {last}".strip()
        if not name:
            continue
        # Sleeper identifies the drafter by user/roster id, never by name, so the
        # name has to come from the league's member list. Where it cannot be
        # resolved the roster number is used — a truthful label, rather than
        # inventing a person. Note `metadata["team"]` is the *player's* NFL team,
        # not the drafter's, which is an easy and silent mistake to make here.
        picked_by = str(pick.get("picked_by") or "")
        roster_id = str(pick.get("roster_id") or "")
        manager_name = (
            drafter_names.get(picked_by)
            or drafter_names.get(roster_id)
            or (f"Roster {roster_id}" if roster_id else "Unknown manager")
        )
        draft.picks.append(
            HistoricalPick(
                overall_pick=_as_int(pick.get("pick_no")) or 0,
                round_number=_as_int(pick.get("round")),
                pick_in_round=_as_int(pick.get("draft_slot")),
                manager_name=manager_name,
                player_name=name,
                position=Position.coerce(metadata.get("position"), None),
                nfl_team=clean_text(metadata.get("team")).upper(),
                league_name=league_name,
                platform=str(Platform.SLEEPER),
                season=season,
            )
        )
    return draft


def _scoring_preset(scoring: dict[str, Any]) -> tuple[ScoringPreset, str]:
    """Infer the app's scoring preset from Sleeper's per-stat scoring settings.

    Sleeper stores raw per-stat values rather than a named format, so the preset is
    read off the reception value — which is exactly what distinguishes the formats.
    """
    try:
        reception = float(scoring.get("rec", 0.0) or 0.0)
    except (TypeError, ValueError):
        reception = 0.0
    try:
        te_bonus = float(scoring.get("bonus_rec_te", 0.0) or 0.0)
    except (TypeError, ValueError):
        te_bonus = 0.0

    if te_bonus >= 0.4:
        return ScoringPreset.TE_PREMIUM, (
            f"Sleeper reports {reception} points per reception with a {te_bonus} tight-end "
            "bonus, read as TE premium."
        )
    if reception >= 0.9:
        return ScoringPreset.FULL_PPR, "Read as full PPR from Sleeper's scoring settings."
    if reception >= 0.4:
        return ScoringPreset.HALF_PPR, "Read as half PPR from Sleeper's scoring settings."
    if reception > 0:
        return ScoringPreset.HALF_PPR, (
            f"Sleeper reports {reception} points per reception, which matches no standard "
            "preset. Half PPR was used — set it manually if that is wrong."
        )
    return ScoringPreset.STANDARD, "Read as standard (no points per reception)."


def espn_league_instructions() -> str:
    """What the ESPN route actually is, which is the paste importer.

    An earlier version of this text said a public ESPN league "imports with just its
    league ID". It does not, and never did — there is no ``fetch_espn_league`` here.
    The promise is removed rather than softened, because a user who follows it goes
    looking for a field that does not exist and concludes the app is broken.
    """
    return (
        "**There is no one-click ESPN connect, and nothing here asks for your ESPN "
        "login.** ESPN's league endpoints need session cookies copied out of your own "
        "browser, which is a poor thing to ask for and a worse thing to depend on — it "
        "breaks whenever ESPN changes them.\n\n"
        "**Use the paste importer instead, on the Draft history tab.** Open your ESPN "
        "league → *Draft Recap*, select the results, copy, and paste. The board is "
        "read directly — by round, by team, or as the draft-board grid — and it gives "
        "the opponent model exactly the same information a live connection would: who "
        "drafted whom, in what order, in which round.\n\n"
        "For your league's *settings* (teams, rounds, scoring, roster slots), fill them "
        "in on the **League settings** tab. There are only a handful and they do not "
        "change between seasons."
    )


def yahoo_league_instructions() -> str:
    """Why Yahoo league import is not implemented, stated plainly."""
    return (
        "Yahoo requires OAuth 2.0 for league data — there is no public read path like "
        "Sleeper's. Using it means registering a developer application with Yahoo, "
        "then completing a browser consent flow to get a token that expires hourly. "
        "That is a real amount of setup for a personal draft tool, so **league import "
        "from Yahoo is not built**.\n\n"
        "Yahoo **ADP still works** and is already part of the player board — it is "
        "only importing your league's managers and past drafts that needs OAuth.\n\n"
        "**Use the paste importer instead, on the Draft history tab.** Yahoo's draft "
        "results page copies cleanly, both the by-team view and the board grid, and "
        "that gives the opponent model the same picks OAuth would have fetched."
    )


def _as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "LeagueImportResult",
    "fetch_sleeper_league",
    "espn_league_instructions",
    "yahoo_league_instructions",
    "SLOT_MAP",
    "MAX_HISTORY_SEASONS",
]
