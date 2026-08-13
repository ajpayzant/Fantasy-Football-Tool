"""Connect a real league: its managers, its settings, and its past drafts.

This is what makes the opponent model about *your* league rather than a generic
one. Everything else in :mod:`services.providers` describes players; this module
describes people.

**Sleeper needs nothing but a league ID.** Its read endpoints are public, so the
full path — league settings, the twelve real managers, and every pick of every
past draft — works with no login. Sleeper also chains seasons through
``previous_league_id``, so one ID walks back through the league's whole history.
That is the strongest source here and the one to prefer.

**ESPN needs nothing at all for a public league**, and two browser cookies for a
private one. Pasting the league URL is enough: ESPN publishes settings, teams,
members and the full draft board on the same undocumented host the player board
already comes from. A private league answers 401 to an anonymous read, so the user
can supply ``espn_s2`` and ``SWID`` once. Those two values are credentials and are
treated as such — passed straight through to the request, never written to the
database, never logged, and never echoed back. The league *data* caches like every
other fetch; the cookies do not exist outside the call.

Because that endpoint is undocumented it can move without notice, so every failure
here is soft: the result comes back with an error on it that names the paste
importer, and Setup keeps working.

**Yahoo league import is still not built.** It requires a registered OAuth
application and an hourly-expiring token, which is a real amount of setup for a
local tool when the draft recap is one Ctrl-C away. :mod:`services.draft_paste`
reads that recap and yields the same picks; :func:`yahoo_league_instructions`
points at it, as does the NFL.com and CBS path.

The ADP and ranking providers for ESPN and Yahoo are separate from this and work
regardless (see :mod:`services.providers.espn` and :mod:`services.providers.yahoo`).

Nothing here raises. A wrong league ID, a private league without cookies, or a
season that predates the platform all come back as a result with errors on it.
"""

from __future__ import annotations

import json
import logging
import re
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
from services.providers import espn_stats
from services.providers.base import DEFAULT_CACHE_TTL_SECONDS, fetch_json
from services.providers.espn import POSITION_IDS, PROTEAM_IDS

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

# ─────────────────────────────────────────────────────────────────────────────
# ESPN
# ─────────────────────────────────────────────────────────────────────────────
ESPN_BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
"""The same host :mod:`services.providers.espn` already reads the player board from.

Undocumented, and the reason every ESPN failure here has to be recoverable rather
than fatal. ESPN has moved this host before (it was ``fantasy.espn.com/apis/v3``).
"""

# How many player ids to resolve to names in one request. ESPN takes the id list in
# a header, so the cap is politeness rather than URL length.
ESPN_PLAYER_CHUNK = 300

ESPN_LINEUP_SLOTS: dict[int, Slot] = {
    0: Slot.QB,
    1: Slot.QB,             # "TQB", team quarterback — a QB seat either way
    2: Slot.RB,
    3: Slot.WR_RB_FLEX,     # RB/WR
    4: Slot.WR,
    5: Slot.WR_TE_FLEX,     # WR/TE
    6: Slot.TE,
    7: Slot.SUPERFLEX,      # "OP", any offensive player
    16: Slot.DST,
    17: Slot.K,
    20: Slot.BENCH,
    23: Slot.FLEX,          # RB/WR/TE
}
"""ESPN's numeric lineup slot ids mapped onto the app's slots.

The ids are stable and public knowledge, but they are not *documented*, which is why
each one is spelled out with what ESPN calls it. Ids missing from this table are
missing on purpose: 8–15 are individual defensive players and 18/19 are punter and
head coach, none of which the engine models.
"""

ESPN_IGNORED_SLOTS: dict[int, str] = {
    8: "DT", 9: "DE", 10: "LB", 11: "DL", 12: "CB", 13: "S", 14: "DB", 15: "DP",
    18: "P", 19: "HC", 25: "Rookie",
}
"""Slots ESPN may report that this app drops, with ESPN's own label for each.

Named rather than silently skipped so the import can tell the user *which* seats
went missing — a user with an IDP league should hear that, not wonder why their
roster came out three short.
"""

# Reserve seats. No pick is spent on them, so counting them would make roster size
# disagree with the number of rounds.
ESPN_NON_DRAFT_SLOTS = frozenset({21, 24})  # IR, ER

# ``ESPNFAN5108403063``, ``espn85157990`` — what ESPN shows for a manager with no display
# name of their own, which on an anonymous read is most of them.
ESPN_PLACEHOLDER_HANDLE = re.compile(r"^espn(?:fan)?\d+$", re.IGNORECASE)

# ESPN's scoring is a list of {statId, points}. These are the ids whose value maps
# onto a named field in :class:`core.config.ScoringRules`, so a real league's rules
# come across as numbers rather than as a guessed preset. The ids themselves are
# defined once, in espn_stats, and imported rather than restated.
ESPN_PER_EVENT_STATS: dict[int, str] = {
    espn_stats.PASS_TD: "pass_td",
    espn_stats.INTERCEPTIONS: "interception",
    espn_stats.PASS_2PT: "pass_2pt",
    espn_stats.RUSH_TD: "rush_td",
    espn_stats.REC_TD: "rec_td",
    espn_stats.RECEPTIONS: "reception",
    espn_stats.FUMBLES_LOST: "fumble_lost",
    espn_stats.RUSH_2PT: "rush_rec_2pt",
    espn_stats.XP_MADE: "kick_xp_made",
    espn_stats.DST_SACKS: "dst_sack",
    espn_stats.DST_INTERCEPTIONS: "dst_interception",
    espn_stats.DST_FUMBLES_RECOVERED: "dst_fumble_recovery",
    espn_stats.DST_TOUCHDOWNS: "dst_touchdown",
    espn_stats.DST_SAFETIES: "dst_safety",
    **{stat_id: rule for stat_id, rule in espn_stats.DST_PA_BUCKETS},
}

# Yardage is scored per yard by ESPN (0.04 for one point per 25), and per *point* by
# this app. Inverted rather than stored as a rate so nothing downstream has to know
# which convention a rule came from.
ESPN_YARDAGE_STATS: dict[int, str] = {
    espn_stats.PASS_YARDS: "pass_yards_per_point",
    espn_stats.RUSH_YARDS: "rush_yards_per_point",
    espn_stats.REC_YARDS: "rec_yards_per_point",
}


def espn_league_reference(text: str) -> tuple[str, int | None]:
    """Pull a league ID (and season, if present) out of anything ESPN shows a user.

    Accepts a bare ID or any ESPN URL that carries one — the modern
    ``fantasy.espn.com/football/league?leagueId=123456`` and team/settings variants,
    and the legacy ``games.espn.com/ffl/...`` ones. Asking a user to find "the number
    after leagueId=" is asking them to do a job a regex does reliably, and getting it
    subtly wrong is the failure that makes the whole feature look broken.

    Returns ``("", None)`` when there is no ID to be found.
    """
    raw = clean_text(text)
    if not raw:
        return "", None
    if raw.isdigit():
        return raw, None
    league = re.search(r"league(?:_?id)?[=/](\d+)", raw, re.IGNORECASE)
    season = re.search(r"season(?:_?id)?=(\d{4})", raw, re.IGNORECASE)
    if not league:
        # A pasted URL with no leagueId at all, but exactly one long number in it, is
        # near-certainly the league. Anything more ambiguous is refused.
        numbers = re.findall(r"\d{4,}", raw)
        if len(numbers) != 1:
            return "", None
        return numbers[0], None
    return league.group(1), _as_int(season.group(1)) if season else None


def sleeper_league_reference(text: str) -> str:
    """Pull a Sleeper league ID out of a bare ID or a pasted league URL.

    Sleeper IDs are long enough that nobody retypes them, so the thing a user
    actually has on the clipboard is ``sleeper.com/leagues/1048.../team``. Accepting
    only the digits meant they had to edit the URL by hand first.
    """
    raw = clean_text(text)
    if not raw:
        return ""
    if raw.isdigit():
        return raw
    match = re.search(r"leagues?/(\d{6,})", raw, re.IGNORECASE)
    if match:
        return match.group(1)
    numbers = re.findall(r"\d{6,}", raw)
    return numbers[0] if len(numbers) == 1 else ""


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

    Takes the bare ID *or* the league URL it came from — nobody retypes a nineteen-digit
    number, so the thing on the clipboard is the URL. No login is required either way;
    these endpoints are public reads.
    """
    result = LeagueImportResult(source=f"Sleeper league {league_id}")
    report = result.report
    league_id = sleeper_league_reference(league_id)
    if not league_id:
        report.error(
            "sleeper_league_id",
            "That does not look like a Sleeper league. Paste your league's URL — for "
            "example `sleeper.com/leagues/1048291234567890123/team` — or just the long "
            "number in it.",
        )
        return result
    result.source = f"Sleeper league {league_id}"

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


# ─────────────────────────────────────────────────────────────────────────────
# ESPN — league import
# ─────────────────────────────────────────────────────────────────────────────
def _espn_cookie_header(espn_s2: str, swid: str) -> dict[str, str]:
    """The one header a private ESPN league needs, or nothing at all.

    Both values are required: ESPN rejects one without the other. ``SWID`` is a GUID
    that ESPN writes with braces, so a user who copies it without them still works.

    Nothing in this module logs the return value, and nothing stores it. It exists for
    the duration of one request.
    """
    s2 = clean_text(espn_s2)
    guid = clean_text(swid)
    if not s2 or not guid:
        return {}
    if not guid.startswith("{"):
        guid = "{" + guid.strip("{}") + "}"
    return {"Cookie": f"espn_s2={s2}; SWID={guid}"}


def _espn_get(
    path: str,
    *,
    params: str,
    cache_key: str,
    ttl: float = LEAGUE_CACHE_TTL_SECONDS,
    headers: dict[str, str] | None = None,
) -> tuple[Any, str]:
    """GET one ESPN endpoint, returning ``(payload, error)``.

    The error string is returned rather than logged-and-dropped because the caller has
    to tell a 401 (private league, cookies needed) apart from a 404 (wrong ID or wrong
    season) apart from a network failure. Those are three different things for the user
    to do next, and collapsing them into "it didn't work" is what makes an import
    feature useless.
    """
    payload, outcome = fetch_json(
        f"{ESPN_BASE}/{path}?{params}",
        cache_key=cache_key,
        ttl_seconds=ttl,
        headers=headers or None,
        timeout_seconds=30.0,
    )
    if not outcome.ok:
        # The path, never the headers: one of them may be the user's session cookie.
        LOGGER.warning("ESPN %s failed: %s", path, outcome.error)
        return None, str(outcome.error or "unknown error")
    return payload, ""


def _espn_failure_message(error: str, league_id: str, season: int, private: bool) -> str:
    """Turn an HTTP failure into the thing the user should actually do next.

    ESPN answers **401 for a league it will not show you, whether or not it exists** —
    verified against the live endpoint — so that message has to cover both a private
    league and a mistyped ID. Naming only the first would send a user with a typo off
    hunting for cookies they do not need.
    """
    if "401" in error or "403" in error:
        if private:
            return (
                f"ESPN refused the cookies for league {league_id}. Either they have "
                "expired — they do when you sign out of ESPN, so copy `espn_s2` and "
                "`SWID` again from a tab that is currently signed in — or the account "
                "they belong to is not in this league, or the league ID is wrong. "
                "Failing all three, the paste importer on the **Draft history** tab "
                "needs no login at all."
            )
        return (
            f"ESPN would not show league {league_id} for {season}. That means one of two "
            "things: the league is private, or there is no such league. If it is private, "
            "open **My league is private** above and paste the two cookies. Otherwise "
            "check the ID against your league's URL — or skip the connection and use the "
            "paste importer on the **Draft history** tab."
        )
    if "404" in error:
        return (
            f"ESPN has no league {league_id} in the {season} season. Check the ID against "
            "your league URL, and check the season — a league that has not been rolled "
            "over to a new season only exists in the old one. Failing that, the paste "
            "importer on the **Draft history** tab needs no league ID at all."
        )
    return (
        f"ESPN could not be read ({error}). Its league API is undocumented and moves "
        "without notice, so this may not be anything you did. The paste importer on the "
        "**Draft history** tab gives the opponent model the same picks and needs no "
        "connection at all."
    )


def _espn_team_name(team: dict[str, Any]) -> str:
    """A team's display name across both payload shapes ESPN has used.

    Newer seasons carry a single ``name``; older ones split it into ``location`` and
    ``nickname``. Both are checked because history import reads seasons years apart.
    """
    name = clean_text(team.get("name"))
    if name:
        return name
    parts = [clean_text(team.get("location")), clean_text(team.get("nickname"))]
    return " ".join(part for part in parts if part).strip()


def _espn_member_names(members: Any) -> dict[str, str]:
    """Map ESPN member GUID → the name to show for that person.

    ``displayName`` is preferred because it is what ESPN itself prints in a draft
    recap, which is what makes a pasted recap and an API import agree on who is who.
    The real name is only a fallback for accounts that have no handle.
    """
    names: dict[str, str] = {}
    for member in members if isinstance(members, list) else []:
        if not isinstance(member, dict) or member.get("id") is None:
            continue
        person = " ".join(
            part for part in (
                clean_text(member.get("firstName")), clean_text(member.get("lastName"))
            ) if part
        ).strip()
        display = clean_text(member.get("displayName")) or person
        if display:
            names[str(member["id"]).upper()] = display
    return names


def _espn_manager_names(payload: dict[str, Any]) -> dict[int, str]:
    """Map ESPN team id → manager name, for one season's payload.

    A team's owner is the person; the team name is the label they chose. The person is
    used, so a manager who renames their team every August still joins onto one profile
    across seasons — which is the entire point of importing history.
    """
    member_names = _espn_member_names(payload.get("members"))
    names: dict[int, str] = {}
    for team in payload.get("teams") or []:
        if not isinstance(team, dict):
            continue
        team_id = _as_int(team.get("id"))
        if team_id is None:
            continue
        owners = team.get("owners")
        owner_ids = [str(o).upper() for o in owners] if isinstance(owners, list) else []
        primary = team.get("primaryOwner")
        if primary:
            owner_ids.insert(0, str(primary).upper())
        owner_name = next(
            (member_names[oid] for oid in owner_ids if oid in member_names), ""
        )
        # ESPN hands out a placeholder handle — ESPNFAN5108403063 — to anyone who never
        # set a display name, and shows it instead of a real one on an anonymous read. It
        # identifies nobody the user could recognise, so the team name is the better
        # label. Identity across seasons does not depend on this either way: that is
        # joined on the owner GUID.
        if owner_name and ESPN_PLACEHOLDER_HANDLE.match(owner_name):
            owner_name = _espn_team_name(team) or owner_name
        names[team_id] = owner_name or _espn_team_name(team) or f"Team {team_id}"
    return names


def _espn_team_owner_ids(payload: dict[str, Any]) -> dict[int, tuple[str, ...]]:
    """Map ESPN team id → the member GUIDs that own it, for one season's payload.

    The GUID is the only thing about a manager that never changes. Team names change
    every August and display names change on a whim, so joining an old pick to a present
    manager on either of those is a guess; joining on the GUID is not.
    """
    owners: dict[int, tuple[str, ...]] = {}
    for team in payload.get("teams") or []:
        if not isinstance(team, dict):
            continue
        team_id = _as_int(team.get("id"))
        if team_id is None:
            continue
        ids: list[str] = []
        primary = team.get("primaryOwner")
        if primary:
            ids.append(str(primary).upper())
        raw = team.get("owners")
        ids.extend(str(o).upper() for o in raw if o) if isinstance(raw, list) else None
        owners[team_id] = tuple(dict.fromkeys(ids))
    return owners


def _espn_roster(settings: dict[str, Any], report: ValidationReport) -> RosterSettings:
    """Read the starting lineup off ESPN's ``lineupSlotCounts``."""
    counts = (settings.get("rosterSettings") or {}).get("lineupSlotCounts") or {}
    slots: dict[Slot, int] = {}
    ignored: list[str] = []
    for key, count in counts.items() if isinstance(counts, dict) else []:
        slot_id = _as_int(key)
        seats = _as_int(count) or 0
        if slot_id is None or seats <= 0 or slot_id in ESPN_NON_DRAFT_SLOTS:
            continue
        slot = ESPN_LINEUP_SLOTS.get(slot_id)
        if slot is None:
            ignored.append(f"{seats}×{ESPN_IGNORED_SLOTS.get(slot_id, f'slot {slot_id}')}")
            continue
        slots[slot] = slots.get(slot, 0) + seats
    if ignored:
        report.warn(
            "espn_unmapped_slots",
            f"These ESPN roster seats have no equivalent here and were dropped: "
            f"{', '.join(sorted(ignored))}. Usually individual defensive players or a "
            "head coach, which this app does not model — your roster will be that many "
            "seats smaller than ESPN's.",
        )
    if not slots:
        report.warn(
            "espn_roster_default",
            "ESPN reported no starting lineup, so a standard one was assumed. Check it "
            "on the League settings tab.",
        )
        slots = {
            Slot.QB: 1, Slot.RB: 2, Slot.WR: 2, Slot.TE: 1,
            Slot.FLEX: 1, Slot.K: 1, Slot.DST: 1, Slot.BENCH: 6,
        }
    return RosterSettings(slots=slots)


def _espn_scoring(
    settings: dict[str, Any], report: ValidationReport
) -> tuple[ScoringRules, ScoringPreset]:
    """Build real scoring rules from ESPN's per-stat table, not just a preset guess.

    ESPN publishes ``scoringItems`` — one entry per stat with the points it is worth —
    and this app already names every one of those stat ids in :mod:`espn_stats` for
    projections. So the league's actual values can be carried across instead of a
    preset's defaults, which matters for the leagues that differ: six-point passing
    touchdowns move quarterback value by roughly a round.

    A preset is still chosen, because the rest of the app labels leagues by one and
    ESPN's own rankings are requested by format. It comes off the reception value,
    which is what distinguishes the formats.
    """
    scoring = settings.get("scoringSettings") or {}
    items = scoring.get("scoringItems")
    te_position_id = next((k for k, v in POSITION_IDS.items() if v == "TE"), 4)

    overrides: dict[str, float] = {}
    reception = 0.0
    te_reception: float | None = None
    long_td_bonus = 0.0
    field_goal: float | None = None
    field_goal_short: float | None = None

    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        stat_id = _as_int(item.get("statId"))
        if stat_id is None:
            continue
        try:
            points = float(item.get("points") or 0.0)
        except (TypeError, ValueError):
            continue

        if stat_id in ESPN_YARDAGE_STATS:
            # 0 < points <= 1 is every real yardage rule (0.04, 0.1, 0.5). Anything
            # outside that is a payload this code has misread, and a bad divisor would
            # silently rescore the whole board.
            if 0 < points <= 1:
                overrides[ESPN_YARDAGE_STATS[stat_id]] = round(1.0 / points, 4)
            continue
        if stat_id in (espn_stats.PASS_TD_40_PLUS, espn_stats.RUSH_TD_40_PLUS,
                       espn_stats.REC_TD_40_PLUS):
            long_td_bonus = max(long_td_bonus, points)
            continue
        if stat_id == espn_stats.FG_MADE:
            field_goal = points
            continue
        if stat_id == espn_stats.FG_MADE_0_39:
            field_goal_short = points
            continue
        if stat_id == espn_stats.RECEPTIONS:
            reception = points
            per_position = item.get("pointsOverrides")
            if isinstance(per_position, dict):
                te_override = per_position.get(str(te_position_id))
                try:
                    te_reception = (
                        float(te_override) if te_override is not None else None
                    )
                except (TypeError, ValueError):
                    te_reception = None
        if stat_id in ESPN_PER_EVENT_STATS:
            overrides[ESPN_PER_EVENT_STATS[stat_id]] = points

    if long_td_bonus:
        overrides["bonus_long_td_40_plus"] = long_td_bonus
    # ESPN scores field goals either as one flat value or split by distance. The flat
    # value wins where both exist; the app models a single figure.
    made = field_goal if field_goal is not None else field_goal_short
    if made is not None:
        overrides["kick_fg_made"] = made

    te_bonus = 0.0
    if te_reception is not None and te_reception - reception >= 0.25:
        te_bonus = round(te_reception - reception, 3)
        overrides["te_premium_reception_bonus"] = te_bonus

    if te_bonus >= 0.4:
        preset = ScoringPreset.TE_PREMIUM
        note = (
            f"{reception} points per reception with a {te_bonus} tight-end bonus — read "
            "as TE premium."
        )
    elif reception >= 0.9:
        preset, note = ScoringPreset.FULL_PPR, "Read as full PPR."
    elif reception >= 0.4:
        preset, note = ScoringPreset.HALF_PPR, "Read as half PPR."
    elif reception > 0:
        preset = ScoringPreset.HALF_PPR
        note = (
            f"{reception} points per reception matches no standard format; half PPR was "
            "used as the label. The actual per-stat values were still imported."
        )
    elif isinstance(items, list) and items:
        preset, note = ScoringPreset.STANDARD, "Read as standard (no points per reception)."
    else:
        # No scoring table at all: fall back to ESPN's own format label, which is the
        # only thing left to go on.
        rank_type = str(scoring.get("playerRankType") or "").upper()
        preset = {
            "PPR": ScoringPreset.FULL_PPR, "STANDARD": ScoringPreset.STANDARD,
        }.get(rank_type, ScoringPreset.HALF_PPR)
        report.warn(
            "espn_scoring_missing",
            "ESPN did not return a scoring table, so scoring was taken from the format "
            f"label it does publish ({rank_type or 'none'} → {preset}). Check it on the "
            "League settings tab.",
        )
        return ScoringRules.from_preset(preset), preset

    report.info(
        "espn_scoring",
        f"{note} {len(overrides)} of your league's own per-stat values were imported, "
        "so projections are scored on your rules rather than a preset's defaults.",
    )
    return ScoringRules.from_preset(preset, **overrides), preset


def _espn_draft_slots(
    settings: dict[str, Any], picks: list[Any]
) -> dict[int, int]:
    """Map ESPN team id → draft slot.

    ``draftSettings.pickOrder`` is the authoritative answer and is published as soon as
    the order is set. Before that it is empty, so round one of a completed draft is used
    instead — which is the same fact recorded a different way.
    """
    draft_settings = settings.get("draftSettings") or {}
    order = draft_settings.get("pickOrder")
    slots: dict[int, int] = {}
    if isinstance(order, list):
        for index, team_id in enumerate(order, start=1):
            parsed = _as_int(team_id)
            if parsed is not None and parsed not in slots:
                slots[parsed] = index
    if slots:
        return slots
    for pick in picks:
        if not isinstance(pick, dict) or _as_int(pick.get("roundId")) != 1:
            continue
        team_id = _as_int(pick.get("teamId"))
        slot = _as_int(pick.get("roundPickNumber"))
        if team_id is not None and slot:
            slots.setdefault(team_id, slot)
    return slots


def _espn_managers(
    payload: dict[str, Any], slot_by_team: dict[int, int], team_count: int
) -> tuple[list[Manager], dict[int, int], str]:
    """Seat every ESPN team in a draft slot, returning the managers and the seating."""
    names = _espn_manager_names(payload)
    teams = [t for t in (payload.get("teams") or []) if isinstance(t, dict)]

    entries: list[tuple[int | None, int, str, str]] = []
    for team in teams:
        team_id = _as_int(team.get("id"))
        if team_id is None:
            continue
        entries.append((
            slot_by_team.get(team_id), team_id,
            names.get(team_id) or f"Team {team_id}", _espn_team_name(team),
        ))

    note = ""
    if any(slot is None for slot, _, _, _ in entries):
        note = (
            "ESPN has not published a draft order for this league yet, so managers were "
            "seated in the order ESPN lists them. Reorder them on the League settings "
            "tab if your real order differs."
        )

    taken = {slot for slot, _, _, _ in entries if slot is not None}
    free = [s for s in range(1, max(team_count, len(entries)) + 1) if s not in taken]
    managers: list[Manager] = []
    seating: dict[int, int] = {}
    for slot, team_id, name, team_name in entries:
        if slot is None:
            slot = free.pop(0) if free else len(managers) + 1
        seating[team_id] = int(slot)
        managers.append(
            Manager(
                name=name,
                draft_slot=int(slot),
                team_name=team_name,
                # Nothing is assumed about how a real manager drafts; their tendencies
                # come from their history.
                archetype=Archetype.BALANCED,
            )
        )
    managers.sort(key=lambda m: m.draft_slot)
    return managers, seating, note


def _espn_player_details(
    season: int, player_ids: list[int]
) -> dict[int, tuple[str, str, str]]:
    """Resolve ESPN player ids to ``(name, position, nfl_team)`` for one season.

    ESPN's draft board records a player id and nothing else, so without this a pick
    list is a column of numbers. Names are looked up per season rather than once,
    because ESPN reuses no ids across seasons but a 2021 draft needs 2021's roster.

    Completed drafts never change, so this caches for the full default term.
    """
    details: dict[int, tuple[str, str, str]] = {}
    ordered = sorted({int(pid) for pid in player_ids})
    for start in range(0, len(ordered), ESPN_PLAYER_CHUNK):
        chunk = ordered[start:start + ESPN_PLAYER_CHUNK]
        payload, _error = _espn_get(
            f"seasons/{season}/players",
            params="view=players_wl",
            cache_key=f"espn_player_names_{season}_{chunk[0]}_{len(chunk)}",
            ttl=DEFAULT_CACHE_TTL_SECONDS,
            headers={"x-fantasy-filter": json.dumps({"filterIds": {"value": chunk}})},
        )
        records = (
            payload if isinstance(payload, list)
            else (payload or {}).get("players", []) if isinstance(payload, dict)
            else []
        )
        for record in records if isinstance(records, list) else []:
            player = record.get("player", record) if isinstance(record, dict) else None
            if not isinstance(player, dict):
                continue
            player_id = _as_int(player.get("id"))
            name = clean_text(player.get("fullName"))
            if player_id is None or not name:
                continue
            details[player_id] = (
                name,
                POSITION_IDS.get(_as_int(player.get("defaultPositionId")), ""),
                PROTEAM_IDS.get(_as_int(player.get("proTeamId")), ""),
            )
    return details


def _espn_parse_draft(
    payload: dict[str, Any],
    *,
    season: int,
    league_name: str,
    current_names: dict[str, str],
    current_by_guid: dict[str, str] | None = None,
) -> HistoricalDraft | None:
    """Turn one season's ESPN payload into a historical draft, or ``None``.

    ``None`` covers three cases that are all "nothing to learn here": the draft has no
    real picks, it is still running, or ESPN would not tell us who the players were. The
    last one matters — picks without names cannot be joined to a player and would enter
    the profile as noise.

    What is deliberately *not* consulted is ``draftDetail.drafted``. For an upcoming
    season ESPN pre-creates the whole board with ``playerId: -1`` in every seat, and for
    a league whose draft was run offline it leaves ``drafted`` false on a board that is
    completely filled in. The picks themselves are the only honest signal, so a pick
    counts only once it names a real player.
    """
    detail = payload.get("draftDetail") or {}
    picks = [
        p for p in (detail.get("picks") or [])
        if isinstance(p, dict) and (_as_int(p.get("playerId")) or 0) > 0
    ]
    if detail.get("inProgress") or not picks:
        return None

    player_ids = [pid for pid in (_as_int(p.get("playerId")) for p in picks) if pid]
    details = _espn_player_details(season, player_ids)
    if not details:
        return None

    # Each season has its own membership, so names come from that season's payload —
    # then a manager still in the league is renamed to their current spelling, which is
    # what joins an old pick onto a present profile. The GUID is tried first and the name
    # only as a fallback: a manager who both renamed their team and changed their display
    # name is still the same person, and only the GUID knows it.
    season_names = _espn_manager_names(payload)
    season_owners = _espn_team_owner_ids(payload)
    by_guid = current_by_guid or {}
    draft = HistoricalDraft(
        season=season,
        league_name=league_name,
        platform=Platform.ESPN,
        source_file=f"espn league {payload.get('id')} draft {season}",
    )
    for pick in picks:
        player_id = _as_int(pick.get("playerId"))
        detail_row = details.get(player_id or -1)
        if detail_row is None:
            continue
        name, position, nfl_team = detail_row
        team_id = _as_int(pick.get("teamId"))
        raw_name = season_names.get(team_id or -1, "")
        manager_name = (
            next(
                (by_guid[guid] for guid in season_owners.get(team_id or -1, ()) if guid in by_guid),
                "",
            )
            or current_names.get(raw_name.lower(), raw_name)
            or (f"Team {team_id}" if team_id else "Unknown manager")
        )
        draft.picks.append(
            HistoricalPick(
                overall_pick=_as_int(pick.get("overallPickNumber")) or 0,
                round_number=_as_int(pick.get("roundId")),
                pick_in_round=_as_int(pick.get("roundPickNumber")),
                manager_name=manager_name,
                player_name=name,
                position=Position.coerce(position, None),
                nfl_team=nfl_team,
                league_name=league_name,
                platform=str(Platform.ESPN),
                season=season,
                is_keeper=bool(pick.get("keeper") or pick.get("reservedForKeeper")),
            )
        )
    return draft if draft.picks else None


def _espn_season_payload(
    league_id: str, season: int, headers: dict[str, str]
) -> tuple[dict[str, Any] | None, str]:
    """Read one past season of a league, trying both paths ESPN has used for it.

    The documented-by-folklore route for an old season is ``leagueHistory``, and as of
    now it answers 404 for every league and every season — including seasons that read
    perfectly well from the ordinary per-season path. So the per-season path is tried
    first and ``leagueHistory`` is kept only as a fallback, in case ESPN turns it back on.
    Whichever one answers, the error returned is the *first* one, because that is the
    failure that actually describes why the season is unavailable.
    """
    views = "view=mSettings&view=mTeam&view=mDraftDetail"
    attempts = (
        (f"seasons/{season}/segments/0/leagues/{league_id}", views),
        (f"leagueHistory/{league_id}", f"seasonId={season}&{views}"),
    )
    first_error = ""
    for index, (path, params) in enumerate(attempts):
        payload, error = _espn_get(
            path,
            params=params,
            cache_key=f"espn_league_history_{league_id}_{season}_{index}",
            ttl=DEFAULT_CACHE_TTL_SECONDS,
            headers=headers,
        )
        # leagueHistory answers with a single-element list.
        if isinstance(payload, list):
            payload = payload[0] if payload else None
        if isinstance(payload, dict):
            return payload, ""
        first_error = first_error or error or "unknown error"
    return None, first_error


def _espn_history(
    league_id: str,
    *,
    current_payload: dict[str, Any],
    current_season: int,
    managers: list[Manager],
    report: ValidationReport,
    max_seasons: int,
    headers: dict[str, str],
) -> tuple[DraftHistory, tuple[int, ...]]:
    """Collect every past ESPN draft this league has, newest first.

    ESPN lists which seasons exist in ``status.previousSeasons``, so the seasons are
    enumerated rather than guessed at. The current season is included when its draft is
    complete: a finished draft is history whether or not the season is.

    Making a league public opens the *current* season only — ESPN keeps the flag per
    season, so an anonymous read of last year's draft is refused even though this year's
    is wide open. That is a different problem from a season ESPN has genuinely lost, and
    it is reported differently, because the fix (the two cookies) is one the user has.
    """
    history = DraftHistory()
    seasons: list[int] = []
    current_names = {
        clean_text(manager.name).lower(): manager.name for manager in managers
    }
    # GUID → the name this manager goes by now, so an old pick lands on a present
    # profile regardless of what the team or the display name was called back then.
    current_team_names = _espn_manager_names(current_payload)
    current_by_guid = {
        guid: current_names.get(name.lower(), name)
        for team_id, guids in _espn_team_owner_ids(current_payload).items()
        if (name := clean_text(current_team_names.get(team_id, "")))
        for guid in guids
    }
    league_name = clean_text(
        (current_payload.get("settings") or {}).get("name")
    ) or f"ESPN league {league_id}"

    current = _espn_parse_draft(
        current_payload, season=current_season, league_name=league_name,
        current_names=current_names, current_by_guid=current_by_guid,
    )
    if current is not None:
        history.add(current)
        seasons.append(current_season)

    previous = (current_payload.get("status") or {}).get("previousSeasons")
    past = sorted(
        {s for s in (_as_int(v) for v in previous or []) if s and s < current_season},
        reverse=True,
    )
    unreadable: list[int] = []
    locked: list[int] = []
    for season in past[: max(0, int(max_seasons) - len(seasons))]:
        payload, error = _espn_season_payload(league_id, season, headers)
        if payload is None:
            (locked if "401" in error or "403" in error else unreadable).append(season)
            LOGGER.info("ESPN history %s unavailable: %s", season, error)
            continue
        parsed = _espn_parse_draft(
            payload, season=season, league_name=league_name,
            current_names=current_names, current_by_guid=current_by_guid,
        )
        if parsed is None:
            unreadable.append(season)
            continue
        history.add(parsed)
        seasons.append(season)

    if locked:
        listed = ", ".join(str(s) for s in sorted(locked))
        report.warn(
            "espn_history_locked",
            f"ESPN would not show season(s) {listed} without a login."
            + (
                " The cookies were accepted for this season but refused for those, which "
                "usually means the account they belong to was not in the league yet."
                if headers else
                " Making a league public only opens the current season — ESPN keeps that "
                "setting per season, so past drafts still need the two cookies from the "
                "**Private league, or past drafts** box above. Add them and connect "
                "again, or paste those recaps on the **Draft history** tab."
            ),
        )
    if unreadable:
        report.warn(
            "espn_history_gaps",
            f"No draft could be read for season(s) "
            f"{', '.join(str(s) for s in sorted(unreadable))}. ESPN drops draft detail "
            "for old seasons and for leagues that changed hands. Paste those recaps on "
            "the Draft history tab if you want them.",
        )
    if not history.drafts and not locked:
        report.warn(
            "espn_no_history",
            "No completed draft was found for this league, so opponents will be modelled "
            "from archetype priors rather than their own habits. If your draft is done, "
            "paste the recap on the Draft history tab.",
        )
    elif history.drafts:
        report.info(
            "espn_history",
            f"Found {len(history.all_picks)} picks across {len(history.drafts)} completed "
            f"draft(s): season(s) {', '.join(str(s) for s in sorted(set(seasons)))}.",
        )
    return history, tuple(sorted(set(seasons)))


def fetch_espn_league(
    league_ref: str,
    *,
    season: int | None = None,
    include_history: bool = True,
    max_seasons: int = MAX_HISTORY_SEASONS,
    espn_s2: str = "",
    swid: str = "",
) -> LeagueImportResult:
    """Build a :class:`League` and its draft history from an ESPN league URL or ID.

    A **public** league needs no credentials whatsoever. A **private** one needs the
    ``espn_s2`` and ``SWID`` cookies from a browser already signed in to it; both are
    optional here and are used for nothing but the request itself.

    ``league_ref`` is anything ESPN puts in front of a user: the league URL, a team
    URL, or the bare ID. ``season`` defaults to the season named in the URL, then to
    the current one — and if ESPN has no such league in that season, the season before
    is tried, because a league that has not rolled over yet only exists in the old one.

    Never raises. Every failure comes back as an error on the result, naming the paste
    importer as the route that always works.
    """
    result = LeagueImportResult()
    report = result.report
    league_id, url_season = espn_league_reference(league_ref)
    if not league_id:
        report.error(
            "espn_league_id",
            "That does not look like an ESPN league. Paste the whole URL from your "
            "league's page — for example "
            "`fantasy.espn.com/football/league?leagueId=123456` — or just the league ID.",
        )
        return result

    headers = _espn_cookie_header(espn_s2, swid)
    private = bool(headers)
    wanted = int(season or url_season or LeagueConfig().season)
    result.source = f"ESPN league {league_id} ({wanted})"

    payload: Any = None
    error = ""
    tried: list[int] = []
    for candidate in (wanted, wanted - 1):
        tried.append(candidate)
        payload, error = _espn_get(
            f"seasons/{candidate}/segments/0/leagues/{league_id}",
            params="view=mSettings&view=mTeam&view=mDraftDetail",
            cache_key=f"espn_league_{league_id}_{candidate}",
            headers=headers,
        )
        if isinstance(payload, dict) and payload.get("id") is not None:
            if candidate != wanted:
                report.warn(
                    "espn_season_fallback",
                    f"ESPN has no {wanted} season for this league, so {candidate} was "
                    f"read instead. Set the season on the League settings tab if you "
                    "are drafting a different one.",
                )
            wanted = candidate
            break
        # A missing league is worth retrying a season earlier — and against the live
        # endpoint, "missing" reads as 401 rather than 404, so an anonymous read retries
        # on both. With cookies a 401 means the cookies are wrong, and the season has
        # nothing to do with it, so that case stops here rather than doubling the wait.
        retryable = "404" in error or (not private and "401" in error)
        if not retryable:
            break
        payload = None

    if not isinstance(payload, dict) or payload.get("id") is None:
        report.error(
            "espn_league_unreachable",
            _espn_failure_message(error, league_id, tried[0], private),
        )
        return result

    result.source = f"ESPN league {league_id} ({wanted})"
    settings = payload.get("settings") or {}
    draft_settings = settings.get("draftSettings") or {}
    picks = [
        p for p in ((payload.get("draftDetail") or {}).get("picks") or [])
        if isinstance(p, dict)
    ]

    roster = _espn_roster(settings, report)
    scoring, preset = _espn_scoring(settings, report)
    slot_by_team = _espn_draft_slots(settings, picks)
    team_count = _as_int(settings.get("size")) or len(payload.get("teams") or []) or 12

    managers, seating, manager_note = _espn_managers(payload, slot_by_team, team_count)
    if manager_note:
        report.warn("espn_managers", manager_note)
    if not managers:
        report.error(
            "espn_no_managers",
            f"ESPN returned no teams for league {league_id}, so there are no opponents "
            "to model. If the league is private, the cookies may be for a different "
            "ESPN account than the one in the league.",
        )
        return result

    draft_type_label = str(draft_settings.get("type") or "").upper()
    if "AUCTION" in draft_type_label:
        report.warn(
            "espn_auction",
            "This is an auction league. The managers, scoring and roster came across, "
            "but the draft is simulated as a snake — auction bidding is not modelled.",
        )

    # ── Which slot is the user's ────────────────────────────────────────────
    # With cookies this is knowable: SWID *is* the member id, so the team it owns is
    # theirs. Without them nothing in a public payload says which team is the user's, and
    # slot 1 is the honest default. This is settled *before* the league is built so the
    # manager can be marked as the user — otherwise validation rightly complains that
    # nobody is, and the draft room simulates the user's own picks.
    user_slot = _espn_user_slot(payload, swid, seating)
    if user_slot:
        for manager in managers:
            manager.is_user = manager.draft_slot == user_slot
        report.info(
            "espn_user_slot",
            f"You were matched to draft slot {user_slot} from your ESPN account.",
        )
    else:
        report.warn(
            "espn_user_slot",
            "Your draft slot defaults to 1, because a public ESPN league does not say "
            "which team is yours. Set it on the League settings tab before drafting — "
            "connecting a private league picks it up automatically.",
        )

    # ESPN does not publish a round count; it is the lineup, which is what a round of
    # picks fills. Reserve seats are already excluded from the roster.
    rounds = roster.roster_size

    config = LeagueConfig(
        name=clean_text(settings.get("name")) or f"ESPN league {league_id}",
        season=wanted,
        platform=Platform.ESPN,
        team_count=len(managers),
        rounds=rounds,
        draft_type=DraftType.SNAKE,
        league_format=LeagueFormat.REDRAFT,
        scoring=scoring,
        roster=roster,
        user_draft_slot=user_slot or 1,
    )
    league = League(config=config, managers=managers)
    league_report = league.validate()
    report.extend(league_report)
    if not league_report.ok:
        # A league that fails its own validation would carry the failure into the draft
        # room. Report it and stop here instead.
        return result
    result.league = league

    report.info(
        "espn_league_loaded",
        f"Loaded '{config.name}': {len(managers)} managers, {rounds} rounds, "
        f"{preset} scoring, {wanted} season.",
    )

    if include_history:
        history, seasons = _espn_history(
            league_id,
            current_payload=payload,
            current_season=wanted,
            managers=managers,
            report=report,
            max_seasons=max_seasons,
            headers=headers,
        )
        result.history = history
        result.seasons_found = seasons

    return result


def _espn_user_slot(
    payload: dict[str, Any], swid: str, seating: dict[int, int]
) -> int | None:
    """Which draft slot belongs to the signed-in user, if that is knowable.

    ESPN's ``SWID`` cookie is the member GUID, so it identifies the user directly in
    the payload's own team ownership. No cookie, no answer — and guessing would be
    worse than the honest default of slot 1.
    """
    guid = clean_text(swid).upper().strip("{}")
    if not guid:
        return None
    for team in payload.get("teams") or []:
        if not isinstance(team, dict):
            continue
        owners = team.get("owners") if isinstance(team.get("owners"), list) else []
        ids = [str(o).upper().strip("{}") for o in owners]
        if team.get("primaryOwner"):
            ids.append(str(team["primaryOwner"]).upper().strip("{}"))
        if guid in ids:
            return seating.get(_as_int(team.get("id")) or -1)
    return None


def espn_league_instructions() -> str:
    """How the ESPN connect works, including the part that can fail.

    This text is the only place a user learns what the two cookie fields are for, so it
    says what they are, where they come from, and that they are not stored. It also
    states plainly that the endpoint is unofficial — a connect that breaks one August
    without warning is much less alarming if the app said in advance that it might.
    """
    return (
        "**Public leagues need nothing but the URL.** Paste your league's address and "
        "the app reads the league name, every team and manager, the scoring rules, the "
        "starting lineup, and this season's draft once it has been held.\n\n"
        "**Past drafts are the exception, and they need the two cookies below — even "
        "for a public league.** ESPN stores the public/private setting one season at a "
        "time, so making your league public opens this year and leaves last year shut. "
        "Past drafts are worth the extra step: they are what lets the model learn each "
        "manager's real habits instead of assuming an average one.\n\n"
        "**Private leagues need the same two cookies**, because ESPN refuses to answer "
        "otherwise. In a browser signed in to your league, open developer tools → "
        "*Application* (Chrome/Edge) or *Storage* (Firefox) → **Cookies** → "
        "`fantasy.espn.com`, and copy the values of `espn_s2` and `SWID`.\n\n"
        "Those two values are your ESPN session. This app sends them to ESPN and does "
        "nothing else with them: they are **never written to the database, never "
        "logged, and never shown back to you**. They live in the browser tab until you "
        "reload the page. They also stop working when you sign out of ESPN, at which "
        "point you copy them again.\n\n"
        "**If the connect fails, nothing is lost.** ESPN's league API is not a "
        "published, supported interface, so it can change without notice. Open your "
        "league → *Draft Recap*, copy the results, and paste them on the **Draft "
        "history** tab: that gives the opponent model exactly the same picks."
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
    "fetch_espn_league",
    "espn_league_reference",
    "sleeper_league_reference",
    "espn_league_instructions",
    "yahoo_league_instructions",
    "ESPN_LINEUP_SLOTS",
    "SLOT_MAP",
    "MAX_HISTORY_SEASONS",
]
