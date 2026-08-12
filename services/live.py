"""One entry point the UI calls to get a real, current draft board.

This module is the seam between the provider layer and the app. It exists so the
Setup page contains no fetching logic and no join logic — it calls
:func:`build_live_board` and renders what comes back.

Two things it guarantees:

* **It always returns something usable, or says exactly why not.** Sources are
  fetched independently and the board is built from whatever answered. Nothing
  raises; every failure arrives as a message on the report.

* **It never invents a person.** The no-history fallback
  (:func:`archetype_opponents`) produces opponents named for their draft slot and
  their *labelled* tendency — "Slot 4 · Zero-RB tendency" — not fictional humans.
  A generic opponent is visibly generic, so it can never be mistaken for a read on
  someone real.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.config import LeagueConfig, RosterSettings, ScoringRules
from core.enums import Archetype, DraftType, LeagueFormat, Platform, ScoringPreset, Slot
from core.validation import ValidationReport
from models.league import League
from models.manager import Manager
from models.player import PlayerPool
from services.importers import import_player_pool
from services.providers import (
    ESPNProvider,
    FFCalculatorProvider,
    SleeperProvider,
    YahooProvider,
)
from services.providers.resolver import ResolvedBoard, board_to_import_frame, resolve_board

LOGGER = logging.getLogger("fantasy_mock_draft.live")

# How many Yahoo players to page for. Yahoo pages 25 at a time and its ADP is a
# corroborating signal rather than the primary one, so this is deliberately capped:
# 300 covers ~25 rounds, past which ADP carries no information anyway.
YAHOO_PLAYER_LIMIT = 300

# The archetype mix used when a league has no draft history. These are the
# tendencies that actually recur in real redraft leagues, and they are spread so
# the simulated field is not uniform — a room where every opponent behaves
# identically produces confidently wrong survival odds.
#
# Order matters: opponents are assigned round-robin from this list by draft slot,
# so a 12-team league gets all twelve and a 10-team league gets the first ten.
FALLBACK_ARCHETYPES: tuple[Archetype, ...] = (
    Archetype.BALANCED,
    Archetype.RANK_FOLLOWER,
    Archetype.BEST_PLAYER_AVAILABLE,
    Archetype.ROBUST_RB,
    Archetype.LATE_QB,
    Archetype.BALANCED,
    Archetype.ZERO_RB,
    Archetype.HERO_RB,
    Archetype.RANK_FOLLOWER,
    Archetype.ELITE_TE,
    Archetype.HIGH_VARIANCE,
    Archetype.EARLY_QB,
)

# Human-readable labels for the fallback opponents. Phrased as a *tendency*, not a
# name, so the UI cannot present one as a person.
ARCHETYPE_LABELS: dict[Archetype, str] = {
    Archetype.BALANCED: "Balanced",
    Archetype.BEST_PLAYER_AVAILABLE: "Best-available",
    Archetype.ZERO_RB: "Zero-RB",
    Archetype.HERO_RB: "Hero-RB",
    Archetype.ROBUST_RB: "Robust-RB",
    Archetype.EARLY_QB: "Early-QB",
    Archetype.LATE_QB: "Late-QB",
    Archetype.ELITE_TE: "Elite-TE",
    Archetype.ROOKIE_HEAVY: "Rookie-heavy",
    Archetype.RANK_FOLLOWER: "Rank-following",
    Archetype.HIGH_VARIANCE: "Unpredictable",
    Archetype.HOMER: "Home-team",
    Archetype.AUTODRAFT: "Autodraft",
    Archetype.CUSTOM: "Custom",
}

# Default lineup when the user has not said otherwise. The most common redraft
# shape: 1QB/2RB/2WR/1TE/1FLEX/1K/1DST and a six-man bench across 15 rounds.
DEFAULT_SLOTS: dict[Slot, int] = {
    Slot.QB: 1, Slot.RB: 2, Slot.WR: 2, Slot.TE: 1,
    Slot.FLEX: 1, Slot.K: 1, Slot.DST: 1, Slot.BENCH: 6,
}


@dataclass(slots=True)
class LiveBoardResult:
    """A live player pool, the board it came from, and full provenance."""

    pool: PlayerPool | None = None
    board: ResolvedBoard | None = None
    report: ValidationReport = field(default_factory=ValidationReport)
    fetched_at: str = ""
    season: int | None = None
    scoring_format: str = ""
    source_status: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.pool is not None and len(self.pool) > 0

    def summary(self) -> str:
        """One line naming what was loaded and from where."""
        if not self.ok:
            return "No live player data loaded."
        live = [
            info.get("label", key)
            for key, info in self.source_status.items()
            if info.get("ok")
        ]
        return (
            f"{len(self.pool)} players, {self.scoring_format or 'unknown'} scoring, "
            f"{self.season or '?'} season — from {', '.join(live) or 'no source'}."
        )


def current_season() -> int:
    """The fantasy season Sleeper itself considers current.

    Read from Sleeper rather than the wall clock because the fantasy season rolls
    over in spring, so "this year" is the wrong answer for several months.
    """
    try:
        season = SleeperProvider().current_season()
        if season:
            return int(season)
    except Exception:  # noqa: BLE001 — a failed lookup must not block the fetch
        LOGGER.warning("Could not read the current season from Sleeper", exc_info=True)
    return datetime.now(timezone.utc).year


def build_live_board(
    *,
    scoring: ScoringPreset | str = ScoringPreset.HALF_PPR,
    team_count: int = 12,
    season: int | None = None,
    league: LeagueConfig | None = None,
    use_espn: bool = True,
    use_yahoo: bool = True,
    force_refresh: bool = False,
) -> LiveBoardResult:
    """Fetch every enabled source, join them, and build a real player pool.

    Each source is fetched independently: ESPN being down costs ESPN's columns and
    nothing else. The result carries the per-source status so the UI can show which
    sources answered and how fresh each one is.
    """
    resolved_season = int(season or current_season())
    preset = ScoringPreset.coerce(scoring, ScoringPreset.HALF_PPR)
    result = LiveBoardResult(season=resolved_season, scoring_format=str(preset))
    report = result.report

    LOGGER.info(
        "Fetching live board: season=%s scoring=%s teams=%s espn=%s yahoo=%s refresh=%s",
        resolved_season, preset, team_count, use_espn, use_yahoo, force_refresh,
    )

    sleeper = SleeperProvider().fetch(force_refresh=force_refresh)
    ffc = FFCalculatorProvider().fetch(
        scoring=preset, team_count=team_count, season=resolved_season,
        force_refresh=force_refresh,
    )
    espn = (
        ESPNProvider().fetch(
            season=resolved_season,
            scoring=preset,
            # ESPN's projected *stat lines* are scored under these rules, so passing
            # the real league makes the projections format-aware: a TE-premium or
            # 6-point-passing-TD league gets projections that reflect it rather than
            # ESPN's fixed default scoring.
            scoring_rules=league.scoring if league is not None else None,
            force_refresh=force_refresh,
        )
        if use_espn else None
    )
    if espn is not None and not espn.ok:
        # ESPN is the one source that can fail for a local reason rather than a
        # network one: its endpoint ignores every documented filter and returns the
        # entire ~39 MB player database, which needs several hundred MB to decode.
        # Worth saying so plainly, because "turn ESPN off" is a fix the user can
        # actually apply and the board is complete without it.
        report.info(
            "espn_optional",
            "ESPN did not load. Its endpoint returns its whole player database "
            "(~39 MB) and cannot be filtered server-side, so it is the one source "
            "that can fail on a busy machine. The board is built from the other "
            "sources; untick ESPN to skip it entirely.",
        )
    yahoo = (
        YahooProvider().fetch(
            player_limit=YAHOO_PLAYER_LIMIT, force_refresh=force_refresh
        )
        if use_yahoo else None
    )

    board = resolve_board(
        sleeper=sleeper, ffc=ffc, espn=espn, yahoo=yahoo,
        season=resolved_season, team_count=team_count, scoring_format=str(preset),
    )
    result.board = board
    report.extend(board.report)

    # Labels for the status table, so the UI does not hard-code source names.
    labels = {
        "sleeper": "Sleeper", "ffc": "Fantasy Football Calculator",
        "espn": "ESPN", "yahoo": "Yahoo",
    }
    for key, info in board.source_status.items():
        info["label"] = labels.get(key, key)
    result.source_status = board.source_status

    if not board.ok:
        report.error(
            "live_board_empty",
            "No live player board could be built. Every data source either failed or "
            "returned nothing usable. You can still import your own rankings file on "
            "the Player pool tab.",
        )
        return result

    fetched_at = ""
    for candidate in (ffc, espn, yahoo, sleeper):
        if candidate is not None and candidate.ok and candidate.fetched_at:
            fetched_at = candidate.fetched_at
            break
    result.fetched_at = fetched_at

    import_result = import_player_pool(
        board_to_import_frame(board.frame),
        league=league,
        source=_source_label(board),
        season=resolved_season,
        platform=Platform.CUSTOM,
        # The whole point of this module: this data is real, so it is never flagged
        # as sample data and never carries the sample-data banner.
        is_sample_data=False,
        imported_at=fetched_at,
    )
    report.extend(import_result.report)
    if import_result.pool is None or not len(import_result.pool):
        report.error(
            "live_import_failed",
            "The live board was fetched but could not be turned into a player pool. "
            "This is a bug — the messages above say which rows were rejected.",
        )
        return result

    result.pool = import_result.pool
    LOGGER.info(
        "Live board ready: %d players for %s %s",
        len(import_result.pool), resolved_season, preset,
    )
    return result


def _source_label(board: ResolvedBoard) -> str:
    """A provenance string naming the sources that actually contributed."""
    live = [
        info.get("label", key) for key, info in board.source_status.items()
        if info.get("ok")
    ]
    return f"live: {', '.join(live)}" if live else "live"


def archetype_opponents(
    team_count: int = 12,
    *,
    user_slot: int = 1,
    user_name: str = "You",
) -> list[Manager]:
    """Generic opponents for a league with no draft history.

    Each opponent is named for their draft slot and their tendency — "Slot 4 ·
    Zero-RB tendency" — because that is exactly what they are. No fictional human
    names: an opponent the model knows nothing specific about must not *look* like
    an opponent the model has read.

    The tendencies come from :data:`FALLBACK_ARCHETYPES`, which the engine already
    has priors for, so these opponents draft plausibly from the first pick.
    """
    count = max(2, int(team_count))
    slot = min(max(1, int(user_slot)), count)
    managers: list[Manager] = []
    for index in range(1, count + 1):
        if index == slot:
            managers.append(
                Manager(name=user_name or "You", draft_slot=index, is_user=True)
            )
            continue
        archetype = FALLBACK_ARCHETYPES[(index - 1) % len(FALLBACK_ARCHETYPES)]
        label = ARCHETYPE_LABELS.get(archetype, str(archetype))
        managers.append(
            Manager(
                name=f"Slot {index} · {label} tendency",
                draft_slot=index,
                team_name=f"Slot {index}",
                archetype=archetype,
            )
        )
    LOGGER.info(
        "Built %d generic archetype opponents (user in slot %d)", count - 1, slot
    )
    return managers


def quick_league(
    *,
    name: str = "My League",
    team_count: int = 12,
    rounds: int = 15,
    user_slot: int = 1,
    scoring: ScoringPreset | str = ScoringPreset.HALF_PPR,
    season: int | None = None,
) -> League:
    """A ready-to-draft league with generic opponents and a standard lineup.

    This is the no-history path: a user who has not connected a league and has no
    past drafts to import still gets a full, working draft room immediately. Its
    opponents are visibly generic (see :func:`archetype_opponents`).
    """
    preset = ScoringPreset.coerce(scoring, ScoringPreset.HALF_PPR)
    config = LeagueConfig(
        name=name.strip() or "My League",
        season=int(season or current_season()),
        platform=Platform.CUSTOM,
        team_count=max(2, int(team_count)),
        rounds=max(1, int(rounds)),
        draft_type=DraftType.SNAKE,
        league_format=LeagueFormat.REDRAFT,
        scoring=ScoringRules.from_preset(preset),
        roster=RosterSettings(slots=dict(DEFAULT_SLOTS)),
        user_draft_slot=min(max(1, int(user_slot)), max(2, int(team_count))),
    )
    return League(
        config=config,
        managers=archetype_opponents(
            config.team_count, user_slot=config.user_draft_slot
        ),
    )


__all__ = [
    "LiveBoardResult",
    "build_live_board",
    "archetype_opponents",
    "quick_league",
    "current_season",
    "FALLBACK_ARCHETYPES",
    "ARCHETYPE_LABELS",
    "DEFAULT_SLOTS",
    "YAHOO_PLAYER_LIMIT",
]
