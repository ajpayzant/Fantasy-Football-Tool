"""Sleeper: the player universe, and the ID crosswalk that joins every source.

Sleeper publishes a documented, key-free read API (https://docs.sleeper.com). Two
things make it the backbone of the pipeline rather than just another source:

* It carries ``espn_id`` and ``yahoo_id`` on each player, which is what lets ESPN
  and Yahoo data be joined by identifier instead of by fuzzy name matching.
* It has real roster metadata — team, depth chart position, injury status, years
  of experience — which no ADP source provides.

It does **not** publish ADP, so it is never the sole source for a draft board.

The endpoint returns every player who has ever been in the database (~12,000
records, ~14 MB), so the fetch is cached aggressively and filtered down to active
fantasy-relevant players on the way in.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from core.constants import NFL_TEAMS
from core.validation import ValidationReport
from services.providers.base import (
    DEFAULT_CACHE_TTL_SECONDS,
    ProviderResult,
    failed_result,
    fetch_json,
)

LOGGER = logging.getLogger("fantasy_mock_draft.providers.sleeper")

PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
STATE_URL = "https://api.sleeper.app/v1/state/nfl"

# The positions a redraft fantasy league actually drafts. Sleeper also carries
# offensive linemen, long snappers and individual defensive players, none of
# which belong on a standard draft board.
FANTASY_POSITIONS = frozenset({"QB", "RB", "WR", "TE", "K", "DEF"})

# Sleeper's team code for a defence differs from the app's, and its "DEF"
# position maps to the app's DST.
POSITION_OVERRIDES = {"DEF": "DST"}

# Sleeper uses a sentinel rank for players it has no draft interest in. Treating
# it as a real rank would sort undrafted practice-squad players onto the board.
UNRANKED_SENTINEL = 9999999


class SleeperProvider:
    """Fetches the Sleeper player universe."""

    key = "sleeper"
    label = "Sleeper"
    description = (
        "Player universe, teams, injury status and cross-platform IDs. Documented "
        "public API, no key required. Does not provide ADP."
    )
    requires_credentials = False

    def current_season(self, *, force_refresh: bool = False) -> int | None:
        """Sleeper's own idea of the current NFL season.

        Used instead of the wall clock because the fantasy season rolls over in
        the spring and a calendar year would ask other providers for a season
        that has no ADP yet.
        """
        payload, outcome = fetch_json(
            STATE_URL, cache_key="sleeper_state", ttl_seconds=6 * 60 * 60,
            force_refresh=force_refresh,
        )
        if not outcome.ok or not isinstance(payload, dict):
            LOGGER.warning("Could not read Sleeper season state: %s", outcome.error)
            return None
        try:
            return int(payload.get("season") or 0) or None
        except (TypeError, ValueError):
            return None

    def fetch(
        self,
        *,
        force_refresh: bool = False,
        ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        include_inactive: bool = False,
        **_: Any,
    ) -> ProviderResult:
        """Return one row per fantasy-relevant player."""
        payload, outcome = fetch_json(
            PLAYERS_URL,
            cache_key="sleeper_players_nfl",
            ttl_seconds=ttl_seconds,
            force_refresh=force_refresh,
            timeout_seconds=90.0,  # ~14 MB payload
        )
        if not outcome.ok or not isinstance(payload, dict):
            return failed_result(
                self.label, outcome,
                hint="Player identity and injury status will be missing, but ADP "
                     "sources can still populate a board.",
            )

        report = ValidationReport()
        rows: list[dict[str, Any]] = []
        skipped_no_name = 0

        for player_id, record in payload.items():
            if not isinstance(record, dict):
                continue
            position = str(record.get("position") or "").upper()
            if position not in FANTASY_POSITIONS:
                continue
            if not include_inactive and not record.get("active"):
                continue

            name = record.get("full_name") or " ".join(
                part for part in (record.get("first_name"), record.get("last_name")) if part
            )
            if not name.strip():
                skipped_no_name += 1
                continue

            team = str(record.get("team") or "").upper()
            search_rank = record.get("search_rank")
            if isinstance(search_rank, (int, float)) and search_rank >= UNRANKED_SENTINEL:
                search_rank = None

            rows.append({
                "sleeper_id": str(player_id),
                "player_name": name.strip(),
                "position": POSITION_OVERRIDES.get(position, position),
                "nfl_team": team if team in NFL_TEAMS else "",
                "experience": record.get("years_exp"),
                "injury_status": record.get("injury_status") or "",
                "depth_chart_order": record.get("depth_chart_order"),
                "age": record.get("age"),
                "sleeper_search_rank": search_rank,
                "espn_id": _as_id(record.get("espn_id")),
                "yahoo_id": _as_id(record.get("yahoo_id")),
                "status": record.get("status") or "",
            })

        if not rows:
            report.error(
                "sleeper_empty",
                "Sleeper returned no fantasy-relevant players. The payload shape "
                "may have changed.",
            )
            return ProviderResult(
                pd.DataFrame(), self.label, url=outcome.url,
                fetched_at=outcome.fetched_at, report=report,
            )

        frame = pd.DataFrame(rows)

        # A free agent is a real state, not an error — but a board full of them
        # means the season rolled over and Sleeper has not repopulated teams yet.
        teamless = int((frame["nfl_team"] == "").sum())
        if teamless:
            report.info(
                "sleeper_free_agents",
                f"{teamless} of {len(frame)} players have no NFL team (free agents "
                "or not on a depth chart).",
            )
        if skipped_no_name:
            report.info(
                "sleeper_unnamed", f"Skipped {skipped_no_name} record(s) with no name."
            )
        report.info(
            "sleeper_ids",
            f"{int(frame['espn_id'].notna().sum())} players carry an ESPN id and "
            f"{int(frame['yahoo_id'].notna().sum())} a Yahoo id, which is how the "
            "sources are joined.",
        )

        LOGGER.info("Sleeper: %d fantasy-relevant players", len(frame))
        return ProviderResult(
            frame=frame,
            source=self.label,
            url=outcome.url,
            fetched_at=outcome.fetched_at,
            from_cache=outcome.from_cache,
            cache_age_seconds=outcome.cache_age_seconds,
            stale_fallback=outcome.stale_fallback,
            report=report,
            notes="Player identity, teams and injury status. No ADP.",
        )


def _as_id(value: Any) -> str | None:
    """Normalise a cross-platform id to a string, or ``None`` if absent.

    Sleeper returns these as ints, strings, or null depending on the player, and a
    join on mixed types silently matches nothing.
    """
    if value is None or value == "":
        return None
    try:
        return str(int(value))
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None


__all__ = ["SleeperProvider", "PLAYERS_URL", "STATE_URL", "FANTASY_POSITIONS"]
