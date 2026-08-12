"""ESPN: draft ranks and average draft position from ESPN's own player database.

ESPN exposes its fantasy player data through an endpoint that answers without
authentication for public season data (``lm-api-reads.fantasy.espn.com``). This
provider reads only that public season-level data. Private-league access is a
separate concern handled in :mod:`services.providers.leagues`, where the user
supplies their own cookie.

Two quirks worth knowing before editing this module:

* **The endpoint ignores the ``limit`` in ``x-fantasy-filter``.** It returns the
  whole player database (~11,500 records, ~39 MB) regardless. It is fast (under a
  second) but the payload is large, so the cache TTL matters more here than
  anywhere else. Filtering happens client-side.

* **ESPN's ADP is not a mock-draft consensus.** It is the average pick across
  ESPN's own leagues, which skews toward casual drafters who follow ESPN's
  rankings. That is a genuinely different signal from a mock-draft site's ADP,
  which is why the pipeline keeps both rather than averaging them away.

The same payload also carries ESPN's projected stat line for the season, which is
the app's only source of a *real* projection — without it the pool falls back to a
curve fitted to draft position, which is a restatement of ADP rather than an
independent opinion. Decoding it is involved enough to live in
:mod:`services.providers.espn_stats`; see that module for why ESPN's own
``appliedTotal`` is not trustworthy.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from core.config import ScoringRules
from core import stats as core_stats
from core.enums import Position, ScoringPreset
from core.validation import ValidationReport
from services.providers import espn_stats
from services.providers.base import (
    DEFAULT_CACHE_TTL_SECONDS,
    ProviderResult,
    failed_result,
    fetch_json,
)

LOGGER = logging.getLogger("fantasy_mock_draft.providers.espn")

BASE_URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons"

# ESPN's internal position ids, confirmed against live payloads by sampling a
# named player at each id (1=EJ Manuel QB, 2=Fozzy Whittaker RB, 16=Falcons D/ST).
POSITION_IDS: dict[int, str] = {
    1: "QB",
    2: "RB",
    3: "WR",
    4: "TE",
    5: "K",
    16: "DST",
}

# ESPN's internal NFL team ids. Derived rather than transcribed: each id was
# resolved by joining ESPN player ids to Sleeper's team field via Sleeper's
# ``espn_id`` crosswalk, and every one of the 32 resolved at 100% agreement
# across all sampled players. Id 31/32 are unused by ESPN; 0 means free agent.
PROTEAM_IDS: dict[int, str] = {
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN",
    8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LAR",
    15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI",
    22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WAS",
    29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU",
}

# Which of ESPN's rank types to read for each of the app's scoring presets.
# ESPN publishes STANDARD and PPR only; half-PPR sits between them, and PPR is the
# closer of the two for the receiving-heavy players where the gap actually bites.
RANK_TYPES: dict[str, str] = {
    "standard": "STANDARD",
    "half_ppr": "PPR",
    "full_ppr": "PPR",
    "te_premium": "PPR",
}


class ESPNProvider:
    """Fetches ESPN draft ranks, ADP and ownership for a season."""

    key = "espn"
    label = "ESPN"
    description = (
        "ESPN's own draft ranks, average draft position and ownership percentage. "
        "Public season data, no login required."
    )
    requires_credentials = False

    def fetch(
        self,
        *,
        season: int,
        scoring: ScoringPreset | str = ScoringPreset.HALF_PPR,
        scoring_rules: ScoringRules | None = None,
        force_refresh: bool = False,
        ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        **_: Any,
    ) -> ProviderResult:
        """Return one row per ranked or owned ESPN player.

        ``scoring_rules`` is what the projected stat lines are scored under. Pass the
        user's actual league rules where they are known; the preset's defaults are
        used otherwise, which is right for a board built before a league exists.
        """
        preset = ScoringPreset.coerce(scoring, ScoringPreset.HALF_PPR)
        rank_type = RANK_TYPES.get(str(preset), "PPR")
        rules = scoring_rules or ScoringRules.from_preset(preset)

        url = f"{BASE_URL}/{int(season)}/players?scoringPeriodId=0&view=kona_player_info"
        # Requested even though ESPN ignores the limit: the sort makes the payload
        # order meaningful, and if ESPN ever honours the filter this asks for the
        # right thing rather than needing a change.
        fantasy_filter = json.dumps({
            "players": {
                "limit": 1500,
                "sortDraftRanks": {
                    "sortPriority": 1, "sortAsc": True, "value": rank_type,
                },
            }
        })

        payload, outcome = fetch_json(
            url,
            cache_key=f"espn_players_{season}_{rank_type}",
            headers={"x-fantasy-filter": fantasy_filter},
            ttl_seconds=ttl_seconds,
            force_refresh=force_refresh,
            timeout_seconds=60.0,  # ~39 MB
        )
        if not outcome.ok:
            return failed_result(
                self.label, outcome,
                hint="ESPN ranks will be missing; other ADP sources still work.",
            )

        report = ValidationReport()
        records = payload if isinstance(payload, list) else (payload or {}).get("players", [])
        if not isinstance(records, list) or not records:
            report.error(
                "espn_shape",
                "ESPN returned an unexpected payload shape. The endpoint may have "
                "changed — ESPN's fantasy API is undocumented and can move without "
                "notice.",
            )
            return ProviderResult(
                pd.DataFrame(), self.label, url=url,
                fetched_at=outcome.fetched_at, report=report,
            )

        rows: list[dict[str, Any]] = []
        for record in records:
            player = record.get("player", record) if isinstance(record, dict) else None
            if not isinstance(player, dict):
                continue
            position = POSITION_IDS.get(player.get("defaultPositionId"))
            if position is None:
                continue  # IDP, coaches, team QB aggregates — not draftable here

            ranks = player.get("draftRanksByRankType") or {}
            rank_block = ranks.get(rank_type) or {}
            rank = rank_block.get("rank")
            ownership = player.get("ownership") or {}
            adp = ownership.get("averageDraftPosition")

            # A player with neither a rank nor an ADP contributes nothing and would
            # only dilute the join. ESPN carries thousands of these.
            if rank is None and adp is None:
                continue

            name = str(player.get("fullName") or "").strip()
            if not name:
                continue

            # Scored under `rules`, not read off ESPN's own total — see espn_stats.
            parsed_position = Position.coerce(position, None)
            stats = espn_stats.season_projection_stats(player, season)
            stat_totals = espn_stats.to_stat_line(stats, parsed_position)
            projection = core_stats.score(stat_totals, parsed_position, rules)
            stat_line = core_stats.labelled(stat_totals, parsed_position)

            rows.append({
                "espn_id": str(player.get("id")) if player.get("id") is not None else None,
                "player_name": name,
                "position": position,
                "nfl_team": PROTEAM_IDS.get(player.get("proTeamId"), ""),
                "espn_rank": _as_int(rank),
                "espn_adp": _as_float(adp),
                "espn_projection": (
                    round(float(projection), 1) if projection is not None else None
                ),
                "espn_stat_line": _format_stat_line(stat_line),
                # The stats themselves, not just the points they came to. This is what
                # lets a change of scoring rules rescore the board in place instead of
                # sending the user back to the network for numbers the app already has.
                "espn_stat_totals": core_stats.to_frame_value(stat_totals),
                "espn_percent_owned": _as_float(ownership.get("percentOwned")),
                "injury_status": str(player.get("injuryStatus") or "").upper(),
                "espn_injured": bool(player.get("injured")),
            })

        frame = pd.DataFrame(rows)
        if frame.empty:
            report.error(
                "espn_empty",
                f"ESPN returned {len(records)} records but none carried a "
                f"{rank_type} rank or ADP for {season}.",
            )
            return ProviderResult(
                frame, self.label, url=url, fetched_at=outcome.fetched_at, report=report
            )

        with_adp = int(frame["espn_adp"].notna().sum())
        with_projection = int(frame["espn_projection"].notna().sum())
        report.info(
            "espn_counts",
            f"{len(frame)} players carry an ESPN {rank_type} rank; {with_adp} also "
            f"have an average draft position.",
        )
        report.info(
            "espn_projections",
            f"{with_projection} of {len(frame)} players carry an ESPN projected stat "
            f"line, scored under your league's rules ({preset}). Players without one "
            "fall back to an estimate from draft position, and the Player Pool page "
            "labels which is which.",
        )
        if with_projection < len(frame) * 0.5:
            report.warn(
                "espn_projection_coverage",
                f"Only {with_projection} of {len(frame)} ESPN players had a projected "
                "stat line. Before ESPN publishes pre-season projections this is "
                "expected; the board still works, on ADP alone.",
            )
        if str(preset) == "half_ppr":
            report.info(
                "espn_rank_substitution",
                "ESPN publishes standard and full-PPR ranks only, so its PPR ranks "
                "were used for half-PPR.",
            )
        report.warn(
            "espn_adp_population",
            "ESPN's ADP is the average pick in ESPN's own leagues, not a mock-draft "
            "consensus — it leans toward drafters following ESPN's rankings. It is "
            "kept as a separate column rather than blended away.",
        )

        LOGGER.info(
            "ESPN %s (%s): %d ranked players, %d with ADP",
            season, rank_type, len(frame), with_adp,
        )
        return ProviderResult(
            frame=frame,
            source=self.label,
            url=url,
            fetched_at=outcome.fetched_at,
            from_cache=outcome.from_cache,
            cache_age_seconds=outcome.cache_age_seconds,
            season=season,
            scoring_format=rank_type,
            report=report,
            notes=f"{rank_type} ranks, {with_adp} players with ADP",
        )


def _format_stat_line(line: dict[str, float]) -> str:
    """Render a projected stat line as one short human-readable string.

    Kept as a string rather than a nested structure because it travels through the
    importer and out to CSV, and its only consumer is a person reading it to see
    what a projection is made of.
    """
    if not line:
        return ""
    parts = [
        f"{value:,.0f} {label.lower()}" if abs(value) >= 10 else f"{value:.1f} {label.lower()}"
        for label, value in line.items()
        if label != "Games"
    ]
    return " · ".join(parts)


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        return None if parsed == 0.0 else parsed
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["ESPNProvider", "POSITION_IDS", "PROTEAM_IDS", "RANK_TYPES", "BASE_URL"]
