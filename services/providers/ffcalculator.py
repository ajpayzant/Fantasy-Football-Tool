"""Fantasy Football Calculator: real ADP with a real distribution around it.

This is the most valuable ADP source in the pipeline, for one reason: it publishes
``stdev``, ``high`` and ``low`` alongside the mean, computed from thousands of
actual mock drafts in a rolling recent window. Every other source gives a point
estimate.

That matters because the simulator models each player's draft position as a
*distribution*, not a number — the whole survival calculation ("will he last two
more picks?") depends on the spread. Given only a mean, the app has to invent a
spread; given FFC's, it does not.

The API is public and key-free (https://fantasyfootballcalculator.com/api-docs).
Formats are separate endpoints, and the sample size varies a lot between them, so
the draft count is surfaced rather than hidden.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from core.constants import NFL_TEAMS
from core.enums import ScoringPreset
from core.validation import ValidationReport
from services.providers.base import (
    DEFAULT_CACHE_TTL_SECONDS,
    ProviderResult,
    failed_result,
    fetch_json,
)

LOGGER = logging.getLogger("fantasy_mock_draft.providers.ffc")

BASE_URL = "https://fantasyfootballcalculator.com/api/v1/adp"

# The app's scoring presets mapped onto FFC's endpoint names. TE-premium has no
# FFC equivalent, so it borrows half-PPR: the alternative is no ADP at all, and
# the substitution is reported rather than hidden.
FORMAT_ENDPOINTS: dict[str, str] = {
    "standard": "standard",
    "half_ppr": "half-ppr",
    "full_ppr": "ppr",
    "te_premium": "half-ppr",
    "2qb": "2qb",
}

# Below this many drafts the mean is too noisy to treat as a market consensus.
# Chosen because FFC's own thin formats (dynasty, rookie) sit in the dozens while
# its main formats sit in the thousands.
MIN_CREDIBLE_DRAFTS = 200

# FFC reports DST as "DEF" and PK as "K" in some responses.
POSITION_OVERRIDES = {"DEF": "DST", "PK": "K", "D/ST": "DST"}


class FFCalculatorProvider:
    """Fetches ADP for one scoring format and team count."""

    key = "ffcalculator"
    label = "Fantasy Football Calculator"
    description = (
        "Consensus ADP from real mock drafts, with the standard deviation and "
        "earliest/latest pick for each player. Public API, no key required."
    )
    requires_credentials = False

    def fetch(
        self,
        *,
        scoring: ScoringPreset | str = ScoringPreset.HALF_PPR,
        team_count: int = 12,
        season: int | None = None,
        force_refresh: bool = False,
        ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        **_: Any,
    ) -> ProviderResult:
        """Return one row per player with ADP and its spread."""
        preset = ScoringPreset.coerce(scoring, ScoringPreset.HALF_PPR)
        endpoint = FORMAT_ENDPOINTS.get(str(preset), "half-ppr")

        query = f"teams={int(team_count)}"
        if season:
            query += f"&year={int(season)}"
        url = f"{BASE_URL}/{endpoint}?{query}"

        payload, outcome = fetch_json(
            url,
            cache_key=f"ffc_adp_{endpoint}_{team_count}_{season or 'current'}",
            ttl_seconds=ttl_seconds,
            force_refresh=force_refresh,
        )
        if not outcome.ok or not isinstance(payload, dict):
            return failed_result(
                self.label, outcome,
                hint="ADP spread will be estimated from ranks instead, which makes "
                     "survival odds less precise.",
            )

        report = ValidationReport()
        if str(payload.get("status", "")).lower() != "success":
            report.error(
                "ffc_status",
                f"Fantasy Football Calculator reported status "
                f"'{payload.get('status')}' for {endpoint}.",
            )
            return ProviderResult(
                pd.DataFrame(), self.label, url=url,
                fetched_at=outcome.fetched_at, report=report,
            )

        meta = payload.get("meta") or {}
        players = payload.get("players") or []
        if not players:
            report.error(
                "ffc_empty",
                f"No {endpoint} ADP data is published yet for "
                f"{season or 'the current season'}. Try a different scoring format, "
                "or a different season.",
            )
            return ProviderResult(
                pd.DataFrame(), self.label, url=url,
                fetched_at=outcome.fetched_at, report=report,
            )

        rows: list[dict[str, Any]] = []
        for record in players:
            if not isinstance(record, dict):
                continue
            name = str(record.get("name") or "").strip()
            if not name:
                continue
            position = POSITION_OVERRIDES.get(
                str(record.get("position") or "").upper(),
                str(record.get("position") or "").upper(),
            )
            team = str(record.get("team") or "").upper()
            rows.append({
                "ffc_id": _as_int(record.get("player_id")),
                "player_name": name,
                "position": position,
                "nfl_team": team if team in NFL_TEAMS else "",
                "bye_week": _as_int(record.get("bye")),
                "ffc_adp": _as_float(record.get("adp")),
                "ffc_adp_stdev": _as_float(record.get("stdev")),
                # FFC's "high" is the earliest pick a player went at, which is the
                # numerically smallest. Naming them min/max_pick here matches the
                # app's Player model and avoids the high/low confusion downstream.
                "ffc_min_pick": _as_int(record.get("high")),
                "ffc_max_pick": _as_int(record.get("low")),
                "ffc_times_drafted": _as_int(record.get("times_drafted")),
            })

        frame = pd.DataFrame(rows)
        if frame.empty:
            report.error("ffc_unparsed", "ADP rows could not be parsed.")
            return ProviderResult(
                frame, self.label, url=url, fetched_at=outcome.fetched_at, report=report
            )

        total_drafts = _as_int(meta.get("total_drafts")) or 0
        window = f"{meta.get('start_date', '?')} to {meta.get('end_date', '?')}"
        report.info(
            "ffc_sample",
            f"{len(frame)} players from {total_drafts:,} real {endpoint} mock drafts "
            f"({window}).",
        )
        if total_drafts and total_drafts < MIN_CREDIBLE_DRAFTS:
            report.warn(
                "ffc_thin_sample",
                f"Only {total_drafts} drafts back this ADP, so the numbers are noisy. "
                f"A format with a larger sample will give steadier survival odds.",
            )
        if str(preset) == "te_premium":
            report.warn(
                "ffc_no_te_premium",
                "Fantasy Football Calculator does not publish TE-premium ADP, so "
                "half-PPR ADP was used. Tight ends will be undervalued relative to "
                "your scoring.",
            )

        LOGGER.info(
            "FFC %s (%d teams): %d players from %s drafts",
            endpoint, team_count, len(frame), total_drafts,
        )
        return ProviderResult(
            frame=frame,
            source=self.label,
            url=url,
            fetched_at=outcome.fetched_at,
            from_cache=outcome.from_cache,
            cache_age_seconds=outcome.cache_age_seconds,
            stale_fallback=outcome.stale_fallback,
            season=season,
            scoring_format=endpoint,
            report=report,
            notes=f"{total_drafts:,} drafts, {window}",
        )

    def available_formats(self) -> list[str]:
        """The app-side scoring presets this provider can serve."""
        return sorted(FORMAT_ENDPOINTS)


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "" or value == "-":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    parsed = _as_float(value)
    return None if parsed is None else int(round(parsed))


__all__ = ["FFCalculatorProvider", "FORMAT_ENDPOINTS", "BASE_URL", "MIN_CREDIBLE_DRAFTS"]
