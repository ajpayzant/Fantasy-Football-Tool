"""Yahoo: average draft position from Yahoo's public read API.

Yahoo runs a read-only public mirror of its fantasy API at
``pub-api-ro.fantasysports.yahoo.com`` which answers without OAuth for public
league data. The ``draft_analysis`` subresource on the player collection is what
this provider reads: it returns average pick, average round and percent drafted,
aggregated across Yahoo's leagues.

Yahoo also returns an average auction cost per player. It is deliberately not
read: this app does not simulate auctions, so carrying the column would mean
storing and displaying a number nothing can act on.

Two structural things about this API:

* **Yahoo's JSON is XML wearing a JSON coat.** Every value is buried in nested
  lists of single-key dicts, and the nesting depth varies by field. Rather than
  hard-coding paths that break on any change, :func:`_flatten` walks each player
  record and collects every scalar it finds. That is resilient to Yahoo moving a
  field one level up or down, which it does.

* **It pages 25 at a time.** ``count`` caps at 25 regardless of what is asked
  for, so a full board needs sequential requests. Each page is a separate cache
  entry, so a partial failure costs one page rather than the whole fetch.

Yahoo reports ``"-"`` for players with no draft data — a real string, not null —
which is why the numeric coercion treats it explicitly.
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
    fetch_json,
)

LOGGER = logging.getLogger("fantasy_mock_draft.providers.yahoo")

# ``nfl.l.public`` is Yahoo's own placeholder league for public season data; it is
# what the API documents for reading league-independent player information.
BASE_URL = (
    "https://pub-api-ro.fantasysports.yahoo.com/fantasy/v2/league/nfl.l.public/players"
)

# Yahoo caps a page at 25 no matter what ``count`` asks for.
PAGE_SIZE = 25

# 300 players is ~25 rounds of a 12-team draft: past that, ADP is noise and the
# players are not being drafted. Twelve requests rather than four hundred.
DEFAULT_PLAYER_LIMIT = 300

# "AR" is Yahoo's actual-rank sort, which orders by draft position rather than by
# its editorial ranking — the order that matters for a draft board.
SORT_KEY = "AR"

POSITION_OVERRIDES = {"DEF": "DST", "D/ST": "DST", "PK": "K"}
FANTASY_POSITIONS = frozenset({"QB", "RB", "WR", "TE", "K", "DST"})

# Yahoo's placeholder for "no draft data for this player".
NO_DATA = "-"


class YahooProvider:
    """Fetches Yahoo ADP via the public read API."""

    key = "yahoo"
    label = "Yahoo"
    description = (
        "Yahoo's average draft pick, round and percent drafted, aggregated across "
        "Yahoo leagues. Public read API, no login required."
    )
    requires_credentials = False

    def fetch(
        self,
        *,
        player_limit: int = DEFAULT_PLAYER_LIMIT,
        force_refresh: bool = False,
        ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        **_: Any,
    ) -> ProviderResult:
        """Page through Yahoo's player list and return draft-analysis rows."""
        report = ValidationReport()
        rows: list[dict[str, Any]] = []
        pages_attempted = 0
        pages_failed = 0
        last_error = ""
        fetched_at = ""
        from_cache_pages = 0
        oldest_cache_age: float | None = None
        url = ""

        for start in range(0, max(PAGE_SIZE, int(player_limit)), PAGE_SIZE):
            url = (
                f"{BASE_URL};position=ALL;count={PAGE_SIZE};start={start};"
                f"sort={SORT_KEY}/draft_analysis?format=json"
            )
            payload, outcome = fetch_json(
                url,
                cache_key=f"yahoo_draft_analysis_{start}_{PAGE_SIZE}",
                ttl_seconds=ttl_seconds,
                force_refresh=force_refresh,
            )
            pages_attempted += 1
            if not outcome.ok or not isinstance(payload, dict):
                pages_failed += 1
                last_error = outcome.error or "unknown error"
                LOGGER.warning("Yahoo page at start=%d failed: %s", start, last_error)
                # One bad page should not discard the pages that worked; Yahoo
                # returns an error for a page past the end of the list, which is
                # also how the natural end of the data is detected.
                break
            fetched_at = outcome.fetched_at or fetched_at
            if outcome.from_cache:
                from_cache_pages += 1
                if outcome.cache_age_seconds is not None:
                    oldest_cache_age = max(
                        oldest_cache_age or 0.0, outcome.cache_age_seconds
                    )

            page_rows = _parse_page(payload)
            if not page_rows:
                break  # end of the list
            rows.extend(page_rows)

        if not rows:
            report.error(
                "yahoo_unavailable",
                f"Yahoo returned no draft data ({last_error or 'empty response'}). "
                "Other ADP sources are unaffected.",
            )
            return ProviderResult(
                pd.DataFrame(), self.label, url=url, fetched_at=fetched_at, report=report
            )

        frame = pd.DataFrame(rows).drop_duplicates(subset=["yahoo_id"], keep="first")

        # Players with no Yahoo draft data at all carry no signal and would only
        # add unmatched rows to the join.
        drafted = frame[frame["yahoo_adp"].notna()].copy()
        dropped = len(frame) - len(drafted)

        if drafted.empty:
            report.error(
                "yahoo_no_adp",
                "Yahoo returned players but none had an average draft pick.",
            )
            return ProviderResult(
                pd.DataFrame(), self.label, url=url, fetched_at=fetched_at, report=report
            )

        report.info(
            "yahoo_counts",
            f"{len(drafted)} players with a Yahoo average draft pick, from "
            f"{pages_attempted} page(s) of {PAGE_SIZE}.",
        )
        if dropped:
            report.info(
                "yahoo_undrafted",
                f"Ignored {dropped} player(s) Yahoo reports no draft data for.",
            )
        if pages_failed:
            report.warn(
                "yahoo_partial",
                f"Stopped after {pages_failed} failed page ({last_error}). The "
                f"{len(drafted)} players retrieved are still usable, but deeper "
                "players may be missing.",
            )

        LOGGER.info("Yahoo: %d players with ADP over %d pages", len(drafted), pages_attempted)
        return ProviderResult(
            frame=drafted.reset_index(drop=True),
            source=self.label,
            url=url,
            fetched_at=fetched_at,
            from_cache=from_cache_pages == pages_attempted and pages_attempted > 0,
            cache_age_seconds=oldest_cache_age,
            report=report,
            notes=f"{len(drafted)} players, {pages_attempted} page(s)",
        )


def _flatten(node: Any, into: dict[str, Any]) -> None:
    """Collect every scalar in Yahoo's nested list-of-dicts into a flat mapping.

    Yahoo nests each field at an inconsistent depth and reorders them between
    responses, so walking for scalars is more robust than indexing known paths.
    Later values win, which is harmless because the keys collected here are unique
    within a player record.
    """
    if isinstance(node, list):
        for item in node:
            _flatten(item, into)
    elif isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, (str, int, float)) or value is None:
                into[key] = value
            else:
                _flatten(value, into)


def _parse_page(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract player rows from one Yahoo page, tolerating shape changes."""
    league = (payload.get("fantasy_content") or {}).get("league")
    if not isinstance(league, list):
        return []

    for block in league:
        if not isinstance(block, dict) or "players" not in block:
            continue
        players = block["players"]
        if not isinstance(players, dict):
            continue
        try:
            count = int(players.get("count") or 0)
        except (TypeError, ValueError):
            count = 0

        rows: list[dict[str, Any]] = []
        for index in range(count):
            entry = players.get(str(index))
            if not isinstance(entry, dict):
                continue
            fields: dict[str, Any] = {}
            _flatten(entry.get("player"), fields)

            name = str(fields.get("full") or "").strip()
            if not name:
                continue
            position = str(
                fields.get("display_position") or fields.get("primary_position") or ""
            ).upper()
            # Multi-eligible players come through as "WR,RB"; the first is primary.
            position = position.split(",")[0].strip()
            position = POSITION_OVERRIDES.get(position, position)
            if position not in FANTASY_POSITIONS:
                continue

            team = str(fields.get("editorial_team_abbr") or "").upper()
            rows.append({
                "yahoo_id": str(fields.get("player_id") or "") or None,
                "player_name": name,
                "position": position,
                "nfl_team": team if team in NFL_TEAMS else "",
                "yahoo_adp": _as_float(fields.get("average_pick")),
                "yahoo_avg_round": _as_float(fields.get("average_round")),
                "yahoo_percent_drafted": _as_float(fields.get("percent_drafted")),
                "yahoo_preseason_adp": _as_float(fields.get("preseason_average_pick")),
            })
        return rows
    return []


def _as_float(value: Any) -> float | None:
    """Coerce a Yahoo numeric, treating its ``"-"`` placeholder as missing."""
    if value is None or value == "" or value == NO_DATA:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["YahooProvider", "BASE_URL", "PAGE_SIZE", "DEFAULT_PLAYER_LIMIT"]
