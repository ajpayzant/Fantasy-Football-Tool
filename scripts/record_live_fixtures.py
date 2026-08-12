"""Record trimmed copies of real provider payloads for offline tests.

The live pipeline is four undocumented third-party endpoints joined together. The
only honest way to test it is against payloads those endpoints actually returned,
so this script takes what is currently in the on-disk fetch cache and writes
smaller versions of it into ``tests/fixtures/live_payloads/``.

Why record rather than hand-write the fixtures:

* A hand-written payload encodes what *I* believe the shape is. These four APIs are
  undocumented, and the shape assumptions are exactly what the tests need to pin
  down — a fixture that agrees with a wrong assumption tests nothing.
* Shape changes are the failure mode the provider layer is built to survive. When
  ESPN moves a field, re-running this script against a fresh cache and watching the
  tests fail is the signal.

Trimming is by *row count only*: every field of every kept record is preserved
verbatim, because a field dropped here is a field the tests can never catch a
regression in. Sizes are reduced roughly 40x, which is what makes it reasonable to
keep them in the repository.

Usage (needs the fetch cache populated, i.e. run the app's fetch once first)::

    python scripts/record_live_fixtures.py
    python scripts/record_live_fixtures.py --refresh   # fetch first, then record
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enums import ScoringPreset  # noqa: E402
from services.providers.base import cache_directory  # noqa: E402

LOGGER = logging.getLogger("fantasy_mock_draft.scripts.record_fixtures")

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "fixtures", "live_payloads",
)

# How many records to keep per source. Chosen so the joins the resolver performs are
# still non-trivial — a 20-player fixture would join perfectly by luck — while the
# files stay small enough to read and to commit.
SLEEPER_KEEP = 700
ESPN_KEEP = 400
# Every page the app asks for (YAHOO_PLAYER_LIMIT / PAGE_SIZE). Recording fewer
# would make the offline board look like a partial Yahoo outage, so a test could
# not tell a real regression from a short fixture.
YAHOO_PAGES = 12


def _read_cache_json(name: str) -> Any | None:
    """Load one cached payload, or ``None`` when it has not been fetched yet."""
    path = os.path.join(cache_directory(), f"{name}.json.gz")
    if not os.path.exists(path):
        LOGGER.warning("No cached payload for %s — skipping", name)
        return None
    with gzip.open(path, "rb") as handle:
        return json.loads(handle.read())


def _write_fixture(name: str, payload: Any) -> int:
    """Write one fixture and return its size in bytes."""
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    path = os.path.join(FIXTURE_DIR, f"{name}.json.gz")
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    with gzip.open(path, "wb") as handle:
        handle.write(encoded)
    return os.path.getsize(path)


def trim_sleeper(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the most-drafted fantasy-relevant players, plus every team defence.

    Ordered by Sleeper's own ``search_rank`` so the kept players are the ones any
    ADP source will also carry, which is what makes the join meaningful. Defences
    are kept unconditionally: they are the join the resolver has a special rule for
    (Sleeper's crosswalk has no ids for them), so a fixture without them would skip
    the one case most likely to break.
    """
    positions = {"QB", "RB", "WR", "TE", "K", "DEF", "DST"}
    ranked: list[tuple[float, str]] = []
    defences: dict[str, Any] = {}
    for player_id, record in payload.items():
        if not isinstance(record, dict):
            continue
        position = str(record.get("position") or "").upper()
        if position not in positions:
            continue
        if position in {"DEF", "DST"}:
            defences[player_id] = record
            continue
        rank = record.get("search_rank")
        ranked.append((float(rank) if isinstance(rank, (int, float)) else 1e9, player_id))

    ranked.sort()
    kept = {pid: payload[pid] for _, pid in ranked[:SLEEPER_KEEP]}
    kept.update(defences)
    LOGGER.info(
        "Sleeper: %d players kept (%d defences) from %d",
        len(kept), len(defences), len(payload),
    )
    return kept


def trim_espn(payload: Any) -> Any:
    """Keep the best-ranked records that carry a rank or an ADP.

    The filter mirrors the provider's own, so the fixture is the useful part of a
    39 MB payload of which the provider discards roughly 90%.

    Sorting by rank before trimming is the part that matters. ESPN returns its
    players in no useful order — the first records in the payload had ranks 776,
    2546 and 209 — so slicing in payload order produced a fixture of deep
    practice-squad players that no ADP source carries, and the resolver's ESPN join
    then looked broken when it was the fixture that was wrong.
    """
    records = payload if isinstance(payload, list) else (payload or {}).get("players", [])
    useful: list[tuple[float, int, Any]] = []
    for index, record in enumerate(records):
        player = record.get("player", record) if isinstance(record, dict) else None
        if not isinstance(player, dict):
            continue
        ranks = player.get("draftRanksByRankType") or {}
        available = [
            (block or {}).get("rank") for block in ranks.values()
            if (block or {}).get("rank") is not None
        ]
        has_adp = (player.get("ownership") or {}).get("averageDraftPosition") is not None
        if not available and not has_adp:
            continue
        # Best (numerically lowest) rank across rank types; un-ranked-but-owned
        # players sort last. The index breaks ties so the sort is stable and the
        # fixture is byte-identical across runs on the same payload.
        best = float(min(available)) if available else 1e9
        useful.append((best, index, record))

    useful.sort(key=lambda entry: (entry[0], entry[1]))
    kept = [record for _, _, record in useful[:ESPN_KEEP]]
    LOGGER.info(
        "ESPN: %d records kept from %d (%d had a rank or ADP)",
        len(kept), len(records), len(useful),
    )
    # Preserved as the same envelope shape the endpoint returns, list or dict, so
    # the provider's shape handling is exercised rather than bypassed.
    return kept if isinstance(payload, list) else {**payload, "players": kept}


def record(*, scoring: ScoringPreset = ScoringPreset.HALF_PPR, teams: int = 12,
           season: int | None = None) -> dict[str, int]:
    """Record every fixture that has a cached payload. Returns name → size."""
    from services.providers.ffcalculator import FORMAT_ENDPOINTS
    from services.providers.sleeper import SleeperProvider

    resolved_season = season or SleeperProvider().current_season()
    endpoint = FORMAT_ENDPOINTS.get(str(scoring), "half-ppr")
    written: dict[str, int] = {}

    state = _read_cache_json("sleeper_state")
    if state is not None:
        written["sleeper_state"] = _write_fixture("sleeper_state", state)

    sleeper = _read_cache_json("sleeper_players_nfl")
    if sleeper is not None:
        written["sleeper_players_nfl"] = _write_fixture(
            "sleeper_players_nfl", trim_sleeper(sleeper)
        )

    ffc_key = f"ffc_adp_{endpoint}_{teams}_{resolved_season or 'current'}"
    ffc = _read_cache_json(ffc_key)
    if ffc is not None:
        # Not trimmed: 208 players is already small, and FFC is the primary ADP
        # source, so the fixture should be the whole board it publishes.
        written[ffc_key] = _write_fixture(ffc_key, ffc)

    for rank_type in ("PPR", "STANDARD"):
        espn_key = f"espn_players_{resolved_season}_{rank_type}"
        espn = _read_cache_json(espn_key)
        if espn is not None:
            written[espn_key] = _write_fixture(espn_key, trim_espn(espn))

    for start in range(0, YAHOO_PAGES * 25, 25):
        yahoo_key = f"yahoo_draft_analysis_{start}_25"
        yahoo = _read_cache_json(yahoo_key)
        if yahoo is not None:
            written[yahoo_key] = _write_fixture(yahoo_key, yahoo)

    manifest = {
        "season": resolved_season,
        "scoring": str(scoring),
        "team_count": teams,
        "ffc_cache_key": ffc_key,
        "yahoo_pages": YAHOO_PAGES,
        "fixtures": sorted(written),
        "note": (
            "Recorded from real provider responses by scripts/record_live_fixtures.py. "
            "Trimmed by row count only; every field of every kept record is verbatim."
        ),
    }
    path = os.path.join(FIXTURE_DIR, "manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    LOGGER.info("Wrote manifest for season %s", resolved_season)
    return written


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true",
        help="Fetch live first so the cache is current, then record.",
    )
    parser.add_argument("--teams", type=int, default=12)
    parser.add_argument("--season", type=int, default=None)
    args = parser.parse_args()

    if args.refresh:
        from services.live import build_live_board

        LOGGER.info("Fetching live data so the cache is current…")
        result = build_live_board(team_count=args.teams, season=args.season)
        LOGGER.info("Fetch: %s", result.summary())

    written = record(teams=args.teams, season=args.season)
    if not written:
        LOGGER.error(
            "Nothing recorded — the fetch cache is empty. Run with --refresh, or "
            "fetch once from the app's Setup page first."
        )
        return 1
    total = sum(written.values())
    for name in sorted(written):
        LOGGER.info("  %-40s %8.1f KB", name, written[name] / 1024)
    LOGGER.info("Recorded %d fixture(s), %.1f KB total", len(written), total / 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
