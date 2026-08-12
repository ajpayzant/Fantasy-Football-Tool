"""Join the live sources into one player board, keeping every source visible.

The problem this module solves: four sources describe the same players with three
different identifier schemes and four different naming conventions, and none of
them agree on what a team defence is called.

**How the join works, in precedence order.** Sleeper is the spine because it is
the only source carrying cross-platform identifiers:

1. **Team defences join on the NFL team code.** No source shares an identifier for
   a defence, and the four naming conventions observed live are mutually
   incompatible — Sleeper says "Houston Texans", FFC "Houston Defense", ESPN
   "Texans D/ST", Yahoo "Texans". For a DST the team *is* the identity, so that is
   the key.
2. **ESPN and Yahoo join on their own ids** via Sleeper's ``espn_id`` and
   ``yahoo_id`` crosswalk. This is exact and needs no name comparison at all.
3. **Everything else joins on a normalised name plus position.** Measured live,
   this matches 192/208 FFC players (92.3%) with every single miss being a team
   defence — which rule 1 then catches. There is deliberately **no fuzzy string
   matching**: the residual after rules 1–3 is small enough that fuzzy matching
   would risk inventing a wrong join to fix a handful of rows, and a wrong join
   silently attributes one player's ADP to another.

**What the blend does and does not do.** Sources are blended into a consensus ADP
by weight, but every source's own value is kept in its own column, and the count
of contributing sources travels with each row. A player with one source's ADP and
a player with four are not presented as equally known.

The one source with a genuine distribution — FFC publishes ``stdev``, ``high`` and
``low`` — is preferred for spread. Where it is absent the spread is estimated from
round depth, and that estimate is flagged as an estimate rather than passed off as
measured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.validation import ValidationReport
from services.normalize import normalize_team, player_key
from services.providers.base import ProviderResult

LOGGER = logging.getLogger("fantasy_mock_draft.providers.resolver")

# Default weight per source in the consensus ADP. FFC leads because it is the only
# source computed from actual mock drafts *and* the only one publishing a spread;
# ESPN and Yahoo are platform-league averages, which is a real but differently
# biased signal (their drafters follow their own site's rankings).
DEFAULT_ADP_WEIGHTS: dict[str, float] = {
    "ffc": 0.50,
    "espn": 0.25,
    "yahoo": 0.25,
}

# Columns holding each source's ADP, and the weight key they map to.
ADP_COLUMNS: dict[str, str] = {
    "ffc_adp": "ffc",
    "espn_adp": "espn",
    "yahoo_adp": "yahoo",
}

# Beyond this disagreement between the earliest and latest source ADP for one
# player, the sources are telling materially different stories and the consensus
# is worth flagging rather than quietly averaging. 24 picks is two full rounds of
# a 12-team draft.
ADP_DISAGREEMENT_PICKS = 24.0

# Spread estimate for players with no FFC data, as a fraction of ADP. Derived from
# the shape of FFC's own published stdev: early picks are tightly clustered and
# late picks are nearly arbitrary, and stdev/ADP is roughly flat across the board
# at about a fifth. Used only as a fallback, and always labelled as estimated.
ESTIMATED_STDEV_FRACTION = 0.20
MIN_ESTIMATED_STDEV = 3.0


@dataclass(slots=True)
class ResolvedBoard:
    """The merged player board, plus what happened while merging it."""

    frame: pd.DataFrame
    report: ValidationReport = field(default_factory=ValidationReport)
    source_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    season: int | None = None
    scoring_format: str = ""
    team_count: int = 12

    @property
    def ok(self) -> bool:
        return not self.frame.empty

    @property
    def player_count(self) -> int:
        return 0 if self.frame is None else int(len(self.frame))

    def successful_sources(self) -> list[str]:
        return [key for key, info in self.source_status.items() if info.get("ok")]

    def failed_sources(self) -> list[str]:
        return [key for key, info in self.source_status.items() if not info.get("ok")]


# Columns every source shares. They are never merged from a secondary source,
# because the spine already has them and re-merging would create _x/_y pairs.
SHARED_COLUMNS = frozenset({"player_name", "position", "nfl_team", "join_key", "bye_week"})


def _column_or_blank(frame: pd.DataFrame, name: str) -> pd.Series:
    """A frame's column, or an all-empty column of the right length.

    Keeps :func:`_join_keys` working on a source that drops a column, instead of
    zipping against an empty series and silently producing zero keys.
    """
    if name in frame.columns:
        return frame[name]
    return pd.Series([""] * len(frame), index=frame.index, dtype=object)


def _join_keys(frame: pd.DataFrame) -> pd.Series:
    """Build the name+position join key for every row.

    Team defences get a team-based key instead, because their names are
    irreconcilable across sources while their team codes are not.
    """
    keys: list[str] = []
    for name, position, team in zip(
        _column_or_blank(frame, "player_name"),
        _column_or_blank(frame, "position"),
        _column_or_blank(frame, "nfl_team"),
        strict=True,
    ):
        position_text = str(position or "").strip().upper()
        if position_text == "DST":
            team_text = normalize_team(team)
            # A defence with no resolvable team cannot be identified by team at all;
            # fall back to its name so the row stays self-consistent rather than
            # collapsing every teamless defence into one key.
            keys.append(
                f"dst_{team_text}" if team_text and team_text != "FA"
                else player_key(name, "DST")
            )
        else:
            keys.append(player_key(name, position_text))
    return pd.Series(keys, index=frame.index, dtype=object)


def resolve_board(
    *,
    sleeper: ProviderResult | None = None,
    ffc: ProviderResult | None = None,
    espn: ProviderResult | None = None,
    yahoo: ProviderResult | None = None,
    adp_weights: dict[str, float] | None = None,
    season: int | None = None,
    team_count: int = 12,
    scoring_format: str = "",
    drop_unranked: bool = True,
) -> ResolvedBoard:
    """Merge whichever provider results are available into one board.

    Every argument is optional: the board is built from whatever succeeded. If no
    source with ADP or ranks succeeded the result is empty and the report says so,
    which the UI turns into a message rather than a crash.
    """
    report = ValidationReport()
    status: dict[str, dict[str, Any]] = {}

    def note(key: str, result: ProviderResult | None) -> None:
        if result is None:
            status[key] = {"ok": False, "rows": 0, "detail": "not requested"}
            return
        status[key] = {
            "ok": result.ok,
            "rows": result.row_count,
            "detail": result.freshness_label(),
            "url": result.url,
            "notes": result.notes,
        }
        # Provider-level messages are re-raised here so the Setup page can show
        # every source's issues in one place.
        for issue in result.report.errors:
            report.error(issue.code, f"{result.source}: {issue.message}")
        for issue in result.report.warnings:
            report.warn(issue.code, f"{result.source}: {issue.message}")
        for issue in result.report.infos:
            report.info(issue.code, f"{result.source}: {issue.message}")

    note("sleeper", sleeper)
    note("ffc", ffc)
    note("espn", espn)
    note("yahoo", yahoo)

    # The spine: Sleeper if it worked, otherwise the widest ADP source available.
    # Without Sleeper there is no crosswalk, so ESPN and Yahoo can only be joined
    # by name — which still works, just less exactly.
    spine_source = ""
    if sleeper is not None and sleeper.ok:
        board = sleeper.frame.copy()
        spine_source = "sleeper"
    else:
        fallback = next(
            (r for r in (ffc, espn, yahoo) if r is not None and r.ok), None
        )
        if fallback is None:
            report.error(
                "no_sources",
                "No live data source could be reached, so no player board could be "
                "built. Check your connection, or import a player file on Setup.",
            )
            return ResolvedBoard(pd.DataFrame(), report, status, season, scoring_format, team_count)
        board = fallback.frame.copy()
        spine_source = fallback.source
        report.warn(
            "no_crosswalk",
            f"Sleeper was unavailable, so {fallback.source} is the base list and "
            "sources are joined by name only. Some players may not merge.",
        )

    board["join_key"] = _join_keys(board)
    board = board.drop_duplicates(subset=["join_key"], keep="first")

    # ── Merge each ADP source ────────────────────────────────────────────────
    merge_stats: dict[str, tuple[int, int]] = {}

    def merge(result: ProviderResult | None, key: str, id_column: str | None) -> None:
        """Fill one source's columns onto the board, by id first, then by name.

        Implemented as column-wise ``map`` rather than ``DataFrame.merge`` on
        purpose: a merge against a source holding two rows for one id duplicates
        board rows, which would put the same player on the board twice. Mapping
        can only ever fill cells, so the board's row count is invariant.
        """
        nonlocal board
        if result is None or not result.ok:
            return
        incoming = result.frame.copy()
        incoming["join_key"] = _join_keys(incoming)

        value_columns = [
            column for column in incoming.columns
            if column not in SHARED_COLUMNS and column not in board.columns
        ]
        for column in value_columns:
            board[column] = np.nan

        # `matched` counts board rows this source actually contributed to. The
        # anchor is the source's ADP column where it has one, because that is the
        # value the consensus depends on.
        anchor = next(
            (c for c in (f"{key}_adp", *value_columns) if c in value_columns), None
        )

        def fill_from(mapping_key: str, rows: pd.DataFrame) -> None:
            """Fill empty board cells from `rows`, keyed on `mapping_key`."""
            if mapping_key not in board.columns or rows.empty:
                return
            deduped = rows.drop_duplicates(subset=[mapping_key], keep="first")
            board_keys = board[mapping_key]
            for column in value_columns:
                lookup = deduped.set_index(mapping_key)[column]
                # Index must be unique for `.map`; drop_duplicates above assures it.
                filled = board_keys.map(lookup)
                board[column] = board[column].where(board[column].notna(), filled)

        # Pass 1: exact id join through Sleeper's crosswalk. Nothing to compare by
        # name, so this cannot mis-join.
        if id_column and id_column in board.columns and id_column in incoming.columns:
            with_id = incoming[incoming[id_column].notna()]
            fill_from(id_column, with_id)

        # Pass 2: fill the remaining gaps by name+position (team code, for defences).
        fill_from("join_key", incoming[incoming["join_key"].astype(bool)])

        matched = int(board[anchor].notna().sum()) if anchor else 0

        # Bye week is worth taking from FFC: Sleeper does not publish it, and it
        # drives the bye-clash warnings in the UI.
        if "bye_week" in incoming.columns:
            bye_map = (
                incoming[incoming["bye_week"].notna()]
                .drop_duplicates(subset=["join_key"], keep="first")
                .set_index("join_key")["bye_week"]
            )
            mapped = board["join_key"].map(bye_map)
            if "bye_week" in board.columns:
                board["bye_week"] = board["bye_week"].where(board["bye_week"].notna(), mapped)
            else:
                board["bye_week"] = mapped

        merge_stats[key] = (matched, result.row_count)
        LOGGER.info(
            "Merged %s: %d of %d rows matched onto the board",
            result.source, matched, result.row_count,
        )

    merge(ffc, "ffc", None)          # FFC has no cross-platform id at all
    merge(espn, "espn", "espn_id")
    merge(yahoo, "yahoo", "yahoo_id")

    for key, (matched, total) in merge_stats.items():
        if total and matched < total * 0.85:
            report.warn(
                f"{key}_join_rate",
                f"Only {matched} of {total} {key.upper()} players matched onto the "
                f"board. Unmatched players keep their other sources' values, but "
                f"their {key.upper()} ADP is not contributing.",
            )
        else:
            report.info(
                f"{key}_join_rate", f"{matched} of {total} {key.upper()} rows merged."
            )

    # ── Consensus ADP ────────────────────────────────────────────────────────
    weights = dict(adp_weights or DEFAULT_ADP_WEIGHTS)
    present = [c for c in ADP_COLUMNS if c in board.columns]
    if not present:
        report.error(
            "no_adp",
            "No source supplied an average draft position, so there is no draft "
            "board to build. Every ADP provider failed.",
        )
        return ResolvedBoard(pd.DataFrame(), report, status, season, scoring_format, team_count)

    weighted_sum = pd.Series(0.0, index=board.index)
    weight_total = pd.Series(0.0, index=board.index)
    for column in present:
        weight = float(weights.get(ADP_COLUMNS[column], 0.0))
        if weight <= 0:
            continue
        values = pd.to_numeric(board[column], errors="coerce")
        available = values.notna()
        weighted_sum = weighted_sum.add(values.fillna(0.0) * weight, fill_value=0.0)
        weight_total = weight_total.add(available.astype(float) * weight, fill_value=0.0)

    # Re-normalised by the weight actually present, so a player seen only by FFC
    # gets FFC's number rather than FFC's number scaled down by the missing 50%.
    board["overall_adp"] = np.where(weight_total > 0, weighted_sum / weight_total, np.nan)
    board["adp_source_count"] = board[present].notna().sum(axis=1).astype(int)

    numeric_adp = board[present].apply(pd.to_numeric, errors="coerce")
    board["adp_disagreement"] = (numeric_adp.max(axis=1) - numeric_adp.min(axis=1)).round(1)

    # ── Spread: measured where possible, estimated where not ─────────────────
    if "ffc_adp_stdev" in board.columns:
        measured = pd.to_numeric(board["ffc_adp_stdev"], errors="coerce")
    else:
        measured = pd.Series(np.nan, index=board.index)
    estimated = (
        pd.to_numeric(board["overall_adp"], errors="coerce") * ESTIMATED_STDEV_FRACTION
    ).clip(lower=MIN_ESTIMATED_STDEV)
    board["adp_stdev"] = measured.where(measured.notna(), estimated)
    board["adp_stdev_is_estimated"] = measured.isna() & board["overall_adp"].notna()

    if "ffc_min_pick" in board.columns:
        board["min_pick"] = pd.to_numeric(board["ffc_min_pick"], errors="coerce")
    if "ffc_max_pick" in board.columns:
        board["max_pick"] = pd.to_numeric(board["ffc_max_pick"], errors="coerce")

    # ── Ranks ────────────────────────────────────────────────────────────────
    # Overall rank is derived from consensus ADP rather than taken from any one
    # source, so the board's own ordering and its ADP can never disagree.
    ranked = board["overall_adp"].notna()
    board["overall_rank"] = np.nan
    board.loc[ranked, "overall_rank"] = (
        board.loc[ranked, "overall_adp"].rank(method="first").astype(int)
    )
    board["position_rank"] = np.nan
    for position, group in board[ranked].groupby("position"):
        board.loc[group.index, "position_rank"] = (
            group["overall_adp"].rank(method="first").astype(int)
        )
    # ``platform_*`` is the single "what one platform thinks" channel the engine's
    # PLATFORM_ADP ranking source reads, and ESPN fills it because it is the only
    # source publishing both a rank and an ADP. Yahoo's and Sleeper's numbers are not
    # lost — they travel in their own named columns all the way to the Player Pool.
    if "espn_rank" in board.columns:
        board["platform_rank"] = pd.to_numeric(board["espn_rank"], errors="coerce")
    if "espn_adp" in board.columns:
        board["platform_adp"] = pd.to_numeric(board["espn_adp"], errors="coerce")

    # ── Trim ─────────────────────────────────────────────────────────────────
    if drop_unranked:
        before = len(board)
        board = board[board["overall_adp"].notna()].copy()
        dropped = before - len(board)
        if dropped:
            report.info(
                "dropped_unranked",
                f"Excluded {dropped} player(s) with no ADP from any source — mostly "
                "practice-squad and deep-roster players no one drafts.",
            )

    if board.empty:
        report.error(
            "empty_board", "No player ended up with a usable draft position."
        )
        return ResolvedBoard(board, report, status, season, scoring_format, team_count)

    board = board.sort_values("overall_adp").reset_index(drop=True)
    board["source"] = f"live:{'+'.join(sorted(merge_stats) or [spine_source])}"

    # ── Report on quality ────────────────────────────────────────────────────
    single = int((board["adp_source_count"] == 1).sum())
    multi = int((board["adp_source_count"] >= 2).sum())
    report.info(
        "consensus_coverage",
        f"{len(board)} players on the board. {multi} have ADP from two or more "
        f"sources; {single} rest on a single source.",
    )
    estimated_count = int(board["adp_stdev_is_estimated"].sum())
    if estimated_count:
        report.info(
            "estimated_spread",
            f"{estimated_count} player(s) have an *estimated* ADP spread because "
            "Fantasy Football Calculator has no data for them. Their survival odds "
            "are less precise than players with a measured spread.",
        )
    disagreements = board[board["adp_disagreement"] > ADP_DISAGREEMENT_PICKS]
    if len(disagreements):
        worst = disagreements.nlargest(3, "adp_disagreement")
        examples = ", ".join(
            f"{row.player_name} ({row.adp_disagreement:.0f} picks)"
            for row in worst.itertuples()
        )
        report.warn(
            "adp_disagreement",
            f"{len(disagreements)} player(s) have sources disagreeing by more than "
            f"{ADP_DISAGREEMENT_PICKS:.0f} picks — e.g. {examples}. The consensus is "
            "a genuine average of genuinely different opinions there.",
        )

    LOGGER.info(
        "Resolved board: %d players from %s", len(board), ", ".join(sorted(merge_stats))
    )
    return ResolvedBoard(
        frame=board, report=report, source_status=status, season=season,
        scoring_format=scoring_format, team_count=team_count,
    )


# Board column → the importer's canonical column. This is an explicit projection
# rather than passing the board through as-is, because the importer's header-alias
# table would otherwise mangle these names: left to itself it maps BOTH ``espn_adp``
# and ``yahoo_adp`` onto ``platform_adp`` (the second silently becoming
# ``platform_adp_2``), and ``source`` onto ``platform``. Naming the mapping here
# means the per-source columns keep their identity and nothing collides.
#
# The per-source entries below are only safe because their canonical names are also
# listed in ``core.constants.PLAYER_IMPORT_COLUMNS``: ``canonical_column`` checks
# that set *before* the alias table, so a known column is passed through untouched.
# Adding a per-source column here without adding it there re-introduces the
# collision this mapping exists to avoid.
IMPORT_COLUMNS: dict[str, str] = {
    "player_name": "player_name",
    "position": "position",
    "nfl_team": "nfl_team",
    "bye_week": "bye_week",
    "experience": "experience",
    "injury_status": "injury_status",
    "overall_adp": "adp",
    "overall_rank": "overall_rank",
    "position_rank": "position_rank",
    "platform_rank": "platform_rank",
    "platform_adp": "platform_adp",
    "adp_stdev": "adp_stdev",
    "min_pick": "min_pick",
    "max_pick": "max_pick",
    # Real projections, the stat line they were computed from, and the stats
    # themselves. ``stat_totals`` is the one that matters structurally: with it the
    # pool can rescore under new scoring rules offline, and without it a change of
    # scoring means refetching from ESPN.
    "espn_projection": "projection",
    "espn_stat_line": "projection_detail",
    "espn_stat_totals": "stat_totals",
    # Each platform's own numbers, kept separate so the Player Pool can show that
    # ESPN and Yahoo disagree rather than only showing the blend.
    "ffc_adp": "ffc_adp",
    "espn_adp": "espn_adp",
    "espn_rank": "espn_rank",
    "yahoo_adp": "yahoo_adp",
    "sleeper_search_rank": "sleeper_rank",
    "adp_source_count": "adp_source_count",
    "adp_disagreement": "adp_disagreement",
    "adp_stdev_is_estimated": "adp_stdev_is_estimated",
}


def board_to_import_frame(board: pd.DataFrame) -> pd.DataFrame:
    """Project a resolved board onto the columns :func:`import_player_pool` reads.

    Only the columns named in :data:`IMPORT_COLUMNS` are passed through — anything
    else on the board (per-provider ids, ownership percentages) is provider
    bookkeeping the player pool has no field for.
    """
    frame = pd.DataFrame(index=board.index)
    for source_column, canonical in IMPORT_COLUMNS.items():
        if source_column in board.columns:
            frame[canonical] = board[source_column]

    if "espn_projection" in board.columns:
        # Named per row rather than once for the pool, because coverage is partial:
        # players ESPN has no stat line for get the pool's ADP-derived estimate, and
        # the two must not look alike.
        frame["projection_source"] = np.where(
            pd.to_numeric(board["espn_projection"], errors="coerce").notna(),
            "ESPN projected stats, scored under your league rules",
            "",
        )

    if "experience" in board.columns:
        # A rookie is zero years of experience. Sleeper reports this reliably, so
        # deriving it here saves the importer guessing.
        frame["rookie_flag"] = pd.to_numeric(
            board["experience"], errors="coerce"
        ).eq(0)
    return frame


__all__ = [
    "ResolvedBoard",
    "resolve_board",
    "board_to_import_frame",
    "DEFAULT_ADP_WEIGHTS",
    "ADP_COLUMNS",
    "IMPORT_COLUMNS",
    "ADP_DISAGREEMENT_PICKS",
    "ESTIMATED_STDEV_FRACTION",
]
