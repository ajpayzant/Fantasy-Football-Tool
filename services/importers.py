"""Importers: raw tables → validated domain objects.

Contract for every importer here:

* Only the strictly required columns are mandatory. Everything else is optional
  and either derived or left unset.
* A row is never silently dropped. Rejected rows land in
  :attr:`ValidationReport.rejected` with a ``rejection_reason`` column so the UI
  can display and export them.
* Nothing raises on bad user data — problems come back as report issues.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import pandas as pd

from core.constants import (
    HISTORICAL_IMPORT_COLUMNS,
    HISTORICAL_REQUIRED_COLUMNS,
    PLAYER_IMPORT_COLUMNS,
    PLAYER_REQUIRED_COLUMNS,
    SAMPLE_DATA_BANNER,
)
from core import stats as core_stats
from core.enums import InjuryStatus, Platform, Position
from core.validation import (
    ValidationReport,
    require_columns,
    to_bool,
    to_float,
    to_int,
    validate_player_pool,
)
from models.draft import DraftHistory, HistoricalDraft, HistoricalPick
from models.league import Keeper
from models.player import Player, PlayerPool, PoolMetadata
from services.normalize import (
    clean_text,
    dedupe_names,
    describe_column_mapping,
    normalize_columns,
    normalize_injury_status,
    normalize_manager_name,
    normalize_player_name,
    normalize_position,
    normalize_season,
    normalize_team,
    player_key,
)

LOGGER = logging.getLogger("fantasy_mock_draft.importers")


@dataclass(slots=True)
class ImportResult:
    """Outcome of an import: the parsed object, the report, and diagnostics."""

    report: ValidationReport = field(default_factory=ValidationReport)
    history: DraftHistory | None = None
    pool: PlayerPool | None = None
    keepers: list[Keeper] = field(default_factory=list)
    column_mapping: dict[str, str] = field(default_factory=dict)
    accepted_rows: int = 0
    rejected_rows: int = 0

    @property
    def ok(self) -> bool:
        return self.report.ok and self.accepted_rows > 0

    def mapping_frame(self) -> pd.DataFrame:
        return describe_column_mapping(self.column_mapping)

    def summary(self) -> str:
        return (
            f"{self.accepted_rows} row(s) accepted, {self.rejected_rows} rejected — "
            f"{self.report.summary()}"
        )


def _reject(rows: list[dict[str, Any]], raw: Any, index: int, reason: str) -> None:
    """Record a rejected row with its original values plus the reason."""
    record = dict(raw) if isinstance(raw, dict) else dict(raw.to_dict())
    record["source_row"] = index + 2  # +2: 1-based, plus the header line
    record["rejection_reason"] = reason
    rows.append(record)


def _attach_rejected(report: ValidationReport, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(list(rows))
    ordered = ["source_row", "rejection_reason"] + [
        c for c in frame.columns if c not in {"source_row", "rejection_reason"}
    ]
    report.rejected = frame[ordered]


# ─────────────────────────────────────────────────────────────────────────────
# Historical drafts
# ─────────────────────────────────────────────────────────────────────────────
def import_historical_drafts(
    frame: pd.DataFrame,
    *,
    default_season: int | None = None,
    default_platform: Platform | str | None = None,
    default_league_name: str = "",
    source_file: str = "",
    merge_manager_spellings: bool = True,
) -> ImportResult:
    """Parse a table of historical picks into a :class:`DraftHistory`.

    Required columns are ``season``, ``manager_name``, ``overall_pick`` and
    ``player_name`` (loose header spellings are accepted). ``season`` may be
    omitted if ``default_season`` is supplied. Round and pick-in-round are
    derived from the overall pick when absent and the team count is inferrable.
    """
    result = ImportResult()
    report = result.report
    if frame is None or frame.empty:
        report.error("empty_input", "No rows to import.")
        return result

    normalized, mapping = normalize_columns(frame)
    result.column_mapping = mapping
    _resolve_pick_columns(normalized, mapping, report)

    required = list(HISTORICAL_REQUIRED_COLUMNS)
    if default_season is not None and "season" not in normalized.columns:
        normalized["season"] = default_season
        report.info(
            "season_defaulted",
            f"No season column found; treating every row as season {default_season}.",
        )
    if (
        "overall_pick" not in normalized.columns
        and {"round", "pick_in_round"} <= set(normalized.columns)
    ):
        # Round + pick-in-round fully determine the overall pick once the team
        # count is known, so accept that pairing in place of an overall column.
        normalized["overall_pick"] = pd.NA
        report.info(
            "overall_pick_derived",
            "No overall pick column found; deriving it from round and "
            "pick-in-round using snake ordering.",
        )
    if not require_columns(normalized, required, report, label="Draft history file"):
        return result

    if merge_manager_spellings:
        canonical = dedupe_names(normalized["manager_name"].tolist())
        merged = {k: v for k, v in canonical.items() if k != v}
        if merged:
            report.info(
                "merged_managers",
                "Merged manager name spellings: "
                + ", ".join(f"'{k}' → '{v}'" for k, v in sorted(merged.items())),
            )
    else:
        canonical = {}

    rejected: list[dict[str, Any]] = []
    parsed: list[tuple[HistoricalPick, str]] = []
    """(pick, draft grouping key)"""

    for index, raw in normalized.iterrows():
        season = normalize_season(raw.get("season")) or default_season
        if season is None:
            _reject(rejected, raw, index, "season is missing or unreadable")
            continue

        manager = normalize_manager_name(raw.get("manager_name"))
        manager = canonical.get(manager, manager)
        if not manager:
            _reject(rejected, raw, index, "manager_name is blank")
            continue

        overall = to_int(raw.get("overall_pick"))
        if overall is None:
            # A round + pick-in-round pair is an acceptable substitute.
            rnd = to_int(raw.get("round"))
            in_round = to_int(raw.get("pick_in_round"))
            if rnd is None or in_round is None:
                _reject(
                    rejected, raw, index,
                    "overall_pick is missing (and round/pick_in_round were not both present)",
                )
                continue
            overall = -1  # resolved once the team count is known
        if overall is not None and overall != -1 and overall < 1:
            _reject(rejected, raw, index, f"overall_pick {overall} is not a positive number")
            continue

        name = normalize_player_name(raw.get("player_name"))
        if not name:
            _reject(rejected, raw, index, "player_name is blank")
            continue

        league_name = clean_text(raw.get("league_name")) or default_league_name
        platform = clean_text(raw.get("platform")) or (
            str(default_platform) if default_platform else ""
        )
        pick = HistoricalPick(
            season=int(season),
            manager_name=manager,
            overall_pick=int(overall),
            player_name=name,
            league_name=league_name,
            platform=platform,
            round_number=to_int(raw.get("round")),
            pick_in_round=to_int(raw.get("pick_in_round")),
            position=normalize_position(raw.get("position")),
            nfl_team=normalize_team(raw.get("nfl_team")) if raw.get("nfl_team") is not None else "",
            adp=to_float(raw.get("adp")),
            platform_rank=to_float(raw.get("platform_rank")),
            projection=to_float(raw.get("projection")),
            tier=to_int(raw.get("tier")),
            is_keeper=to_bool(raw.get("keeper_flag")),
            is_rookie=to_bool(raw.get("rookie_flag")),
            bye_week=to_int(raw.get("bye_week")),
            draft_date=clean_text(raw.get("draft_date")) or None,
        )
        parsed.append((pick, f"{season}||{league_name}"))

    _attach_rejected(report, rejected)
    result.rejected_rows = len(rejected)
    result.accepted_rows = len(parsed)

    if rejected:
        report.warn(
            "rows_rejected",
            f"{len(rejected)} row(s) could not be imported. Download the rejected "
            "rows to see the reason for each.",
        )
    if not parsed:
        report.error("no_valid_rows", "No importable rows were found.")
        return result

    history = DraftHistory()
    for group in sorted({key for _, key in parsed}):
        season_text, league_name = group.split("||", 1)
        picks = [p for p, key in parsed if key == group]
        picks.sort(key=lambda p: (p.overall_pick if p.overall_pick > 0 else 10**6))
        draft = HistoricalDraft(
            season=int(season_text),
            league_name=league_name,
            platform=picks[0].platform if picks else "",
            picks=picks,
            draft_date=next((p.draft_date for p in picks if p.draft_date), None),
            source_file=source_file,
        )
        _finalize_draft(draft, report)
        history.add(draft)

    result.history = history
    report.info(
        "import_summary",
        f"Imported {len(parsed)} pick(s) across {len(history)} draft(s): "
        f"seasons {', '.join(str(s) for s in history.seasons)}.",
    )
    LOGGER.info("Imported %s historical picks in %s drafts", len(parsed), len(history))
    return result


def _resolve_pick_columns(
    frame: pd.DataFrame, mapping: dict[str, str], report: ValidationReport
) -> None:
    """Disambiguate files that carry both a within-round and an overall pick.

    Headers like ``Pick`` and ``Overall`` both alias to ``overall_pick``, leaving
    a suffixed duplicate (``overall_pick_2``). A header that says "overall" wins
    outright; otherwise the column with the larger maximum is the overall pick
    and the other is the pick within a round. Mutates ``frame`` and ``mapping``.
    """
    duplicates = [c for c in frame.columns if str(c).startswith("overall_pick_")]
    if not duplicates:
        return

    candidates = ["overall_pick", *duplicates]
    maxima: dict[str, float] = {}
    for column in candidates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        maxima[column] = float(values.max()) if values.notna().any() else -1.0
    if len(maxima) < 2:
        return

    # Reverse the mapping so we can consult the user's original header text.
    original_of = {canonical: original for original, canonical in mapping.items()}
    named_overall = [
        column for column in maxima
        if "overall" in str(original_of.get(column, "")).lower()
    ]
    if len(named_overall) == 1:
        overall_column = named_overall[0]
        rule = "the column named 'overall'"
    else:
        overall_column = max(maxima, key=lambda c: maxima[c])
        rule = "their value ranges"
    renames: dict[str, str] = {}
    if overall_column != "overall_pick":
        renames[overall_column] = "overall_pick"
        # The incumbent becomes pick_in_round unless that column already exists.
        renames["overall_pick"] = (
            "overall_pick_unused" if "pick_in_round" in frame.columns else "pick_in_round"
        )
    else:
        for column in duplicates:
            renames[column] = (
                "overall_pick_unused" if "pick_in_round" in frame.columns else "pick_in_round"
            )

    frame.rename(columns=renames, inplace=True)
    for original, canonical in list(mapping.items()):
        if canonical in renames:
            mapping[original] = renames[canonical]

    described = ", ".join(
        f"'{orig}' read as {mapping[orig]}"
        for orig, canon in mapping.items()
        if canon in {"overall_pick", "pick_in_round"}
    )
    report.info(
        "pick_columns_resolved",
        f"The file had more than one pick-number column. Told them apart using "
        f"{rule}: {described}.",
    )


def _finalize_draft(draft: HistoricalDraft, report: ValidationReport) -> None:
    """Infer team count and rounds, then fill any missing pick coordinates."""
    team_count = draft.infer_team_count()
    draft.team_count = team_count

    for pick in draft.picks:
        if pick.overall_pick <= 0 and pick.round_number and pick.pick_in_round:
            # Snake ordering: even rounds run backwards.
            in_round = (
                pick.pick_in_round if pick.round_number % 2
                else team_count - pick.pick_in_round + 1
            )
            pick.overall_pick = (pick.round_number - 1) * team_count + in_round
        if pick.round_number is None and pick.overall_pick > 0 and team_count:
            pick.round_number = (pick.overall_pick - 1) // team_count + 1
        if pick.pick_in_round is None and pick.overall_pick > 0 and team_count:
            pick.pick_in_round = (pick.overall_pick - 1) % team_count + 1

    draft.picks.sort(key=lambda p: p.overall_pick)
    draft.rounds = draft.infer_rounds()

    duplicates = _duplicate_values([p.overall_pick for p in draft.picks])
    if duplicates:
        preview = ", ".join(str(d) for d in duplicates[:8])
        report.warn(
            "duplicate_picks",
            f"Season {draft.season}: pick number(s) {preview} appear more than once. "
            "The picks are kept, but ordering features may be less accurate.",
        )

    counts = pd.Series([p.manager_name for p in draft.picks]).value_counts()
    if len(counts) > 1 and counts.max() - counts.min() > 2:
        report.warn(
            "uneven_pick_counts",
            f"Season {draft.season}: managers have uneven pick counts "
            f"({counts.min()}–{counts.max()}). Keeper or traded picks can cause this.",
        )


def _duplicate_values(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    duplicates: list[Any] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def historical_template() -> pd.DataFrame:
    """An empty template with the expected columns and one example row."""
    example = {
        "season": 2025, "league_name": "My League", "platform": "ESPN",
        "manager_name": "Alice", "round": 1, "pick_in_round": 1,
        "overall_pick": 1, "player_name": "Example Player", "position": "RB",
        "nfl_team": "KC", "adp": 2.4, "platform_rank": 1, "projection": 285.0,
        "tier": 1, "keeper_flag": "N", "rookie_flag": "N", "draft_date": "2025-08-24",
    }
    return pd.DataFrame([example], columns=list(HISTORICAL_IMPORT_COLUMNS))


# ─────────────────────────────────────────────────────────────────────────────
# Player pools
# ─────────────────────────────────────────────────────────────────────────────
def import_player_pool(
    frame: pd.DataFrame,
    *,
    league: Any | None = None,
    source: str = "upload",
    season: int | None = None,
    platform: Platform | str | None = None,
    is_sample_data: bool = False,
    imported_at: str = "",
) -> ImportResult:
    """Parse a player table into a :class:`PlayerPool`.

    Only ``player_name`` and ``position`` are required; ADP, ranks and
    projections are imputed by the pool when absent, and what was imputed is
    recorded in the pool metadata.
    """
    result = ImportResult()
    report = result.report
    if frame is None or frame.empty:
        report.error("empty_input", "No rows to import.")
        return result

    normalized, mapping = normalize_columns(frame)
    result.column_mapping = mapping
    if not require_columns(normalized, PLAYER_REQUIRED_COLUMNS, report, label="Player file"):
        return result

    rejected: list[dict[str, Any]] = []
    players: list[Player] = []
    seen_keys: dict[str, str] = {}

    for index, raw in normalized.iterrows():
        name = normalize_player_name(raw.get("player_name"))
        if not name:
            _reject(rejected, raw, index, "player_name is blank")
            continue
        position = normalize_position(raw.get("position"))
        if position is None:
            _reject(
                rejected, raw, index,
                f"position '{clean_text(raw.get('position'))}' is not one of "
                f"{', '.join(Position.values())}",
            )
            continue

        key = player_key(name, position)
        if key in seen_keys:
            _reject(
                rejected, raw, index,
                f"duplicate of '{seen_keys[key]}' (same name and position)",
            )
            continue
        seen_keys[key] = name

        adp = to_float(raw.get("adp"))
        espn_adp = to_float(raw.get("espn_adp"))
        yahoo_adp = to_float(raw.get("yahoo_adp"))
        espn_rank = to_float(raw.get("espn_rank"))
        # The per-platform columns are canonical in their own right, so a file
        # carrying only "ESPN ADP" no longer fills ``platform_adp`` by aliasing.
        # Fall back explicitly: ``platform_adp`` is what the engine's platform
        # ranking lens reads, and leaving it empty would silently disable that lens.
        platform_adp = to_float(raw.get("platform_adp"))
        if platform_adp is None:
            platform_adp = espn_adp if espn_adp is not None else yahoo_adp
        platform_rank = to_float(raw.get("platform_rank"))
        if platform_rank is None:
            platform_rank = (
                espn_rank if espn_rank is not None else to_float(raw.get("yahoo_rank"))
            )
        players.append(
            Player(
                player_id=key,
                name=name,
                position=position,
                nfl_team=normalize_team(raw.get("nfl_team")),
                bye_week=to_int(raw.get("bye_week")),
                experience=to_int(raw.get("experience")),
                is_rookie=to_bool(raw.get("rookie_flag")),
                injury_status=InjuryStatus.coerce(
                    normalize_injury_status(raw.get("injury_status")),
                    InjuryStatus.HEALTHY,
                ),
                suspended=to_bool(raw.get("suspended")),
                projection=to_float(raw.get("projection")),
                overall_rank=to_float(raw.get("overall_rank")),
                position_rank=to_int(raw.get("position_rank")),
                platform_rank=platform_rank,
                overall_adp=adp if adp is not None else to_float(raw.get("overall_adp")),
                platform_adp=platform_adp,
                adp_stdev=to_float(raw.get("adp_stdev")),
                min_pick=to_int(raw.get("min_pick")),
                max_pick=to_int(raw.get("max_pick")),
                tier=to_int(raw.get("tier")),
                ceiling=to_float(raw.get("ceiling")),
                floor=to_float(raw.get("floor")),
                risk_score=to_float(raw.get("risk_score")),
                value_over_replacement=to_float(raw.get("value_over_replacement")),
                notes=clean_text(raw.get("notes")),
                source=source,
                ffc_adp=to_float(raw.get("ffc_adp")),
                espn_adp=espn_adp,
                espn_rank=espn_rank,
                yahoo_adp=yahoo_adp,
                yahoo_rank=to_float(raw.get("yahoo_rank")),
                sleeper_rank=to_float(raw.get("sleeper_rank")),
                adp_source_count=to_int(raw.get("adp_source_count")),
                adp_disagreement=to_float(raw.get("adp_disagreement")),
                adp_stdev_is_estimated=to_bool(raw.get("adp_stdev_is_estimated")),
                projection_source=clean_text(raw.get("projection_source")),
                projection_detail=clean_text(raw.get("projection_detail")),
                # The stat line behind the projection, if the file carried one. This is
                # what lets :meth:`PlayerPool.rescore` change scoring rules without
                # going back to the network — see :mod:`core.stats`.
                stat_totals=core_stats.from_frame_value(raw.get("stat_totals")),
            )
        )

    _attach_rejected(report, rejected)
    result.rejected_rows = len(rejected)
    result.accepted_rows = len(players)

    if rejected:
        report.warn(
            "rows_rejected",
            f"{len(rejected)} player row(s) could not be imported. Download the "
            "rejected rows to see why.",
        )
    if not players:
        report.error("no_valid_players", "No importable player rows were found.")
        return result

    metadata = PoolMetadata(
        source=source,
        imported_at=imported_at,
        season=season,
        platform=str(platform) if platform else None,
        player_count=len(players),
        is_sample_data=is_sample_data,
        notes=SAMPLE_DATA_BANNER if is_sample_data else "",
    )
    pool = PlayerPool(players, league=league, metadata=metadata)
    result.pool = pool

    if league is not None:
        report.extend(validate_player_pool(pool.to_frame(), league))
    if metadata.imputed_fields:
        detail = ", ".join(
            f"{count} {field_name}" for field_name, count in sorted(metadata.imputed_fields.items())
        )
        report.info(
            "imputed_values",
            f"Filled in missing values so every player is draftable: {detail}. "
            "Supplying these columns yourself gives better simulations.",
        )
    report.info("import_summary", f"Imported {len(players)} player(s) from {source}.")
    LOGGER.info("Imported %s players from %s", len(players), source)
    return result


def player_template() -> pd.DataFrame:
    """An empty player-file template with one example row."""
    example = {
        "player_name": "Example Player", "position": "RB", "nfl_team": "KC",
        "bye_week": 10, "experience": 3, "rookie_flag": "N",
        "injury_status": "Healthy", "projection": 285.0, "overall_rank": 1,
        "position_rank": 1, "platform_rank": 1, "overall_adp": 2.4,
        "platform_adp": 2.1, "adp_stdev": 1.5, "min_pick": 1, "max_pick": 6,
        "tier": 1, "ceiling": 330.0, "floor": 210.0, "risk_score": 0.3,
        "value_over_replacement": 95.0, "notes": "",
        # Optional, and the only column here that buys something the others cannot:
        # supply the stat line and the app can rescore this player when the league's
        # scoring changes. Supply only ``projection`` and it is frozen at whatever
        # rules produced it. Field names are the canonical ones in :mod:`core.stats`.
        "stat_totals": core_stats.to_frame_value({
            "rush_attempts": 260.0, "rush_yards": 1180.0, "rush_td": 9.0,
            "targets": 72.0, "receptions": 56.0, "rec_yards": 480.0, "rec_td": 3.0,
            "fumbles_lost": 2.0, "games": 17.0,
        }),
    }
    return pd.DataFrame([example], columns=list(PLAYER_IMPORT_COLUMNS))


# ─────────────────────────────────────────────────────────────────────────────
# Keepers
# ─────────────────────────────────────────────────────────────────────────────
KEEPER_REQUIRED_COLUMNS: tuple[str, ...] = ("manager_name", "player_name")
KEEPER_IMPORT_COLUMNS: tuple[str, ...] = (
    "manager_name", "player_name", "keeper_round", "overall_pick",
    "removes_pick", "salary", "position", "nfl_team", "notes",
)


def import_keepers(frame: pd.DataFrame) -> ImportResult:
    """Parse a keeper table. Only manager and player names are required."""
    result = ImportResult()
    report = result.report
    if frame is None or frame.empty:
        report.error("empty_input", "No rows to import.")
        return result

    normalized, mapping = normalize_columns(frame)
    result.column_mapping = mapping
    if not require_columns(normalized, KEEPER_REQUIRED_COLUMNS, report, label="Keeper file"):
        return result

    rejected: list[dict[str, Any]] = []
    keepers: list[Keeper] = []
    for index, raw in normalized.iterrows():
        manager = normalize_manager_name(raw.get("manager_name"))
        player = normalize_player_name(raw.get("player_name"))
        if not manager:
            _reject(rejected, raw, index, "manager_name is blank")
            continue
        if not player:
            _reject(rejected, raw, index, "player_name is blank")
            continue
        keepers.append(
            Keeper(
                manager_name=manager,
                player_name=player,
                keeper_round=to_int(raw.get("keeper_round")),
                overall_pick=to_int(raw.get("overall_pick")),
                removes_pick=to_bool(raw.get("removes_pick"), default=True),
                salary=to_float(raw.get("salary")),
                position=normalize_position(raw.get("position")),
                nfl_team=normalize_team(raw.get("nfl_team")) if raw.get("nfl_team") is not None else None,
                notes=clean_text(raw.get("notes")),
            )
        )

    _attach_rejected(report, rejected)
    result.rejected_rows = len(rejected)
    result.accepted_rows = len(keepers)
    result.keepers = keepers
    if rejected:
        report.warn("rows_rejected", f"{len(rejected)} keeper row(s) were rejected.")
    if not keepers:
        report.error("no_valid_keepers", "No importable keeper rows were found.")
        return result

    missing_round = [k.player_name for k in keepers if k.removes_pick and k.keeper_round is None and k.overall_pick is None]
    if missing_round:
        report.warn(
            "keeper_no_round",
            f"{len(missing_round)} keeper(s) have no round or pick assigned, so they "
            "cost no draft pick: " + ", ".join(missing_round[:6]),
        )
    report.info("import_summary", f"Imported {len(keepers)} keeper(s).")
    return result


def keeper_template() -> pd.DataFrame:
    example = {
        "manager_name": "Alice", "player_name": "Example Player",
        "keeper_round": 3, "overall_pick": "", "removes_pick": "Y",
        "salary": "", "position": "WR", "nfl_team": "KC", "notes": "",
    }
    return pd.DataFrame([example], columns=list(KEEPER_IMPORT_COLUMNS))


# ─────────────────────────────────────────────────────────────────────────────
# Manual entry helpers
# ─────────────────────────────────────────────────────────────────────────────
def manual_history_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Build a history frame from UI data-editor rows, dropping blank ones."""
    frame = pd.DataFrame(list(rows), columns=list(HISTORICAL_IMPORT_COLUMNS))
    mask = frame["player_name"].astype(str).str.strip().ne("") if "player_name" in frame else None
    return frame if mask is None else frame[mask].reset_index(drop=True)


def rejected_rows_csv(report: ValidationReport) -> bytes:
    """Rejected rows as CSV bytes for a download button (empty frame if none)."""
    frame = report.rejected if report.rejected is not None else pd.DataFrame(
        columns=["source_row", "rejection_reason"]
    )
    return frame.to_csv(index=False).encode("utf-8")


__all__ = [
    "ImportResult", "import_historical_drafts", "import_player_pool",
    "import_keepers", "historical_template", "player_template",
    "keeper_template", "manual_history_frame", "rejected_rows_csv",
    "KEEPER_IMPORT_COLUMNS", "KEEPER_REQUIRED_COLUMNS",
]
