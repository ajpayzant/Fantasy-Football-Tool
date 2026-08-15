"""Persistence for domain objects: maps dataclasses to/from SQLAlchemy rows.

The engine never touches this module — simulation works on plain domain objects,
and the UI calls these functions at load/save boundaries. Every ``save_*``
function is an upsert keyed on a natural key so repeated saves from a Streamlit
rerun do not create duplicates.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from core import freshness as core_freshness
from core import stats as core_stats
from core.config import LeagueConfig, RosterSettings, ScoringRules, ShrinkageConfig
from core.enums import Archetype, InjuryStatus, Platform, Position, Slot
from models.database import (
    ApplicationSettingRow,
    HistoricalDraftRow,
    HistoricalPickRow,
    KeeperRow,
    LeagueRosterSlotRow,
    LeagueRow,
    LeagueScoringRuleRow,
    ManagerManualPreferenceRow,
    ManagerProfileRow,
    ManagerRow,
    MockDraftPickRow,
    MockDraftRosterRow,
    MockDraftRow,
    PlayerDataSourceRow,
    PlayerRankingRow,
    PlayerRow,
    SimulationResultRow,
    SimulationRunRow,
    get_setting,
    session_scope,
    set_setting,
    utcnow,
)
from models.draft import (
    DraftHistory,
    HistoricalDraft,
    HistoricalPick,
    MockDraftResult,
    Pick,
    TeamRoster,
)
from models.league import Keeper, League
from models.manager import (
    Manager,
    ManagerPreferences,
    ManagerProfile,
    normalize_manager_key,
)
from models.player import Player, PlayerPool, PoolMetadata

LOGGER = logging.getLogger("fantasy_mock_draft.repository")

SCORING_PRESET_KEY = "__preset__"
"""Sentinel rule key holding the non-numeric preset name."""


# ─────────────────────────────────────────────────────────────────────────────
# Leagues
# ─────────────────────────────────────────────────────────────────────────────
def save_league(session: Session, league: League) -> int:
    """Insert or update a league with its slots, scoring, managers and keepers.

    Returns the league id, which is also written back onto ``league.config``.
    """
    row: LeagueRow | None = None
    if league.config.league_id is not None:
        row = session.get(LeagueRow, league.config.league_id)
    if row is None:
        row = session.execute(
            select(LeagueRow).where(
                LeagueRow.name == league.config.name,
                LeagueRow.season == int(league.config.season),
            )
        ).scalar_one_or_none()
    if row is None:
        row = LeagueRow()
        session.add(row)

    config = league.config
    row.name = config.name
    row.season = int(config.season)
    row.platform = str(config.platform)
    row.team_count = int(config.team_count)
    row.rounds = int(config.rounds)
    row.draft_type = str(config.draft_type)
    row.league_format = str(config.league_format)
    row.user_draft_slot = int(config.user_draft_slot)
    row.draft_date = config.draft_date
    row.reversal_round = int(config.reversal_round)
    row.custom_round_order = {str(k): list(v) for k, v in config.custom_round_order.items()}
    row.notes = config.notes
    session.flush()

    _replace_roster_slots(session, row, config.roster)
    _replace_scoring_rules(session, row, config.scoring)
    _replace_managers(session, row, league.managers)
    _replace_keepers(session, row, league.keepers)
    session.flush()

    config.league_id = int(row.id)
    LOGGER.info("Saved league '%s' (%s) id=%s", row.name, row.season, row.id)
    return int(row.id)


def _replace_roster_slots(session: Session, row: LeagueRow, roster: RosterSettings) -> None:
    # Clear through the relationship (not a bulk DELETE) so delete-orphan
    # cascades reach child rows and the ORM identity map stays consistent.
    row.roster_slots.clear()
    session.flush()
    for slot, count in roster.slots.items():
        row.roster_slots.append(
            LeagueRosterSlotRow(slot=str(slot), count=int(count))
        )
    # Position min/max are league-wide, not per slot; store them on a synthetic
    # row per position so the table stays the single source of roster limits.
    for position, maximum in roster.position_max.items():
        row.roster_slots.append(
            LeagueRosterSlotRow(
                slot=f"max:{position}", count=0, position_max=int(maximum)
            )
        )
    for position, minimum in roster.position_min.items():
        row.roster_slots.append(
            LeagueRosterSlotRow(
                slot=f"min:{position}", count=0, position_min=int(minimum)
            )
        )


def _replace_scoring_rules(session: Session, row: LeagueRow, scoring: ScoringRules) -> None:
    row.scoring_rules.clear()
    session.flush()
    payload = scoring.to_dict()
    preset = payload.pop("preset", None)
    row.scoring_rules.append(
        LeagueScoringRuleRow(
            rule_key=SCORING_PRESET_KEY,
            rule_text=str(preset) if preset else None,
        )
    )
    for key, value in payload.items():
        row.scoring_rules.append(
            LeagueScoringRuleRow(
                rule_key=key,
                rule_value=None if value is None else float(value),
            )
        )


def _replace_managers(session: Session, row: LeagueRow, managers: Sequence[Manager]) -> None:
    """Upsert managers by key, so cached profiles survive a league re-save."""
    existing = {m.manager_key: m for m in row.managers}
    incoming_keys = {m.key for m in managers}

    for key, manager_row in list(existing.items()):
        if key not in incoming_keys:
            row.managers.remove(manager_row)
    session.flush()

    # ``draft_slot`` is unique per league, so shift every surviving row out of
    # the valid range before reassigning — otherwise swapping two managers'
    # slots would transiently collide.
    for manager_row in row.managers:
        manager_row.draft_slot = -abs(manager_row.draft_slot) - 1000
    session.flush()

    for manager in managers:
        manager_row = existing.get(manager.key)
        if manager_row is None or manager_row not in row.managers:
            manager_row = ManagerRow(manager_key=manager.key)
            row.managers.append(manager_row)
        manager_row.name = manager.name
        manager_row.team_name = manager.team_name
        manager_row.draft_slot = int(manager.draft_slot)
        manager_row.is_user = bool(manager.is_user)
        manager_row.archetype = str(manager.archetype)

        if manager.preferences.has_any:
            payload = manager.preferences.to_dict()
            if manager_row.preferences is None:
                manager_row.preferences = ManagerManualPreferenceRow(payload=payload)
            else:
                manager_row.preferences.payload = payload
        elif manager_row.preferences is not None:
            manager_row.preferences = None
    session.flush()

    by_key = {m.manager_key: m for m in row.managers}
    for manager in managers:
        manager_row = by_key.get(manager.key)
        if manager_row is not None:
            manager.manager_id = int(manager_row.id)


def _replace_keepers(session: Session, row: LeagueRow, keepers: Sequence[Keeper]) -> None:
    row.keepers.clear()
    session.flush()
    for keeper in keepers:
        row.keepers.append(
            KeeperRow(
                manager_name=keeper.manager_name,
                player_name=keeper.player_name,
                keeper_round=keeper.keeper_round,
                overall_pick=keeper.overall_pick,
                removes_pick=bool(keeper.removes_pick),
                salary=keeper.salary,
                position=str(keeper.position) if keeper.position else None,
                nfl_team=keeper.nfl_team,
                notes=keeper.notes,
            )
        )


def load_league(session: Session, league_id: int) -> League | None:
    """Reconstruct a full :class:`League` from its id, or ``None`` if absent."""
    row = session.get(LeagueRow, league_id)
    if row is None:
        return None

    roster = _roster_from_rows(row.roster_slots)
    scoring = _scoring_from_rows(row.scoring_rules)
    config = LeagueConfig(
        name=row.name,
        season=int(row.season),
        platform=Platform.coerce(row.platform, Platform.CUSTOM),
        team_count=int(row.team_count),
        rounds=int(row.rounds),
        draft_type=row.draft_type,
        league_format=row.league_format,
        scoring=scoring,
        roster=roster,
        user_draft_slot=int(row.user_draft_slot),
        draft_date=row.draft_date,
        reversal_round=int(row.reversal_round),
        custom_round_order={
            int(k): list(v) for k, v in (row.custom_round_order or {}).items()
        },
        notes=row.notes or "",
        league_id=int(row.id),
    )

    managers: list[Manager] = []
    for manager_row in sorted(row.managers, key=lambda m: m.draft_slot):
        preferences = ManagerPreferences.from_dict(
            manager_row.preferences.payload if manager_row.preferences else None
        )
        managers.append(
            Manager(
                name=manager_row.name,
                draft_slot=int(manager_row.draft_slot),
                team_name=manager_row.team_name or "",
                is_user=bool(manager_row.is_user),
                manager_id=int(manager_row.id),
                archetype=Archetype.coerce(manager_row.archetype, Archetype.BALANCED),
                preferences=preferences,
            )
        )

    keepers = [
        Keeper(
            manager_name=k.manager_name,
            player_name=k.player_name,
            keeper_round=k.keeper_round,
            overall_pick=k.overall_pick,
            removes_pick=bool(k.removes_pick),
            salary=k.salary,
            position=k.position,
            nfl_team=k.nfl_team,
            notes=k.notes or "",
            keeper_id=int(k.id),
        )
        for k in row.keepers
    ]
    return League(config=config, managers=managers, keepers=keepers)


def _roster_from_rows(rows: Iterable[LeagueRosterSlotRow]) -> RosterSettings:
    slots: dict[Slot, int] = {}
    position_max: dict[Position, int] = {}
    position_min: dict[Position, int] = {}
    for row in rows:
        label = row.slot or ""
        if label.startswith("max:"):
            position = Position.coerce(label[4:], None)
            if position is not None and row.position_max is not None:
                position_max[position] = int(row.position_max)
            continue
        if label.startswith("min:"):
            position = Position.coerce(label[4:], None)
            if position is not None and row.position_min is not None:
                position_min[position] = int(row.position_min)
            continue
        slot = Slot.coerce(label, None)
        if slot is not None:
            slots[slot] = int(row.count)
    if not slots:
        return RosterSettings(position_max=position_max, position_min=position_min)
    return RosterSettings(
        slots=slots, position_max=position_max, position_min=position_min
    )


def _scoring_from_rows(rows: Iterable[LeagueScoringRuleRow]) -> ScoringRules:
    payload: dict[str, Any] = {}
    for row in rows:
        if row.rule_key == SCORING_PRESET_KEY:
            if row.rule_text:
                payload["preset"] = row.rule_text
            continue
        payload[row.rule_key] = row.rule_value
    return ScoringRules.from_dict(payload)


def list_leagues(session: Session) -> list[dict[str, Any]]:
    """Summary rows for the league picker, newest first."""
    rows = session.execute(
        select(LeagueRow).order_by(LeagueRow.updated_at.desc())
    ).scalars().all()
    return [
        {
            "league_id": int(r.id),
            "name": r.name,
            "season": int(r.season),
            "platform": r.platform,
            "team_count": int(r.team_count),
            "rounds": int(r.rounds),
            "draft_type": r.draft_type,
            "updated_at": r.updated_at,
        }
        for r in rows
    ]


def delete_league(session: Session, league_id: int) -> bool:
    row = session.get(LeagueRow, league_id)
    if row is None:
        return False
    session.delete(row)
    LOGGER.warning("Deleted league id=%s", league_id)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Historical drafts
# ─────────────────────────────────────────────────────────────────────────────
def save_history(
    session: Session, history: DraftHistory, league_id: int | None = None
) -> list[int]:
    """Persist every draft in ``history``, replacing same-season records."""
    return [save_historical_draft(session, d, league_id) for d in history.drafts]


def save_historical_draft(
    session: Session, draft: HistoricalDraft, league_id: int | None = None
) -> int:
    """Upsert one historical draft keyed on (league, season, league_name)."""
    row = session.execute(
        select(HistoricalDraftRow).where(
            HistoricalDraftRow.league_id == league_id,
            HistoricalDraftRow.season == int(draft.season),
            HistoricalDraftRow.league_name == (draft.league_name or ""),
        )
    ).scalar_one_or_none()
    if row is None:
        row = HistoricalDraftRow(league_id=league_id)
        session.add(row)

    row.season = int(draft.season)
    row.league_name = draft.league_name or ""
    row.platform = draft.platform or ""
    row.team_count = draft.team_count
    row.rounds = draft.rounds
    row.draft_date = draft.draft_date
    row.source_file = draft.source_file or ""
    session.flush()

    row.picks.clear()
    session.flush()
    for pick in draft.picks:
        row.picks.append(_historical_pick_row(pick))
    session.flush()
    draft.draft_id = int(row.id)
    LOGGER.info(
        "Saved historical draft season=%s picks=%s", draft.season, len(draft.picks)
    )
    return int(row.id)


_FEATURE_FIELDS: tuple[str, ...] = (
    "adp_delta", "rank_delta", "rank_inversions", "position_count_before",
    "roster_size_before",
    "open_starting_slots_before", "filled_starting_slot", "draft_phase",
    "started_run", "continued_run", "was_stack", "was_handcuff",
    "picks_until_next", "position_picks_in_window",
)


def _historical_pick_row(pick: HistoricalPick) -> HistoricalPickRow:
    """Build a child row; ``draft_id`` is set by the relationship append."""
    return HistoricalPickRow(
        season=int(pick.season),
        manager_name=pick.manager_name,
        manager_key=pick.manager_key,
        overall_pick=int(pick.overall_pick),
        round_number=pick.round_number,
        pick_in_round=pick.pick_in_round,
        player_name=pick.player_name,
        position=str(pick.position) if pick.position else None,
        nfl_team=pick.nfl_team or "",
        adp=pick.adp,
        platform_rank=pick.platform_rank,
        projection=pick.projection,
        is_keeper=bool(pick.is_keeper),
        is_rookie=bool(pick.is_rookie),
        bye_week=pick.bye_week,
        draft_date=pick.draft_date,
        platform=pick.platform or "",
        league_name=pick.league_name or "",
        features={name: getattr(pick, name) for name in _FEATURE_FIELDS},
    )


def load_history(session: Session, league_id: int | None = None) -> DraftHistory:
    """Load all historical drafts for a league (or every orphan draft)."""
    statement = select(HistoricalDraftRow).order_by(HistoricalDraftRow.season)
    if league_id is not None:
        statement = statement.where(HistoricalDraftRow.league_id == league_id)
    history = DraftHistory()
    for row in session.execute(statement).scalars().all():
        history.add(_historical_draft_from_row(row))
    return history


def _historical_draft_from_row(row: HistoricalDraftRow) -> HistoricalDraft:
    picks: list[HistoricalPick] = []
    for pick_row in sorted(row.picks, key=lambda p: p.overall_pick):
        features = dict(pick_row.features or {})
        pick = HistoricalPick(
            season=int(pick_row.season),
            manager_name=pick_row.manager_name,
            overall_pick=int(pick_row.overall_pick),
            player_name=pick_row.player_name,
            league_name=pick_row.league_name or "",
            platform=pick_row.platform or "",
            round_number=pick_row.round_number,
            pick_in_round=pick_row.pick_in_round,
            position=pick_row.position,
            nfl_team=pick_row.nfl_team or "",
            adp=pick_row.adp,
            platform_rank=pick_row.platform_rank,
            projection=pick_row.projection,
            is_keeper=bool(pick_row.is_keeper),
            is_rookie=bool(pick_row.is_rookie),
            bye_week=pick_row.bye_week,
            draft_date=pick_row.draft_date,
            historical_pick_id=int(pick_row.id),
        )
        for name in _FEATURE_FIELDS:
            if name in features and features[name] is not None:
                setattr(pick, name, features[name])
        picks.append(pick)
    return HistoricalDraft(
        season=int(row.season),
        league_name=row.league_name or "",
        platform=row.platform or "",
        team_count=row.team_count,
        rounds=row.rounds,
        draft_date=row.draft_date,
        picks=picks,
        draft_id=int(row.id),
        source_file=row.source_file or "",
    )


def delete_historical_draft(session: Session, draft_id: int) -> bool:
    row = session.get(HistoricalDraftRow, draft_id)
    if row is None:
        return False
    session.delete(row)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Manager profiles
# ─────────────────────────────────────────────────────────────────────────────
def save_manager_profile(
    session: Session,
    league_id: int,
    profile: ManagerProfile,
    shrinkage: ShrinkageConfig | None = None,
) -> int | None:
    """Cache a modelled profile against its manager row.

    Returns ``None`` when no manager in the league matches the profile key —
    profiles for non-league managers (e.g. league-average) are not persisted.
    """
    manager_row = session.execute(
        select(ManagerRow).where(
            ManagerRow.league_id == league_id,
            ManagerRow.manager_key == profile.manager_key,
        )
    ).scalar_one_or_none()
    if manager_row is None:
        return None

    row = manager_row.profile
    if row is None:
        row = ManagerProfileRow(manager_id=manager_row.id)
        session.add(row)
    row.archetype = str(profile.archetype)
    row.sample_picks = float(profile.sample_picks)
    row.sample_drafts = int(profile.sample_drafts)
    row.payload = profile.to_dict()
    row.built_from_seasons = list(profile.seasons_seen)
    row.shrinkage_config = shrinkage.to_dict() if shrinkage else None
    session.flush()
    return int(row.id)


def load_manager_profiles(session: Session, league_id: int) -> dict[str, ManagerProfile]:
    """Manager key → cached profile, for every manager that has one."""
    rows = session.execute(
        select(ManagerProfileRow)
        .join(ManagerRow, ManagerProfileRow.manager_id == ManagerRow.id)
        .where(ManagerRow.league_id == league_id)
    ).scalars().all()
    out: dict[str, ManagerProfile] = {}
    for row in rows:
        if not row.payload:
            continue
        profile = ManagerProfile.from_dict(row.payload)
        out[profile.manager_key] = profile
    return out


def clear_manager_profiles(session: Session, league_id: int) -> int:
    """Drop cached profiles so they rebuild — used when history changes."""
    manager_ids = session.execute(
        select(ManagerRow.id).where(ManagerRow.league_id == league_id)
    ).scalars().all()
    if not manager_ids:
        return 0
    result = session.execute(
        delete(ManagerProfileRow).where(ManagerProfileRow.manager_id.in_(manager_ids))
    )
    return int(result.rowcount or 0)


# ─────────────────────────────────────────────────────────────────────────────
# Player pools
# ─────────────────────────────────────────────────────────────────────────────
def save_player_pool(
    session: Session, pool: PlayerPool, *, source_kind: str = "upload",
    file_name: str = "", validation_summary: dict[str, Any] | None = None,
) -> int:
    """Persist a pool as a data source plus its players and rankings."""
    metadata = pool.metadata
    row = session.execute(
        select(PlayerDataSourceRow).where(
            PlayerDataSourceRow.name == metadata.source,
            PlayerDataSourceRow.season == metadata.season,
        )
    ).scalar_one_or_none()
    if row is None:
        row = PlayerDataSourceRow(name=metadata.source, season=metadata.season)
        session.add(row)

    row.platform = metadata.platform
    row.source_kind = source_kind
    row.file_name = file_name
    # The data's own timestamp, not the time of this save. Stamping ``utcnow()`` here
    # meant a board fetched three days ago and saved today reloaded tomorrow looking
    # brand new, which is the one thing the staleness warning has to be able to see.
    row.imported_at = (
        core_freshness.parse_timestamp(metadata.imported_at) or utcnow()
    ).replace(tzinfo=None)
    row.timestamp_basis = metadata.timestamp_basis or core_freshness.FETCHED
    row.player_count = len(pool)
    row.is_sample_data = bool(metadata.is_sample_data)
    row.notes = metadata.notes or ""
    row.validation_summary = validation_summary
    session.flush()

    row.players.clear()
    row.rankings.clear()
    session.flush()

    platform = metadata.platform or "Custom"
    for player in pool:
        player_row = PlayerRow(
            player_key=player.player_id,
            player_name=player.name,
            position=str(player.position),
            nfl_team=player.nfl_team,
            bye_week=player.bye_week,
            experience=player.experience,
            is_rookie=bool(player.is_rookie),
            injury_status=str(player.injury_status),
            suspended=bool(player.suspended),
            projection=player.projection,
            overall_rank=player.overall_rank,
            position_rank=player.position_rank,
            ceiling=player.ceiling,
            floor=player.floor,
            risk_score=player.risk_score,
            value_over_replacement=player.value_over_replacement,
            notes=player.notes,
            # Provenance. Without ``stat_totals`` a reloaded pool cannot be rescored
            # when the league's scoring changes, and without the rest a reloaded pool
            # forgets which of its numbers were measured and which this app guessed.
            stat_totals=core_stats.to_frame_value(player.stat_totals),
            projection_imputed=bool(player.projection_imputed),
            projection_source=player.projection_source,
            projection_detail=player.projection_detail,
            outcome_band_source=player.outcome_band_source,
            adp_stdev_is_estimated=bool(player.adp_stdev_is_estimated),
            ffc_adp=player.ffc_adp,
            espn_adp=player.espn_adp,
            espn_rank=player.espn_rank,
            yahoo_adp=player.yahoo_adp,
            yahoo_rank=player.yahoo_rank,
            sleeper_rank=player.sleeper_rank,
            adp_source_count=player.adp_source_count,
            adp_disagreement=player.adp_disagreement,
        )
        player_row.rankings.append(
            PlayerRankingRow(
                source_id=row.id,
                platform=platform,
                ranking_kind="platform_adp",
                overall_adp=player.overall_adp,
                platform_adp=player.platform_adp,
                platform_rank=player.platform_rank,
                adp_stdev=player.adp_stdev,
                min_pick=player.min_pick,
                max_pick=player.max_pick,
            )
        )
        row.players.append(player_row)
    session.flush()
    LOGGER.info("Saved player pool '%s' (%s players)", row.name, row.player_count)
    return int(row.id)


def load_player_pool(
    session: Session, source_id: int, league: LeagueConfig | None = None
) -> PlayerPool | None:
    """Rebuild a :class:`PlayerPool` from a stored data source."""
    row = session.get(PlayerDataSourceRow, source_id)
    if row is None:
        return None

    players: list[Player] = []
    for player_row in row.players:
        ranking = player_row.rankings[0] if player_row.rankings else None
        players.append(
            Player(
                player_id=player_row.player_key,
                name=player_row.player_name,
                position=Position.coerce(player_row.position, Position.WR),
                nfl_team=player_row.nfl_team or "FA",
                bye_week=player_row.bye_week,
                experience=player_row.experience,
                is_rookie=bool(player_row.is_rookie),
                injury_status=InjuryStatus.coerce(
                    player_row.injury_status, InjuryStatus.HEALTHY
                ),
                suspended=bool(player_row.suspended),
                projection=player_row.projection,
                overall_rank=player_row.overall_rank,
                position_rank=player_row.position_rank,
                platform_rank=ranking.platform_rank if ranking else None,
                overall_adp=ranking.overall_adp if ranking else None,
                platform_adp=ranking.platform_adp if ranking else None,
                adp_stdev=ranking.adp_stdev if ranking else None,
                min_pick=ranking.min_pick if ranking else None,
                max_pick=ranking.max_pick if ranking else None,
                ceiling=player_row.ceiling,
                floor=player_row.floor,
                risk_score=player_row.risk_score,
                value_over_replacement=player_row.value_over_replacement,
                notes=player_row.notes or "",
                source=row.name,
                stat_totals=core_stats.from_frame_value(player_row.stat_totals),
                projection_imputed=bool(player_row.projection_imputed),
                projection_source=player_row.projection_source or "",
                projection_detail=player_row.projection_detail or "",
                outcome_band_source=player_row.outcome_band_source or "",
                adp_stdev_is_estimated=bool(player_row.adp_stdev_is_estimated),
                ffc_adp=player_row.ffc_adp,
                espn_adp=player_row.espn_adp,
                espn_rank=player_row.espn_rank,
                yahoo_adp=player_row.yahoo_adp,
                yahoo_rank=player_row.yahoo_rank,
                sleeper_rank=player_row.sleeper_rank,
                adp_source_count=player_row.adp_source_count,
                adp_disagreement=player_row.adp_disagreement,
            )
        )

    metadata = PoolMetadata(
        source=row.name,
        imported_at=row.imported_at.isoformat(timespec="seconds") if row.imported_at else "",
        timestamp_basis=row.timestamp_basis or core_freshness.FETCHED,
        season=row.season,
        platform=row.platform,
        player_count=len(players),
        is_sample_data=bool(row.is_sample_data),
        notes=row.notes or "",
    )
    return PlayerPool(players, league=league, metadata=metadata)


def list_player_sources(session: Session) -> list[dict[str, Any]]:
    rows = session.execute(
        select(PlayerDataSourceRow).order_by(PlayerDataSourceRow.imported_at.desc())
    ).scalars().all()
    return [
        {
            "source_id": int(r.id),
            "name": r.name,
            "season": r.season,
            "platform": r.platform,
            "source_kind": r.source_kind,
            "player_count": int(r.player_count or 0),
            "is_sample_data": bool(r.is_sample_data),
            "imported_at": r.imported_at,
        }
        for r in rows
    ]


def delete_player_source(session: Session, source_id: int) -> bool:
    row = session.get(PlayerDataSourceRow, source_id)
    if row is None:
        return False
    session.delete(row)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Mock drafts
# ─────────────────────────────────────────────────────────────────────────────
def save_mock_draft(
    session: Session,
    result: MockDraftResult,
    *,
    league_id: int | None = None,
    rosters: Sequence[TeamRoster] | None = None,
    roster_metrics: dict[str, dict[str, float]] | None = None,
    user_slots: Iterable[int] | None = None,
) -> int:
    """Persist a completed (or in-progress) mock draft with picks and rosters."""
    row: MockDraftRow | None = None
    if result.mock_id is not None:
        row = session.get(MockDraftRow, result.mock_id)
    if row is None:
        row = MockDraftRow()
        session.add(row)

    row.league_id = league_id
    row.name = result.name
    row.league_name = result.league_name
    row.season = int(result.season) if result.season else None
    row.mode = str(result.mode)
    row.status = "complete"
    row.random_seed = result.random_seed
    row.user_slots = list(result.user_slots or tuple(user_slots or ()))
    row.settings_snapshot = dict(result.settings_snapshot or {})
    row.notes = result.notes or ""
    session.flush()

    row.picks.clear()
    row.rosters.clear()
    session.flush()

    for pick in result.picks:
        row.picks.append(_mock_pick_row(pick))

    metrics = roster_metrics or {}
    for roster in rosters or []:
        stats = metrics.get(roster.manager_name, {})
        row.rosters.append(
            MockDraftRosterRow(
                manager_name=roster.manager_name,
                draft_slot=int(roster.draft_slot),
                is_user=int(roster.draft_slot) in set(row.user_slots or []),
                payload=roster.to_dict(),
                starter_projection=stats.get("starter_projection"),
                total_projection=stats.get("total_projection"),
                adp_value=stats.get("adp_value"),
            )
        )
    session.flush()
    result.mock_id = int(row.id)
    LOGGER.info("Saved mock draft '%s' (%s picks)", row.name, len(result.picks))
    return int(row.id)


def _mock_pick_row(pick: Pick) -> MockDraftPickRow:
    """Build a child row; ``mock_id`` is set by the relationship append."""
    return MockDraftPickRow(
        overall_pick=int(pick.overall_pick),
        round_number=int(pick.round_number),
        pick_in_round=int(pick.pick_in_round),
        draft_slot=int(pick.draft_slot),
        manager_name=pick.manager_name,
        player_id=pick.player_id,
        player_name=pick.player_name,
        position=str(pick.position),
        nfl_team=pick.nfl_team,
        assigned_slot=str(pick.assigned_slot),
        is_keeper=bool(pick.is_keeper),
        is_user_pick=bool(pick.is_user_pick),
        was_manual_override=bool(pick.was_manual_override),
        adp_at_pick=pick.adp_at_pick,
        platform_rank_at_pick=pick.platform_rank_at_pick,
        projection=pick.projection,
        pick_probability=pick.pick_probability,
        alternatives=list(pick.alternatives or []),
        explanation=pick.explanation or "",
    )


def load_mock_draft(session: Session, mock_id: int) -> MockDraftResult | None:
    row = session.get(MockDraftRow, mock_id)
    if row is None:
        return None
    picks = [
        Pick(
            overall_pick=int(p.overall_pick),
            round_number=int(p.round_number),
            pick_in_round=int(p.pick_in_round),
            draft_slot=int(p.draft_slot),
            manager_name=p.manager_name,
            player_id=p.player_id,
            player_name=p.player_name,
            position=Position.coerce(p.position, Position.WR),
            nfl_team=p.nfl_team or "FA",
            is_keeper=bool(p.is_keeper),
            is_user_pick=bool(p.is_user_pick),
            was_manual_override=bool(p.was_manual_override),
            assigned_slot=Slot.coerce(p.assigned_slot, Slot.BENCH),
            adp_at_pick=p.adp_at_pick,
            platform_rank_at_pick=p.platform_rank_at_pick,
            projection=p.projection,
            pick_probability=p.pick_probability,
            alternatives=list(p.alternatives or []),
            explanation=p.explanation or "",
        )
        for p in sorted(row.picks, key=lambda p: p.overall_pick)
    ]
    return MockDraftResult(
        name=row.name,
        league_name=row.league_name or "",
        season=int(row.season or 0),
        picks=picks,
        user_slots=tuple(row.user_slots or ()),
        random_seed=row.random_seed,
        mode=row.mode or "interactive",
        created_at=row.created_at.isoformat(timespec="seconds") if row.created_at else "",
        notes=row.notes or "",
        mock_id=int(row.id),
        settings_snapshot=dict(row.settings_snapshot or {}),
    )


def list_mock_drafts(
    session: Session, league_id: int | None = None
) -> list[dict[str, Any]]:
    statement = select(MockDraftRow).order_by(MockDraftRow.created_at.desc())
    if league_id is not None:
        statement = statement.where(MockDraftRow.league_id == league_id)
    rows = session.execute(statement).scalars().all()
    return [
        {
            "mock_id": int(r.id),
            "name": r.name,
            "league_name": r.league_name,
            "season": r.season,
            "mode": r.mode,
            "random_seed": r.random_seed,
            "pick_count": len(r.picks),
            "created_at": r.created_at,
        }
        for r in rows
    ]


def delete_mock_draft(session: Session, mock_id: int) -> bool:
    row = session.get(MockDraftRow, mock_id)
    if row is None:
        return False
    session.delete(row)
    return True


def load_mock_roster_payloads(session: Session, mock_id: int) -> list[dict[str, Any]]:
    """Stored roster payloads for a mock, ordered by draft slot."""
    rows = session.execute(
        select(MockDraftRosterRow)
        .where(MockDraftRosterRow.mock_id == mock_id)
        .order_by(MockDraftRosterRow.draft_slot)
    ).scalars().all()
    return [
        {
            "manager_name": r.manager_name,
            "draft_slot": int(r.draft_slot),
            "is_user": bool(r.is_user),
            "starter_projection": r.starter_projection,
            "total_projection": r.total_projection,
            "adp_value": r.adp_value,
            "roster": dict(r.payload or {}),
        }
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runs
# ─────────────────────────────────────────────────────────────────────────────
def save_simulation_run(
    session: Session,
    *,
    name: str,
    run_kind: str,
    iterations: int,
    results: Sequence[dict[str, Any]],
    league_id: int | None = None,
    random_seed: int | None = None,
    user_slot: int | None = None,
    settings_snapshot: dict[str, Any] | None = None,
    duration_seconds: float | None = None,
    notes: str = "",
) -> int:
    """Store an aggregated simulation run.

    Each entry in ``results`` needs ``metric_kind`` and ``subject``; ``context``,
    ``value`` and ``payload`` are optional.
    """
    row = SimulationRunRow(
        league_id=league_id,
        name=name,
        run_kind=run_kind,
        iterations=int(iterations),
        random_seed=random_seed,
        user_slot=user_slot,
        settings_snapshot=settings_snapshot,
        duration_seconds=duration_seconds,
        notes=notes,
    )
    session.add(row)
    session.flush()
    for entry in results:
        row.results.append(
            SimulationResultRow(
                metric_kind=str(entry.get("metric_kind", "metric")),
                subject=str(entry.get("subject", "")),
                context=str(entry.get("context", "")),
                value=entry.get("value"),
                payload=entry.get("payload"),
            )
        )
    session.flush()
    LOGGER.info("Saved simulation run '%s' (%s results)", name, len(results))
    return int(row.id)


def load_simulation_results(session: Session, run_id: int) -> list[dict[str, Any]]:
    rows = session.execute(
        select(SimulationResultRow).where(SimulationResultRow.run_id == run_id)
    ).scalars().all()
    return [
        {
            "metric_kind": r.metric_kind,
            "subject": r.subject,
            "context": r.context,
            "value": r.value,
            "payload": r.payload,
        }
        for r in rows
    ]


def list_simulation_runs(
    session: Session, league_id: int | None = None
) -> list[dict[str, Any]]:
    statement = select(SimulationRunRow).order_by(SimulationRunRow.created_at.desc())
    if league_id is not None:
        statement = statement.where(SimulationRunRow.league_id == league_id)
    rows = session.execute(statement).scalars().all()
    return [
        {
            "run_id": int(r.id),
            "name": r.name,
            "run_kind": r.run_kind,
            "iterations": int(r.iterations or 0),
            "random_seed": r.random_seed,
            "duration_seconds": r.duration_seconds,
            "created_at": r.created_at,
        }
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Settings passthrough
# ─────────────────────────────────────────────────────────────────────────────
def read_setting(key: str, default: Any = None) -> Any:
    """Read an application setting using a short-lived session."""
    with session_scope() as session:
        value = get_setting(session, key)
    return default if value is None else value


def write_setting(key: str, value: Any) -> None:
    """Write an application setting using a short-lived session."""
    with session_scope() as session:
        set_setting(session, key, value)


def all_settings() -> dict[str, Any]:
    with session_scope() as session:
        rows = session.execute(select(ApplicationSettingRow)).scalars().all()
        return {
            r.key: (r.value_json if r.value_json is not None else r.value_text)
            for r in rows
        }


__all__ = [
    "save_league", "load_league", "list_leagues", "delete_league",
    "save_history", "save_historical_draft", "load_history",
    "delete_historical_draft",
    "save_manager_profile", "load_manager_profiles", "clear_manager_profiles",
    "save_player_pool", "load_player_pool", "list_player_sources",
    "delete_player_source",
    "save_mock_draft", "load_mock_draft", "list_mock_drafts",
    "delete_mock_draft", "load_mock_roster_payloads",
    "save_simulation_run", "load_simulation_results", "list_simulation_runs",
    "read_setting", "write_setting", "all_settings",
    "normalize_manager_key",
]
