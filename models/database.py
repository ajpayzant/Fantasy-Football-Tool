"""SQLAlchemy schema and session management for the local SQLite database.

The schema is normalised (see the table list in the README) and versioned via
``application_settings.schema_version``. :func:`init_db` is idempotent and
performs additive migrations when the stored version is behind
:data:`core.constants.SCHEMA_VERSION`.

JSON-ish payloads (scoring rules, model weights, profile parameter bags) are
stored in ``JSON`` columns rather than exploded into columns, because they are
read and written as whole objects and their keys evolve with the model.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    inspect,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker
from sqlalchemy.pool import StaticPool

from core.config import DEFAULT_PATHS
from core.constants import SCHEMA_VERSION

LOGGER = logging.getLogger("fantasy_mock_draft.database")


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp (SQLite stores naive, so we normalise)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    """Declarative base with created/updated timestamps on every table."""


class TimestampMixin:
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)


# ─────────────────────────────────────────────────────────────────────────────
# League
# ─────────────────────────────────────────────────────────────────────────────
class LeagueRow(Base, TimestampMixin):
    __tablename__ = "leagues"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    season = Column(Integer, nullable=False)
    platform = Column(String(40), nullable=False, default="Custom")
    team_count = Column(Integer, nullable=False, default=12)
    rounds = Column(Integer, nullable=False, default=16)
    draft_type = Column(String(40), nullable=False, default="snake")
    league_format = Column(String(20), nullable=False, default="redraft")
    user_draft_slot = Column(Integer, nullable=False, default=1)
    draft_date = Column(String(40), nullable=True)
    reversal_round = Column(Integer, nullable=False, default=3)
    custom_round_order = Column(JSON, nullable=True)
    notes = Column(Text, default="")

    roster_slots = relationship(
        "LeagueRosterSlotRow", back_populates="league",
        cascade="all, delete-orphan", lazy="selectin",
    )
    scoring_rules = relationship(
        "LeagueScoringRuleRow", back_populates="league",
        cascade="all, delete-orphan", lazy="selectin",
    )
    managers = relationship(
        "ManagerRow", back_populates="league",
        cascade="all, delete-orphan", lazy="selectin",
    )
    keepers = relationship(
        "KeeperRow", back_populates="league",
        cascade="all, delete-orphan", lazy="selectin",
    )

    __table_args__ = (UniqueConstraint("name", "season", name="uq_league_name_season"),)


class LeagueRosterSlotRow(Base, TimestampMixin):
    """One row per slot type per league (e.g. WR → 3)."""

    __tablename__ = "league_roster_slots"

    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey("leagues.id", ondelete="CASCADE"),
                       nullable=False, index=True)
    slot = Column(String(20), nullable=False)
    count = Column(Integer, nullable=False, default=0)
    position_min = Column(Integer, nullable=True)
    position_max = Column(Integer, nullable=True)

    league = relationship("LeagueRow", back_populates="roster_slots")

    __table_args__ = (UniqueConstraint("league_id", "slot", name="uq_slot_per_league"),)


class LeagueScoringRuleRow(Base, TimestampMixin):
    """Scoring stored as key/value rows so custom rules need no migration."""

    __tablename__ = "league_scoring_rules"

    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey("leagues.id", ondelete="CASCADE"),
                       nullable=False, index=True)
    rule_key = Column(String(60), nullable=False)
    rule_value = Column(Float, nullable=True)
    rule_text = Column(String(60), nullable=True)
    """Used for non-numeric values such as the preset name."""

    league = relationship("LeagueRow", back_populates="scoring_rules")

    __table_args__ = (UniqueConstraint("league_id", "rule_key", name="uq_rule_per_league"),)


# ─────────────────────────────────────────────────────────────────────────────
# Managers
# ─────────────────────────────────────────────────────────────────────────────
class ManagerRow(Base, TimestampMixin):
    __tablename__ = "managers"

    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey("leagues.id", ondelete="CASCADE"),
                       nullable=False, index=True)
    name = Column(String(80), nullable=False)
    manager_key = Column(String(80), nullable=False, index=True)
    team_name = Column(String(120), default="")
    draft_slot = Column(Integer, nullable=False)
    is_user = Column(Boolean, nullable=False, default=False)
    archetype = Column(String(40), nullable=False, default="balanced")

    league = relationship("LeagueRow", back_populates="managers")
    preferences = relationship(
        "ManagerManualPreferenceRow", back_populates="manager",
        cascade="all, delete-orphan", uselist=False, lazy="selectin",
    )
    profile = relationship(
        "ManagerProfileRow", back_populates="manager",
        cascade="all, delete-orphan", uselist=False, lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint("league_id", "manager_key", name="uq_manager_per_league"),
        UniqueConstraint("league_id", "draft_slot", name="uq_slot_per_league_mgr"),
    )


class ManagerManualPreferenceRow(Base, TimestampMixin):
    """User-asserted knowledge about a manager (kept separate from the model)."""

    __tablename__ = "manager_manual_preferences"

    id = Column(Integer, primary_key=True)
    manager_id = Column(Integer, ForeignKey("managers.id", ondelete="CASCADE"),
                        nullable=False, unique=True, index=True)
    payload = Column(JSON, nullable=False, default=dict)
    """Serialised :class:`models.manager.ManagerPreferences`."""

    manager = relationship("ManagerRow", back_populates="preferences")


class ManagerProfileRow(Base, TimestampMixin):
    """Cached modelled profile, rebuilt whenever history or settings change."""

    __tablename__ = "manager_profiles"

    id = Column(Integer, primary_key=True)
    manager_id = Column(Integer, ForeignKey("managers.id", ondelete="CASCADE"),
                        nullable=False, unique=True, index=True)
    archetype = Column(String(40), nullable=False, default="balanced")
    sample_picks = Column(Float, nullable=False, default=0.0)
    sample_drafts = Column(Integer, nullable=False, default=0)
    payload = Column(JSON, nullable=False, default=dict)
    """Serialised :class:`models.manager.ManagerProfile`."""
    built_from_seasons = Column(JSON, nullable=True)
    shrinkage_config = Column(JSON, nullable=True)

    manager = relationship("ManagerRow", back_populates="profile")


# ─────────────────────────────────────────────────────────────────────────────
# Keepers
# ─────────────────────────────────────────────────────────────────────────────
class KeeperRow(Base, TimestampMixin):
    __tablename__ = "keepers"

    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey("leagues.id", ondelete="CASCADE"),
                       nullable=False, index=True)
    manager_name = Column(String(80), nullable=False)
    player_name = Column(String(120), nullable=False)
    keeper_round = Column(Integer, nullable=True)
    overall_pick = Column(Integer, nullable=True)
    removes_pick = Column(Boolean, nullable=False, default=True)
    salary = Column(Float, nullable=True)
    position = Column(String(10), nullable=True)
    nfl_team = Column(String(10), nullable=True)
    notes = Column(Text, default="")

    league = relationship("LeagueRow", back_populates="keepers")


# ─────────────────────────────────────────────────────────────────────────────
# Historical drafts
# ─────────────────────────────────────────────────────────────────────────────
class HistoricalDraftRow(Base, TimestampMixin):
    __tablename__ = "historical_drafts"

    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey("leagues.id", ondelete="CASCADE"),
                       nullable=True, index=True)
    season = Column(Integer, nullable=False, index=True)
    league_name = Column(String(120), default="")
    platform = Column(String(40), default="")
    team_count = Column(Integer, nullable=True)
    rounds = Column(Integer, nullable=True)
    draft_date = Column(String(40), nullable=True)
    source_file = Column(String(255), default="")

    picks = relationship(
        "HistoricalPickRow", back_populates="draft",
        cascade="all, delete-orphan", lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint("league_id", "season", "league_name",
                         name="uq_hist_draft_season"),
    )


class HistoricalPickRow(Base, TimestampMixin):
    """A historical pick plus its engineered features."""

    __tablename__ = "historical_picks"

    id = Column(Integer, primary_key=True)
    draft_id = Column(Integer, ForeignKey("historical_drafts.id", ondelete="CASCADE"),
                      nullable=False, index=True)
    season = Column(Integer, nullable=False, index=True)
    manager_name = Column(String(80), nullable=False)
    manager_key = Column(String(80), nullable=False, index=True)
    overall_pick = Column(Integer, nullable=False)
    round_number = Column(Integer, nullable=True)
    pick_in_round = Column(Integer, nullable=True)
    player_name = Column(String(120), nullable=False)
    position = Column(String(10), nullable=True)
    nfl_team = Column(String(10), default="")
    adp = Column(Float, nullable=True)
    platform_rank = Column(Float, nullable=True)
    projection = Column(Float, nullable=True)
    tier = Column(Integer, nullable=True)
    is_keeper = Column(Boolean, default=False)
    is_rookie = Column(Boolean, default=False)
    bye_week = Column(Integer, nullable=True)
    draft_date = Column(String(40), nullable=True)
    platform = Column(String(40), default="")
    league_name = Column(String(120), default="")
    features = Column(JSON, nullable=True)
    """Engineered feature bag (adp_delta, run flags, stack/handcuff, …)."""

    draft = relationship("HistoricalDraftRow", back_populates="picks")

    __table_args__ = (
        Index("ix_hist_pick_manager_season", "manager_key", "season"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Players and rankings
# ─────────────────────────────────────────────────────────────────────────────
class PlayerDataSourceRow(Base, TimestampMixin):
    """Provenance for one import of player data."""

    __tablename__ = "player_data_sources"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    season = Column(Integer, nullable=True, index=True)
    platform = Column(String(40), nullable=True)
    source_kind = Column(String(30), default="upload")
    """upload | sample | api | manual"""
    file_name = Column(String(255), default="")
    imported_at = Column(DateTime, default=utcnow, nullable=False)
    player_count = Column(Integer, default=0)
    is_sample_data = Column(Boolean, default=False)
    notes = Column(Text, default="")
    validation_summary = Column(JSON, nullable=True)

    players = relationship(
        "PlayerRow", back_populates="source",
        cascade="all, delete-orphan", lazy="selectin",
    )
    rankings = relationship(
        "PlayerRankingRow", back_populates="source",
        cascade="all, delete-orphan", lazy="selectin",
    )


class PlayerRow(Base, TimestampMixin):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("player_data_sources.id", ondelete="CASCADE"),
                       nullable=False, index=True)
    player_key = Column(String(120), nullable=False, index=True)
    """Normalised name+position key used for cross-source joins."""
    player_name = Column(String(120), nullable=False)
    position = Column(String(10), nullable=False, index=True)
    nfl_team = Column(String(10), default="FA")
    bye_week = Column(Integer, nullable=True)
    experience = Column(Integer, nullable=True)
    is_rookie = Column(Boolean, default=False)
    injury_status = Column(String(20), default="Healthy")
    suspended = Column(Boolean, default=False)
    projection = Column(Float, nullable=True)
    overall_rank = Column(Float, nullable=True)
    position_rank = Column(Integer, nullable=True)
    tier = Column(Integer, nullable=True)
    ceiling = Column(Float, nullable=True)
    floor = Column(Float, nullable=True)
    risk_score = Column(Float, nullable=True)
    value_over_replacement = Column(Float, nullable=True)
    notes = Column(Text, default="")

    source = relationship("PlayerDataSourceRow", back_populates="players")
    rankings = relationship(
        "PlayerRankingRow", back_populates="player",
        cascade="all, delete-orphan", lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint("source_id", "player_key", name="uq_player_per_source"),
    )


class PlayerRankingRow(Base, TimestampMixin):
    """Platform-specific ranks / ADP, allowing several sets to coexist."""

    __tablename__ = "player_rankings"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id", ondelete="CASCADE"),
                       nullable=False, index=True)
    source_id = Column(Integer, ForeignKey("player_data_sources.id", ondelete="CASCADE"),
                       nullable=False, index=True)
    platform = Column(String(40), nullable=False, default="Custom")
    ranking_kind = Column(String(30), default="platform_adp")
    overall_adp = Column(Float, nullable=True)
    platform_adp = Column(Float, nullable=True)
    platform_rank = Column(Float, nullable=True)
    adp_stdev = Column(Float, nullable=True)
    min_pick = Column(Integer, nullable=True)
    max_pick = Column(Integer, nullable=True)

    player = relationship("PlayerRow", back_populates="rankings")
    source = relationship("PlayerDataSourceRow", back_populates="rankings")

    __table_args__ = (
        UniqueConstraint("player_id", "source_id", "ranking_kind",
                         name="uq_ranking_per_player_source"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mock drafts
# ─────────────────────────────────────────────────────────────────────────────
class MockDraftRow(Base, TimestampMixin):
    __tablename__ = "mock_drafts"

    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey("leagues.id", ondelete="CASCADE"),
                       nullable=True, index=True)
    name = Column(String(120), nullable=False)
    league_name = Column(String(120), default="")
    season = Column(Integer, nullable=True)
    mode = Column(String(30), default="interactive")
    status = Column(String(20), default="complete")
    random_seed = Column(Integer, nullable=True)
    user_slots = Column(JSON, nullable=True)
    settings_snapshot = Column(JSON, nullable=True)
    """Full league + model settings at run time, so a reload is faithful."""
    notes = Column(Text, default="")

    picks = relationship(
        "MockDraftPickRow", back_populates="mock",
        cascade="all, delete-orphan", lazy="selectin",
    )
    rosters = relationship(
        "MockDraftRosterRow", back_populates="mock",
        cascade="all, delete-orphan", lazy="selectin",
    )


class MockDraftPickRow(Base, TimestampMixin):
    __tablename__ = "mock_draft_picks"

    id = Column(Integer, primary_key=True)
    mock_id = Column(Integer, ForeignKey("mock_drafts.id", ondelete="CASCADE"),
                     nullable=False, index=True)
    overall_pick = Column(Integer, nullable=False)
    round_number = Column(Integer, nullable=False)
    pick_in_round = Column(Integer, nullable=False)
    draft_slot = Column(Integer, nullable=False)
    manager_name = Column(String(80), nullable=False)
    player_id = Column(String(120), nullable=False)
    player_name = Column(String(120), nullable=False)
    position = Column(String(10), nullable=False)
    nfl_team = Column(String(10), default="FA")
    assigned_slot = Column(String(20), default="BN")
    is_keeper = Column(Boolean, default=False)
    is_user_pick = Column(Boolean, default=False)
    was_manual_override = Column(Boolean, default=False)
    adp_at_pick = Column(Float, nullable=True)
    platform_rank_at_pick = Column(Float, nullable=True)
    projection = Column(Float, nullable=True)
    tier = Column(Integer, nullable=True)
    pick_probability = Column(Float, nullable=True)
    alternatives = Column(JSON, nullable=True)
    explanation = Column(Text, default="")

    mock = relationship("MockDraftRow", back_populates="picks")

    __table_args__ = (
        UniqueConstraint("mock_id", "overall_pick", name="uq_pick_per_mock"),
    )


class MockDraftRosterRow(Base, TimestampMixin):
    """Denormalised final roster per team — cheap to read for review pages."""

    __tablename__ = "mock_draft_rosters"

    id = Column(Integer, primary_key=True)
    mock_id = Column(Integer, ForeignKey("mock_drafts.id", ondelete="CASCADE"),
                     nullable=False, index=True)
    manager_name = Column(String(80), nullable=False)
    draft_slot = Column(Integer, nullable=False)
    is_user = Column(Boolean, default=False)
    payload = Column(JSON, nullable=False, default=dict)
    """Serialised :class:`models.draft.TeamRoster` plus analysis metrics."""
    starter_projection = Column(Float, nullable=True)
    total_projection = Column(Float, nullable=True)
    adp_value = Column(Float, nullable=True)

    mock = relationship("MockDraftRow", back_populates="rosters")


# ─────────────────────────────────────────────────────────────────────────────
# Simulations
# ─────────────────────────────────────────────────────────────────────────────
class SimulationRunRow(Base, TimestampMixin):
    __tablename__ = "simulation_runs"

    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey("leagues.id", ondelete="CASCADE"),
                       nullable=True, index=True)
    name = Column(String(120), default="")
    run_kind = Column(String(30), default="monte_carlo")
    """monte_carlo | availability | backtest"""
    iterations = Column(Integer, default=0)
    random_seed = Column(Integer, nullable=True)
    user_slot = Column(Integer, nullable=True)
    settings_snapshot = Column(JSON, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    notes = Column(Text, default="")

    results = relationship(
        "SimulationResultRow", back_populates="run",
        cascade="all, delete-orphan", lazy="selectin",
    )


class SimulationResultRow(Base, TimestampMixin):
    """One aggregated metric row from a simulation run."""

    __tablename__ = "simulation_results"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("simulation_runs.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    metric_kind = Column(String(40), nullable=False, index=True)
    """availability | exposure | strategy | run_frequency | backtest_metric …"""
    subject = Column(String(160), nullable=False, default="")
    """Player name, position, manager, or metric name."""
    context = Column(String(80), default="")
    """Pick number, round, or strategy label the metric applies to."""
    value = Column(Float, nullable=True)
    payload = Column(JSON, nullable=True)

    run = relationship("SimulationRunRow", back_populates="results")

    __table_args__ = (
        Index("ix_sim_result_kind_subject", "metric_kind", "subject"),
    )


class ApplicationSettingRow(Base, TimestampMixin):
    """Key/value app settings, including the schema version."""

    __tablename__ = "application_settings"

    id = Column(Integer, primary_key=True)
    key = Column(String(80), nullable=False, unique=True, index=True)
    value_text = Column(Text, nullable=True)
    value_json = Column(JSON, nullable=True)


ALL_TABLES: tuple[str, ...] = (
    "leagues", "league_roster_slots", "league_scoring_rules", "managers",
    "manager_manual_preferences", "historical_drafts", "historical_picks",
    "players", "player_data_sources", "player_rankings", "manager_profiles",
    "mock_drafts", "mock_draft_picks", "mock_draft_rosters", "simulation_runs",
    "simulation_results", "keepers", "application_settings",
)


# ─────────────────────────────────────────────────────────────────────────────
# Engine / session management
# ─────────────────────────────────────────────────────────────────────────────
_ENGINE: Any = None
_SESSION_FACTORY: Any = None
_DB_PATH: str | None = None


def database_path() -> str:
    return _DB_PATH or DEFAULT_PATHS.database


def get_engine(db_path: str | None = None, *, echo: bool = False) -> Any:
    """Return the process-wide engine, creating it on first use.

    ``db_path=None`` means "whichever database is already configured", falling
    back to :data:`core.config.DEFAULT_PATHS`. Passing an explicit, different
    path rebuilds the engine — used by tests to point at a temporary database.
    """
    global _ENGINE, _SESSION_FACTORY, _DB_PATH

    target = db_path or _DB_PATH or DEFAULT_PATHS.database
    if _ENGINE is not None and target == _DB_PATH:
        return _ENGINE

    extra_kwargs: dict[str, Any] = {}
    if target != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        url = f"sqlite:///{target}"
    else:
        url = "sqlite://"
        # Without StaticPool every in-memory connection gets its own blank
        # database, so tables created in one session vanish in the next.
        extra_kwargs["poolclass"] = StaticPool

    engine = create_engine(
        url,
        echo=echo,
        future=True,
        # Streamlit reruns touch the DB from several threads.
        connect_args={"check_same_thread": False},
        **extra_kwargs,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    _ENGINE = engine
    _SESSION_FACTORY = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    _DB_PATH = target
    return engine


def get_session_factory(db_path: str | None = None) -> Any:
    get_engine(db_path)
    return _SESSION_FACTORY


@contextmanager
def session_scope(db_path: str | None = None) -> Iterator[Session]:
    """Transactional session context. Commits on success, rolls back on error."""
    factory = get_session_factory(db_path)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(db_path: str | None = None, *, echo: bool = False) -> str:
    """Create or migrate the schema. Idempotent; returns the database path."""
    engine = get_engine(db_path, echo=echo)
    Base.metadata.create_all(engine)

    with session_scope(db_path) as session:
        stored = get_setting(session, "schema_version")
        current = int(stored) if stored is not None else None
        if current is None:
            set_setting(session, "schema_version", SCHEMA_VERSION)
            LOGGER.info("Initialised database schema v%s at %s",
                        SCHEMA_VERSION, database_path())
        elif current < SCHEMA_VERSION:
            _migrate(session, current, SCHEMA_VERSION)
            set_setting(session, "schema_version", SCHEMA_VERSION)
            LOGGER.info("Migrated database schema v%s → v%s", current, SCHEMA_VERSION)
        elif current > SCHEMA_VERSION:
            LOGGER.warning(
                "Database schema v%s is newer than this build (v%s). "
                "Some columns may be ignored.", current, SCHEMA_VERSION,
            )
    return database_path()


def _add_column_if_missing(session: Session, table: str, column: str, ddl_type: str) -> bool:
    """Add a column to an existing table when it is absent. Idempotent."""
    inspector = inspect(session.get_bind())
    if table not in inspector.get_table_names():
        return False
    if column in {c["name"] for c in inspector.get_columns(table)}:
        return False
    session.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))
    LOGGER.info("Migration: added %s.%s", table, column)
    return True


def _migrate(session: Session, from_version: int, to_version: int) -> None:
    """Apply additive migrations between schema versions.

    ``create_all`` already creates new *tables*; this handles new *columns* on
    tables that already exist. Steps are written so a partially-applied
    migration can be re-run safely.

    Version 1 is the initial schema, so there is nothing to apply yet. Future
    steps take the form::

        if from_version < 2:
            _add_column_if_missing(session, "players", "adot", "FLOAT")
    """
    LOGGER.debug("No migration steps defined for v%s → v%s", from_version, to_version)


def get_setting(session: Session, key: str) -> Any:
    """Read an application setting (JSON value preferred over text)."""
    row = session.execute(
        select(ApplicationSettingRow).where(ApplicationSettingRow.key == key)
    ).scalar_one_or_none()
    if row is None:
        return None
    if row.value_json is not None:
        return row.value_json
    return row.value_text


def set_setting(session: Session, key: str, value: Any) -> None:
    """Upsert an application setting. Scalars go to text, structures to JSON."""
    row = session.execute(
        select(ApplicationSettingRow).where(ApplicationSettingRow.key == key)
    ).scalar_one_or_none()
    if row is None:
        row = ApplicationSettingRow(key=key)
        session.add(row)
    if isinstance(value, (dict, list)):
        row.value_json = value
        row.value_text = None
    else:
        row.value_text = None if value is None else str(value)
        row.value_json = None


def reset_database(db_path: str | None = None) -> str:
    """Drop and recreate every table. Destructive — the UI confirms first."""
    engine = get_engine(db_path)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with session_scope(db_path) as session:
        set_setting(session, "schema_version", SCHEMA_VERSION)
    LOGGER.warning("Database reset at %s", database_path())
    return database_path()


def table_counts(db_path: str | None = None) -> dict[str, int]:
    """Row counts per table, for the Settings page."""
    engine = get_engine(db_path)
    counts: dict[str, int] = {}
    inspector = inspect(engine)
    present = set(inspector.get_table_names())
    with engine.connect() as connection:
        for table in ALL_TABLES:
            if table not in present:
                counts[table] = 0
                continue
            result = connection.execute(text(f"SELECT COUNT(*) FROM {table}"))
            counts[table] = int(result.scalar() or 0)
    return counts


def dispose_engine() -> None:
    """Drop the cached engine — used between tests."""
    global _ENGINE, _SESSION_FACTORY, _DB_PATH
    if _ENGINE is not None:
        _ENGINE.dispose()
    _ENGINE = None
    _SESSION_FACTORY = None
    _DB_PATH = None
