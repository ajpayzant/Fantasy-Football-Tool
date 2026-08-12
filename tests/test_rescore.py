"""Changing a league's scoring must move its projections, offline.

The bug these tests exist for was silent and expensive. Projections were scored
once, when the board was fetched, and the points total was the only thing kept — so
a user who set up a half-PPR league, fetched a board, then corrected their scoring
to full PPR had a board that *looked* recomputed (tiers, VOR, ceiling and floor all
moved) while the projections underneath it were still half-PPR. The only remedy the
app could offer was "download the season from ESPN again", which is a network round
trip to redo arithmetic on numbers already in hand, and which quietly replaces the
whole board's ADP with whatever ESPN's has drifted to since.

The fix is to store the projected *stat line* beside the points. These tests assert
the three things that has to be true for that to be worth anything:

1. Rescoring gives the same answer as scoring the stat line from scratch would.
2. The stat line survives every round trip the app puts a pool through — frame,
   CSV import and the SQLite database — because a rescore is only available where
   the stats are.
3. Everything downstream of a projection is re-derived, not left on the old scale.

Point 3 has a trap in it that a naive implementation falls into: a projection
*estimated from draft position* is a point on a curve fitted to the range of the
real projections, so it is meaningful only on the scale those are on. Rescored, it
would be wrong; left alone, it would be wrong differently. It has to be discarded
and re-derived, and :func:`test_an_estimated_projection_is_re_derived` is the test
that catches an implementation that forgets.
"""

from __future__ import annotations

import pytest

from core import stats as core_stats
from core.config import LeagueConfig, ScoringRules
from core.enums import Position, ScoringPreset
from models.player import Player, PlayerPool, PoolMetadata

# A receiving-heavy stat line per player, so a change in the value of a reception is
# the loudest possible signal. Receptions differ between players so the pool has a
# real spread to fit tiers and bands to.
RECEIVERS: tuple[tuple[str, float, float, float], ...] = (
    # name, receptions, rec_yards, rec_td
    ("Alpha Receiver", 110.0, 1480.0, 11.0),
    ("Bravo Receiver", 92.0, 1210.0, 8.0),
    ("Charlie Receiver", 74.0, 980.0, 6.0),
    ("Delta Receiver", 58.0, 720.0, 4.0),
    ("Echo Receiver", 41.0, 505.0, 2.0),
)

STANDARD = ScoringRules.from_preset(ScoringPreset.STANDARD)
FULL_PPR = ScoringRules.from_preset(ScoringPreset.FULL_PPR)


def _receiver(name: str, receptions: float, yards: float, tds: float,
              adp: float, scoring: ScoringRules) -> Player:
    """A wide receiver whose points were scored from his own stat line."""
    stats = {
        "receptions": receptions, "rec_yards": yards, "rec_td": tds,
        "targets": receptions * 1.5, "games": 17.0,
    }
    return Player(
        player_id=name.lower().replace(" ", "_"),
        name=name,
        position=Position.WR,
        projection=core_stats.score(stats, Position.WR, scoring),
        overall_adp=adp,
        adp_stdev=adp * 0.2,
        stat_totals=stats,
        projection_source="Supplied by your source",
    )


def _pool(
    scoring: ScoringRules = STANDARD,
    *,
    league: LeagueConfig | None = None,
    extra: list[Player] | None = None,
) -> PlayerPool:
    players = [
        _receiver(name, rec, yards, tds, adp=float(index * 8 + 1), scoring=scoring)
        for index, (name, rec, yards, tds) in enumerate(RECEIVERS, start=1)
    ]
    players.extend(extra or [])
    return PlayerPool(
        players, league=league, metadata=PoolMetadata(source="test fixture")
    )


def _league(scoring: ScoringRules) -> LeagueConfig:
    return LeagueConfig(name="Test", season=2026, scoring=scoring)


# ─────────────────────────────────────────────────────────────────────────────
# The arithmetic
# ─────────────────────────────────────────────────────────────────────────────
def test_rescoring_equals_scoring_the_stat_line_from_scratch() -> None:
    """The invariant the whole feature rests on.

    If a rescored projection ever differs from what the scorer would produce given
    the same stats and the same rules, then "rescore in place" and "refetch and
    rescore" disagree, and the user has no way to tell which number they are
    looking at.
    """
    pool = _pool(STANDARD)
    pool.rescore(FULL_PPR)
    for player in pool:
        expected = core_stats.score(player.stat_totals, player.position, FULL_PPR)
        assert expected is not None
        assert player.projection == pytest.approx(round(expected, 1), abs=1e-9), (
            f"{player.name} rescored to {player.projection}, but scoring his stat "
            f"line directly gives {expected}"
        )


def test_a_reception_becoming_worth_a_point_moves_every_receiver() -> None:
    """Standard → full PPR is +1.0 per reception and nothing else. Assert exactly that."""
    pool = _pool(STANDARD)
    before = {p.player_id: float(p.projection) for p in pool}

    outcome = pool.rescore(FULL_PPR)

    assert outcome.rescored == len(RECEIVERS)
    assert outcome.no_stat_line == 0
    for player in pool:
        gained = float(player.projection) - before[player.player_id]
        assert gained == pytest.approx(player.stat_totals["receptions"], abs=0.05), (
            f"{player.name} gained {gained:.1f} points from {player.stat_totals['receptions']:.0f} "
            "receptions; full PPR pays exactly one point each"
        )


def test_rescoring_is_reversible() -> None:
    """There and back again lands on the original number.

    Not a tautology: it fails if a rescore reads its input from the *previous*
    projection rather than from the stored stats, which is the shortcut that would
    make repeated scoring changes compound instead of replace.
    """
    pool = _pool(STANDARD)
    original = {p.player_id: float(p.projection) for p in pool}

    pool.rescore(FULL_PPR)
    pool.rescore(STANDARD)

    for player in pool:
        assert player.projection == pytest.approx(original[player.player_id], abs=0.05)


def test_the_projection_detail_text_is_rewritten_too() -> None:
    """The human-readable stat line must not survive as a stale copy."""
    pool = _pool(STANDARD)
    for player in pool:
        player.projection_detail = "stale text from the old board"

    pool.rescore(FULL_PPR)

    for player in pool:
        assert "stale" not in player.projection_detail
        assert player.projection_detail == core_stats.describe(
            player.stat_totals, player.position
        )


# ─────────────────────────────────────────────────────────────────────────────
# The three kinds of player, handled differently
# ─────────────────────────────────────────────────────────────────────────────
def test_a_projection_with_no_stat_line_is_left_alone_and_counted() -> None:
    """A user's points-only CSV is their number. Converting it would invent data."""
    hand_written = Player(
        player_id="hand_written",
        name="Hand Written",
        position=Position.WR,
        projection=222.0,
        overall_adp=20.0,
        projection_source="From my own spreadsheet",
    )
    pool = _pool(STANDARD, extra=[hand_written])

    outcome = pool.rescore(FULL_PPR)

    assert hand_written.projection == 222.0
    assert outcome.no_stat_line == 1
    assert outcome.rescored == len(RECEIVERS)
    assert "cannot be rescored" in outcome.describe()


def test_an_estimated_projection_is_re_derived() -> None:
    """The trap. An ADP-derived estimate is only meaningful on the old scale.

    Full PPR lifts every real receiver by 40-110 points. An estimate left behind
    would end up ranked among players it was fitted to sit below, so it has to be
    thrown away and re-fitted to the *new* range.
    """
    nobody = Player(
        player_id="nobody",
        name="No Projection Anywhere",
        position=Position.WR,
        overall_adp=200.0,
    )
    pool = _pool(STANDARD, extra=[nobody])
    assert nobody.projection_imputed, "the fixture must actually exercise imputation"
    estimated_before = float(nobody.projection)

    outcome = pool.rescore(FULL_PPR)

    assert outcome.reimputed == 1
    assert nobody.projection_imputed, "still an estimate — it did not become real"
    assert float(nobody.projection) != pytest.approx(estimated_before, abs=0.5), (
        "the estimate is unchanged, so it is still on the pre-rescore scale"
    )
    real = [float(p.projection) for p in pool if p.stat_totals]
    assert min(real) <= float(nobody.projection) <= max(real), (
        "a re-derived estimate must land inside the range of the real projections "
        "it is fitted to"
    )


def test_a_player_given_a_real_projection_stops_being_an_estimate() -> None:
    """Otherwise a stale flag makes the next rescore discard a real number."""
    nobody = Player(
        player_id="nobody", name="Late Arrival", position=Position.WR, overall_adp=200.0
    )
    pool = _pool(STANDARD, extra=[nobody])
    assert nobody.projection_imputed

    nobody.projection = 191.0
    nobody.stat_totals = {"receptions": 60.0, "rec_yards": 700.0, "rec_td": 4.0}
    pool.rescore(STANDARD)

    assert not nobody.projection_imputed
    assert float(nobody.projection) == pytest.approx(
        round(core_stats.score(nobody.stat_totals, Position.WR, STANDARD), 1)
    )


def test_a_board_with_nothing_to_rescore_says_so_rather_than_pretending() -> None:
    """An older save carries points but no stats. The UI needs to be able to tell."""
    pool = PlayerPool(
        [
            Player(player_id=f"p{i}", name=f"Player {i}", position=Position.WR,
                   projection=200.0 - i * 10, overall_adp=float(i))
            for i in range(1, 6)
        ],
        metadata=PoolMetadata(source="points only"),
    )
    outcome = pool.rescore(FULL_PPR)

    assert outcome.rescored == 0
    assert outcome.changed == 0
    assert outcome.no_stat_line == 5
    assert "No projections could be rescored" in outcome.describe()


def test_per_game_bonuses_are_reported_as_unappliable() -> None:
    """A season total cannot say how many single games cleared 100 yards.

    Silently ignoring the rule would be the worst option: the user set a value, and
    every projection would quietly omit it.
    """
    with_bonus = ScoringRules.from_preset(
        ScoringPreset.FULL_PPR, bonus_rec_100_yards=3.0
    )
    pool = _pool(STANDARD)
    outcome = pool.rescore(with_bonus)

    assert any("100-yard receiving" in rule for rule in outcome.unscorable_rules)
    assert _pool(STANDARD).rescore(FULL_PPR).unscorable_rules == []


# ─────────────────────────────────────────────────────────────────────────────
# Everything downstream of a projection
# ─────────────────────────────────────────────────────────────────────────────
def test_derived_tiers_and_bands_are_re_derived() -> None:
    """Both are read off the position's projection curve, which has just moved."""
    pool = _pool(STANDARD)
    tiers_before = {p.player_id: p.tier for p in pool}
    ceilings_before = {p.player_id: p.ceiling for p in pool}
    assert all(p.tier_source for p in pool), "fixture must have derived tiers"

    pool.rescore(FULL_PPR)

    assert all(p.tier is not None for p in pool), "a cleared tier must be refilled"
    assert all(p.tier_source for p in pool), "and re-explained"
    assert all(p.ceiling is not None and p.floor is not None for p in pool)
    moved = [
        p.player_id for p in pool
        if p.ceiling != ceilings_before[p.player_id]
    ]
    assert moved, "ceilings are in points, so they must move with the scoring scale"
    assert tiers_before  # kept for the diff a failure above would want


def test_a_supplied_tier_or_band_is_not_overwritten() -> None:
    """A source's own opinion is not this app's to replace.

    Identifiable because ``tier_source`` and ``outcome_band_source`` are only ever
    written when this app derived the value.
    """
    opinionated = _receiver("Opinion Holder", 80.0, 1000.0, 7.0, adp=12.0,
                            scoring=STANDARD)
    opinionated.tier = 9
    opinionated.ceiling = 1234.0
    opinionated.floor = 111.0
    opinionated.risk_score = 0.42
    pool = _pool(STANDARD, extra=[opinionated])
    assert not opinionated.tier_source and not opinionated.outcome_band_source

    pool.rescore(FULL_PPR)

    assert opinionated.tier == 9
    assert opinionated.ceiling == 1234.0
    assert opinionated.floor == 111.0
    assert opinionated.risk_score == 0.42


def test_value_over_replacement_follows_the_new_projections() -> None:
    """VOR is projection minus the replacement's projection, both of which moved."""
    league = _league(STANDARD)
    pool = _pool(STANDARD, league=league)
    before = {p.player_id: float(p.value_over_replacement) for p in pool}

    pool.rescore(FULL_PPR)

    for player in pool:
        assert player.value_over_replacement == pytest.approx(
            float(player.projection) - float(player.replacement_points), abs=1e-6
        )
    assert any(
        abs(float(p.value_over_replacement) - before[p.player_id]) > 1.0 for p in pool
    ), "no VOR moved, so it was not recomputed"


def test_expected_points_tracks_the_projection() -> None:
    """The engine reads ``expected_points``; a stale copy would undo the rescore."""
    pool = _pool(STANDARD)
    pool.rescore(FULL_PPR)
    for player in pool:
        assert player.expected_points == pytest.approx(float(player.projection))


def test_rescoring_does_not_disturb_adp_or_ranks() -> None:
    """The point of rescoring offline is that the draft-position data stays put.

    A refetch is the alternative, and a refetch changes ADP — which is exactly the
    hidden cost this feature removes.
    """
    pool = _pool(STANDARD)
    before = {
        p.player_id: (p.overall_adp, p.platform_adp, p.overall_rank, p.adp_stdev)
        for p in pool
    }
    pool.rescore(FULL_PPR)
    for player in pool:
        assert (
            player.overall_adp, player.platform_adp, player.overall_rank,
            player.adp_stdev,
        ) == before[player.player_id]


# ─────────────────────────────────────────────────────────────────────────────
# The stat line has to survive every round trip, or there is nothing to rescore
# ─────────────────────────────────────────────────────────────────────────────
def test_the_stat_line_survives_a_frame_round_trip() -> None:
    """``to_frame`` → ``from_frame`` is how the pool reaches the UI and CSV export."""
    pool = _pool(STANDARD)
    frame = pool.to_frame()
    assert "stat_totals" in frame.columns

    restored = PlayerPool.from_frame(frame)
    for player in pool:
        other = restored.require(player.player_id)
        assert other.stat_totals == pytest.approx(player.stat_totals)
        assert other.projection_imputed == player.projection_imputed

    restored.rescore(FULL_PPR)
    for player in restored:
        assert player.projection == pytest.approx(
            round(core_stats.score(player.stat_totals, player.position, FULL_PPR), 1)
        )


def test_the_stat_line_survives_the_csv_importer() -> None:
    """A user can export a board, reopen it, and still change their scoring."""
    from services.importers import import_player_pool

    pool = _pool(STANDARD)
    frame = pool.to_frame().rename(columns={"player_name": "player_name"})
    result = import_player_pool(frame, source="round trip")
    assert result.pool is not None

    for player in result.pool:
        assert player.stat_totals, f"{player.name} lost its stat line through import"
    assert result.pool.rescore(FULL_PPR).rescored == len(RECEIVERS)


def test_the_import_template_documents_the_stat_line_column() -> None:
    """The template is the format spec for anyone supplying their own projections."""
    from services.importers import player_template

    template = player_template()
    assert "stat_totals" in template.columns
    decoded = core_stats.from_frame_value(template.iloc[0]["stat_totals"])
    assert decoded, "the example row must show what the column looks like"
    assert core_stats.score(decoded, Position.RB, FULL_PPR) is not None, (
        "the example stat line must actually be scorable, or it teaches the wrong shape"
    )


def test_the_stat_line_survives_the_database(tmp_path) -> None:
    """Saving a pool and loading it back must not cost the ability to rescore.

    This is the round trip that was silently lossy: ``players`` had no column for any
    of the provenance fields, so a reloaded board came back with no per-platform ADP,
    no projection provenance, and no stats to rescore from.
    """
    from models.database import dispose_engine, init_db, session_scope
    from services.repository import load_player_pool, save_player_pool

    dispose_engine()
    db = str(tmp_path / "round_trip.db")
    init_db(db)
    try:
        pool = _pool(STANDARD)
        for index, player in enumerate(pool, start=1):
            player.espn_adp = float(index)
            player.yahoo_adp = float(index) + 0.5
            player.ffc_adp = float(index) + 0.25
            player.sleeper_rank = float(index)
            player.adp_source_count = 3
            player.adp_stdev_is_estimated = True

        with session_scope(db) as session:
            source_id = save_player_pool(session, pool)
        with session_scope(db) as session:
            restored = load_player_pool(session, source_id)

        assert restored is not None and len(restored) == len(pool)
        for player in pool:
            other = restored.require(player.player_id)
            assert other.stat_totals == pytest.approx(player.stat_totals)
            assert other.projection_source == player.projection_source
            assert other.espn_adp == player.espn_adp
            assert other.yahoo_adp == player.yahoo_adp
            assert other.ffc_adp == player.ffc_adp
            assert other.sleeper_rank == player.sleeper_rank
            assert other.adp_source_count == 3
            assert other.adp_stdev_is_estimated is True

        assert restored.rescore(FULL_PPR).rescored == len(RECEIVERS)
        for player in restored:
            assert player.projection == pytest.approx(
                round(core_stats.score(player.stat_totals, player.position, FULL_PPR), 1)
            )
    finally:
        dispose_engine()


def test_an_older_database_gains_the_provenance_columns(tmp_path) -> None:
    """The migration, run against a database genuinely missing the columns.

    Asserted rather than assumed because the failure mode is a hard crash on the
    user's existing database — the one place a schema mistake is unrecoverable.
    """
    import sqlalchemy

    from models.database import (
        _V2_PLAYER_COLUMNS, dispose_engine, get_engine, init_db, session_scope,
        set_setting,
    )
    from services.repository import load_player_pool, save_player_pool

    dispose_engine()
    db = str(tmp_path / "legacy.db")
    init_db(db)
    try:
        # Put a real pool in first, so the migration runs over rows rather than an
        # empty table — a rewrite-the-table migration would lose them.
        pool = _pool(STANDARD)
        with session_scope(db) as session:
            source_id = save_player_pool(session, pool)

        # Wind the schema back to v1: drop every column v2 added, and say so.
        engine = get_engine(db)
        with engine.begin() as connection:
            for column, _ddl in _V2_PLAYER_COLUMNS:
                connection.execute(
                    sqlalchemy.text(f"ALTER TABLE players DROP COLUMN {column}")
                )
        with session_scope(db) as session:
            set_setting(session, "schema_version", 1)

        present = {
            c["name"] for c in sqlalchemy.inspect(engine).get_columns("players")
        }
        assert not present & {c for c, _ in _V2_PLAYER_COLUMNS}

        init_db(db)  # the upgrade a user gets by opening the app

        present = {
            c["name"] for c in sqlalchemy.inspect(get_engine(db)).get_columns("players")
        }
        assert {c for c, _ in _V2_PLAYER_COLUMNS} <= present

        with session_scope(db) as session:
            restored = load_player_pool(session, source_id)
        assert restored is not None and len(restored) == len(pool)
        # The stats were never stored under v1, so they cannot come back. What must
        # not happen is a crash, or a pool that claims to be rescorable and is not.
        assert restored.rescore(FULL_PPR).rescored == 0

        init_db(db)  # idempotent: re-running must not fail on existing columns
    finally:
        dispose_engine()


def test_the_espn_board_carries_stat_totals_through_to_the_pool() -> None:
    """The fetch path, end to end, without the recorded payloads.

    A hand-made board rather than a fixture, because what is being checked is the
    column plumbing — ``espn_stat_totals`` → ``stat_totals`` → ``Player`` — and that
    breaks by a column being renamed on one side only.
    """
    import pandas as pd

    from services.importers import import_player_pool
    from services.providers.resolver import IMPORT_COLUMNS, board_to_import_frame

    assert IMPORT_COLUMNS.get("espn_stat_totals") == "stat_totals"

    stats = {"receptions": 88.0, "rec_yards": 1150.0, "rec_td": 9.0}
    board = pd.DataFrame([{
        "player_name": "Board Receiver",
        "position": "WR",
        "adp": 14.0,
        "espn_projection": core_stats.score(stats, Position.WR, STANDARD),
        "espn_stat_totals": core_stats.to_frame_value(stats),
    }])

    result = import_player_pool(board_to_import_frame(board), source="espn")
    assert result.pool is not None
    player = result.pool.require("Board Receiver")
    assert player.stat_totals == pytest.approx(stats)
    assert not player.projection_imputed
