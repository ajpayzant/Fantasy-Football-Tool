"""Uploading your own projections, and what has to be true for them to be usable.

A projection upload is easy to build badly in ways nobody notices. The three failure
modes these tests exist for, in rough order of how expensive they are:

1. **Silence.** A column header the app cannot place, a name that is not on the board,
   a sheet whose only recognised columns were targets and games played — each one
   produces a board that looks updated and is not. Every one of those has a test here
   asserting it is *reported*, because a user who is told can fix it and a user who is
   not will trust a number built from nothing.
2. **Divergence.** If uploaded projections were scored anywhere other than
   :mod:`core.stats`, the same player would be worth different points depending on
   which source he came from, for no reason a user could discover. The tests here score
   expectations with :func:`core.stats.score` directly, so an implementation that grew
   its own arithmetic fails.
3. **Erasure.** A sheet covering only receiving stats must not zero a running back's
   rushing projection, and a points total must not be silently reverted to the
   provider's number the next time scoring changes. Both are asserted below.

The distinction that runs through all of it: a **stat line** is a projection this app
can keep working with, and a **points total** is a number frozen at whatever rules
produced it. Both are accepted — plenty of sheets only export points — but the app has
to know which it has and say so.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core import stats as core_stats
from core.config import LeagueConfig, ScoringRules
from core.enums import Position, ProjectionMode, ScoringPreset
from models.player import Player, PlayerPool, PoolMetadata
from services.importers import (
    import_player_pool,
    import_projections,
    projection_template,
)
from services.normalize import canonical_column

STANDARD = ScoringRules.from_preset(ScoringPreset.STANDARD)
HALF_PPR = ScoringRules.from_preset(ScoringPreset.HALF_PPR)
FULL_PPR = ScoringRules.from_preset(ScoringPreset.FULL_PPR)

# A board wide enough for tiers, replacement level and outcome bands to mean something,
# and receiving-heavy so a change in the value of a reception is the loudest signal.
BOARD: tuple[tuple[str, str, float, float, float], ...] = (
    # name, position, receptions, rec_yards, rec_td
    ("Alpha Receiver", "WR", 110.0, 1480.0, 11.0),
    ("Bravo Receiver", "WR", 92.0, 1210.0, 8.0),
    ("Charlie Receiver", "WR", 74.0, 980.0, 6.0),
    ("Delta Receiver", "WR", 58.0, 720.0, 4.0),
    ("Echo Receiver", "WR", 41.0, 505.0, 2.0),
    ("Foxtrot Receiver", "WR", 30.0, 360.0, 1.0),
)


def _league(scoring: ScoringRules = HALF_PPR) -> LeagueConfig:
    return LeagueConfig(name="Test", season=2026, scoring=scoring)


def _receiver(
    name: str, receptions: float, yards: float, tds: float, *,
    adp: float, scoring: ScoringRules, position: Position = Position.WR,
) -> Player:
    stats = {
        "receptions": receptions, "rec_yards": yards, "rec_td": tds,
        "targets": receptions * 1.5, "games": 17.0,
    }
    return Player(
        player_id=f"{name.lower().replace(' ', '')}_{position}".lower(),
        name=name,
        position=position,
        projection=round(core_stats.score(stats, position, scoring) or 0.0, 1),
        overall_adp=adp,
        adp_stdev=adp * 0.2,
        stat_totals=stats,
        source="test fixture",
    )


def _pool(
    scoring: ScoringRules = HALF_PPR,
    *,
    league: LeagueConfig | None = None,
    extra: list[Player] | None = None,
) -> PlayerPool:
    players = [
        _receiver(
            name, rec, yards, tds,
            adp=float(index * 6), scoring=scoring, position=Position.coerce(pos),
        )
        for index, (name, pos, rec, yards, tds) in enumerate(BOARD, start=1)
    ]
    players.extend(extra or [])
    return PlayerPool(
        players,
        league=league if league is not None else _league(scoring),
        metadata=PoolMetadata(source="test fixture"),
    )


def _apply(frame: pd.DataFrame, pool: PlayerPool, **kwargs):
    scoring = kwargs.pop("scoring", pool.league.scoring)
    return import_projections(frame, pool=pool, scoring=scoring, **kwargs)


def _messages(result) -> str:
    return " ".join(issue.message for issue in result.report.issues)


# ─────────────────────────────────────────────────────────────────────────────
# Reading the file
# ─────────────────────────────────────────────────────────────────────────────
def test_stat_columns_are_read_under_the_spellings_real_sheets_use() -> None:
    """Nobody exports ``rec_yards``. The headers people really have have to work."""
    pool = _pool()
    frame = pd.DataFrame([
        {
            "Player": "Alpha Receiver", "Pos": "WR",
            "REC": 100, "Rec Yds": 1300, "Rec TD": 9,
            "Rush Att": 4, "Rush Yds": 30, "FUM": 1,
        },
    ])
    result = _apply(frame, pool)

    assert result.stat_columns == {
        "REC": "receptions", "Rec Yds": "rec_yards", "Rec TD": "rec_td",
        "Rush Att": "rush_attempts", "Rush Yds": "rush_yards", "FUM": "fumbles_lost",
    }
    assert not result.ignored_columns


def test_a_points_column_is_accepted_on_its_own() -> None:
    """The universal fallback: every projection export ever made has a points column."""
    pool = _pool()
    frame = pd.DataFrame([{"Player": "Alpha Receiver", "FPTS": 275.5}])
    result = _apply(frame, pool)

    assert result.report.ok
    assert result.outcome is not None
    assert result.outcome.from_points == 1
    assert pool.get("alphareceiver_wr").projection == pytest.approx(275.5)


def test_a_column_the_app_cannot_place_is_named_rather_than_dropped() -> None:
    """An ignored column is the usual reason an uploaded projection comes out low.

    Dropping it silently means the user compares their sheet to the board, sees a
    smaller number, and has no way to learn that their "YDS" column was never read.
    """
    pool = _pool()
    frame = pd.DataFrame([
        {"Player": "Alpha Receiver", "REC": 100, "Rec Yds": 1300, "YDS": 1300,
         "Auction Value": 42},
    ])
    result = _apply(frame, pool)

    assert set(result.ignored_columns) == {"YDS", "Auction Value"}
    message = _messages(result)
    assert "YDS" in message and "Auction Value" in message


def test_a_file_with_names_but_no_projection_is_refused() -> None:
    """A list of names is not a projection, and pretending otherwise changes nothing."""
    pool = _pool()
    before = {p.player_id: p.projection for p in pool}
    result = _apply(pd.DataFrame([{"Player": "Alpha Receiver", "Pos": "WR"}]), pool)

    assert not result.report.ok
    assert "points column" in _messages(result)
    assert {p.player_id: p.projection for p in pool} == before


def test_a_file_with_no_name_column_is_refused() -> None:
    pool = _pool()
    result = _apply(pd.DataFrame([{"Rec": 100, "Rec Yds": 1300}]), pool)

    assert not result.report.ok
    assert "player-name column" in _messages(result)


def test_an_upload_with_no_board_to_apply_it_to_is_refused() -> None:
    """Projections edit the loaded board; there is nothing to edit without one."""
    empty = PlayerPool([], league=_league())
    result = import_projections(
        pd.DataFrame([{"Player": "Alpha Receiver", "FPTS": 200}]),
        pool=empty, scoring=HALF_PPR,
    )
    assert not result.report.ok
    assert "Load a player pool first" in _messages(result)


# ─────────────────────────────────────────────────────────────────────────────
# Matching a spreadsheet name to a board
# ─────────────────────────────────────────────────────────────────────────────
def test_matching_survives_punctuation_and_generational_suffixes() -> None:
    """``A.J. Brown Jr.`` and ``AJ Brown`` are one player, and sheets disagree."""
    pool = _pool(extra=[
        _receiver("A.J. Brown Jr.", 90.0, 1200.0, 8.0, adp=5.0, scoring=HALF_PPR),
    ])
    result = _apply(
        pd.DataFrame([{"Player": "AJ Brown", "Rec": 100, "Rec Yds": 1400, "Rec TD": 10}]),
        pool,
    )

    assert result.outcome is not None and result.outcome.from_stats == 1
    assert not result.unmatched


def test_a_name_not_on_the_board_is_named_and_skipped() -> None:
    pool = _pool()
    result = _apply(
        pd.DataFrame([
            {"Player": "Alpha Receiver", "Rec": 100, "Rec Yds": 1300, "Rec TD": 9},
            {"Player": "Nobody At All", "Rec": 90, "Rec Yds": 1100, "Rec TD": 7},
        ]),
        pool,
    )

    assert result.unmatched == ["Nobody At All"]
    assert "Nobody At All" in _messages(result)
    assert result.rejected_rows == 1
    assert result.outcome is not None and result.outcome.applied == 1


def test_a_position_column_tells_two_players_with_one_name_apart() -> None:
    pool = _pool(extra=[
        _receiver("Same Name", 60.0, 700.0, 5.0, adp=40.0, scoring=HALF_PPR),
        _receiver("Same Name", 30.0, 300.0, 2.0, adp=90.0, scoring=HALF_PPR,
                  position=Position.TE),
    ])
    result = _apply(
        pd.DataFrame([
            {"Player": "Same Name", "Pos": "TE", "Rec": 80, "Rec Yds": 900, "Rec TD": 7},
        ]),
        pool,
    )

    assert result.outcome is not None and result.outcome.from_stats == 1
    assert pool.get("samename_te").projection == pytest.approx(
        round(core_stats.score(
            {"receptions": 80.0, "rec_yards": 900.0, "rec_td": 7.0},
            Position.TE, HALF_PPR,
        ), 1)
    )
    # The receiver of the same name was not touched.
    assert pool.get("samename_wr").stat_totals["receptions"] == 60.0


def test_one_name_matching_two_players_without_a_position_is_refused() -> None:
    """Guessing which one the user meant would be wrong half the time and invisible."""
    pool = _pool(extra=[
        _receiver("Same Name", 60.0, 700.0, 5.0, adp=40.0, scoring=HALF_PPR),
        _receiver("Same Name", 30.0, 300.0, 2.0, adp=90.0, scoring=HALF_PPR,
                  position=Position.TE),
    ])
    result = _apply(
        pd.DataFrame([{"Player": "Same Name", "Rec": 80, "Rec Yds": 900, "Rec TD": 7}]),
        pool,
    )

    assert result.ambiguous == ["Same Name"]
    assert "add a position column" in _messages(result).lower()
    assert pool.get("samename_te").stat_totals["receptions"] == 30.0
    assert pool.get("samename_wr").stat_totals["receptions"] == 60.0


def test_a_second_row_for_the_same_player_is_rejected_not_silently_applied() -> None:
    """Two rows disagreeing about one player is a file problem, not a merge problem."""
    pool = _pool()
    result = _apply(
        pd.DataFrame([
            {"Player": "Alpha Receiver", "Rec": 100, "Rec Yds": 1300, "Rec TD": 9},
            {"Player": "Alpha Receiver", "Rec": 40, "Rec Yds": 400, "Rec TD": 2},
        ]),
        pool,
    )

    assert result.rejected_rows == 1
    assert result.accepted_rows == 1
    assert pool.get("alphareceiver_wr").stat_totals["receptions"] == 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Scoring what was uploaded
# ─────────────────────────────────────────────────────────────────────────────
def test_uploaded_stats_are_scored_by_the_one_scorer_under_league_rules() -> None:
    """The whole reason ``core.stats`` exists: one implementation, no drift.

    An upload scored anywhere else would make the same player worth different points
    depending on where he came from.
    """
    pool = _pool(league=_league(FULL_PPR), scoring=FULL_PPR)
    uploaded = {"receptions": 100.0, "rec_yards": 1300.0, "rec_td": 9.0}
    result = _apply(
        pd.DataFrame([{
            "Player": "Alpha Receiver", "Rec": 100, "Rec Yds": 1300, "Rec TD": 9,
        }]),
        pool,
    )

    assert result.outcome is not None and result.outcome.from_stats == 1
    expected = core_stats.score(uploaded, Position.WR, FULL_PPR)
    assert expected is not None
    assert pool.get("alphareceiver_wr").projection == pytest.approx(
        round(expected, 1), abs=1e-9
    )


def test_uploaded_stats_survive_a_later_scoring_change() -> None:
    """The payoff of taking stats rather than points: they stay rescorable.

    Upload under standard scoring, then switch the league to full PPR. A projection
    built from a stat line has to move by exactly the receptions; one that had been
    stored as points could not move at all.
    """
    pool = _pool(league=_league(STANDARD), scoring=STANDARD)
    _apply(
        pd.DataFrame([{
            "Player": "Alpha Receiver", "Rec": 100, "Rec Yds": 1300, "Rec TD": 9,
        }]),
        pool,
    )
    before = pool.get("alphareceiver_wr").projection

    pool.rescore(FULL_PPR)
    after = pool.get("alphareceiver_wr").projection

    assert after == pytest.approx(before + 100.0, abs=0.05), (
        "a projection built from an uploaded stat line must rescore like any other"
    )


def test_a_points_total_is_taken_exactly_as_given() -> None:
    pool = _pool()
    _apply(pd.DataFrame([{"Player": "Bravo Receiver", "Projection": 301.4}]), pool)
    assert pool.get("bravoreceiver_wr").projection == pytest.approx(301.4)


def test_a_points_total_clears_the_stat_line_so_it_cannot_be_reverted() -> None:
    """The trap: keeping the provider's stats under a user's points total.

    A later scoring change rescores from stats, so the user's own number would be
    thrown away and replaced by the provider's — silently, and with nothing on the row
    to explain why the projection they uploaded is not the one on screen.
    """
    pool = _pool()
    _apply(pd.DataFrame([{"Player": "Bravo Receiver", "FPTS": 301.4}]), pool)
    player = pool.get("bravoreceiver_wr")
    assert player.stat_totals == {}

    pool.rescore(FULL_PPR)
    assert pool.get("bravoreceiver_wr").projection == pytest.approx(301.4)


def test_the_projection_source_says_which_of_the_two_it_was() -> None:
    """A user has to be able to tell a rescorable projection from a frozen one."""
    pool = _pool()
    _apply(
        pd.DataFrame([
            {"Player": "Alpha Receiver", "Rec": 100, "Rec Yds": 1300, "Rec TD": 9,
             "FPTS": None},
            {"Player": "Bravo Receiver", "Rec": None, "Rec Yds": None, "Rec TD": None,
             "FPTS": 250.0},
        ]),
        pool, source="my own sheet",
    )

    from_stats = pool.get("alphareceiver_wr").projection_source
    from_points = pool.get("bravoreceiver_wr").projection_source
    assert "my own sheet" in from_stats and "league's rules" in from_stats
    assert "my own sheet" in from_points and "cannot rescore" in from_points


def test_a_partial_stat_line_does_not_erase_the_rest_of_the_projection() -> None:
    """A receiving-only sheet must not zero a running back's rushing projection."""
    back = Player(
        player_id="hybrid_back_rb", name="Hybrid Back", position=Position.RB,
        overall_adp=3.0, adp_stdev=1.0,
        stat_totals={
            "rush_attempts": 260.0, "rush_yards": 1180.0, "rush_td": 9.0,
            "receptions": 40.0, "rec_yards": 300.0, "rec_td": 1.0,
        },
    )
    back.projection = round(
        core_stats.score(back.stat_totals, Position.RB, HALF_PPR) or 0.0, 1
    )
    pool = _pool(extra=[back])

    result = _apply(
        pd.DataFrame([{"Player": "Hybrid Back", "Rec": 60, "Rec Yds": 520, "Rec TD": 3}]),
        pool,
    )

    updated = pool.get("hybrid_back_rb")
    assert updated.stat_totals["rush_yards"] == 1180.0, "rushing was erased"
    assert updated.stat_totals["receptions"] == 60.0
    assert result.outcome is not None and result.outcome.partial_merge == 1
    assert "kept from the previous projection" in updated.projection_source


def test_stats_that_score_no_points_are_dropped_and_reported() -> None:
    """A sheet whose only readable columns were targets and games played.

    Left in place, an unscorable stat line would suppress the points column on the same
    row — ``apply_projections`` prefers a stat line — so the projection would silently
    not change at all.
    """
    pool = _pool()
    result = _apply(
        pd.DataFrame([{"Player": "Alpha Receiver", "TGT": 150, "G": 17, "FPTS": 280.0}]),
        pool,
    )

    assert result.thin and "Alpha Receiver" in result.thin[0]
    assert "score no points" in _messages(result)
    assert pool.get("alphareceiver_wr").projection == pytest.approx(280.0)


def test_a_row_with_nothing_scorable_and_no_points_is_rejected() -> None:
    pool = _pool()
    before = pool.get("alphareceiver_wr").projection
    result = _apply(
        pd.DataFrame([{"Player": "Alpha Receiver", "TGT": 150, "G": 17}]), pool
    )

    assert result.rejected_rows == 1
    assert result.accepted_rows == 0
    assert pool.get("alphareceiver_wr").projection == before


# ─────────────────────────────────────────────────────────────────────────────
# The three modes
# ─────────────────────────────────────────────────────────────────────────────
def test_replace_mode_lets_your_number_win() -> None:
    pool = _pool()
    _apply(pd.DataFrame([{"Player": "Alpha Receiver", "FPTS": 100.0}]), pool,
           mode=ProjectionMode.REPLACE)
    assert pool.get("alphareceiver_wr").projection == pytest.approx(100.0)


def test_blend_mode_averages_the_two_stat_lines() -> None:
    """Averaged stat by stat, not as points, so the stored line still explains itself."""
    pool = _pool()
    before = pool.get("alphareceiver_wr").stat_totals["receptions"]
    result = _apply(
        pd.DataFrame([{
            "Player": "Alpha Receiver", "Rec": 90, "Rec Yds": 1000, "Rec TD": 5,
        }]),
        pool, mode=ProjectionMode.BLEND,
    )

    updated = pool.get("alphareceiver_wr")
    assert updated.stat_totals["receptions"] == pytest.approx((before + 90.0) / 2)
    assert result.outcome is not None and result.outcome.blended == 1
    # And the points still equal the scorer's answer for the blended line, so nothing
    # on the row disagrees with anything else on it.
    assert updated.projection == pytest.approx(
        round(core_stats.score(updated.stat_totals, Position.WR, HALF_PPR), 1),
        abs=1e-9,
    )


def test_blend_mode_averages_a_points_total_with_the_board() -> None:
    pool = _pool()
    before = pool.get("alphareceiver_wr").projection
    _apply(pd.DataFrame([{"Player": "Alpha Receiver", "FPTS": 100.0}]), pool,
           mode=ProjectionMode.BLEND)
    assert pool.get("alphareceiver_wr").projection == pytest.approx(
        round((before + 100.0) / 2, 1)
    )


def test_fill_gaps_mode_leaves_a_real_projection_alone() -> None:
    pool = _pool()
    before = pool.get("alphareceiver_wr").projection
    result = _apply(pd.DataFrame([{"Player": "Alpha Receiver", "FPTS": 100.0}]), pool,
                    mode=ProjectionMode.FILL_GAPS)

    assert pool.get("alphareceiver_wr").projection == before
    assert result.outcome is not None and result.outcome.skipped_had_real == 1
    assert result.outcome.applied == 0


def test_fill_gaps_mode_replaces_a_projection_estimated_from_draft_position() -> None:
    """An estimate is a restatement of ADP, not a projection, so it is not a gap-filler.

    Treating it as "already there" would make fill-gaps mode a no-op on exactly the
    players it exists for: the late-round tail nobody published a projection for.
    """
    unknown = Player(
        player_id="unknown_deep_wr", name="Unknown Deep", position=Position.WR,
        overall_adp=140.0, adp_stdev=30.0,
    )
    pool = _pool(extra=[unknown])
    assert pool.get("unknown_deep_wr").projection_imputed is True

    result = _apply(
        pd.DataFrame([{
            "Player": "Unknown Deep", "Rec": 55, "Rec Yds": 640, "Rec TD": 4,
        }]),
        pool, mode=ProjectionMode.FILL_GAPS,
    )

    updated = pool.get("unknown_deep_wr")
    assert result.outcome is not None and result.outcome.from_stats == 1
    assert updated.projection_imputed is False
    assert updated.projection == pytest.approx(
        round(core_stats.score(
            {"receptions": 55.0, "rec_yards": 640.0, "rec_td": 4.0},
            Position.WR, HALF_PPR,
        ), 1),
        abs=1e-9,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Everything downstream of a projection
# ─────────────────────────────────────────────────────────────────────────────
def test_tiers_bands_and_value_over_replacement_follow_the_upload() -> None:
    """All three are read off the position's projection curve, so all three move.

    A board where a player's projection says one thing and his tier and VOR still
    reflect the old one is worse than one that was never updated: the row disagrees
    with itself and there is no way to tell which half is current.
    """
    pool = _pool()
    last = pool.get("foxtrotreceiver_wr")
    before_tier = last.tier
    before_vor = last.value_over_replacement
    before_ceiling = last.ceiling

    # Make the worst receiver on the board the best by a wide margin.
    _apply(
        pd.DataFrame([{
            "Player": "Foxtrot Receiver", "Rec": 140, "Rec Yds": 2000, "Rec TD": 18,
        }]),
        pool,
    )

    updated = pool.get("foxtrotreceiver_wr")
    assert updated.tier is not None and updated.tier < before_tier
    assert updated.value_over_replacement > before_vor
    assert updated.ceiling != before_ceiling


def test_a_points_only_upload_still_re_derives_the_board() -> None:
    """The path with no stat line anywhere, which an early return can skip.

    ``rescore`` returns without re-deriving when it had nothing to rescore, which is
    right on its own and wrong here: the projections moved anyway.
    """
    # Decaying rather than evenly spaced: tiers break where a gap is unusual for the
    # position, so a straight line puts every player in tier 1 and the test proves
    # nothing.
    plain = [
        Player(
            player_id=f"plain_{index}_wr", name=f"Plain {index}", position=Position.WR,
            projection=round(320.0 * 0.7 ** index, 1), overall_adp=float(index * 6 + 1),
            adp_stdev=2.0,
        )
        for index in range(1, 15)
    ]
    pool = PlayerPool(plain, league=_league())
    before_tier = pool.get("plain_14_wr").tier
    before_vor = pool.get("plain_14_wr").value_over_replacement
    assert before_tier is not None and before_tier > 1, "fixture must spread over tiers"

    result = _apply(pd.DataFrame([{"Player": "Plain 14", "FPTS": 400.0}]), pool)

    assert result.outcome is not None and result.outcome.rescore is not None
    assert result.outcome.rescore.changed == 0, "fixture must carry no stat lines"
    updated = pool.get("plain_14_wr")
    assert updated.tier is not None and updated.tier < before_tier
    assert updated.value_over_replacement > before_vor


def test_an_upload_does_not_disturb_adp_or_ranks() -> None:
    """Projections and draft position are separate claims, from separate sources."""
    pool = _pool()
    before = {
        p.player_id: (p.overall_adp, p.platform_adp, p.overall_rank, p.adp_stdev)
        for p in pool
    }
    _apply(
        pd.DataFrame([{
            "Player": "Alpha Receiver", "Rec": 140, "Rec Yds": 2000, "Rec TD": 18,
        }]),
        pool,
    )
    after = {
        p.player_id: (p.overall_adp, p.platform_adp, p.overall_rank, p.adp_stdev)
        for p in pool
    }
    assert after == before


# ─────────────────────────────────────────────────────────────────────────────
# The template, and the pool importer's own stat column
# ─────────────────────────────────────────────────────────────────────────────
def test_the_template_is_readable_by_the_importer_it_documents() -> None:
    """A template the importer rejects is worse than none: it teaches the wrong format."""
    template = projection_template()
    pool = PlayerPool(
        [
            Player(player_id="example_quarterback_qb", name="Example Quarterback",
                   position=Position.QB, overall_adp=20.0, adp_stdev=4.0),
            Player(player_id="example_running_back_rb", name="Example Running Back",
                   position=Position.RB, overall_adp=8.0, adp_stdev=2.0),
            Player(player_id="example_receiver_wr", name="Example Receiver",
                   position=Position.WR, overall_adp=14.0, adp_stdev=3.0),
        ],
        league=_league(),
    )
    result = import_projections(template, pool=pool, scoring=HALF_PPR)

    assert result.report.ok, _messages(result)
    assert not result.ignored_columns
    assert not result.unmatched
    assert result.outcome is not None
    assert result.outcome.from_stats == 2
    assert result.outcome.from_points == 1


def test_a_pool_file_with_stat_columns_is_scored_not_estimated_from_adp() -> None:
    """The pool importer documents a ``stat_totals`` column, so it has to use it.

    Before this, a file supplying stat lines and no ``projection`` column produced a
    board where every projection was an estimate read off draft position while the real
    stats sat in the row unused — the exact opposite of what supplying them asks for.
    """
    frame = pd.DataFrame([
        {
            "player_name": name, "position": pos, "overall_adp": float(index * 6),
            "stat_totals": core_stats.to_frame_value({
                "receptions": rec, "rec_yards": yards, "rec_td": tds,
            }),
        }
        for index, (name, pos, rec, yards, tds) in enumerate(BOARD, start=1)
    ])
    result = import_player_pool(frame, league=_league(HALF_PPR), source="mine.csv")

    assert result.pool is not None
    for player in result.pool:
        assert player.projection_imputed is False, f"{player.name} was estimated"
        expected = core_stats.score(player.stat_totals, player.position, HALF_PPR)
        assert player.projection == pytest.approx(round(expected, 1), abs=1e-9)
    assert "under your league's scoring rules" in _messages(result)


def test_a_projection_computed_from_stats_stops_claiming_to_be_an_estimate() -> None:
    """The provenance line has to match the number it describes.

    A pool whose projections were imputed and then scored from stats kept the "estimated
    from draft position" text, which is a plain false statement about the row it is on.
    """
    from models.player import STAT_LINE_PROJECTION_SOURCE

    frame = pd.DataFrame([{
        "player_name": "Alpha Receiver", "position": "WR", "overall_adp": 3.0,
        "stat_totals": core_stats.to_frame_value(
            {"receptions": 100.0, "rec_yards": 1300.0, "rec_td": 9.0}
        ),
    }])
    result = import_player_pool(frame, league=_league(HALF_PPR), source="mine.csv")

    assert result.pool is not None
    player = result.pool.players[0]
    assert player.projection_source == STAT_LINE_PROJECTION_SOURCE
    assert "Estimated from draft position" not in player.projection_source


def test_no_header_is_both_a_stat_and_a_column_the_importer_reads() -> None:
    """The reason header resolution can be ordered at all.

    :func:`services.importers._projection_headers` resolves stat names first and
    identity columns second, on the grounds that the two vocabularies do not overlap.
    That is true today and nothing enforces it: adding a ``COLUMN_ALIASES`` entry for a
    header that is also a stat name — ``yds``, say, or ``td`` — would quietly route a
    stat column into the identity map, where it would be ignored and the projection
    built from it would come out low. Ordering cannot be tested through behaviour while
    the vocabularies are disjoint, so the disjointness itself is what is pinned here.
    """
    from core.stats import _ALIASES
    from services.importers import _PROJECTION_IDENTITY_COLUMNS

    conflicts = {
        spelling: canonical_column(spelling)
        for spelling in set(_ALIASES) | core_stats.STAT_FIELD_SET
        if canonical_column(spelling) in _PROJECTION_IDENTITY_COLUMNS
    }
    assert not conflicts, (
        "these headers are both a stat and an identity column, so header resolution "
        f"order is now load-bearing and needs a real test: {conflicts}"
    )
