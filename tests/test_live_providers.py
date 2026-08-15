"""Tests for the live data pipeline, run offline against recorded payloads.

Two distinct things are asserted here, and keeping them separate matters:

1. **Against real recorded payloads** (``tests/fixtures/live_payloads/``, written by
   ``scripts/record_live_fixtures.py``) each provider shapes what the endpoint
   actually returns, and the resolver joins the four sources into one board. These
   are the tests that would catch a source changing its field names — the failure
   mode an undocumented third-party API has most often.

2. **Against a simulated outage** every provider degrades to a validation error
   rather than an exception, and the board is still built from whatever answered.
   This is the "do not make the application dependent on a fragile external
   endpoint" constraint, asserted rather than asserted-to.

No test in this file touches the network. The recorded payloads are fed in through
the providers' own on-disk cache — the same path a real cache hit takes — so the
providers are exercised whole rather than having their parsing poked at directly.
Network access is additionally blocked (see :func:`_no_network`), so a provider that
grew a new un-cached request would fail here rather than quietly reaching out.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from core.enums import Position, ScoringPreset
from services.providers import base as provider_base
from services.providers import (
    ESPNProvider,
    FFCalculatorProvider,
    SleeperProvider,
    YahooProvider,
)
from services.providers.resolver import board_to_import_frame, resolve_board

PAYLOAD_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "live_payloads")
MANIFEST_PATH = os.path.join(PAYLOAD_DIR, "manifest.json")

# Every test in this module needs the recorded payloads. Skipping rather than
# failing when they are absent is deliberate: a fresh clone that has not run the
# recorder yet has a legitimately incomplete fixture set, and a hard failure there
# would say "the code is broken" when the truth is "the fixtures are not recorded".
requires_payloads = pytest.mark.skipif(
    not os.path.exists(MANIFEST_PATH),
    reason="recorded payloads missing — run scripts/record_live_fixtures.py",
)


def _manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        pytest.skip("recorded payloads missing")
    return _manifest()


@pytest.fixture
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any real HTTP request a test failure.

    Without this the tests would still pass by silently fetching live, which would
    make them slow, non-deterministic, and dependent on four third parties being
    up. The point of the recorded payloads is that none of that is true.
    """
    import urllib.request

    def _blocked(*args, **kwargs):
        raise AssertionError(
            "a provider tried to reach the network; these tests run offline against "
            "recorded payloads"
        )

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch, _no_network: None) -> dict:
    """Point the providers' cache at the recorded payloads, read-only.

    ``cache_directory`` is redirected rather than the fetch functions, so every
    provider runs its real code path: cache lookup, JSON decode, shaping, and
    report construction. ``write_cache`` is stubbed out so a test can never
    overwrite a recorded fixture with whatever it happened to be holding.
    """
    if not os.path.exists(MANIFEST_PATH):
        pytest.skip("recorded payloads missing")
    monkeypatch.setattr(provider_base, "cache_directory", lambda: PAYLOAD_DIR)
    monkeypatch.setattr(provider_base, "write_cache", lambda *a, **k: None)
    # The payloads were recorded once and are older than any provider's TTL, so
    # every fetch would be a miss and fall through to the network. -1 disables the
    # expiry check, which is the same switch the stale-cache fallback uses.
    monkeypatch.setattr(provider_base, "DEFAULT_CACHE_TTL_SECONDS", -1)
    return _manifest()


def _fetch_all(manifest: dict) -> dict:
    """Every provider's result for the recorded season and format."""
    season = int(manifest["season"])
    scoring = ScoringPreset.coerce(manifest["scoring"], ScoringPreset.HALF_PPR)
    return {
        "sleeper": SleeperProvider().fetch(ttl_seconds=-1),
        "ffc": FFCalculatorProvider().fetch(
            scoring=scoring, team_count=int(manifest["team_count"]),
            season=season, ttl_seconds=-1,
        ),
        "espn": ESPNProvider().fetch(season=season, scoring=scoring, ttl_seconds=-1),
        "yahoo": YahooProvider().fetch(
            player_limit=int(manifest["yahoo_pages"]) * 25, ttl_seconds=-1
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Each provider shapes what its endpoint really returned
# ─────────────────────────────────────────────────────────────────────────────
@requires_payloads
def test_sleeper_shapes_players_and_carries_the_crosswalk(recorded) -> None:
    """Sleeper is the identity spine: without its id columns nothing else joins."""
    result = SleeperProvider().fetch(ttl_seconds=-1)
    assert result.ok, result.report.summary()
    assert {"player_name", "position", "nfl_team", "espn_id", "yahoo_id"} <= set(
        result.frame.columns
    )
    assert len(result.frame) > 200
    # The crosswalk has to actually be populated, not merely present. Measured on a
    # full payload it covers the large majority of players.
    assert int(result.frame["espn_id"].notna().sum()) > 100
    assert int(result.frame["yahoo_id"].notna().sum()) > 100


@requires_payloads
def test_sleeper_includes_team_defences(recorded) -> None:
    """DSTs are the one join with no shared identifier, so they must be present."""
    result = SleeperProvider().fetch(ttl_seconds=-1)
    defences = result.frame[result.frame["position"] == "DST"]
    assert len(defences) >= 20, "team defences are missing from the Sleeper board"
    # And they are precisely the rows the crosswalk cannot help with — the reason
    # the resolver keys them on team code instead.
    assert defences["espn_id"].isna().all()


@requires_payloads
def test_ffc_publishes_adp_with_a_real_spread(recorded, manifest) -> None:
    """FFC is the only source with a distribution, which the simulator needs."""
    result = FFCalculatorProvider().fetch(
        scoring=ScoringPreset.coerce(manifest["scoring"]),
        team_count=int(manifest["team_count"]),
        season=int(manifest["season"]),
        ttl_seconds=-1,
    )
    assert result.ok, result.report.summary()
    assert {"ffc_adp", "ffc_adp_stdev", "ffc_min_pick", "ffc_max_pick"} <= set(
        result.frame.columns
    )
    assert result.frame["ffc_adp"].notna().all()
    assert result.frame["ffc_adp_stdev"].notna().sum() > len(result.frame) * 0.9
    # ADP must be ordered sensibly and bounded by the draft it came from.
    assert result.frame["ffc_adp"].min() >= 1.0


@requires_payloads
def test_espn_shapes_ranks_and_flags_its_adp_population(recorded, manifest) -> None:
    """ESPN's ADP is its own leagues' average, not a mock consensus — and says so."""
    result = ESPNProvider().fetch(
        season=int(manifest["season"]),
        scoring=ScoringPreset.coerce(manifest["scoring"]),
        ttl_seconds=-1,
    )
    assert result.ok, result.report.summary()
    assert {"espn_id", "espn_rank", "espn_adp"} <= set(result.frame.columns)
    codes = {issue.code for issue in result.report.issues}
    assert "espn_adp_population" in codes, (
        "ESPN's ADP must be labelled as a platform-league average"
    )


@requires_payloads
def test_espn_stat_map_identities_still_hold(recorded, manifest) -> None:
    """The stat ids are derived from the payload's own arithmetic — re-check it.

    ESPN's fantasy API is undocumented, and the stat ids are numbers with no labels
    attached. If ESPN renumbers them, every projection in the app silently becomes
    wrong. These three identities are the evidence the map rests on, so a change
    surfaces here, named, instead of as quietly bad projections.
    """
    import gzip
    import json as _json

    from services.providers import espn_stats

    season = int(manifest["season"])
    rank_type = "PPR" if "ppr" in str(manifest["scoring"]).lower() else "STANDARD"
    path = os.path.join(PAYLOAD_DIR, f"espn_players_{season}_{rank_type}.json.gz")
    if not os.path.exists(path):
        pytest.skip(f"no recorded ESPN payload at {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        records = _json.load(handle)

    problems = espn_stats.verify_stat_map(records, season)
    assert not problems, "\n".join(problems)


@requires_payloads
def test_espn_projections_are_scored_under_the_league_rules_not_espns(
    recorded, manifest
) -> None:
    """The projection must respond to *our* scoring, and be sanely scaled.

    ESPN's own ``appliedTotal`` is unusable (it scores field-goal yardage as points,
    giving a kicker 5,237), so the provider re-scores the raw projected stat line.
    Two things must be true for that to be worth doing: the numbers have to land in
    the range real fantasy seasons occupy, and they have to *move* when the scoring
    rules move. A projection that ignores the rules would be ESPN's, not the league's.
    """
    from core.config import ScoringRules
    from core.enums import Position

    season = int(manifest["season"])
    preset = ScoringPreset.coerce(manifest["scoring"])

    def _fetch(rules: ScoringRules):
        return ESPNProvider().fetch(
            season=season, scoring=preset, scoring_rules=rules, ttl_seconds=-1
        ).frame.set_index("player_name")

    standard = _fetch(ScoringRules.from_preset(ScoringPreset.STANDARD))
    full_ppr = _fetch(ScoringRules.from_preset(ScoringPreset.FULL_PPR))

    assert "espn_projection" in standard.columns
    covered = standard["espn_projection"].notna()
    assert covered.sum() > len(standard) * 0.5, (
        f"only {int(covered.sum())} of {len(standard)} ESPN players carry a projection"
    )

    # A full-PPR point per catch has to raise receivers and leave quarterbacks alone,
    # which is the sharpest available check that the stat line is being read correctly
    # (receptions is stat 53; confusing it with targets, 58, would inflate both).
    receivers = standard[standard["position"].isin(["WR", "TE", "RB"])].index
    lifted = (
        full_ppr.loc[receivers, "espn_projection"]
        - standard.loc[receivers, "espn_projection"]
    ).dropna()
    assert (lifted >= 0).all(), "full PPR can never lower a receiver"
    # Not `all > 0`: a projection with no receptions at all is legitimately unmoved.
    assert (lifted > 0).mean() > 0.95, (
        "full PPR barely moved the receivers — receptions (stat 53) may be being read "
        "as something else"
    )
    quarterbacks = standard[standard["position"] == "QB"].index
    unchanged = (
        full_ppr.loc[quarterbacks, "espn_projection"]
        - standard.loc[quarterbacks, "espn_projection"]
    ).dropna()
    assert unchanged.abs().max() < 0.05, "PPR must not move a quarterback"

    # Scale check per position. Wide bounds on purpose: this catches a stat id that has
    # moved or a unit that is off by an order of magnitude, not a projection we
    # disagree with.
    plausible: dict[str, tuple[float, float]] = {
        "QB": (150.0, 500.0), "RB": (150.0, 450.0), "WR": (130.0, 420.0),
        "TE": (90.0, 320.0), "K": (90.0, 220.0), "DST": (60.0, 220.0),
    }
    for position, (low, high) in plausible.items():
        group = full_ppr[full_ppr["position"] == position]["espn_projection"].dropna()
        if group.empty:
            continue
        best = float(group.max())
        assert low <= best <= high, (
            f"the best projected {position} scores {best:.0f}, outside the plausible "
            f"{low:.0f}-{high:.0f} — a stat id has probably moved"
        )
        assert Position.coerce(position, None) is not None


@requires_payloads
def test_defence_projections_respond_to_the_points_allowed_tiers(
    recorded, manifest
) -> None:
    """Points allowed is the biggest term in real defence scoring, so it must bite.

    A defence scored on sacks and turnovers alone projects roughly 25% low, which is
    why the eight tiers exist as explicit fields rather than one shutout bonus. This
    asserts the whole chain works: a league that scores points allowed differently
    gets different defences, and nothing else changes. It also pins the tiers as the
    dominant term, since a map that read the wrong stat ids would still move the
    number a little.
    """
    from dataclasses import fields, replace

    from core.config import ScoringRules

    season = int(manifest["season"])
    preset = ScoringPreset.coerce(manifest["scoring"])
    base_rules = ScoringRules.from_preset(preset)
    tier_fields = [
        f.name for f in fields(ScoringRules)
        if f.name.startswith("dst_points_allowed_")
    ]
    assert len(tier_fields) == 8, "expected eight points-allowed bands"

    def _defences(rules: ScoringRules):
        frame = ESPNProvider().fetch(
            season=season, scoring=preset, scoring_rules=rules, ttl_seconds=-1
        ).frame
        return (
            frame[frame["position"] == "DST"]
            .set_index("player_name")["espn_projection"]
            .dropna()
        )

    base = _defences(base_rules)
    if base.empty:
        pytest.skip("no projected defences in this fixture set")

    # Double every tier. Points allowed is per game, so a full season of games in a
    # band should roughly double that component — a change no other term can produce.
    doubled = _defences(
        replace(base_rules, **{
            name: getattr(base_rules, name) * 2.0 for name in tier_fields
        })
    )
    shared = base.index.intersection(doubled.index)
    assert len(shared) >= 10, "too few defences to compare"
    moved = (doubled.loc[shared] - base.loc[shared]).abs()
    assert (moved > 1.0).mean() > 0.9, (
        "doubling the points-allowed tiers barely moved the defences — the buckets "
        "(stat ids 129-136) are probably not being read"
    )

    # And they must be reachable the way the UI reaches them: an override on top of a
    # named preset, which is exactly what the Setup page's advanced editor builds.
    custom = ScoringRules.from_preset(preset, dst_points_allowed_0=25.0)
    assert custom.dst_points_allowed_0 == 25.0
    assert custom.dst_points_allowed_7_13 == base_rules.dst_points_allowed_7_13, (
        "an override must not disturb the other tiers"
    )
    shutout_heavy = _defences(custom)
    assert (shutout_heavy.loc[shared] >= base.loc[shared] - 0.05).all(), (
        "paying more for a shutout can never lower a defence"
    )

    # Kickers share no stat ids with defences; if they move, the buckets are wrong.
    def _kickers(rules: ScoringRules):
        frame = ESPNProvider().fetch(
            season=season, scoring=preset, scoring_rules=rules, ttl_seconds=-1
        ).frame
        return (
            frame[frame["position"] == "K"]
            .set_index("player_name")["espn_projection"]
            .dropna()
        )

    kick_base, kick_custom = _kickers(base_rules), _kickers(custom)
    kick_shared = kick_base.index.intersection(kick_custom.index)
    if len(kick_shared):
        assert (kick_custom.loc[kick_shared] - kick_base.loc[kick_shared]).abs().max() < 0.05, (
            "a defence scoring change moved the kickers — a stat id is shared by mistake"
        )


@requires_payloads
def test_yahoo_returns_only_players_with_draft_data(recorded, manifest) -> None:
    result = YahooProvider().fetch(
        player_limit=int(manifest["yahoo_pages"]) * 25, ttl_seconds=-1
    )
    assert result.ok, result.report.summary()
    assert {"yahoo_id", "yahoo_adp"} <= set(result.frame.columns)
    assert result.frame["yahoo_adp"].notna().all(), (
        "a row with no Yahoo ADP carries no signal and would only add join noise"
    )


# ─────────────────────────────────────────────────────────────────────────────
# The resolver joins them into one board
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def board(recorded, manifest):
    """One resolved board built from all four recorded sources."""
    results = _fetch_all(manifest)
    return results, resolve_board(
        sleeper=results["sleeper"], ffc=results["ffc"],
        espn=results["espn"], yahoo=results["yahoo"],
        season=int(manifest["season"]), team_count=int(manifest["team_count"]),
        scoring_format=str(manifest["scoring"]),
    )


@requires_payloads
def test_board_has_one_row_per_player(board) -> None:
    """The join must be able only to fill cells, never to duplicate a player.

    This is the invariant the resolver's map-based merge exists to guarantee: a
    ``DataFrame.merge`` against a source holding two rows for one id would put the
    same player on the board twice, and a duplicated player would be recommendable
    twice in the same draft.
    """
    _, resolved = board
    assert resolved.ok
    assert resolved.frame["join_key"].is_unique, "the join duplicated a player"
    assert not resolved.frame["player_name"].duplicated().any()


@requires_payloads
def test_every_source_contributes_to_the_board(board) -> None:
    _, resolved = board
    assert sorted(resolved.successful_sources()) == ["espn", "ffc", "sleeper", "yahoo"]
    for column in ("ffc_adp", "espn_adp", "yahoo_adp"):
        assert int(resolved.frame[column].notna().sum()) > 50, (
            f"{column} contributed almost nothing — the join for that source broke"
        )


@requires_payloads
def test_defences_join_across_four_naming_conventions(board) -> None:
    """The measured reason DSTs key on team code rather than name.

    Sleeper says "Denver Broncos", FFC "Denver Defense", ESPN "Broncos D/ST" and
    Yahoo "Broncos". Keyed on name these are four different players; keyed on team
    they are one. If this regresses, every defence silently loses its ADP.
    """
    results, resolved = board
    defences = resolved.frame[resolved.frame["position"] == "DST"]
    assert len(defences) >= 20

    # The bar is *every defence FFC publishes*, not every defence on the board.
    # FFC only ranks the ~16 defences worth drafting, so requiring all 23 board
    # defences to have FFC ADP would be asserting data FFC does not have.
    ffc_frame = results["ffc"].frame
    ffc_defences = int((ffc_frame["position"] == "DST").sum())
    assert ffc_defences > 0, "FFC published no defences, so this proves nothing"
    joined = int(defences["ffc_adp"].notna().sum())
    assert joined == ffc_defences, (
        f"{joined} of {ffc_defences} FFC defences joined onto the board — the "
        "team-code join for DSTs has regressed"
    )
    # And a defence FFC ranks should also have picked up the other sources.
    multi = defences[defences["adp_source_count"] >= 2]
    assert len(multi) >= 10, "defences are not joining across sources"


@requires_payloads
def test_consensus_is_not_diluted_by_a_missing_source(board) -> None:
    """A player only one source knows gets that source's number, undiluted.

    Weights are renormalised by the weight actually present. Without that, a player
    with only FFC's ADP would come out at half of it and be ranked a round early.
    """
    _, resolved = board
    single = resolved.frame[
        (resolved.frame["adp_source_count"] == 1) & resolved.frame["ffc_adp"].notna()
    ]
    if single.empty:
        pytest.skip("no single-source players in this fixture set")
    assert (single["overall_adp"] - single["ffc_adp"]).abs().max() < 0.01


@requires_payloads
def test_rank_and_adp_agree_by_construction(board) -> None:
    """``overall_rank`` is derived from consensus ADP, so they cannot disagree.

    Two orderings of the same board is a bug the user sees as a player being ranked
    above another while having a later ADP.
    """
    _, resolved = board
    ordered = resolved.frame.sort_values("overall_rank")
    adp = ordered["overall_adp"].to_numpy()
    assert (adp[1:] >= adp[:-1] - 1e-9).all(), "rank order contradicts ADP order"


@requires_payloads
def test_estimated_spread_is_flagged_as_estimated(board) -> None:
    """Imputed spread must be labelled, not passed off as measured."""
    _, resolved = board
    assert "adp_stdev_is_estimated" in resolved.frame.columns
    estimated = resolved.frame[resolved.frame["adp_stdev_is_estimated"]]
    measured = resolved.frame[~resolved.frame["adp_stdev_is_estimated"]]
    assert len(measured) > 100, "FFC's published spread is not being used"
    # Every estimated row must still have a usable number — the estimate exists so
    # the simulator always has a distribution to draw from.
    assert estimated["adp_stdev"].notna().all()
    assert (estimated["adp_stdev"] > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# The board becomes a real player pool
# ─────────────────────────────────────────────────────────────────────────────
@requires_payloads
def test_board_imports_into_a_pool_that_is_not_flagged_as_sample(recorded, manifest) -> None:
    """End to end, offline: four payloads in, one real player pool out."""
    from services.importers import import_player_pool

    results = _fetch_all(manifest)
    resolved = resolve_board(
        sleeper=results["sleeper"], ffc=results["ffc"],
        espn=results["espn"], yahoo=results["yahoo"],
        season=int(manifest["season"]), team_count=int(manifest["team_count"]),
    )
    imported = import_player_pool(
        board_to_import_frame(resolved.frame),
        source="live: recorded fixtures",
        season=int(manifest["season"]),
        is_sample_data=False,
    )
    assert imported.report.ok, imported.report.summary()
    pool = imported.pool
    assert pool is not None and len(pool) > 200
    assert pool.metadata.is_sample_data is False
    assert "SAMPLE" not in pool.metadata.describe().upper()
    # Real names, not generated ones. The generator's players were "WR 2025-14"
    # style; a real board has no digits in a player's name.
    names = [player.name for player in pool.players[:50]]
    assert not any(any(ch.isdigit() for ch in name) for name in names), names


@requires_payloads
def test_import_projection_does_not_collide_on_aliased_columns(recorded, manifest) -> None:
    """``espn_adp`` and ``yahoo_adp`` both alias to ``platform_adp`` in the importer.

    Passing the board through as-is silently produced a ``platform_adp_2`` column and
    lost one source, which is why :data:`resolver.IMPORT_COLUMNS` is an explicit
    projection. This asserts the projection, not the alias table.
    """
    results = _fetch_all(manifest)
    resolved = resolve_board(
        sleeper=results["sleeper"], ffc=results["ffc"],
        espn=results["espn"], yahoo=results["yahoo"],
        season=int(manifest["season"]),
    )
    frame = board_to_import_frame(resolved.frame)
    assert not frame.columns.duplicated().any()
    assert not any(str(c).endswith("_2") for c in frame.columns), list(frame.columns)
    assert "adp" in frame.columns and frame["adp"].notna().any()

    # Each platform's own number must survive with its own identity. This is the
    # regression the collision used to cause: Yahoo's ADP landing in a column nothing
    # reads, so the Player Pool could only ever show ESPN.
    for column in ("ffc_adp", "espn_adp", "espn_rank", "yahoo_adp"):
        assert column in frame.columns, f"{column} was dropped on the way in"
        assert frame[column].notna().any(), f"{column} arrived empty"
    espn = frame["espn_adp"].dropna()
    yahoo = frame["yahoo_adp"].dropna()
    assert not espn.equals(yahoo), "ESPN and Yahoo ADP must not be the same column"


def _live_pool(manifest):
    """The board every source contributed to, imported as the app imports it."""
    from services.importers import import_player_pool
    from services.live import quick_league

    results = _fetch_all(manifest)
    resolved = resolve_board(
        sleeper=results["sleeper"], ffc=results["ffc"],
        espn=results["espn"], yahoo=results["yahoo"],
        season=int(manifest["season"]), team_count=int(manifest["team_count"]),
    )
    league = quick_league(
        team_count=int(manifest["team_count"]), scoring=str(manifest["scoring"]),
        season=int(manifest["season"]),
    )
    pool = import_player_pool(
        board_to_import_frame(resolved.frame),
        league=league.config,
        source="live: recorded fixtures",
    ).pool
    assert pool is not None and len(pool) > 200
    return pool


@requires_payloads
def test_pool_derives_bands_that_bracket_the_projection(recorded, manifest) -> None:
    """Ceiling, floor and risk must be present, ordered, and differentiated.

    All three used to be blank for every player on a live board. The band is derived
    from draft-pick disagreement mapped onto the position's projection curve, which
    only means anything if three things hold: the band actually brackets the
    projection (it is anchored on the player's own value), it is *not* constant (a
    contested player has a wider one than a consensus player), and the paperwork
    saying it was derived rather than supplied is filled in.
    """
    pool = _live_pool(manifest)

    for player in pool:
        assert player.ceiling is not None, player.name
        assert player.floor is not None, player.name
        assert player.risk_score is not None, player.name
        assert player.value_over_replacement is not None, player.name
        # Anchored on the player's own projection, so the band cannot straddle
        # somewhere else on the curve. This is exactly what an ADP-ordered curve got
        # wrong: 75 of 245 players came out with a floor above their projection.
        assert player.floor <= player.projection + 0.05 <= player.ceiling + 0.1, (
            f"{player.name}: floor {player.floor} / proj {player.projection} / "
            f"ceiling {player.ceiling} are out of order"
        )
        assert 0.0 <= player.risk_score <= 1.0
        assert player.outcome_band_source, player.name

    widths = [
        (p.ceiling - p.floor) / max(1.0, p.projection or 1.0) for p in pool
    ]
    assert max(widths) > min(widths) * 3, (
        "every player got the same band width, so the disagreement signal is not "
        "reaching the band"
    )

    # The provenance the Player Pool page reads to explain itself.
    derived = pool.metadata.imputed_fields
    for field in ("ceiling", "floor", "risk_score"):
        assert derived.get(field), f"{field} was derived but not recorded as derived"

    real = [p for p in pool if "ESPN projected stats" in p.projection_source]
    assert len(real) > 150, f"only {len(real)} players carry a real projection"
    assert all(p.projection_detail for p in real), (
        "a real projection must carry the stat line it was computed from"
    )
    estimated = [p for p in pool if "Estimated from draft position" in p.projection_source]
    assert all(not p.projection_detail for p in estimated)


@requires_payloads
def test_an_estimated_projection_never_outranks_a_real_one_above_it(
    recorded, manifest
) -> None:
    """Half a live board arrives unprojected, and where the estimates land is the board.

    ESPN projects a stat line for the players it thinks matter and nothing for the rest —
    520 of 1003 players on a real import. Those 520 are estimated from where the board
    ranks them, so the one invariant that has to hold is *ordering*: a player the board
    ranks 134th cannot come out projected above a player it ranks 4th who carries a real
    number. It used to, by a mile. Every player was placed on one pool-wide curve by
    overall rank, spanning the best projection in the pool to the worst, and that curve
    decays so slowly across a thousand players that the 134th came out at 77% of the top
    of it — 286 points, second among all WRs, ahead of the real 277 of the WR ranked 4th.
    """
    pool = _live_pool(manifest)
    estimated = [p for p in pool if p.projection_imputed]
    assert len(estimated) > 20, f"only {len(estimated)} estimates — board too small"

    for position in {p.position for p in estimated}:
        group = sorted(
            (p for p in pool if p.position is position),
            key=lambda p: (p.overall_rank if p.overall_rank is not None else 1e9, p.name),
        )
        best_real_above: float | None = None
        for player in group:
            if not player.projection_imputed:
                real = float(player.projection)
                best_real_above = (
                    real if best_real_above is None else max(best_real_above, real)
                )
                continue
            assert best_real_above is None or player.projection <= best_real_above, (
                f"{player.name} ({position}, board rank {player.overall_rank}) is "
                f"estimated at {player.projection}, above a projected player ranked "
                f"ahead of him"
            )


@requires_payloads
def test_no_estimated_projection_reaches_the_top_of_a_position(
    recorded, manifest
) -> None:
    """An estimate is a restatement of draft position. It must never lead a position.

    Estimates are read off the curve of the *real* projections, and an earlier version
    read them off a curve that had not been forced to decrease, so a WR the board ranked
    134th came out with a 286-point estimate — ahead of every genuinely projected
    receiver. The pick model reads projection as a strength signal, so that one number
    priced a replacement-level receiver as a second-round pick.
    """
    pool = _live_pool(manifest)

    for position in (Position.WR, Position.RB):
        group = sorted(
            (p for p in pool if p.position is position),
            key=lambda p: -(p.projection or 0.0),
        )
        assert len(group) >= 50, f"{position}: only {len(group)} players"
        top = group[:12]
        assert not any(p.projection_imputed for p in top), (
            f"{position}: "
            + ", ".join(
                f"{p.name} (board rank {p.overall_rank}, estimated {p.projection})"
                for p in top if p.projection_imputed
            )
            + " leads the position on an estimated projection"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Degradation: the app survives every source failing
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def all_sources_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a total outage at the transport layer.

    Patched at ``fetch_bytes`` rather than at the socket so each provider's own
    failure handling runs — including the stale-cache fallback, which is skipped
    here by pointing the cache at an empty directory.
    """
    from services.providers.base import FetchOutcome

    def _down(url: str, **kwargs) -> FetchOutcome:
        return FetchOutcome(None, url, error="network error: simulated outage")

    monkeypatch.setattr(provider_base, "cache_directory", lambda: _empty_dir())
    for module_name in ("sleeper", "ffcalculator", "espn", "yahoo", "leagues"):
        module = __import__(
            f"services.providers.{module_name}", fromlist=["fetch_bytes"]
        )
        if hasattr(module, "fetch_bytes"):
            monkeypatch.setattr(module, "fetch_bytes", _down)
    monkeypatch.setattr(provider_base, "fetch_bytes", _down)


def _empty_dir() -> str:
    """A real, empty directory, so the stale-cache fallback finds nothing."""
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "fmd_empty_cache")
    os.makedirs(path, exist_ok=True)
    return path


def test_no_provider_raises_when_every_source_is_down(all_sources_down) -> None:
    """The guarantee the whole live layer rests on: failures are values, not raises."""
    results = {
        "sleeper": SleeperProvider().fetch(),
        "ffc": FFCalculatorProvider().fetch(season=2026),
        "espn": ESPNProvider().fetch(season=2026),
        "yahoo": YahooProvider().fetch(player_limit=50),
    }
    for name, result in results.items():
        assert not result.ok, f"{name} claimed success during a simulated outage"
        assert result.report.errors, f"{name} failed without saying why"
        message = " ".join(issue.message for issue in result.report.errors)
        assert "unavailable" in message.lower() or "no draft data" in message.lower()


def test_live_board_reports_a_total_outage_without_raising(all_sources_down) -> None:
    """``build_live_board`` is what the UI calls, so it must not raise either."""
    from services.live import build_live_board

    result = build_live_board(season=2026, use_espn=True, use_yahoo=True)
    assert not result.ok
    assert result.pool is None
    codes = {issue.code for issue in result.report.errors}
    assert "live_board_empty" in codes
    # The message has to name the fallback, or a user with no network is stuck.
    text = " ".join(issue.message for issue in result.report.errors)
    assert "import your own" in text.lower()


@requires_payloads
def test_board_is_built_from_whatever_answered(recorded, manifest) -> None:
    """ESPN and Yahoo off still yields a full, usable board.

    This is the realistic degraded case, not a contrived one: ESPN's endpoint
    returns its whole 39 MB player database and can genuinely fail on a busy
    machine, so "the board without ESPN" is a state real users will hit.
    """
    results = _fetch_all(manifest)
    resolved = resolve_board(
        sleeper=results["sleeper"], ffc=results["ffc"], espn=None, yahoo=None,
        season=int(manifest["season"]), team_count=int(manifest["team_count"]),
    )
    assert resolved.ok
    assert len(resolved.frame) >= 200
    assert resolved.frame["overall_adp"].notna().all()
    assert sorted(resolved.successful_sources()) == ["ffc", "sleeper"]


def test_resolver_survives_every_source_being_empty() -> None:
    """No source at all is an empty board with an explanation, not a crash."""
    resolved = resolve_board(sleeper=None, ffc=None, espn=None, yahoo=None, season=2026)
    assert not resolved.ok
    assert resolved.player_count == 0
    assert resolved.report.errors


def test_resolver_survives_a_source_that_lost_its_adp_column() -> None:
    """A source whose payload shape changed yields an explanation, not a crash.

    Modelled on the real failure: an endpoint that starts returning a different
    field name arrives here as a frame missing the column the resolver wants. With
    no ADP from anywhere there is no draft board to build — a list of names in
    arbitrary order would be worse than nothing, because the user cannot tell it
    apart from a real one — so the correct outcome is an empty board that says why.
    """
    from services.providers.base import ProviderResult

    ffc = ProviderResult(
        frame=pd.DataFrame({"player_name": ["Real Player"], "position": ["RB"]}),
        source="Fantasy Football Calculator",
    )
    resolved = resolve_board(sleeper=None, ffc=ffc, espn=None, yahoo=None, season=2026)
    assert not resolved.ok
    assert {issue.code for issue in resolved.report.errors} == {"no_adp"}


def test_a_player_no_adp_source_knows_is_dropped_not_ranked_last() -> None:
    """Sleeper knows thousands of players no one drafts; they must not reach the board.

    Keeping them would put un-drafted practice-squad players in the recommendation
    pool with an imputed ADP, which reads as a real opinion about them.
    """
    from services.providers.base import ProviderResult

    sleeper = ProviderResult(
        frame=pd.DataFrame({
            "player_name": ["Drafted Back", "Practice Squad Guy"],
            "position": ["RB", "RB"],
            "nfl_team": ["DET", "DET"],
        }),
        source="Sleeper",
    )
    ffc = ProviderResult(
        frame=pd.DataFrame({
            "player_name": ["Drafted Back"], "position": ["RB"],
            "nfl_team": ["DET"], "ffc_adp": [12.5],
        }),
        source="Fantasy Football Calculator",
    )
    resolved = resolve_board(sleeper=sleeper, ffc=ffc, season=2026)
    assert resolved.player_count == 1
    assert resolved.frame["player_name"].iloc[0] == "Drafted Back"
    assert "dropped_unranked" in {issue.code for issue in resolved.report.issues}


# ─────────────────────────────────────────────────────────────────────────────
# ESPN league import
# ─────────────────────────────────────────────────────────────────────────────
# There is no recorded payload for this one, and there cannot be a checked-in real
# one: an ESPN league payload is somebody's actual league, names and all. So these run
# against a payload built to ESPN's shape-by-observation — the numeric lineup slot
# ids, the scoringItems list, pickOrder, and draftDetail.picks carrying a playerId and
# nothing else — and assert two things. First, that a league comes across whole.
# Second, and more important for an endpoint nobody publishes, that every way it can
# fail comes back as a message naming the paste importer rather than as an exception.
ESPN_LINEUP_COUNTS = {
    "0": 1, "2": 2, "4": 2, "6": 1, "23": 1, "16": 1, "17": 1, "20": 6,
    "21": 1,   # IR — a seat no pick is spent on
    "10": 2,   # LB — individual defence, which this app does not model
}

ESPN_SCORING_ITEMS = [
    {"statId": 3, "points": 0.04},                                  # 1 pt / 25 pass yds
    {"statId": 4, "points": 6.0},                                   # 6-point passing TD
    {"statId": 20, "points": -2.0},
    {"statId": 24, "points": 0.1},
    {"statId": 25, "points": 6.0},
    {"statId": 42, "points": 0.1},
    {"statId": 43, "points": 6.0},
    {"statId": 53, "points": 1.0, "pointsOverrides": {"4": 1.5}},   # PPR, TE premium
    {"statId": 72, "points": -2.0},
]


def _espn_league_payload(
    season: int,
    *,
    teams: int = 4,
    drafted: bool = True,
    placeholder: bool = False,
    renamed: bool = False,
) -> dict:
    """One season of an ESPN league, shaped the way ESPN shapes it.

    ``placeholder`` gives the board ESPN publishes for a draft that has not happened:
    every seat present, ``playerId`` -1 in all of them. ``renamed`` changes both the team
    names and the manager display names while keeping the owner GUIDs — a league between
    two Augusts.
    """
    label = "was" if renamed else "manager"
    return {
        "id": 123456,
        "seasonId": season,
        "status": {"previousSeasons": [season - 1, season - 2]},
        "settings": {
            "name": "The Gauntlet",
            "size": teams,
            # Deliberately not 1,2,3,4: a seating bug that happens to be the identity
            # map is invisible against an already-sorted pick order.
            "draftSettings": {"type": "SNAKE", "pickOrder": [3, 1, 4, 2][:teams]},
            "rosterSettings": {"lineupSlotCounts": ESPN_LINEUP_COUNTS},
            "scoringSettings": {
                "playerRankType": "PPR", "scoringItems": ESPN_SCORING_ITEMS,
            },
        },
        "teams": [
            {
                "id": i,
                "location": "Old" if renamed else "Team",
                "nickname": f"name {i}" if renamed else str(i),
                "owners": [f"{{GUID-{i}}}"], "primaryOwner": f"{{GUID-{i}}}",
            }
            for i in range(1, teams + 1)
        ],
        "members": [
            {"id": f"{{GUID-{i}}}", "displayName": f"{label}{i}"}
            for i in range(1, teams + 1)
        ],
        "draftDetail": {
            "drafted": drafted,
            "picks": [
                {
                    "playerId": -1 if placeholder else 1000 + i,
                    "teamId": (i % teams) + 1,
                    "roundId": (i // teams) + 1, "roundPickNumber": (i % teams) + 1,
                    "overallPickNumber": i + 1, "keeper": False,
                }
                for i in range(teams * 3)
            ],
        },
    }


def _espn_transport(
    *,
    seasons: tuple[int, ...] = (2026,),
    fail: str = "",
    missing: str = "HTTP 404 Not Found",
    leaguehistory_error: str = "",
    rename_history: bool = False,
    drafted: bool = True,
    placeholder: bool = False,
):
    """A stand-in for ``leagues._espn_get``, at the request boundary.

    Patched here rather than at ``fetch_bytes`` so the ESPN-specific URL and header
    construction runs too — including the ``x-fantasy-filter`` carrying the player-id
    list, which is the part with no second chance if it is wrong.
    """
    calls: list[tuple[str, str, dict]] = []

    def _get(path, *, params, cache_key, ttl=0, headers=None):
        calls.append((path, params, dict(headers or {})))
        if fail and "/players" not in path:
            return None, fail
        if "/players" in path:
            ids = json.loads(headers["x-fantasy-filter"])["filterIds"]["value"]
            return [
                {
                    "id": pid, "fullName": f"Player {pid}",
                    "defaultPositionId": (pid % 4) + 1, "proTeamId": 1,
                }
                for pid in ids
            ], ""
        if path.startswith("leagueHistory"):
            if leaguehistory_error:
                return None, leaguehistory_error
            year = int(params.split("seasonId=")[1].split("&")[0])
            return [_espn_league_payload(year, renamed=rename_history)], ""
        year = int(path.split("seasons/")[1].split("/")[0])
        if year not in seasons:
            return None, missing
        # The highest season the transport serves stands in for "this year"; the flags
        # that describe an unheld draft belong to it and not to the archive.
        if year == max(seasons):
            return _espn_league_payload(
                year, drafted=drafted, placeholder=placeholder
            ), ""
        return _espn_league_payload(year, renamed=rename_history), ""

    _get.calls = calls
    return _get


@pytest.fixture
def espn_leagues(monkeypatch: pytest.MonkeyPatch):
    """The league importer with its transport replaced, and a hook to configure it."""
    from services.providers import leagues as leagues_module

    def _with(**kwargs):
        transport = _espn_transport(**kwargs)
        monkeypatch.setattr(leagues_module, "_espn_get", transport)
        return leagues_module, transport

    return _with


def test_a_public_espn_league_imports_whole_from_its_url(espn_leagues) -> None:
    """The claim the Connect tab makes: paste a link, get a league you can draft."""
    leagues_module, _ = espn_leagues()
    result = leagues_module.fetch_espn_league(
        "https://fantasy.espn.com/football/league?leagueId=123456", season=2026
    )
    assert result.ok, [i.message for i in result.report.errors]
    config = result.league.config
    assert config.name == "The Gauntlet"
    assert config.team_count == 4
    assert config.season == 2026
    assert str(config.platform) == "ESPN"
    # Managers are people, not team names, and are seated in ESPN's own pick order.
    assert [(m.draft_slot, m.name) for m in result.league.managers] == [
        (1, "manager3"), (2, "manager1"), (3, "manager4"), (4, "manager2")
    ]
    assert [m.team_name for m in result.league.managers][0] == "Team 3"


def test_espn_roster_slots_come_across_and_unmodelled_seats_are_reported(
    espn_leagues,
) -> None:
    """Two failures that look identical to a user: a dropped seat and a silent one."""
    from core.enums import Slot

    leagues_module, _ = espn_leagues()
    result = leagues_module.fetch_espn_league(
        "123456", season=2026, include_history=False
    )
    assert dict(result.league.config.roster.slots) == {
        Slot.QB: 1, Slot.RB: 2, Slot.WR: 2, Slot.TE: 1, Slot.FLEX: 1,
        Slot.DST: 1, Slot.K: 1, Slot.BENCH: 6,
    }
    # IR consumes no pick, so it is not a round. The two LB seats are not modelled at
    # all, and the user is told which — rather than left three seats short in silence.
    assert result.league.config.rounds == 15
    warnings = {i.code: i.message for i in result.report.warnings}
    assert "espn_unmapped_slots" in warnings
    assert "LB" in warnings["espn_unmapped_slots"]
    assert "IR" not in warnings["espn_unmapped_slots"]


def test_espn_scoring_is_imported_as_values_not_guessed_from_a_preset(
    espn_leagues,
) -> None:
    """The league's own numbers, because the ones that differ change the board.

    Six-point passing touchdowns move quarterback value by about a round, and a preset
    would have quietly used four.
    """
    from core.enums import ScoringPreset

    leagues_module, _ = espn_leagues()
    result = leagues_module.fetch_espn_league(
        "123456", season=2026, include_history=False
    )
    scoring = result.league.config.scoring
    assert scoring.pass_td == 6.0
    assert scoring.pass_yards_per_point == 25.0    # inverted from ESPN's 0.04 per yard
    assert scoring.rush_yards_per_point == 10.0
    assert scoring.reception == 1.0
    assert scoring.te_premium_reception_bonus == 0.5
    assert scoring.interception == -2.0
    assert scoring.preset is ScoringPreset.TE_PREMIUM


def test_espn_draft_picks_get_real_player_names(espn_leagues) -> None:
    """ESPN's draft board is player *ids*; without the second lookup it is numbers."""
    leagues_module, transport = espn_leagues()
    result = leagues_module.fetch_espn_league("123456", season=2026)
    picks = result.history.all_picks
    assert picks, "no history imported"
    assert all(pick.player_name.startswith("Player ") for pick in picks)
    assert all(pick.manager_name.startswith("manager") for pick in picks)
    # The current season plus the two ESPN says exist, all from the one league link.
    assert result.seasons_found == (2024, 2025, 2026)
    assert {pick.season for pick in picks} == {2024, 2025, 2026}
    # Names came from each season's own player list, not from the current one — ids do
    # not survive across seasons, so reusing 2026's would misname a 2024 pick.
    assert {
        path for path, _params, _headers in transport.calls if "/players" in path
    } == {
        "seasons/2024/players", "seasons/2025/players", "seasons/2026/players",
    }


def test_espn_cookies_go_to_the_request_and_nowhere_else(espn_leagues) -> None:
    """The user's ESPN session is a credential, and this is where that is enforced.

    It has to reach the request — a private league is unreadable otherwise — and it must
    not reach the cache key, which becomes a filename on disk.
    """
    leagues_module, transport = espn_leagues()
    result = leagues_module.fetch_espn_league(
        "123456", season=2026, espn_s2="secret-s2", swid="GUID-2",
    )
    assert result.ok
    cookies = [headers.get("Cookie", "") for _p, _q, headers in transport.calls]
    assert any("espn_s2=secret-s2" in cookie for cookie in cookies)
    # SWID is braced on the way out even when the user pasted it bare.
    assert any("SWID={GUID-2}" in cookie for cookie in cookies)
    assert not any("secret-s2" in f"{path}?{params}" for path, params, _h in transport.calls)
    # And with the cookie, the app knows which team is the user's without being told.
    assert result.league.config.user_draft_slot == 4
    assert [m.draft_slot for m in result.league.managers if m.is_user] == [4]


def test_a_public_espn_league_does_not_guess_which_team_is_yours(espn_leagues) -> None:
    """Slot 1 and a warning, rather than a confident wrong answer."""
    leagues_module, _ = espn_leagues()
    result = leagues_module.fetch_espn_league(
        "123456", season=2026, include_history=False
    )
    assert result.league.config.user_draft_slot == 1
    assert "espn_user_slot" in {i.code for i in result.report.warnings}


@pytest.mark.parametrize("missing", ["HTTP 404 Not Found", "HTTP 401 Unauthorized"])
def test_a_league_not_rolled_over_yet_falls_back_a_season(
    espn_leagues, missing: str
) -> None:
    """August, and the user's league still only exists in last year. Not a dead end.

    Both statuses are covered because the live endpoint answers 401 for a league it has
    no record of, not 404 — so retrying only on 404 would leave this case broken for
    exactly the month people are drafting in.
    """
    leagues_module, _ = espn_leagues(seasons=(2025,), missing=missing)
    result = leagues_module.fetch_espn_league(
        "123456", season=2026, include_history=False
    )
    assert result.ok
    assert result.league.config.season == 2025
    assert "espn_season_fallback" in {i.code for i in result.report.warnings}


def test_bad_cookies_do_not_set_off_a_second_season_of_requests(espn_leagues) -> None:
    """A 401 means something different once cookies are involved.

    Without them it may be a league that has not rolled over. With them it is the
    cookies, and the season has nothing to do with it — so retrying is a wasted round
    trip and a misleading message about the season.
    """
    leagues_module, transport = espn_leagues(fail="HTTP 401 Unauthorized")
    result = leagues_module.fetch_espn_league(
        "123456", season=2026, espn_s2="stale", swid="GUID-1",
    )
    assert not result.ok
    assert len(transport.calls) == 1
    assert "expired" in " ".join(i.message for i in result.report.errors)


@pytest.mark.parametrize(
    "failure, expected",
    [
        # 401 is what the live endpoint returns for a private league *and* for one that
        # does not exist, so its message has to name both possibilities.
        ("HTTP 401 Unauthorized", "no such league"),
        ("HTTP 404 Not Found", "no league"),
        ("network error: simulated outage", "undocumented"),
    ],
)
def test_every_espn_failure_is_a_message_naming_the_paste_importer(
    espn_leagues, failure: str, expected: str
) -> None:
    """The constraint this whole feature is built under.

    ESPN's league API is not a published interface and can move without notice, so no
    failure of it may raise, and none may leave the user without a route. Each of the
    three distinct failures says something different — "it didn't work" is not an
    instruction — and all three name the importer that needs no connection at all.
    """
    leagues_module, _ = espn_leagues(fail=failure)
    result = leagues_module.fetch_espn_league("123456", season=2026)
    assert not result.ok
    assert result.league is None
    message = " ".join(i.message for i in result.report.errors).lower()
    assert expected in message
    assert "paste importer" in message and "draft history" in message


def test_an_espn_league_that_is_not_a_league_is_refused_before_any_request(
    espn_leagues,
) -> None:
    leagues_module, transport = espn_leagues()
    result = leagues_module.fetch_espn_league("my league name")
    assert not result.ok
    assert transport.calls == []
    assert "espn_league_id" in {i.code for i in result.report.errors}


def test_an_espn_draft_with_no_resolvable_players_is_skipped_not_imported() -> None:
    """Picks without names would enter a manager's profile as noise.

    No transport is patched here, so the name lookup cannot succeed — which is exactly
    the real case where ESPN drops player detail for an old season.
    """
    from services.providers import leagues as leagues_module

    assert leagues_module._espn_parse_draft(
        _espn_league_payload(2026), season=2026, league_name="X", current_names={},
    ) is None


def test_an_espn_manager_with_no_display_name_is_labelled_by_their_team() -> None:
    """A real public league came back as nine rows of ``ESPNFAN5108403063``.

    That is what ESPN shows for a manager who never set a display name, and it identifies
    nobody. The team name is at least something the user recognises. A real handle is
    still preferred over a team name, because it survives the annual renaming.
    """
    from services.providers import leagues as leagues_module

    payload = {
        "teams": [
            {"id": 1, "location": "Kittlehead", "nickname": "IPA",
             "primaryOwner": "{G-1}", "owners": ["{G-1}"]},
            {"id": 2, "location": "Mac n", "nickname": "Chase",
             "primaryOwner": "{G-2}", "owners": ["{G-2}"]},
            {"id": 3, "location": "JT", "nickname": "Coming",
             "primaryOwner": "{G-3}", "owners": ["{G-3}"]},
        ],
        "members": [
            {"id": "{G-1}", "displayName": "ESPNFAN5108403063"},
            {"id": "{G-2}", "displayName": "espn85157990"},
            {"id": "{G-3}", "displayName": "AJPayzant21"},
        ],
    }
    assert leagues_module._espn_manager_names(payload) == {
        1: "Kittlehead IPA", 2: "Mac n Chase", 3: "AJPayzant21",
    }


def test_a_past_espn_season_is_read_from_whichever_path_still_answers(
    espn_leagues,
) -> None:
    """``leagueHistory`` is the folklore route for an old season, and it is currently 404.

    It answers 404 for every league and every season, including ones that read perfectly
    well from the ordinary per-season path — so a history import that only knew
    ``leagueHistory`` returned nothing for everybody. Both are tried, per-season first.
    """
    leagues_module, transport = espn_leagues(
        seasons=(2026, 2025), leaguehistory_error="HTTP 404 Not Found",
    )
    result = leagues_module.fetch_espn_league("123456", season=2026)
    assert result.ok
    assert 2025 in result.seasons_found
    assert {pick.season for pick in result.history.all_picks} >= {2025}
    # 2025 came off the per-season path, so leagueHistory was never asked for it.
    assert not any(
        path.startswith("leagueHistory") and "2025" in params
        for path, params, _headers in transport.calls
    )


def test_a_public_league_says_why_last_years_draft_is_missing(espn_leagues) -> None:
    """The user made their league public, connected, and got no history. Why, exactly.

    ESPN stores public/private one season at a time, so opening this year leaves last
    year shut and an anonymous read of it is refused. That is not a lost season and it is
    not a broken app — it is a fixable thing, and the fix is the two cookies. Saying "no
    draft could be read" instead would send the user looking for a problem ESPN does not
    have.
    """
    leagues_module, _ = espn_leagues(
        seasons=(2026,), missing="HTTP 401 Unauthorized",
        leaguehistory_error="HTTP 404 Not Found", placeholder=True,
    )
    result = leagues_module.fetch_espn_league("123456", season=2026)
    assert result.ok, [i.message for i in result.report.errors]
    warnings = {i.code: i.message for i in result.report.warnings}
    assert "espn_history_locked" in warnings
    message = warnings["espn_history_locked"]
    assert "2024, 2025" in message
    assert "only opens the current season" in message
    assert "cookies" in message and "Draft history" in message
    # And not the wrong diagnosis on top of the right one.
    assert "espn_history_gaps" not in warnings
    assert "espn_no_history" not in warnings


def test_a_finished_draft_is_imported_even_when_espn_denies_it_happened(
    espn_leagues,
) -> None:
    """``draftDetail.drafted`` is not reliable, in both directions.

    A real league — 10 teams, 16 rounds, a full board — reported ``drafted: false`` on
    160 picks, and gating on the flag threw the whole draft away. Meanwhile the flag says
    nothing useful about an *upcoming* season, where ESPN pre-creates every seat with
    ``playerId: -1``. So the picks decide, not the flag: a seat counts once it names a
    real player.
    """
    leagues_module, _ = espn_leagues(drafted=False)
    result = leagues_module.fetch_espn_league("123456", season=2026)
    assert 2026 in result.seasons_found
    assert len([p for p in result.history.all_picks if p.season == 2026]) == 12

    leagues_module, _ = espn_leagues(placeholder=True, drafted=True)
    result = leagues_module.fetch_espn_league("123456", season=2026)
    assert 2026 not in result.seasons_found, "an unheld draft was imported as history"
    assert not [p for p in result.history.all_picks if p.season == 2026]


def test_espn_history_follows_the_manager_not_the_team_name(espn_leagues) -> None:
    """Whoever drafted in 2024 is the same person now, whatever they renamed themselves.

    Team names change every August and display names change on a whim. The owner GUID
    does not, so it is joined on first — otherwise an old draft lands on a manager who
    does not exist and the present manager looks like a first-timer.
    """
    leagues_module, _ = espn_leagues(seasons=(2026, 2025, 2024), rename_history=True)
    result = leagues_module.fetch_espn_league("123456", season=2026)
    current = {manager.name for manager in result.league.managers}
    assert current == {"manager1", "manager2", "manager3", "manager4"}
    old_picks = [pick for pick in result.history.all_picks if pick.season < 2026]
    assert old_picks
    # Not "was1"/"Old name 1", which is what the payload called them at the time.
    assert {pick.manager_name for pick in old_picks} == current
    # Each old team still maps to its own manager, rather than all collapsing onto one.
    by_team = {
        (pick.season, pick.pick_in_round): pick.manager_name for pick in old_picks
    }
    assert len(set(by_team.values())) == 4


def test_a_pasted_sleeper_url_reaches_the_same_league_as_the_bare_id() -> None:
    """Nobody retypes a nineteen-digit id; the thing on the clipboard is the URL."""
    from services.providers.leagues import sleeper_league_reference

    assert sleeper_league_reference(
        "https://sleeper.com/leagues/1048291234567890123/team"
    ) == "1048291234567890123"
    assert sleeper_league_reference("1048291234567890123") == "1048291234567890123"
    assert sleeper_league_reference(
        "sleeper.app/leagues/999888777666555/draft"
    ) == "999888777666555"
    # Ambiguity is refused rather than guessed at.
    assert sleeper_league_reference("my league") == ""
    assert sleeper_league_reference("") == ""


def test_an_espn_url_yields_its_league_and_its_season() -> None:
    """Every shape ESPN has put a league id into, including the legacy host."""
    from services.providers.leagues import espn_league_reference

    assert espn_league_reference(
        "https://fantasy.espn.com/football/league?leagueId=123456"
    ) == ("123456", None)
    assert espn_league_reference(
        "https://fantasy.espn.com/football/team?leagueId=98765&teamId=3&seasonId=2024"
    ) == ("98765", 2024)
    assert espn_league_reference(
        "http://games.espn.com/ffl/leaguesetup/ownerinfo?leagueId=555&seasonId=2018"
    ) == ("555", 2018)
    assert espn_league_reference("123456") == ("123456", None)
    assert espn_league_reference("my league") == ("", None)
