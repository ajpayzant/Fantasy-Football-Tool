"""Tests for the bundled sample dataset.

Three things are being asserted here, in increasing order of interest:

1. The generated frames flow through the *real* importers without validation
   errors, so the sample data exercises the same path a user's export does.
2. The dataset is labelled as sample data everywhere it enters the app, which is
   an explicit product requirement rather than a nice-to-have.
3. The opponent model recovers all twelve designed archetypes. This is the one that
   matters: it is the only test in the suite where the right answer is known by
   construction rather than asserted from the implementation's own output, so it is
   the only one that can catch the estimator quietly ceasing to estimate.
"""

from __future__ import annotations

import pytest

from core.enums import Archetype, Position
from tests.fixtures.sample_league import (
    DEFAULT_PLAYER_COUNT,
    HISTORY_SEASONS,
    MANAGER_ARCHETYPES,
    SAMPLE_DATA_NOTICE,
    SAMPLE_ROUNDS,
    SAMPLE_TEAM_COUNT,
    sample_bundle,
    sample_draft_history,
    sample_history_frame,
    sample_history_summary,
    sample_league,
    sample_player_frame,
    sample_player_pool,
)
from engine.features import annotate_history
from engine.opponent_model import build_profiles, observe_manager
from services.adapters import (
    SampleDataAdapter,
    available_adapters,
    get_adapter,
    register_adapter,
    unregister_adapter,
)
from services.importers import import_historical_drafts, import_player_pool


@pytest.fixture(scope="module")
def bundle():
    """League, pool and annotated history — built once, since generation is slow."""
    league, pool, history = sample_bundle()
    annotate_history(history, pool=pool, roster=league.config.roster)
    return league, pool, history


# ─────────────────────────────────────────────────────────────────────────────
# The frames import cleanly through the real importers
# ─────────────────────────────────────────────────────────────────────────────
def test_player_frame_imports_without_errors() -> None:
    result = import_player_pool(sample_player_frame(), is_sample_data=True)
    assert result.report.ok, result.report.messages
    assert len(result.pool) >= 200, "the brief calls for a 200+ player pool"
    assert len(result.pool) == DEFAULT_PLAYER_COUNT
    assert not result.rejected_rows


def test_history_frame_imports_without_errors() -> None:
    result = import_historical_drafts(sample_history_frame())
    assert result.report.ok, result.report.messages
    assert not result.rejected_rows
    assert len(result.history.drafts) == len(HISTORY_SEASONS)
    expected = SAMPLE_TEAM_COUNT * SAMPLE_ROUNDS * len(HISTORY_SEASONS)
    assert len(result.history.all_picks) == expected


def test_no_player_drafted_twice_in_a_season() -> None:
    """A board that hands the same player to two managers would silently corrupt
    every positional statistic derived from the history."""
    for draft in sample_draft_history().drafts:
        names = [p.player_name for p in draft.picks]
        assert len(names) == len(set(names)), f"duplicate pick in {draft.season}"


def test_every_manager_ends_with_a_full_roster(bundle) -> None:
    league, _, history = bundle
    for draft in history.drafts:
        for manager in league.managers:
            picks = [p for p in draft.picks if p.manager_key == manager.key]
            assert len(picks) == SAMPLE_ROUNDS


def test_generation_is_deterministic() -> None:
    """Same seed, same data — otherwise the tests below are not reproducible."""
    first = sample_history_frame()
    second = sample_history_frame()
    assert first.equals(second)


def test_seasons_carry_independent_information() -> None:
    """Each season draws from a freshly generated pool, so the three drafts are not
    the same draft three times. Without this the history would look like one season
    of evidence to a shrinkage estimator that is counting three."""
    by_season = {
        season: {
            p.player_name for p in draft.picks if (p.round_number or 99) <= 2
        }
        for draft in sample_draft_history().drafts
        for season in [draft.season]
    }
    seasons = sorted(by_season)
    for earlier, later in zip(seasons, seasons[1:]):
        overlap = by_season[earlier] & by_season[later]
        assert len(overlap) < 12, "consecutive seasons open with the same players"


# ─────────────────────────────────────────────────────────────────────────────
# Labelling — a product requirement, not a detail
# ─────────────────────────────────────────────────────────────────────────────
def test_pool_is_flagged_as_sample_data() -> None:
    pool = sample_player_pool()
    assert pool.metadata.is_sample_data is True
    assert "SAMPLE DATA" in pool.metadata.describe()


def test_notice_names_it_as_fictional() -> None:
    lowered = SAMPLE_DATA_NOTICE.lower()
    assert "sample" in lowered and "fictional" in lowered
    assert "not real" in lowered


def test_league_notes_say_sample() -> None:
    assert "SAMPLE" in sample_league().config.notes.upper()


def test_sample_adapter_is_not_reachable_from_the_app() -> None:
    """The app must offer no route to fictional players.

    The adapter class still exists and works (see the test below), but it is not in
    the default registry — so nothing in the UI can present generated players as a
    data source.
    """
    assert "sample" not in available_adapters()


def test_sample_adapter_flags_its_output_when_registered() -> None:
    """Registered explicitly with the fixture loader, it labels its output.

    This is the labelling contract itself: any adapter that serves synthetic data
    must mark it, so a pool built from it can never look real downstream.
    """
    register_adapter("sample", SampleDataAdapter(loader=sample_player_frame))
    try:
        result = get_adapter("sample").read()
        assert result.ok, result.report.summary()
        assert result.is_sample_data is True
        assert len(result.frame) >= 200
        notices = " ".join(issue.message for issue in result.report.issues).lower()
        assert "fictional" in notices, notices
    finally:
        # The registry is module-level state; leaving the adapter in it would let
        # this test change what a later test sees.
        unregister_adapter("sample")


# ─────────────────────────────────────────────────────────────────────────────
# The point of the exercise: does the model recover what was designed?
# ─────────────────────────────────────────────────────────────────────────────
def test_every_designed_archetype_is_recovered(bundle) -> None:
    """The estimator, given only the picks, arrives at all twelve designed labels.

    Deliberately all-or-nothing. A partial threshold ("at least nine of twelve")
    would let a regression in one branch of ``infer_archetype`` pass unnoticed,
    which is exactly the failure this dataset exists to catch.
    """
    league, pool, history = bundle
    profiles = build_profiles(league, history, pool=pool, annotate=False)

    missed = []
    for manager, designed in zip(league.managers, MANAGER_ARCHETYPES):
        inferred = profiles[manager.draft_slot].archetype
        if inferred is not designed:
            missed.append(f"{manager.name}: designed {designed}, inferred {inferred}")
    assert not missed, "archetype recovery regressed:\n" + "\n".join(missed)


def test_the_designed_archetypes_cover_the_estimators_vocabulary() -> None:
    """Every label ``infer_archetype`` can return is exercised by some seat.

    Two labels are excluded. ``BEST_PLAYER_AVAILABLE`` because no branch returns it
    — see the docstring on ``MANAGER_ARCHETYPES``. ``CUSTOM`` because it means "the
    user described this manager themselves", which is not something an estimator can
    infer and not something a generated seat can demonstrate. If a branch is ever
    added for either, this test fails and the sample league gains a seat for it.
    """
    unreachable = {Archetype.BEST_PLAYER_AVAILABLE, Archetype.CUSTOM}
    assert set(MANAGER_ARCHETYPES) == set(Archetype) - unreachable


def test_managers_differ_in_their_estimated_statistics(bundle) -> None:
    """A demo in which every profile reads the same would prove nothing."""
    _, _, history = bundle
    inversions = {
        name: observe_manager(name, history).rank_inversions.mean
        for name in (m.name for m in sample_league().managers)
    }
    values = [v for v in inversions.values() if v is not None]
    assert len(values) == SAMPLE_TEAM_COUNT
    assert max(values) - min(values) > 5.0, inversions


@pytest.mark.parametrize(
    ("archetype", "position", "predicate", "reason"),
    [
        (Archetype.ZERO_RB, Position.RB, lambda share: share <= 0.05,
         "the zero-RB manager takes almost no early backs"),
        (Archetype.ROBUST_RB, Position.RB, lambda share: share >= 0.9,
         "the robust-RB manager takes almost nothing else early"),
    ],
)
def test_designed_positional_extremes_show_up_in_the_data(
    bundle, archetype, position, predicate, reason
) -> None:
    """The caricatures really are caricatures in the generated picks.

    Guards the generator rather than the estimator: a scripting bug that quietly
    stopped honouring plans would still produce twelve plausible-looking drafts.
    """
    league, _, history = bundle
    name = next(
        manager.name
        for manager, designed in zip(league.managers, MANAGER_ARCHETYPES)
        if designed is archetype
    )
    share = observe_manager(name, history).early_position_share.get(position, 0.0)
    assert predicate(share), f"{reason}; measured {share:.2f}"


def test_plan_summary_publishes_the_answer_key() -> None:
    """The designed plans are surfaced for display, so the demo can be honest about
    what the estimator was up against rather than only showing its output."""
    frame = sample_history_summary()
    assert len(frame) == SAMPLE_TEAM_COUNT
    assert {"manager_name", "designed_archetype"} <= set(frame.columns)
    assert frame["designed_archetype"].nunique() == SAMPLE_TEAM_COUNT
