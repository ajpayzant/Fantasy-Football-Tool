"""Tests for the probabilistic pick engine.

Three things are being pinned here, and they are the three that broke by hand
during development:

1. **ADP is a distribution, not an order.** A player near his ADP must stay
   plausible, and the value term must have the sign a human would expect.
2. **The shortlist must be able to reach every startable position**, including
   the ones whose ADP puts them outside the top of the board.
3. **The softmax must be decisive without being deterministic**, and must order
   predictable managers ahead of erratic ones.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from core.config import LeagueConfig, RosterSettings, SimulationConfig
from core.enums import Archetype, Platform, Position, Slot
from engine.draft_state import DraftState
from engine.features import annotate_history
from engine.opponent_model import build_profiles
from engine.pick_model import (
    NEUTRAL_WHEN_UNKNOWN,
    adp_availability,
    adp_sigma,
    candidate_shortlist,
    choose_player,
    context_for,
    expected_survival,
    most_likely_player,
    pick_probabilities,
    position_probabilities,
    score_candidate,
    score_candidates,
)
from models.draft import DraftHistory
from models.league import League
from models.manager import Manager, ManagerProfile
from models.player import Player, PlayerPool, PoolMetadata

from tests._smoke_pool import build as build_smoke_pool
from tests.conftest import PLANS, ROUNDS, SEASONS, TEAM_COUNT, _build_draft


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def smoke_league() -> tuple[LeagueConfig, PlayerPool, League]:
    """A 4-team, 10-round league whose managers match the synthetic history."""
    config, pool = build_smoke_pool(TEAM_COUNT, ROUNDS)
    managers = [
        Manager(name=name, draft_slot=slot, archetype=Archetype.BALANCED)
        for slot, name in enumerate(PLANS, start=1)
    ]
    return config, pool, League(config=config, managers=managers)


@pytest.fixture
def profiles(
    smoke_league: tuple[LeagueConfig, PlayerPool, League], settings: SimulationConfig
) -> dict[int, ManagerProfile]:
    _, _, league = smoke_league
    history = DraftHistory(drafts=[_build_draft(s) for s in SEASONS])
    annotate_history(history)
    return build_profiles(league, history, settings=settings, annotate=False)


@pytest.fixture
def state(
    smoke_league: tuple[LeagueConfig, PlayerPool, League], settings: SimulationConfig
) -> DraftState:
    _, pool, league = smoke_league
    return DraftState(league=league, pool=pool, settings=settings, seed=7)


def _run_draft(
    state: DraftState, profiles: dict[int, ManagerProfile], seed: int
) -> DraftState:
    """Drive every pick of a draft through the model."""
    rng = random.Random(seed)
    while not state.is_complete:
        slot = state.current_slot
        context = context_for(state, profiles[slot.draft_slot], rng=rng)
        chosen = choose_player(context)
        if chosen is None:
            break
        state.make_pick(chosen.player.player_id)
    return state


def _player(pid: str, position: Position, adp: float | None, **kw) -> Player:
    return Player(
        player_id=pid, name=pid, position=position, overall_adp=adp, **kw
    )


# ─────────────────────────────────────────────────────────────────────────────
# ADP as a distribution
# ─────────────────────────────────────────────────────────────────────────────
class TestAdpSigma:
    def test_a_supplied_stdev_is_used(self, settings: SimulationConfig) -> None:
        player = _player("p", Position.RB, 20.0, adp_stdev=14.0)
        assert adp_sigma(player, 1, settings) == 14.0

    def test_a_supplied_stdev_below_the_floor_is_raised(
        self, settings: SimulationConfig
    ) -> None:
        """A file claiming near-zero uncertainty is not to be believed."""
        player = _player("p", Position.RB, 20.0, adp_stdev=0.5)
        assert adp_sigma(player, 1, settings) == settings.adp_sigma_floor

    def test_uncertainty_grows_with_the_round(self, settings: SimulationConfig) -> None:
        player = _player("p", Position.RB, 20.0)
        early = adp_sigma(player, 1, settings)
        late = adp_sigma(player, 12, settings)
        assert early == settings.adp_sigma_floor
        assert late > early


class TestAdpAvailability:
    def test_a_player_at_his_adp_scores_one(self, settings: SimulationConfig) -> None:
        player = _player("p", Position.RB, 20.0)
        assert adp_availability(player, 20, settings, 1) == pytest.approx(1.0)

    def test_a_nearby_player_stays_plausible(self, settings: SimulationConfig) -> None:
        """The core rule: ADP is a distribution, so ADP 24 at pick 20 is fine."""
        player = _player("p", Position.RB, 24.0)
        assert adp_availability(player, 20, settings, 1) > 0.5

    def test_plausibility_decays_with_distance(
        self, settings: SimulationConfig
    ) -> None:
        near = _player("near", Position.RB, 24.0)
        far = _player("far", Position.RB, 90.0)
        assert adp_availability(near, 20, settings, 1) > adp_availability(
            far, 20, settings, 1
        )

    def test_a_missing_adp_is_neutral(self, settings: SimulationConfig) -> None:
        """Absence of evidence must neither reward nor punish."""
        player = _player("p", Position.RB, None)
        assert adp_availability(player, 20, settings, 1) == NEUTRAL_WHEN_UNKNOWN


class TestExpectedSurvival:
    def test_back_to_back_picks_guarantee_survival(
        self, settings: SimulationConfig
    ) -> None:
        """With no intervening picks nobody can be taken in between."""
        player = _player("p", Position.RB, 20.0)
        assert expected_survival(player, 0, 20, settings, 1) == 1.0

    def test_a_player_well_past_his_adp_is_unlikely_to_last(
        self, settings: SimulationConfig
    ) -> None:
        player = _player("p", Position.RB, 22.0)
        assert expected_survival(player, 20, 20, settings, 1) < 0.2

    def test_a_player_far_from_his_adp_probably_lasts(
        self, settings: SimulationConfig
    ) -> None:
        player = _player("p", Position.RB, 120.0)
        assert expected_survival(player, 10, 20, settings, 1) > 0.9

    def test_survival_falls_as_the_gap_widens(self, settings: SimulationConfig) -> None:
        player = _player("p", Position.RB, 40.0)
        short = expected_survival(player, 4, 20, settings, 1)
        long = expected_survival(player, 24, 20, settings, 1)
        assert short > long


# ─────────────────────────────────────────────────────────────────────────────
# The value term's sign
# ─────────────────────────────────────────────────────────────────────────────
class TestValueTermSign:
    """Regression: the value term rewarded reaches and punished bargains.

    ``adp - pick`` was labelled "fallen past ADP", but a player who fell has a
    *higher* pick number than his ADP. Inverted, the term made the latest-ADP
    player on the board look like the biggest bargain, which is how kickers came
    to be treated as round-1 value.
    """

    def test_a_player_taken_at_his_adp_is_neither(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """At pick 1, the ADP-1 player is exactly on time — no value either way."""
        context = context_for(state, profiles[1])
        on_time = next(
            p for p in state.available_players() if (p.adp_for() or 0) <= 2.0
        )
        assert score_candidate(on_time, context).components["adp_value"] == (
            pytest.approx(0.0, abs=0.05)
        )

    def test_reaching_scores_negative(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Taking the last player on the board at pick 1 is the maximum reach."""
        context = context_for(state, profiles[1])
        reach = max(state.pool.players, key=lambda p: p.adp_for() or 0.0)
        assert score_candidate(reach, context).components["adp_value"] < 0.0

    def test_value_rises_as_a_player_slides(
        self, smoke_league: tuple[LeagueConfig, PlayerPool, League],
        settings: SimulationConfig, profiles: dict[int, ManagerProfile],
    ) -> None:
        """The same player must be better value at pick 30 than at pick 1."""
        _, pool, league = smoke_league
        target = next(p for p in pool.players if (p.adp_for() or 0) >= 20.0)
        early_state = DraftState(league=league, pool=pool, settings=settings, seed=1)
        first = score_candidate(
            target, context_for(early_state, profiles[1])
        ).components["adp_value"]
        # Advance the draft past the player's ADP without taking him.
        while early_state.pick_index < 20:
            nxt = next(
                p for p in early_state.available_players()
                if p.player_id != target.player_id
            )
            early_state.make_pick(nxt.player_id)
        later = score_candidate(
            target, context_for(early_state, profiles[1])
        ).components["adp_value"]
        assert later > first


class TestValueScalesToTheDraftNotTheFile:
    """Regression: every value term was normalised on the *player file*.

    A live import brought 1,003 players instead of the ~300 a spreadsheet holds, and
    that alone changed how the model drafted: the ADP span went to 250 picks, so a
    45-pick reach cost 0.18 utility against a 0.60-weighted positional tendency, and
    projection percentiles at the top of the board compressed into 0.95–1.00 where no
    weight could separate them. Round one filled with fifth-round players.
    """

    def test_the_adp_span_follows_the_round_not_the_file(
        self, state: DraftState, profiles: dict[int, ManagerProfile],
        settings: SimulationConfig,
    ) -> None:
        """A fixed reach costs more in round 1 than the same reach in round 8."""
        context = context_for(state, profiles[1])
        player = _player("reacher", Position.RB, context.overall_pick + 20.0)
        early = score_candidate(player, context).components["adp_value"]
        late = score_candidate(
            player, dataclasses.replace(context, round_number=8, view=None)
        ).components["adp_value"]
        assert early < late < 0.0
        # Two sigma of the round's own spread is the span, so 20 picks early in round 1
        # is a touch under two sigma of six.
        assert early == pytest.approx(-20.0 / (2.0 * settings.adp_sigma_floor))

    def test_a_bigger_file_does_not_change_the_score(
        self, smoke_league: tuple[LeagueConfig, PlayerPool, League],
        settings: SimulationConfig, profiles: dict[int, ManagerProfile],
    ) -> None:
        """The same board plus 400 undrafted names must score the same pick."""
        _, pool, league = smoke_league
        target = next(p for p in pool.players if (p.adp_for() or 0) >= 20.0)
        base = score_candidate(
            target,
            context_for(
                DraftState(league=league, pool=pool, settings=settings, seed=1),
                profiles[1],
            ),
        )
        padded = PlayerPool(
            list(pool.players) + [
                _player(f"filler-{i}", Position.WR, 400.0 + i)
                for i in range(400)
            ],
            league=league.config,
            metadata=PoolMetadata(),
        )
        widened = score_candidate(
            target,
            context_for(
                DraftState(league=league, pool=padded, settings=settings, seed=1),
                profiles[1],
            ),
        )
        assert widened.components["adp_value"] == pytest.approx(
            base.components["adp_value"]
        )

    def test_the_top_of_the_board_stays_distinguishable(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """The best player must out-score the tenth by more than a rounding error."""
        context = context_for(state, profiles[1])
        ranked = sorted(
            (p for p in state.pool.players if p.projection is not None),
            key=lambda p: -(p.projection or 0.0),
        )
        best = score_candidate(ranked[0], context).components["projection"]
        tenth = score_candidate(ranked[9], context).components["projection"]
        assert best - tenth > 0.02

    def test_an_absurd_reach_costs_more_than_a_bad_one(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Regression: clipped at -1, a 45-pick reach and an 80-pick reach tied.

        With the gradient dead past the clip, nothing outweighed the small positive
        from roster need, and kickers came off the board in round one.
        """
        context = context_for(state, profiles[1])
        span = 2.0 * context.settings.adp_sigma_floor
        bad = _player("bad", Position.RB, context.overall_pick + span * 1.5)
        worse = _player("worse", Position.RB, context.overall_pick + span * 2.5)
        bad_value = score_candidate(bad, context).components["adp_value"]
        worse_value = score_candidate(worse, context).components["adp_value"]
        assert bad_value < -1.0
        assert worse_value < bad_value

    def test_a_wide_market_spread_does_not_buy_an_early_round_pass(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Regression: a kicker's 23-pick ADP spread bought him a 46-pick span.

        Taken at face value it put Brandon Aubrey in round three of a live mock. The
        value term uses the tighter of the player's spread and the round's, so a wide
        one cannot forgive an early reach — while :func:`adp_availability`, which asks
        the different question of where he will actually go, still honours it.
        """
        context = context_for(state, profiles[1])
        adp = context.overall_pick + 24.0
        settled = _player("settled", Position.RB, adp, adp_stdev=2.0)
        divisive = _player("divisive", Position.RB, adp, adp_stdev=23.0)
        assert (
            score_candidate(divisive, context).components["adp_value"]
            == pytest.approx(score_candidate(settled, context).components["adp_value"])
        )
        assert adp_availability(
            divisive, context.overall_pick, context.settings, 1
        ) > adp_availability(settled, context.overall_pick, context.settings, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Candidate shortlist
# ─────────────────────────────────────────────────────────────────────────────
class TestCandidateShortlist:
    def test_shortlist_is_capped_but_not_empty(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        context = context_for(state, profiles[1])
        shortlist = candidate_shortlist(context)
        assert shortlist
        # The cap plus at most one extra per open starting slot.
        ceiling = context.settings.candidate_pool_size + len(
            context.config.roster.starting_slots
        ) * len(Position)
        assert len(shortlist) <= ceiling

    def test_late_adp_positions_are_reachable(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Regression: K/DST sit outside the top of the board by construction.

        A board-order-only shortlist can never contain them, so those seats would
        stay empty for the whole draft no matter how hard the imbalance penalty
        pushed. The shortlist must add the best available player per open slot.
        """
        context = context_for(state, profiles[1])
        positions = {p.position for p in candidate_shortlist(context)}
        assert Position.K in positions
        assert Position.DST in positions

    def test_a_filled_slot_stops_being_force_included(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Once the K seat is filled, kickers drop back out of contention."""
        context = context_for(state, profiles[1])
        kicker = next(p for p in context.state.available_at_position(Position.K, 1))
        context.roster.add(kicker)
        refreshed = context_for(state, profiles[1])
        assert Position.K not in {
            p.position for p in candidate_shortlist(refreshed)
        }

    def test_no_duplicates(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        context = context_for(state, profiles[1])
        ids = [p.player_id for p in candidate_shortlist(context)]
        assert len(ids) == len(set(ids))

    def test_drafted_players_never_appear(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        taken = state.available_players(limit=1)[0]
        state.make_pick(taken.player_id)
        context = context_for(state, profiles[2])
        assert taken.player_id not in {
            p.player_id for p in candidate_shortlist(context)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Roster-imbalance penalty
# ─────────────────────────────────────────────────────────────────────────────
class TestImbalancePenalty:
    def test_no_penalty_while_spare_picks_remain(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """At pick 1 nothing is stranded, so nothing should be penalised."""
        context = context_for(state, profiles[1])
        player = state.available_players(limit=1)[0]
        components = score_candidate(player, context).components
        assert components["roster_imbalance_penalty"] == 0.0

    def test_a_pick_that_fills_a_needed_seat_is_never_penalised(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        context = context_for(state, profiles[1])
        kicker = state.available_at_position(Position.K, limit=1)[0]
        assert (
            score_candidate(kicker, context).components["roster_imbalance_penalty"]
            == 0.0
        )

    def test_the_penalty_grades_with_the_shortfall(self) -> None:
        """Regression: a flat penalty lost to board value four picks running.

        Two seats open with one pick left must hurt strictly more than two seats
        open with two picks left, because the second seat is already lost.
        """
        from engine.pick_model import _imbalance_penalty

        roster_settings = RosterSettings(
            slots={Slot.QB: 1, Slot.RB: 1, Slot.K: 1, Slot.DST: 1}
        )
        config = LeagueConfig(
            name="L", season=2026, platform=Platform.ESPN, team_count=2,
            rounds=4, roster=roster_settings,
        )
        from models.draft import TeamRoster

        roster = TeamRoster(
            manager_name="M", draft_slot=1, settings=roster_settings
        )
        roster.add(_player("qb1", Position.QB, 5.0))
        roster.add(_player("rb1", Position.RB, 6.0))
        luxury = _player("wr1", Position.WR, 10.0)
        exact = _imbalance_penalty(luxury, roster, config, picks_left=2)
        behind = _imbalance_penalty(luxury, roster, config, picks_left=1)
        assert 0.0 < exact < behind


# ─────────────────────────────────────────────────────────────────────────────
# Softmax
# ─────────────────────────────────────────────────────────────────────────────
def _fake_scored(utilities: list[float]) -> list:
    from engine.pick_model import ScoredCandidate

    return [
        ScoredCandidate(player=_player(f"p{i}", Position.RB, 10.0), utility=u)
        for i, u in enumerate(utilities)
    ]


class TestSoftmax:
    def test_probabilities_sum_to_one(self) -> None:
        scored = pick_probabilities(_fake_scored([2.0, 1.5, 1.0, 0.5]), 0.5)
        assert sum(c.probability for c in scored) == pytest.approx(1.0)

    def test_the_best_utility_gets_the_highest_probability(self) -> None:
        scored = pick_probabilities(_fake_scored([2.0, 1.5, 1.0]), 0.5)
        assert scored[0].probability == max(c.probability for c in scored)

    def test_a_cold_temperature_concentrates_probability(self) -> None:
        cold = pick_probabilities(_fake_scored([2.0, 1.5, 1.0]), 0.05)
        hot = pick_probabilities(_fake_scored([2.0, 1.5, 1.0]), 2.0)
        assert cold[0].probability > hot[0].probability

    def test_temperature_is_relative_to_the_spread(self) -> None:
        """The same relative ordering must give the same probabilities whether
        utilities are wide (round 1) or narrow (round 12).

        Absolute temperature cannot do this: a fixed value that models a decisive
        manager in round 1 turns him into a coin-flipper by round 12.
        """
        wide = pick_probabilities(_fake_scored([4.0, 2.0, 0.0]), 0.5)
        narrow = pick_probabilities(_fake_scored([0.4, 0.2, 0.0]), 0.5)
        for a, b in zip(wide, narrow):
            assert a.probability == pytest.approx(b.probability, abs=1e-9)

    def test_identical_utilities_give_a_uniform_distribution(self) -> None:
        scored = pick_probabilities(_fake_scored([1.0, 1.0, 1.0, 1.0]), 0.5)
        for candidate in scored:
            assert candidate.probability == pytest.approx(0.25)

    def test_an_empty_field_is_handled(self) -> None:
        assert pick_probabilities([], 0.5) == []

    def test_one_hopeless_candidate_does_not_flatten_the_field(self) -> None:
        """Regression: the spread was measured from the best to the *worst* candidate.

        So adding an obviously terrible option widened the spread, the wider spread
        raised the temperature, and the higher temperature made every *good* option
        less likely — the shortlist's tail set everybody's decisiveness. Measuring to
        the median instead leaves the top of the field alone.
        """
        field = [2.0, 1.9, 1.8, 1.7, 1.6]
        clean = pick_probabilities(_fake_scored(field), 0.5)
        with_outlier = pick_probabilities(_fake_scored([*field, -8.0]), 0.5)
        assert with_outlier[0].probability == pytest.approx(
            clean[0].probability, rel=0.1
        )
        assert with_outlier[-1].probability < 1e-6


class TestPredictabilityOrdersDecisiveness:
    def test_a_predictable_manager_takes_his_top_player_more_often(
        self, state: DraftState, profiles: dict[int, ManagerProfile],
        settings: SimulationConfig,
    ) -> None:
        """Predictability must actually change behaviour, not just a stored number."""
        context = context_for(state, profiles[1])
        sure = pick_probabilities(
            score_candidates(context), settings.temperature_for(0.95)
        )
        erratic = pick_probabilities(
            score_candidates(context), settings.temperature_for(0.05)
        )
        assert sure[0].probability > erratic[0].probability

    def test_even_a_predictable_manager_is_not_deterministic(
        self, state: DraftState, profiles: dict[int, ManagerProfile],
        settings: SimulationConfig,
    ) -> None:
        """No manager is ever locked to one outcome — that is the whole premise."""
        context = context_for(state, profiles[1])
        scored = pick_probabilities(
            score_candidates(context), settings.temperature_for(1.0)
        )
        assert scored[0].probability < 0.999
        assert scored[1].probability > 0.0


class TestEarlyRoundsAreColderThanLateOnes:
    """The first two rounds of a real draft barely deviate from consensus.

    Regression: one temperature for the whole draft made round one as loose as round
    twelve. Jaxon Smith-Njigba went first overall and Drake London — ranked 14th — went
    second, in a room where the top two were not close to being in doubt.
    """

    def test_round_one_is_colder_than_the_middle_rounds(
        self, settings: SimulationConfig
    ) -> None:
        assert settings.temperature_for(0.6, 1) < settings.temperature_for(0.6, 2)
        assert settings.temperature_for(0.6, 2) < settings.temperature_for(0.6, 4)

    def test_the_discount_is_gone_by_the_middle_of_the_draft(
        self, settings: SimulationConfig
    ) -> None:
        baseline = settings.temperature_for(0.6)
        assert settings.temperature_for(0.6, settings.early_round_rounds) == pytest.approx(baseline)
        assert settings.temperature_for(0.6, 12) == pytest.approx(baseline)
        # Omitting the round asks for the manager's own temperature, undiscounted.
        assert settings.temperature_for(0.6, 1) < baseline

    def test_ordering_by_predictability_survives_the_discount(
        self, settings: SimulationConfig
    ) -> None:
        """The discount must scale the curve, not flatten the managers together."""
        assert settings.temperature_for(0.9, 1) < settings.temperature_for(0.2, 1)

    def test_a_consensus_top_pick_dominates_the_first_overall_pick(
        self, state: DraftState, profiles: dict[int, ManagerProfile],
        settings: SimulationConfig,
    ) -> None:
        """Most of the probability at pick 1 must sit on the top few by ADP."""
        context = context_for(state, profiles[1])
        predictability = float(profiles[1].get("predictability"))
        top = {
            p.player_id for p in sorted(
                (p for p in state.pool.players if p.adp_for() is not None),
                key=lambda p: p.adp_for(),
            )[:5]
        }

        def share(round_number: int | None) -> float:
            ranked = pick_probabilities(
                score_candidates(context),
                settings.temperature_for(predictability, round_number),
            )
            return sum(c.probability for c in ranked if c.player.player_id in top)

        assert share(1) > 0.5
        # And the discount is what puts it there.
        assert share(1) > share(None)


class TestKickersAndDefencesWaitForTheEnd:
    """Nothing else in the model expresses the one convention every league keeps.

    A blended board can read a kicker's ADP as round nine of a ten-team draft, and with
    ADP the only thing holding him back one went in round five about every fourth draft.
    """

    def test_the_penalty_fades_as_the_draft_runs_out(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        context = context_for(state, profiles[1])
        kicker = _player("k", Position.K, 90.0)
        rounds = context.config.rounds
        early = score_candidate(kicker, context).components["premature_kicker_penalty"]
        middle = score_candidate(
            kicker, dataclasses.replace(context, round_number=rounds // 2, view=None)
        ).components["premature_kicker_penalty"]
        last = score_candidate(
            kicker, dataclasses.replace(context, round_number=rounds, view=None)
        ).components["premature_kicker_penalty"]
        assert early < middle < 0.0
        assert last == 0.0

    def test_only_kickers_and_defences_pay_it(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        context = context_for(state, profiles[1])
        for position in (Position.QB, Position.RB, Position.WR, Position.TE):
            player = _player(f"p-{position}", position, 90.0)
            assert score_candidate(
                player, context
            ).components["premature_kicker_penalty"] == 0.0
        for position in (Position.K, Position.DST):
            player = _player(f"p-{position}", position, 90.0)
            assert score_candidate(
                player, context
            ).components["premature_kicker_penalty"] < 0.0

    def test_a_round_one_kicker_loses_to_a_comparable_skill_player(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Same ADP, same round: the kicker must not be the pick."""
        context = context_for(state, profiles[1])
        adp = float(context.overall_pick)
        kicker = score_candidate(_player("k", Position.K, adp), context)
        back = score_candidate(_player("rb", Position.RB, adp), context)
        assert kicker.utility < back.utility


# ─────────────────────────────────────────────────────────────────────────────
# Selection
# ─────────────────────────────────────────────────────────────────────────────
class TestChoosePlayer:
    def test_the_same_seed_gives_the_same_pick(
        self, smoke_league: tuple[LeagueConfig, PlayerPool, League],
        profiles: dict[int, ManagerProfile], settings: SimulationConfig,
    ) -> None:
        _, pool, league = smoke_league
        picks = []
        for _ in range(2):
            fresh = DraftState(league=league, pool=pool, settings=settings, seed=7)
            context = context_for(fresh, profiles[1], rng=random.Random(7))
            chosen = choose_player(context)
            picks.append(chosen.player_id)
        assert picks[0] == picks[1]

    def test_different_seeds_can_give_different_picks(
        self, smoke_league: tuple[LeagueConfig, PlayerPool, League],
        profiles: dict[int, ManagerProfile], settings: SimulationConfig,
    ) -> None:
        _, pool, league = smoke_league
        seen = set()
        for seed in range(40):
            fresh = DraftState(league=league, pool=pool, settings=settings, seed=seed)
            context = context_for(fresh, profiles[1], rng=random.Random(seed))
            seen.add(choose_player(context).player_id)
        assert len(seen) > 1

    def test_a_chosen_candidate_carries_its_explanation(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        chosen = choose_player(context_for(state, profiles[1]))
        assert chosen.components
        assert chosen.probability > 0.0
        assert chosen.explain() != "no distinguishing factors"

    def test_an_exhausted_board_returns_none(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        context = context_for(state, profiles[1])
        assert choose_player(context, candidates=[]) is None

    def test_most_likely_player_is_deterministic(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        context = context_for(state, profiles[1])
        first = most_likely_player(context)
        second = most_likely_player(context)
        assert first.player_id == second.player_id

    def test_position_probabilities_sum_to_one(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        context = context_for(state, profiles[1])
        scored = pick_probabilities(score_candidates(context), 0.5)
        totals = position_probabilities(scored)
        assert sum(totals.values()) == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Preference satiation
# ─────────────────────────────────────────────────────────────────────────────
class TestPreferenceSatiation:
    """A positional tendency must be *spent* by acting on it, not repeated.

    Regression: an early-QB manager's quarterback bias fired identically in
    rounds 1, 2 and 3, so he drafted three quarterbacks into a one-QB lineup.
    """

    def test_the_bias_fades_once_the_seat_is_filled(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        slot, qui = next(
            (s, p) for s, p in profiles.items()
            if p.manager_name == "Qui Quarterback"
        )
        context = context_for(state, qui, draft_slot=slot)
        quarterback = state.available_at_position(Position.QB, limit=1)[0]
        before = score_candidate(quarterback, context).components[
            "round_specific_preference"
        ]
        assert before > 0.0

        context.roster.add(quarterback)
        second = state.available_at_position(Position.QB, limit=2)[1]
        after = score_candidate(second, context).components[
            "round_specific_preference"
        ]
        assert 0.0 < after < before

    def test_an_unfilled_seat_keeps_the_full_bias(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        slot, qui = next(
            (s, p) for s, p in profiles.items()
            if p.manager_name == "Qui Quarterback"
        )
        context = context_for(state, qui, draft_slot=slot)
        quarterback = state.available_at_position(Position.QB, limit=1)[0]
        bias = qui.early_round_position_bias.get(Position.QB, 0.0)
        component = score_candidate(quarterback, context).components[
            "round_specific_preference"
        ]
        expected = bias * context.weights.round_specific_preference
        assert component == pytest.approx(expected)


# ─────────────────────────────────────────────────────────────────────────────
# Whole-draft behaviour
# ─────────────────────────────────────────────────────────────────────────────
class TestFullDraft:
    def test_a_draft_completes_and_fills_every_seat_it_can(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        finished = _run_draft(state, profiles, seed=7)
        assert finished.is_complete
        # 9 starting slots in 10 rounds leaves a single spare pick, so at most one
        # seat can be stranded by an early double-up. More than that is a model
        # failure, not bad luck.
        for manager in finished.league.managers:
            open_seats = sum(
                finished.roster(manager.draft_slot).open_starting_slots().values()
            )
            assert open_seats <= 1, manager.name

    def test_no_manager_exceeds_a_positional_limit(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        finished = _run_draft(state, profiles, seed=11)
        for manager in finished.league.managers:
            roster = finished.roster(manager.draft_slot)
            for position in Position:
                limit = finished.config.roster.max_for(position)
                if limit is None:
                    continue
                assert roster.count_at(position) <= limit, (
                    f"{manager.name} exceeded the {position.value} limit"
                )

    def test_every_player_is_drafted_at_most_once(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        finished = _run_draft(state, profiles, seed=23)
        ids = [p.player_id for p in finished.picks]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize(
        "manager_name,position",
        [("Zed Zero", Position.WR), ("Rob Robust", Position.RB)],
    )
    def test_designed_signatures_survive_a_whole_draft(
        self, smoke_league: tuple[LeagueConfig, PlayerPool, League],
        profiles: dict[int, ManagerProfile], settings: SimulationConfig,
        manager_name: str, position: Position,
    ) -> None:
        """The point of the whole exercise: the AI drafts like the real manager.

        Zed was designed as zero-RB and Rob as robust-RB, and neither label was
        ever given to the pick model — it has only their pick histories. Their
        opening rounds must still come out looking like the plan.
        """
        _, pool, league = smoke_league
        hits = 0
        seeds = (7, 11, 23, 42, 99)
        for seed in seeds:
            fresh = DraftState(league=league, pool=pool, settings=settings, seed=seed)
            finished = _run_draft(fresh, profiles, seed=seed)
            slot = next(
                m.draft_slot for m in league.managers if m.name == manager_name
            )
            opening = [p.position for p in finished.picks_by_slot(slot)][:3]
            if opening.count(position) >= 2:
                hits += 1
        # Probabilistic by design, so this is a strong majority rather than
        # every seed: a manager who always did the same thing would not be a
        # model of a human.
        assert hits >= len(seeds) - 1

    def test_a_draft_is_reproducible_end_to_end(
        self, smoke_league: tuple[LeagueConfig, PlayerPool, League],
        profiles: dict[int, ManagerProfile], settings: SimulationConfig,
    ) -> None:
        _, pool, league = smoke_league
        runs = []
        for _ in range(2):
            fresh = DraftState(league=league, pool=pool, settings=settings, seed=42)
            finished = _run_draft(fresh, profiles, seed=42)
            runs.append([p.player_id for p in finished.picks])
        assert runs[0] == runs[1]


# ─────────────────────────────────────────────────────────────────────────────
# Robustness
# ─────────────────────────────────────────────────────────────────────────────
class TestSparseData:
    def test_a_pool_with_no_adp_still_drafts(
        self, settings: SimulationConfig
    ) -> None:
        """The model must run on a bare name-and-position file."""
        roster = RosterSettings(slots={Slot.QB: 1, Slot.RB: 1, Slot.BENCH: 1})
        config = LeagueConfig(
            name="Sparse", season=2026, platform=Platform.ESPN, team_count=2,
            rounds=3, roster=roster,
        )
        players = [
            Player(player_id=f"p{i}", name=f"P{i}",
                   position=Position.QB if i % 2 else Position.RB)
            for i in range(12)
        ]
        pool = PlayerPool(
            players, league=config,
            metadata=PoolMetadata(source="test", is_sample_data=True),
        )
        managers = [
            Manager(name=f"M{s}", draft_slot=s, archetype=Archetype.BALANCED)
            for s in (1, 2)
        ]
        league = League(config=config, managers=managers)
        built = build_profiles(league, DraftHistory(), settings=settings)
        finished = _run_draft(
            DraftState(league=league, pool=pool, settings=settings, seed=3),
            built, seed=3,
        )
        assert finished.is_complete
        assert len(finished.picks) == 6

    def test_scoring_a_player_with_no_data_does_not_raise(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        context = context_for(state, profiles[1])
        bare = Player(player_id="bare", name="Bare", position=Position.WR)
        scored = score_candidate(bare, context)
        assert isinstance(scored.utility, float)


class TestContextFor:
    def test_a_complete_draft_has_no_pick_to_score(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        finished = _run_draft(state, profiles, seed=7)
        with pytest.raises(ValueError):
            context_for(finished, profiles[1])

    def test_picks_left_includes_the_current_pick(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Regression: 'two seats to fill, two picks left' must not read as slack."""
        context = context_for(state, profiles[1])
        assert context.picks_left_for_manager == ROUNDS

    def test_context_reports_the_right_round_and_pick(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        context = context_for(state, profiles[1])
        assert context.overall_pick == 1
        assert context.round_number == 1


class TestNamedPlayerPreferences:
    """What the user says about a specific player has to reach the pick.

    This is the lever behind the Manager Profiles editor: the estimator can only
    describe the history file, and a user who knows their league knows things it
    does not. The tests below pin the three properties that make it trustworthy —
    it does nothing unasked, it moves the pick when asked, and it matches names the
    same forgiving way the importers do.
    """

    def _context(self, state: DraftState, profiles, preferences):
        from models.manager import ManagerPreferences

        profile = profiles[1]
        profile.preferences = preferences or ManagerPreferences()
        return context_for(state, profile)

    def test_no_named_players_means_no_effect_at_all(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """The default has to be inert, or every existing simulation shifts."""
        context = self._context(state, profiles, None)
        for player in context.pool.players[:20]:
            scored = score_candidate(player, context)
            assert scored.components["named_player"] == 0.0

    def test_a_favourite_is_pushed_up_and_a_dislike_pushed_down(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        from models.manager import ManagerPreferences

        pool = state.pool
        loved, hated = pool.players[8], pool.players[9]
        plain = self._context(state, profiles, None)
        baseline_loved = score_candidate(loved, plain).utility
        baseline_hated = score_candidate(hated, plain).utility

        context = self._context(
            state, profiles,
            ManagerPreferences(
                favorite_players=[loved.name], disliked_players=[hated.name]
            ),
        )
        assert score_candidate(loved, context).utility > baseline_loved
        assert score_candidate(hated, context).utility < baseline_hated
        # And it is this term doing it, not a side effect elsewhere.
        assert score_candidate(loved, context).components["named_player"] > 0
        assert score_candidate(hated, context).components["named_player"] < 0

    def test_players_nobody_named_are_left_alone(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Naming one player must not disturb the rest of the board."""
        from models.manager import ManagerPreferences

        pool = state.pool
        named = pool.players[3]
        others = [p for p in pool.players[:15] if p.player_id != named.player_id]
        plain = self._context(state, profiles, None)
        before = {p.player_id: score_candidate(p, plain).utility for p in others}

        context = self._context(
            state, profiles, ManagerPreferences(favorite_players=[named.name])
        )
        for player in others:
            assert score_candidate(player, context).utility == pytest.approx(
                before[player.player_id]
            )

    def test_a_name_is_matched_the_way_the_importers_match_it(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Typing "aj brown" must find "A.J. Brown Jr." — punctuation and suffix free.

        Without this the feature looks broken in exactly the case it is most likely
        to be used in: a name typed from memory.
        """
        from models.manager import ManagerPreferences
        from models.player import Player

        target = Player(
            player_id="ajb", name="A.J. Brown Jr.", position=Position.WR,
            overall_adp=5.0, projection=250.0,
        )
        plain = self._context(state, profiles, None)
        baseline = score_candidate(target, plain).utility
        context = self._context(
            state, profiles, ManagerPreferences(favorite_players=["  aj brown  "])
        )
        # The component is the term times its weight, so a full match reads as the
        # weight itself — anything less would mean the name did not match.
        assert score_candidate(target, context).components["named_player"] == pytest.approx(
            context.weights.named_player_preference
        )
        assert score_candidate(target, context).utility > baseline

    def test_the_effect_is_large_enough_to_actually_change_a_pick(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """A weight too small to move the choice would be a lie to the user.

        Checked against the model's own decision rather than the raw utility: the
        second-best player on the board, named as a favourite, has to become the
        likely pick.
        """
        from models.manager import ManagerPreferences

        plain = self._context(state, profiles, None)
        ranked = sorted(
            score_candidates(plain, candidate_shortlist(plain)),
            key=lambda c: -c.utility,
        )
        assert len(ranked) >= 2
        favourite = ranked[1].player
        assert most_likely_player(plain) is not None

        context = self._context(
            state, profiles, ManagerPreferences(favorite_players=[favourite.name])
        )
        assert most_likely_player(context).player_id == favourite.player_id
