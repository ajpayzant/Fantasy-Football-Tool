"""Tests for the eight-lens recommendation engine.

The engine's job is to *disagree with itself usefully*. So the properties worth
pinning are less about any single number than about whether the lenses stay
distinct, stay honest about uncertainty, and never recommend something illegal:

1. **Every lens that fires must name an available player** — a recommendation the
   user cannot act on is worse than no recommendation.
2. **Lenses must be able to diverge.** If they always agree the UI is eight copies
   of one answer, and the trade-off the engine exists to surface is invisible.
3. **Conditional lenses must stay silent rather than fabricate.** BEST_VALUE with
   nobody falling, LAST_CHANCE with everybody safe, and SCARCITY with no run are
   all *absences*, and inventing an answer for them trains a user to ignore them.
4. **The engine must never mutate the draft.** It rolls the board forward
   internally; if that leaks, the user's board fills with simulated picks.
"""

from __future__ import annotations

import pytest

from core.config import LeagueConfig, SimulationConfig
from core.enums import Archetype, Position, RecommendationLens, RiskBand
from engine.draft_state import DraftState
from engine.opponent_model import build_profiles
from engine.recommender import (
    ALTERNATIVE_MIN_UTILITY_SHARE,
    BYE_STACK_WARNING,
    LAST_CHANCE_SURVIVAL,
    RecommendationEngine,
    RecommendationSet,
    recommend_for,
)
from engine.simulator import AvailabilityReport, DraftSimulator, PlayerAvailability
from models.league import League
from models.manager import Manager, ManagerProfile
from models.player import PlayerPool

from tests._smoke_pool import build as build_smoke_pool

TEAMS = 8
ROUNDS = 6
USER_SLOT = 3
FAST_SIMS = 8
"""Rollouts per test. Small deliberately: these tests assert on structure and
ordering, not on convergence of the percentages, and 120 rollouts per test would
make the suite slow enough that people stop running it."""


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def league_pool() -> tuple[LeagueConfig, PlayerPool, League]:
    config, pool = build_smoke_pool(TEAMS, ROUNDS)
    managers = [
        Manager(
            name=f"Manager {slot}", draft_slot=slot,
            archetype=Archetype.BALANCED, is_user=(slot == USER_SLOT),
        )
        for slot in range(1, TEAMS + 1)
    ]
    return config, pool, League(config=config, managers=managers)


@pytest.fixture
def state(
    league_pool: tuple[LeagueConfig, PlayerPool, League],
    settings: SimulationConfig,
) -> DraftState:
    _, pool, league = league_pool
    return DraftState(league=league, pool=pool, settings=settings, seed=21)


@pytest.fixture
def profiles(
    league_pool: tuple[LeagueConfig, PlayerPool, League],
    settings: SimulationConfig,
) -> dict[int, ManagerProfile]:
    _, pool, league = league_pool
    return build_profiles(league, settings=settings, pool=pool)


@pytest.fixture
def at_user_turn(
    state: DraftState, profiles: dict[int, ManagerProfile]
) -> DraftState:
    """The board advanced to the user's first pick."""
    DraftSimulator(state, profiles).simulate_until_user()
    return state


@pytest.fixture
def recommendations(
    at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
) -> RecommendationSet:
    return RecommendationEngine(at_user_turn, profiles).recommend(
        simulations=FAST_SIMS, seed=4
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shape and safety
# ─────────────────────────────────────────────────────────────────────────────
class TestRecommendationShape:
    def test_it_recommends_at_the_users_turn(
        self, recommendations: RecommendationSet, at_user_turn: DraftState
    ) -> None:
        assert recommendations.recommendations
        assert recommendations.draft_slot == USER_SLOT
        assert recommendations.overall_pick == at_user_turn.current_slot.overall_pick
        assert recommendations.round_number == 1

    def test_every_recommended_player_is_actually_available(
        self, recommendations: RecommendationSet, at_user_turn: DraftState
    ) -> None:
        """A recommendation the user cannot act on is a bug, not a suggestion."""
        for rec in recommendations.recommendations:
            assert at_user_turn.is_available(rec.player.player_id), rec.lens

    def test_lenses_are_unique(self, recommendations: RecommendationSet) -> None:
        lenses = [r.lens for r in recommendations.recommendations]
        assert len(lenses) == len(set(lenses))

    def test_best_overall_always_fires(
        self, recommendations: RecommendationSet
    ) -> None:
        """The unconditional anchor the other lenses are alternatives to."""
        assert recommendations.by_lens(RecommendationLens.BEST_OVERALL) is not None
        assert recommendations.primary is not None

    def test_best_overall_is_the_highest_utility_candidate(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        engine = RecommendationEngine(at_user_turn, profiles)
        context = engine._context(USER_SLOT)
        shortlist, _hidden = engine._shortlist(context, 12)
        best = max(shortlist, key=lambda c: c.utility)
        result = engine.recommend(simulations=FAST_SIMS, seed=4)
        assert result.by_lens(
            RecommendationLens.BEST_OVERALL
        ).player.player_id == best.player.player_id

    def test_every_recommendation_explains_itself(
        self, recommendations: RecommendationSet
    ) -> None:
        """No placeholder text: the 'why' is the product."""
        for rec in recommendations.recommendations:
            assert rec.headline
            assert rec.player.name in rec.headline
            assert rec.detail
            assert all(bullet for bullet in rec.detail)
            assert rec.components

    def test_availability_and_bands_are_attached(
        self, recommendations: RecommendationSet
    ) -> None:
        for rec in recommendations.recommendations:
            assert 0.0 <= rec.survival <= 1.0
            assert isinstance(rec.risk_band, RiskBand)

    def test_the_roster_summary_describes_the_user(
        self, recommendations: RecommendationSet
    ) -> None:
        assert "starting seats open" in recommendations.roster_summary

    def test_it_serialises_for_the_ui(
        self, recommendations: RecommendationSet
    ) -> None:
        frame = recommendations.to_frame()
        assert len(frame) == len(recommendations.recommendations)
        assert {"lens", "player_name", "headline", "survival"} <= set(frame.columns)
        payload = recommendations.recommendations[0].to_dict()
        assert payload["lens"]
        assert payload["player_name"]


class TestItDoesNotMutateTheDraft:
    def test_recommending_commits_nothing(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        before = len(at_user_turn.picks)
        available = at_user_turn.available_count()
        RecommendationEngine(at_user_turn, profiles).recommend(simulations=FAST_SIMS)
        assert len(at_user_turn.picks) == before
        assert at_user_turn.available_count() == available

    def test_recommending_twice_is_stable_under_one_seed(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        engine = RecommendationEngine(at_user_turn, profiles)
        first = engine.recommend(simulations=FAST_SIMS, seed=7)
        second = engine.recommend(simulations=FAST_SIMS, seed=7)
        assert [r.player_id for r in first.recommendations] == [
            r.player_id for r in second.recommendations
        ]


# ─────────────────────────────────────────────────────────────────────────────
# The lenses must be able to disagree
# ─────────────────────────────────────────────────────────────────────────────
class TestLensesDiverge:
    def test_over_a_draft_the_lenses_do_not_always_agree(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """The engine's whole purpose: surfacing a trade-off, not one answer.

        Checked across a whole draft rather than one pick because early picks
        legitimately agree — the best player available really is the best fit when
        every seat is open. Disagreement should appear as the roster fills.
        """
        simulator = DraftSimulator(state, profiles)
        engine = RecommendationEngine(state, profiles)
        distinct_counts: list[int] = []
        while not state.is_complete:
            simulator.simulate_until_user()
            if state.is_complete:
                break
            result = engine.recommend(simulations=FAST_SIMS, seed=3)
            distinct_counts.append(
                len({r.player_id for r in result.recommendations})
            )
            state.make_pick(result.primary.player, is_user_pick=True)
        assert distinct_counts
        assert max(distinct_counts) > 1

    def test_best_fit_prefers_a_startable_player_when_one_exists(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Fit means "fills a hole".

        Asserted across the user's turns rather than at one pick, and only when the
        shortlist actually contains someone startable — late in a draft every
        remaining candidate may be bench depth, and the lens is right to name one
        of them then.
        """
        simulator = DraftSimulator(state, profiles)
        engine = RecommendationEngine(state, profiles)
        checked = 0
        while not state.is_complete:
            simulator.simulate_until_user()
            if state.is_complete:
                break
            context = engine._context(USER_SLOT)
            shortlist, _hidden = engine._shortlist(context, 12)
            startable = [
                c for c in shortlist
                if context.view.fills_starting_slot(c.player.position)
            ]
            result = engine.recommend(simulations=FAST_SIMS, seed=5)
            fit = result.by_lens(RecommendationLens.BEST_FIT)
            assert fit is not None
            if startable:
                assert context.view.fills_starting_slot(fit.player.position), (
                    f"best fit {fit.player.position} fills no open seat although "
                    f"{len(startable)} shortlisted players do"
                )
                checked += 1
            state.make_pick(result.primary.player, is_user_pick=True)
        assert checked, "the scenario never arose, so nothing was tested"

    def test_the_alternative_is_a_different_position_from_the_headline(
        self, recommendations: RecommendationSet
    ) -> None:
        alternative = recommendations.by_lens(RecommendationLens.ALTERNATIVE)
        best = recommendations.by_lens(RecommendationLens.BEST_OVERALL)
        if alternative is not None:
            assert alternative.player.position is not best.player.position

    def test_the_alternative_is_held_to_a_utility_floor(
        self, recommendations: RecommendationSet
    ) -> None:
        """A token contrast that is much worse is not a real option."""
        alternative = recommendations.by_lens(RecommendationLens.ALTERNATIVE)
        best = recommendations.by_lens(RecommendationLens.BEST_OVERALL)
        if alternative is not None:
            assert alternative.utility >= best.utility * ALTERNATIVE_MIN_UTILITY_SHARE

    def test_consensus_is_marked_when_lenses_agree(
        self, recommendations: RecommendationSet
    ) -> None:
        counts: dict[str, int] = {}
        for rec in recommendations.recommendations:
            counts[rec.player_id] = counts.get(rec.player_id, 0) + 1
        for rec in recommendations.recommendations:
            assert rec.is_consensus == (counts[rec.player_id] > 1)
        for player in recommendations.consensus_players:
            assert counts[player.player_id] > 1


# ─────────────────────────────────────────────────────────────────────────────
# Conditional lenses stay silent rather than fabricate
# ─────────────────────────────────────────────────────────────────────────────
class TestConditionalLenses:
    @staticmethod
    def _report(state: DraftState, survival: float) -> AvailabilityReport:
        """A hand-built report giving every available player one fixed survival.

        Hand-built rather than simulated because the point is to test the *lens
        thresholds*, and a rollout cannot be asked to produce a board where
        everyone is certainly gone.
        """
        players = {
            p.player_id: PlayerAvailability(
                player=p, survival=survival, simulations=50,
                picks_until_next=10, target_pick=99,
            )
            for p in state.available_players()
        }
        return AvailabilityReport(
            players=players, simulations=50, picks_until_next=10, target_pick=99,
        )

    def test_last_chance_is_silent_when_everyone_is_safe(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        result = RecommendationEngine(at_user_turn, profiles).recommend(
            availability=self._report(at_user_turn, 1.0)
        )
        assert result.by_lens(RecommendationLens.LAST_CHANCE) is None

    def test_last_chance_fires_when_the_board_is_about_to_empty(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        result = RecommendationEngine(at_user_turn, profiles).recommend(
            availability=self._report(at_user_turn, 0.0)
        )
        last = result.by_lens(RecommendationLens.LAST_CHANCE)
        assert last is not None
        assert last.survival <= LAST_CHANCE_SURVIVAL

    def test_last_chance_is_silent_just_above_the_threshold(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Pins the boundary, not just the extremes."""
        result = RecommendationEngine(at_user_turn, profiles).recommend(
            availability=self._report(at_user_turn, LAST_CHANCE_SURVIVAL + 0.01)
        )
        assert result.by_lens(RecommendationLens.LAST_CHANCE) is None

    def test_last_chance_fires_exactly_at_the_threshold(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        result = RecommendationEngine(at_user_turn, profiles).recommend(
            availability=self._report(at_user_turn, LAST_CHANCE_SURVIVAL)
        )
        assert result.by_lens(RecommendationLens.LAST_CHANCE) is not None

    def test_a_cliff_produces_a_warning(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        result = RecommendationEngine(at_user_turn, profiles).recommend(
            availability=self._report(at_user_turn, 0.0)
        )
        assert any("cliff" in w for w in result.warnings)

    def test_no_cliff_warning_when_the_board_is_calm(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        result = RecommendationEngine(at_user_turn, profiles).recommend(
            availability=self._report(at_user_turn, 1.0)
        )
        assert not any("cliff" in w for w in result.warnings)

    def test_a_small_rollout_count_is_flagged_as_indicative(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Overstating what 5 rollouts know is how a user learns to distrust it."""
        result = RecommendationEngine(at_user_turn, profiles).recommend(
            simulations=5, seed=1
        )
        assert any("indicative" in w for w in result.warnings)

    def test_a_full_rollout_count_is_not_flagged(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        result = RecommendationEngine(at_user_turn, profiles).recommend(
            simulations=30, seed=1
        )
        assert not any("indicative" in w for w in result.warnings)

    def test_scarcity_only_names_a_position_the_user_can_start(
        self, recommendations: RecommendationSet
    ) -> None:
        scarcity = recommendations.by_lens(RecommendationLens.SCARCITY)
        if scarcity is not None:
            assert scarcity.player.position in recommendations.pressure

    def test_best_value_only_fires_on_an_actual_faller(
        self, recommendations: RecommendationSet
    ) -> None:
        value = recommendations.by_lens(RecommendationLens.BEST_VALUE)
        if value is not None:
            adp = value.player.adp_for()
            assert adp is not None
            assert recommendations.overall_pick > adp


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_a_complete_draft_recommends_nothing_and_says_why(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        while not state.is_complete:
            state.make_pick(state.best_available())
        result = RecommendationEngine(state, profiles).recommend()
        assert result.recommendations == []
        assert result.warnings
        assert result.primary is None

    def test_a_missing_user_profile_falls_back_to_a_baseline(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """A brand-new user has no history; that must not block a recommendation."""
        thinned = {k: v for k, v in profiles.items() if k != USER_SLOT}
        result = RecommendationEngine(at_user_turn, thinned).recommend(
            simulations=FAST_SIMS, seed=2
        )
        assert result.recommendations

    def test_it_warns_when_the_lineup_cannot_be_filled(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """6 rounds against 9 starting seats is unfillable by arithmetic."""
        DraftSimulator(state, profiles).simulate_until_user()
        result = RecommendationEngine(state, profiles).recommend(
            simulations=FAST_SIMS, seed=1
        )
        assert any("starting seats to fill" in w for w in result.warnings)

    def test_a_supplied_availability_report_is_reused(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """So the UI's availability table and its recommendations always agree."""
        supplied = AvailabilityReport(
            players={}, simulations=42, picks_until_next=9, target_pick=77,
        )
        result = RecommendationEngine(at_user_turn, profiles).recommend(
            availability=supplied
        )
        assert result.availability is supplied
        assert result.picks_until_next == 9

    def test_the_module_level_helper_works(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        result = recommend_for(
            at_user_turn, profiles, simulations=FAST_SIMS, seed=6
        )
        assert result.recommendations

    def test_it_recommends_for_a_slot_that_is_not_on_the_clock(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """The roster-analysis page asks 'what should slot 5 do?' out of turn."""
        result = RecommendationEngine(state, profiles).recommend(
            draft_slot=5, simulations=FAST_SIMS, seed=8
        )
        assert result.draft_slot == 5
        assert result.recommendations


# ─────────────────────────────────────────────────────────────────────────────
# Lenses that need outcome-distribution data
#
# The bare smoke pool publishes no ceiling, floor or risk score, so the upside and
# safety lenses have nothing to rank on there — SAFEST degenerates to whatever the
# shortlist order gives it and HIGHEST_UPSIDE stays silent. Both behaviours are
# correct on a pool that cannot answer the question, and both are also untested
# by the fixtures above, so these tests supply the missing columns explicitly.
# ─────────────────────────────────────────────────────────────────────────────
class TestDistributionLenses:
    @staticmethod
    def _enrich(state: DraftState) -> tuple[str, str]:
        """Give the pool a boom/bust player and a rock-solid one.

        Returns their ids. Both are planted near the top of the board so they reach
        the shortlist; everyone else gets a narrow, unremarkable distribution so the
        two lenses have an unambiguous winner to find.
        """
        available = state.available_players()
        for player in available:
            projection = float(player.projection or 100.0)
            player.ceiling = projection * 1.10
            player.floor = projection * 0.90
            player.risk_score = 0.10

        boom, rock = available[1], available[2]
        boom.ceiling = float(boom.projection) * 2.0
        boom.floor = float(boom.projection) * 0.85
        boom.risk_score = 0.60
        rock.ceiling = float(rock.projection) * 1.05
        rock.floor = float(rock.projection) * 0.99
        rock.risk_score = 0.0
        return boom.player_id, rock.player_id

    def test_the_upside_lens_finds_the_boom_bust_player(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        boom, _ = self._enrich(at_user_turn)
        result = RecommendationEngine(at_user_turn, profiles).recommend(
            simulations=FAST_SIMS, seed=11
        )
        upside = result.by_lens(RecommendationLens.HIGHEST_UPSIDE)
        assert upside is not None
        assert upside.player_id == boom

    def test_the_safety_lens_finds_the_high_floor_player(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        _, rock = self._enrich(at_user_turn)
        result = RecommendationEngine(at_user_turn, profiles).recommend(
            simulations=FAST_SIMS, seed=11
        )
        safest = result.by_lens(RecommendationLens.SAFEST)
        assert safest is not None
        assert safest.player_id == rock

    def test_upside_and_safety_disagree_on_an_enriched_pool(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """The trade-off the two lenses exist to show must actually surface."""
        self._enrich(at_user_turn)
        result = RecommendationEngine(at_user_turn, profiles).recommend(
            simulations=FAST_SIMS, seed=11
        )
        upside = result.by_lens(RecommendationLens.HIGHEST_UPSIDE)
        safest = result.by_lens(RecommendationLens.SAFEST)
        assert upside is not None and safest is not None
        assert upside.player_id != safest.player_id

    def test_the_upside_lens_stays_silent_without_ceiling_data(
        self, recommendations: RecommendationSet
    ) -> None:
        """The un-enriched pool: no ceilings, so no upside claim to make."""
        assert recommendations.by_lens(RecommendationLens.HIGHEST_UPSIDE) is None

    def test_a_suspended_player_is_not_the_safest_pick(
        self, at_user_turn: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        _, rock = self._enrich(at_user_turn)
        at_user_turn.pool.require(rock).suspended = True
        result = RecommendationEngine(at_user_turn, profiles).recommend(
            simulations=FAST_SIMS, seed=11
        )
        assert result.by_lens(RecommendationLens.SAFEST).player_id != rock

    def test_it_warns_about_a_bye_week_stack(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Four starters on one bye is a week with half the lineup out."""
        simulator = DraftSimulator(state, profiles)
        for _ in range(BYE_STACK_WARNING):
            simulator.simulate_until_user()
            if state.is_complete:
                break
            player = state.best_available()
            player.bye_week = 9
            state.make_pick(player, is_user_pick=True)
        simulator.simulate_until_user()
        result = RecommendationEngine(state, profiles).recommend(
            simulations=FAST_SIMS, seed=12
        )
        assert any("week 9 bye" in w for w in result.warnings)
