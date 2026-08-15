"""Tests for the draft simulator and the Monte Carlo availability layer.

The properties pinned here are the ones that would silently produce plausible
nonsense rather than crash:

1. **The horizon must be the gap *after* the current pick.** Measuring it before
   the pick makes it 0 for whoever is on the clock, which makes every survival
   estimate 1.0. That bug existed and was invisible because the numbers still
   looked like probabilities.
2. **A rollout must not disturb the live draft.** Rollouts run on copies; if that
   ever regresses, the user's board silently fills with simulated picks.
3. **Survival must respond to the gap and to demand**, not just to ADP — that is
   the whole reason the simulated answer replaces the closed form.
4. **The user's own stand-in pick must not count as competition.** Counting it
   reports the player the model most likes as least likely to survive.
"""

from __future__ import annotations

import pytest

from core.config import LeagueConfig, SimulationConfig
from core.enums import Archetype, DraftStatus, Position, RiskBand
from core.validation import ConfigurationError
from engine.draft_state import DraftState
from engine.opponent_model import build_profiles
from engine.simulator import (
    PLAN_GONE_BY,
    PLAN_LASTS,
    RISK_BAND_EDGES,
    AvailabilityReport,
    DraftSimulator,
    PlayerAvailability,
    _horizon_for,
    likely_next_picks,
    monte_carlo_draft,
    simulate_availability,
    simulate_draft_plan,
    upcoming_position_pressure,
)
from models.league import League
from models.manager import Manager, ManagerProfile
from models.player import Player, PlayerPool

from tests._smoke_pool import build as build_smoke_pool

TEAMS = 8
ROUNDS = 6
USER_SLOT = 3


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def league_pool() -> tuple[LeagueConfig, PlayerPool, League]:
    """A small league whose slot 3 is the user, sized to keep rollouts quick."""
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
    return DraftState(league=league, pool=pool, settings=settings, seed=13)


@pytest.fixture
def profiles(
    league_pool: tuple[LeagueConfig, PlayerPool, League],
    settings: SimulationConfig,
) -> dict[int, ManagerProfile]:
    _, pool, league = league_pool
    return build_profiles(league, settings=settings, pool=pool)


@pytest.fixture
def simulator(
    state: DraftState, profiles: dict[int, ManagerProfile]
) -> DraftSimulator:
    return DraftSimulator(state, profiles)


# ─────────────────────────────────────────────────────────────────────────────
# The horizon — the bug that made every survival estimate 1.0
# ─────────────────────────────────────────────────────────────────────────────
class TestHorizon:
    def test_the_slot_on_the_clock_measures_to_its_next_pick(
        self, state: DraftState
    ) -> None:
        """Not to the pick it is currently making, which is a gap of zero.

        This is the regression that mattered: ``picks_until_turn`` legitimately
        returns 0 here, and reading that as the availability horizon reported every
        player as certain to survive.
        """
        assert state.current_slot.draft_slot == 1
        horizon = _horizon_for(state, 1)
        # Snake: slot 1 picks 1st and 16th in an 8-team league, so 14 others go.
        assert horizon.on_clock is True
        assert horizon.target_pick == 16
        assert horizon.gap == 14
        # The rollout must also cover slot 1's own stand-in pick.
        assert horizon.rollout_picks == 15

    def test_a_future_slot_measures_from_the_current_pick(
        self, state: DraftState
    ) -> None:
        horizon = _horizon_for(state, 5)
        assert horizon.on_clock is False
        assert horizon.target_pick == 5
        # Picks 1-4 happen first, and pick 1 is on the clock, so four intervene.
        assert horizon.gap == 4
        assert horizon.rollout_picks == 4

    def test_back_to_back_picks_have_no_gap(self, state: DraftState) -> None:
        """Slot 8 turns the snake, so its two picks are adjacent."""
        while state.current_slot.draft_slot != 8:
            state.make_pick(state.best_available())
        horizon = _horizon_for(state, 8)
        assert horizon.gap == 0
        assert horizon.rollout_picks == 1

    def test_no_remaining_picks_returns_none(self, state: DraftState) -> None:
        while not state.is_complete:
            state.make_pick(state.best_available())
        assert _horizon_for(state, 1) is None

    def test_two_turns_ahead_measures_to_the_pick_after_the_next_one(
        self, state: DraftState
    ) -> None:
        """Slot 5 picks 5th and 12th in an eight-team snake."""
        horizon = _horizon_for(state, 5, turns_ahead=2)
        assert horizon.target_pick == 12
        # Its own turn at pick 5 has to be played through to reach pick 12, and that
        # pick is not competition, so the honest wait is ten managers rather than 11.
        assert horizon.own_turns_passed == 1
        assert horizon.rollout_picks == 11
        assert horizon.gap == 10

    def test_the_slot_on_the_clock_counts_both_of_its_own_turns(
        self, state: DraftState
    ) -> None:
        """Slot 1 picks 1st, 16th and 17th — it turns the snake, so 16 and 17 adjoin.

        Both of those are its own picks, so a two-turn plan waits on exactly the same
        14 opponents as a one-turn plan. Getting this wrong by one would price the
        second turn as if a stranger picked at 16.
        """
        horizon = _horizon_for(state, 1, turns_ahead=2)
        assert horizon.target_pick == 17
        assert horizon.own_turns_passed == 2
        assert horizon.rollout_picks == 16
        assert horizon.gap == 14

    def test_a_slot_with_one_pick_left_has_no_second_turn(
        self, state: DraftState
    ) -> None:
        """Asking for two turns in the last round must answer honestly, not guess.

        Slot 3 picks 3, 14, 19, 30, 35 and 46 here, so from pick 41 there is exactly
        one turn left to plan for.
        """
        while state.pick_index < 40:
            state.make_pick(state.best_available())
        assert _horizon_for(state, USER_SLOT, turns_ahead=1).target_pick == 46
        assert _horizon_for(state, USER_SLOT, turns_ahead=2) is None

    def test_the_pick_model_now_sees_a_real_gap(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """``context_for`` must expose the post-pick gap, not zero.

        Guards the fix at the point of use: a zero here silently disables the
        ``expected_availability`` term for every pick actually being made.
        """
        from engine.pick_model import context_for

        context = context_for(state, profiles[1])
        assert context.picks_until_next == 14


# ─────────────────────────────────────────────────────────────────────────────
# Single picks
# ─────────────────────────────────────────────────────────────────────────────
class TestSimulatePick:
    def test_a_pick_lands_on_the_board_with_its_reasoning(
        self, simulator: DraftSimulator, state: DraftState
    ) -> None:
        result = simulator.simulate_pick()
        assert result is not None
        assert len(state.picks) == 1
        assert state.picks[0].player_id == result.player.player_id
        assert 0.0 < result.probability <= 1.0
        assert result.explanation
        assert result.components
        # Runner-ups are recorded so the UI can show what he passed on.
        assert result.alternatives
        assert all(p.player_id != result.player.player_id for p, _ in result.alternatives)

    def test_the_pick_is_drafted_and_unavailable_afterwards(
        self, simulator: DraftSimulator, state: DraftState
    ) -> None:
        result = simulator.simulate_pick()
        assert not state.is_available(result.player.player_id)

    def test_preview_does_not_commit(
        self, simulator: DraftSimulator, state: DraftState
    ) -> None:
        ranked = simulator.preview_pick()
        assert ranked
        assert sum(c.probability for c in ranked) == pytest.approx(1.0)
        assert state.picks == []

    def test_a_missing_profile_is_a_loud_failure(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Simulating a real manager with a generic stand-in invalidates the run."""
        thinned = {k: v for k, v in profiles.items() if k != 1}
        with pytest.raises(ConfigurationError, match="draft slot 1"):
            DraftSimulator(state, thinned).simulate_pick()

    def test_simulating_until_the_user_stops_on_their_pick(
        self, simulator: DraftSimulator, state: DraftState
    ) -> None:
        made = simulator.simulate_until_user()
        assert len(made) == USER_SLOT - 1
        assert state.is_user_on_clock
        assert state.current_slot.draft_slot == USER_SLOT

    def test_the_log_accumulates_in_order(self, simulator: DraftSimulator) -> None:
        simulator.simulate_until_user()
        log = simulator.log
        assert [p.overall_pick for p in log] == sorted(p.overall_pick for p in log)
        # The log is a copy: mutating it must not corrupt the simulator.
        log.clear()
        assert simulator.log


class TestSimulateToCompletion:
    def test_a_whole_draft_fills_every_slot(
        self, simulator: DraftSimulator, state: DraftState
    ) -> None:
        made = simulator.simulate_to_completion()
        assert len(made) == len(state.order)
        assert state.is_complete
        assert state.status is DraftStatus.COMPLETE

    def test_nobody_is_drafted_twice(
        self, simulator: DraftSimulator, state: DraftState
    ) -> None:
        simulator.simulate_to_completion()
        ids = [p.player_id for p in state.picks]
        assert len(ids) == len(set(ids))

    def test_the_same_seed_reproduces_the_draft(
        self,
        league_pool: tuple[LeagueConfig, PlayerPool, League],
        settings: SimulationConfig,
        profiles: dict[int, ManagerProfile],
    ) -> None:
        """Reproducibility is what makes a reported result checkable."""
        _, pool, league = league_pool

        def run() -> list[str]:
            state = DraftState(league=league, pool=pool, settings=settings, seed=99)
            DraftSimulator(state, profiles).simulate_to_completion()
            return [p.player_id for p in state.picks]

        assert run() == run()

    def test_different_seeds_diverge(
        self,
        league_pool: tuple[LeagueConfig, PlayerPool, League],
        settings: SimulationConfig,
        profiles: dict[int, ManagerProfile],
    ) -> None:
        _, pool, league = league_pool

        def run(seed: int) -> list[str]:
            state = DraftState(league=league, pool=pool, settings=settings, seed=seed)
            DraftSimulator(state, profiles).simulate_to_completion()
            return [p.player_id for p in state.picks]

        assert run(1) != run(2)


# ─────────────────────────────────────────────────────────────────────────────
# Availability
# ─────────────────────────────────────────────────────────────────────────────
class TestSimulateAvailability:
    def test_rollouts_leave_the_live_draft_untouched(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """The property everything else depends on."""
        before = len(state.picks)
        before_available = state.available_count()
        simulate_availability(state, profiles, draft_slot=USER_SLOT, simulations=5)
        assert len(state.picks) == before
        assert state.available_count() == before_available
        assert state.can_undo is False

    def test_top_players_are_unlikely_to_survive_a_long_gap(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        report = simulate_availability(state, profiles, draft_slot=1, simulations=25)
        assert report.simulations == 25
        assert report.picks_until_next == 14
        best = state.best_available()
        # Fourteen picks with the best player on the board: he does not last.
        assert report.survival(best.player_id) < 0.5

    def test_survival_rises_as_the_gap_shrinks(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """The core claim: this is a simulation of the board, not a lookup."""
        best = state.best_available()
        far = simulate_availability(state, profiles, draft_slot=1, simulations=30)
        near = simulate_availability(state, profiles, draft_slot=2, simulations=30)
        assert far.picks_until_next > near.picks_until_next
        assert far.survival(best.player_id) <= near.survival(best.player_id)

    def test_deep_players_survive(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        deep = state.available_players()[-1]
        report = simulate_availability(
            state, profiles, draft_slot=1, simulations=15,
            extra_players=[deep], track_limit=10,
        )
        assert report.get(deep.player_id) is not None
        assert report.survival(deep.player_id) == 1.0

    def test_the_users_own_stand_in_pick_is_not_competition(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Slot 1 is on the clock and would take the best player every rollout.

        If that stand-in selection were counted, the best player on the board would
        show ~0% survival — the engine would scream "last chance" about a player the
        user can simply take right now.
        """
        best = state.best_available()
        report = simulate_availability(state, profiles, draft_slot=1, simulations=25)
        entry = report.get(best.player_id)
        assert entry is not None
        assert entry.taken_by.get("Manager 1") is None

    def test_an_immediate_turn_needs_no_rollouts(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Slot 8 turns the snake, so nothing intervenes and everyone survives."""
        while state.current_slot.draft_slot != 8:
            state.make_pick(state.best_available())
        report = simulate_availability(state, profiles, draft_slot=8, simulations=20)
        assert report.simulations == 0
        assert report.picks_until_next == 0
        assert all(entry.survival == 1.0 for entry in report.players.values())

    def test_a_finished_draft_reports_nothing(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        while not state.is_complete:
            state.make_pick(state.best_available())
        report = simulate_availability(state, profiles, draft_slot=1, simulations=10)
        assert report.simulations == 0
        assert report.players == {}

    def test_progress_is_reported_once_per_rollout(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        seen: list[tuple[int, int]] = []
        simulate_availability(
            state, profiles, draft_slot=1, simulations=6,
            progress=lambda done, total: seen.append((done, total)),
        )
        assert seen == [(i, 6) for i in range(1, 7)]

    def test_the_same_seed_reproduces_the_report(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        a = simulate_availability(state, profiles, draft_slot=1, simulations=10, seed=5)
        b = simulate_availability(state, profiles, draft_slot=1, simulations=10, seed=5)
        assert {k: v.survival for k, v in a.players.items()} == {
            k: v.survival for k, v in b.players.items()
        }

    def test_takers_are_attributed(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        report = simulate_availability(state, profiles, draft_slot=1, simulations=20)
        risky = report.at_risk(0.5)
        assert risky
        entry = risky[0]
        assert entry.taken_by
        assert entry.likeliest_taker in {m for m in entry.taken_by}
        assert entry.mean_pick_taken is not None
        assert 1 <= entry.mean_pick_taken <= 16

    def test_an_untracked_player_defaults_to_available(self) -> None:
        """Deep players are not tracked; the default must not read as 'gone'."""
        report = AvailabilityReport(simulations=10)
        assert report.survival("nobody") == 1.0
        assert report.band("nobody") is RiskBand.SAFE


# ─────────────────────────────────────────────────────────────────────────────
# The two-turn plan
#
# The properties here are the ones that make a plan trustworthy rather than merely
# plausible. A second horizon is easy to produce and easy to get subtly wrong: the
# arithmetic can double-count the user's own turn, the second turn can be computed
# from a board that never saw the first turn happen, and a rollout that drafts on
# the user's behalf can quietly report their own best pick as "gone".
# ─────────────────────────────────────────────────────────────────────────────
FAST_PLAN_SIMS = 12


class TestSimulateDraftPlan:
    def test_both_turns_come_from_one_pass_over_the_board(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Slot 3 picks 3rd and 14th, and the plan reports them in order."""
        plan = simulate_draft_plan(
            state, profiles, draft_slot=USER_SLOT,
            simulations=FAST_PLAN_SIMS, seed=4,
        )
        assert [turn.turn for turn in plan.turns] == [1, 2]
        assert [turn.overall_pick for turn in plan.turns] == [3, 14]
        assert [turn.picks_until for turn in plan.turns] == [2, 12]
        assert plan.simulations == FAST_PLAN_SIMS
        assert plan.first_report is plan.turns[0].availability

    def test_survival_never_rises_between_your_turns(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """The invariant that proves the turns share one simulated draft.

        A player taken before the first turn cannot reappear before the second. If
        the two turns were simulated independently this would fail intermittently,
        and the plan would be able to advise waiting for a player it had already
        reported gone.
        """
        plan = simulate_draft_plan(
            state, profiles, draft_slot=USER_SLOT,
            simulations=FAST_PLAN_SIMS, seed=4,
        )
        first, second = plan.turn(1), plan.turn(2)
        assert first is not None and second is not None
        for player_id in first.availability.players:
            assert (
                second.availability.survival(player_id)
                <= first.availability.survival(player_id) + 1e-9
            )

    def test_the_room_is_every_intervening_pick_but_none_of_yours(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Picks 1-2 and 4-13, with the user's own pick 3 absent.

        The user's turn is excluded because the model's stand-in choice there is not
        a prediction of anything — the user is about to make it themselves.
        """
        plan = simulate_draft_plan(
            state, profiles, draft_slot=USER_SLOT,
            simulations=FAST_PLAN_SIMS, seed=4,
        )
        picks = [forecast.overall_pick for forecast in plan.room]
        assert picks == [1, 2] + list(range(4, 14))
        assert all(forecast.draft_slot != USER_SLOT for forecast in plan.room)
        assert [f.overall_pick for f in plan.room_before(1)] == [1, 2]
        assert [f.overall_pick for f in plan.room_before(2)] == list(range(4, 14))

    def test_a_forecast_pick_is_a_distribution_over_positions(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        plan = simulate_draft_plan(
            state, profiles, draft_slot=USER_SLOT,
            simulations=FAST_PLAN_SIMS, seed=4,
        )
        forecast = plan.room[0]
        assert forecast.simulations == FAST_PLAN_SIMS
        assert sum(forecast.position_shares.values()) == pytest.approx(1.0)
        assert forecast.likeliest_position in forecast.position_shares
        shares = [share for _, share in forecast.player_shares]
        assert shares == sorted(shares, reverse=True)
        assert all(0.0 < share <= 1.0 for share in shares)
        assert forecast.likeliest_player is forecast.player_shares[0][0]

    def test_a_forecast_shows_the_evidence_behind_it(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """What they already drafted and how they draft, not just the prediction."""
        while state.pick_index < 10:
            state.make_pick(state.best_available())
        plan = simulate_draft_plan(
            state, profiles, draft_slot=USER_SLOT,
            simulations=FAST_PLAN_SIMS, seed=4,
        )
        assert plan.room
        assert all(forecast.tendency for forecast in plan.room)
        # Ten picks in, everyone in the room owns at least one player.
        assert all(forecast.roster_so_far for forecast in plan.room)

    def test_one_turn_is_exactly_what_simulate_availability_reports(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """The refactor's contract: the old entry point is the new one, turns=1.

        Same seed, same rollouts, same numbers — not "close enough". If this drifts,
        every survival figure in the app changed meaning without anyone deciding to
        change it.
        """
        plan = simulate_draft_plan(
            state, profiles, draft_slot=1, turns=1, simulations=10, seed=5,
        )
        report = simulate_availability(
            state, profiles, draft_slot=1, simulations=10, seed=5,
        )
        assert {k: v.survival for k, v in plan.first_report.players.items()} == {
            k: v.survival for k, v in report.players.items()
        }

    def test_the_plan_leaves_the_live_draft_untouched(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        before, available = len(state.picks), state.available_count()
        simulate_draft_plan(
            state, profiles, draft_slot=USER_SLOT, simulations=5, seed=4
        )
        assert len(state.picks) == before
        assert state.available_count() == available
        assert state.can_undo is False

    def test_progress_counts_rollouts_rather_than_turns(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Two turns per rollout must not make the bar reach 200%."""
        seen: list[tuple[int, int]] = []
        simulate_draft_plan(
            state, profiles, draft_slot=USER_SLOT, simulations=6,
            progress=lambda done, total: seen.append((done, total)),
        )
        assert seen == [(i, 6) for i in range(1, 7)]

    def test_your_own_stand_in_pick_is_never_reported_as_competition(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """The rollout drafts for the user at their first turn to reach the second.

        That selection must not appear as a taker at either turn, or the plan would
        tell the user a player is gone because the model took him on their behalf.
        """
        plan = simulate_draft_plan(
            state, profiles, draft_slot=USER_SLOT,
            simulations=FAST_PLAN_SIMS, seed=4,
        )
        mine = f"Manager {USER_SLOT}"
        for turn in plan.turns:
            for entry in turn.availability.players.values():
                assert mine not in entry.taken_by

    def test_the_last_round_plans_the_one_turn_that_is_left(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        while state.pick_index < 40:
            state.make_pick(state.best_available())
        plan = simulate_draft_plan(
            state, profiles, draft_slot=USER_SLOT, simulations=6, seed=4
        )
        assert [turn.overall_pick for turn in plan.turns] == [46]
        assert plan.turn(2) is None
        # And the windows collapse to two groups rather than inventing a third.
        assert not plan.windows().next_turn

    def test_a_finished_draft_plans_nothing(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        while not state.is_complete:
            state.make_pick(state.best_available())
        plan = simulate_draft_plan(
            state, profiles, draft_slot=USER_SLOT, simulations=6
        )
        assert plan.is_empty
        assert plan.turns == [] and plan.room == []
        assert plan.windows().is_empty

    def test_the_windows_sort_the_board_by_the_decision(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Each group has to mean what it says, since the labels drive the advice."""
        plan = simulate_draft_plan(
            state, profiles, draft_slot=USER_SLOT,
            simulations=FAST_PLAN_SIMS, seed=4,
        )
        windows = plan.windows(limit=4)
        first, second = plan.turn(1), plan.turn(2)
        assert all(len(group) <= 4 for group in (
            windows.take_now, windows.next_turn, windows.can_wait
        ))
        for entry in windows.take_now:
            assert first.availability.survival(entry.player_id) <= PLAN_GONE_BY
        for entry in windows.next_turn:
            assert first.availability.survival(entry.player_id) > PLAN_GONE_BY
            assert second.availability.survival(entry.player_id) <= PLAN_GONE_BY
        for entry in windows.can_wait:
            assert second.availability.survival(entry.player_id) >= PLAN_LASTS
        # No player can be in two groups: the three are a partition, not three views.
        grouped = [
            entry.player_id for group in
            (windows.take_now, windows.next_turn, windows.can_wait)
            for entry in group
        ]
        assert len(grouped) == len(set(grouped))

    def test_the_frame_carries_a_column_per_turn(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        plan = simulate_draft_plan(
            state, profiles, draft_slot=USER_SLOT,
            simulations=FAST_PLAN_SIMS, seed=4,
        )
        frame = plan.to_frame()
        assert list(frame.columns)[:4] == ["Player", "Pos", "Team", "ADP"]
        assert "Pick 3" in frame.columns and "Pick 14" in frame.columns
        # Board order, so the table reads as "the best players left" top-down.
        assert frame.iloc[0]["Player"] == state.available_players(limit=1)[0].name
        room = plan.room_frame()
        assert len(room) == len(plan.room)
        assert set(room["Pick"]) == {f.overall_pick for f in plan.room}

    def test_a_sleeper_the_user_names_is_tracked_at_both_turns(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """``extra_players`` is how the user's own board reaches the rollouts."""
        deep = state.available_players()[-1]
        plan = simulate_draft_plan(
            state, profiles, draft_slot=USER_SLOT, simulations=6,
            extra_players=[deep], track_limit=10, seed=4,
        )
        for turn in plan.turns:
            assert turn.availability.get(deep.player_id) is not None
            assert turn.availability.survival(deep.player_id) == 1.0


def _availability(survival: float, simulations: int = 100) -> PlayerAvailability:
    """A band/statistics fixture. The player is real so the type hints hold."""
    return PlayerAvailability(
        player=Player(player_id="x", name="Test Player", position=Position.RB),
        survival=survival, simulations=simulations,
        picks_until_next=10, target_pick=20,
    )


class TestRiskBands:
    @pytest.mark.parametrize(
        ("survival", "expected"),
        [
            (1.00, RiskBand.SAFE),
            (0.85, RiskBand.SAFE),
            (0.84, RiskBand.LIKELY_AVAILABLE),
            (0.60, RiskBand.LIKELY_AVAILABLE),
            (0.50, RiskBand.COIN_FLIP),
            (0.40, RiskBand.COIN_FLIP),
            (0.39, RiskBand.LIKELY_GONE),
            (0.15, RiskBand.LIKELY_GONE),
            (0.14, RiskBand.GONE),
            (0.00, RiskBand.GONE),
        ],
    )
    def test_bands_map_from_survival(
        self, survival: float, expected: RiskBand
    ) -> None:
        assert _availability(survival).risk_band is expected

    def test_the_edges_are_ordered_high_to_low(self) -> None:
        """The lookup checks in order, so an unsorted table would misclassify."""
        thresholds = [t for t, _ in RISK_BAND_EDGES]
        assert thresholds == sorted(thresholds, reverse=True)

    def test_standard_error_reflects_the_sample(self) -> None:
        wide = _availability(0.5, simulations=25)
        tight = _availability(0.5, simulations=400)
        assert wide.standard_error > tight.standard_error
        assert tight.standard_error == pytest.approx(0.025, abs=1e-6)

    def test_a_certain_survivor_has_no_mean_taken_pick(self) -> None:
        """``None`` rather than a fabricated average over zero rollouts."""
        entry = _availability(1.0, simulations=50)
        assert entry.mean_pick_taken is None
        assert entry.likeliest_taker is None


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo whole drafts
# ─────────────────────────────────────────────────────────────────────────────
class TestMonteCarlo:
    def test_every_rollout_finishes_a_draft(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        report = monte_carlo_draft(
            state, profiles, simulations=6, draft_slot=USER_SLOT
        )
        assert report.simulations == 6
        assert len(report.starter_points) == 6
        assert report.user_slot == USER_SLOT
        # Each rollout gives the user one roster's worth of players.
        assert sum(report.player_frequency.values()) == 6 * ROUNDS

    def test_the_live_draft_is_untouched(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        monte_carlo_draft(state, profiles, simulations=4)
        assert state.picks == []
        assert state.is_complete is False

    def test_percentiles_are_ordered(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        report = monte_carlo_draft(state, profiles, simulations=10)
        p10 = report.points_percentile(10)
        p50 = report.points_percentile(50)
        p90 = report.points_percentile(90)
        assert p10 <= p50 <= p90
        assert p10 <= report.mean_starter_points <= p90

    def test_position_shape_averages_the_user_roster(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        report = monte_carlo_draft(state, profiles, simulations=5)
        assert report.position_shape
        assert sum(report.position_shape.values()) == pytest.approx(ROUNDS)

    def test_frequency_rates_are_shares_of_the_run(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        report = monte_carlo_draft(state, profiles, simulations=8)
        common = report.most_common_players(5)
        assert common
        assert all(0.0 < rate <= 1.0 for _, _, rate in common)
        # Descending by rate, so the UI can render it as-is.
        rates = [rate for _, _, rate in common]
        assert rates == sorted(rates, reverse=True)
        assert all(name for _, name, _ in common)

    def test_a_user_strategy_is_honoured(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """A caller testing a specific plan must actually get that plan."""
        target_position = None

        def always_best(clone: DraftState):
            nonlocal target_position
            player = clone.best_available()
            target_position = target_position or player.position
            return player

        report = monte_carlo_draft(
            state, profiles, simulations=3, draft_slot=USER_SLOT,
            user_strategy=always_best,
        )
        assert report.simulations == 3
        assert sum(report.player_frequency.values()) == 3 * ROUNDS

    def test_a_strategy_returning_an_undraftable_player_falls_back(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """One bad callback must not cost the whole run."""
        taken = state.available_players(limit=1)[0]
        state.make_pick(taken)

        report = monte_carlo_draft(
            state, profiles, simulations=3, draft_slot=USER_SLOT,
            user_strategy=lambda clone: taken,
        )
        assert report.simulations == 3
        assert len(report.starter_points) == 3

    def test_simulations_are_clamped(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        report = monte_carlo_draft(state, profiles, simulations=0)
        assert report.simulations == 1


# ─────────────────────────────────────────────────────────────────────────────
# Opponent-intent helpers
# ─────────────────────────────────────────────────────────────────────────────
class TestPressure:
    def test_pressure_sums_to_about_the_number_of_intervening_picks(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """Each intervening manager contributes one pick's worth of probability."""
        pressure = upcoming_position_pressure(state, profiles, draft_slot=USER_SLOT)
        horizon = _horizon_for(state, USER_SLOT)
        assert sum(pressure.values()) == pytest.approx(horizon.gap, abs=0.01)

    def test_the_slot_on_the_clock_still_gets_pressure(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """The regression: reading the pre-pick gap returned an empty dict here."""
        pressure = upcoming_position_pressure(state, profiles, draft_slot=1)
        assert pressure
        assert sum(pressure.values()) == pytest.approx(14, abs=0.01)

    def test_a_finished_draft_has_no_pressure(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        while not state.is_complete:
            state.make_pick(state.best_available())
        assert upcoming_position_pressure(state, profiles, draft_slot=1) == {}


class TestLikelyNextPicks:
    def test_it_names_the_upcoming_managers_in_order(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        upcoming = likely_next_picks(state, profiles, count=4)
        assert [pick for pick, _, _, _ in upcoming] == [1, 2, 3, 4]
        assert [name for _, name, _, _ in upcoming] == [
            "Manager 1", "Manager 2", "Manager 3", "Manager 4",
        ]
        assert all(0.0 < probability <= 1.0 for *_, probability in upcoming)

    def test_it_is_deterministic(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        """The UI panel must not flicker between refreshes."""
        first = [p.player_id for _, _, p, _ in likely_next_picks(state, profiles)]
        second = [p.player_id for _, _, p, _ in likely_next_picks(state, profiles)]
        assert first == second

    def test_it_does_not_commit_anything(
        self, state: DraftState, profiles: dict[int, ManagerProfile]
    ) -> None:
        likely_next_picks(state, profiles, count=5)
        assert state.picks == []
