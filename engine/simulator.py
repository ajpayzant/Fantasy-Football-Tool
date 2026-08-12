"""Draft simulation: drive AI picks, roll the board forward, run Monte Carlo.

Three layers, deliberately separated because they answer different questions and
have very different costs:

1. :class:`DraftSimulator` — advances a real :class:`~engine.draft_state.DraftState`
   one AI pick at a time. This is the thing the interactive UI drives: it never
   guesses, it commits, and it records why each pick happened.
2. :func:`simulate_availability` — "will he still be there at my next pick?"
   answered by rolling *copies* of the board forward through the same pick model
   and counting. This replaces the closed-form ADP approximation in
   :func:`engine.pick_model.expected_survival` with the honest simulated answer.
3. :func:`monte_carlo_draft` — "how do whole drafts from here tend to go?" Many
   complete drafts from the current state, summarised.

Nothing here imports Streamlit or touches a database. The UI consumes the result
dataclasses; it does not participate in producing them.

**Why rollouts use copies rather than undo.** ``DraftState.undo`` exists and
works, but a rollout that mutated the live state and unwound it would leave the
user's undo stack and RNG stream perturbed on every recommendation refresh.
:meth:`~engine.draft_state.DraftState.copy_for_simulation` is the supported path
and costs a roster clone, not a pool copy.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from core.config import SimulationConfig
from core.enums import DraftStatus, Position, RiskBand
from core.validation import ConfigurationError
from engine.draft_state import DraftState
from engine.pick_model import (
    ScoredCandidate,
    choose_player,
    context_for,
    pick_probabilities,
    position_probabilities,
    score_candidates,
)
from models.draft import Pick
from models.manager import ManagerProfile
from models.player import Player

LOGGER = logging.getLogger("fantasy_mock_draft.simulator")

# ─────────────────────────────────────────────────────────────────────────────
# Tuning constants. Everything a caller might reasonably want to change lives in
# SimulationConfig; these are structural choices, documented where they are not
# self-evident.
# ─────────────────────────────────────────────────────────────────────────────
ALTERNATIVES_RECORDED: int = 4
"""How many runner-up candidates each AI pick stores for the explainability panel."""

RISK_BAND_EDGES: tuple[tuple[float, RiskBand], ...] = (
    (0.85, RiskBand.SAFE),
    (0.60, RiskBand.LIKELY_AVAILABLE),
    (0.40, RiskBand.COIN_FLIP),
    (0.15, RiskBand.LIKELY_GONE),
)
"""Survival probability → risk band, checked high to low; below the last is GONE.

The bands are deliberately asymmetric around 0.5. "Coin flip" is a genuinely
narrow window (0.40–0.60) because the whole point of the label is to tell a user
when they cannot rely on a player lasting — stretching it to 0.3–0.7 would call
a 2-in-3 favourite a coin flip and make the label useless for deciding whether to
wait a round.
"""

AVAILABILITY_TRACK_LIMIT: int = 80
"""Players whose survival is tracked per rollout.

Survival is counted for the top N of the board rather than all ~300 available:
past the top 80 at any given pick the answer is 1.0 for everyone and the
book-keeping is pure cost. Callers asking about a specific deeper player get it
via ``extra_players``.
"""

MIN_SIMULATIONS: int = 1
MAX_SIMULATIONS: int = 5_000
"""Guard rails. The ceiling exists so a bad config value cannot hang the UI."""


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class SimulatedPick:
    """One AI pick, with the model's reasoning attached.

    Wraps rather than replaces :class:`models.draft.Pick`: the committed pick is
    the record of what happened, while this carries the decision context the UI
    renders and the draft log explains from.
    """

    pick: Pick
    player: Player
    probability: float = 0.0
    """Softmax probability the model gave this player at this pick."""
    alternatives: list[tuple[Player, float]] = field(default_factory=list)
    """Runner-up candidates and their probabilities, most likely first."""
    components: dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    was_forced: bool = False
    """True when only one candidate was available, so the model had no choice."""

    @property
    def overall_pick(self) -> int:
        return self.pick.overall_pick

    @property
    def summary(self) -> str:
        who = f"{self.player.name} ({self.player.position})"
        if self.was_forced:
            return f"{self.pick.label} {self.pick.manager_name}: {who} — only option left"
        return (
            f"{self.pick.label} {self.pick.manager_name}: {who} "
            f"at {self.probability:.0%} — {self.explanation}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_pick": self.pick.overall_pick,
            "manager_name": self.pick.manager_name,
            "player_id": self.player.player_id,
            "player_name": self.player.name,
            "position": str(self.player.position),
            "probability": round(float(self.probability), 4),
            "was_forced": self.was_forced,
            "explanation": self.explanation,
            "alternatives": [
                {"player_name": p.name, "position": str(p.position),
                 "probability": round(float(q), 4)}
                for p, q in self.alternatives
            ],
        }


@dataclass(slots=True)
class PlayerAvailability:
    """Simulated odds one player survives to a given pick."""

    player: Player
    survival: float
    """Share of rollouts in which the player was still on the board."""
    simulations: int
    picks_until_next: int
    target_pick: int
    mean_pick_taken: float | None = None
    """Average overall pick at which he went, across rollouts where he went.

    ``None`` when he survived every rollout — there is no honest average of an
    empty set, and returning the target pick would read as "he goes right at your
    turn" when the truth is "he never went at all".
    """
    taken_by: dict[str, int] = field(default_factory=dict)
    """Manager name → how many rollouts they took him in."""

    @property
    def player_id(self) -> str:
        return self.player.player_id

    @property
    def risk_band(self) -> RiskBand:
        for threshold, band in RISK_BAND_EDGES:
            if self.survival >= threshold:
                return band
        return RiskBand.GONE

    @property
    def standard_error(self) -> float:
        """Binomial standard error on the survival estimate.

        Reported rather than hidden: at the default 120 rollouts a 0.50 estimate
        carries an SE of 0.046, so a UI that presents survival to the percent is
        overstating what the simulation knows.
        """
        if self.simulations <= 0:
            return 0.0
        p = min(1.0, max(0.0, self.survival))
        return float(math.sqrt(max(0.0, p * (1.0 - p)) / self.simulations))

    @property
    def likeliest_taker(self) -> str | None:
        if not self.taken_by:
            return None
        return max(self.taken_by.items(), key=lambda kv: kv[1])[0]

    def describe(self) -> str:
        pct = f"{self.survival:.0%}"
        if self.survival >= 0.99:
            return f"{self.player.name} lasted to pick {self.target_pick} in every rollout"
        taker = self.likeliest_taker
        tail = f"; most often taken by {taker}" if taker else ""
        return (
            f"{self.player.name}: {pct} chance he is there at pick "
            f"{self.target_pick} (±{self.standard_error:.0%}){tail}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player.player_id,
            "player_name": self.player.name,
            "position": str(self.player.position),
            "survival": round(float(self.survival), 4),
            "standard_error": round(self.standard_error, 4),
            "risk_band": str(self.risk_band),
            "simulations": self.simulations,
            "target_pick": self.target_pick,
            "picks_until_next": self.picks_until_next,
            "mean_pick_taken": (
                round(float(self.mean_pick_taken), 2)
                if self.mean_pick_taken is not None else None
            ),
            "likeliest_taker": self.likeliest_taker,
        }


@dataclass(slots=True)
class AvailabilityReport:
    """Survival odds for every tracked player, plus what the rollouts cost."""

    players: dict[str, PlayerAvailability] = field(default_factory=dict)
    simulations: int = 0
    picks_until_next: int = 0
    target_pick: int = 0
    from_pick: int = 0
    elapsed_seconds: float = 0.0
    position_gone: dict[Position, float] = field(default_factory=dict)
    """Position → mean share of the *tracked* pool at that position taken."""

    def get(self, player_id: str) -> PlayerAvailability | None:
        return self.players.get(player_id)

    def survival(self, player_id: str, default: float = 1.0) -> float:
        """Survival for one player, defaulting to ``default`` when untracked.

        Defaults to 1.0 rather than 0.0 on purpose: an untracked player is one
        deep enough on the board that nobody is taking him, so "he will be there"
        is the right answer, and a 0.0 default would make the recommendation
        engine frantically reach for irrelevant players.
        """
        entry = self.players.get(player_id)
        return float(entry.survival) if entry else float(default)

    def band(self, player_id: str) -> RiskBand:
        entry = self.players.get(player_id)
        return entry.risk_band if entry else RiskBand.SAFE

    def at_risk(self, threshold: float = 0.60) -> list[PlayerAvailability]:
        """Tracked players unlikely to last, least likely first."""
        risky = [a for a in self.players.values() if a.survival < threshold]
        risky.sort(key=lambda a: a.survival)
        return risky

    def safest(self, threshold: float = 0.85) -> list[PlayerAvailability]:
        safe = [a for a in self.players.values() if a.survival >= threshold]
        safe.sort(key=lambda a: -a.survival)
        return safe

    def to_frame(self):
        """Tabulate for the UI. Imported locally so the engine stays UI-free."""
        import pandas as pd

        rows = [a.to_dict() for a in self.players.values()]
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values("survival").reset_index(drop=True)
        return frame


@dataclass(slots=True)
class MonteCarloReport:
    """Aggregate of many complete simulated drafts."""

    simulations: int = 0
    from_pick: int = 0
    elapsed_seconds: float = 0.0
    user_slot: int | None = None
    player_frequency: dict[str, int] = field(default_factory=dict)
    """Player id → rollouts in which the *user* ended up with them."""
    position_shape: dict[Position, float] = field(default_factory=dict)
    """Position → mean count on the user's roster."""
    starter_points: list[float] = field(default_factory=list)
    """The user's summed starter projection, one entry per rollout."""
    open_starter_counts: Counter[int] = field(default_factory=Counter)
    """Unfilled starting seats → how many rollouts finished that way."""
    names: dict[str, str] = field(default_factory=dict)
    """Player id → name, so a report survives without the pool."""

    @property
    def mean_starter_points(self) -> float:
        return (
            sum(self.starter_points) / len(self.starter_points)
            if self.starter_points else 0.0
        )

    def points_percentile(self, percentile: float) -> float:
        """A percentile of the user's outcome distribution (0-100)."""
        if not self.starter_points:
            return 0.0
        ordered = sorted(self.starter_points)
        if len(ordered) == 1:
            return ordered[0]
        position = (min(100.0, max(0.0, percentile)) / 100.0) * (len(ordered) - 1)
        low = int(math.floor(position))
        high = min(len(ordered) - 1, low + 1)
        weight = position - low
        return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)

    def most_common_players(self, count: int = 15) -> list[tuple[str, str, float]]:
        """(player id, name, rate) for the players the user lands most often."""
        if not self.simulations:
            return []
        ranked = sorted(self.player_frequency.items(), key=lambda kv: -kv[1])
        return [
            (pid, self.names.get(pid, pid), n / self.simulations)
            for pid, n in ranked[:count]
        ]

    @property
    def unfilled_starter_rate(self) -> float:
        """Share of rollouts that finished with at least one empty starting seat."""
        if not self.simulations:
            return 0.0
        bad = sum(n for seats, n in self.open_starter_counts.items() if seats > 0)
        return bad / self.simulations

    def to_frame(self):
        import pandas as pd

        rows = [
            {"player_id": pid, "player_name": name, "rate": round(rate, 4)}
            for pid, name, rate in self.most_common_players(count=200)
        ]
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# The interactive simulator
# ─────────────────────────────────────────────────────────────────────────────
class DraftSimulator:
    """Advances a live draft through the pick model, one AI pick at a time.

    Holds no state of its own beyond the profiles and an RNG: the draft lives in
    :class:`~engine.draft_state.DraftState`, which is what the UI persists and
    what undo operates on. That separation is what lets a user undo an AI pick
    and get a *different* pick on the redo — the simulator is not caching a plan.
    """

    __slots__ = ("state", "profiles", "rng", "_log")

    def __init__(
        self,
        state: DraftState,
        profiles: Mapping[int, ManagerProfile],
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.state = state
        self.profiles = dict(profiles)
        self.rng = rng or state.rng
        self._log: list[SimulatedPick] = []

    # -- profiles --------------------------------------------------------
    def profile_for(self, draft_slot: int) -> ManagerProfile:
        """The profile for a slot, or a loud failure.

        Deliberately not a silent baseline fallback: a missing profile means the
        league and the profile map disagree, and simulating a whole draft with a
        generic drafter standing in for a real manager would quietly invalidate
        every result the user is about to read.
        """
        profile = self.profiles.get(int(draft_slot))
        if profile is None:
            raise ConfigurationError(
                f"No manager profile for draft slot {draft_slot}. Build profiles "
                "for every slot in the league before simulating."
            )
        return profile

    @property
    def log(self) -> list[SimulatedPick]:
        """The AI picks this simulator has made, oldest first (a copy)."""
        return list(self._log)

    # -- single picks ----------------------------------------------------
    def preview_pick(self, draft_slot: int | None = None) -> list[ScoredCandidate]:
        """Score and rank the current pick's candidates *without* committing.

        Used by the "what will he do?" panel and by the recommendation engine's
        opponent-threat estimate. Probabilities are filled in, so the caller sees
        the same distribution the sampler would draw from.
        """
        slot = self.state.current_slot
        if slot is None:
            return []
        target = int(draft_slot if draft_slot is not None else slot.draft_slot)
        profile = self.profile_for(target)
        context = context_for(self.state, profile, draft_slot=target, rng=self.rng)
        ranked = score_candidates(context)
        temperature = self.state.settings.temperature_for(
            float(profile.get("predictability"))
        )
        return pick_probabilities(ranked, temperature)

    def simulate_pick(self) -> SimulatedPick | None:
        """Make the pick on the clock as its manager would, and commit it.

        Returns ``None`` when the draft is complete or nobody is draftable.
        """
        slot = self.state.current_slot
        if slot is None:
            return None
        profile = self.profile_for(slot.draft_slot)
        context = context_for(self.state, profile, rng=self.rng)
        ranked = score_candidates(context)
        if not ranked:
            LOGGER.warning(
                "No draftable candidates at pick %s; the pool is exhausted", slot.label
            )
            return None
        chosen = choose_player(context, scored=ranked)
        if chosen is None:
            return None

        alternatives = [c for c in ranked if c.player_id != chosen.player_id]
        alternatives.sort(key=lambda c: -c.probability)
        top_alternatives = alternatives[:ALTERNATIVES_RECORDED]
        explanation = chosen.explain()
        pick = self.state.make_pick(
            chosen.player,
            pick_probability=chosen.probability,
            alternatives=[
                {"player_id": c.player_id, "player_name": c.player.name,
                 "position": str(c.player.position),
                 "probability": round(float(c.probability), 4)}
                for c in top_alternatives
            ],
            explanation=explanation,
        )
        simulated = SimulatedPick(
            pick=pick,
            player=chosen.player,
            probability=float(chosen.probability),
            alternatives=[(c.player, float(c.probability)) for c in top_alternatives],
            components=dict(chosen.components),
            explanation=explanation,
            was_forced=len(ranked) == 1,
        )
        self._log.append(simulated)
        LOGGER.debug("Simulated %s", simulated.summary)
        return simulated

    def simulate_until_user(self, max_picks: int | None = None) -> list[SimulatedPick]:
        """Run AI picks until a user slot is on the clock or the draft ends.

        ``max_picks`` bounds the run so a league with no user slot — a fully
        simulated draft watched from outside — cannot loop past the board.
        """
        made: list[SimulatedPick] = []
        limit = max_picks if max_picks is not None else len(self.state.order)
        while len(made) < limit:
            if self.state.is_complete or self.state.is_user_on_clock:
                break
            result = self.simulate_pick()
            if result is None:
                break
            made.append(result)
        return made

    def simulate_to_completion(self) -> list[SimulatedPick]:
        """Finish the draft, AI-picking on the user's behalf too.

        This is the "instant draft" mode. It deliberately does not consult the
        user: a caller wanting to stop at the user's turn wants
        :meth:`simulate_until_user`.
        """
        made: list[SimulatedPick] = []
        guard = len(self.state.order) + 1
        while not self.state.is_complete and len(made) < guard:
            result = self.simulate_pick()
            if result is None:
                break
            made.append(result)
        if self.state.is_complete:
            self.state.status = DraftStatus.COMPLETE
        return made


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo availability
# ─────────────────────────────────────────────────────────────────────────────
def _clamp_simulations(requested: int) -> int:
    return int(min(MAX_SIMULATIONS, max(MIN_SIMULATIONS, int(requested))))


@dataclass(frozen=True, slots=True)
class _Horizon:
    """The stretch of board between now and a slot's next meaningful turn.

    Exists so the availability rollouts, the pressure estimate and the UI all
    derive the same window from one rule. When they each computed it inline they
    disagreed: whoever is on the clock has ``picks_until_turn == 0``, so a naive
    reading concluded "no gap" and reported every player as certain to survive.
    """

    from_pick: int
    target_pick: int
    on_clock: bool
    """True when the slot being asked about is the one currently picking."""

    @property
    def rollout_picks(self) -> int:
        """Selections needed to bring ``target_pick`` to the clock."""
        return max(0, self.target_pick - self.from_pick)

    @property
    def gap(self) -> int:
        """Picks by *other* managers before the target pick — the honest wait."""
        return max(0, self.rollout_picks - (1 if self.on_clock else 0))

    @property
    def is_empty(self) -> bool:
        return self.rollout_picks <= 0


def _horizon_for(state: DraftState, draft_slot: int) -> _Horizon | None:
    """Resolve the wait a slot faces, or ``None`` when it has no picks left.

    For the slot on the clock this skips the pick they are deciding right now and
    measures to the one after it: "will he last until I pick again" is the only
    version of the question with a non-trivial answer.
    """
    current = state.current_slot
    if current is None:
        return None
    upcoming = state.next_pick_numbers(int(draft_slot), count=2)
    on_clock = current.draft_slot == int(draft_slot)
    horizon = upcoming[1:] if on_clock else upcoming
    if not horizon:
        return None
    return _Horizon(
        from_pick=int(current.overall_pick),
        target_pick=int(horizon[0]),
        on_clock=on_clock,
    )


def _roll_forward(
    state: DraftState,
    profiles: Mapping[int, ManagerProfile],
    picks: int,
    rng: random.Random,
    *,
    stop_at_slot: int | None = None,
    skip_stops: int = 0,
) -> DraftState:
    """Advance a *copy* of ``state`` by up to ``picks`` AI picks.

    Stops early at the draft's end, when nobody is draftable, or when
    ``stop_at_slot`` reaches the clock — the last is what makes an availability
    rollout stop at the user's turn instead of drafting on their behalf.

    ``skip_stops`` passes *through* that slot's turn that many times before
    stopping. It exists for the on-the-clock case: to learn what survives to the
    user's **next** pick, the rollout has to get past the pick they are currently
    deciding, which means letting the model take a player on their behalf. That
    selection is excluded when survival is counted, so it cannot bias the answer.
    """
    clone = state.copy_for_simulation()
    remaining_skips = max(0, int(skip_stops))
    for _ in range(max(0, int(picks))):
        slot = clone.current_slot
        if slot is None:
            break
        if stop_at_slot is not None and slot.draft_slot == int(stop_at_slot):
            if remaining_skips <= 0:
                break
            remaining_skips -= 1
        profile = profiles.get(slot.draft_slot)
        if profile is None:
            raise ConfigurationError(
                f"No manager profile for draft slot {slot.draft_slot}"
            )
        context = context_for(clone, profile, rng=rng)
        chosen = choose_player(context)
        if chosen is None:
            break
        clone.make_pick(chosen.player, is_user_pick=False)
    return clone


def simulate_availability(
    state: DraftState,
    profiles: Mapping[int, ManagerProfile],
    *,
    draft_slot: int | None = None,
    simulations: int | None = None,
    extra_players: Iterable[Player] | None = None,
    track_limit: int = AVAILABILITY_TRACK_LIMIT,
    seed: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> AvailabilityReport:
    """Simulate the picks between now and ``draft_slot``'s next turn.

    This is the honest counterpart to :func:`engine.pick_model.expected_survival`.
    The closed form asks only "where is his ADP relative to the gap", which cannot
    know that the three managers picking before you all need a tight end and one
    of them reaches. Rolling the actual pick model forward can, because it runs
    the same opponent models that will really be making those picks.

    Returns survival for the top ``track_limit`` players on the board, plus any
    ``extra_players`` the caller names explicitly (a deep sleeper the user is
    considering, which would otherwise fall outside the tracked window).

    ``progress`` is called as ``(completed, total)`` after each rollout so a UI
    can show a bar; it is never called from a thread.
    """
    started = time.perf_counter()
    settings = state.settings
    runs = _clamp_simulations(
        simulations if simulations is not None else settings.availability_simulations
    )

    current = state.current_slot
    if current is None:
        return AvailabilityReport(simulations=0, elapsed_seconds=0.0)
    target_slot = int(
        draft_slot if draft_slot is not None
        else (next(iter(sorted(state.league.user_slots)), current.draft_slot))
    )
    from_pick = int(current.overall_pick)
    horizon = _horizon_for(state, target_slot)
    if horizon is None or horizon.is_empty:
        # No further picks for this slot: nothing left to wait for.
        return AvailabilityReport(
            simulations=0, picks_until_next=0, target_pick=from_pick,
            from_pick=from_pick, elapsed_seconds=time.perf_counter() - started,
        )
    target_pick = horizon.target_pick
    gap = horizon.gap
    on_clock = horizon.on_clock

    tracked: dict[str, Player] = {
        p.player_id: p for p in state.available_players(limit=max(1, int(track_limit)))
    }
    for player in extra_players or ():
        if state.is_available(player.player_id):
            tracked.setdefault(player.player_id, player)

    if gap <= 0 or not tracked:
        # Already on the clock: everyone available survives by definition. Report
        # that rather than running rollouts that cannot change anything.
        return AvailabilityReport(
            players={
                pid: PlayerAvailability(
                    player=player, survival=1.0, simulations=0,
                    picks_until_next=0, target_pick=target_pick,
                )
                for pid, player in tracked.items()
            },
            simulations=0,
            picks_until_next=0,
            target_pick=target_pick,
            from_pick=from_pick,
            elapsed_seconds=time.perf_counter() - started,
        )

    rng = random.Random(seed if seed is not None else state.rng.random())
    survived: Counter[str] = Counter()
    taken_pick_total: Counter[str] = Counter()
    taken_count: Counter[str] = Counter()
    taken_by: dict[str, Counter[str]] = {}
    position_taken: dict[Position, int] = {}
    position_pool: dict[Position, int] = {}
    for player in tracked.values():
        position_pool[player.position] = position_pool.get(player.position, 0) + 1

    for run_index in range(runs):
        clone = _roll_forward(
            state, profiles, horizon.rollout_picks, rng,
            stop_at_slot=target_slot,
            skip_stops=1 if on_clock else 0,
        )
        # Which tracked players went, and to whom. Only picks made *during* this
        # rollout count, so picks already on the board are not attributed to it.
        gone: dict[str, Pick] = {}
        for pick in clone.picks[len(state.picks):]:
            # The stand-in pick made on the user's own behalf is not competition
            # for them — counting it would report every player the model likes as
            # unlikely to survive, when the user is free to simply take him now.
            if on_clock and pick.draft_slot == target_slot:
                continue
            if pick.player_id in tracked:
                gone[pick.player_id] = pick
        for pid, player in tracked.items():
            pick = gone.get(pid)
            if pick is None:
                survived[pid] += 1
                continue
            taken_pick_total[pid] += int(pick.overall_pick)
            taken_count[pid] += 1
            taken_by.setdefault(pid, Counter())[pick.manager_name] += 1
            position_taken[player.position] = position_taken.get(player.position, 0) + 1
        if progress is not None:
            progress(run_index + 1, runs)

    players = {
        pid: PlayerAvailability(
            player=player,
            survival=survived[pid] / runs,
            simulations=runs,
            picks_until_next=int(gap),
            target_pick=target_pick,
            mean_pick_taken=(
                taken_pick_total[pid] / taken_count[pid] if taken_count[pid] else None
            ),
            taken_by=dict(taken_by.get(pid, {})),
        )
        for pid, player in tracked.items()
    }
    report = AvailabilityReport(
        players=players,
        simulations=runs,
        picks_until_next=int(gap),
        target_pick=target_pick,
        from_pick=from_pick,
        elapsed_seconds=time.perf_counter() - started,
        position_gone={
            position: position_taken.get(position, 0) / (runs * count)
            for position, count in position_pool.items() if count
        },
    )
    LOGGER.info(
        "Availability: %d rollouts over %d picks (slot %d, pick %d) in %.2fs",
        runs, gap, target_slot, target_pick, report.elapsed_seconds,
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo whole drafts
# ─────────────────────────────────────────────────────────────────────────────
def _starter_points(
    state: DraftState, draft_slot: int, projections: Mapping[str, float]
) -> float:
    """Summed projection of a slot's starting lineup.

    Rebuilds the lineup against ``projections`` first. That is not redundant with
    the rebuild :meth:`~models.draft.TeamRoster.add` already did: the in-draft
    rebuild runs without a projection map, so it fills seats legally but does not
    guarantee the *best* player got the flex seat. Scoring a roster is exactly
    when that ordering matters. The map is passed in rather than rebuilt here
    because it is identical for every rollout.
    """
    roster = state.roster(draft_slot)
    roster.rebuild(projections)
    return float(sum(projections.get(pid, 0.0) for pid in roster.starters()))


def monte_carlo_draft(
    state: DraftState,
    profiles: Mapping[int, ManagerProfile],
    *,
    simulations: int | None = None,
    draft_slot: int | None = None,
    user_strategy: Callable[[DraftState], Player | None] | None = None,
    seed: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> MonteCarloReport:
    """Run many complete drafts from the current state and summarise them.

    ``user_strategy`` decides the user's own picks. Defaults to the same pick
    model everyone else uses, driven by the user's profile — which is the right
    default because the question this answers is "how do drafts from here tend to
    go", not "how well would a perfect drafter do". A caller wanting to test a
    specific plan passes a strategy that implements it.

    Each rollout starts from a fresh copy, so the live draft is untouched.
    """
    started = time.perf_counter()
    settings: SimulationConfig = state.settings
    runs = _clamp_simulations(
        simulations if simulations is not None else settings.monte_carlo_default_runs
    )
    current = state.current_slot
    user_slot = (
        int(draft_slot) if draft_slot is not None
        else next(iter(sorted(state.league.user_slots)), None)
    )
    rng = random.Random(seed if seed is not None else state.rng.random())

    report = MonteCarloReport(
        simulations=runs,
        from_pick=int(current.overall_pick) if current else len(state.order),
        user_slot=user_slot,
    )
    position_totals: dict[Position, int] = {}
    # Built once: the pool is read-only for the whole run, and rebuilding it per
    # rollout would be a full pool scan × 200.
    projections = {
        p.player_id: float(p.projection or 0.0) for p in state.pool
    }

    for run_index in range(runs):
        clone = state.copy_for_simulation()
        guard = len(clone.order) + 1
        steps = 0
        while not clone.is_complete and steps < guard:
            steps += 1
            slot = clone.current_slot
            if slot is None:
                break
            if (
                user_strategy is not None
                and user_slot is not None
                and slot.draft_slot == user_slot
            ):
                chosen_player = user_strategy(clone)
                if chosen_player is None or not clone.is_available(
                    chosen_player.player_id
                ):
                    # A strategy that names an unavailable player is a caller bug,
                    # but failing the whole run would lose 199 good rollouts to
                    # one bad callback. Fall through to the model instead.
                    LOGGER.debug(
                        "user_strategy returned an undraftable player at %s; "
                        "using the pick model for this pick", slot.label,
                    )
                else:
                    clone.make_pick(chosen_player, is_user_pick=True)
                    continue
            profile = profiles.get(slot.draft_slot)
            if profile is None:
                raise ConfigurationError(
                    f"No manager profile for draft slot {slot.draft_slot}"
                )
            context = context_for(clone, profile, rng=rng)
            chosen = choose_player(context)
            if chosen is None:
                break
            clone.make_pick(chosen.player)

        if user_slot is not None:
            # The whole finished roster, not just picks this rollout added: the
            # question is "what does my team look like when drafts play out from
            # here", and players already banked are part of that team.
            for pick in clone.picks_by_slot(user_slot):
                if not pick.player_id:
                    continue
                report.player_frequency[pick.player_id] = (
                    report.player_frequency.get(pick.player_id, 0) + 1
                )
                report.names.setdefault(pick.player_id, pick.player_name)
                position_totals[pick.position] = position_totals.get(pick.position, 0) + 1
            report.starter_points.append(
                _starter_points(clone, user_slot, projections)
            )
            report.open_starter_counts[
                sum(clone.roster(user_slot).open_starting_slots().values())
            ] += 1
        if progress is not None:
            progress(run_index + 1, runs)

    report.position_shape = {
        position: total / runs for position, total in position_totals.items()
    }
    report.elapsed_seconds = time.perf_counter() - started
    LOGGER.info(
        "Monte Carlo: %d complete drafts from pick %d in %.2fs",
        runs, report.from_pick, report.elapsed_seconds,
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Opponent-intent helpers, used by the recommendation engine
# ─────────────────────────────────────────────────────────────────────────────
def upcoming_position_pressure(
    state: DraftState,
    profiles: Mapping[int, ManagerProfile],
    *,
    draft_slot: int | None = None,
) -> dict[Position, float]:
    """Expected number of each position taken before a slot's next turn.

    A single deterministic pass over the intervening managers' *distributions* —
    not a rollout — so it is cheap enough to call on every UI refresh. It answers
    "the four managers ahead of me are collectively about 2.3 running backs
    hungry", which is what makes a scarcity warning concrete.

    The window is the same one :func:`simulate_availability` uses, via
    :func:`_horizon_for`, so the scarcity warning and the survival percentages can
    never describe different stretches of the board.

    It deliberately does not decrement the board as it goes: two managers who
    both want the same running back each contribute their own probability, which
    slightly over-counts. The alternative — sequentially removing the likeliest
    pick — would make it a single deterministic rollout and lose the distribution
    entirely, which is a worse trade for a summary statistic.
    """
    current = state.current_slot
    if current is None:
        return {}
    target = int(
        draft_slot if draft_slot is not None
        else next(iter(sorted(state.league.user_slots)), current.draft_slot)
    )
    horizon = _horizon_for(state, target)
    if horizon is None or horizon.gap <= 0:
        return {}
    pressure: dict[Position, float] = {}
    index = state.pick_index
    for offset in range(horizon.rollout_picks):
        position_in_order = index + offset
        if position_in_order >= len(state.order):
            break
        slot = state.order[position_in_order]
        if slot.draft_slot == target:
            # The target's own picks are not competition for them: skip past the
            # one on the clock rather than stopping, so the managers between it and
            # their next turn are still counted.
            continue
        profile = profiles.get(slot.draft_slot)
        if profile is None:
            continue
        context = context_for(state, profile, draft_slot=slot.draft_slot)
        # Scored against the *current* board for every intervening pick, which is
        # the approximation this function exists to make: no rollout, one pass.
        ranked = score_candidates(context)
        if not ranked:
            continue
        temperature = state.settings.temperature_for(
            float(profile.get("predictability"))
        )
        pick_probabilities(ranked, temperature)
        for position, probability in position_probabilities(ranked).items():
            pressure[position] = pressure.get(position, 0.0) + float(probability)
    return pressure


def likely_next_picks(
    state: DraftState,
    profiles: Mapping[int, ManagerProfile],
    *,
    count: int = 5,
) -> list[tuple[int, str, Player, float]]:
    """The most likely pick for each of the next ``count`` slots.

    ``(overall pick, manager name, player, probability)``. Deterministic — it
    takes each manager's modal choice rather than sampling — so the UI's "coming
    up" panel does not flicker between refreshes.
    """
    out: list[tuple[int, str, Player, float]] = []
    index = state.pick_index
    for offset in range(max(0, int(count))):
        position_in_order = index + offset
        if position_in_order >= len(state.order):
            break
        slot = state.order[position_in_order]
        profile = profiles.get(slot.draft_slot)
        if profile is None:
            continue
        context = context_for(state, profile, draft_slot=slot.draft_slot)
        ranked = score_candidates(context)
        if not ranked:
            continue
        temperature = state.settings.temperature_for(
            float(profile.get("predictability"))
        )
        pick_probabilities(ranked, temperature)
        best = ranked[0]
        manager = state.league.manager_by_slot(slot.draft_slot)
        out.append((
            int(slot.overall_pick),
            manager.name if manager else f"Slot {slot.draft_slot}",
            best.player,
            float(best.probability),
        ))
    return out


__all__ = [
    "DraftSimulator", "SimulatedPick", "PlayerAvailability", "AvailabilityReport",
    "SimulatedDraftResult", "MonteCarloReport", "simulate_availability",
    "monte_carlo_draft", "upcoming_position_pressure", "likely_next_picks",
    "RISK_BAND_EDGES", "AVAILABILITY_TRACK_LIMIT", "ALTERNATIVES_RECORDED",
]
