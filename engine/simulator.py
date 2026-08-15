"""Draft simulation: drive AI picks, roll the board forward, run Monte Carlo.

Three layers, deliberately separated because they answer different questions and
have very different costs:

1. :class:`DraftSimulator` — advances a real :class:`~engine.draft_state.DraftState`
   one AI pick at a time. This is the thing the interactive UI drives: it never
   guesses, it commits, and it records why each pick happened.
2. :func:`simulate_draft_plan` — "what should be there at my next two picks, and
   who takes what in between?" answered by rolling *copies* of the board forward
   through the same pick model and counting. This replaces the closed-form ADP
   approximation in :func:`engine.pick_model.expected_survival` with the honest
   simulated answer. :func:`simulate_availability` is the one-turn case of it.
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

PLAN_TURNS: int = 2
"""How many of your own upcoming picks :func:`simulate_draft_plan` looks at.

Two, because that is the shortest look-ahead that changes a decision: with one
turn the only question is "take him or lose him", and it takes a second turn for
"take the receiver now because the tight end lasts" to be expressible. Each extra
turn roughly doubles the rollout cost and compounds the error — by the third turn
the board has been guessed twenty-odd picks deep and the answer is not worth
acting on.
"""

PLAN_ROOM_PLAYERS: int = 4
"""Named players kept per intervening pick.

A single pick's named-player distribution has a long tail of 2% guesses; past the
fourth the names are noise dressed as information.
"""

PLAN_GONE_BY: float = 0.35
"""Survival at or below which a plan calls a player gone by a turn.

The same edge as :data:`engine.recommender.LAST_CHANCE_SURVIVAL`, deliberately:
"now or never" has to mean one thing across the app, or the two panels of the
draft room argue with each other about the same player.
"""

PLAN_LASTS: float = 0.65
"""Survival at or above which a plan says a player will still be there.

Looser than :data:`engine.recommender.SAFE_TO_WAIT_SURVIVAL` (0.80) because it is
asked of a longer wait. Two turns out is twenty-odd picks in a 12-team league, and
demanding 80% there would leave the list permanently empty — which reads as "wait
for nobody" rather than the truth, "this is a two-in-three bet".
"""


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
class RoomPickForecast:
    """What one specific pick between now and your turn is likely to become.

    Aggregated from the same rollouts that produced the survival numbers, which is
    the point: the players disappearing in one panel are disappearing *because* of
    the picks in the other. Two independently simulated panels would let the app
    tell a user "the tight ends are safe" beside "this manager takes a tight end".
    """

    overall_pick: int
    round_label: str
    draft_slot: int
    manager_name: str
    simulations: int = 0
    before_turn: int = 1
    """Which of your turns this pick falls before (1 = your very next one)."""
    roster_so_far: str = ""
    """What they have already drafted, e.g. ``"2 RB · 1 WR"`` — empty in round 1."""
    tendency: str = ""
    """A short tag for how they draft, from their modelled profile."""
    position_shares: dict[Position, float] = field(default_factory=dict)
    """Position → share of rollouts in which they spent this pick on it."""
    player_shares: list[tuple[Player, float]] = field(default_factory=list)
    """The players they took here, most frequent first, with each one's share."""

    @property
    def likeliest_position(self) -> Position | None:
        if not self.position_shares:
            return None
        return max(self.position_shares.items(), key=lambda kv: kv[1])[0]

    @property
    def likeliest_player(self) -> Player | None:
        return self.player_shares[0][0] if self.player_shares else None

    @property
    def position_summary(self) -> str:
        """The positional distribution, biggest first — the honest headline.

        Named players at a single pick are usually a scatter of 3% guesses, while
        the position is often 60% certain. Leading with the position is leading with
        the part the simulation actually knows.
        """
        ordered = sorted(self.position_shares.items(), key=lambda kv: -kv[1])
        return " · ".join(
            f"{position} {share:.0%}" for position, share in ordered[:4] if share >= 0.05
        )

    def describe(self) -> str:
        position = self.likeliest_position
        if position is None:
            return f"Pick {self.overall_pick} ({self.manager_name}): no clear lean"
        share = self.position_shares.get(position, 0.0)
        player = self.likeliest_player
        tail = f", most often {player.name}" if player is not None else ""
        return (
            f"Pick {self.overall_pick} ({self.manager_name}): {position} in "
            f"{share:.0%} of rollouts{tail}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_pick": self.overall_pick,
            "round": self.round_label,
            "draft_slot": self.draft_slot,
            "manager_name": self.manager_name,
            "before_turn": self.before_turn,
            "roster_so_far": self.roster_so_far,
            "tendency": self.tendency,
            "likeliest_position": (
                str(self.likeliest_position) if self.likeliest_position else None
            ),
            "position_shares": {
                str(position): round(float(share), 4)
                for position, share in self.position_shares.items()
            },
            "likely_players": [
                {"player_name": player.name, "position": str(player.position),
                 "share": round(float(share), 4)}
                for player, share in self.player_shares
            ],
        }


@dataclass(slots=True)
class PlannedTurn:
    """One of your own upcoming picks, with what the board looks like when it lands."""

    turn: int
    """1 = your next pick, 2 = the one after it."""
    overall_pick: int
    round_label: str
    picks_until: int
    """Picks by *other* managers between now and this turn."""
    availability: AvailabilityReport

    @property
    def label(self) -> str:
        return f"{self.round_label} (pick {self.overall_pick})"

    def expected_best(
        self, *, minimum: float = 0.60, limit: int = 8
    ) -> list[PlayerAvailability]:
        """The best players on the board likely to reach this turn, best first.

        "Best" is board order, the same ordering the rest of the app uses, rather
        than "most likely to survive" — sorting by survival answers a different
        question and puts the deepest sleeper on the board at the top.
        """
        keep = [
            entry for entry in self.availability.players.values()
            if entry.survival >= minimum
        ]
        return keep[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "overall_pick": self.overall_pick,
            "round": self.round_label,
            "picks_until": self.picks_until,
            "simulations": self.availability.simulations,
        }


@dataclass(slots=True)
class PlanWindows:
    """The three groups a two-turn plan sorts the board into.

    This is the whole point of looking two turns ahead rather than one. With one
    turn the only question is "take him or lose him"; the second turn is what lets
    a plan say "take the receiver now, the tight end will still be here" — which is
    the sentence a drafter is actually trying to write.
    """

    take_now: list[PlayerAvailability] = field(default_factory=list)
    """On the board now and unlikely to reach your next turn."""
    next_turn: list[PlayerAvailability] = field(default_factory=list)
    """Likely to reach your next turn but not the one after it."""
    can_wait: list[PlayerAvailability] = field(default_factory=list)
    """Likely to still be there two turns from now."""

    @property
    def is_empty(self) -> bool:
        return not (self.take_now or self.next_turn or self.can_wait)


@dataclass(slots=True)
class DraftPlan:
    """Two turns of look-ahead from one shared set of rollouts.

    The question this answers is the one drafters actually ask: not "will he last
    to my next pick", but "which of these two can I get later, so I take the other
    one now". It has to come from a single simulation to be coherent — the second
    turn's numbers are conditional on the first turn's picks having happened.

    The stand-in pick the model makes on the user's behalf at the intervening turn
    is excluded from survival counting, exactly as it is for the on-the-clock case
    in :func:`simulate_availability`. So the second turn's survival reads as "the
    room leaves him alone that long", which is the only part the user does not
    control; whoever they actually take at the first turn is obviously gone.
    """

    draft_slot: int = 0
    from_pick: int = 0
    simulations: int = 0
    elapsed_seconds: float = 0.0
    turns: list[PlannedTurn] = field(default_factory=list)
    room: list[RoomPickForecast] = field(default_factory=list)
    """Every intervening pick by another manager, in board order."""

    @property
    def is_empty(self) -> bool:
        return not self.turns

    def turn(self, number: int) -> PlannedTurn | None:
        """Your ``number``-th upcoming pick (1-based), or ``None``."""
        for planned in self.turns:
            if planned.turn == int(number):
                return planned
        return None

    @property
    def first_report(self) -> AvailabilityReport | None:
        """The next turn's availability, reusable by the recommendation engine."""
        return self.turns[0].availability if self.turns else None

    def survival(self, player_id: str, turn: int = 1, default: float = 1.0) -> float:
        planned = self.turn(turn)
        return planned.availability.survival(player_id, default) if planned else default

    def room_before(self, turn: int) -> list[RoomPickForecast]:
        return [entry for entry in self.room if entry.before_turn == int(turn)]

    def windows(
        self,
        *,
        limit: int = 6,
        gone_by: float = PLAN_GONE_BY,
        lasts: float = PLAN_LASTS,
    ) -> PlanWindows:
        """Sort the board into take-now, next-turn and can-wait, best first.

        Board order is preserved and each group is capped at ``limit``, so what a
        user reads is "the best few players in each case" rather than a hundred deep
        sleepers who were never going anywhere.

        With only one turn to plan for, ``next_turn`` and ``can_wait`` collapse into
        one question and everything not at risk lands in ``can_wait``: a single-turn
        plan cannot honestly distinguish them, and inventing the distinction is
        worse than omitting it.
        """
        first = self.turn(1)
        if first is None:
            return PlanWindows()
        second = self.turn(2)
        windows = PlanWindows()
        for player_id, entry in first.availability.players.items():
            if entry.survival <= gone_by:
                if len(windows.take_now) < limit:
                    windows.take_now.append(entry)
                continue
            if second is None:
                if entry.survival >= lasts and len(windows.can_wait) < limit:
                    windows.can_wait.append(entry)
                continue
            later = second.availability.get(player_id)
            survival_later = later.survival if later is not None else 1.0
            if survival_later <= gone_by:
                if len(windows.next_turn) < limit:
                    windows.next_turn.append(later or entry)
            elif survival_later >= lasts and len(windows.can_wait) < limit:
                windows.can_wait.append(later or entry)
        return windows

    def to_frame(self):
        """One row per tracked player, one survival column per turn.

        Board order is preserved, so the first rows are the best players left and
        the table reads top-down as "here is what should still be there".
        """
        import pandas as pd

        first = self.turns[0] if self.turns else None
        if first is None:
            return pd.DataFrame()
        rows = []
        for player_id, entry in first.availability.players.items():
            row: dict[str, Any] = {
                "Player": entry.player.name,
                "Pos": str(entry.player.position),
                "Team": entry.player.nfl_team or "FA",
                "ADP": entry.player.overall_adp,
            }
            for planned in self.turns:
                own = planned.availability.get(player_id)
                row[f"Pick {planned.overall_pick}"] = (
                    float(own.survival) if own is not None else 1.0
                )
            row["Most likely taken by"] = entry.likeliest_taker or "—"
            rows.append(row)
        return pd.DataFrame(rows)

    def room_frame(self):
        import pandas as pd

        # Name the turn each pick comes before rather than its index — "before your
        # 4.07 (pick 43)" is readable on its own, "before your 1" is not.
        labels = {planned.turn: planned.label for planned in self.turns}
        return pd.DataFrame([
            {
                "Pick": entry.overall_pick,
                "Round": entry.round_label,
                "Manager": entry.manager_name,
                "Likely position": entry.position_summary or "—",
                "Most likely player": (
                    entry.likeliest_player.name
                    if entry.likeliest_player is not None else "—"
                ),
                "Odds": (
                    entry.player_shares[0][1] if entry.player_shares else None
                ),
                "Roster so far": entry.roster_so_far or "—",
                "How they draft": entry.tendency or "—",
                "Before your pick": labels.get(
                    entry.before_turn, f"#{entry.before_turn}"
                ),
            }
            for entry in self.room
        ])

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_slot": self.draft_slot,
            "from_pick": self.from_pick,
            "simulations": self.simulations,
            "elapsed_seconds": round(float(self.elapsed_seconds), 3),
            "turns": [planned.to_dict() for planned in self.turns],
            "room": [entry.to_dict() for entry in self.room],
        }


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
            float(profile.get("predictability")), context.round_number
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
    turns_ahead: int = 1
    """Which of the slot's own turns this measures to (1 = its next one)."""

    @property
    def rollout_picks(self) -> int:
        """Selections needed to bring ``target_pick`` to the clock."""
        return max(0, self.target_pick - self.from_pick)

    @property
    def own_turns_passed(self) -> int:
        """The slot's *own* turns the rollout must play through to get here.

        One for the pick it is deciding right now, plus one for each earlier turn
        of its own between then and the target. These are the picks a rollout has
        to make on the slot's behalf, and they are the picks excluded when
        survival is counted.
        """
        return (1 if self.on_clock else 0) + max(0, int(self.turns_ahead) - 1)

    @property
    def gap(self) -> int:
        """Picks by *other* managers before the target pick — the honest wait."""
        return max(0, self.rollout_picks - self.own_turns_passed)

    @property
    def is_empty(self) -> bool:
        return self.rollout_picks <= 0


def _horizon_for(
    state: DraftState, draft_slot: int, *, turns_ahead: int = 1
) -> _Horizon | None:
    """Resolve the wait a slot faces, or ``None`` when it has no picks left.

    For the slot on the clock this skips the pick they are deciding right now and
    measures to the one after it: "will he last until I pick again" is the only
    version of the question with a non-trivial answer.

    ``turns_ahead`` looks further down the board: 2 measures to the turn *after*
    the next one, which is what plans a pair of picks together ("take the receiver
    now, the tight end lasts"). ``None`` once the slot runs out of turns, so a
    caller asking for two horizons late in the draft gets one.
    """
    current = state.current_slot
    if current is None:
        return None
    wanted = max(1, int(turns_ahead))
    upcoming = state.next_pick_numbers(int(draft_slot), count=wanted + 1)
    on_clock = current.draft_slot == int(draft_slot)
    horizon = upcoming[1:] if on_clock else upcoming
    if len(horizon) < wanted:
        return None
    return _Horizon(
        from_pick=int(current.overall_pick),
        target_pick=int(horizon[wanted - 1]),
        on_clock=on_clock,
        turns_ahead=wanted,
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
    A two-turn plan uses it the same way for the turn in between.
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


def _gone_from(
    picks: Iterable[Pick], tracked: Mapping[str, Player], own_slot: int
) -> dict[str, Pick]:
    """Tracked players taken during a rollout, keyed by player id.

    Picks made by ``own_slot`` are ignored. Those are the stand-in selections the
    rollout makes on the user's behalf to get past their own turns, and they are not
    competition: counting them would report every player the model likes as unlikely
    to survive, when the user is free to simply take him.
    """
    gone: dict[str, Pick] = {}
    for pick in picks:
        if pick.draft_slot == int(own_slot):
            continue
        if pick.player_id in tracked:
            gone[pick.player_id] = pick
    return gone


def _roster_shorthand(state: DraftState, draft_slot: int) -> str:
    """What a slot has drafted so far, as ``"2 RB · 1 WR"``, most-drafted first.

    Half of "who will they take" is "what do they already have", and a plan that
    showed the prediction without the roster behind it would be asking the user to
    take the model's word for it.
    """
    counts: Counter[Position] = Counter(
        pick.position for pick in state.picks_by_slot(int(draft_slot))
    )
    if not counts:
        return ""
    return " · ".join(f"{count} {position}" for position, count in counts.most_common())


def _tendency_tag(profile: ManagerProfile | None) -> str:
    """A few words on how a manager drafts, sized for a table cell.

    :meth:`models.manager.ManagerProfile.describe` is the full paragraph and belongs
    on the Manager Profiles page. What a planning table needs is the two or three
    traits that explain the prediction sitting beside them.
    """
    if profile is None:
        return ""
    tags: list[str] = []
    lean = max(
        (
            (position, float(profile.position_bias.get(position, 0.0)))
            for position in (Position.RB, Position.WR, Position.TE, Position.QB)
        ),
        key=lambda item: abs(item[1]),
        default=None,
    )
    if lean is not None and abs(lean[1]) >= 0.08:
        tags.append(f"{lean[0]} {'early' if lean[1] > 0 else 'late'}")
    reach = profile.reach_mean
    if reach >= 3:
        tags.append(f"reaches ~{reach:.0f}")
    elif reach <= -3:
        tags.append(f"waits ~{abs(reach):.0f}")
    else:
        tags.append("near ADP")
    if profile.predictability >= 0.70:
        tags.append("predictable")
    elif profile.predictability <= 0.32:
        tags.append("erratic")
    return " · ".join(tags)


def simulate_draft_plan(
    state: DraftState,
    profiles: Mapping[int, ManagerProfile],
    *,
    draft_slot: int | None = None,
    turns: int = PLAN_TURNS,
    simulations: int | None = None,
    extra_players: Iterable[Player] | None = None,
    track_limit: int = AVAILABILITY_TRACK_LIMIT,
    seed: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> DraftPlan:
    """Roll the board forward through ``draft_slot``'s next ``turns`` turns.

    The engine behind the two-pick plan, and the single source of every survival
    number in the app: :func:`simulate_availability` is this function with
    ``turns=1``. One implementation because the counting rule is subtle — which
    picks are competition, which are the user's own stand-ins, and what "gap" means
    for whoever is on the clock — and two copies of it would drift.

    Each rollout is run as one continuous draft, stopping at each of the slot's
    turns in order to snapshot the board before carrying on. That is what makes the
    second turn's numbers *conditional*: the managers picking between the two turns
    have seen the first turn happen, so a run on tight ends that the first pick
    triggers is priced into the second.

    Everything the caller asked about is derived from those same rollouts — survival
    per turn, and a per-pick forecast of what each intervening manager does — so no
    two panels of the UI can disagree about one simulated draft.

    ``progress`` is called as ``(completed, total)`` once per rollout, not once per
    turn, so a progress bar advances evenly; it is never called from a thread.
    """
    started = time.perf_counter()
    settings = state.settings
    runs = _clamp_simulations(
        simulations if simulations is not None else settings.availability_simulations
    )
    current = state.current_slot
    if current is None:
        return DraftPlan()
    target_slot = int(
        draft_slot if draft_slot is not None
        else (next(iter(sorted(state.league.user_slots)), current.draft_slot))
    )
    from_pick = int(current.overall_pick)
    plan = DraftPlan(draft_slot=target_slot, from_pick=from_pick)

    # Ask for turns one at a time and stop at the first one that does not exist:
    # in the last round there is no second turn to plan for, and reporting one
    # turn is the right answer rather than an error.
    horizons: list[_Horizon] = []
    for ahead in range(1, max(1, int(turns)) + 1):
        horizon = _horizon_for(state, target_slot, turns_ahead=ahead)
        if horizon is None or horizon.is_empty:
            break
        horizons.append(horizon)
    if not horizons:
        plan.elapsed_seconds = time.perf_counter() - started
        return plan

    tracked: dict[str, Player] = {
        p.player_id: p for p in state.available_players(limit=max(1, int(track_limit)))
    }
    for player in extra_players or ():
        if state.is_available(player.player_id):
            tracked.setdefault(player.player_id, player)
    position_pool: dict[Position, int] = {}
    for player in tracked.values():
        position_pool[player.position] = position_pool.get(player.position, 0) + 1

    # Book-keeping, index-aligned with ``horizons``: one set per turn.
    survived: list[Counter[str]] = [Counter() for _ in horizons]
    taken_pick_total: list[Counter[str]] = [Counter() for _ in horizons]
    taken_count: list[Counter[str]] = [Counter() for _ in horizons]
    taken_by: list[dict[str, Counter[str]]] = [{} for _ in horizons]
    position_taken: list[dict[Position, int]] = [{} for _ in horizons]
    # And one set per intervening pick, keyed by its overall pick number. The draft
    # order is fixed, so pick 34 is the same manager's pick in every rollout and the
    # counts across rollouts are comparable.
    room_positions: dict[int, Counter[Position]] = {}
    room_players: dict[int, Counter[str]] = {}
    room_managers: dict[int, tuple[int, str]] = {}

    # A slot that turns the snake picks again immediately, so nothing intervenes and
    # everyone available survives by definition. Report that rather than running
    # rollouts that cannot change the answer.
    live = any(horizon.gap > 0 for horizon in horizons)
    completed = runs if (live and tracked) else 0
    baseline = len(state.picks)

    if completed:
        rng = random.Random(seed if seed is not None else state.rng.random())
        for run_index in range(completed):
            clone = state
            leg_start = from_pick
            for index, horizon in enumerate(horizons):
                clone = _roll_forward(
                    clone, profiles, horizon.target_pick - leg_start, rng,
                    stop_at_slot=target_slot,
                    # Leg one has to get past the pick the user is deciding right
                    # now, if it is theirs. Every later leg starts *at* one of their
                    # turns and has to play through it.
                    skip_stops=(1 if horizon.on_clock else 0) if index == 0 else 1,
                )
                leg_start = horizon.target_pick
                # Survival to a later turn includes everyone taken in the earlier
                # legs, so each turn reads the whole rollout so far rather than its
                # own leg. Only picks made during this rollout count: picks already
                # on the board when it started are not attributed to it.
                gone = _gone_from(clone.picks[baseline:], tracked, target_slot)
                for pid, player in tracked.items():
                    pick = gone.get(pid)
                    if pick is None:
                        survived[index][pid] += 1
                        continue
                    taken_pick_total[index][pid] += int(pick.overall_pick)
                    taken_count[index][pid] += 1
                    taken_by[index].setdefault(pid, Counter())[pick.manager_name] += 1
                    position_taken[index][player.position] = (
                        position_taken[index].get(player.position, 0) + 1
                    )
            for pick in clone.picks[baseline:]:
                if pick.draft_slot == target_slot:
                    continue
                room_positions.setdefault(
                    pick.overall_pick, Counter()
                )[pick.position] += 1
                room_players.setdefault(
                    pick.overall_pick, Counter()
                )[pick.player_id] += 1
                room_managers[pick.overall_pick] = (pick.draft_slot, pick.manager_name)
            if progress is not None:
                progress(run_index + 1, completed)

    slots_by_pick = {slot.overall_pick: slot for slot in state.order}
    elapsed = time.perf_counter() - started
    for index, horizon in enumerate(horizons):
        # A zero-gap turn reports zero simulations because none were needed, not
        # because the answer is unknown: survival is 1.0 for everyone.
        counted = completed if horizon.gap > 0 else 0
        players = {
            pid: PlayerAvailability(
                player=player,
                survival=(survived[index][pid] / completed) if completed else 1.0,
                simulations=counted,
                picks_until_next=int(horizon.gap),
                target_pick=horizon.target_pick,
                mean_pick_taken=(
                    taken_pick_total[index][pid] / taken_count[index][pid]
                    if taken_count[index][pid] else None
                ),
                taken_by=dict(taken_by[index].get(pid, {})),
            )
            for pid, player in tracked.items()
        }
        target = slots_by_pick.get(horizon.target_pick)
        plan.turns.append(PlannedTurn(
            turn=index + 1,
            overall_pick=horizon.target_pick,
            round_label=target.label if target is not None else str(horizon.target_pick),
            picks_until=int(horizon.gap),
            availability=AvailabilityReport(
                players=players,
                simulations=counted,
                picks_until_next=int(horizon.gap),
                target_pick=horizon.target_pick,
                from_pick=from_pick,
                elapsed_seconds=elapsed,
                position_gone={
                    position: position_taken[index].get(position, 0) / (completed * count)
                    for position, count in position_pool.items() if count
                } if counted else {},
            ),
        ))

    for overall_pick in sorted(room_positions):
        pick_slot = slots_by_pick.get(overall_pick)
        slot_number, manager_name = room_managers[overall_pick]
        likely: list[tuple[Player, float]] = []
        for player_id, count in room_players[overall_pick].most_common(
            PLAN_ROOM_PLAYERS
        ):
            player = state.pool.get(player_id)
            if player is not None:
                likely.append((player, count / completed))
        plan.room.append(RoomPickForecast(
            overall_pick=int(overall_pick),
            round_label=pick_slot.label if pick_slot is not None else str(overall_pick),
            draft_slot=int(slot_number),
            manager_name=manager_name,
            simulations=completed,
            before_turn=next(
                (
                    index + 1 for index, horizon in enumerate(horizons)
                    if overall_pick < horizon.target_pick
                ),
                len(horizons),
            ),
            roster_so_far=_roster_shorthand(state, slot_number),
            tendency=_tendency_tag(profiles.get(int(slot_number))),
            position_shares={
                position: count / completed
                for position, count in room_positions[overall_pick].items()
            },
            player_shares=likely,
        ))

    plan.simulations = completed
    plan.elapsed_seconds = time.perf_counter() - started
    LOGGER.info(
        "Draft plan: %d rollouts to slot %d's next %d turn(s) (picks %s) in %.2fs",
        completed, target_slot, len(horizons),
        ", ".join(str(horizon.target_pick) for horizon in horizons),
        plan.elapsed_seconds,
    )
    return plan


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

    One turn of :func:`simulate_draft_plan`, which is where the rollout and counting
    rules live. Callers wanting the pick after this one should ask that function for
    both at once: it costs one set of rollouts instead of two, and the two answers
    are then consistent with each other.
    """
    plan = simulate_draft_plan(
        state, profiles,
        draft_slot=draft_slot, turns=1, simulations=simulations,
        extra_players=extra_players, track_limit=track_limit, seed=seed,
        progress=progress,
    )
    report = plan.first_report
    if report is not None:
        return report
    # No further picks for this slot: nothing left to wait for.
    return AvailabilityReport(
        simulations=0, picks_until_next=0, target_pick=plan.from_pick,
        from_pick=plan.from_pick, elapsed_seconds=plan.elapsed_seconds,
    )


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
            float(profile.get("predictability")), context.round_number
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
            float(profile.get("predictability")), context.round_number
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
    "DraftPlan", "PlannedTurn", "PlanWindows", "RoomPickForecast",
    "simulate_draft_plan",
    "monte_carlo_draft", "upcoming_position_pressure", "likely_next_picks",
    "RISK_BAND_EDGES", "AVAILABILITY_TRACK_LIMIT", "ALTERNATIVES_RECORDED",
    "PLAN_TURNS", "PLAN_ROOM_PLAYERS", "PLAN_GONE_BY", "PLAN_LASTS",
]
