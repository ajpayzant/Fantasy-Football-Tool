"""Probabilistic pick selection for AI-controlled managers.

Scoring a candidate happens in two stages, kept separate on purpose:

1. :func:`score_candidate` produces a *utility* — a weighted sum of independent
   terms (value, need, tendency, run-chasing, …), each normalised to roughly the
   same scale so :class:`~core.config.ModelWeights` reads as relative importance.
2. :func:`choose_player` turns those utilities into a *probability distribution*
   via a softmax whose temperature comes from the manager's predictability, then
   samples from it.

The two-stage split is what makes the simulator league-aware rather than
deterministic. A predictable manager gets a cold temperature and almost always
takes their top-utility player; an erratic one gets a hot temperature and will
happily take their fourth choice. Neither ever becomes a pure
best-player-available bot, and no manager is ever locked to one outcome.

ADP is treated as a *distribution*, never an exact order: a player with ADP 20
is merely likely to go near pick 20, and :func:`adp_availability` scores that
likelihood from a normal centred on their ADP.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from statistics import median
from typing import Iterable, Mapping, Sequence

from core.config import LeagueConfig, ModelWeights, SimulationConfig
from core.constants import SLOT_ELIGIBILITY
from core.enums import Position, RecommendationLens, Slot
from engine.draft_state import DraftState, RunSnapshot
from models.draft import TeamRoster
from models.manager import ManagerProfile
from models.player import Player, PlayerPool, available_slot_for
# Same join key the importers and ``engine.features`` use, so a name the user types
# matches the board the same way a name in an uploaded file does.
from services.normalize import player_key

LOGGER = logging.getLogger("fantasy_mock_draft.pick_model")

PASS_CATCHERS: frozenset[Position] = frozenset({Position.WR, Position.TE})
"""Positions that stack with a quarterback. Matches ``engine.features``."""

SCARCITY_HORIZON: float = 0.25
"""Share of the league's starting seats at a position treated as *imminent* need.

Scarcity is about the run that is about to happen, not the whole season's
demand: only a fraction of the league is realistically shopping at a position at
any one pick.
"""

VALUE_SPAN_SIGMAS: float = 2.0
"""ADP standard deviations over which the value term spans +/-1.

The denominator for "how far off ADP is this pick" has to be the market's own
uncertainty about the player, not a constant. A quarter of the draft — the previous
rule — was the same 40 picks in round 1 as in round 14, and 40 picks is the whole top
of the board early and barely two rounds late. Priced that way, taking the WR8 first
overall cost 0.45 utility, which a positional tendency worth 0.60 could simply outvote:
Drake London went second overall in a live mock.

:func:`adp_sigma` already models that uncertainty and already grows with the round —
about 6 picks in round 1, 17 by round 8, 30 by round 16 — so the value term becomes a
z-score: how many standard deviations early is this? Two sigma is the span, so a
two-sigma reach costs a full point. That makes the first two rounds track consensus
closely (an ADP-19 player at pick 2 now costs 1.4, which nothing outbids) while leaving
the late rounds as loose as they really are. It also inherits a player's own
``adp_stdev`` when the file supplies one, so a rookie the platforms genuinely disagree
about gets the wider tolerance he deserves and a settled top-five pick does not.

The span is the *tighter* of the player's own spread and the round's, because inheriting
a supplied spread is right only up to a point and kickers are where it stops being right.
This board has Brandon Aubrey at ADP 84.5 with a spread of 22.9 and a platform rank of
251 — that spread is not a market which thinks he might go at pick 60, it is an average
over boards that disagree about whether he is draftable at all, and taken at face value
it bought him a 46-pick span and a place in round three. Only the value term takes the
tighter figure. :func:`adp_availability` and :func:`expected_survival` keep the honest
one, because they ask where a player will *actually* go and a player the boards disagree
about really is harder to predict.
"""

BOARD_HORIZON_SLACK: float = 1.25
"""Board depth graded by the value terms, as a multiple of the picks in the draft.

Projection, value-over-replacement and platform rank are all *standings* — "where does
this player sit" — and a standing needs a denominator. Using the file's length made the
answer meaningless at the top of the board: in a 1,003-player file the best player scored
1.000 and the fortieth scored 0.961, so a term weighted 0.55 separated the two by 0.02.
Grading over the players who will actually be drafted, plus a quarter more for the ones
just past the end, restores the difference the weight was written to express.
"""

MIN_BOARD_HORIZON: int = 48
"""Floor for that depth, so a short draft still grades a reasonable board."""

MAX_REACH_PENALTY: float = 3.0
"""How far below -1 the ADP term may go for an absurd reach.

Clipped symmetrically at -1, the term stopped distinguishing a bad pick from a
ridiculous one: with a 40-pick span, reaching 45 picks and reaching 80 scored exactly
the same, so nothing outweighed the small positive from roster need and a kicker could
come off the board at pick seven. Value on the table genuinely does saturate — a player
who lasted 200 picks past his ADP is not five times the bargain of one who lasted 40,
because you can only use him once — but reaching gets steadily more indefensible with
every pick, and the utility should say so.
"""

ADP_PLAUSIBILITY_SHARE: float = 0.25
"""Plausibility's weight as a share of the main ADP weight.

Being takeable *near this pick* matters, but far less than the value on the
board, so it rides at a quarter of the ADP weight rather than carrying its own.
"""

NEUTRAL_WHEN_UNKNOWN: float = 0.5
"""Score for ADP-derived terms when a player has no ADP at all.

Deliberately neutral: a missing field is an absence of evidence, so it must
neither reward nor punish the player.
"""

REPEAT_SEASONS_SATURATION: float = 3.0
"""Seasons of re-drafting at which the loyalty term is at full strength.

Two seasons is a coincidence worth half the term; three or more is a pattern. Above
that it saturates rather than growing without limit — a manager who has taken the same
running back four years running is not eight times as attached as one who took him
twice, and letting the term keep climbing would make an old favourite unbeatable by any
board value.
"""

MAX_LOOKAHEAD_PICKS: int = 64
"""Cap on how many of a manager's future picks are enumerated per pick.

Only the *count* matters, for the roster-imbalance penalty, and no realistic
league drafts more rounds than this.
"""

UTILITY_SPREAD_FLOOR: float = 0.10
"""Smallest utility spread the softmax will scale against.

When every candidate really is interchangeable the spread collapses toward zero;
dividing by it would manufacture false confidence in whichever player won by a
rounding error. The floor keeps such a pick close to a coin flip, which is the
honest answer.
"""

SPREAD_TEMPERATURE_DIVISOR: float = 2.0
"""Divides the utility spread before it scales temperature.

Calibrated so the resulting probabilities match how real draft rooms behave over
a ~40-player shortlist: a highly predictable manager takes their top-utility
player about half the time and stays inside their top three almost always, while
an erratic one takes the top player only about one time in six. Without the
divisor the distribution is far too flat — every candidate lands within a few
percent of every other, and even a clearly correct pick wins only rarely.

Halved from 4.0 when the spread it divides became a robust one — see
:func:`pick_probabilities` — which is about half the size of the full best-to-worst
range it replaced. The two changes together leave a normal pick's sharpness where it
was and stop one outlier from setting everybody's temperature.
"""


@dataclass(slots=True)
class ScoredCandidate:
    """One candidate player with its utility broken out by term.

    ``components`` is kept for explainability: the recommendation engine and the
    "why did he take that?" panel both render it directly, so a pick can always
    be traced back to the terms that drove it.
    """

    player: Player
    utility: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    probability: float = 0.0
    """Softmax probability, filled in by :func:`pick_probabilities`."""

    @property
    def player_id(self) -> str:
        return self.player.player_id

    def top_reasons(self, count: int = 3) -> list[tuple[str, float]]:
        """The terms that contributed most, largest absolute value first."""
        ranked = sorted(
            self.components.items(), key=lambda kv: abs(kv[1]), reverse=True
        )
        return [(k, v) for k, v in ranked[:count] if abs(v) > 1e-9]

    def explain(self, count: int = 3) -> str:
        """A short human-readable reason string for the draft log."""
        reasons = self.top_reasons(count)
        if not reasons:
            return "no distinguishing factors"
        return ", ".join(
            f"{name.replace('_', ' ')} {value:+.2f}" for name, value in reasons
        )


# ─────────────────────────────────────────────────────────────────────────────
# ADP as a distribution
# ─────────────────────────────────────────────────────────────────────────────
def baseline_adp_sigma(round_number: int, settings: SimulationConfig) -> float:
    """The round's own ADP spread, before any player-specific number.

    Grows with the round: consensus is tight at the top of the draft and wide by the
    double-digit rounds, where a player's range spans whole rounds.
    """
    growth = float(settings.adp_sigma_round_growth) * max(0, int(round_number) - 1)
    return float(settings.adp_sigma_floor) + growth


def adp_sigma(
    player: Player, round_number: int, settings: SimulationConfig
) -> float:
    """Standard deviation of a player's likely draft slot, in picks.

    Uses the player file's own ``adp_stdev`` when present, floored so a consensus ADP is
    never treated as a certainty. Otherwise it falls back to
    :func:`baseline_adp_sigma`.
    """
    supplied = player.adp_stdev
    if supplied is not None and float(supplied) > 0:
        return max(float(settings.adp_sigma_floor), float(supplied))
    return baseline_adp_sigma(round_number, settings)


def adp_availability(
    player: Player,
    overall_pick: int,
    settings: SimulationConfig,
    round_number: int | None = None,
) -> float:
    """How *natural* it is for this player to come off the board at this pick.

    A normal density centred on the player's ADP, normalised so a player being
    taken exactly at ADP scores 1.0. This is the "ADP is a distribution, not an
    order" rule: at pick 20 an ADP-24 player is nearly as plausible as the ADP-20
    player, and the model must not treat the board as a strict queue.

    Returns 0.5 — deliberately neutral rather than 0 — when a player has no ADP
    at all, so a missing field neither rewards nor punishes them.
    """
    adp = player.adp_for()
    if adp is None:
        return NEUTRAL_WHEN_UNKNOWN
    rnd = round_number if round_number is not None else 1
    sigma = max(1e-6, adp_sigma(player, rnd, settings))
    z = (float(overall_pick) - float(adp)) / sigma
    return float(math.exp(-0.5 * z * z))


def expected_survival(
    player: Player,
    picks_until_next: int,
    overall_pick: int,
    settings: SimulationConfig,
    round_number: int | None = None,
) -> float:
    """Rough probability a player is still there at the manager's next turn.

    A closed-form approximation over the ADP distribution, used as a *utility
    term*. The honest, simulated answer comes from the Monte Carlo availability
    rollouts; this is the cheap version that runs inside the scoring loop for
    every candidate.
    """
    if picks_until_next <= 0:
        return 1.0
    adp = player.adp_for()
    if adp is None:
        return NEUTRAL_WHEN_UNKNOWN
    rnd = round_number if round_number is not None else 1
    sigma = max(1e-6, adp_sigma(player, rnd, settings))
    target = float(overall_pick) + float(picks_until_next)
    # P(draft slot > target) under a normal centred on ADP.
    z = (target - float(adp)) / (sigma * math.sqrt(2.0))
    return float(max(0.0, min(1.0, 0.5 * (1.0 - math.erf(z)))))


# ─────────────────────────────────────────────────────────────────────────────
# Utility terms
# ─────────────────────────────────────────────────────────────────────────────
def _value_term(
    player: Player,
    overall_pick: int,
    settings: SimulationConfig,
    round_number: int,
) -> float:
    """Positive when a player has fallen past his ADP, negative when reaching.

    ``pick - adp``: a player who *fell* is still on the board later than the crowd
    would have taken him, so his pick number is higher than his ADP. Getting this
    backwards makes the model reward the latest-ADP player available — it reads a
    25-pick reach as the biggest bargain on the board — which is precisely
    inverted.

    Note this is deliberately the *negation* of
    :attr:`models.draft.Pick.adp_delta`, which is reach-positive. The two measure
    opposite things and are named accordingly: ``adp_delta`` asks "how far did
    this manager reach?", while this term asks "how much value is on the table?".
    Conflating the two is what produced the original inversion.

    Measured in the player's own ADP standard deviations — see
    :data:`VALUE_SPAN_SIGMAS` — so it is strict where consensus is tight (the first two
    rounds) and forgiving where it is not (the last ten), and independent of how many
    names the player file happened to hold. Bargains cap at +1; reaches run to
    :data:`MAX_REACH_PENALTY`, for the reason given there.
    """
    adp = player.adp_for()
    if adp is None:
        return 0.0
    sigma = min(
        adp_sigma(player, round_number, settings),
        baseline_adp_sigma(round_number, settings),
    )
    span = max(1.0, VALUE_SPAN_SIGMAS * sigma)
    value = (float(overall_pick) - float(adp)) / span
    return float(max(-MAX_REACH_PENALTY, min(1.0, value)))


def _projection_term(player: Player, pool: PlayerPool, horizon: int) -> float:
    """The player's projection as a 0-1 percentile of the draftable board."""
    projection = player.projection
    if projection is None:
        return 0.0
    return pool.projection_percentile(player, horizon=horizon)


def _vor_term(player: Player, pool: PlayerPool, horizon: int) -> float:
    """Value over replacement, normalised to 0-1 across the draftable board."""
    return pool.vor_percentile(player, horizon=horizon)


def _tier_term(player: Player, pool: PlayerPool) -> float:
    """Higher for better tiers, so tier 1 outranks tier 4."""
    if player.tier is None:
        return 0.0
    # Tiers are 1-based and lower is better; map to a decaying 0-1 score.
    return float(1.0 / (1.0 + max(0, int(player.tier) - 1)))


class RosterView:
    """Per-pick roster and board facts, computed once and reused per candidate.

    Every fact here depends only on the roster and the board, both of which are
    frozen for the duration of one pick — but each is asked once per *candidate*,
    forty times a pick. Recomputed naively they dominated the profile:
    ``available_slot_for``, ``open_starting_slots`` and ``filled_starting_slots``
    together accounted for more time than the rest of the scoring loop.

    That matters beyond tidiness. A Monte Carlo run is 200 whole drafts and an
    availability estimate is ~120 partial ones, so anything per-candidate is
    multiplied by roughly 40 × picks × simulations before a user sees a number.

    Roster facts are keyed to :attr:`models.draft.TeamRoster.version`, so a
    caller that adds a player to a roster mid-pick — which the recommendation
    engine's "what if I took him?" evaluation does deliberately — gets a fresh
    answer rather than a stale one. Board facts are *not* versioned: the board
    only moves when a pick is committed, which ends the pick this view describes.

    It takes the board and the pool rather than a :class:`DraftState` so a caller
    with a bare roster — a test, or the roster-analysis page — can build one.
    """

    __slots__ = (
        "roster", "config", "board", "pool", "_version", "_filled", "_slot_for",
        "_seats", "_remaining", "_team_positions", "_open_starters",
    )

    def __init__(
        self,
        roster: TeamRoster,
        config: LeagueConfig,
        *,
        board: DraftState | None = None,
        pool: PlayerPool | None = None,
    ) -> None:
        self.roster = roster
        self.config = config
        self.board = board
        self.pool = pool if pool is not None else (board.pool if board else None)
        self._seats: dict[Position, int] = {}
        self._remaining: dict[Position, int] = {}
        self._version = -1
        self._refresh()

    def _refresh(self) -> None:
        """Recompute the roster-dependent facts and re-stamp the version."""
        self._version = self.roster.version
        self._filled: dict[Slot, int] = self.roster.filled_starting_slots()
        self._open_starters: int = sum(self.roster.open_starting_slots().values())
        self._slot_for: dict[Position, Slot | None] = {}
        self._team_positions: dict[str, set[Position]] = {}

    def _sync(self) -> None:
        if self._version != self.roster.version:
            self._refresh()

    @property
    def filled_slots(self) -> dict[Slot, int]:
        """Starting slot → seats already used. Do not mutate."""
        self._sync()
        return self._filled

    @property
    def open_starter_count(self) -> int:
        """Total unfilled starting seats on this roster."""
        self._sync()
        return self._open_starters

    def starting_slot_for(self, position: Position) -> Slot | None:
        """First open starting slot this position could fill, or ``None``."""
        self._sync()
        if position not in self._slot_for:
            self._slot_for[position] = available_slot_for(
                position, self._filled, self.config.roster
            )
        return self._slot_for[position]

    def fills_starting_slot(self, position: Position) -> bool:
        return self.starting_slot_for(position) is not None

    def league_starting_seats(self, position: Position) -> int:
        """Seats league-wide that this position is eligible to fill, per team."""
        if position not in self._seats:
            self._seats[position] = sum(
                count for slot, count in self.config.roster.starting_slots.items()
                if position in SLOT_ELIGIBILITY.get(slot, frozenset())
            )
        return self._seats[position]

    def remaining_at(self, position: Position, limit: int = 64) -> int:
        """How many are still on the board at a position, capped at ``limit``.

        0 with no board attached, which reads as "no supply left" and so is the
        conservative answer for a view built off a bare roster.
        """
        if self.board is None:
            return 0
        if position not in self._remaining:
            self._remaining[position] = len(
                self.board.available_at_position(position, limit=limit)
            )
        return self._remaining[position]

    def positions_on_nfl_team(self, team: str) -> set[Position]:
        """Positions this roster already holds from one NFL team."""
        if self.pool is None:
            return set()
        self._sync()
        key = (team or "").upper()
        cached = self._team_positions.get(key)
        if cached is None:
            cached = _positions_on_team(self.roster, self.pool, key)
            self._team_positions[key] = cached
        return cached


def _need_term(player: Player, view: RosterView) -> float:
    """How much this pick advances the manager's starting lineup.

    1.0 when it fills an empty starting slot, tapering for depth once the
    starters are set. Data-driven via :func:`available_slot_for`, so a superflex
    or TE-premium lineup is handled without special cases.
    """
    if view.fills_starting_slot(player.position):
        return 1.0
    # Starters are covered at this position; depth still has some value, less so
    # the more of that position they already hold.
    depth = view.roster.count_at(player.position)
    return float(1.0 / (1.0 + max(1, depth)))


def _scarcity_term(player: Player, view: RosterView) -> float:
    """Rises as the startable supply at a position dries up.

    Measured against how many starters the whole league still needs at that
    position, which is what actually makes a position scarce.
    """
    remaining = view.remaining_at(player.position)
    if remaining <= 0:
        return 0.0
    seats = view.league_starting_seats(player.position)
    league_need = max(
        1.0, float(seats) * float(view.config.team_count) * SCARCITY_HORIZON
    )
    return float(max(0.0, min(1.0, league_need / (league_need + remaining))))


def _run_term(
    player: Player, runs: RunSnapshot, profile: ManagerProfile,
    settings: SimulationConfig,
) -> float:
    """Positional-run pressure, signed by whether this manager chases runs.

    ``run_chase`` near 1 means the manager joins runs; near 0 means they fade
    them and take the position everyone else is ignoring.
    """
    windows = settings.run_windows
    if not windows:
        return 0.0
    heat = max(runs.rate(window, player.position) for window in windows)
    # Centre on the share a position would hold if picks were spread evenly.
    baseline = 1.0 / max(1, len(Position))
    pressure = float(max(-1.0, min(1.0, (heat - baseline) / max(1e-6, baseline))))
    lean = (float(profile.get("run_chase")) - 0.5) * 2.0
    return pressure * lean


def _preference_satiation(player: Player, view: RosterView) -> float:
    """How much of this manager's appetite for the position is still unspent.

    A positional tendency describes how a manager *allocates* picks across
    positions, so it has to fade as that allocation is actually made. Left
    unsatiated it compounds: an early-QB manager's quarterback bias fires just as
    hard in round 3 as in round 1, so he takes three quarterbacks instead of the
    one he really wants.

    1.0 while a starting seat at the position is still open — the appetite is
    genuinely unspent — then tapering with each extra body already held.

    It scales negative biases too, which is the right behaviour rather than a
    convenient side effect: a zero-RB manager avoids running backs *until* he
    starts taking them, and from that point on he is a normal drafter at the
    position. Suppressing the taper for avoidance would make him refuse backs all
    draft, which is not what zero-RB means.
    """
    if view.fills_starting_slot(player.position):
        return 1.0
    return float(1.0 / (1.0 + max(0, view.roster.count_at(player.position))))


def _position_preference_term(
    player: Player, view: RosterView, profile: ManagerProfile
) -> float:
    """This manager's standing bias for or against the position, once satiated."""
    bias = float(profile.position_bias.get(player.position, 0.0))
    return bias * _preference_satiation(player, view)


def _round_preference_term(
    player: Player, view: RosterView, profile: ManagerProfile, round_number: int,
    settings: SimulationConfig,
) -> float:
    """Round-specific positional bias — the early-round tendencies.

    Only applies inside the configured early-round window, which is where
    managers' positional signatures actually live, and is satiated by what the
    roster already holds so the tendency is spent rather than repeated.
    """
    if int(round_number) > int(settings.estimation.early_rounds):
        return 0.0
    bias = float(profile.early_round_position_bias.get(player.position, 0.0))
    return bias * _preference_satiation(player, view)


def _rank_term(player: Player, horizon: int) -> float:
    """Agreement with the platform's own ranking, as a 0-1 score.

    Weighted at pick time by the manager's ``rank_dependence``, so an autodrafter
    follows the list closely and an independent thinker mostly ignores it.

    Graded over ``horizon`` picks rather than the whole file, for the reason given at
    :data:`BOARD_HORIZON_SLACK`: divided by 1,003 the difference between the first and
    fortieth ranked player is 0.039, which no weight can turn back into a preference.
    """
    rank = player.rank_for()
    if rank is None:
        return 0.0
    n = max(1, int(horizon))
    return float(max(0.0, 1.0 - (float(rank) - 1.0) / float(n)))


def _rookie_term(player: Player, profile: ManagerProfile) -> float:
    if not player.is_rookie:
        return 0.0
    # Centred so an average rookie appetite is neutral rather than a bonus.
    return (float(profile.get("rookie_rate")) - 0.5) * 2.0


def _favorite_team_term(player: Player, profile: ManagerProfile) -> float:
    favorite = (profile.preferences.favorite_nfl_team or "").strip().upper()
    if not favorite or (player.nfl_team or "").upper() != favorite:
        return 0.0
    return float(profile.get("favorite_team_rate"))


def _named_player_term(player: Player, profile: ManagerProfile) -> float:
    """Players the user named as this manager's must-haves or won't-touches.

    Returns ``+1`` for a favourite, ``-1`` for a disliked player, ``0`` — the case
    for every manager the user has not annotated — otherwise. Deliberately not
    scaled by an estimated rate: this is not a tendency inferred from picks that
    could be over-trusted, it is the user stating a fact about someone they know.

    Matched on :func:`services.normalize.player_key`, so "AJ Brown" typed into the
    editor finds "A.J. Brown Jr." on the board. Position is left out of the key on
    purpose — the user typed a name, not a name and a position.
    """
    preferences = profile.preferences
    if not preferences.favorite_players and not preferences.disliked_players:
        return 0.0
    key = player_key(player.name)
    if not key:
        return 0.0
    if any(player_key(name) == key for name in preferences.favorite_players):
        return 1.0
    if any(player_key(name) == key for name in preferences.disliked_players):
        return -1.0
    return 0.0


def _repeat_player_term(player: Player, profile: ManagerProfile) -> float:
    """Managers come back to players they have drafted before, in earlier seasons.

    ``profile.repeat_players`` counts *distinct seasons*, and only players with two or
    more are in it, so any entry here is already evidence of a habit rather than one
    memorable pick. The term rises with the number of seasons and saturates at
    :data:`REPEAT_SEASONS_SATURATION`.

    Unlike :func:`_named_player_term` this is inferred from a draft history rather than
    stated by the user, which is why it carries a smaller weight: "he took Kupp three
    years running" is real evidence, but it is also consistent with Kupp simply being
    the best player available at that manager's slot three years running.

    Matched on :func:`services.normalize.player_key` so history spellings meet board
    spellings — the two usually come from different sources.
    """
    repeats = profile.repeat_players
    if not repeats:
        return 0.0
    key = player_key(player.name)
    if not key:
        return 0.0
    for name, seasons in repeats.items():
        if player_key(name) != key:
            continue
        count = float(seasons)
        if count < 2:
            return 0.0
        span = max(1.0, REPEAT_SEASONS_SATURATION - 1.0)
        return float(min(1.0, (count - 1.0) / span))
    return 0.0


def _stack_term(
    player: Player, view: RosterView, profile: ManagerProfile
) -> float:
    """Reward pairing a quarterback with his pass-catchers, if this manager does.

    Mirrors the historical ``was_stack`` definition: QB↔WR/TE only, since that is
    the correlation a stacker is actually buying.
    """
    team = (player.nfl_team or "").upper()
    if not team or team == "FA":
        return 0.0
    held = view.positions_on_nfl_team(team)
    if not held:
        return 0.0
    is_stack = (
        (player.position is Position.QB and bool(held & PASS_CATCHERS))
        or (player.position in PASS_CATCHERS and Position.QB in held)
    )
    return float(profile.get("stack_rate")) if is_stack else 0.0


def _handcuff_term(
    player: Player, view: RosterView, profile: ManagerProfile
) -> float:
    """Reward a second running back from a team the manager already owns."""
    if player.position is not Position.RB:
        return 0.0
    team = (player.nfl_team or "").upper()
    if not team or team == "FA":
        return 0.0
    if Position.RB not in view.positions_on_nfl_team(team):
        return 0.0
    return float(profile.get("handcuff_rate"))


def _positions_on_team(
    roster: TeamRoster, pool: PlayerPool, team: str
) -> set[Position]:
    """Positions this roster already holds from one NFL team.

    ``TeamRoster`` deliberately stores only positions, so NFL teams are resolved
    through the pool by player id.
    """
    out: set[Position] = set()
    for pid in roster.player_ids:
        held = pool.get(pid)
        if held is not None and (held.nfl_team or "").upper() == team:
            out.add(held.position)
    return out


def _injury_term(player: Player) -> float:
    """Penalty magnitude for injury / suspension status (always >= 0)."""
    return float(player.injury_penalty)


def _limit_penalty(player: Player, roster: TeamRoster) -> float:
    """1.0 when the position is already at its configured maximum."""
    return 1.0 if roster.at_position_limit(player.position) else 0.0


LATE_ROUND_POSITIONS: frozenset[Position] = frozenset({Position.K, Position.DST})
"""Positions that real drafts leave until the end, near-universally."""

LATE_ROUND_GRACE: int = 3
"""Rounds at the end of a draft where taking a kicker or defence is simply normal."""


def _premature_penalty(
    player: Player, round_number: int, config: LeagueConfig
) -> float:
    """How out of place a kicker or defence is this early, from 1.0 down to 0.

    Full strength in round one and fading linearly to nothing over the last
    :data:`LATE_ROUND_GRACE` rounds. Graded rather than a cutoff because the convention
    is itself graded: round twelve is early for a kicker and round three is absurd, and
    a flag cannot say both. Every other position returns 0 and is unaffected.
    """
    if player.position not in LATE_ROUND_POSITIONS:
        return 0.0
    rounds = max(1, int(config.rounds))
    # The first round where taking one is unremarkable.
    normal_from = max(2, rounds - int(LATE_ROUND_GRACE) + 1)
    current = max(1, int(round_number))
    if current >= normal_from:
        return 0.0
    return float(normal_from - current) / float(normal_from - 1)


def _imbalance_penalty(
    player: Player,
    roster: TeamRoster,
    config: LeagueConfig,
    picks_left: int,
    view: "RosterView | None" = None,
) -> float:
    """Penalise picks that would leave a starting slot unfillable.

    Late in a draft, taking a fourth receiver when the kicker seat is empty is a
    real mistake, and a manager who never does it should be modelled as such.

    The penalty scales with how badly the roster is behind: one spare pick in
    hand makes a luxury pick merely questionable, while needing every remaining
    pick for a different seat makes it indefensible. Returning a graded value
    rather than a flag is what lets it actually outweigh a strong board term —
    a flat penalty was small enough that four straight luxury picks still beat
    filling the seat.

    ``view`` is an optional pre-computed :class:`RosterView`; without one the
    roster facts are recomputed, which keeps the function usable standalone.
    """
    view = view if view is not None else RosterView(roster, config)
    unmet = view.open_starter_count
    if unmet <= 0:
        return 0.0
    if picks_left > unmet:
        return 0.0
    # Every remaining pick is needed for a starting seat; does this fill one?
    if view.fills_starting_slot(player.position):
        return 0.0
    # Shortfall of 0 means "exactly enough picks left"; each pick further behind
    # compounds, since the seat can never be recovered later.
    shortfall = unmet - picks_left
    return float(1.0 + shortfall)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class PickContext:
    """Everything the scorer needs about the moment a pick is made.

    Built once per pick by :func:`context_for` and reused across every candidate,
    so per-pick work (run snapshot, roster lookups, gap to the next turn) happens
    once rather than once per player.
    """

    state: DraftState
    profile: ManagerProfile
    roster: TeamRoster
    overall_pick: int
    round_number: int
    picks_until_next: int
    picks_left_for_manager: int
    runs: RunSnapshot
    settings: SimulationConfig
    weights: ModelWeights
    config: LeagueConfig
    rng: random.Random
    view: RosterView = None  # type: ignore[assignment]
    """Memoized roster/board facts. Built by :func:`context_for`; never ``None``
    in practice, but defaulted so a hand-built context still constructs."""

    horizon: int = 0
    """Board depth the value terms grade against — see :data:`BOARD_HORIZON_SLACK`.
    Derived in ``__post_init__`` when left at 0, which is every real caller."""

    def __post_init__(self) -> None:
        if self.view is None:
            self.view = RosterView(self.roster, self.config, board=self.state)
        # Scales the board to the draft rather than to the player file, and is the same
        # for every candidate on the pick, so it is settled once here instead of forty
        # times in the scorer.
        picks = max(1, int(self.config.total_picks))
        if not self.horizon:
            self.horizon = max(
                MIN_BOARD_HORIZON, int(round(picks * BOARD_HORIZON_SLACK))
            )

    @property
    def pool(self) -> PlayerPool:
        return self.state.pool


def context_for(
    state: DraftState,
    profile: ManagerProfile,
    *,
    draft_slot: int | None = None,
    rng: random.Random | None = None,
) -> PickContext:
    """Assemble the scoring context for whoever is on the clock."""
    slot = state.current_slot
    if slot is None:
        raise ValueError("the draft is complete; there is no pick to score")
    draft_slot = int(draft_slot if draft_slot is not None else slot.draft_slot)
    roster = state.roster(draft_slot)
    # Inclusive of the pick being made right now, which is what the
    # roster-imbalance penalty needs: "two seats to fill and two picks left" must
    # not read as slack.
    upcoming = state.next_pick_numbers(draft_slot, count=MAX_LOOKAHEAD_PICKS)
    picks_left = len(upcoming)
    # The gap *after* this slot's pick, not before it. Measured from the slot's own
    # next two picks rather than from the clock, because this context may describe
    # a manager several picks away (the recommendation engine's opponent-threat
    # pass does exactly that) and the gap they face starts at their pick, not ours.
    #
    # Taking the gap before the pick instead would leave this at 0 for whoever is
    # on the clock — and 0 makes expected_survival return 1.0 for everyone, which
    # silently disables the expected_availability term on every pick actually being
    # made. It only ever looked correct because the term was never exercised.
    gap = (upcoming[1] - upcoming[0] - 1) if len(upcoming) >= 2 else 0
    return PickContext(
        state=state,
        profile=profile,
        roster=roster,
        overall_pick=int(slot.overall_pick),
        round_number=int(slot.round_number),
        picks_until_next=int(gap) if gap is not None else 0,
        picks_left_for_manager=picks_left,
        runs=state.run_snapshot(),
        settings=state.settings,
        weights=state.settings.weights,
        config=state.config,
        rng=rng or state.rng,
        view=RosterView(roster, state.config, board=state),
    )


def score_candidate(player: Player, context: PickContext) -> ScoredCandidate:
    """Score one player, returning both the utility and its breakdown.

    Every term is computed independently and multiplied by its configured weight.
    Terms whose inputs are missing contribute exactly zero rather than a guess,
    which is what lets the model run on a sparse player file.
    """
    w = context.weights
    profile = context.profile
    pool = context.pool
    view = context.view
    components: dict[str, float] = {}

    horizon = context.horizon
    components["adp_value"] = w.adp * _value_term(
        player, context.overall_pick, context.settings, context.round_number
    )
    components["projection"] = w.projection * _projection_term(player, pool, horizon)
    components["tier"] = w.tier * _tier_term(player, pool)
    components["value_over_replacement"] = w.value_over_replacement * _vor_term(
        player, pool, horizon
    )
    components["roster_need"] = (
        w.roster_need
        * float(profile.get("need_dependence"))
        * _need_term(player, view)
    )
    components["positional_scarcity"] = w.positional_scarcity * _scarcity_term(
        player, view
    )
    components["manager_position_preference"] = (
        w.manager_position_preference
        * _position_preference_term(player, view, profile)
    )
    components["round_specific_preference"] = (
        w.round_specific_preference
        * _round_preference_term(
            player, view, profile, context.round_number, context.settings,
        )
    )
    components["platform_rank"] = (
        w.platform_rank_dependence
        * float(profile.get("rank_dependence"))
        * _rank_term(player, horizon)
    )
    components["rookie"] = w.rookie_preference * _rookie_term(player, profile)
    components["favorite_team"] = w.favorite_team_preference * _favorite_team_term(
        player, profile
    )
    components["named_player"] = w.named_player_preference * _named_player_term(
        player, profile
    )
    components["repeat_player"] = w.repeat_player_affinity * _repeat_player_term(
        player, profile
    )
    components["stack"] = w.stack * _stack_term(player, view, profile)
    components["handcuff"] = w.handcuff * _handcuff_term(player, view, profile)
    components["positional_run"] = w.positional_run * _run_term(
        player, context.runs, profile, context.settings
    )
    # Wanting a player who will not last is what drives a manager to reach.
    components["expected_availability"] = w.expected_availability * (
        1.0
        - expected_survival(
            player, context.picks_until_next, context.overall_pick,
            context.settings, context.round_number,
        )
    )
    components["adp_plausibility"] = w.adp * ADP_PLAUSIBILITY_SHARE * adp_availability(
        player, context.overall_pick, context.settings, context.round_number
    )

    # Penalties, subtracted so a positive weight always means "avoid this".
    components["injury_penalty"] = -w.injury_penalty * _injury_term(player)
    components["positional_limit_penalty"] = -w.positional_limit_penalty * _limit_penalty(
        player, context.roster
    )
    components["premature_kicker_penalty"] = -w.premature_kicker_penalty * (
        _premature_penalty(player, context.round_number, context.config)
    )
    components["roster_imbalance_penalty"] = -w.roster_imbalance_penalty * (
        _imbalance_penalty(
            player, context.roster, context.config, context.picks_left_for_manager,
            view=view,
        )
    )

    utility = float(sum(components.values()))
    return ScoredCandidate(player=player, utility=utility, components=components)


def candidate_shortlist(context: PickContext) -> list[Player]:
    """The players this manager might plausibly take at this pick.

    Starts with the top ``candidate_pool_size`` available players by board order
    — a manager is not plausibly taking the 300th-ranked player, and the softmax
    would give them a vanishing probability anyway.

    It then *adds* the best available player for every unfilled starting slot.
    Without that, positions with a late ADP could never be drafted at all:
    kickers and defenses sit outside the top of the board by construction, so a
    board-order shortlist would leave those seats empty forever no matter how
    hard the roster-imbalance penalty pushed.
    """
    state = context.state
    limit = max(1, int(context.settings.candidate_pool_size))
    shortlist = list(state.available_players(limit=limit))
    seen = {p.player_id for p in shortlist}

    for slot in context.roster.open_starting_slots():
        for position in SLOT_ELIGIBILITY.get(slot, frozenset()):
            if not context.view.fills_starting_slot(position):
                continue
            for player in state.available_at_position(position, limit=1):
                if player.player_id not in seen:
                    shortlist.append(player)
                    seen.add(player.player_id)
    return shortlist


def score_candidates(
    context: PickContext, candidates: Iterable[Player] | None = None
) -> list[ScoredCandidate]:
    """Score a shortlist of players, best utility first."""
    if candidates is None:
        candidates = candidate_shortlist(context)
    scored = [score_candidate(player, context) for player in candidates]
    scored.sort(key=lambda c: c.utility, reverse=True)
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# Softmax selection
# ─────────────────────────────────────────────────────────────────────────────
def pick_probabilities(
    scored: Sequence[ScoredCandidate], temperature: float
) -> list[ScoredCandidate]:
    """Fill in each candidate's softmax probability, in place, and return them.

    Temperature controls how sharply utility translates into probability: cold
    means the top player is nearly certain, hot means the field is competitive.

    Temperature is interpreted *relative to the spread of utilities on the
    board*, not in raw utility units. Absolute temperature cannot work here: the
    gap between the best and worst plausible candidate is wide in round 1 and
    narrow in round 12, so a fixed temperature that models a decisive manager
    early would make the same manager look like a coin-flipper late. Scaling by
    the observed spread keeps "predictable" meaning the same thing all draft.

    The spread is measured from the best candidate to the *median* one, not to the
    worst. Best-to-worst let a single hopeless candidate set the temperature for
    everybody: one player carrying a large reach penalty widened the spread, the wider
    spread raised the temperature, and the higher temperature flattened the whole
    distribution — so the presence of an obviously bad option made every good option
    less likely. Half the board is a stable ruler; the tail of it is not.

    The max-utility offset keeps ``exp`` from overflowing on large utilities.
    """
    if not scored:
        return []
    utilities = [c.utility for c in scored]
    best = max(utilities)
    spread = best - median(utilities)
    scale = max(UTILITY_SPREAD_FLOOR, spread) / SPREAD_TEMPERATURE_DIVISOR
    t = max(1e-3, float(temperature)) * scale
    weights = [math.exp((c.utility - best) / t) for c in scored]
    total = sum(weights)
    if total <= 0:
        # Degenerate case: fall back to a uniform distribution rather than
        # silently returning zeros that would break sampling.
        uniform = 1.0 / len(scored)
        for candidate in scored:
            candidate.probability = uniform
        return list(scored)
    for candidate, weight in zip(scored, weights):
        candidate.probability = weight / total
    return list(scored)


def choose_player(
    context: PickContext,
    candidates: Iterable[Player] | None = None,
    *,
    scored: Sequence[ScoredCandidate] | None = None,
) -> ScoredCandidate | None:
    """Sample this manager's pick from the softmax over candidate utilities.

    Returns ``None`` only when there is nobody available to draft. The chosen
    candidate keeps its component breakdown and probability so the caller can log
    *why* the pick happened, not just what it was.
    """
    ranked = list(scored) if scored is not None else score_candidates(context, candidates)
    if not ranked:
        return None
    temperature = context.settings.temperature_for(
        float(context.profile.get("predictability")), context.round_number
    )
    pick_probabilities(ranked, temperature)
    roll = context.rng.random()
    cumulative = 0.0
    for candidate in ranked:
        cumulative += candidate.probability
        if roll <= cumulative:
            return candidate
    # Floating-point shortfall: the last candidate absorbs the remainder.
    return ranked[-1]


def most_likely_player(
    context: PickContext, candidates: Iterable[Player] | None = None
) -> ScoredCandidate | None:
    """The single highest-utility candidate, with no sampling.

    Used by the recommendation engine's "what will he do?" lens and by the
    ADP-only / rank-only evaluation baselines, which must be deterministic.
    """
    ranked = score_candidates(context, candidates)
    return ranked[0] if ranked else None


def position_probabilities(
    scored: Sequence[ScoredCandidate],
) -> dict[Position, float]:
    """Aggregate pick probability by position — 'he probably takes a RB here'."""
    out: dict[Position, float] = {}
    for candidate in scored:
        position = candidate.player.position
        out[position] = out.get(position, 0.0) + float(candidate.probability)
    return out


__all__ = [
    "ScoredCandidate", "PickContext", "RosterView", "context_for", "score_candidate",
    "score_candidates", "candidate_shortlist", "pick_probabilities", "choose_player",
    "most_likely_player", "position_probabilities", "adp_sigma",
    "adp_availability", "expected_survival",
]
