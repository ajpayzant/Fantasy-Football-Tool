"""Three seasons of fictional draft history for the sample league.

**Why this is generated rather than random.** The opponent model's entire job is to
recover a manager's tendencies from their past picks. To demonstrate — or test —
that it works, the history has to contain tendencies that are *known by
construction*: each of the twelve managers drafts to a plan, and the model should
independently arrive at a profile resembling that plan. Random history would make
every profile a shrug, and a demo of a shrug teaches a user nothing.

**Why the plans are exaggerated.** Three drafts of sixteen rounds is 48 picks per
manager, which is a thin sample by the standards of any estimator. Real
tendencies at that sample size are indistinguishable from noise. So the sample
managers are caricatures — the zero-RB manager takes *no* running backs in the
first three rounds, not merely fewer than average. The point is a legible demo,
not a realistic one.

Each manager also drafts with a per-season *reach* tendency, so the derived
predictability and reach statistics differ between managers rather than all
collapsing to the league mean.

Nothing here is real. Player names come from the generated sample pool, so the
history joins cleanly to it and the tier / ADP features have data to read.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from core.constants import HISTORICAL_IMPORT_COLUMNS
from core.enums import Archetype, Platform, Position

from .league import (
    HISTORY_SEASONS,
    HOMER_FAVOURITE_TEAM,
    MANAGER_ARCHETYPES,
    SAMPLE_LEAGUE_NAME,
    SAMPLE_ROUNDS,
    SAMPLE_TEAM_COUNT,
)
from .names import MANAGER_NAMES
from .players import SAMPLE_SEED, sample_player_frame

HISTORY_SEED = 20260402
"""Separate from the pool seed so regenerating one does not perturb the other."""

# ─────────────────────────────────────────────────────────────────────────────
# Draft plans
#
# One positional script per archetype *per season*, round 1 → 16. These are what
# the opponent model is meant to rediscover. A script is a *preference*, not a
# command: the generator takes the best available player at the scripted position
# and falls back when the board no longer has one — which is what a real manager
# does, and what keeps the history from being perfectly predictable.
#
# The scripts are designed against the *whole* of ``infer_archetype``, not just
# the branch they are meant to hit. That function is a fixed-priority chain, so a
# label is only reachable when every earlier test fails: a manager designed as a
# homer who also happens to spend two-thirds of their early picks on running backs
# is labelled ``robust_rb``, because the RB test runs first — and correctly so,
# since that is the more informative description. Rounds 1-3 (see
# ``ProfileEstimationConfig.early_rounds``) therefore hold each manager's early-RB
# share in a deliberate band, and the round of their first QB and TE is chosen to
# clear the early-QB and elite-TE tests, before their signature trait is expressed.
# ─────────────────────────────────────────────────────────────────────────────
Q, R, W, T, K, D = (
    Position.QB, Position.RB, Position.WR, Position.TE, Position.K, Position.DST
)

# Rounds 4-16 for a manager with no early-round agenda. TE lands in round 5 and QB
# in round 6, which clears the elite-TE test (first TE at or before round 4) and
# both QB tests (early at or before round 5, late at or after round 10) — so none
# of the positional labels can fire and whatever the manager's own signature is
# gets to be the label that does.
_NEUTRAL_TAIL: tuple[Position, ...] = (W, T, Q, R, W, R, W, R, T, W, R, K, D)

# Rounds 1-3 for the same manager. Two of the three openings take one RB and the
# third takes two, which puts the recency-weighted early-RB share near 0.45:
# clear of the hero-RB band's 0.40 ceiling and clear of the robust-RB floor of
# 0.55, with room on both sides for the noise a 48-pick sample carries.
_NEUTRAL_EARLY: tuple[tuple[Position | None, ...], ...] = (
    (R, W, W), (R, R, W), (W, R, W),
)

# Rounds 1-3 for the two managers whose whole trait is *not* having a plan. Fully
# scripted despite that, and deliberately identical to the neutral opening: the
# obvious alternative — leaving a round unscripted so they take the board — was
# tried and makes their early-RB share a property of whatever the rankings happened
# to offer, which drifted to 0.56 and had the chain labelling both of them
# ``robust_rb``. Rounds 4-16 are where their list-following is expressed, and
# thirteen of sixteen rounds is plenty to measure it over.
_LIST_FOLLOWER_EARLY: tuple[tuple[Position | None, ...], ...] = _NEUTRAL_EARLY


def _plan(
    early: tuple[tuple[Position | None, ...], ...],
    tail: tuple[Position, ...] = _NEUTRAL_TAIL,
) -> tuple[tuple[Position | None, ...], ...]:
    """One full 16-round script per season, from per-season openings and a tail."""
    return tuple(opening + tail for opening in early)


DRAFT_PLANS: dict[Archetype, tuple[tuple[Position | None, ...], ...]] = {
    # No running back at all in the opening rounds; receivers loaded early.
    Archetype.ZERO_RB: _plan(((W, W, W),) * 3),
    # Backs early and often — every opening pick.
    Archetype.ROBUST_RB: _plan(((R, R, R),) * 3),
    # Quarterback in round 2, well before the league takes one.
    Archetype.EARLY_QB: _plan(((W, Q, R), (R, Q, W), (W, Q, R))),
    # Tight end in round 2, which nobody else does.
    Archetype.ELITE_TE: _plan(((R, T, W), (R, T, W), (W, T, R))),
    # Exactly one early back and the rest receivers: the hero-RB band.
    Archetype.HERO_RB: _plan(((R, W, W),) * 3),
    # Quarterback deliberately last; two of them, late, as a hedge.
    Archetype.LATE_QB: _plan(
        _NEUTRAL_EARLY, (W, T, R, W, R, W, R, W, Q, R, Q, K, D)
    ),
    # These two follow the ranking list: the neutral opening, then no tail at all,
    # so from round 4 they take whatever the board hands them. See
    # ``_LIST_FOLLOWER_EARLY`` for why the opening is scripted at all.
    Archetype.AUTODRAFT: _LIST_FOLLOWER_EARLY,
    Archetype.RANK_FOLLOWER: _LIST_FOLLOWER_EARLY,
    # Reaches for rookies far ahead of their ADP.
    Archetype.ROOKIE_HEAVY: _plan(_NEUTRAL_EARLY),
    # Takes their own NFL team's players wherever they are on the board.
    Archetype.HOMER: _plan(_NEUTRAL_EARLY),
    # Chases ceilings; reaches wildly and inconsistently.
    Archetype.HIGH_VARIANCE: _plan(_NEUTRAL_EARLY),
    # No signature at all: the league-average manager, and the label the chain
    # falls through to.
    Archetype.BALANCED: _plan(_NEUTRAL_EARLY),
}

EARLY_ROUNDS = 3
"""Rounds in which a manager never deviates from their script.

Mirrors :attr:`core.config.ProfileEstimationConfig.early_rounds`, which is the
window every early-round positional test in ``infer_archetype`` measures over. One
off-script pick out of nine early picks moves a share by 0.11, which is wider than
the gap between the hero-RB band and its neighbours — so a manager who drifts here
gets labelled by the drift rather than by the trait they were built to show.
Deviation from round 4 onward is unconstrained, and is what keeps the history from
being perfectly predictable.
"""

FALLBACK_ORDER: tuple[Position, ...] = (R, W, T, Q, D, K)
"""Positions to try, in order, when the scripted position is exhausted.

K and DST last: taking a kicker in round 4 because no receiver was left would be
a strange pick to attribute to a manager, and would corrupt the very positional
statistics this history exists to supply.
"""


@dataclass(frozen=True, slots=True)
class DraftStyle:
    """How a manager deviates from their plan and from the board.

    These are the *causes* of the statistics the opponent model estimates. They are
    set per archetype so that the model's ``predictability``, ``reach_*`` and
    ``rookie_rate`` outputs differ between managers for a reason, rather than all
    landing on the league mean.
    """

    reach_mean: float
    """Mean picks ahead of ADP this manager drafts. Positive = reaches."""
    reach_stdev: float
    """Spread of that reach. Large = erratic, and a low predictability estimate."""
    plan_adherence: float
    """Probability of following the script for a given round, 0-1."""
    rookie_bonus: float
    """Extra probability of preferring a rookie among comparable players."""
    ceiling_bonus: float
    """Extra probability of preferring the wider outcome distribution."""
    favourite_team: str = ""
    """When set, the manager over-drafts this NFL team's players."""
    favourite_team_rate: float = 0.0
    candidate_window: int = 0
    """How deep past the top of the board they look. 0 uses the shared default."""
    avoids: tuple[Position, ...] = ()
    """Positions this manager will not take in the early rounds.

    A guard rather than a preference: several archetype tests key off *when* a
    manager first takes a QB or a TE, and those tests run before the traits some
    of these managers exist to demonstrate. Without the guard a manager can be
    labelled by an accident of what the board offered in round 2 — which is the
    demo lying about the estimator, not the estimator being wrong.
    """
    avoid_through_round: int = 6
    """How long :attr:`avoids` applies. Past it the position is fair game."""


DRAFT_STYLES: dict[Archetype, DraftStyle] = {
    Archetype.ZERO_RB: DraftStyle(2.0, 7.0, 0.98, 0.04, 0.30),
    Archetype.ROBUST_RB: DraftStyle(3.5, 8.0, 0.98, 0.02, 0.10),
    Archetype.EARLY_QB: DraftStyle(6.0, 9.0, 0.98, 0.05, 0.15),
    Archetype.ELITE_TE: DraftStyle(4.0, 8.5, 0.98, 0.03, 0.18),
    Archetype.HERO_RB: DraftStyle(1.5, 7.5, 0.98, 0.06, 0.20),
    Archetype.LATE_QB: DraftStyle(1.0, 6.5, 0.98, 0.05, 0.16),
    # Autodraft: takes the top of the board every time, so it leaves nobody
    # better-ranked behind and its reach spread is near zero — the two things
    # the autodraft label tests for.
    Archetype.AUTODRAFT: DraftStyle(
        0.0, 0.0, 1.00, 0.00, 0.00, candidate_window=1,
        avoids=(Q, T), avoid_through_round=5,
    ),
    # Rank-follower: the same idea, loosened. Chooses within a band of players near
    # the top of the board rather than always the first. The band has to be wide
    # enough that the two are separable by *some* threshold — at a window of four
    # the difference from a pure autodrafter was a third of an inversion per pick,
    # which no threshold can split. At 34 it is 1.0 against 3.3, with clearance on
    # both sides of the threshold between them.
    Archetype.RANK_FOLLOWER: DraftStyle(
        0.0, 12.0, 1.00, 0.02, 0.05, candidate_window=34,
        avoids=(Q, T), avoid_through_round=5,
    ),
    # Rookie-heavy: the bonus has to be large enough to survive the board being
    # picked over by eleven rivals between their turns, and the window wide enough
    # that a rookie is usually in view at all.
    Archetype.ROOKIE_HEAVY: DraftStyle(
        9.0, 11.0, 0.95, 0.90, 0.35, candidate_window=30
    ),
    # Avoids QB early: the favourite-team pull is strong enough to drag a KC
    # quarterback into the opening rounds, which would label them early-QB. The wide
    # window is what makes the pull *visible* — a favourite-team player they never
    # look at cannot be drafted, and eleven rivals are emptying the board between
    # their picks, so a narrow window would leave the trait to chance.
    Archetype.HOMER: DraftStyle(4.0, 12.0, 0.95, 0.05, 0.15,
                                favourite_team=HOMER_FAVOURITE_TEAM,
                                favourite_team_rate=1.0,
                                candidate_window=40,
                                avoids=(Q,)),
    # High variance: a reach spread so wide it swamps the board's own ordering, so
    # this manager picks near-uniformly from everyone left at the position and
    # leaves far more better-ranked players behind than anyone else — which is
    # both what ``predictability`` measures and what the high-variance label tests.
    # Rookie bonus deliberately zero: the rookie test runs earlier in the chain,
    # and a ceiling-chaser drifts toward rookies on its own.
    Archetype.HIGH_VARIANCE: DraftStyle(
        9.0, 60.0, 0.95, 0.00, 0.60, candidate_window=45
    ),
    Archetype.BALANCED: DraftStyle(1.0, 6.0, 0.95, 0.06, 0.15),
}

DEFAULT_CANDIDATE_WINDOW = 14
"""How deep into the board a manager will consider players at their pick.

A manager who only ever takes the single best player at their position produces
zero reach variance and no distinguishable style, so the window is what lets reach
tendencies express themselves at all. Per-manager via
:attr:`DraftStyle.candidate_window`, because the width of the window *is* the
list-following trait: a window of one is an autodraft, and a wide one is how a
high-variance manager reaches.
"""

SEASON_POOL_DRIFT = 1000
"""Seed offset per season.

Each historical season is drafted from a *differently generated* pool, because
players change between years. Reusing one pool would make the same three names go
first in all three drafts and the history would carry no independent information.
"""


@dataclass(slots=True)
class _Board:
    """The available players for one historical draft, indexed for fast lookup."""

    by_position: dict[Position, list[dict[str, object]]]
    taken: set[str]

    @classmethod
    def build(cls, frame: pd.DataFrame) -> "_Board":
        by_position: dict[Position, list[dict[str, object]]] = {}
        for row in frame.to_dict("records"):
            position = Position.coerce(row["position"], None)
            if position is None:
                continue
            by_position.setdefault(position, []).append(row)
        for rows in by_position.values():
            rows.sort(key=lambda r: float(r["overall_adp"]))
        return cls(by_position=by_position, taken=set())

    def candidates(self, position: Position, window: int) -> list[dict[str, object]]:
        """The best ``window`` undrafted players at ``position``, board order."""
        rows = self.by_position.get(position, [])
        out: list[dict[str, object]] = []
        for row in rows:
            if str(row["player_name"]) in self.taken:
                continue
            out.append(row)
            if len(out) >= max(1, window):
                break
        return out

    def best_overall(self, window: int) -> list[dict[str, object]]:
        """Best undrafted players regardless of position, board order."""
        pool: list[dict[str, object]] = []
        for position in FALLBACK_ORDER:
            pool.extend(self.candidates(position, window))
        pool.sort(key=lambda r: float(r["overall_adp"]))
        return pool[:max(1, window)]

    def take(self, row: dict[str, object]) -> None:
        self.taken.add(str(row["player_name"]))


def _score_candidate(
    row: dict[str, object],
    round_number: int,
    style: DraftStyle,
    rng: random.Random,
) -> float:
    """How much this manager wants this player at this pick. Lower is better.

    The manager's reach tendency is applied as a shift on the player's ADP, so a
    manager with a positive reach mean systematically takes players earlier than
    the board says — which is precisely the statistic
    :func:`engine.opponent_model.observe_manager` measures back out.
    """
    adp = float(row["overall_adp"])
    score = adp - rng.gauss(style.reach_mean, style.reach_stdev)
    if style.rookie_bonus and str(row.get("rookie_flag")) == "Y":
        score -= style.rookie_bonus * 30.0
    if style.ceiling_bonus:
        ceiling = float(row.get("ceiling") or 0.0)
        projection = float(row.get("projection") or 1.0)
        spread = max(0.0, ceiling / projection - 1.0)
        score -= style.ceiling_bonus * spread * 90.0
    if style.favourite_team and str(row.get("nfl_team")) == style.favourite_team:
        if rng.random() < style.favourite_team_rate:
            score -= 45.0
    return score


def _is_banned(position: Position, style: DraftStyle, round_number: int) -> bool:
    """Whether ``style`` refuses ``position`` in ``round_number``."""
    return (
        bool(style.avoids)
        and round_number <= style.avoid_through_round
        and position in style.avoids
    )


def _allowed(
    rows: list[dict[str, object]], style: DraftStyle, round_number: int
) -> list[dict[str, object]]:
    """Drop positions this manager refuses this early.

    A filter rather than a score penalty because a list-follower's candidate window
    is a single player: with only one candidate, ``min`` returns it whatever its
    score, so a penalty would be silently ignored exactly where it matters most.

    May return empty. The caller widens its search in that case rather than having
    this function quietly hand back the banned players, which would make the guard
    look applied when it was not.
    """
    if not style.avoids or round_number > style.avoid_through_round:
        return rows
    banned = {str(p) for p in style.avoids}
    return [r for r in rows if str(r.get("position")) not in banned]


def _favourite_team_rows(
    board: "_Board", style: DraftStyle, window: int
) -> list[dict[str, object]]:
    """Undrafted players from this manager's favourite team, across all positions."""
    if not style.favourite_team:
        return []
    out: list[dict[str, object]] = []
    for position in FALLBACK_ORDER:
        out.extend(
            row for row in board.candidates(position, window)
            if str(row.get("nfl_team")) == style.favourite_team
        )
    return out


def _plan_position(
    archetype: Archetype,
    season_index: int,
    round_number: int,
    rng: random.Random,
    style: DraftStyle,
) -> Position | None:
    """The position this manager intends to take, or ``None`` for best-available."""
    seasons = DRAFT_PLANS.get(archetype, ())
    if not seasons:
        return None
    if round_number > EARLY_ROUNDS and rng.random() > style.plan_adherence:
        return None
    plan = seasons[season_index % len(seasons)]
    index = max(0, round_number - 1)
    if index >= len(plan):
        # Past the end of the script the manager has no agenda left and simply
        # takes the board. A plan shorter than the draft is how a manager with an
        # opinion about the early rounds only is expressed.
        return None
    return plan[index]


def _choose(
    board: _Board,
    archetype: Archetype,
    style: DraftStyle,
    season_index: int,
    round_number: int,
    pick: int,
    rng: random.Random,
) -> dict[str, object] | None:
    """One manager's pick, following their plan where the board allows."""
    wanted = _plan_position(archetype, season_index, round_number, rng, style)
    window = style.candidate_window or DEFAULT_CANDIDATE_WINDOW
    if wanted is not None and _is_banned(wanted, style, round_number):
        # The script names a position this manager refuses this early. The guard
        # wins: it exists precisely because these rounds decide the archetype
        # label, and a scripted round-6 QB is exactly the accident it guards
        # against. They take the board instead.
        wanted = None

    candidates: list[dict[str, object]] = []
    if wanted is not None:
        candidates = board.candidates(wanted, window)
        if not candidates:
            # The scripted position is gone. Fall through the fallback order
            # rather than taking a kicker in round 3.
            for position in FALLBACK_ORDER:
                candidates = _allowed(
                    board.candidates(position, window), style, round_number
                )
                if candidates:
                    break
    else:
        # A board-follower looks at the whole board, so the guard has to be applied
        # to a wider slice than their window and the window taken afterwards —
        # otherwise a banned position at the very top would be all they ever see.
        wide = board.best_overall(window + len(style.avoids) * 4)
        candidates = (_allowed(wide, style, round_number) or wide)[:max(1, window)]

    if style.favourite_team:
        # A homer abandons their plan for their team's players, so favourite-team
        # candidates are added across *all* positions. Without this the pull is
        # confined to whatever position the script named, and a team with nobody at
        # that position exerts no pull at all — a limitation of the generator
        # rather than a trait of the manager.
        candidates = candidates + _allowed(
            _favourite_team_rows(board, style, window), style, round_number
        )
    if not candidates:
        return None
    return min(candidates, key=lambda r: _score_candidate(r, round_number, style, rng))


def _slot_order(round_number: int, team_count: int) -> list[int]:
    """Snake order for one round: 1..N on odd rounds, N..1 on even."""
    slots = list(range(1, team_count + 1))
    return slots if round_number % 2 else list(reversed(slots))


def sample_history_frame(
    *,
    seasons: tuple[int, ...] = HISTORY_SEASONS,
    rounds: int = SAMPLE_ROUNDS,
    team_count: int = SAMPLE_TEAM_COUNT,
    seed: int = HISTORY_SEED,
    pool_seed: int = SAMPLE_SEED,
) -> pd.DataFrame:
    """Historical picks for the sample league, in the app's import format.

    Returned as a frame so it travels through
    :func:`services.importers.import_historical_drafts` exactly like a user's
    exported draft recap — the sample history therefore exercises the real import
    and feature-annotation path rather than bypassing it.

    Draft slots rotate by one place each season, as a real league's do, so a
    manager's history is not confounded with one fixed board position.
    """
    rows: list[dict[str, object]] = []
    for season_index, season in enumerate(seasons):
        # A fresh pool per season: rosters change between years, and reusing one
        # board would make all three drafts near-identical.
        pool = sample_player_frame(seed=pool_seed + season_index * SEASON_POOL_DRIFT)
        board = _Board.build(pool)
        rng = random.Random(seed + season_index * 7919)

        # Slots rotate one place per season.
        assignment = {
            (index + season_index) % team_count + 1: MANAGER_NAMES[index]
            for index in range(min(team_count, len(MANAGER_NAMES)))
        }
        archetypes = {
            (index + season_index) % team_count + 1: MANAGER_ARCHETYPES[index]
            for index in range(min(team_count, len(MANAGER_ARCHETYPES)))
        }

        overall = 0
        for round_number in range(1, int(rounds) + 1):
            for position_in_round, slot in enumerate(
                _slot_order(round_number, team_count), start=1
            ):
                overall += 1
                manager = assignment.get(slot)
                if manager is None:
                    continue
                archetype = archetypes.get(slot, Archetype.BALANCED)
                style = DRAFT_STYLES.get(archetype, DRAFT_STYLES[Archetype.BALANCED])
                chosen = _choose(
                    board, archetype, style, season_index, round_number, overall, rng
                )
                if chosen is None:
                    continue
                board.take(chosen)
                rows.append({
                    "season": season,
                    "league_name": SAMPLE_LEAGUE_NAME,
                    "platform": str(Platform.ESPN),
                    "manager_name": manager,
                    "round": round_number,
                    "pick_in_round": position_in_round,
                    "overall_pick": overall,
                    "player_name": chosen["player_name"],
                    "position": chosen["position"],
                    "nfl_team": chosen["nfl_team"],
                    "adp": chosen["overall_adp"],
                    "platform_rank": chosen["platform_rank"],
                    "projection": chosen["projection"],
                    "tier": chosen["tier"],
                    "keeper_flag": "N",
                    "rookie_flag": chosen["rookie_flag"],
                    "draft_date": f"{season}-08-26",
                })
    return pd.DataFrame(rows, columns=list(HISTORICAL_IMPORT_COLUMNS))


DESIGNED_TELL: dict[Archetype, str] = {
    Archetype.ZERO_RB: "No running back in the first three rounds, ever.",
    Archetype.ROBUST_RB: "Nothing but running backs in the first three rounds.",
    Archetype.EARLY_QB: "Quarterback in round 2 every season.",
    Archetype.ELITE_TE: "Tight end in round 2 every season.",
    Archetype.HERO_RB: "One back in round 1, then receivers.",
    Archetype.LATE_QB: "No quarterback until round 12, then two of them.",
    Archetype.AUTODRAFT: "Takes the single best-ranked player left, with no reaching at all.",
    Archetype.RANK_FOLLOWER: "Follows the rankings loosely — a wide reach spread, but no plan.",
    Archetype.ROOKIE_HEAVY: "Prefers a rookie over a comparable veteran 90% of the time.",
    Archetype.HOMER: f"Takes {HOMER_FAVOURITE_TEAM} players whenever one is anywhere near the board.",
    Archetype.HIGH_VARIANCE: "Reach spread of 60 picks — the board barely constrains them.",
    Archetype.BALANCED: "Nothing distinctive: the neutral opening and average reaching.",
}
"""What each sample manager was built to demonstrate, in one sentence.

Six of the twelve share the neutral opening because their trait is expressed
elsewhere — in rookie preference, team loyalty, reach spread, or the late rounds. The
opening columns alone would make those six look identical, so the tell is stated
rather than left to be inferred from the plan.
"""


def plan_summary() -> pd.DataFrame:
    """The intended plan for each sample manager — the demo's answer key.

    Surfaced in the UI beside the *inferred* profiles so a user can see what the
    opponent model recovered and what it missed. Publishing the answer key is the
    honest way to demonstrate an estimator.
    """
    rows = []
    for index, name in enumerate(MANAGER_NAMES):
        archetype = MANAGER_ARCHETYPES[index]
        plan = DRAFT_PLANS.get(archetype, ())
        style = DRAFT_STYLES.get(archetype, DRAFT_STYLES[Archetype.BALANCED])
        rows.append({
            "draft_slot": index + 1,
            "manager_name": name,
            "designed_archetype": str(archetype),
            "designed_tell": DESIGNED_TELL.get(archetype, ""),
            # ``plan`` is one script *per season*, so the opening is plan[0][:5] —
            # not plan[:5], which would slice the seasons and print three whole
            # sixteen-round scripts.
            "first_five_rounds": (
                " ".join(
                    str(p) if p is not None else "board"
                    for p in plan[0][:EARLY_ROUNDS + 2]
                )
                if plan else "follows the ranking list"
            ),
            "scripted_rounds": len(plan[0]) if plan else 0,
            "designed_reach_mean": style.reach_mean,
            "designed_reach_stdev": style.reach_stdev,
            "plan_adherence": style.plan_adherence,
        })
    return pd.DataFrame(rows)


__all__ = [
    "sample_history_frame", "plan_summary", "DRAFT_PLANS", "DRAFT_STYLES",
    "DESIGNED_TELL", "HISTORY_SEED",
]
