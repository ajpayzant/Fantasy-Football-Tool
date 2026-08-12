"""Feature engineering over imported historical drafts.

Importers produce raw :class:`~models.draft.HistoricalPick` rows: who took whom,
when. The opponent model needs *context* — was that a reach, did it fill a
starting slot, was it part of a positional run, how long until that manager
picked again. :func:`annotate_history` computes those fields once, in place, so
profile building and backtesting never recompute them.

Backtest isolation is at *season* granularity: a few features (``picks_until_next``,
``rank_inversions``) read later picks within the same completed draft, which is
sound because a backtest withholds whole seasons — see ``reference_season`` in
:func:`engine.opponent_model.observe_manager`. Nothing here reads a season the
caller has withheld.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from core.config import ProfileEstimationConfig, RosterSettings
from core.constants import SLOT_ELIGIBILITY, SLOT_FILL_PRIORITY
from core.enums import Position, Slot
from models.draft import DraftHistory, HistoricalDraft, HistoricalPick
from models.player import PlayerPool, available_slot_for
from services.normalize import player_key

LOGGER = logging.getLogger("fantasy_mock_draft.features")

DRAFT_PHASES: tuple[str, str, str] = ("early", "middle", "late")


@dataclass(slots=True)
class HistoryStats:
    """League-wide aggregates computed alongside the per-pick features.

    These are the denominators the opponent model shrinks toward: what a
    *typical* manager in this league did, so an individual can be described as
    above or below it.
    """

    pick_count: int = 0
    manager_count: int = 0
    positions_seen: tuple[Position, ...] = ()
    mean_pick_by_position: dict[Position, float] | None = None
    """Average overall pick at which each position went, league-wide."""
    share_by_position: dict[Position, float] | None = None
    """Share of all picks spent on each position, league-wide."""
    first_pick_by_position: dict[Position, float] | None = None
    """Average *round* of each manager's first pick at a position."""
    early_share_by_position: dict[Position, float] | None = None
    """Share of *early-round* picks spent on each position, league-wide."""
    early_rounds: int = 0
    """How many rounds ``early_share_by_position`` covers."""

    def __post_init__(self) -> None:
        self.mean_pick_by_position = self.mean_pick_by_position or {}
        self.share_by_position = self.share_by_position or {}
        self.first_pick_by_position = self.first_pick_by_position or {}
        self.early_share_by_position = self.early_share_by_position or {}


def annotate_history(
    history: DraftHistory,
    *,
    pool: PlayerPool | None = None,
    roster: RosterSettings | None = None,
    config: ProfileEstimationConfig | None = None,
) -> HistoryStats:
    """Fill in every engineered field on every pick in ``history``, in place.

    ``pool`` supplies position / rookie / ADP data for picks whose file did not
    carry it — matched by normalised name. It is optional: without it the
    features that need player metadata are simply left at their defaults rather
    than being guessed.
    """
    config = config or ProfileEstimationConfig()
    roster = roster or RosterSettings()
    for draft in history.drafts:
        annotate_draft(draft, pool=pool, roster=roster, config=config)
    stats = summarize_history(history)
    LOGGER.info(
        "Annotated %d historical picks across %d draft(s), %d manager(s)",
        stats.pick_count, len(history.drafts), stats.manager_count,
    )
    return stats


def annotate_draft(
    draft: HistoricalDraft,
    *,
    pool: PlayerPool | None = None,
    roster: RosterSettings | None = None,
    config: ProfileEstimationConfig | None = None,
) -> None:
    """Annotate one season's picks. Safe to call repeatedly (idempotent)."""
    config = config or ProfileEstimationConfig()
    roster = roster or RosterSettings()
    if not draft.picks:
        return

    picks = sorted(draft.picks, key=lambda p: p.overall_pick)
    team_count = draft.infer_team_count()
    total_picks = max(p.overall_pick for p in picks)

    _backfill_from_pool(picks, pool)
    _fill_round_numbers(picks, team_count)
    _annotate_rank_inversions(picks)

    # Running per-manager roster reconstruction, and the league-wide run window.
    counts: dict[str, dict[Position, int]] = {}
    sizes: dict[str, int] = {}
    filled_slots: dict[str, dict[Slot, int]] = {}
    owned_by_team: dict[str, dict[str, set[Position]]] = {}
    """manager_key → NFL team → positions that manager already holds there."""
    recent: list[Position] = []
    remaining_by_tier = _tier_inventory(picks)
    next_pick_by_manager = _next_pick_lookup(picks)

    for pick in picks:
        key = pick.manager_key
        manager_counts = counts.setdefault(key, {})
        manager_filled = filled_slots.setdefault(key, {})

        pick.adp_delta = _adp_delta(pick, config)
        pick.rank_delta = (
            float(pick.platform_rank) - float(pick.overall_pick)
            if pick.platform_rank is not None else None
        )
        pick.roster_size_before = sizes.get(key, 0)
        pick.draft_phase = draft_phase(pick.overall_pick, total_picks)

        position = pick.position
        pick.position_count_before = manager_counts.get(position, 0) if position else 0
        pick.open_starting_slots_before = _open_starter_count(roster, manager_filled)

        if position is not None:
            target = available_slot_for(position, manager_filled, roster)
            pick.filled_starting_slot = target is not None
            if target is not None:
                manager_filled[target] = manager_filled.get(target, 0) + 1
            # Run detection over the configured window, before this pick lands.
            window = recent[-int(config.run_window_picks):]
            same = window.count(position)
            pick.position_picks_in_window = same
            pick.continued_run = same >= int(config.run_threshold_picks)
            pick.started_run = same == 0
            owned = owned_by_team.setdefault(key, {})
            pick.was_stack = _is_stack(pick, owned)
            pick.was_handcuff = _is_handcuff(pick, owned)
            if pick.nfl_team:
                owned.setdefault(pick.nfl_team.upper(), set()).add(position)
            recent.append(position)
            manager_counts[position] = manager_counts.get(position, 0) + 1
        else:
            pick.filled_starting_slot = False
            pick.position_picks_in_window = 0
            pick.continued_run = False
            pick.started_run = False

        following = next_pick_by_manager.get((key, pick.overall_pick))
        pick.picks_until_next = (
            following - pick.overall_pick - 1 if following is not None else None
        )
        pick.same_tier_remaining = _consume_tier(remaining_by_tier, pick)
        sizes[key] = sizes.get(key, 0) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Per-pick helpers
# ─────────────────────────────────────────────────────────────────────────────
def draft_phase(overall_pick: int, total_picks: int) -> str:
    """``early`` / ``middle`` / ``late`` — equal thirds of the draft."""
    if total_picks <= 0:
        return DRAFT_PHASES[0]
    fraction = (float(overall_pick) - 1.0) / float(total_picks)
    if fraction < 1 / 3:
        return DRAFT_PHASES[0]
    if fraction < 2 / 3:
        return DRAFT_PHASES[1]
    return DRAFT_PHASES[2]


def _adp_delta(pick: HistoricalPick, config: ProfileEstimationConfig) -> float | None:
    """ADP minus pick (positive = reach), discarding implausible values as bad data.

    Keeper picks are excluded: a keeper's "pick" is an accounting artefact, not a
    decision, so counting it would make every keeper league look like it reaches.
    """
    if pick.adp is None or pick.is_keeper:
        return None
    delta = float(pick.adp) - float(pick.overall_pick)
    if abs(delta) > float(config.reach_clip_picks):
        return None
    return delta


def _annotate_rank_inversions(picks: Sequence[HistoricalPick]) -> None:
    """Count, per pick, how many later picks in the draft were ranked better.

    This is the scale-free counterpart to :attr:`HistoricalPick.rank_delta`. A
    manager who takes whoever the list says is next leaves nobody better-ranked
    behind them, so their count is zero whether the league has 4 teams or 14;
    ``rank - pick`` cannot make that distinction because it also absorbs every
    *other* manager's reaching, which grows with league size.

    Computed as a single reverse sweep over a sorted list of the ranks still to
    come, so the whole draft costs O(n log n) rather than O(n²).
    """
    ordered = sorted(picks, key=lambda p: p.overall_pick)
    later_ranks: list[float] = []
    for pick in reversed(ordered):
        rank = pick.platform_rank
        if rank is None:
            pick.rank_inversions = None
            continue
        # Everything already in `later_ranks` was drafted after this pick.
        pick.rank_inversions = bisect.bisect_left(later_ranks, float(rank))
        bisect.insort(later_ranks, float(rank))


def _open_starter_count(roster: RosterSettings, filled: Mapping[Slot, int]) -> int:
    """Starting seats still unfilled, given how many of each slot are used."""
    return sum(
        max(0, roster.count(slot) - int(filled.get(slot, 0)))
        for slot in roster.starting_slots
    )


PASS_CATCHERS: frozenset[Position] = frozenset({Position.WR, Position.TE})


def _is_stack(pick: HistoricalPick, owned: Mapping[str, set[Position]]) -> bool:
    """True when this pick pairs a quarterback with one of his pass-catchers.

    Owning an unrelated player from the same team (a running back, say) is not a
    stack — the correlation a stacker is buying is passer-to-receiver.
    """
    if not pick.nfl_team or pick.position is None:
        return False
    held = owned.get(pick.nfl_team.upper())
    if not held:
        return False
    if pick.position is Position.QB:
        return bool(held & PASS_CATCHERS)
    if pick.position in PASS_CATCHERS:
        return Position.QB in held
    return False


def _is_handcuff(pick: HistoricalPick, owned: Mapping[str, set[Position]]) -> bool:
    """True when this pick adds a second running back from a team already owned."""
    if not pick.nfl_team or pick.position is not Position.RB:
        return False
    held = owned.get(pick.nfl_team.upper())
    return bool(held and Position.RB in held)


def _fill_round_numbers(picks: Sequence[HistoricalPick], team_count: int) -> None:
    """Derive round / pick-in-round from the overall pick when the file lacked them."""
    if team_count < 1:
        return
    for pick in picks:
        if pick.round_number is None:
            pick.round_number = (pick.overall_pick - 1) // team_count + 1
        if pick.pick_in_round is None:
            pick.pick_in_round = (pick.overall_pick - 1) % team_count + 1


def _next_pick_lookup(
    picks: Sequence[HistoricalPick],
) -> dict[tuple[str, int], int]:
    """``(manager_key, overall_pick)`` → that manager's following overall pick."""
    by_manager: dict[str, list[int]] = {}
    for pick in picks:
        by_manager.setdefault(pick.manager_key, []).append(pick.overall_pick)
    out: dict[tuple[str, int], int] = {}
    for key, numbers in by_manager.items():
        numbers.sort()
        for index, number in enumerate(numbers[:-1]):
            out[(key, number)] = numbers[index + 1]
    return out


def _tier_inventory(picks: Sequence[HistoricalPick]) -> dict[tuple[Position, int], int]:
    """How many players of each (position, tier) appear in this draft.

    A within-draft inventory is the only tier count available from history alone:
    the undrafted remainder of a tier is unknown, so ``same_tier_remaining``
    counts players in that tier still to be drafted *in this draft*.
    """
    inventory: dict[tuple[Position, int], int] = {}
    for pick in picks:
        if pick.position is None or pick.tier is None:
            continue
        key = (pick.position, int(pick.tier))
        inventory[key] = inventory.get(key, 0) + 1
    return inventory


def _consume_tier(
    inventory: dict[tuple[Position, int], int], pick: HistoricalPick
) -> int | None:
    """Decrement and return how many of this player's tier remained after them."""
    if pick.position is None or pick.tier is None:
        return None
    key = (pick.position, int(pick.tier))
    remaining = inventory.get(key)
    if remaining is None:
        return None
    remaining = max(0, remaining - 1)
    inventory[key] = remaining
    return remaining


def _backfill_from_pool(
    picks: Sequence[HistoricalPick], pool: PlayerPool | None
) -> None:
    """Fill missing position / team / ADP / tier fields from a player pool.

    Only *missing* fields are touched — a value in the user's file always wins,
    since the pool may describe a different season.
    """
    if pool is None:
        return
    index = {player_key(p.name): p for p in pool}
    matched = 0
    for pick in picks:
        player = index.get(player_key(pick.player_name))
        if player is None:
            continue
        matched += 1
        if pick.position is None:
            pick.position = player.position
        if not pick.nfl_team:
            pick.nfl_team = player.nfl_team
        if pick.adp is None:
            pick.adp = player.adp_for()
        if pick.platform_rank is None:
            pick.platform_rank = player.rank_for()
        if pick.projection is None:
            pick.projection = player.projection
        if pick.tier is None:
            pick.tier = player.tier
        if not pick.is_rookie:
            pick.is_rookie = bool(player.is_rookie)
    if picks:
        LOGGER.debug(
            "Backfilled %d/%d historical picks from the player pool",
            matched, len(picks),
        )


# ─────────────────────────────────────────────────────────────────────────────
# League-wide aggregates
# ─────────────────────────────────────────────────────────────────────────────
def summarize_history(
    history: DraftHistory, *, early_rounds: int = 3
) -> HistoryStats:
    """League-wide positional timing and share, used as the shrinkage target."""
    picks = [p for p in history.all_picks if not p.is_keeper]
    if not picks:
        return HistoryStats(early_rounds=int(early_rounds))

    sums: dict[Position, float] = {}
    counts: dict[Position, int] = {}
    for pick in picks:
        if pick.position is None:
            continue
        sums[pick.position] = sums.get(pick.position, 0.0) + float(pick.overall_pick)
        counts[pick.position] = counts.get(pick.position, 0) + 1

    total = sum(counts.values()) or 1
    mean_pick = {pos: sums[pos] / counts[pos] for pos in counts}
    share = {pos: counts[pos] / total for pos in counts}

    # Early-round share, computed exactly rather than approximated: it is the
    # denominator for every early-round positional-bias comparison.
    early_counts: dict[Position, int] = {}
    for pick in picks:
        if pick.position is None:
            continue
        if int(pick.round_number or 1) <= int(early_rounds):
            early_counts[pick.position] = early_counts.get(pick.position, 0) + 1
    early_total = sum(early_counts.values()) or 1
    early_share = {pos: n / early_total for pos, n in early_counts.items()}

    # Average round of each manager's *first* pick at a position, per season, so
    # "when does this league take its first QB" has a league-wide answer.
    first_rounds: dict[Position, list[float]] = {}
    for draft in history.drafts:
        seen: set[tuple[str, Position]] = set()
        for pick in sorted(draft.picks, key=lambda p: p.overall_pick):
            if pick.position is None or pick.is_keeper:
                continue
            marker = (pick.manager_key, pick.position)
            if marker in seen:
                continue
            seen.add(marker)
            first_rounds.setdefault(pick.position, []).append(
                float(pick.round_number or 1)
            )

    return HistoryStats(
        pick_count=len(picks),
        manager_count=len({p.manager_key for p in picks}),
        positions_seen=tuple(sorted(counts, key=lambda p: str(p))),
        mean_pick_by_position=mean_pick,
        share_by_position=share,
        first_pick_by_position={
            pos: sum(values) / len(values) for pos, values in first_rounds.items()
        },
        early_share_by_position=early_share,
        early_rounds=int(early_rounds),
    )


def feature_frame(history: DraftHistory):
    """Annotated picks as a frame, for the model-evaluation and debug views."""
    return history.to_frame()


__all__ = [
    "HistoryStats", "annotate_history", "annotate_draft", "summarize_history",
    "draft_phase", "feature_frame", "DRAFT_PHASES",
]
