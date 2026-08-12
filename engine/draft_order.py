"""Draft order generation for every supported draft type.

The order is computed once, up front, as a list of :class:`PickSlot` values.
Keeper picks are marked in place rather than removed, so pick numbering matches
what the user sees on their platform and the board has no holes.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

from core.config import LeagueConfig
from core.enums import DraftType
from core.validation import ConfigurationError
from models.draft import PickSlot
from models.league import Keeper, League

LOGGER = logging.getLogger("fantasy_mock_draft.draft_order")


def round_slot_order(config: LeagueConfig, round_number: int) -> list[int]:
    """Draft-slot order for a single round, per the league's draft type.

    * ``snake`` — odd rounds ascending, even rounds descending.
    * ``linear`` — every round ascending.
    * ``third_round_reversal`` — snake, except the reversal round repeats the
      previous round's direction, flipping the pattern from there on.
    * ``custom`` — the explicit order given for that round, falling back to
      snake for rounds the user did not specify.
    * ``auction`` — nominal ascending order (auctions are not simulated in
      Phase 1; :func:`core.validation.validate_league` warns about this).
    """
    team_count = int(config.team_count)
    if team_count < 1:
        raise ConfigurationError("team_count must be at least 1")
    if round_number < 1:
        raise ConfigurationError("round_number is 1-based")

    ascending = list(range(1, team_count + 1))
    descending = list(reversed(ascending))

    draft_type = config.draft_type
    if draft_type is DraftType.LINEAR or draft_type is DraftType.AUCTION:
        return ascending

    if draft_type is DraftType.CUSTOM:
        explicit = config.custom_round_order.get(round_number)
        if explicit:
            return list(explicit)
        return ascending if round_number % 2 else descending

    if draft_type is DraftType.THIRD_ROUND_REVERSAL:
        reversal = int(config.reversal_round)
        if round_number < reversal:
            return ascending if round_number % 2 else descending
        # From the reversal round on, the direction is flipped relative to plain
        # snake: the reversal round repeats the prior round's direction.
        return descending if round_number % 2 else ascending

    # Snake (the default).
    return ascending if round_number % 2 else descending


def snake_position_in_round(team_count: int, round_number: int, draft_slot: int) -> int:
    """Where a slot picks within a round under plain snake ordering (1-based)."""
    return draft_slot if round_number % 2 else team_count - draft_slot + 1


def build_pick_slots(
    config: LeagueConfig,
    keeper_picks: Mapping[int, Keeper] | None = None,
) -> list[PickSlot]:
    """Generate every pick in the draft, in order, marking keeper picks.

    ``keeper_picks`` maps an overall pick number to the keeper occupying it, as
    produced by :meth:`models.league.League.consumed_picks`.
    """
    keeper_picks = dict(keeper_picks or {})
    slots: list[PickSlot] = []
    overall = 0
    for round_number in range(1, int(config.rounds) + 1):
        order = round_slot_order(config, round_number)
        for position, draft_slot in enumerate(order, start=1):
            overall += 1
            keeper = keeper_picks.get(overall)
            slots.append(
                PickSlot(
                    overall_pick=overall,
                    round_number=round_number,
                    pick_in_round=position,
                    draft_slot=int(draft_slot),
                    is_keeper_pick=keeper is not None,
                    keeper_player_name=keeper.player_name if keeper else None,
                )
            )
    return slots


def build_order_for_league(league: League) -> list[PickSlot]:
    """Convenience wrapper resolving the league's keepers onto its draft order."""
    return build_pick_slots(league.config, league.consumed_picks())


def validate_custom_order(config: LeagueConfig) -> list[str]:
    """Human-readable problems with a custom order (empty when it is usable)."""
    if config.draft_type is not DraftType.CUSTOM:
        return []
    problems: list[str] = []
    expected = set(range(1, config.team_count + 1))
    for round_number, order in sorted(config.custom_round_order.items()):
        if not 1 <= round_number <= config.rounds:
            problems.append(f"Round {round_number} is outside rounds 1–{config.rounds}.")
            continue
        given = list(order)
        if len(given) != config.team_count:
            problems.append(
                f"Round {round_number} lists {len(given)} slot(s) but the league has "
                f"{config.team_count} teams."
            )
        missing = sorted(expected - set(given))
        duplicated = sorted({s for s in given if given.count(s) > 1})
        if missing:
            problems.append(
                f"Round {round_number} is missing slot(s): "
                + ", ".join(map(str, missing))
            )
        if duplicated:
            problems.append(
                f"Round {round_number} repeats slot(s): "
                + ", ".join(map(str, duplicated))
            )
    return problems


def picks_for_slot(slots: Sequence[PickSlot], draft_slot: int) -> list[PickSlot]:
    """Every pick belonging to one draft slot, in order."""
    return [s for s in slots if s.draft_slot == int(draft_slot)]


def next_pick_for_slot(
    slots: Sequence[PickSlot], draft_slot: int, after_overall: int
) -> PickSlot | None:
    """The slot's next pick strictly after ``after_overall``."""
    for slot in slots:
        if slot.draft_slot == int(draft_slot) and slot.overall_pick > after_overall:
            return slot
    return None


def picks_until_next_turn(
    slots: Sequence[PickSlot], draft_slot: int, current_overall: int
) -> int | None:
    """How many other picks happen before this slot picks again.

    ``0`` means they pick again immediately (a snake turn). ``None`` means they
    have no picks left.
    """
    following = next_pick_for_slot(slots, draft_slot, current_overall)
    if following is None:
        return None
    return following.overall_pick - current_overall - 1


def draft_order_frame(slots: Sequence[PickSlot], league: League | None = None):
    """Tabulate the order for display, naming managers when a league is given."""
    import pandas as pd

    names = {m.draft_slot: m.name for m in (league.managers if league else [])}
    return pd.DataFrame(
        [
            {
                "overall": s.overall_pick,
                "round": s.round_number,
                "pick": s.pick_in_round,
                "label": s.label,
                "draft_slot": s.draft_slot,
                "manager": names.get(s.draft_slot, f"Slot {s.draft_slot}"),
                "keeper": s.keeper_player_name or "",
            }
            for s in slots
        ]
    )


__all__ = [
    "round_slot_order", "snake_position_in_round", "build_pick_slots",
    "build_order_for_league", "validate_custom_order", "picks_for_slot",
    "next_pick_for_slot", "picks_until_next_turn", "draft_order_frame",
]
