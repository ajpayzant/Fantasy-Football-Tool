"""Mutable draft state: the single source of truth during a mock draft.

Everything the pick model needs is derived from here — who is on the clock, who
is available, each team's roster, recent positional runs, and how long until a
given team picks again. State changes only through :meth:`DraftState.make_pick`
and :meth:`DraftState.undo`, so history stays consistent and the UI can offer
undo/redo without special cases.

Callers get defensive copies of internal collections; nothing outside this class
mutates its rosters or pick list.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from core.config import LeagueConfig, SimulationConfig
from core.enums import DraftStatus, Position, Slot
from core.validation import ConfigurationError
from engine.draft_order import build_pick_slots, picks_until_next_turn
from models.draft import Pick, PickSlot, TeamRoster
from models.league import League
from models.player import Player, PlayerPool

LOGGER = logging.getLogger("fantasy_mock_draft.draft_state")


@dataclass(slots=True)
class RunSnapshot:
    """How hot each position is right now, over several look-back windows."""

    counts_by_window: dict[int, dict[Position, int]] = field(default_factory=dict)
    picks_since_position: dict[Position, int | None] = field(default_factory=dict)

    def count(self, window: int, position: Position) -> int:
        return self.counts_by_window.get(window, {}).get(position, 0)

    def rate(self, window: int, position: Position) -> float:
        """Share of the last ``window`` picks spent on ``position``."""
        if window <= 0:
            return 0.0
        return self.count(window, position) / float(window)

    def hottest(self, window: int) -> Position | None:
        counts = self.counts_by_window.get(window, {})
        if not counts:
            return None
        return max(counts, key=lambda p: counts[p])


class DraftState:
    """The live state of one mock draft.

    Construct with a league and a player pool; the draft order is generated up
    front and keeper picks are applied immediately so the state is always
    consistent with the board the user sees.
    """

    __slots__ = (
        "league", "pool", "settings", "order", "picks", "rosters",
        "_drafted_ids", "_available_ids", "_undo_stack", "_redo_stack",
        "_rng", "_seed", "status", "_keeper_names", "_strategy_notes",
        "_reserved_ids", "_forfeited", "_avail_cache", "_avail_pos_cache",
    )

    def __init__(
        self,
        league: League,
        pool: PlayerPool,
        settings: SimulationConfig | None = None,
        *,
        seed: int | None = None,
        apply_keepers: bool = True,
    ) -> None:
        self.league = league
        self.pool = pool
        self.settings = settings or SimulationConfig()
        self.order: list[PickSlot] = build_pick_slots(
            league.config, league.consumed_picks()
        )
        self.picks: list[Pick] = []
        self.rosters: dict[int, TeamRoster] = {
            manager.draft_slot: TeamRoster(
                manager_name=manager.name,
                draft_slot=manager.draft_slot,
                settings=league.config.roster,
            )
            for manager in league.managers
        }
        self._drafted_ids: set[str] = set()
        self._available_ids: list[str] = [p.player_id for p in pool]
        self._undo_stack: list[Pick] = []
        self._redo_stack: list[Pick] = []
        self._keeper_names: dict[int, list[str]] = {}
        self._strategy_notes: dict[int, list[str]] = {}
        self._reserved_ids: dict[str, int] = {}
        """Player id → the overall pick reserving them for a keeper."""
        self._forfeited: set[int] = set()
        self._avail_cache: list[Player] | None = None
        """Board-ordered available players, invalidated whenever availability moves.

        Not an optimisation for its own sake: the pick model's scarcity term asks
        "how many are left at this position" once per candidate, and the
        recommendation engine's rollouts repeat a whole draft ~120 times. Rebuilt
        naively that is one full sort of the pool per candidate per pick, which
        profiled at 5.1s of a 7.9s single draft on a 280-player pool — rollouts
        would have multiplied it by the simulation count. Every mutation path
        (:meth:`_commit`, :meth:`undo`, keeper reservation) clears it.
        """
        self._avail_pos_cache: dict[Position, list[Player]] | None = None

        self._seed = seed if seed is not None else self.settings.random_seed
        self._rng = random.Random(self._seed)
        self.status = DraftStatus.NOT_STARTED

        if apply_keepers:
            self._apply_keepers()

    # -- identity --------------------------------------------------------
    @property
    def config(self) -> LeagueConfig:
        return self.league.config

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def rng(self) -> random.Random:
        """The seeded RNG. Shared deliberately so one seed drives the whole run."""
        return self._rng

    def reseed(self, seed: int | None) -> None:
        """Reset the RNG. Only valid before the draft starts."""
        if self.picks:
            raise ConfigurationError("Cannot reseed a draft that has already started")
        self._seed = seed
        self._rng = random.Random(seed)

    def rng_state(self) -> tuple:
        """The generator's exact position in its stream.

        Needed to save and resume a draft. The seed alone is not enough: the stream
        advances with every simulated pick, so a draft rebuilt from the seed would
        start drawing from the beginning again and the opponents would behave
        differently from the pick after the resume onward — with nothing on screen to
        explain why.
        """
        return self._rng.getstate()

    def set_rng_state(self, state: Any) -> bool:
        """Put the generator back where it was. ``False`` if the state is unusable.

        A state saved by a different Python build can be rejected by ``setstate``.
        That is worth a warning and a slightly different room, not a lost draft, so
        the failure is reported rather than raised.
        """
        try:
            self._rng.setstate(state)
        except (TypeError, ValueError) as error:
            LOGGER.warning("Could not restore the draft's RNG position: %s", error)
            return False
        return True

    # -- keepers ---------------------------------------------------------
    def _apply_keepers(self) -> None:
        """Reserve keeper players and resolve any keeper picks at the front.

        Keepers are *not* committed up front: the clock is derived from the
        number of picks made, so pre-committing a keeper assigned to a later
        pick would skip pick 1 and re-use that later slot. Instead the players
        are reserved as unavailable, and each keeper pick is committed when the
        clock reaches it (see :meth:`_resolve_keeper_picks`).

        A keeper whose player is missing from the pool is logged and its pick is
        left open, rather than silently producing a wrong roster.
        """
        for slot in self.order:
            if not slot.is_keeper_pick or not slot.keeper_player_name:
                continue
            player = self.pool.get(slot.keeper_player_name)
            if player is None:
                LOGGER.warning(
                    "Keeper '%s' is not in the player pool; pick %s left open",
                    slot.keeper_player_name, slot.label,
                )
                continue
            # Reserved so no other team can draft them before their keeper pick.
            self._reserved_ids[player.player_id] = slot.overall_pick
            self._keeper_names.setdefault(slot.draft_slot, []).append(player.name)
        self._invalidate_availability()
        self._resolve_keeper_picks()

    def _resolve_keeper_picks(self) -> None:
        """Auto-commit keeper selections while the clock sits on a keeper pick."""
        while True:
            slot = self.current_slot
            if slot is None or not slot.is_keeper_pick or not slot.keeper_player_name:
                return
            player = self.pool.get(slot.keeper_player_name)
            if player is None or player.player_id in self._drafted_ids:
                # Nothing to commit; treat the pick as forfeited so the clock
                # can move on instead of deadlocking.
                LOGGER.warning("Keeper pick %s forfeited (player unavailable)", slot.label)
                self._forfeited.add(slot.overall_pick)
                self.picks.append(self._forfeit_pick(slot))
                continue
            self._reserved_ids.pop(player.player_id, None)
            self._commit(player, slot, is_keeper=True, record_undo=False)

    def _forfeit_pick(self, slot: PickSlot) -> Pick:
        """A placeholder pick so a forfeited keeper slot still occupies a number."""
        manager = self.league.manager_by_slot(slot.draft_slot)
        return Pick(
            overall_pick=slot.overall_pick,
            round_number=slot.round_number,
            pick_in_round=slot.pick_in_round,
            draft_slot=slot.draft_slot,
            manager_name=manager.name if manager else f"Slot {slot.draft_slot}",
            player_id="",
            player_name=slot.keeper_player_name or "(forfeited)",
            position=Position.RB,
            is_keeper=True,
            assigned_slot=Slot.BENCH,
            explanation="Keeper pick forfeited: the player was not in the pool.",
        )

    def keepers_for_slot(self, draft_slot: int) -> list[str]:
        return list(self._keeper_names.get(int(draft_slot), []))

    @property
    def reserved_ids(self) -> frozenset[str]:
        """Players held for an upcoming keeper pick — undraftable by others."""
        return frozenset(self._reserved_ids)

    # -- clock -----------------------------------------------------------
    @property
    def pick_index(self) -> int:
        """0-based index of the pick on the clock."""
        return len(self.picks)

    @property
    def current_slot(self) -> PickSlot | None:
        """The pick on the clock, or ``None`` when the draft is complete."""
        if self.pick_index >= len(self.order):
            return None
        return self.order[self.pick_index]

    @property
    def is_complete(self) -> bool:
        return self.pick_index >= len(self.order)

    @property
    def current_round(self) -> int:
        slot = self.current_slot
        return slot.round_number if slot else int(self.config.rounds)

    @property
    def on_the_clock_slot(self) -> int | None:
        slot = self.current_slot
        return slot.draft_slot if slot else None

    def manager_on_clock(self):
        """The :class:`models.manager.Manager` currently picking, if any."""
        slot = self.current_slot
        if slot is None:
            return None
        return self.league.manager_by_slot(slot.draft_slot)

    @property
    def is_user_on_clock(self) -> bool:
        slot = self.current_slot
        return bool(slot and slot.draft_slot in self.league.user_slots)

    def picks_until_turn(self, draft_slot: int) -> int | None:
        """Picks between now and ``draft_slot``'s next turn (0 = on the clock)."""
        current = self.current_slot
        if current is None:
            return None
        if current.draft_slot == int(draft_slot):
            return 0
        return picks_until_next_turn(
            self.order, int(draft_slot), current.overall_pick - 1
        )

    def picks_until_following_turn(self, draft_slot: int) -> int | None:
        """Picks between the pick on the clock and this slot's *next* one after it.

        Distinct from :meth:`picks_until_turn`, and the distinction matters. That
        method answers "how long until this slot picks?", which is 0 for whoever is
        on the clock. This one answers "once this pick is spent, how long until
        they pick again?" — which is the question behind every wait-or-take
        decision: availability, scarcity, and the roster-imbalance horizon all need
        the gap *after* the current selection, not before it.

        ``None`` when the slot has no further picks. Back-to-back turns give 0.
        """
        current = self.current_slot
        if current is None:
            return None
        return picks_until_next_turn(
            self.order, int(draft_slot), current.overall_pick
        )

    def next_pick_numbers(self, draft_slot: int, count: int = 3) -> list[int]:
        """The slot's next ``count`` upcoming overall pick numbers."""
        current = self.current_slot
        floor = 0 if current is None else current.overall_pick - 1
        return [
            s.overall_pick for s in self.order
            if s.draft_slot == int(draft_slot) and s.overall_pick > floor
        ][:count]

    # -- availability ----------------------------------------------------
    @property
    def drafted_ids(self) -> frozenset[str]:
        return frozenset(self._drafted_ids)

    def is_available(self, player_id: str) -> bool:
        """True when the player is neither drafted nor held for a keeper pick."""
        return player_id not in self._drafted_ids and player_id not in self._reserved_ids

    def _invalidate_availability(self) -> None:
        """Drop the cached board. Called by every path that drafts or un-drafts."""
        self._avail_cache = None
        self._avail_pos_cache = None

    def available_players(self, limit: int | None = None) -> list[Player]:
        """Undrafted, unreserved players in draft-order preference, best first.

        The returned list is a slice, so callers cannot mutate the cache.
        """
        if self._avail_cache is None:
            players = [
                self.pool.require(pid) for pid in self._available_ids
                if self.is_available(pid)
            ]
            players.sort(key=self.pool.order_value)
            self._avail_cache = players
        cached = self._avail_cache
        return cached[:limit] if limit else cached[:]

    def available_at_position(self, position: Position, limit: int | None = None) -> list[Player]:
        """Available players at one position, best first.

        Bucketed by position on first use rather than filtered per call: the
        scarcity term asks this question once for every candidate at every pick.
        """
        if self._avail_pos_cache is None:
            buckets: dict[Position, list[Player]] = {}
            for player in self.available_players():
                buckets.setdefault(player.position, []).append(player)
            self._avail_pos_cache = buckets
        cached = self._avail_pos_cache.get(position, [])
        return cached[:limit] if limit else cached[:]

    def available_count(self) -> int:
        return sum(1 for pid in self._available_ids if self.is_available(pid))

    def best_available(self) -> Player | None:
        players = self.available_players(limit=1)
        return players[0] if players else None

    # -- rosters ---------------------------------------------------------
    def roster(self, draft_slot: int) -> TeamRoster:
        """The live roster for a slot (mutating it corrupts state — use copies)."""
        roster = self.rosters.get(int(draft_slot))
        if roster is None:
            raise ConfigurationError(f"No roster for draft slot {draft_slot}")
        return roster

    def roster_copy(self, draft_slot: int) -> TeamRoster:
        """A defensive copy, safe to mutate (used by simulation rollouts)."""
        return self.roster(draft_slot).copy()

    def current_roster(self) -> TeamRoster | None:
        slot = self.current_slot
        return self.roster(slot.draft_slot) if slot else None

    def picks_by_slot(self, draft_slot: int) -> list[Pick]:
        return [p for p in self.picks if p.draft_slot == int(draft_slot)]

    def picks_by_manager(self, manager_name: str) -> list[Pick]:
        return [p for p in self.picks if p.manager_name == manager_name]

    # -- positional runs -------------------------------------------------
    def run_snapshot(self, windows: Sequence[int] | None = None) -> RunSnapshot:
        """Positional pick counts over each look-back window.

        Keeper picks are excluded: they reflect last season's roster, not the
        momentum of the room.
        """
        windows = tuple(windows or self.settings.run_windows)
        real_picks = [p for p in self.picks if not p.is_keeper]
        snapshot = RunSnapshot()
        for window in windows:
            counts: dict[Position, int] = {}
            for pick in real_picks[-window:]:
                counts[pick.position] = counts.get(pick.position, 0) + 1
            snapshot.counts_by_window[window] = counts

        for position in Position:
            gap: int | None = None
            for distance, pick in enumerate(reversed(real_picks), start=1):
                if pick.position is position:
                    gap = distance - 1
                    break
            snapshot.picks_since_position[position] = gap
        return snapshot

    def position_counts_drafted(self) -> dict[Position, int]:
        """League-wide count of players taken at each position."""
        counts: dict[Position, int] = {}
        for pick in self.picks:
            counts[pick.position] = counts.get(pick.position, 0) + 1
        return counts

    # -- making picks ----------------------------------------------------
    def make_pick(
        self,
        player: Player | str,
        *,
        is_user_pick: bool | None = None,
        was_manual_override: bool = False,
        pick_probability: float | None = None,
        alternatives: Sequence[Mapping[str, Any]] | None = None,
        explanation: str = "",
    ) -> Pick:
        """Draft ``player`` at the current pick and advance the clock.

        Raises :class:`core.validation.ConfigurationError` when the draft is over
        or the player is already gone — both are programmer errors, since the UI
        only offers available players.
        """
        slot = self.current_slot
        if slot is None:
            raise ConfigurationError("The draft is already complete")

        resolved = player if isinstance(player, Player) else self.pool.require(player)
        if resolved.player_id in self._drafted_ids:
            raise ConfigurationError(
                f"{resolved.name} has already been drafted at pick "
                f"{self._pick_of(resolved.player_id)}"
            )
        reserved_at = self._reserved_ids.get(resolved.player_id)
        if reserved_at is not None:
            raise ConfigurationError(
                f"{resolved.name} is a keeper held for pick {reserved_at} and "
                "cannot be drafted here"
            )

        if is_user_pick is None:
            is_user_pick = slot.draft_slot in self.league.user_slots

        pick = self._commit(
            resolved, slot,
            is_user_pick=is_user_pick,
            was_manual_override=was_manual_override,
            pick_probability=pick_probability,
            alternatives=alternatives,
            explanation=explanation,
        )
        self._redo_stack.clear()
        if self.status is DraftStatus.NOT_STARTED:
            self.status = DraftStatus.IN_PROGRESS
        # The next pick may itself be a keeper pick; settle it now so the clock
        # always rests on a pick somebody actually has to make.
        self._resolve_keeper_picks()
        if self.is_complete:
            self.status = DraftStatus.COMPLETE
            LOGGER.info("Draft complete: %s picks", len(self.picks))
        return pick

    def _commit(
        self,
        player: Player,
        slot: PickSlot,
        *,
        is_keeper: bool = False,
        is_user_pick: bool = False,
        was_manual_override: bool = False,
        pick_probability: float | None = None,
        alternatives: Sequence[Mapping[str, Any]] | None = None,
        explanation: str = "",
        record_undo: bool = True,
    ) -> Pick:
        """Apply a selection to the roster and pick list. No validation here."""
        manager = self.league.manager_by_slot(slot.draft_slot)
        manager_name = manager.name if manager else f"Slot {slot.draft_slot}"
        roster = self.roster(slot.draft_slot)
        assigned = roster.add(player)

        pick = Pick(
            overall_pick=slot.overall_pick,
            round_number=slot.round_number,
            pick_in_round=slot.pick_in_round,
            draft_slot=slot.draft_slot,
            manager_name=manager_name,
            player_id=player.player_id,
            player_name=player.name,
            position=player.position,
            nfl_team=player.nfl_team,
            is_keeper=is_keeper,
            is_user_pick=is_user_pick,
            was_manual_override=was_manual_override,
            assigned_slot=assigned,
            adp_at_pick=player.adp_for(),
            platform_rank_at_pick=player.rank_for(),
            projection=player.projection,
            tier=player.tier,
            pick_probability=pick_probability,
            alternatives=[dict(a) for a in (alternatives or [])],
            explanation=explanation,
        )
        self.picks.append(pick)
        self._drafted_ids.add(player.player_id)
        self._invalidate_availability()
        if record_undo:
            self._undo_stack.append(pick)
        return pick

    def _pick_of(self, player_id: str) -> str:
        for pick in self.picks:
            if pick.player_id == player_id:
                return pick.label
        return "?"

    # -- undo / redo -----------------------------------------------------
    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> Pick | None:
        """Reverse the most recent user/AI pick.

        Keeper picks are auto-committed rather than chosen, so they are never on
        the undo stack; any keeper picks sitting after the undone pick are rolled
        back too and then re-applied, keeping the clock on a real decision.
        """
        if not self._undo_stack:
            return None
        pick = self._undo_stack.pop()
        self._rollback_keepers_after(pick.overall_pick)
        # The undo stack only ever holds the newest picks, so this is the last one.
        if self.picks and self.picks[-1] is pick:
            self.picks.pop()
        else:
            self.picks = [p for p in self.picks if p is not pick]
        self._drafted_ids.discard(pick.player_id)
        self._invalidate_availability()
        roster = self.roster(pick.draft_slot)
        roster.remove(pick.player_id)
        roster.rebuild(self._projection_map())
        self._redo_stack.append(pick)
        self.status = (
            DraftStatus.IN_PROGRESS if self.picks else DraftStatus.NOT_STARTED
        )
        self._resolve_keeper_picks()
        LOGGER.debug("Undid pick %s (%s)", pick.label, pick.player_name)
        return pick

    def _rollback_keepers_after(self, overall_pick: int) -> None:
        """Remove auto-committed keeper picks at or after ``overall_pick``."""
        while self.picks and self.picks[-1].overall_pick > int(overall_pick):
            trailing = self.picks[-1]
            if not trailing.is_keeper:
                break
            self.picks.pop()
            if trailing.player_id:
                self._drafted_ids.discard(trailing.player_id)
                self._reserved_ids[trailing.player_id] = trailing.overall_pick
                self._invalidate_availability()
                roster = self.roster(trailing.draft_slot)
                roster.remove(trailing.player_id)
                roster.rebuild(self._projection_map())
            self._forfeited.discard(trailing.overall_pick)

    def redo(self) -> Pick | None:
        """Re-apply the most recently undone pick, if the clock still matches."""
        if not self._redo_stack:
            return None
        pick = self._redo_stack.pop()
        slot = self.current_slot
        if slot is None or slot.overall_pick != pick.overall_pick:
            LOGGER.debug("Redo discarded: the clock moved past %s", pick.label)
            return None
        player = self.pool.get(pick.player_id)
        if player is None:
            return None
        return self._commit(
            player, slot,
            is_keeper=pick.is_keeper,
            is_user_pick=pick.is_user_pick,
            was_manual_override=pick.was_manual_override,
            pick_probability=pick.pick_probability,
            alternatives=pick.alternatives,
            explanation=pick.explanation,
        )

    def undo_to(self, overall_pick: int) -> int:
        """Undo back to just before ``overall_pick``. Returns picks reversed."""
        undone = 0
        while self._undo_stack and self._undo_stack[-1].overall_pick >= int(overall_pick):
            if self.undo() is None:
                break
            undone += 1
        return undone

    def _projection_map(self) -> dict[str, float]:
        return {
            p.player_id: float(p.projection or 0.0) for p in self.pool
        }

    # -- in-mock strategy notes ------------------------------------------
    def note_strategy(self, draft_slot: int, note: str) -> None:
        """Record an observation about a manager's behaviour during this mock."""
        self._strategy_notes.setdefault(int(draft_slot), []).append(note)

    def strategy_notes(self, draft_slot: int) -> list[str]:
        return list(self._strategy_notes.get(int(draft_slot), []))

    # -- snapshots -------------------------------------------------------
    def board_state(self) -> dict[str, Any]:
        """Compact snapshot for logging, exports and debugging."""
        slot = self.current_slot
        return {
            "status": str(self.status),
            "picks_made": len(self.picks),
            "total_picks": len(self.order),
            "on_the_clock": slot.label if slot else None,
            "on_the_clock_manager": (
                self.manager_on_clock().name if self.manager_on_clock() else None
            ),
            "available_players": self.available_count(),
            "seed": self._seed,
        }

    def summary_line(self) -> str:
        slot = self.current_slot
        if slot is None:
            return f"Draft complete — {len(self.picks)} picks made."
        manager = self.manager_on_clock()
        who = manager.label if manager else f"Slot {slot.draft_slot}"
        return (
            f"Pick {slot.label} (overall {slot.overall_pick} of {len(self.order)}) — "
            f"{who} on the clock."
        )

    def copy_for_simulation(self) -> "DraftState":
        """A deep-enough copy for rollouts: rosters and pick lists are cloned.

        The player pool is shared (it is read-only during a draft) and the RNG is
        re-created from the current seed so a rollout cannot disturb the parent's
        random stream.
        """
        clone = DraftState.__new__(DraftState)
        clone.league = self.league
        clone.pool = self.pool
        clone.settings = self.settings
        clone.order = self.order
        clone.picks = list(self.picks)
        clone.rosters = {slot: r.copy() for slot, r in self.rosters.items()}
        clone._drafted_ids = set(self._drafted_ids)
        clone._available_ids = list(self._available_ids)
        clone._undo_stack = []
        clone._redo_stack = []
        clone._keeper_names = {k: list(v) for k, v in self._keeper_names.items()}
        clone._reserved_ids = dict(self._reserved_ids)
        clone._forfeited = set(self._forfeited)
        clone._strategy_notes = {k: list(v) for k, v in self._strategy_notes.items()}
        clone._seed = self._seed
        clone._rng = random.Random(self._rng.random())
        clone.status = self.status
        # The availability cache is shared, not copied: it is a read-only
        # snapshot of the current board (callers only ever receive slices of it)
        # and the clone's first pick replaces its own reference rather than
        # mutating the list. Copying it would cost a sort per rollout.
        clone._avail_cache = self._avail_cache
        clone._avail_pos_cache = self._avail_pos_cache
        return clone


__all__ = ["DraftState", "RunSnapshot"]
