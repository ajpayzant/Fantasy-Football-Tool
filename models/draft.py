"""Pick records, team rosters, and historical draft containers.

These are the value types that flow between the engine, analytics, and
persistence layers. The mutable live-draft object is
:class:`engine.draft_state.DraftState`; everything here is either a record or a
derived view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from core.config import RosterSettings
from core.constants import SLOT_ELIGIBILITY, SLOT_FILL_PRIORITY
from core.enums import Position, Slot
from models.manager import normalize_manager_key
from models.player import Player

REACH_THRESHOLD_PICKS: float = 6.0
"""How far ahead of ADP a pick must be before it is labelled a reach.

Roughly half a round in a 12-team league: inside that, the gap says more about
which ADP source was used than about the manager's intent.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Picks
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class Pick:
    """A single selection in a mock draft."""

    overall_pick: int
    round_number: int
    pick_in_round: int
    draft_slot: int
    manager_name: str
    player_id: str
    player_name: str
    position: Position
    nfl_team: str = "FA"
    is_keeper: bool = False
    is_user_pick: bool = False
    was_manual_override: bool = False
    """True when the user picked on an AI team's behalf."""
    assigned_slot: Slot = Slot.BENCH
    adp_at_pick: float | None = None
    platform_rank_at_pick: float | None = None
    projection: float | None = None
    tier: int | None = None
    pick_probability: float | None = None
    """Model probability assigned to this player at this pick (AI picks)."""
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    """Top other candidates the model considered, for explainability."""
    explanation: str = ""

    def __post_init__(self) -> None:
        self.position = Position.coerce(self.position, Position.RB) or Position.RB
        self.assigned_slot = Slot.coerce(self.assigned_slot, Slot.BENCH) or Slot.BENCH

    @property
    def manager_key(self) -> str:
        return normalize_manager_key(self.manager_name)

    @property
    def adp_delta(self) -> float | None:
        """Picks *ahead* of ADP: ADP minus the actual pick.

        Positive = reached (taken earlier than the crowd would have). Negative =
        value (he fell past his ADP). A player with an ADP of 100 taken 5th is a
        95-pick reach, not a bargain.
        """
        if self.adp_at_pick is None:
            return None
        return float(self.adp_at_pick) - float(self.overall_pick)

    @property
    def is_reach(self) -> bool:
        delta = self.adp_delta
        return delta is not None and delta > REACH_THRESHOLD_PICKS

    @property
    def is_value(self) -> bool:
        delta = self.adp_delta
        return delta is not None and delta < -REACH_THRESHOLD_PICKS

    @property
    def label(self) -> str:
        return f"{self.round_number}.{self.pick_in_round:02d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_pick": self.overall_pick,
            "round_number": self.round_number,
            "pick_in_round": self.pick_in_round,
            "draft_slot": self.draft_slot,
            "manager_name": self.manager_name,
            "player_id": self.player_id,
            "player_name": self.player_name,
            "position": str(self.position),
            "nfl_team": self.nfl_team,
            "is_keeper": self.is_keeper,
            "is_user_pick": self.is_user_pick,
            "was_manual_override": self.was_manual_override,
            "assigned_slot": str(self.assigned_slot),
            "adp_at_pick": self.adp_at_pick,
            "platform_rank_at_pick": self.platform_rank_at_pick,
            "projection": self.projection,
            "tier": self.tier,
            "pick_probability": self.pick_probability,
            "alternatives": list(self.alternatives),
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Pick":
        raw = dict(raw or {})
        return cls(
            overall_pick=int(raw["overall_pick"]),
            round_number=int(raw["round_number"]),
            pick_in_round=int(raw["pick_in_round"]),
            draft_slot=int(raw["draft_slot"]),
            manager_name=raw.get("manager_name", ""),
            player_id=raw.get("player_id", ""),
            player_name=raw.get("player_name", ""),
            position=raw.get("position", Position.RB),
            nfl_team=raw.get("nfl_team", "FA"),
            is_keeper=bool(raw.get("is_keeper", False)),
            is_user_pick=bool(raw.get("is_user_pick", False)),
            was_manual_override=bool(raw.get("was_manual_override", False)),
            assigned_slot=raw.get("assigned_slot", Slot.BENCH),
            adp_at_pick=raw.get("adp_at_pick"),
            platform_rank_at_pick=raw.get("platform_rank_at_pick"),
            projection=raw.get("projection"),
            tier=raw.get("tier"),
            pick_probability=raw.get("pick_probability"),
            alternatives=list(raw.get("alternatives") or []),
            explanation=raw.get("explanation", ""),
        )

    @classmethod
    def from_selection(
        cls,
        *,
        slot_info: "PickSlot",
        manager_name: str,
        player: Player,
        assigned_slot: Slot = Slot.BENCH,
        is_user_pick: bool = False,
        is_keeper: bool = False,
        was_manual_override: bool = False,
        pick_probability: float | None = None,
        alternatives: Sequence[Mapping[str, Any]] = (),
        explanation: str = "",
    ) -> "Pick":
        """Build a pick from a scheduled slot plus the chosen player."""
        return cls(
            overall_pick=slot_info.overall_pick,
            round_number=slot_info.round_number,
            pick_in_round=slot_info.pick_in_round,
            draft_slot=slot_info.draft_slot,
            manager_name=manager_name,
            player_id=player.player_id,
            player_name=player.name,
            position=player.position,
            nfl_team=player.nfl_team,
            is_keeper=is_keeper,
            is_user_pick=is_user_pick,
            was_manual_override=was_manual_override,
            assigned_slot=assigned_slot,
            adp_at_pick=player.adp_for(),
            platform_rank_at_pick=player.rank_for(),
            projection=player.projection,
            tier=player.tier,
            pick_probability=pick_probability,
            alternatives=[dict(a) for a in alternatives],
            explanation=explanation,
        )


@dataclass(frozen=True, slots=True)
class PickSlot:
    """One scheduled position in the draft order."""

    overall_pick: int
    round_number: int
    pick_in_round: int
    draft_slot: int
    is_keeper_pick: bool = False
    keeper_player_name: str | None = None

    @property
    def label(self) -> str:
        return f"{self.round_number}.{self.pick_in_round:02d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_pick": self.overall_pick,
            "round_number": self.round_number,
            "pick_in_round": self.pick_in_round,
            "draft_slot": self.draft_slot,
            "is_keeper_pick": self.is_keeper_pick,
            "keeper_player_name": self.keeper_player_name,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PickSlot":
        return cls(
            overall_pick=int(raw["overall_pick"]),
            round_number=int(raw["round_number"]),
            pick_in_round=int(raw["pick_in_round"]),
            draft_slot=int(raw["draft_slot"]),
            is_keeper_pick=bool(raw.get("is_keeper_pick", False)),
            keeper_player_name=raw.get("keeper_player_name"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Rosters
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class TeamRoster:
    """A single team's drafted players, with lineup assignment.

    Slot assignment is recomputed from scratch by :meth:`rebuild` so it is
    always optimal for the current player set — greedy fill of the most
    restrictive slots first, then best-projection players into flex seats.
    """

    manager_name: str
    draft_slot: int
    settings: RosterSettings
    player_ids: list[str] = field(default_factory=list)
    positions: dict[str, Position] = field(default_factory=dict)
    """player_id → position, so counts work without the pool."""
    lineup: dict[Slot, list[str]] = field(default_factory=dict)
    bench: list[str] = field(default_factory=list)
    version: int = 0
    """Bumped by every :meth:`rebuild`, which is the single funnel all mutation
    goes through. Consumers that memoize roster facts — :class:`engine.pick_model.RosterView`
    — compare it so a roster mutated underneath them invalidates rather than
    returning a stale answer."""

    # -- counts ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.player_ids)

    @property
    def is_full(self) -> bool:
        return len(self.player_ids) >= self.settings.roster_size

    def position_counts(self) -> dict[Position, int]:
        counts: dict[Position, int] = {}
        for pid in self.player_ids:
            position = self.positions.get(pid)
            if position is not None:
                counts[position] = counts.get(position, 0) + 1
        return counts

    def count_at(self, position: Position) -> int:
        return self.position_counts().get(position, 0)

    def filled_starting_slots(self) -> dict[Slot, int]:
        return {slot: len(pids) for slot, pids in self.lineup.items() if pids}

    def open_starting_slots(self) -> dict[Slot, int]:
        """Slot → how many seats remain unfilled."""
        filled = self.filled_starting_slots()
        return {
            slot: self.settings.count(slot) - filled.get(slot, 0)
            for slot in self.settings.starting_slots
            if self.settings.count(slot) - filled.get(slot, 0) > 0
        }

    @property
    def open_starter_count(self) -> int:
        return sum(self.open_starting_slots().values())

    @property
    def bench_open(self) -> int:
        return max(0, self.settings.bench_total - len(self.bench))

    def has_position(self, position: Position) -> bool:
        return self.count_at(position) > 0

    def at_position_limit(self, position: Position) -> bool:
        maximum = self.settings.max_for(position)
        return maximum is not None and self.count_at(position) >= maximum

    # -- mutation --------------------------------------------------------
    def add(self, player: Player) -> Slot:
        """Add a player and return the slot they were assigned to."""
        self.player_ids.append(player.player_id)
        self.positions[player.player_id] = player.position
        return self.rebuild_and_slot_of(player.player_id)

    def remove(self, player_id: str) -> None:
        if player_id in self.player_ids:
            self.player_ids.remove(player_id)
        self.positions.pop(player_id, None)
        self.rebuild()

    def rebuild(self, projections: Mapping[str, float] | None = None) -> None:
        """Recompute the optimal lineup / bench split from ``player_ids``."""
        projections = projections or {}
        self.version += 1
        self.lineup = {}
        self.bench = []

        # Best players first so the strongest fill starting slots.
        ordered = sorted(
            self.player_ids,
            key=lambda pid: -float(projections.get(pid, 0.0)),
        )
        # Two passes: dedicated slots (most restrictive) then flex slots.
        remaining: list[str] = []
        for pid in ordered:
            position = self.positions.get(pid)
            if position is None:
                remaining.append(pid)
                continue
            placed = False
            for slot in SLOT_FILL_PRIORITY:
                seats = self.settings.count(slot)
                if not seats:
                    continue
                eligible = SLOT_ELIGIBILITY.get(slot, frozenset())
                # Dedicated pass: only single-position slots.
                if len(eligible) != 1 or position not in eligible:
                    continue
                if len(self.lineup.get(slot, [])) < seats:
                    self.lineup.setdefault(slot, []).append(pid)
                    placed = True
                    break
            if not placed:
                remaining.append(pid)

        for pid in remaining:
            position = self.positions.get(pid)
            placed = False
            if position is not None:
                for slot in SLOT_FILL_PRIORITY:
                    seats = self.settings.count(slot)
                    if not seats:
                        continue
                    eligible = SLOT_ELIGIBILITY.get(slot, frozenset())
                    if len(eligible) <= 1 or position not in eligible:
                        continue
                    if len(self.lineup.get(slot, [])) < seats:
                        self.lineup.setdefault(slot, []).append(pid)
                        placed = True
                        break
            if not placed:
                self.bench.append(pid)

    def rebuild_and_slot_of(self, player_id: str) -> Slot:
        self.rebuild()
        return self.slot_of(player_id)

    def slot_of(self, player_id: str) -> Slot:
        for slot, pids in self.lineup.items():
            if player_id in pids:
                return slot
        return Slot.BENCH

    def starters(self) -> list[str]:
        out: list[str] = []
        for slot in SLOT_FILL_PRIORITY:
            out.extend(self.lineup.get(slot, []))
        for slot, pids in self.lineup.items():
            if slot not in SLOT_FILL_PRIORITY:
                out.extend(pids)
        return out

    def copy(self) -> "TeamRoster":
        clone = TeamRoster(
            manager_name=self.manager_name,
            draft_slot=self.draft_slot,
            settings=self.settings,
            player_ids=list(self.player_ids),
            positions=dict(self.positions),
        )
        clone.lineup = {slot: list(pids) for slot, pids in self.lineup.items()}
        clone.bench = list(self.bench)
        clone.version = self.version
        return clone

    def to_dict(self) -> dict[str, Any]:
        return {
            "manager_name": self.manager_name,
            "draft_slot": self.draft_slot,
            "player_ids": list(self.player_ids),
            "positions": {pid: str(pos) for pid, pos in self.positions.items()},
            "lineup": {str(slot): list(pids) for slot, pids in self.lineup.items()},
            "bench": list(self.bench),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], settings: RosterSettings) -> "TeamRoster":
        raw = dict(raw or {})
        roster = cls(
            manager_name=raw.get("manager_name", ""),
            draft_slot=int(raw.get("draft_slot", 1)),
            settings=settings,
            player_ids=list(raw.get("player_ids") or []),
            positions={
                pid: Position.coerce(pos, Position.RB) or Position.RB
                for pid, pos in (raw.get("positions") or {}).items()
            },
        )
        roster.rebuild()
        return roster


# ─────────────────────────────────────────────────────────────────────────────
# Historical drafts
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class HistoricalPick:
    """One pick from a previous draft, with every context field the model uses.

    Fields after ``draft_date`` are engineered features filled in by
    :mod:`data.historical_drafts`; they are stored so profile building and
    backtesting never have to recompute them.
    """

    season: int
    manager_name: str
    overall_pick: int
    player_name: str
    league_name: str = ""
    platform: str = ""
    round_number: int | None = None
    pick_in_round: int | None = None
    position: Position | None = None
    nfl_team: str = ""
    adp: float | None = None
    platform_rank: float | None = None
    projection: float | None = None
    tier: int | None = None
    is_keeper: bool = False
    is_rookie: bool = False
    bye_week: int | None = None
    draft_date: str | None = None
    # -- engineered features --------------------------------------------
    adp_delta: float | None = None
    """ADP minus overall pick. Positive = reached ahead of ADP."""
    rank_delta: float | None = None
    rank_inversions: int | None = None
    """How many players taken *later* in this draft were ranked better.

    Zero for a manager drafting the ranking list verbatim, at any league size.
    :attr:`rank_delta` cannot measure that: ``rank − pick`` absorbs the whole
    league's reaching, so in a 12-team league it reads 11-20 picks even for a
    manager who never deviates. This counts only the manager's own deviations.
    """
    position_count_before: int = 0
    """How many of this position the manager already had."""
    roster_size_before: int = 0
    open_starting_slots_before: int = 0
    filled_starting_slot: bool = False
    draft_phase: str = ""
    """early | middle | late — thirds of the draft."""
    started_run: bool = False
    continued_run: bool = False
    was_stack: bool = False
    was_handcuff: bool = False
    picks_until_next: int | None = None
    same_tier_remaining: int | None = None
    position_picks_in_window: int = 0
    historical_pick_id: int | None = None

    def __post_init__(self) -> None:
        self.season = int(self.season)
        self.overall_pick = int(self.overall_pick)
        self.manager_name = str(self.manager_name).strip()
        self.player_name = str(self.player_name).strip()
        self.position = Position.coerce(self.position, None)

    @property
    def manager_key(self) -> str:
        return normalize_manager_key(self.manager_name)

    @property
    def reach_picks(self) -> float | None:
        """Picks *ahead* of ADP (positive = reach).

        Alias of :attr:`adp_delta`, kept because "reach" reads more clearly than
        "delta" at the call sites that only care about the reach direction.
        """
        if self.adp_delta is None:
            return None
        return float(self.adp_delta)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        raw = asdict(self)
        raw["position"] = str(self.position) if self.position else None
        return raw

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HistoricalPick":
        raw = dict(raw or {})
        known = {
            "season", "manager_name", "overall_pick", "player_name", "league_name",
            "platform", "round_number", "pick_in_round", "position", "nfl_team",
            "adp", "platform_rank", "projection", "tier", "is_keeper", "is_rookie",
            "bye_week", "draft_date", "adp_delta", "rank_delta", "rank_inversions",
            "position_count_before", "roster_size_before",
            "open_starting_slots_before", "filled_starting_slot", "draft_phase",
            "started_run", "continued_run", "was_stack", "was_handcuff",
            "picks_until_next", "same_tier_remaining", "position_picks_in_window",
            "historical_pick_id",
        }
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass(slots=True)
class HistoricalDraft:
    """All picks from one season of one league."""

    season: int
    league_name: str = ""
    platform: str = ""
    team_count: int | None = None
    rounds: int | None = None
    draft_date: str | None = None
    picks: list[HistoricalPick] = field(default_factory=list)
    draft_id: int | None = None
    source_file: str = ""

    @property
    def manager_names(self) -> list[str]:
        seen: dict[str, str] = {}
        for pick in self.picks:
            seen.setdefault(pick.manager_key, pick.manager_name)
        return list(seen.values())

    @property
    def pick_count(self) -> int:
        return len(self.picks)

    def picks_for(self, manager: str) -> list[HistoricalPick]:
        key = normalize_manager_key(manager)
        return sorted(
            (p for p in self.picks if p.manager_key == key),
            key=lambda p: p.overall_pick,
        )

    def infer_team_count(self) -> int:
        """Derive team count from distinct managers, falling back to pick math."""
        if self.team_count:
            return int(self.team_count)
        managers = len({p.manager_key for p in self.picks})
        return managers or 12

    def infer_rounds(self) -> int:
        if self.rounds:
            return int(self.rounds)
        teams = self.infer_team_count()
        if not self.picks:
            return 0
        return max(1, int(round(max(p.overall_pick for p in self.picks) / max(1, teams))))

    def to_frame(self) -> pd.DataFrame:
        if not self.picks:
            return pd.DataFrame()
        return pd.DataFrame([p.to_dict() for p in self.picks])

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "league_name": self.league_name,
            "platform": self.platform,
            "team_count": self.team_count,
            "rounds": self.rounds,
            "draft_date": self.draft_date,
            "picks": [p.to_dict() for p in self.picks],
            "draft_id": self.draft_id,
            "source_file": self.source_file,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HistoricalDraft":
        raw = dict(raw or {})
        return cls(
            season=int(raw.get("season", 0)),
            league_name=raw.get("league_name", ""),
            platform=raw.get("platform", ""),
            team_count=raw.get("team_count"),
            rounds=raw.get("rounds"),
            draft_date=raw.get("draft_date"),
            picks=[HistoricalPick.from_dict(p) for p in (raw.get("picks") or [])],
            draft_id=raw.get("draft_id"),
            source_file=raw.get("source_file", ""),
        )


@dataclass(slots=True)
class DraftHistory:
    """Every historical draft available for profile building."""

    drafts: list[HistoricalDraft] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.drafts)

    def __iter__(self) -> Iterable[HistoricalDraft]:
        return iter(self.drafts)

    @property
    def all_picks(self) -> list[HistoricalPick]:
        return [pick for draft in self.drafts for pick in draft.picks]

    @property
    def seasons(self) -> tuple[int, ...]:
        return tuple(sorted({d.season for d in self.drafts}))

    @property
    def latest_season(self) -> int | None:
        seasons = self.seasons
        return seasons[-1] if seasons else None

    def manager_names(self) -> list[str]:
        seen: dict[str, str] = {}
        for draft in self.drafts:
            for pick in draft.picks:
                seen.setdefault(pick.manager_key, pick.manager_name)
        return sorted(seen.values())

    def picks_for(self, manager: str) -> list[HistoricalPick]:
        key = normalize_manager_key(manager)
        return sorted(
            (p for p in self.all_picks if p.manager_key == key),
            key=lambda p: (p.season, p.overall_pick),
        )

    def drafts_for_manager(self, manager: str) -> list[HistoricalDraft]:
        key = normalize_manager_key(manager)
        return [d for d in self.drafts if any(p.manager_key == key for p in d.picks)]

    def before_season(self, season: int) -> "DraftHistory":
        """Subset used by backtests to avoid look-ahead bias."""
        return DraftHistory([d for d in self.drafts if d.season < int(season)])

    def for_season(self, season: int) -> HistoricalDraft | None:
        for draft in self.drafts:
            if draft.season == int(season):
                return draft
        return None

    def add(self, draft: HistoricalDraft) -> None:
        """Add a draft, replacing any existing draft for the same season+league."""
        self.drafts = [
            d for d in self.drafts
            if not (d.season == draft.season and d.league_name == draft.league_name)
        ]
        self.drafts.append(draft)
        self.drafts.sort(key=lambda d: d.season)

    def to_frame(self) -> pd.DataFrame:
        frames = [d.to_frame() for d in self.drafts if d.picks]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def to_dict(self) -> dict[str, Any]:
        return {"drafts": [d.to_dict() for d in self.drafts]}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DraftHistory":
        return cls(drafts=[HistoricalDraft.from_dict(d)
                           for d in (dict(raw or {}).get("drafts") or [])])


# ─────────────────────────────────────────────────────────────────────────────
# Completed mock container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class MockDraftResult:
    """A finished (or abandoned) mock draft, ready to save / review / compare."""

    name: str
    league_name: str
    season: int
    picks: list[Pick] = field(default_factory=list)
    user_slots: tuple[int, ...] = ()
    random_seed: int | None = None
    mode: str = "interactive"
    created_at: str = ""
    notes: str = ""
    mock_id: int | None = None
    settings_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def rounds(self) -> int:
        return max((p.round_number for p in self.picks), default=0)

    @property
    def pick_count(self) -> int:
        return len(self.picks)

    def picks_for(self, manager: str) -> list[Pick]:
        key = normalize_manager_key(manager)
        return [p for p in self.picks if p.manager_key == key]

    def user_picks(self) -> list[Pick]:
        return [p for p in self.picks if p.is_user_pick]

    def board_frame(self) -> pd.DataFrame:
        """Long-form board: one row per pick."""
        if not self.picks:
            return pd.DataFrame()
        return pd.DataFrame([p.to_dict() for p in self.picks]).sort_values("overall_pick")

    def board_grid(self) -> pd.DataFrame:
        """Wide board: rounds as rows, draft slots as columns."""
        if not self.picks:
            return pd.DataFrame()
        rows: dict[int, dict[str, str]] = {}
        for pick in self.picks:
            cell = f"{pick.player_name}\n{pick.position} · {pick.nfl_team}"
            if pick.is_keeper:
                cell = f"[K] {cell}"
            rows.setdefault(pick.round_number, {})[f"Slot {pick.draft_slot}"] = cell
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "Round"
        ordered = sorted(frame.columns, key=lambda c: int(str(c).split()[-1]))
        return frame[ordered].sort_index()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "league_name": self.league_name,
            "season": self.season,
            "picks": [p.to_dict() for p in self.picks],
            "user_slots": list(self.user_slots),
            "random_seed": self.random_seed,
            "mode": self.mode,
            "created_at": self.created_at,
            "notes": self.notes,
            "mock_id": self.mock_id,
            "settings_snapshot": dict(self.settings_snapshot),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MockDraftResult":
        raw = dict(raw or {})
        return cls(
            name=raw.get("name", "Mock Draft"),
            league_name=raw.get("league_name", ""),
            season=int(raw.get("season", 0)),
            picks=[Pick.from_dict(p) for p in (raw.get("picks") or [])],
            user_slots=tuple(int(s) for s in (raw.get("user_slots") or [])),
            random_seed=raw.get("random_seed"),
            mode=raw.get("mode", "interactive"),
            created_at=raw.get("created_at", ""),
            notes=raw.get("notes", ""),
            mock_id=raw.get("mock_id"),
            settings_snapshot=dict(raw.get("settings_snapshot") or {}),
        )
