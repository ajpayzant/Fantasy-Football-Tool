"""League aggregate: config + managers + keepers, with validation entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from core.config import LeagueConfig
from core.enums import Position
from core.validation import (
    ValidationReport,
    validate_keepers,
    validate_league,
    validate_managers,
)
from models.manager import Manager, normalize_manager_key


@dataclass(slots=True)
class Keeper:
    """A player retained before the draft, optionally consuming a pick."""

    manager_name: str
    player_name: str
    keeper_round: int | None = None
    overall_pick: int | None = None
    removes_pick: bool = True
    """When True the assigned pick is consumed and skipped in the draft order."""
    salary: float | None = None
    """Contract value in a salary-cap keeper league. Carried through import, save and
    load so a user's own keeper sheet round-trips intact, but nothing computes with
    it — the draft engine costs a keeper in picks, not dollars."""
    position: Position | None = None
    nfl_team: str | None = None
    notes: str = ""
    keeper_id: int | None = None

    def __post_init__(self) -> None:
        self.manager_name = str(self.manager_name).strip()
        self.player_name = str(self.player_name).strip()
        self.position = Position.coerce(self.position, None)
        if self.keeper_round is not None:
            self.keeper_round = int(self.keeper_round)
        if self.overall_pick is not None:
            self.overall_pick = int(self.overall_pick)

    @property
    def manager_key(self) -> str:
        return normalize_manager_key(self.manager_name)

    def resolve_pick(self, team_count: int, draft_slot: int) -> int | None:
        """Derive the overall pick this keeper costs, if not given explicitly.

        Uses the keeper's round with the manager's slot under snake ordering.
        Returns ``None`` when the keeper costs no pick.
        """
        if not self.removes_pick:
            return None
        if self.overall_pick is not None:
            return self.overall_pick
        if self.keeper_round is None:
            return None
        rnd = int(self.keeper_round)
        # Snake: even rounds run in reverse slot order.
        position_in_round = draft_slot if rnd % 2 else (team_count - draft_slot + 1)
        return (rnd - 1) * team_count + position_in_round

    def to_dict(self) -> dict[str, Any]:
        return {
            "manager_name": self.manager_name,
            "player_name": self.player_name,
            "keeper_round": self.keeper_round,
            "overall_pick": self.overall_pick,
            "removes_pick": self.removes_pick,
            "salary": self.salary,
            "position": str(self.position) if self.position else None,
            "nfl_team": self.nfl_team,
            "notes": self.notes,
            "keeper_id": self.keeper_id,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Keeper":
        raw = dict(raw or {})
        return cls(
            manager_name=raw.get("manager_name", ""),
            player_name=raw.get("player_name", ""),
            keeper_round=raw.get("keeper_round"),
            overall_pick=raw.get("overall_pick"),
            removes_pick=bool(raw.get("removes_pick", True)),
            salary=raw.get("salary"),
            position=raw.get("position"),
            nfl_team=raw.get("nfl_team"),
            notes=raw.get("notes", ""),
            keeper_id=raw.get("keeper_id"),
        )


@dataclass(slots=True)
class League:
    """A league plus its participants — the unit the UI edits and persists."""

    config: LeagueConfig
    managers: list[Manager] = field(default_factory=list)
    keepers: list[Keeper] = field(default_factory=list)

    # -- lookups ---------------------------------------------------------
    @property
    def name(self) -> str:
        return self.config.name

    @property
    def team_count(self) -> int:
        return self.config.team_count

    @property
    def league_id(self) -> int | None:
        return self.config.league_id

    def manager_by_slot(self, slot: int) -> Manager | None:
        for manager in self.managers:
            if manager.draft_slot == int(slot):
                return manager
        return None

    def manager_by_name(self, name: str) -> Manager | None:
        key = normalize_manager_key(name)
        for manager in self.managers:
            if manager.key == key:
                return manager
        return None

    def require_manager_by_slot(self, slot: int) -> Manager:
        manager = self.manager_by_slot(slot)
        if manager is None:
            raise KeyError(f"No manager assigned to draft slot {slot}")
        return manager

    @property
    def user_managers(self) -> list[Manager]:
        return [m for m in self.managers if m.is_user]

    @property
    def user_manager(self) -> Manager | None:
        users = self.user_managers
        return users[0] if users else None

    @property
    def user_slots(self) -> set[int]:
        return {m.draft_slot for m in self.managers if m.is_user}

    def slots_in_order(self) -> list[int]:
        return sorted(m.draft_slot for m in self.managers)

    def keepers_for(self, manager: Manager | str) -> list[Keeper]:
        key = manager.key if isinstance(manager, Manager) else normalize_manager_key(manager)
        return [k for k in self.keepers if k.manager_key == key]

    def kept_player_names(self) -> set[str]:
        return {k.player_name for k in self.keepers if k.player_name}

    def consumed_picks(self) -> dict[int, Keeper]:
        """Overall pick number → keeper occupying it."""
        out: dict[int, Keeper] = {}
        for keeper in self.keepers:
            manager = self.manager_by_name(keeper.manager_name)
            if manager is None:
                continue
            pick = keeper.resolve_pick(self.team_count, manager.draft_slot)
            if pick is not None:
                out[pick] = keeper
        return out

    # -- mutation --------------------------------------------------------
    def add_manager(self, manager: Manager) -> Manager:
        self.managers.append(manager)
        return manager

    def remove_manager(self, name: str) -> bool:
        key = normalize_manager_key(name)
        before = len(self.managers)
        self.managers = [m for m in self.managers if m.key != key]
        self.keepers = [k for k in self.keepers if k.manager_key != key]
        return len(self.managers) != before

    def autofill_managers(self, prefix: str = "Manager") -> None:
        """Create placeholder managers for any unassigned draft slot."""
        taken = {m.draft_slot for m in self.managers}
        for slot in range(1, self.team_count + 1):
            if slot not in taken:
                self.managers.append(Manager(name=f"{prefix} {slot}", draft_slot=slot))
        self.managers.sort(key=lambda m: m.draft_slot)

    def set_user_slot(self, slot: int) -> None:
        """Mark exactly the manager at ``slot`` as the user."""
        for manager in self.managers:
            manager.is_user = manager.draft_slot == int(slot)
        self.config.user_draft_slot = int(slot)

    # -- validation ------------------------------------------------------
    def validate(self) -> ValidationReport:
        report = validate_league(self.config)
        report.extend(validate_managers(self.managers, self.config))
        report.extend(
            validate_keepers(
                self.keepers, self.config,
                manager_names=[m.name for m in self.managers],
            )
        )
        return report

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "managers": [m.to_dict() for m in self.managers],
            "keepers": [k.to_dict() for k in self.keepers],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "League":
        raw = dict(raw or {})
        return cls(
            config=LeagueConfig.from_dict(raw.get("config") or {}),
            managers=[Manager.from_dict(m) for m in (raw.get("managers") or [])],
            keepers=[Keeper.from_dict(k) for k in (raw.get("keepers") or [])],
        )

    @classmethod
    def new(
        cls,
        config: LeagueConfig | None = None,
        *,
        manager_names: Sequence[str] | None = None,
    ) -> "League":
        """Create a league, optionally seeding managers by name in slot order."""
        config = config or LeagueConfig()
        league = cls(config=config)
        if manager_names:
            for slot, name in enumerate(manager_names, start=1):
                if slot > config.team_count:
                    break
                league.add_manager(Manager(name=name, draft_slot=slot))
        league.autofill_managers()
        league.set_user_slot(config.user_draft_slot)
        return league

    def duplicate(self, new_name: str | None = None) -> "League":
        """Deep copy with database ids cleared, for 'duplicate league'."""
        clone = League.from_dict(self.to_dict())
        clone.config.league_id = None
        clone.config.name = new_name or f"{self.config.name} (copy)"
        for manager in clone.managers:
            manager.manager_id = None
        for keeper in clone.keepers:
            keeper.keeper_id = None
        return clone
