"""Validation primitives and league/draft rule checks.

Two ideas drive this module:

* Nothing is silently dropped. Every rejected row or broken rule becomes an
  :class:`Issue` carrying enough context for the UI to explain and export it.
* Validation is pure. It never raises for *data* problems — it returns a
  :class:`ValidationReport`. Programmer errors still raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import pandas as pd

from .config import LeagueConfig, RosterSettings
from .constants import SLOT_ELIGIBILITY
from .enums import DraftType, Position, Slot

ERROR = "error"
WARNING = "warning"
INFO = "info"


class ConfigurationError(ValueError):
    """Raised for programmer-level misuse (not user data problems)."""


@dataclass(slots=True)
class Issue:
    """A single validation finding."""

    severity: str
    code: str
    message: str
    row: int | None = None
    column: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.severity == ERROR

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "row": self.row,
            "column": self.column,
            **{f"ctx_{k}": v for k, v in self.context.items()},
        }

    def __str__(self) -> str:
        where = f" (row {self.row})" if self.row is not None else ""
        return f"[{self.severity}] {self.message}{where}"


@dataclass(slots=True)
class ValidationReport:
    """Collected issues plus the rows that failed hard validation."""

    issues: list[Issue] = field(default_factory=list)
    rejected: pd.DataFrame | None = None

    # -- construction ----------------------------------------------------
    def add(
        self,
        severity: str,
        code: str,
        message: str,
        *,
        row: int | None = None,
        column: str | None = None,
        context: dict[str, Any] | None = None,
        **extra: Any,
    ) -> Issue:
        """Record an issue. Extra context may be passed as ``context=`` or kwargs."""
        merged = dict(context or {})
        merged.update(extra)
        issue = Issue(severity, code, message, row=row, column=column, context=merged)
        self.issues.append(issue)
        return issue

    def error(self, code: str, message: str, **kw: Any) -> Issue:
        return self.add(ERROR, code, message, **kw)

    def warn(self, code: str, message: str, **kw: Any) -> Issue:
        return self.add(WARNING, code, message, **kw)

    def info(self, code: str, message: str, **kw: Any) -> Issue:
        return self.add(INFO, code, message, **kw)

    def extend(self, other: "ValidationReport") -> "ValidationReport":
        self.issues.extend(other.issues)
        if other.rejected is not None and len(other.rejected):
            self.rejected = (
                other.rejected if self.rejected is None
                else pd.concat([self.rejected, other.rejected], ignore_index=True)
            )
        return self

    # -- inspection ------------------------------------------------------
    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def infos(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == INFO]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def rejected_count(self) -> int:
        return 0 if self.rejected is None else int(len(self.rejected))

    def to_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=["severity", "code", "message", "row", "column"])
        return pd.DataFrame([i.as_dict() for i in self.issues])

    def summary(self) -> str:
        return (
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s), "
            f"{self.rejected_count} rejected row(s)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Scalar coercion helpers — used heavily by the importers.
# ─────────────────────────────────────────────────────────────────────────────
_TRUTHY = {"1", "true", "t", "yes", "y", "keeper", "k", "rookie", "r"}
_FALSEY = {"0", "false", "f", "no", "n", "", "nan", "none", "null", "-"}


def to_bool(value: Any, default: bool = False) -> bool:
    """Parse loose spreadsheet truthiness (``Y``, ``TRUE``, ``1``, ``keeper``…)."""
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSEY:
        return False
    return default


def to_float(value: Any, default: float | None = None) -> float | None:
    """Parse a float, tolerating ``$``, commas, percent signs and blanks."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        try:
            if pd.isna(value):
                return default
        except (TypeError, ValueError):
            pass
        return float(value)
    text = str(value).strip().replace(",", "").replace("$", "").replace("%", "")
    if text in ("", "-", "--", "nan", "None", "NULL", "N/A", "n/a"):
        return default
    try:
        return float(text)
    except ValueError:
        return default


def to_int(value: Any, default: int | None = None) -> int | None:
    """Parse an int via :func:`to_float` so ``"3.0"`` works."""
    parsed = to_float(value, None)
    if parsed is None:
        return default
    try:
        return int(round(parsed))
    except (ValueError, OverflowError):
        return default


def require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    report: ValidationReport,
    *,
    label: str = "file",
) -> bool:
    """Record an error for each missing required column. Returns True if all present."""
    missing = [c for c in required if c not in frame.columns]
    if missing:
        report.error(
            "missing_columns",
            f"{label} is missing required column(s): {', '.join(missing)}. "
            f"Found: {', '.join(map(str, frame.columns)) or '(none)'}",
            missing=", ".join(missing),
        )
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# League configuration validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_league(config: LeagueConfig) -> ValidationReport:
    """Check a league config for internal consistency."""
    report = ValidationReport()

    if config.team_count < 2:
        report.error("team_count", "A league needs at least 2 teams.")
    elif config.team_count > 32:
        report.error("team_count", "Team count above 32 is not supported.")
    elif config.team_count % 2 and config.draft_type is DraftType.SNAKE:
        report.warn(
            "odd_team_count",
            f"{config.team_count} teams is an odd count — snake order still works "
            "but real leagues rarely do this. Double-check the setting.",
        )

    if config.rounds < 1:
        report.error("rounds", "A draft needs at least 1 round.")
    elif config.rounds > 40:
        report.error("rounds", "Round count above 40 is not supported.")

    if not (1 <= config.user_draft_slot <= max(1, config.team_count)):
        report.error(
            "user_slot",
            f"Your draft slot ({config.user_draft_slot}) must be between 1 and "
            f"{config.team_count}.",
        )

    roster = config.roster
    if roster.starters_total < 1:
        report.error("no_starters", "The roster has no starting slots configured.")

    if roster.roster_size > config.rounds:
        report.error(
            "roster_exceeds_rounds",
            f"Roster size ({roster.roster_size} = {roster.starters_total} starters + "
            f"{roster.bench_total} bench) exceeds the {config.rounds} draft rounds. "
            "Add rounds or remove slots.",
        )
    elif roster.roster_size < config.rounds:
        report.warn(
            "rounds_exceed_roster",
            f"{config.rounds} rounds fills {roster.roster_size} roster spots — "
            f"{config.rounds - roster.roster_size} extra pick(s) have nowhere to go.",
        )

    report.extend(validate_roster_settings(roster, config))

    if config.draft_type is DraftType.CUSTOM:
        if not config.custom_round_order:
            report.error(
                "custom_order_missing",
                "Custom draft type selected but no round-by-round order was provided.",
            )
        else:
            expected = set(range(1, config.team_count + 1))
            for rnd in range(1, config.rounds + 1):
                order = config.custom_round_order.get(rnd)
                if not order:
                    report.error(
                        "custom_order_round",
                        f"Custom order is missing round {rnd}.",
                        context={"round": rnd},
                    )
                    continue
                if set(order) != expected or len(order) != config.team_count:
                    report.error(
                        "custom_order_invalid",
                        f"Round {rnd} custom order must list each slot 1-"
                        f"{config.team_count} exactly once (got {order}).",
                        context={"round": rnd},
                    )

    if config.draft_type is DraftType.THIRD_ROUND_REVERSAL:
        if not (2 <= config.reversal_round <= config.rounds):
            report.error(
                "reversal_round",
                f"Reversal round ({config.reversal_round}) must be between 2 and "
                f"{config.rounds}.",
            )

    return report


def validate_roster_settings(
    roster: RosterSettings, config: LeagueConfig | None = None
) -> ValidationReport:
    """Check slot/position limits for contradictions."""
    report = ValidationReport()
    demand = roster.starting_demand()

    for position, minimum in roster.position_min.items():
        maximum = roster.position_max.get(position)
        if maximum is not None and minimum > maximum:
            report.error(
                "min_gt_max",
                f"{position} minimum ({minimum}) exceeds its maximum ({maximum}).",
                context={"position": str(position)},
            )
        if minimum > roster.roster_size:
            report.error(
                "min_gt_roster",
                f"{position} minimum ({minimum}) exceeds the roster size "
                f"({roster.roster_size}).",
                context={"position": str(position)},
            )

    for position, maximum in roster.position_max.items():
        needed = demand.get(position, 0.0)
        # A dedicated slot count is a hard requirement; flex share is not.
        dedicated = 0
        for slot, n in roster.starting_slots.items():
            eligible = SLOT_ELIGIBILITY.get(slot, frozenset())
            if eligible == frozenset({position}):
                dedicated += n
        if maximum < dedicated:
            report.error(
                "max_below_starters",
                f"{position} maximum ({maximum}) is below the {dedicated} dedicated "
                f"{position} starting slot(s).",
                context={"position": str(position)},
            )
        elif maximum < needed:
            report.warn(
                "max_tight",
                f"{position} maximum ({maximum}) is below expected starting demand "
                f"({needed:.1f}) once flex slots are counted.",
                context={"position": str(position)},
            )

    total_min = sum(roster.position_min.values())
    if total_min > roster.roster_size:
        report.error(
            "min_sum_exceeds_roster",
            f"Position minimums total {total_min}, above the {roster.roster_size} "
            "roster spots available.",
        )

    if config is not None:
        unreachable = {
            slot for slot in roster.slots
            if slot not in SLOT_ELIGIBILITY
        }
        if unreachable:
            report.error(
                "unknown_slot",
                f"Unrecognised roster slot(s): {', '.join(map(str, unreachable))}.",
            )

    return report


def validate_managers(
    managers: Sequence[Any], config: LeagueConfig
) -> ValidationReport:
    """Check manager count, slot assignment uniqueness, and user ownership.

    ``managers`` items need ``name``, ``draft_slot`` and ``is_user`` attributes
    (see :class:`models.manager.Manager`).
    """
    report = ValidationReport()

    if len(managers) != config.team_count:
        report.error(
            "manager_count",
            f"{len(managers)} manager(s) entered but the league has "
            f"{config.team_count} teams.",
        )

    slots: dict[int, list[str]] = {}
    for manager in managers:
        slot = int(getattr(manager, "draft_slot", 0) or 0)
        slots.setdefault(slot, []).append(str(getattr(manager, "name", "?")))

    for slot, names in sorted(slots.items()):
        if len(names) > 1:
            report.error(
                "duplicate_slot",
                f"Draft slot {slot} is assigned to {len(names)} managers: "
                f"{', '.join(names)}.",
                context={"slot": slot},
            )
        if not (1 <= slot <= config.team_count):
            report.error(
                "invalid_slot",
                f"Draft slot {slot} (assigned to {', '.join(names)}) is outside "
                f"1-{config.team_count}.",
                context={"slot": slot},
            )

    unassigned = [s for s in range(1, config.team_count + 1) if s not in slots]
    if unassigned:
        report.error(
            "unassigned_slots",
            f"Draft slot(s) with no manager: {', '.join(map(str, unassigned))}.",
        )

    names = [str(getattr(m, "name", "")).strip().lower() for m in managers]
    duplicates = {n for n in names if n and names.count(n) > 1}
    if duplicates:
        report.error(
            "duplicate_manager",
            f"Duplicate manager name(s): {', '.join(sorted(duplicates))}. "
            "Names must be unique so history maps to the right person.",
        )
    if any(not n for n in names):
        report.error("blank_manager", "Every manager needs a name.")

    users = [m for m in managers if getattr(m, "is_user", False)]
    if not users:
        report.warn(
            "no_user_team",
            "No manager is marked as you — the whole draft will be simulated.",
        )
    elif len(users) > 1:
        report.info(
            "multiple_user_teams",
            f"You control {len(users)} teams: "
            f"{', '.join(str(getattr(m, 'name', '?')) for m in users)}.",
        )

    return report


def validate_keepers(
    keepers: Sequence[Any], config: LeagueConfig, *, manager_names: Iterable[str] = ()
) -> ValidationReport:
    """Check keeper assignments for double-claims and pick collisions.

    Keeper items need ``manager_name``, ``player_name``, ``keeper_round`` and
    ``overall_pick`` attributes or keys.
    """
    report = ValidationReport()
    if not keepers:
        return report

    known = {str(n).strip().lower() for n in manager_names}

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    if not config.allows_keepers:
        report.warn(
            "keepers_in_redraft",
            f"{len(keepers)} keeper(s) assigned but the league format is "
            f"{config.league_format}. Set the format to Keeper or Dynasty.",
        )

    by_player: dict[str, list[str]] = {}
    by_pick: dict[int, list[str]] = {}

    for index, keeper in enumerate(keepers):
        manager = str(_get(keeper, "manager_name", "") or "").strip()
        player = str(_get(keeper, "player_name", "") or "").strip()
        rnd = to_int(_get(keeper, "keeper_round"), None)
        overall = to_int(_get(keeper, "overall_pick"), None)

        if not player:
            report.error("keeper_no_player", "A keeper row has no player name.", row=index)
        if not manager:
            report.error("keeper_no_manager",
                         f"Keeper '{player or '?'}' has no manager.", row=index)
        elif known and manager.lower() not in known:
            report.error(
                "keeper_unknown_manager",
                f"Keeper '{player}' is assigned to '{manager}', who is not in the league.",
                row=index,
            )

        if rnd is not None and not (1 <= rnd <= config.rounds):
            report.error(
                "keeper_round_range",
                f"Keeper '{player}' has round {rnd}, outside 1-{config.rounds}.",
                row=index,
            )
        if overall is not None and not (1 <= overall <= config.total_picks):
            report.error(
                "keeper_pick_range",
                f"Keeper '{player}' costs overall pick {overall}, outside "
                f"1-{config.total_picks}.",
                row=index,
            )

        if player:
            by_player.setdefault(player.lower(), []).append(manager or "?")
        if overall is not None:
            by_pick.setdefault(overall, []).append(f"{manager or '?'}/{player or '?'}")

    for player, owners in by_player.items():
        if len(owners) > 1:
            report.error(
                "keeper_double_claim",
                f"'{player}' is kept by {len(owners)} managers: {', '.join(owners)}.",
                context={"player": player},
            )

    for pick, claims in by_pick.items():
        if len(claims) > 1:
            report.error(
                "keeper_pick_collision",
                f"Overall pick {pick} is claimed by {len(claims)} keepers: "
                f"{', '.join(claims)}.",
                context={"overall_pick": pick},
            )

    return report


def validate_player_pool(
    players: pd.DataFrame, config: LeagueConfig
) -> ValidationReport:
    """Sanity-check a normalised player pool against league needs."""
    report = ValidationReport()
    if players is None or players.empty:
        report.error("empty_pool", "The player pool is empty — import player data first.")
        return report

    if "player_name" in players.columns:
        dupes = players["player_name"].astype(str).str.strip().str.lower()
        counts = dupes.value_counts()
        repeated = counts[counts > 1]
        for name, count in repeated.items():
            report.warn(
                "duplicate_player",
                f"'{name}' appears {count} times in the player pool.",
                context={"player": name},
            )

    needed = config.roster.roster_size * config.team_count
    if len(players) < needed:
        report.warn(
            "thin_pool",
            f"Only {len(players)} players for a draft that consumes up to {needed}. "
            "Import a deeper list or reduce rounds.",
        )

    if "position" in players.columns:
        available = set(players["position"].dropna().astype(str))
        from .config import positions_in_use

        for position in positions_in_use(config.roster):
            if str(position) not in available:
                report.error(
                    "missing_position",
                    f"The league starts {position} but no {position} exists in the "
                    "player pool.",
                    context={"position": str(position)},
                )
                continue
            count = int((players["position"].astype(str) == str(position)).sum())
            dedicated = sum(
                n for slot, n in config.roster.starting_slots.items()
                if SLOT_ELIGIBILITY.get(slot, frozenset()) == frozenset({position})
            )
            required = dedicated * config.team_count
            if required and count < required:
                report.error(
                    "insufficient_position",
                    f"Only {count} {position}(s) available but {required} are needed "
                    f"to fill every team's dedicated {position} slot(s).",
                    context={"position": str(position)},
                )

    for column, label in (("overall_adp", "ADP"), ("platform_rank", "platform rank"),
                          ("projection", "projection")):
        if column not in players.columns:
            report.warn("missing_field", f"No {label} column — the model will fall back "
                                         "to the fields that are present.")
            continue
        missing = int(players[column].isna().sum())
        if missing:
            report.warn(
                "partial_field",
                f"{missing} of {len(players)} players have no {label}; those values "
                "are imputed from overall rank.",
                context={"column": column},
            )

    return report


def validate_draft_completeness(
    picks: Sequence[Any], config: LeagueConfig
) -> ValidationReport:
    """Verify a finished/imported draft: pick counts, duplicates, ordering."""
    report = ValidationReport()

    expected = config.total_picks
    if len(picks) != expected:
        severity = report.warn if len(picks) < expected else report.error
        severity(
            "pick_count",
            f"{len(picks)} pick(s) recorded but {config.team_count} teams x "
            f"{config.rounds} rounds = {expected}.",
        )

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    seen_players: dict[str, int] = {}
    seen_picks: dict[int, int] = {}
    for index, pick in enumerate(picks):
        player = str(_get(pick, "player_name", "") or "").strip().lower()
        overall = to_int(_get(pick, "overall_pick"), None)
        if player:
            if player in seen_players:
                report.error(
                    "player_drafted_twice",
                    f"'{player}' was drafted twice (picks "
                    f"{seen_players[player]} and {overall}).",
                    row=index,
                )
            else:
                seen_players[player] = overall or index + 1
        if overall is not None:
            if overall in seen_picks:
                report.error(
                    "duplicate_pick_number",
                    f"Overall pick {overall} appears more than once.",
                    row=index,
                )
            else:
                seen_picks[overall] = index

    if seen_picks:
        gaps = [n for n in range(1, max(seen_picks) + 1) if n not in seen_picks]
        if gaps:
            preview = ", ".join(map(str, gaps[:10]))
            more = "" if len(gaps) <= 10 else f" (+{len(gaps) - 10} more)"
            report.warn(
                "pick_gaps",
                f"Missing overall pick number(s): {preview}{more}. These may be "
                "traded or keeper picks.",
                context={"gap_count": len(gaps)},
            )

    return report


def validate_roster_legality(
    roster_positions: Sequence[Position], roster: RosterSettings
) -> ValidationReport:
    """Confirm a completed roster respects size and positional limits."""
    report = ValidationReport()
    counts: dict[Position, int] = {}
    for position in roster_positions:
        counts[position] = counts.get(position, 0) + 1

    if len(roster_positions) > roster.roster_size:
        report.error(
            "roster_overfull",
            f"Roster holds {len(roster_positions)} players, above the "
            f"{roster.roster_size} available spots.",
        )

    for position, maximum in roster.position_max.items():
        if counts.get(position, 0) > maximum:
            report.error(
                "position_max_exceeded",
                f"{counts[position]} {position}(s) drafted, above the max of {maximum}.",
                context={"position": str(position)},
            )
    for position, minimum in roster.position_min.items():
        if counts.get(position, 0) < minimum:
            report.warn(
                "position_min_unmet",
                f"Only {counts.get(position, 0)} {position}(s), below the minimum "
                f"of {minimum}.",
                context={"position": str(position)},
            )

    from .config import positions_in_use

    for position in positions_in_use(roster):
        dedicated = sum(
            n for slot, n in roster.starting_slots.items()
            if SLOT_ELIGIBILITY.get(slot, frozenset()) == frozenset({position})
        )
        if dedicated and counts.get(position, 0) < dedicated:
            report.warn(
                "starters_unfilled",
                f"{counts.get(position, 0)} {position}(s) cannot fill {dedicated} "
                f"starting {position} slot(s).",
                context={"position": str(position)},
            )

    return report
