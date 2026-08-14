"""Managers, their user-entered preferences, and their modelled draft profile.

A :class:`Manager` is the identity (name, slot, whether it's you). A
:class:`ManagerProfile` is the *behavioural* model the simulator consumes.
Every profile value is paired with a :class:`core.enums.ProvenanceKind` so the
UI can always say whether a number was observed, inferred, typed in by the
user, or borrowed from a fallback level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from core.config import ArchetypeParams, ShrinkageConfig, archetype_params
from core.enums import Archetype, Position, ProvenanceKind


@dataclass(slots=True)
class ManagerPreferences:
    """Everything the user can assert manually about an opponent.

    These are treated as *evidence*, blended into the profile rather than
    silently overwriting the model — except for the explicit ``*_override``
    fields, which do win outright.
    """

    favorite_nfl_team: str | None = None
    favorite_players: list[str] = field(default_factory=list)
    disliked_players: list[str] = field(default_factory=list)
    preferred_positions: list[Position] = field(default_factory=list)
    avoided_positions: list[Position] = field(default_factory=list)
    typical_strategy: Archetype | None = None
    experience_level: str = "average"      # new | average | veteran | expert
    risk_tolerance: float | None = None    # 0 floor-seeking … 1 ceiling-seeking
    rookie_preference: float | None = None
    stack_preference: float | None = None
    handcuff_preference: float | None = None
    rank_reliance: float | None = None
    draft_speed: str = "normal"            # slow | normal | fast | autodraft
    predictability: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        self.preferred_positions = [
            p for p in (Position.coerce(x, None) for x in self.preferred_positions) if p
        ]
        self.avoided_positions = [
            p for p in (Position.coerce(x, None) for x in self.avoided_positions) if p
        ]
        self.typical_strategy = Archetype.coerce(self.typical_strategy, None)
        self.favorite_players = [str(p).strip() for p in self.favorite_players if str(p).strip()]
        self.disliked_players = [str(p).strip() for p in self.disliked_players if str(p).strip()]

    @property
    def has_any(self) -> bool:
        """True when the user supplied anything at all."""
        return bool(
            self.favorite_nfl_team or self.favorite_players or self.disliked_players
            or self.preferred_positions or self.avoided_positions
            or self.typical_strategy is not None
            or self.risk_tolerance is not None or self.rookie_preference is not None
            or self.stack_preference is not None or self.handcuff_preference is not None
            or self.rank_reliance is not None or self.predictability is not None
            or self.notes.strip()
        )

    def experience_confidence(self) -> float:
        """How strongly experience level nudges predictability."""
        return {
            "new": 0.30, "average": 0.50, "veteran": 0.62, "expert": 0.70,
        }.get(str(self.experience_level).lower(), 0.50)

    def to_dict(self) -> dict[str, Any]:
        return {
            "favorite_nfl_team": self.favorite_nfl_team,
            "favorite_players": list(self.favorite_players),
            "disliked_players": list(self.disliked_players),
            "preferred_positions": [str(p) for p in self.preferred_positions],
            "avoided_positions": [str(p) for p in self.avoided_positions],
            "typical_strategy": str(self.typical_strategy) if self.typical_strategy else None,
            "experience_level": self.experience_level,
            "risk_tolerance": self.risk_tolerance,
            "rookie_preference": self.rookie_preference,
            "stack_preference": self.stack_preference,
            "handcuff_preference": self.handcuff_preference,
            "rank_reliance": self.rank_reliance,
            "draft_speed": self.draft_speed,
            "predictability": self.predictability,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "ManagerPreferences":
        raw = dict(raw or {})
        return cls(
            favorite_nfl_team=raw.get("favorite_nfl_team"),
            favorite_players=list(raw.get("favorite_players") or []),
            disliked_players=list(raw.get("disliked_players") or []),
            preferred_positions=list(raw.get("preferred_positions") or []),
            avoided_positions=list(raw.get("avoided_positions") or []),
            typical_strategy=raw.get("typical_strategy"),
            experience_level=raw.get("experience_level", "average"),
            risk_tolerance=_opt_float(raw.get("risk_tolerance")),
            rookie_preference=_opt_float(raw.get("rookie_preference")),
            stack_preference=_opt_float(raw.get("stack_preference")),
            handcuff_preference=_opt_float(raw.get("handcuff_preference")),
            rank_reliance=_opt_float(raw.get("rank_reliance")),
            draft_speed=raw.get("draft_speed", "normal"),
            predictability=_opt_float(raw.get("predictability")),
            notes=raw.get("notes", ""),
        )


@dataclass(slots=True)
class Manager:
    """A team in the league."""

    name: str
    draft_slot: int
    team_name: str = ""
    is_user: bool = False
    manager_id: int | None = None
    archetype: Archetype = Archetype.BALANCED
    preferences: ManagerPreferences = field(default_factory=ManagerPreferences)

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        self.draft_slot = int(self.draft_slot)
        self.archetype = Archetype.coerce(self.archetype, Archetype.BALANCED) or Archetype.BALANCED
        if not self.team_name:
            self.team_name = f"{self.name}'s Team"

    @property
    def key(self) -> str:
        """Normalised identity key used to join historical picks."""
        return normalize_manager_key(self.name)

    @property
    def label(self) -> str:
        suffix = " (you)" if self.is_user else ""
        return f"{self.name}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "draft_slot": self.draft_slot,
            "team_name": self.team_name,
            "is_user": self.is_user,
            "manager_id": self.manager_id,
            "archetype": str(self.archetype),
            "preferences": self.preferences.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Manager":
        raw = dict(raw or {})
        return cls(
            name=raw.get("name", "Manager"),
            draft_slot=int(raw.get("draft_slot", 1)),
            team_name=raw.get("team_name", ""),
            is_user=bool(raw.get("is_user", False)),
            manager_id=raw.get("manager_id"),
            archetype=raw.get("archetype", Archetype.BALANCED),
            preferences=ManagerPreferences.from_dict(raw.get("preferences")),
        )


@dataclass(slots=True)
class ProfileValue:
    """A single modelled parameter plus where it came from."""

    value: float
    provenance: ProvenanceKind = ProvenanceKind.BASELINE
    sample_size: float = 0.0
    manager_weight: float = 0.0
    observed_value: float | None = None
    """The raw manager-only estimate before shrinkage (None when unavailable)."""

    def __float__(self) -> float:
        return float(self.value)

    @property
    def is_personalized(self) -> bool:
        return self.manager_weight >= 0.25

    def describe(self) -> str:
        label = {
            ProvenanceKind.OBSERVED: "observed in their drafts",
            ProvenanceKind.MODEL_INFERRED: "model estimate",
            ProvenanceKind.USER_ENTERED: "you told us",
            ProvenanceKind.LEAGUE_FALLBACK: "league average",
            ProvenanceKind.PLATFORM_FALLBACK: "platform average",
            ProvenanceKind.BASELINE: "general prior",
        }[self.provenance]
        if self.sample_size:
            return f"{label} ({self.sample_size:.0f} picks, {self.manager_weight:.0%} personalised)"
        return label


@dataclass(slots=True)
class ManagerProfile:
    """The behavioural model for one manager, consumed by the pick engine.

    Scalar behavioural parameters live in :attr:`values` (each a
    :class:`ProfileValue`); positional tendencies live in the dict fields.
    Convenience properties expose plain floats for the hot path.
    """

    manager_key: str
    manager_name: str
    archetype: Archetype = Archetype.BALANCED
    values: dict[str, ProfileValue] = field(default_factory=dict)
    position_bias: dict[Position, float] = field(default_factory=dict)
    early_round_position_bias: dict[Position, float] = field(default_factory=dict)
    position_rate_by_round: dict[int, dict[Position, float]] = field(default_factory=dict)
    """Round → position → observed selection rate (recency weighted)."""
    favorite_teams: dict[str, float] = field(default_factory=dict)
    """NFL team → share of this manager's picks."""
    repeat_players: dict[str, int] = field(default_factory=dict)
    """Player name → how many *distinct seasons* this manager drafted him.

    Only names with two or more seasons appear, so every entry is a loyalty signal by
    construction. Counted in seasons rather than picks on purpose: drafting someone
    twice in one season says nothing about liking him, and two mocks of the same league
    would otherwise read as devotion.
    """
    sample_picks: float = 0.0
    sample_drafts: int = 0
    seasons_seen: tuple[int, ...] = ()
    preferences: ManagerPreferences = field(default_factory=ManagerPreferences)
    notes: str = ""

    # -- parameter access -------------------------------------------------
    def get(self, key: str, default: float = 0.0) -> float:
        """Plain float for a parameter."""
        entry = self.values.get(key)
        if entry is None:
            return float(_PARAM_DEFAULTS.get(key, default))
        return float(entry.value)

    def provenance(self, key: str) -> ProvenanceKind:
        entry = self.values.get(key)
        return entry.provenance if entry else ProvenanceKind.BASELINE

    def set(
        self,
        key: str,
        value: float,
        provenance: ProvenanceKind = ProvenanceKind.MODEL_INFERRED,
        *,
        sample_size: float = 0.0,
        manager_weight: float = 0.0,
        observed_value: float | None = None,
    ) -> None:
        self.values[key] = ProfileValue(
            value=float(value),
            provenance=provenance,
            sample_size=float(sample_size),
            manager_weight=float(manager_weight),
            observed_value=observed_value,
        )

    # -- hot-path convenience --------------------------------------------
    @property
    def reach_mean(self) -> float:
        """Average picks drafted *ahead* of ADP (positive = reaches)."""
        return self.get("reach_mean_picks")

    @property
    def reach_stdev(self) -> float:
        return max(1.0, self.get("reach_stdev_picks"))

    @property
    def rank_dependence(self) -> float:
        return _clamp01(self.get("rank_dependence"))

    @property
    def need_dependence(self) -> float:
        return _clamp01(self.get("need_dependence"))

    @property
    def rookie_rate(self) -> float:
        return _clamp01(self.get("rookie_rate"))

    @property
    def stack_rate(self) -> float:
        return _clamp01(self.get("stack_rate"))

    @property
    def handcuff_rate(self) -> float:
        return _clamp01(self.get("handcuff_rate"))

    @property
    def favorite_team_rate(self) -> float:
        return _clamp01(self.get("favorite_team_rate"))

    @property
    def run_chase(self) -> float:
        return _clamp01(self.get("run_chase"))

    @property
    def tier_sensitivity(self) -> float:
        return _clamp01(self.get("tier_sensitivity"))

    @property
    def predictability(self) -> float:
        return _clamp01(self.get("predictability"))

    @property
    def risk_preference(self) -> float:
        return _clamp01(self.get("risk_preference"))

    @property
    def first_qb_round(self) -> float:
        return max(1.0, self.get("first_qb_round"))

    @property
    def first_te_round(self) -> float:
        return max(1.0, self.get("first_te_round"))

    @property
    def confidence(self) -> float:
        """0-1 confidence in personalisation, driven by sample size."""
        return _clamp01(self.sample_picks / max(1.0, self.sample_picks + 24.0))

    @property
    def top_favorite_team(self) -> str | None:
        if self.preferences.favorite_nfl_team:
            return self.preferences.favorite_nfl_team
        if not self.favorite_teams:
            return None
        team, share = max(self.favorite_teams.items(), key=lambda kv: kv[1])
        # Only call it a preference when it is well above uniform.
        return team if share >= 0.18 else None

    def position_rate(self, round_number: int, position: Position) -> float | None:
        """Observed rate at which this manager took ``position`` in that round."""
        rates = self.position_rate_by_round.get(int(round_number))
        if not rates:
            return None
        return rates.get(position)

    # -- narrative -------------------------------------------------------
    def describe(self, league_average: "ManagerProfile | None" = None) -> str:
        """Plain-language summary shown on the Manager Profiles page."""
        parts: list[str] = []
        name = self.manager_name

        if self.sample_picks <= 0:
            params = archetype_params(self.archetype)
            return (
                f"{name} has no draft history on file, so they are simulated as a "
                f"“{params.label}” drafter: {params.description.lower().rstrip('.')}."
            )

        # Positional lean vs league average.
        leans: list[str] = []
        for position in (Position.RB, Position.WR, Position.TE, Position.QB):
            bias = self.position_bias.get(position, 0.0)
            if abs(bias) < 0.08:
                continue
            direction = "earlier than" if bias > 0 else "later than"
            leans.append(f"{_position_plural(position)} {direction} league average")
        if leans:
            parts.append(f"{name} drafts " + _join(leans))
        else:
            parts.append(f"{name} drafts close to league-average positional timing")

        qb_round = self.first_qb_round
        te_round = self.first_te_round
        parts.append(
            f"typically takes a first quarterback around round {qb_round:.0f} and a "
            f"first tight end around round {te_round:.0f}"
        )

        reach = self.reach_mean
        if reach >= 3:
            parts.append(
                f"is willing to reach roughly {reach:.0f} picks ahead of ADP for "
                "players they want"
            )
        elif reach <= -3:
            parts.append(f"tends to let value come to them, drafting about {abs(reach):.0f} "
                         "picks behind ADP on average")
        else:
            parts.append("drafts close to ADP")

        if self.rank_dependence >= 0.75:
            parts.append("sticks closely to the platform's default rankings")
        elif self.rank_dependence <= 0.30:
            parts.append("largely ignores the platform's default rankings")

        if self.predictability >= 0.70:
            parts.append("and is quite predictable pick to pick")
        elif self.predictability <= 0.32:
            parts.append("and is unpredictable — expect surprises")

        # Base-form verbs: these are appended after "They also ...".
        extras: list[str] = []
        if self.rookie_rate >= 0.20:
            extras.append(f"draft rookies at {self.rookie_rate:.0%} of picks")
        favorite = self.top_favorite_team
        if favorite and self.favorite_team_rate >= 0.15:
            extras.append(f"favour {favorite} players ({self.favorite_team_rate:.0%} of picks)")
        if self.stack_rate >= 0.15:
            extras.append("stack quarterbacks with their receivers")
        if self.handcuff_rate >= 0.12:
            extras.append("handcuff their running backs")

        sentence = _join(parts, final=", ") + "."
        if extras:
            sentence += " They also " + _join(extras) + "."
        sentence += (
            f" (Based on {self.sample_picks:.0f} picks across {self.sample_drafts} draft(s)"
            + (f", seasons {min(self.seasons_seen)}–{max(self.seasons_seen)}"
               if self.seasons_seen else "")
            + f"; {self.confidence:.0%} model confidence.)"
        )
        return sentence[0].upper() + sentence[1:]

    def provenance_summary(self) -> dict[str, int]:
        """Count of parameters by provenance — drives the honesty badges."""
        counts: dict[str, int] = {}
        for entry in self.values.values():
            key = str(entry.provenance)
            counts[key] = counts.get(key, 0) + 1
        return counts

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "manager_key": self.manager_key,
            "manager_name": self.manager_name,
            "archetype": str(self.archetype),
            "values": {
                k: {
                    "value": v.value,
                    "provenance": str(v.provenance),
                    "sample_size": v.sample_size,
                    "manager_weight": v.manager_weight,
                    "observed_value": v.observed_value,
                }
                for k, v in self.values.items()
            },
            "position_bias": {str(k): v for k, v in self.position_bias.items()},
            "early_round_position_bias": {
                str(k): v for k, v in self.early_round_position_bias.items()
            },
            "position_rate_by_round": {
                str(rnd): {str(p): r for p, r in rates.items()}
                for rnd, rates in self.position_rate_by_round.items()
            },
            "favorite_teams": dict(self.favorite_teams),
            "repeat_players": dict(self.repeat_players),
            "sample_picks": self.sample_picks,
            "sample_drafts": self.sample_drafts,
            "seasons_seen": list(self.seasons_seen),
            "preferences": self.preferences.to_dict(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ManagerProfile":
        raw = dict(raw or {})
        profile = cls(
            manager_key=raw.get("manager_key", ""),
            manager_name=raw.get("manager_name", ""),
            archetype=Archetype.coerce(raw.get("archetype"), Archetype.BALANCED)
            or Archetype.BALANCED,
            position_bias={
                Position.coerce(k, Position.RB): float(v)
                for k, v in (raw.get("position_bias") or {}).items()
            },
            early_round_position_bias={
                Position.coerce(k, Position.RB): float(v)
                for k, v in (raw.get("early_round_position_bias") or {}).items()
            },
            position_rate_by_round={
                int(rnd): {Position.coerce(p, Position.RB): float(r)
                           for p, r in rates.items()}
                for rnd, rates in (raw.get("position_rate_by_round") or {}).items()
            },
            favorite_teams={str(k): float(v)
                            for k, v in (raw.get("favorite_teams") or {}).items()},
            repeat_players={str(k): int(v)
                            for k, v in (raw.get("repeat_players") or {}).items()},
            sample_picks=float(raw.get("sample_picks", 0.0)),
            sample_drafts=int(raw.get("sample_drafts", 0)),
            seasons_seen=tuple(int(s) for s in (raw.get("seasons_seen") or [])),
            preferences=ManagerPreferences.from_dict(raw.get("preferences")),
            notes=raw.get("notes", ""),
        )
        for key, entry in (raw.get("values") or {}).items():
            profile.values[key] = ProfileValue(
                value=float(entry.get("value", 0.0)),
                provenance=ProvenanceKind.coerce(
                    entry.get("provenance"), ProvenanceKind.BASELINE
                ) or ProvenanceKind.BASELINE,
                sample_size=float(entry.get("sample_size", 0.0)),
                manager_weight=float(entry.get("manager_weight", 0.0)),
                observed_value=_opt_float(entry.get("observed_value")),
            )
        return profile

    @classmethod
    def from_archetype(
        cls,
        manager: Manager,
        archetype: Archetype | None = None,
        *,
        provenance: ProvenanceKind = ProvenanceKind.BASELINE,
    ) -> "ManagerProfile":
        """Build a profile purely from archetype priors (no history)."""
        chosen = (
            Archetype.coerce(archetype, None)
            or manager.preferences.typical_strategy
            or manager.archetype
        )
        params = archetype_params(chosen)
        profile = cls(
            manager_key=manager.key,
            manager_name=manager.name,
            archetype=chosen,
            preferences=manager.preferences,
        )
        apply_archetype_params(profile, params, provenance)
        return profile


# ─────────────────────────────────────────────────────────────────────────────
# Parameter registry
# ─────────────────────────────────────────────────────────────────────────────
PARAM_KEYS: tuple[str, ...] = (
    "reach_mean_picks", "reach_stdev_picks", "rank_dependence", "need_dependence",
    "rookie_rate", "stack_rate", "handcuff_rate", "favorite_team_rate",
    "run_chase", "tier_sensitivity", "predictability", "risk_preference",
    "first_qb_round", "first_te_round",
)

PARAM_LABELS: dict[str, str] = {
    "reach_mean_picks": "Average reach vs ADP (picks)",
    "reach_stdev_picks": "Reach variability (picks)",
    "rank_dependence": "Reliance on platform rankings",
    "need_dependence": "Reliance on roster need",
    "rookie_rate": "Rookie selection rate",
    "stack_rate": "QB/WR stacking rate",
    "handcuff_rate": "RB handcuff rate",
    "favorite_team_rate": "Favourite-team selection rate",
    "run_chase": "Chases positional runs",
    "tier_sensitivity": "Tier-based drafting",
    "predictability": "Predictability",
    "risk_preference": "Upside preference (vs floor)",
    "first_qb_round": "Round of first QB",
    "first_te_round": "Round of first TE",
}

# Parameters observed once per draft rather than once per pick, and therefore
# already shrunk on the number of seasons before they reach the pick-based
# shrinkage. The per-draft discount in
# :meth:`core.config.ShrinkageConfig.cluster_weight` skips these so it is not
# charged twice.
SEASON_SCALED_PARAMS: frozenset[str] = frozenset({
    "first_qb_round", "first_te_round",
})

# Parameters bounded to [0, 1]; others are unbounded/round numbers.
UNIT_PARAMS: frozenset[str] = frozenset({
    "rank_dependence", "need_dependence", "rookie_rate", "stack_rate",
    "handcuff_rate", "favorite_team_rate", "run_chase", "tier_sensitivity",
    "predictability", "risk_preference",
})

_PARAM_DEFAULTS: dict[str, float] = {
    "reach_mean_picks": 0.0,
    "reach_stdev_picks": 9.0,
    "rank_dependence": 0.5,
    "need_dependence": 0.5,
    "rookie_rate": 0.10,
    "stack_rate": 0.08,
    "handcuff_rate": 0.06,
    "favorite_team_rate": 0.05,
    "run_chase": 0.5,
    "tier_sensitivity": 0.5,
    "predictability": 0.5,
    "risk_preference": 0.5,
    "first_qb_round": 8.0,
    "first_te_round": 8.0,
}


def param_default(key: str) -> float:
    return float(_PARAM_DEFAULTS.get(key, 0.0))


def apply_archetype_params(
    profile: ManagerProfile,
    params: ArchetypeParams,
    provenance: ProvenanceKind = ProvenanceKind.BASELINE,
) -> None:
    """Write archetype priors into a profile's parameter set."""
    mapping = {
        "reach_mean_picks": params.reach_mean_picks,
        "reach_stdev_picks": params.reach_stdev_picks,
        "rank_dependence": params.rank_dependence,
        "need_dependence": params.need_dependence,
        "rookie_rate": params.rookie_rate,
        "stack_rate": params.stack_rate,
        "handcuff_rate": params.handcuff_rate,
        "favorite_team_rate": params.favorite_team_rate,
        "run_chase": params.run_chase,
        "tier_sensitivity": params.tier_sensitivity,
        "predictability": params.predictability,
        "risk_preference": params.risk_preference,
        "first_qb_round": params.first_qb_round,
        "first_te_round": params.first_te_round,
    }
    for key, value in mapping.items():
        profile.set(key, value, provenance)
    profile.position_bias = dict(params.position_bias)
    profile.early_round_position_bias = dict(params.early_round_position_bias)


def baseline_profile(name: str = "baseline") -> ManagerProfile:
    """The general fantasy-football prior used as the last fallback level."""
    profile = ManagerProfile(manager_key=normalize_manager_key(name), manager_name=name)
    apply_archetype_params(
        profile, archetype_params(Archetype.BALANCED), ProvenanceKind.BASELINE
    )
    return profile


def normalize_manager_key(name: str) -> str:
    """Case/punctuation-insensitive manager identity key."""
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


# ─────────────────────────────────────────────────────────────────────────────
# small helpers
# ─────────────────────────────────────────────────────────────────────────────
def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _position_plural(position: Position) -> str:
    return {
        Position.QB: "quarterbacks", Position.RB: "running backs",
        Position.WR: "wide receivers", Position.TE: "tight ends",
        Position.K: "kickers", Position.DST: "defenses",
    }.get(position, str(position))


def _join(items: Iterable[str], final: str = " and ") -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + final + items[-1]
