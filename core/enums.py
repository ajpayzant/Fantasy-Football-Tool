"""Enumerations shared across the application.

All enums are ``str``-based so they serialise cleanly to JSON / SQLite and
compare directly against values loaded from user files.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """A string enum whose ``str()`` is the raw value (Py3.10-compatible)."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)

    @classmethod
    def values(cls) -> list[str]:
        return [member.value for member in cls]

    @classmethod
    def coerce(cls, raw: object, default: "StrEnum | None" = None) -> "StrEnum | None":
        """Best-effort parse of arbitrary user input into a member."""
        if isinstance(raw, cls):
            return raw
        if raw is None:
            return default
        text = str(raw).strip()
        if not text:
            return default
        lowered = text.lower().replace(" ", "_").replace("-", "_")
        for member in cls:
            if member.value.lower() == lowered or member.name.lower() == lowered:
                return member
        return default


class Position(StrEnum):
    """Fantasy-relevant player positions."""

    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"
    DST = "DST"


class Slot(StrEnum):
    """Roster slots. Eligibility is data-driven via ``SLOT_ELIGIBILITY``."""

    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    FLEX = "FLEX"           # RB/WR/TE
    WR_RB_FLEX = "WR/RB"    # RB/WR only
    WR_TE_FLEX = "WR/TE"    # WR/TE only
    SUPERFLEX = "SUPERFLEX"  # QB/RB/WR/TE
    K = "K"
    DST = "DST"
    BENCH = "BN"
    IR = "IR"


class DraftType(StrEnum):
    """Supported pick-ordering schemes.

    Every member here is fully simulated. There was an ``AUCTION`` member that was
    selectable and accepted but modelled nothing — it fell through to plain
    ascending order and emitted a warning — and it has been removed rather than
    left as a trap for someone who picks it and believes their bidding is being
    modelled.

    A league saved earlier with ``draft_type="auction"`` still loads: ``coerce``
    does not recognise the string and :class:`core.config.LeagueConfig` falls back
    to :attr:`SNAKE`, which is what the engine did with it anyway.
    """

    SNAKE = "snake"
    LINEAR = "linear"
    THIRD_ROUND_REVERSAL = "third_round_reversal"
    CUSTOM = "custom"


class LeagueFormat(StrEnum):
    """League lifecycle format."""

    REDRAFT = "redraft"
    KEEPER = "keeper"
    DYNASTY = "dynasty"


class ScoringPreset(StrEnum):
    """Common scoring presets; each expands into explicit rule values."""

    STANDARD = "standard"
    HALF_PPR = "half_ppr"
    FULL_PPR = "full_ppr"
    TE_PREMIUM = "te_premium"
    CUSTOM = "custom"


class Platform(StrEnum):
    """Fantasy platforms recognised for ADP / ranking provenance."""

    ESPN = "ESPN"
    YAHOO = "Yahoo"
    SLEEPER = "Sleeper"
    NFL = "NFL"
    CBS = "CBS"
    UNDERDOG = "Underdog"
    CUSTOM = "Custom"


class Archetype(StrEnum):
    """Fallback draft archetypes used when manager history is thin."""

    BALANCED = "balanced"
    BEST_PLAYER_AVAILABLE = "best_player_available"
    ZERO_RB = "zero_rb"
    HERO_RB = "hero_rb"
    ROBUST_RB = "robust_rb"
    EARLY_QB = "early_qb"
    LATE_QB = "late_round_qb"
    ELITE_TE = "elite_te"
    ROOKIE_HEAVY = "rookie_heavy"
    RANK_FOLLOWER = "platform_rank_follower"
    HIGH_VARIANCE = "high_variance"
    HOMER = "favorite_team_homer"
    AUTODRAFT = "autodraft_like"
    CUSTOM = "custom"


class DraftStatus(StrEnum):
    """Lifecycle of a mock draft."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETE = "complete"


class DraftMode(StrEnum):
    """How a mock draft is executed."""

    INTERACTIVE = "interactive"
    INSTANT = "instant"
    MONTE_CARLO = "monte_carlo"
    HISTORICAL_REPLAY = "historical_replay"


class RankingSource(StrEnum):
    """Which ordering to use when scoring candidates."""

    PLATFORM_ADP = "platform_adp"
    OVERALL_ADP = "overall_adp"
    EXPERT_CONSENSUS = "expert_consensus"
    PERSONAL = "personal"
    PROJECTION = "projection"
    LEAGUE_ADJUSTED = "league_adjusted"
    BLEND = "blend"


class InjuryStatus(StrEnum):
    """Player availability flags."""

    HEALTHY = "Healthy"
    QUESTIONABLE = "Questionable"
    DOUBTFUL = "Doubtful"
    OUT = "Out"
    IR = "IR"
    PUP = "PUP"
    SUSPENDED = "Suspended"


class ProvenanceKind(StrEnum):
    """Where a profile value came from — surfaced in the UI for honesty."""

    OBSERVED = "observed"           # computed from this manager's real picks
    MODEL_INFERRED = "inferred"     # shrunk / smoothed model estimate
    USER_ENTERED = "user"           # typed in by the user
    LEAGUE_FALLBACK = "league"      # borrowed from league-wide behaviour
    PLATFORM_FALLBACK = "platform"  # borrowed from platform behaviour
    BASELINE = "baseline"           # general fantasy prior / archetype


class RiskBand(StrEnum):
    """Coarse availability risk classification."""

    GONE = "very_likely_gone"
    LIKELY_GONE = "likely_gone"
    COIN_FLIP = "coin_flip"
    LIKELY_AVAILABLE = "likely_available"
    SAFE = "very_likely_available"


class RecommendationLens(StrEnum):
    """Perspectives offered by the recommendation engine."""

    BEST_OVERALL = "best_overall"
    BEST_FIT = "best_roster_fit"
    BEST_VALUE = "best_value"
    SAFEST = "safest"
    HIGHEST_UPSIDE = "highest_upside"
    SCARCITY = "scarcity"
    LAST_CHANCE = "last_chance"
    ALTERNATIVE = "strategic_alternative"
