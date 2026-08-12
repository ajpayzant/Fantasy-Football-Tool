"""Typed configuration objects.

Everything the simulation needs is expressed here as plain dataclasses so the
engine can be exercised from tests without Streamlit or a database. Model
weights live in :class:`ModelWeights` — there are no magic numbers buried in
the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from .constants import (
    REPLACEMENT_RANK_PER_TEAM,
    SLOT_ELIGIBILITY,
    SLOT_FILL_PRIORITY,
    STARTING_SLOTS,
)
from .enums import (
    Archetype,
    DraftType,
    LeagueFormat,
    Platform,
    Position,
    RankingSource,
    ScoringPreset,
    Slot,
)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ScoringRules:
    """Explicit per-event scoring values.

    Presets expand into these fields so downstream code never branches on a
    preset name. Fractional values are supported throughout.
    """

    # Passing
    pass_yards_per_point: float = 25.0
    pass_td: float = 4.0
    interception: float = -2.0
    pass_2pt: float = 2.0
    # Rushing
    rush_yards_per_point: float = 10.0
    rush_td: float = 6.0
    # Receiving
    rec_yards_per_point: float = 10.0
    rec_td: float = 6.0
    reception: float = 0.0
    te_premium_reception_bonus: float = 0.0
    # Misc
    fumble_lost: float = -2.0
    rush_rec_2pt: float = 2.0
    # Bonuses (yardage thresholds → extra points)
    bonus_pass_300_yards: float = 0.0
    bonus_pass_400_yards: float = 0.0
    bonus_rush_100_yards: float = 0.0
    bonus_rush_200_yards: float = 0.0
    bonus_rec_100_yards: float = 0.0
    bonus_rec_200_yards: float = 0.0
    bonus_long_td_40_plus: float = 0.0
    # Kicking
    kick_fg_made: float = 3.0
    kick_xp_made: float = 1.0
    # Defence / special teams
    dst_sack: float = 1.0
    dst_interception: float = 2.0
    dst_fumble_recovery: float = 2.0
    dst_touchdown: float = 6.0
    dst_safety: float = 2.0
    # Points-allowed tiers, in points per game finishing in that band. These are
    # the largest single component of real defence scoring — a defence scored on
    # sacks and turnovers alone projects roughly 25% low — so they are explicit
    # rather than folded into a single "shutout" bonus. Defaults are ESPN's.
    dst_points_allowed_0: float = 5.0
    dst_points_allowed_1_6: float = 4.0
    dst_points_allowed_7_13: float = 3.0
    dst_points_allowed_14_17: float = 1.0
    dst_points_allowed_18_21: float = 0.0
    dst_points_allowed_22_27: float = -1.0
    dst_points_allowed_28_34: float = -3.0
    dst_points_allowed_35_plus: float = -5.0
    preset: ScoringPreset = ScoringPreset.HALF_PPR

    @classmethod
    def from_preset(cls, preset: ScoringPreset | str, **overrides: Any) -> "ScoringRules":
        """Build rules from a named preset, then apply explicit overrides."""
        preset = ScoringPreset.coerce(preset, ScoringPreset.CUSTOM) or ScoringPreset.CUSTOM
        base: dict[str, Any] = {"preset": preset}
        if preset is ScoringPreset.STANDARD:
            base["reception"] = 0.0
        elif preset is ScoringPreset.HALF_PPR:
            base["reception"] = 0.5
        elif preset is ScoringPreset.FULL_PPR:
            base["reception"] = 1.0
        elif preset is ScoringPreset.TE_PREMIUM:
            base.update(reception=1.0, te_premium_reception_bonus=0.5)
        base.update(overrides)
        return cls(**base)

    def reception_value(self, position: Position) -> float:
        """Points per catch for a position, including any TE premium."""
        bonus = self.te_premium_reception_bonus if position is Position.TE else 0.0
        return self.reception + bonus

    @property
    def is_ppr(self) -> bool:
        return self.reception > 0

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        raw = asdict(self)
        raw["preset"] = str(self.preset)
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ScoringRules":
        raw = dict(raw or {})
        # ``dst_shutout`` was this field's name before the points-allowed tiers
        # existed. Migrated rather than dropped, because the filter below discards
        # unknown keys silently and a league saved earlier would lose the value.
        if "dst_shutout" in raw and "dst_points_allowed_0" not in raw:
            raw["dst_points_allowed_0"] = raw["dst_shutout"]
        known = {f for f in cls.__slots__}
        clean = {k: v for k, v in raw.items() if k in known}
        if "preset" in clean:
            clean["preset"] = ScoringPreset.coerce(clean["preset"], ScoringPreset.CUSTOM)
        return cls(**clean)


# ─────────────────────────────────────────────────────────────────────────────
# Roster
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class RosterSettings:
    """Slot counts plus positional limits.

    ``slots`` maps a :class:`Slot` to how many of that slot each team carries.
    Any slot may be zero or absent. Nothing assumes a "standard" lineup.
    """

    slots: dict[Slot, int] = field(default_factory=lambda: {
        Slot.QB: 1, Slot.RB: 2, Slot.WR: 2, Slot.TE: 1,
        Slot.FLEX: 1, Slot.K: 1, Slot.DST: 1, Slot.BENCH: 7,
    })
    position_max: dict[Position, int] = field(default_factory=dict)
    position_min: dict[Position, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.slots = {
            Slot.coerce(k, None) or k: int(v)
            for k, v in self.slots.items() if int(v) > 0
        }
        self.position_max = {
            Position.coerce(k, None) or k: int(v) for k, v in self.position_max.items()
        }
        self.position_min = {
            Position.coerce(k, None) or k: int(v) for k, v in self.position_min.items()
        }

    def count(self, slot: Slot) -> int:
        return int(self.slots.get(slot, 0))

    @property
    def starting_slots(self) -> dict[Slot, int]:
        return {s: n for s, n in self.slots.items() if s in STARTING_SLOTS and n > 0}

    @property
    def starters_total(self) -> int:
        return sum(self.starting_slots.values())

    @property
    def bench_total(self) -> int:
        return self.count(Slot.BENCH)

    @property
    def ir_total(self) -> int:
        return self.count(Slot.IR)

    @property
    def roster_size(self) -> int:
        """Draftable roster size (IR slots are not drafted into)."""
        return self.starters_total + self.bench_total

    @property
    def is_superflex(self) -> bool:
        return self.count(Slot.SUPERFLEX) > 0

    @property
    def is_two_qb(self) -> bool:
        return self.count(Slot.QB) >= 2

    def ordered_starting_slots(self) -> list[Slot]:
        """Starting slots expanded to one entry per seat, most-restrictive first."""
        out: list[Slot] = []
        for slot in SLOT_FILL_PRIORITY:
            out.extend([slot] * self.count(slot))
        # Any starting slot not in the priority list (future additions).
        for slot, n in self.starting_slots.items():
            if slot not in SLOT_FILL_PRIORITY:
                out.extend([slot] * n)
        return out

    def starting_demand(self) -> dict[Position, float]:
        """Expected starters demanded per position, splitting flex slots evenly."""
        demand: dict[Position, float] = {p: 0.0 for p in Position}
        for slot, n in self.starting_slots.items():
            eligible = SLOT_ELIGIBILITY.get(slot, frozenset())
            if not eligible:
                continue
            share = n / len(eligible)
            for pos in eligible:
                demand[pos] += share
        return demand

    def max_for(self, position: Position) -> int | None:
        return self.position_max.get(position)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots": {str(s): n for s, n in self.slots.items()},
            "position_max": {str(p): n for p, n in self.position_max.items()},
            "position_min": {str(p): n for p, n in self.position_min.items()},
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RosterSettings":
        raw = raw or {}
        return cls(
            slots={Slot.coerce(k, Slot.BENCH): int(v)
                   for k, v in (raw.get("slots") or {}).items()},
            position_max={Position.coerce(k, Position.RB): int(v)
                          for k, v in (raw.get("position_max") or {}).items()},
            position_min={Position.coerce(k, Position.RB): int(v)
                          for k, v in (raw.get("position_min") or {}).items()},
        )


# ─────────────────────────────────────────────────────────────────────────────
# League
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class LeagueConfig:
    """Complete definition of a league's rules and draft structure."""

    name: str = "My League"
    season: int = 2026
    platform: Platform = Platform.ESPN
    team_count: int = 12
    rounds: int = 16
    draft_type: DraftType = DraftType.SNAKE
    league_format: LeagueFormat = LeagueFormat.REDRAFT
    scoring: ScoringRules = field(default_factory=ScoringRules)
    roster: RosterSettings = field(default_factory=RosterSettings)
    user_draft_slot: int = 1
    draft_date: str | None = None
    reversal_round: int = 3
    """Round at which a third-round-reversal draft flips (configurable)."""
    custom_round_order: dict[int, list[int]] = field(default_factory=dict)
    """Round number → explicit list of draft slots, for ``DraftType.CUSTOM``."""
    notes: str = ""
    league_id: int | None = None

    def __post_init__(self) -> None:
        self.platform = Platform.coerce(self.platform, Platform.CUSTOM) or Platform.CUSTOM
        self.draft_type = DraftType.coerce(self.draft_type, DraftType.SNAKE) or DraftType.SNAKE
        self.league_format = (
            LeagueFormat.coerce(self.league_format, LeagueFormat.REDRAFT) or LeagueFormat.REDRAFT
        )
        self.team_count = int(self.team_count)
        self.rounds = int(self.rounds)
        self.user_draft_slot = int(self.user_draft_slot)
        self.custom_round_order = {
            int(k): [int(s) for s in v] for k, v in (self.custom_round_order or {}).items()
        }

    @property
    def total_picks(self) -> int:
        return self.team_count * self.rounds

    @property
    def allows_keepers(self) -> bool:
        return self.league_format in (LeagueFormat.KEEPER, LeagueFormat.DYNASTY)

    def replacement_rank(self, position: Position) -> float:
        """Approximate positional rank at which value hits replacement level."""
        per_team = REPLACEMENT_RANK_PER_TEAM.get(position, 1.0)
        if position is Position.QB and self.roster.is_superflex:
            per_team = 1.9
        if position is Position.QB and self.roster.is_two_qb:
            per_team = 2.2
        return max(1.0, per_team * self.team_count)

    def with_(self, **changes: Any) -> "LeagueConfig":
        """Return a modified copy (configs are treated as immutable in the engine)."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "season": self.season,
            "platform": str(self.platform),
            "team_count": self.team_count,
            "rounds": self.rounds,
            "draft_type": str(self.draft_type),
            "league_format": str(self.league_format),
            "scoring": self.scoring.to_dict(),
            "roster": self.roster.to_dict(),
            "user_draft_slot": self.user_draft_slot,
            "draft_date": self.draft_date,
            "reversal_round": self.reversal_round,
            "custom_round_order": {str(k): v for k, v in self.custom_round_order.items()},
            "notes": self.notes,
            "league_id": self.league_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LeagueConfig":
        raw = dict(raw or {})
        return cls(
            name=raw.get("name", "My League"),
            season=int(raw.get("season", 2026)),
            platform=raw.get("platform", Platform.ESPN),
            team_count=int(raw.get("team_count", 12)),
            rounds=int(raw.get("rounds", 16)),
            draft_type=raw.get("draft_type", DraftType.SNAKE),
            league_format=raw.get("league_format", LeagueFormat.REDRAFT),
            scoring=ScoringRules.from_dict(raw.get("scoring") or {}),
            roster=RosterSettings.from_dict(raw.get("roster") or {}),
            user_draft_slot=int(raw.get("user_draft_slot", 1)),
            draft_date=raw.get("draft_date"),
            reversal_round=int(raw.get("reversal_round", 3)),
            custom_round_order={int(k): v for k, v in (raw.get("custom_round_order") or {}).items()},
            notes=raw.get("notes", ""),
            league_id=raw.get("league_id"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model weights
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ModelWeights:
    """Weights for every term in the candidate utility score.

    Utility is computed on a roughly comparable scale per component (most
    components are normalised to ~[-1, 1] or [0, 1] before weighting), so these
    numbers are directly interpretable as relative importance.
    """

    adp: float = 1.00
    projection: float = 0.55
    tier: float = 0.30
    value_over_replacement: float = 0.45
    roster_need: float = 0.70
    positional_scarcity: float = 0.35
    manager_position_preference: float = 0.60
    round_specific_preference: float = 0.40
    platform_rank_dependence: float = 0.50
    rookie_preference: float = 0.25
    favorite_team_preference: float = 0.20
    named_player_preference: float = 0.45
    """Pull toward (or away from) players the user named for this manager.

    Set above the positional and team preferences because it is far more specific:
    "he always takes Kupp" is a statement about one player, and a weight that only
    matched a positional lean would be drowned out by board value on every pick.
    Zero for every manager the user has said nothing about, so it changes no
    simulation until someone fills it in.
    """
    stack: float = 0.15
    handcuff: float = 0.15
    positional_run: float = 0.30
    expected_availability: float = 0.35
    strategy: float = 0.45
    randomness: float = 0.10
    injury_penalty: float = 0.60
    roster_imbalance_penalty: float = 1.50
    """Cost of a pick that strands a starting slot, per seat irrecoverably lost.

    Set above the widest realistic board-value gap between two candidates on
    purpose. This term only engages once a manager has no spare picks left, and at
    that point the choice is between a slightly better bench player and starting
    the season with an empty seat that scores zero every week. No board edge is
    worth that, so the penalty has to dominate rather than merely compete: at 0.30
    it lost to a 0.6 value edge and rosters finished with unfilled K/DST seats.
    """
    positional_limit_penalty: float = 0.80

    def to_dict(self) -> dict[str, float]:
        return {f: float(getattr(self, f)) for f in self.__slots__}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelWeights":
        clean = {k: float(v) for k, v in (raw or {}).items() if k in cls.__slots__}
        return cls(**clean)

    def scaled(self, factor: float) -> "ModelWeights":
        return ModelWeights(**{k: v * factor for k, v in self.to_dict().items()})


@dataclass(slots=True)
class ShrinkageConfig:
    """Controls hierarchical blending of manager / league / platform / baseline.

    ``manager_weight = n / (n + prior_strength)`` — a manager with
    ``prior_strength`` observations gets 50% personalisation. The remainder is
    split across the fallback levels by the ratios below.
    """

    prior_strength: float = 24.0
    """Pseudo-count for manager-level shrinkage (in picks)."""
    season_prior_strength: float = 2.0
    """Pseudo-count in *seasons*, for metrics observed once per draft rather than
    once per pick (e.g. the round of a manager's first quarterback)."""
    league_share: float = 0.45
    platform_share: float = 0.30
    baseline_share: float = 0.25
    recency_half_life_seasons: float = 1.5
    """Half-life of the exponential recency decay applied to historical picks."""
    min_picks_for_observed: int = 8
    """Below this many picks a metric is labelled inferred rather than observed."""

    def fallback_shares(self) -> tuple[float, float, float]:
        total = self.league_share + self.platform_share + self.baseline_share
        if total <= 0:
            return (0.0, 0.0, 1.0)
        return (
            self.league_share / total,
            self.platform_share / total,
            self.baseline_share / total,
        )

    def manager_weight(self, sample_size: float) -> float:
        n = max(0.0, float(sample_size))
        return n / (n + max(1e-9, self.prior_strength))

    def season_weight(self, seasons: float) -> float:
        """Shrinkage weight for a per-season observation (e.g. first-QB round)."""
        n = max(0.0, float(seasons))
        return n / (n + max(1e-9, self.season_prior_strength))

    def recency_decay(self, seasons_ago: float) -> float:
        """Exponential weight for a pick made ``seasons_ago`` seasons back."""
        if self.recency_half_life_seasons <= 0:
            return 1.0
        return float(0.5 ** (max(0.0, seasons_ago) / self.recency_half_life_seasons))

    def to_dict(self) -> dict[str, float]:
        return {f: float(getattr(self, f)) for f in self.__slots__}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ShrinkageConfig":
        clean = {k: float(v) for k, v in (raw or {}).items() if k in cls.__slots__}
        if "min_picks_for_observed" in clean:
            clean["min_picks_for_observed"] = int(clean["min_picks_for_observed"])
        return cls(**clean)


@dataclass(slots=True)
class ProfileEstimationConfig:
    """Scales that convert raw historical statistics into model parameters.

    Every number here is an *anchor*: the observed statistic is divided by (or
    exponentially damped against) the anchor so the resulting parameter lands on
    the same 0-1 scale the archetype priors use. Anchors are configurable so a
    user whose league behaves unusually can recalibrate without touching code.
    """

    reach_clip_picks: float = 60.0
    """|ADP − pick| beyond this is treated as bad data and dropped."""
    rank_delta_scale_picks: float = 18.0
    """Mean |rank − pick| at which ``rank_dependence`` decays to 1/e (~0.37)."""
    predictability_scale_picks: float = 14.0
    """Reach standard deviation at which ``predictability`` decays to 1/e."""
    fill_rate_anchor: float = 0.62
    """Share of picks that fill an open starting slot for a league-average
    manager. Maps to ``need_dependence`` = 0.5."""
    run_continue_anchor: float = 0.34
    """Share of picks continuing a positional run for a league-average manager.
    Maps to ``run_chase`` = 0.5."""
    tier_cliff_anchor: float = 0.18
    """Share of picks that take the last player of a tier at their position for a
    league-average manager. Maps to ``tier_sensitivity`` = 0.5."""
    upside_anchor: float = 0.5
    """Share of picks with above-median ceiling that maps to ``risk_preference``
    = 0.5. Only used when the historical file carries ceiling data."""
    position_bias_timing_scale: float = 1.5
    """Utility units per unit of normalised draft-position lean vs the league."""
    position_bias_share_scale: float = 2.0
    """Utility units per unit of positional pick-share lean vs the league."""
    position_bias_clip: float = 0.60
    """Maximum magnitude of any derived positional bias, in utility units."""
    early_rounds: int = 3
    """Rounds counted as 'early' for the early-round positional bias."""
    favorite_team_min_share: float = 0.15
    """Minimum share of picks before a team counts as a favourite."""
    user_preference_weight: float = 0.60
    """How strongly a value the user typed in pulls a parameter toward it. 1.0
    would make user input an outright override."""
    run_window_picks: int = 6
    """Look-back window (picks) used when labelling historical positional runs."""
    run_threshold_picks: int = 2
    """Same-position picks inside the window that constitute a run."""

    def to_dict(self) -> dict[str, float]:
        return {f: float(getattr(self, f)) for f in self.__slots__}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ProfileEstimationConfig":
        clean = {k: float(v) for k, v in (raw or {}).items() if k in cls.__slots__}
        for key in ("early_rounds", "run_window_picks", "run_threshold_picks"):
            if key in clean:
                clean[key] = int(clean[key])
        return cls(**clean)


@dataclass(slots=True)
class SimulationConfig:
    """Knobs for the pick model and Monte Carlo layer."""

    weights: ModelWeights = field(default_factory=ModelWeights)
    shrinkage: ShrinkageConfig = field(default_factory=ShrinkageConfig)
    estimation: ProfileEstimationConfig = field(default_factory=ProfileEstimationConfig)
    ranking_source: RankingSource = RankingSource.BLEND
    blend_weights: dict[str, float] = field(default_factory=lambda: {
        "platform_adp": 0.45, "overall_adp": 0.25, "projection": 0.30,
    })
    base_temperature: float = 0.55
    """Softmax temperature at average predictability."""
    temperature_range: tuple[float, float] = (0.22, 1.35)
    """Clamp for per-manager temperature (predictable → volatile)."""
    candidate_pool_size: int = 40
    """How many top-utility players enter the softmax. Keeps sims fast."""
    adp_sigma_floor: float = 6.0
    """Minimum ADP standard deviation when a player file supplies none."""
    adp_sigma_round_growth: float = 1.6
    """Extra ADP sigma per round — later rounds are noisier."""
    run_windows: tuple[int, ...] = (3, 6, 12)
    """Look-back windows (in picks) used to detect positional runs."""
    availability_simulations: int = 120
    """Rollouts used when estimating survival to the user's next pick."""
    monte_carlo_default_runs: int = 200
    reach_scale_picks: float = 12.0
    """ADP delta (picks) treated as one unit of 'reach' when scoring utility."""
    random_seed: int | None = None

    def temperature_for(self, predictability: float) -> float:
        """Map predictability in [0, 1] to a softmax temperature.

        Higher predictability → lower temperature → more deterministic picks.
        """
        lo, hi = self.temperature_range
        p = min(1.0, max(0.0, float(predictability)))
        # Predictability 0.5 reproduces base_temperature.
        if p >= 0.5:
            t = lo + (self.base_temperature - lo) * ((1.0 - p) / 0.5)
        else:
            t = self.base_temperature + (hi - self.base_temperature) * ((0.5 - p) / 0.5)
        return float(min(hi, max(lo, t)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.to_dict(),
            "shrinkage": self.shrinkage.to_dict(),
            "estimation": self.estimation.to_dict(),
            "ranking_source": str(self.ranking_source),
            "blend_weights": dict(self.blend_weights),
            "base_temperature": self.base_temperature,
            "temperature_range": list(self.temperature_range),
            "candidate_pool_size": self.candidate_pool_size,
            "adp_sigma_floor": self.adp_sigma_floor,
            "adp_sigma_round_growth": self.adp_sigma_round_growth,
            "run_windows": list(self.run_windows),
            "availability_simulations": self.availability_simulations,
            "monte_carlo_default_runs": self.monte_carlo_default_runs,
            "reach_scale_picks": self.reach_scale_picks,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SimulationConfig":
        raw = dict(raw or {})
        cfg = cls(
            weights=ModelWeights.from_dict(raw.get("weights") or {}),
            shrinkage=ShrinkageConfig.from_dict(raw.get("shrinkage") or {}),
            estimation=ProfileEstimationConfig.from_dict(raw.get("estimation") or {}),
            ranking_source=RankingSource.coerce(
                raw.get("ranking_source"), RankingSource.BLEND) or RankingSource.BLEND,
            blend_weights=dict(raw.get("blend_weights") or {
                "platform_adp": 0.45, "overall_adp": 0.25, "projection": 0.30}),
        )
        for key in ("base_temperature", "adp_sigma_floor", "adp_sigma_round_growth",
                    "reach_scale_picks"):
            if key in raw:
                setattr(cfg, key, float(raw[key]))
        for key in ("candidate_pool_size", "availability_simulations",
                    "monte_carlo_default_runs"):
            if key in raw:
                setattr(cfg, key, int(raw[key]))
        if raw.get("temperature_range"):
            lo, hi = raw["temperature_range"]
            cfg.temperature_range = (float(lo), float(hi))
        if raw.get("run_windows"):
            cfg.run_windows = tuple(int(w) for w in raw["run_windows"])
        if raw.get("random_seed") is not None:
            cfg.random_seed = int(raw["random_seed"])
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Archetype parameter table
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ArchetypeParams:
    """Starting model parameters for a fallback archetype.

    These are *priors*, not rules: they seed a profile which the probabilistic
    engine then blends with roster need, runs, and ADP.
    """

    label: str
    description: str
    position_bias: dict[Position, float] = field(default_factory=dict)
    """Additive utility bias per position, in utility units."""
    early_round_position_bias: dict[Position, float] = field(default_factory=dict)
    """Extra bias applied in rounds 1-3 only."""
    first_qb_round: float = 8.0
    first_te_round: float = 8.0
    reach_mean_picks: float = 0.0
    """Positive = drafts ahead of ADP on average."""
    reach_stdev_picks: float = 9.0
    rank_dependence: float = 0.5
    """0 = ignores platform ranks, 1 = drafts strictly by rank."""
    need_dependence: float = 0.5
    rookie_rate: float = 0.10
    stack_rate: float = 0.08
    handcuff_rate: float = 0.06
    favorite_team_rate: float = 0.05
    run_chase: float = 0.5
    """0 = disciplined against positional runs, 1 = chases them."""
    tier_sensitivity: float = 0.5
    predictability: float = 0.5
    risk_preference: float = 0.5
    """0 = floor-seeking, 1 = ceiling-seeking."""


def _pos(**kwargs: float) -> dict[Position, float]:
    return {Position[k]: float(v) for k, v in kwargs.items()}


ARCHETYPE_PARAMS: dict[Archetype, ArchetypeParams] = {
    Archetype.BALANCED: ArchetypeParams(
        label="Balanced",
        description="Drafts near consensus, fills needs without extreme lean.",
    ),
    Archetype.BEST_PLAYER_AVAILABLE: ArchetypeParams(
        label="Best Player Available",
        description="Follows value over need; rarely reaches.",
        need_dependence=0.22, rank_dependence=0.72, reach_mean_picks=-1.5,
        reach_stdev_picks=6.0, predictability=0.72, tier_sensitivity=0.68,
    ),
    Archetype.ZERO_RB: ArchetypeParams(
        label="Zero RB",
        description="Avoids RB early, loads WR/TE, attacks RB from the middle rounds.",
        position_bias=_pos(RB=-0.25, WR=0.30, TE=0.15),
        early_round_position_bias=_pos(RB=-0.95, WR=0.60),
        need_dependence=0.35, predictability=0.62, risk_preference=0.68,
    ),
    Archetype.HERO_RB: ArchetypeParams(
        label="Hero RB",
        description="One elite RB early, then WR-heavy for several rounds.",
        early_round_position_bias=_pos(RB=0.35, WR=0.25),
        position_bias=_pos(RB=-0.10, WR=0.18),
        predictability=0.58, risk_preference=0.55,
    ),
    Archetype.ROBUST_RB: ArchetypeParams(
        label="Robust RB",
        description="Prioritises multiple early running backs.",
        position_bias=_pos(RB=0.32),
        early_round_position_bias=_pos(RB=0.85, WR=-0.20),
        need_dependence=0.6, predictability=0.62, risk_preference=0.38,
    ),
    Archetype.EARLY_QB: ArchetypeParams(
        label="Early QB",
        description="Takes a quarterback well before league average.",
        position_bias=_pos(QB=0.45), first_qb_round=4.0,
        reach_mean_picks=4.0, predictability=0.55,
    ),
    Archetype.LATE_QB: ArchetypeParams(
        label="Late-Round QB",
        description="Waits on quarterback, often the last starter drafted.",
        position_bias=_pos(QB=-0.35), first_qb_round=11.0,
        early_round_position_bias=_pos(QB=-1.10), predictability=0.65,
    ),
    Archetype.ELITE_TE: ArchetypeParams(
        label="Elite TE",
        description="Pays up for a top tight end inside the first few rounds.",
        position_bias=_pos(TE=0.40), first_te_round=3.0,
        early_round_position_bias=_pos(TE=0.75), reach_mean_picks=3.0,
    ),
    Archetype.ROOKIE_HEAVY: ArchetypeParams(
        label="Rookie Heavy",
        description="Reaches for first-year players and upside profiles.",
        rookie_rate=0.34, reach_mean_picks=6.0, reach_stdev_picks=13.0,
        risk_preference=0.80, predictability=0.40,
    ),
    Archetype.RANK_FOLLOWER: ArchetypeParams(
        label="Platform Rank Follower",
        description="Drafts almost strictly off the platform's default list.",
        rank_dependence=0.95, need_dependence=0.28, reach_mean_picks=-0.5,
        reach_stdev_picks=3.5, predictability=0.90, tier_sensitivity=0.35,
    ),
    Archetype.HIGH_VARIANCE: ArchetypeParams(
        label="High Variance",
        description="Unpredictable; frequent large reaches in both directions.",
        reach_stdev_picks=20.0, predictability=0.15, risk_preference=0.75,
        rank_dependence=0.30, run_chase=0.7,
    ),
    Archetype.HOMER: ArchetypeParams(
        label="Favorite-Team Homer",
        description="Repeatedly drafts players from one favourite NFL team.",
        favorite_team_rate=0.30, reach_mean_picks=5.0, predictability=0.42,
    ),
    Archetype.AUTODRAFT: ArchetypeParams(
        label="Autodraft-Like",
        description="Effectively drafts the default queue in order.",
        rank_dependence=1.0, need_dependence=0.45, reach_stdev_picks=2.0,
        predictability=0.96, run_chase=0.15, rookie_rate=0.05,
    ),
    Archetype.CUSTOM: ArchetypeParams(
        label="Custom",
        description="User-defined parameters only.",
    ),
}


def archetype_params(archetype: Archetype | str | None) -> ArchetypeParams:
    """Look up archetype priors, defaulting to Balanced."""
    key = Archetype.coerce(archetype, Archetype.BALANCED) or Archetype.BALANCED
    return ARCHETYPE_PARAMS.get(key, ARCHETYPE_PARAMS[Archetype.BALANCED])


# ─────────────────────────────────────────────────────────────────────────────
# Paths / app settings
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class AppPaths:
    """Filesystem locations, resolved relative to the package root."""

    root: str = ""
    database: str = ""
    cache: str = ""
    exports: str = ""

    @classmethod
    def default(cls) -> "AppPaths":
        import os

        # There is deliberately no ``sample_data`` path here any more: the synthetic
        # league moved to ``tests/fixtures/sample_league`` and the app has no route
        # to it, so a path pointing at a directory that no longer exists would only
        # invite something to start reading from it again.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return cls(
            root=root,
            database=os.path.join(root, "data", "fantasy_mock_draft.db"),
            cache=os.path.join(root, "data", "cache"),
            exports=os.path.join(root, "data", "exports"),
        )


DEFAULT_PATHS = AppPaths.default()


def league_presets() -> dict[str, dict[str, Any]]:
    """Convenience roster presets offered in the UI (fully editable after load)."""
    return {
        "Standard 1QB (12-team)": {
            "slots": {Slot.QB: 1, Slot.RB: 2, Slot.WR: 2, Slot.TE: 1,
                      Slot.FLEX: 1, Slot.K: 1, Slot.DST: 1, Slot.BENCH: 6},
        },
        "Superflex (12-team)": {
            "slots": {Slot.QB: 1, Slot.RB: 2, Slot.WR: 2, Slot.TE: 1,
                      Slot.FLEX: 1, Slot.SUPERFLEX: 1, Slot.BENCH: 7},
        },
        "Two-QB (12-team)": {
            "slots": {Slot.QB: 2, Slot.RB: 2, Slot.WR: 3, Slot.TE: 1,
                      Slot.FLEX: 1, Slot.BENCH: 7},
        },
        "3WR PPR (10-team)": {
            "slots": {Slot.QB: 1, Slot.RB: 2, Slot.WR: 3, Slot.TE: 1,
                      Slot.FLEX: 1, Slot.K: 1, Slot.DST: 1, Slot.BENCH: 6},
        },
        "No K/DST (12-team)": {
            "slots": {Slot.QB: 1, Slot.RB: 2, Slot.WR: 3, Slot.TE: 1,
                      Slot.FLEX: 2, Slot.BENCH: 7},
        },
    }


def eligible_slots_for(position: Position, roster: RosterSettings) -> list[Slot]:
    """Starting slots in this league that ``position`` can legally fill."""
    return [
        slot for slot in roster.ordered_starting_slots()
        if position in SLOT_ELIGIBILITY.get(slot, frozenset())
    ]


def positions_in_use(roster: RosterSettings) -> set[Position]:
    """Positions worth drafting given the league's slot configuration."""
    used: set[Position] = set()
    for slot, n in roster.starting_slots.items():
        if n > 0:
            used |= set(SLOT_ELIGIBILITY.get(slot, frozenset()))
    if not used:
        used = {Position.QB, Position.RB, Position.WR, Position.TE}
    return used


def summarize_positions(positions: Iterable[Position]) -> str:
    order = [Position.QB, Position.RB, Position.WR, Position.TE, Position.K, Position.DST]
    present = [p for p in order if p in set(positions)]
    return "/".join(str(p) for p in present)
