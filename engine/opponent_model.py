"""Building manager behaviour profiles from historical draft data.

This is the module that makes the simulator league-aware. For each manager it
estimates the parameters in :data:`models.manager.PARAM_KEYS` from that
manager's own picks, then blends the estimate with wider evidence so a manager
with two picks on file is not modelled as confidently as one with sixty:

    value = manager_weight · manager_estimate
          + (1 − manager_weight) · (league · w_l + platform · w_p + baseline · w_b)

``manager_weight = n / (n + prior_strength)`` (see
:meth:`core.config.ShrinkageConfig.manager_weight`), where *n* is the
recency-weighted pick count — picks from older seasons count for less via an
exponential half-life.

Every resulting value carries a :class:`~core.enums.ProvenanceKind` so the UI can
state plainly whether a number was observed, inferred, typed in by the user, or
borrowed from a fallback. Anything the model cannot estimate falls back to the
manager's archetype rather than to a silent zero.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from core.config import (
    LeagueConfig,
    ProfileEstimationConfig,
    ShrinkageConfig,
    SimulationConfig,
    archetype_params,
)
from core.enums import Archetype, Position, ProvenanceKind
from models.draft import DraftHistory, HistoricalPick
from models.league import League
from models.manager import (
    PARAM_KEYS,
    UNIT_PARAMS,
    Manager,
    ManagerPreferences,
    ManagerProfile,
    apply_archetype_params,
    baseline_profile,
    normalize_manager_key,
    param_default,
)
from engine.features import HistoryStats, annotate_history, summarize_history

LOGGER = logging.getLogger("fantasy_mock_draft.opponent_model")


# ─────────────────────────────────────────────────────────────────────────────
# Weighted statistics
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class WeightedStat:
    """A recency-weighted mean, tracking its own effective sample size."""

    total: float = 0.0
    weight: float = 0.0
    square_total: float = 0.0
    count: int = 0

    def add(self, value: float, weight: float = 1.0) -> None:
        if weight <= 0:
            return
        self.total += float(value) * weight
        self.square_total += float(value) ** 2 * weight
        self.weight += weight
        self.count += 1

    @property
    def mean(self) -> float | None:
        if self.weight <= 0:
            return None
        return self.total / self.weight

    @property
    def stdev(self) -> float | None:
        """Population standard deviation of the weighted sample."""
        mean = self.mean
        if mean is None or self.count < 2:
            return None
        variance = max(0.0, self.square_total / self.weight - mean**2)
        return math.sqrt(variance)

    @property
    def n(self) -> float:
        """Effective (recency-weighted) sample size."""
        return self.weight


@dataclass(slots=True)
class ManagerObservations:
    """Raw recency-weighted statistics for one manager, before any shrinkage.

    Kept separate from the profile so the Manager Profile page can show the raw
    observation next to the shrunk model value, and so tests can assert on the
    estimation step independently of the blending step.
    """

    manager_key: str
    manager_name: str
    picks: int = 0
    weighted_picks: float = 0.0
    seasons: tuple[int, ...] = ()
    drafts: int = 0
    reach: WeightedStat = field(default_factory=WeightedStat)
    """Picks *ahead* of ADP (positive = reach)."""
    rank_gap: WeightedStat = field(default_factory=WeightedStat)
    """|platform rank − overall pick|, the inverse signal for rank dependence."""
    rank_inversions: WeightedStat = field(default_factory=WeightedStat)
    """Better-ranked players this manager left on the board, per pick.

    Unlike :attr:`rank_gap` this is comparable across league sizes, so it — not
    the gap — is what the list-follower archetype thresholds test. See
    :attr:`models.draft.HistoricalPick.rank_inversions`.
    """
    fill_rate: WeightedStat = field(default_factory=WeightedStat)
    run_continue_rate: WeightedStat = field(default_factory=WeightedStat)
    tier_cliff_rate: WeightedStat = field(default_factory=WeightedStat)
    rookie_rate: WeightedStat = field(default_factory=WeightedStat)
    stack_rate: WeightedStat = field(default_factory=WeightedStat)
    handcuff_rate: WeightedStat = field(default_factory=WeightedStat)
    position_share: dict[Position, float] = field(default_factory=dict)
    early_position_share: dict[Position, float] = field(default_factory=dict)
    """Share of this manager's *early-round* picks spent on each position."""
    early_picks: float = 0.0
    """Recency-weighted count of early-round picks behind ``early_position_share``."""
    mean_pick_by_position: dict[Position, WeightedStat] = field(default_factory=dict)
    first_round_by_position: dict[Position, WeightedStat] = field(default_factory=dict)
    position_rate_by_round: dict[int, dict[Position, float]] = field(default_factory=dict)
    team_share: dict[str, float] = field(default_factory=dict)
    repeat_players: dict[str, int] = field(default_factory=dict)

    @property
    def has_data(self) -> bool:
        return self.picks > 0


# ─────────────────────────────────────────────────────────────────────────────
# Observation extraction
# ─────────────────────────────────────────────────────────────────────────────
def observe_manager(
    manager_name: str,
    history: DraftHistory,
    *,
    shrinkage: ShrinkageConfig | None = None,
    estimation: ProfileEstimationConfig | None = None,
    reference_season: int | None = None,
) -> ManagerObservations:
    """Compute recency-weighted statistics for one manager's historical picks.

    ``reference_season`` is the season being *predicted*. Picks from that season
    and later are excluded outright — in a backtest they are the answer, so
    reading them would be look-ahead — and recency decay for the remainder is
    measured back from it. Without it, every season in ``history`` is used and
    decay is measured from the most recent.
    """
    shrinkage = shrinkage or ShrinkageConfig()
    estimation = estimation or ProfileEstimationConfig()
    key = normalize_manager_key(manager_name)
    picks = [p for p in history.picks_for(manager_name) if not p.is_keeper]
    if reference_season is not None:
        picks = [p for p in picks if p.season < int(reference_season)]

    observations = ManagerObservations(manager_key=key, manager_name=manager_name)
    if not picks:
        return observations

    latest = reference_season or (history.latest_season or max(p.season for p in picks))
    observations.picks = len(picks)
    observations.seasons = tuple(sorted({p.season for p in picks}))
    observations.drafts = len(observations.seasons)

    position_weight: dict[Position, float] = {}
    team_weight: dict[str, float] = {}
    round_weight: dict[int, dict[Position, float]] = {}
    round_total: dict[int, float] = {}
    early_weight: dict[Position, float] = {}
    early_total = 0.0
    total_weight = 0.0
    first_at_position: dict[tuple[int, Position], HistoricalPick] = {}

    for pick in picks:
        # Picks at or after `latest` were already filtered out above when a
        # reference season was given, so this only damps genuinely older picks.
        seasons_ago = max(0.0, float(latest) - float(pick.season))
        weight = shrinkage.recency_decay(seasons_ago)
        if weight <= 0:
            continue
        total_weight += weight

        if pick.adp_delta is not None:
            # adp_delta is already signed reach-positive; no inversion needed.
            observations.reach.add(float(pick.adp_delta), weight)
        if pick.rank_delta is not None:
            observations.rank_gap.add(abs(float(pick.rank_delta)), weight)
        if pick.rank_inversions is not None:
            observations.rank_inversions.add(float(pick.rank_inversions), weight)
        observations.fill_rate.add(1.0 if pick.filled_starting_slot else 0.0, weight)
        observations.run_continue_rate.add(1.0 if pick.continued_run else 0.0, weight)
        if pick.same_tier_remaining is not None:
            observations.tier_cliff_rate.add(
                1.0 if pick.same_tier_remaining == 0 else 0.0, weight
            )
        observations.rookie_rate.add(1.0 if pick.is_rookie else 0.0, weight)
        observations.stack_rate.add(1.0 if pick.was_stack else 0.0, weight)
        observations.handcuff_rate.add(1.0 if pick.was_handcuff else 0.0, weight)

        if pick.position is not None:
            position_weight[pick.position] = position_weight.get(pick.position, 0.0) + weight
            observations.mean_pick_by_position.setdefault(
                pick.position, WeightedStat()
            ).add(float(pick.overall_pick), weight)
            rnd = int(pick.round_number or 1)
            round_weight.setdefault(rnd, {})[pick.position] = (
                round_weight.setdefault(rnd, {}).get(pick.position, 0.0) + weight
            )
            round_total[rnd] = round_total.get(rnd, 0.0) + weight
            if rnd <= int(estimation.early_rounds):
                early_weight[pick.position] = (
                    early_weight.get(pick.position, 0.0) + weight
                )
                early_total += weight
            marker = (pick.season, pick.position)
            if marker not in first_at_position:
                first_at_position[marker] = pick

        if pick.nfl_team:
            team = pick.nfl_team.upper()
            team_weight[team] = team_weight.get(team, 0.0) + weight
        if pick.player_name:
            observations.repeat_players[pick.player_name] = (
                observations.repeat_players.get(pick.player_name, 0) + 1
            )

    observations.weighted_picks = total_weight
    if total_weight > 0:
        observations.position_share = {
            pos: w / total_weight for pos, w in position_weight.items()
        }
        observations.team_share = {t: w / total_weight for t, w in team_weight.items()}
    observations.early_picks = early_total
    if early_total > 0:
        observations.early_position_share = {
            pos: w / early_total for pos, w in early_weight.items()
        }
    observations.position_rate_by_round = {
        rnd: {pos: w / round_total[rnd] for pos, w in rates.items()}
        for rnd, rates in round_weight.items()
        if round_total.get(rnd)
    }

    # First-pick-at-position rounds, one observation per season.
    for (season, position), pick in first_at_position.items():
        weight = shrinkage.recency_decay(max(0.0, float(latest) - float(season)))
        observations.first_round_by_position.setdefault(position, WeightedStat()).add(
            float(pick.round_number or 1), weight
        )
    return observations


# ─────────────────────────────────────────────────────────────────────────────
# Estimation: observations → parameter values
# ─────────────────────────────────────────────────────────────────────────────
def estimate_parameters(
    observations: ManagerObservations,
    estimation: ProfileEstimationConfig,
    *,
    shrinkage: ShrinkageConfig | None = None,
) -> dict[str, tuple[float, float]]:
    """Map raw observations to ``{param: (value, effective sample size)}``.

    A parameter is absent from the result when the data cannot support it — the
    caller then falls back rather than inventing a number. Sample sizes are
    returned per parameter because they differ: a manager may have 60 picks but
    ADP on only 12 of them.
    """
    shrinkage = shrinkage or ShrinkageConfig()
    out: dict[str, tuple[float, float]] = {}

    reach_mean = observations.reach.mean
    if reach_mean is not None:
        out["reach_mean_picks"] = (reach_mean, observations.reach.n)
    reach_stdev = observations.reach.stdev
    if reach_stdev is not None:
        out["reach_stdev_picks"] = (max(1.0, reach_stdev), observations.reach.n)
        # Predictability: tight reach distribution → predictable. Exponential so
        # it saturates smoothly instead of clipping at an arbitrary cutoff.
        out["predictability"] = (
            math.exp(-reach_stdev / max(1e-6, estimation.predictability_scale_picks)),
            observations.reach.n,
        )

    rank_gap = observations.rank_gap.mean
    if rank_gap is not None:
        out["rank_dependence"] = (
            math.exp(-rank_gap / max(1e-6, estimation.rank_delta_scale_picks)),
            observations.rank_gap.n,
        )

    fill = observations.fill_rate.mean
    if fill is not None:
        out["need_dependence"] = (
            _anchor_to_unit(fill, estimation.fill_rate_anchor),
            observations.fill_rate.n,
        )

    run = observations.run_continue_rate.mean
    if run is not None:
        out["run_chase"] = (
            _anchor_to_unit(run, estimation.run_continue_anchor),
            observations.run_continue_rate.n,
        )

    cliff = observations.tier_cliff_rate.mean
    if cliff is not None:
        out["tier_sensitivity"] = (
            _anchor_to_unit(cliff, estimation.tier_cliff_anchor),
            observations.tier_cliff_rate.n,
        )

    for key, stat in (
        ("rookie_rate", observations.rookie_rate),
        ("stack_rate", observations.stack_rate),
        ("handcuff_rate", observations.handcuff_rate),
    ):
        mean = stat.mean
        if mean is not None:
            out[key] = (mean, stat.n)

    if observations.team_share:
        top_share = max(observations.team_share.values())
        out["favorite_team_rate"] = (top_share, observations.weighted_picks)

    # First-QB / first-TE round: a per-season observation, so it shrinks on
    # seasons rather than picks.
    for key, position in (("first_qb_round", Position.QB), ("first_te_round", Position.TE)):
        stat = observations.first_round_by_position.get(position)
        mean = stat.mean if stat else None
        if mean is not None:
            # Convert the season count into a pick-equivalent sample so the
            # caller's shrinkage (which is in picks) yields the same weight the
            # season-based prior would.
            seasons = float(stat.n)
            equivalent = _pick_equivalent(
                shrinkage.season_weight(seasons), shrinkage.prior_strength
            )
            out[key] = (mean, equivalent)

    return out


def _anchor_to_unit(observed: float, anchor: float) -> float:
    """Map a rate onto [0, 1] such that ``observed == anchor`` gives 0.5.

    Below the anchor the mapping is linear to 0; above it, linear to 1. This
    keeps a league-average manager at the midpoint of the parameter's range no
    matter how common the underlying behaviour is in that league.
    """
    anchor = min(0.999, max(0.001, float(anchor)))
    value = min(1.0, max(0.0, float(observed)))
    if value <= anchor:
        return 0.5 * value / anchor
    return 0.5 + 0.5 * (value - anchor) / (1.0 - anchor)


def _pick_equivalent(weight: float, prior_strength: float) -> float:
    """Invert ``n / (n + prior)`` so a target weight becomes a pick count."""
    weight = min(0.999, max(0.0, float(weight)))
    if weight <= 0:
        return 0.0
    return float(prior_strength) * weight / (1.0 - weight)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback levels
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class FallbackLevels:
    """The three non-manager evidence levels a parameter can shrink toward."""

    league: dict[str, float] = field(default_factory=dict)
    platform: dict[str, float] = field(default_factory=dict)
    baseline: dict[str, float] = field(default_factory=dict)
    league_stats: HistoryStats = field(default_factory=HistoryStats)
    league_sample: float = 0.0
    platform_sample: float = 0.0

    def value(self, key: str) -> tuple[float, ProvenanceKind]:
        """The best available fallback for a parameter, and which level it is."""
        for source, provenance in (
            (self.league, ProvenanceKind.LEAGUE_FALLBACK),
            (self.platform, ProvenanceKind.PLATFORM_FALLBACK),
            (self.baseline, ProvenanceKind.BASELINE),
        ):
            if key in source:
                return float(source[key]), provenance
        return param_default(key), ProvenanceKind.BASELINE

    def blended(self, key: str, shrinkage: ShrinkageConfig) -> float:
        """Weighted mix of the fallback levels for one parameter."""
        league_w, platform_w, baseline_w = shrinkage.fallback_shares()
        default = param_default(key)
        parts: list[tuple[float, float]] = []
        if self.league_sample > 0 and key in self.league:
            parts.append((float(self.league[key]), league_w))
        if self.platform_sample > 0 and key in self.platform:
            parts.append((float(self.platform[key]), platform_w))
        parts.append((float(self.baseline.get(key, default)), baseline_w))
        total = sum(w for _, w in parts)
        if total <= 0:
            return default
        return sum(v * w for v, w in parts) / total


def build_fallback_levels(
    history: DraftHistory,
    *,
    settings: SimulationConfig | None = None,
    platform_history: Mapping[str, DraftHistory] | None = None,
    platform: str | None = None,
    archetype: Archetype = Archetype.BALANCED,
    reference_season: int | None = None,
) -> FallbackLevels:
    """Estimate league-wide, platform-wide and general-prior parameter levels.

    The *league* level pools every manager in the uploaded history; the
    *platform* level pools drafts from the same platform (useful when a user
    uploads both their home league and, say, public Sleeper drafts); the
    *baseline* level is the archetype prior, which always exists.
    """
    settings = settings or SimulationConfig()
    estimation = settings.estimation
    shrinkage = settings.shrinkage

    baseline = _archetype_parameter_map(archetype)
    levels = FallbackLevels(baseline=baseline)
    levels.league_stats = summarize_history(
        history, early_rounds=int(estimation.early_rounds)
    )

    league_pool = _pooled_observations(
        history, shrinkage=shrinkage, estimation=estimation,
        reference_season=reference_season,
    )
    if league_pool.has_data:
        levels.league = {
            key: value
            for key, (value, _) in estimate_parameters(
                league_pool, estimation, shrinkage=shrinkage
            ).items()
        }
        levels.league_sample = league_pool.weighted_picks

    if platform:
        subset = _platform_subset(history, platform, platform_history)
        if subset is not None:
            pool = _pooled_observations(
                subset, shrinkage=shrinkage, estimation=estimation,
                reference_season=reference_season,
            )
            if pool.has_data:
                levels.platform = {
                    key: value
                    for key, (value, _) in estimate_parameters(
                        pool, estimation, shrinkage=shrinkage
                    ).items()
                }
                levels.platform_sample = pool.weighted_picks
    return levels


def _platform_subset(
    history: DraftHistory,
    platform: str,
    platform_history: Mapping[str, DraftHistory] | None,
) -> DraftHistory | None:
    """Drafts matching ``platform``, from an explicit map or the main history."""
    if platform_history and platform in platform_history:
        return platform_history[platform]
    wanted = str(platform).strip().lower()
    drafts = [d for d in history.drafts if str(d.platform).strip().lower() == wanted]
    return DraftHistory(drafts) if drafts else None


def _pooled_observations(
    history: DraftHistory,
    *,
    shrinkage: ShrinkageConfig,
    estimation: ProfileEstimationConfig,
    reference_season: int | None = None,
) -> ManagerObservations:
    """Observations over every manager at once — the league/platform level."""
    pooled = ManagerObservations(manager_key="__pool__", manager_name="League average")
    for name in history.manager_names():
        single = observe_manager(
            name, history, shrinkage=shrinkage, estimation=estimation,
            reference_season=reference_season,
        )
        if not single.has_data:
            continue
        pooled.picks += single.picks
        pooled.weighted_picks += single.weighted_picks
        pooled.drafts = max(pooled.drafts, single.drafts)
        pooled.seasons = tuple(sorted(set(pooled.seasons) | set(single.seasons)))
        for attribute in (
            "reach", "rank_gap", "rank_inversions", "fill_rate",
            "run_continue_rate", "tier_cliff_rate", "rookie_rate",
            "stack_rate", "handcuff_rate",
        ):
            target: WeightedStat = getattr(pooled, attribute)
            source: WeightedStat = getattr(single, attribute)
            target.total += source.total
            target.square_total += source.square_total
            target.weight += source.weight
            target.count += source.count
        for position, stat in single.first_round_by_position.items():
            merged = pooled.first_round_by_position.setdefault(position, WeightedStat())
            merged.total += stat.total
            merged.square_total += stat.square_total
            merged.weight += stat.weight
            merged.count += stat.count
        # Favourite-team rate is a per-manager concept; pooling the top shares
        # would overstate it, so the league level deliberately omits it.
    return pooled


def _archetype_parameter_map(archetype: Archetype) -> dict[str, float]:
    """The archetype priors as a plain ``{param: value}`` map."""
    params = archetype_params(archetype)
    return {
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


# ─────────────────────────────────────────────────────────────────────────────
# Profile assembly
# ─────────────────────────────────────────────────────────────────────────────
def build_profile(
    manager: Manager,
    history: DraftHistory,
    *,
    settings: SimulationConfig | None = None,
    fallbacks: FallbackLevels | None = None,
    reference_season: int | None = None,
) -> ManagerProfile:
    """Build one manager's behavioural profile with full provenance.

    With no history at all this returns the archetype profile — the same object
    :meth:`ManagerProfile.from_archetype` produces — so the simulator always has
    a usable opponent.
    """
    settings = settings or SimulationConfig()
    shrinkage = settings.shrinkage
    estimation = settings.estimation

    observations = observe_manager(
        manager.name, history, shrinkage=shrinkage, estimation=estimation,
        reference_season=reference_season,
    )
    archetype = _resolve_archetype(manager, observations, estimation)
    if fallbacks is None:
        fallbacks = build_fallback_levels(
            history, settings=settings, archetype=archetype,
            reference_season=reference_season,
        )
    # The manager's own archetype is the last-resort prior, not whatever
    # archetype the league-wide fallback happened to be built with.
    fallbacks = FallbackLevels(
        league=dict(fallbacks.league),
        platform=dict(fallbacks.platform),
        baseline=_archetype_parameter_map(archetype),
        league_stats=fallbacks.league_stats,
        league_sample=fallbacks.league_sample,
        platform_sample=fallbacks.platform_sample,
    )

    profile = ManagerProfile(
        manager_key=manager.key,
        manager_name=manager.name,
        archetype=archetype,
        preferences=manager.preferences,
        sample_picks=float(observations.weighted_picks),
        sample_drafts=observations.drafts,
        seasons_seen=observations.seasons,
    )

    estimates = estimate_parameters(observations, estimation, shrinkage=shrinkage)
    user_values = _user_entered_values(manager.preferences)

    for key in PARAM_KEYS:
        estimate = estimates.get(key)
        fallback_value = fallbacks.blended(key, shrinkage)
        if estimate is None:
            _, provenance = fallbacks.value(key)
            value, weight, observed = fallback_value, 0.0, None
        else:
            observed, sample = estimate
            weight = shrinkage.manager_weight(sample)
            value = weight * observed + (1.0 - weight) * fallback_value
            provenance = (
                ProvenanceKind.OBSERVED
                if sample >= shrinkage.min_picks_for_observed
                else ProvenanceKind.MODEL_INFERRED
            )
        sample_size = estimate[1] if estimate else 0.0

        if key in user_values:
            value = _apply_user_value(value, user_values[key], estimation, key)
            provenance = ProvenanceKind.USER_ENTERED

        if key in UNIT_PARAMS:
            value = min(1.0, max(0.0, value))
        profile.set(
            key, value, provenance,
            sample_size=sample_size, manager_weight=weight, observed_value=observed,
        )

    _apply_positional_tendencies(profile, observations, fallbacks, settings)
    _apply_preference_adjustments(profile, manager.preferences, estimation)
    profile.favorite_teams = dict(observations.team_share)
    profile.repeat_players = {
        name: count for name, count in observations.repeat_players.items() if count > 1
    }
    profile.position_rate_by_round = {
        rnd: dict(rates) for rnd, rates in observations.position_rate_by_round.items()
    }
    LOGGER.debug(
        "Profile for %s: %.1f weighted picks, %d draft(s), archetype %s",
        manager.name, observations.weighted_picks, observations.drafts, archetype,
    )
    return profile


def _resolve_archetype(
    manager: Manager,
    observations: ManagerObservations,
    estimation: ProfileEstimationConfig,
) -> Archetype:
    """Pick the fallback archetype: user's choice first, else inferred, else set."""
    if manager.preferences.typical_strategy is not None:
        return manager.preferences.typical_strategy
    if observations.has_data:
        inferred = infer_archetype(observations, estimation)
        if inferred is not None:
            return inferred
    return manager.archetype


# Thresholds for archetype labelling. These describe what makes a tendency
# *noteworthy* rather than tuning the model — the profile's actual numbers come
# from the manager's picks either way, so a borderline label costs nothing.
ARCHETYPE_MIN_PICKS = 8
"""Below this many picks no label is assigned; the manager keeps their default."""
ZERO_RB_MAX_EARLY_RB_SHARE = 0.12
"""Early-round RB share at or below which a manager is avoiding RBs outright."""
ROBUST_RB_MIN_EARLY_RB_SHARE = 0.55
"""Early-round RB share at or above which more than half the early picks are RBs."""
HERO_RB_EARLY_RB_RANGE = (0.20, 0.40)
"""Early-round RB share consistent with taking exactly one early RB."""
HERO_RB_MIN_EARLY_WR_SHARE = 0.40
"""Hero-RB also requires the remaining early picks to lean WR."""
EARLY_QB_MAX_ROUND = 5.0
"""Average round of first QB at or before which the manager is an early-QB drafter."""
LATE_QB_MIN_ROUND = 10.0
ELITE_TE_MAX_ROUND = 4.0
ROOKIE_HEAVY_MIN_RATE = 0.28
HOMER_MIN_TEAM_SHARE = 0.25
"""Share of picks from one NFL team that reads as genuine team loyalty."""
HIGH_VARIANCE_MIN_INVERSIONS = 16.0
"""Better-ranked players left behind per pick, at or above which the manager is
routinely going a long way off the list.

Measured in inversions rather than reach standard deviation for the same reason as
the thresholds below: ``adp - pick`` spreads out as the draft gets deeper, so its
standard deviation is 15-25 picks for *every* manager in a 12-team league and no
absolute threshold on it separates anybody. Inversions are comparable across
league sizes and depths."""
AUTODRAFT_MAX_INVERSIONS = 1.5
"""Better-ranked players left on the board per pick, at or below which the manager
is taking whoever the list says is next.

Deliberately *not* expressed as a mean ``|rank - pick|`` gap: that quantity grows
with league size because it absorbs every other manager's reaching, so any fixed
threshold on it is unreachable outside the small league it was tuned on. Leaving
one or two better-ranked players behind per pick is list-following in any league —
not zero, because a roster has slots to fill and even an autodrafting platform
skips a position a team is already full at."""
RANK_FOLLOWER_MAX_INVERSIONS = 5.0
"""Looser than autodraft: broadly follows the list but makes real choices within it.

The ceiling matters less than the clearance below it. A manager with any positional
plan at all leaves 7-10 better-ranked players behind per pick in a 12-team league,
because their plan and the rankings disagree constantly, so anything under 5 is
someone who is essentially reading off the list."""


def infer_archetype(
    observations: ManagerObservations,
    estimation: ProfileEstimationConfig | None = None,
) -> Archetype | None:
    """Label a manager with the archetype their history most resembles.

    Only a *label* — the profile's numbers come from their actual picks. Returns
    ``None`` when the history is too thin or matches nothing distinctive, so the
    caller keeps whatever archetype was already assigned.
    """
    estimation = estimation or ProfileEstimationConfig()
    if observations.picks < ARCHETYPE_MIN_PICKS:
        return None

    early_rb = _early_share(observations, Position.RB, estimation)
    early_wr = _early_share(observations, Position.WR, estimation)
    qb_round = _first_round(observations, Position.QB)
    te_round = _first_round(observations, Position.TE)
    inversions = observations.rank_inversions.mean
    rookie = observations.rookie_rate.mean or 0.0
    top_team = max(observations.team_share.values(), default=0.0)

    # Positional-strategy labels come first: they describe *what* a manager
    # drafts, which is more informative than *how tightly* they track a list, and
    # a rank-follower who also happens to load up on RBs is better described by
    # the strategy. Rank-based labels are the fallback for managers with no
    # distinctive positional signature.
    if early_rb is not None and early_rb <= ZERO_RB_MAX_EARLY_RB_SHARE:
        return Archetype.ZERO_RB
    if early_rb is not None and early_rb >= ROBUST_RB_MIN_EARLY_RB_SHARE:
        return Archetype.ROBUST_RB
    if qb_round is not None and qb_round <= EARLY_QB_MAX_ROUND:
        return Archetype.EARLY_QB
    if te_round is not None and te_round <= ELITE_TE_MAX_ROUND:
        return Archetype.ELITE_TE
    if rookie >= ROOKIE_HEAVY_MIN_RATE:
        return Archetype.ROOKIE_HEAVY
    if top_team >= HOMER_MIN_TEAM_SHARE:
        return Archetype.HOMER
    if (
        early_rb is not None and early_wr is not None
        and HERO_RB_EARLY_RB_RANGE[0] <= early_rb <= HERO_RB_EARLY_RB_RANGE[1]
        and early_wr >= HERO_RB_MIN_EARLY_WR_SHARE
    ):
        return Archetype.HERO_RB
    if qb_round is not None and qb_round >= LATE_QB_MIN_ROUND:
        return Archetype.LATE_QB
    if inversions is not None and inversions >= HIGH_VARIANCE_MIN_INVERSIONS:
        return Archetype.HIGH_VARIANCE
    if inversions is not None and inversions <= AUTODRAFT_MAX_INVERSIONS:
        return Archetype.AUTODRAFT
    if inversions is not None and inversions <= RANK_FOLLOWER_MAX_INVERSIONS:
        return Archetype.RANK_FOLLOWER
    return Archetype.BALANCED


def _early_share(
    observations: ManagerObservations,
    position: Position,
    estimation: ProfileEstimationConfig | None = None,
) -> float | None:
    """Share of this manager's early-round picks spent on ``position``.

    Pick-weighted, not an average of per-round rates: a round in which they made
    one pick must not count as much as a round in which they made five.
    """
    if observations.early_picks <= 0:
        return None
    return observations.early_position_share.get(position, 0.0)


def _first_round(observations: ManagerObservations, position: Position) -> float | None:
    stat = observations.first_round_by_position.get(position)
    return stat.mean if stat else None


def _apply_positional_tendencies(
    profile: ManagerProfile,
    observations: ManagerObservations,
    fallbacks: FallbackLevels,
    settings: SimulationConfig,
) -> None:
    """Derive per-position utility bias from timing and share vs the league.

    Two independent signals are combined: *timing* (this manager takes the
    position earlier or later than the league does) and *share* (they spend more
    or fewer picks on it). Both are expressed relative to the league so the bias
    means "compared to the people you actually draft against".
    """
    estimation = settings.estimation
    shrinkage = settings.shrinkage
    stats = fallbacks.league_stats
    archetype_bias = dict(archetype_params(profile.archetype).position_bias)
    archetype_early = dict(archetype_params(profile.archetype).early_round_position_bias)

    weight = shrinkage.manager_weight(observations.weighted_picks)
    bias: dict[Position, float] = {}

    for position in Position:
        league_mean = (stats.mean_pick_by_position or {}).get(position)
        league_share = (stats.share_by_position or {}).get(position)
        signals: list[float] = []

        manager_stat = observations.mean_pick_by_position.get(position)
        manager_mean = manager_stat.mean if manager_stat else None
        if manager_mean is not None and league_mean:
            # Earlier than the league (smaller pick number) → positive bias.
            timing = (float(league_mean) - float(manager_mean)) / float(league_mean)
            signals.append(timing * float(estimation.position_bias_timing_scale))

        manager_share = observations.position_share.get(position)
        if manager_share is not None and league_share:
            share_lean = (manager_share - league_share) / league_share
            signals.append(share_lean * float(estimation.position_bias_share_scale))

        prior = float(archetype_bias.get(position, 0.0))
        if not signals:
            bias[position] = prior
            continue
        observed = sum(signals) / len(signals)
        blended = weight * observed + (1.0 - weight) * prior
        clip = float(estimation.position_bias_clip)
        bias[position] = float(min(clip, max(-clip, blended)))

    profile.position_bias = {p: v for p, v in bias.items() if abs(v) > 1e-9}

    # Early-round bias: the same comparison restricted to the opening rounds, and
    # shrunk on the early-pick count rather than the full sample — three seasons
    # of a 16-round draft give only nine early picks to learn from.
    early_bias: dict[Position, float] = {}
    early_rounds = int(estimation.early_rounds)
    early_weight = shrinkage.manager_weight(observations.early_picks)
    for position in Position:
        manager_early = _early_share(observations, position, estimation)
        league_early = _league_early_share(stats, position, early_rounds)
        prior = float(archetype_early.get(position, 0.0))
        if manager_early is None or not league_early:
            if abs(prior) > 1e-9:
                early_bias[position] = prior
            continue
        lean = (manager_early - league_early) / max(1e-6, league_early)
        observed = lean * float(estimation.position_bias_share_scale)
        blended = early_weight * observed + (1.0 - early_weight) * prior
        clip = float(estimation.position_bias_clip) * 2.0
        value = float(min(clip, max(-clip, blended)))
        if abs(value) > 1e-9:
            early_bias[position] = value
    profile.early_round_position_bias = early_bias


def _league_early_share(
    stats: HistoryStats, position: Position, early_rounds: int
) -> float | None:
    """League share of early-round picks at a position.

    ``None`` when the pooled stats were built for a different early-round window
    than the one being asked about, so the caller falls back to the prior instead
    of comparing against a mismatched denominator.
    """
    if not stats.early_share_by_position or stats.early_rounds != int(early_rounds):
        return None
    return stats.early_share_by_position.get(position, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# User-entered preferences
# ─────────────────────────────────────────────────────────────────────────────
_PREFERENCE_TO_PARAM: dict[str, str] = {
    "risk_tolerance": "risk_preference",
    "rookie_preference": "rookie_rate",
    "stack_preference": "stack_rate",
    "handcuff_preference": "handcuff_rate",
    "rank_reliance": "rank_dependence",
    "predictability": "predictability",
}


def _user_entered_values(preferences: ManagerPreferences) -> dict[str, float]:
    """Parameters the user typed a value for, keyed by parameter name."""
    out: dict[str, float] = {}
    for attribute, key in _PREFERENCE_TO_PARAM.items():
        value = getattr(preferences, attribute, None)
        if value is not None:
            out[key] = float(value)
    return out


def _apply_user_value(
    modelled: float,
    stated: float,
    estimation: ProfileEstimationConfig,
    key: str,
) -> float:
    """Pull a modelled value toward what the user said, without erasing evidence.

    The spec treats manual entry as strong evidence rather than gospel: a user
    who says "he loves rookies" should move the parameter a long way, but sixty
    observed picks should not be thrown away entirely.
    """
    pull = min(1.0, max(0.0, float(estimation.user_preference_weight)))
    return (1.0 - pull) * float(modelled) + pull * float(stated)


def _apply_preference_adjustments(
    profile: ManagerProfile,
    preferences: ManagerPreferences,
    estimation: ProfileEstimationConfig,
) -> None:
    """Fold non-numeric user statements into the profile.

    Preferred / avoided positions become positional bias; a stated favourite team
    raises the favourite-team rate to at least the threshold that makes it
    visible; experience level nudges predictability when nothing was observed.
    """
    pull = float(estimation.user_preference_weight)
    clip = float(estimation.position_bias_clip)
    for position in preferences.preferred_positions:
        current = profile.position_bias.get(position, 0.0)
        profile.position_bias[position] = min(clip, current + pull * clip)
    for position in preferences.avoided_positions:
        current = profile.position_bias.get(position, 0.0)
        profile.position_bias[position] = max(-clip, current - pull * clip)

    if preferences.favorite_nfl_team:
        team = str(preferences.favorite_nfl_team).upper()
        floor = float(estimation.favorite_team_min_share)
        current = profile.get("favorite_team_rate")
        if current < floor:
            profile.set(
                "favorite_team_rate", floor, ProvenanceKind.USER_ENTERED,
                sample_size=0.0, manager_weight=0.0, observed_value=current,
            )
        profile.favorite_teams.setdefault(team, floor)

    if preferences.predictability is None and profile.provenance("predictability") in (
        ProvenanceKind.BASELINE, ProvenanceKind.LEAGUE_FALLBACK,
        ProvenanceKind.PLATFORM_FALLBACK,
    ):
        # No observation and no explicit number: experience level is the only
        # signal available, so use it rather than leaving the prior untouched.
        entry = profile.values.get("predictability")
        blended = 0.5 * (profile.get("predictability") + preferences.experience_confidence())
        profile.set(
            "predictability", blended, ProvenanceKind.USER_ENTERED,
            sample_size=entry.sample_size if entry else 0.0,
            manager_weight=entry.manager_weight if entry else 0.0,
            observed_value=entry.observed_value if entry else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Whole-league entry point
# ─────────────────────────────────────────────────────────────────────────────
def build_profiles(
    league: League,
    history: DraftHistory | None = None,
    *,
    settings: SimulationConfig | None = None,
    annotate: bool = True,
    pool: object | None = None,
    reference_season: int | None = None,
) -> dict[int, ManagerProfile]:
    """Build a profile for every manager in ``league``, keyed by draft slot.

    Set ``annotate=False`` when the history has already been through
    :func:`engine.features.annotate_history` — annotation is idempotent but not
    free.
    """
    settings = settings or SimulationConfig()
    history = history or DraftHistory()
    if annotate and history.drafts:
        annotate_history(
            history, pool=pool, roster=league.config.roster,
            config=settings.estimation,
        )

    shared = build_fallback_levels(
        history, settings=settings, platform=str(league.config.platform),
        reference_season=reference_season,
    )
    profiles: dict[int, ManagerProfile] = {}
    for manager in league.managers:
        profiles[manager.draft_slot] = build_profile(
            manager, history, settings=settings, fallbacks=shared,
            reference_season=reference_season,
        )
    matched = sum(1 for p in profiles.values() if p.sample_picks > 0)
    LOGGER.info(
        "Built %d manager profile(s); %d matched to draft history",
        len(profiles), matched,
    )
    return profiles


def league_average_profile(
    history: DraftHistory,
    *,
    settings: SimulationConfig | None = None,
    name: str = "League average",
) -> ManagerProfile:
    """A profile representing the league as a whole, for side-by-side comparison."""
    settings = settings or SimulationConfig()
    pooled = _pooled_observations(
        history, shrinkage=settings.shrinkage, estimation=settings.estimation
    )
    profile = baseline_profile(name)
    profile.manager_name = name
    if not pooled.has_data:
        return profile
    profile.sample_picks = pooled.weighted_picks
    profile.sample_drafts = pooled.drafts
    profile.seasons_seen = pooled.seasons
    estimates = estimate_parameters(
        pooled, settings.estimation, shrinkage=settings.shrinkage
    )
    for key, (value, sample) in estimates.items():
        profile.set(
            key, value, ProvenanceKind.LEAGUE_FALLBACK,
            sample_size=sample, manager_weight=1.0, observed_value=value,
        )
    stats = summarize_history(history)
    profile.position_bias = {}
    profile.notes = (
        f"Pooled across {pooled.picks} picks from {stats.manager_count} manager(s)."
    )
    return profile


def profiles_frame(profiles: Mapping[int, ManagerProfile]):
    """Tabulate profiles for the comparison view."""
    import pandas as pd

    rows = []
    for slot, profile in sorted(profiles.items()):
        row: dict[str, object] = {
            "draft_slot": slot,
            "manager": profile.manager_name,
            "archetype": str(profile.archetype),
            "sample_picks": round(profile.sample_picks, 1),
            "confidence": round(profile.confidence, 3),
        }
        for key in PARAM_KEYS:
            row[key] = round(profile.get(key), 3)
            row[f"{key}__provenance"] = str(profile.provenance(key))
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "WeightedStat", "ManagerObservations", "FallbackLevels",
    "observe_manager", "estimate_parameters", "build_fallback_levels",
    "build_profile", "build_profiles", "infer_archetype",
    "league_average_profile", "profiles_frame",
]
