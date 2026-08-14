"""Tests for the opponent model: estimation, shrinkage, and archetype labelling.

The assertions here are written against the *designed* behaviour of the
synthetic managers in ``conftest.py`` — Zed drafts zero-RB, Rob drafts robust-RB,
Qui takes a quarterback early, Auto sits on ADP — so a passing test means the
model recovered a strategy it was never told about.
"""

from __future__ import annotations

import pytest

from core.config import ProfileEstimationConfig, ShrinkageConfig, SimulationConfig
from core.enums import Archetype, Position, ProvenanceKind
from engine.features import annotate_history
from engine.opponent_model import (
    ARCHETYPE_MIN_PICKS,
    FallbackLevels,
    _archetype_parameter_map,
    build_fallback_levels,
    build_profile,
    build_profiles,
    estimate_parameters,
    infer_archetype,
    league_average_profile,
    observe_manager,
    profiles_frame,
)
from models.draft import DraftHistory
from models.league import League
from models.manager import ManagerPreferences, Manager


@pytest.fixture
def annotated(synthetic_history: DraftHistory) -> DraftHistory:
    annotate_history(synthetic_history)
    return synthetic_history


@pytest.fixture
def profiles(
    annotated: DraftHistory, synthetic_league: League, settings: SimulationConfig
) -> dict[int, object]:
    return build_profiles(
        synthetic_league, annotated, settings=settings, annotate=False
    )


def _by_name(profiles: dict[int, object]) -> dict[str, object]:
    return {p.manager_name: p for p in profiles.values()}


class TestObserveManager:
    def test_observations_are_recorded_for_a_known_manager(
        self, annotated: DraftHistory, settings: SimulationConfig
    ) -> None:
        obs = observe_manager(
            "Zed Zero", annotated,
            shrinkage=settings.shrinkage, estimation=settings.estimation,
        )
        assert obs.has_data
        assert obs.drafts == 3
        assert obs.picks == 30  # 10 rounds x 3 seasons

    def test_unknown_manager_has_no_data(
        self, annotated: DraftHistory, settings: SimulationConfig
    ) -> None:
        obs = observe_manager(
            "Nobody At All", annotated,
            shrinkage=settings.shrinkage, estimation=settings.estimation,
        )
        assert not obs.has_data

    def test_recency_decay_weights_recent_seasons_higher(
        self, annotated: DraftHistory
    ) -> None:
        """Weighted picks must fall short of the raw count once decay applies."""
        settings = SimulationConfig()
        settings.shrinkage = ShrinkageConfig(recency_half_life_seasons=1.0)
        obs = observe_manager(
            "Zed Zero", annotated,
            shrinkage=settings.shrinkage, estimation=settings.estimation,
        )
        assert obs.picks == 30
        assert obs.weighted_picks < 30

    def test_reference_season_excludes_itself_and_later(
        self, annotated: DraftHistory, settings: SimulationConfig
    ) -> None:
        """No look-ahead: the season being predicted is not evidence about itself."""
        obs = observe_manager(
            "Zed Zero", annotated,
            shrinkage=settings.shrinkage, estimation=settings.estimation,
            reference_season=2024,
        )
        assert obs.seasons == (2023,)
        assert obs.drafts == 1
        assert obs.picks == 10  # one 10-round season

    def test_reference_season_before_all_history_yields_nothing(
        self, annotated: DraftHistory, settings: SimulationConfig
    ) -> None:
        obs = observe_manager(
            "Zed Zero", annotated,
            shrinkage=settings.shrinkage, estimation=settings.estimation,
            reference_season=2023,
        )
        assert not obs.has_data

    def test_early_picks_are_counted(
        self, annotated: DraftHistory, settings: SimulationConfig
    ) -> None:
        obs = observe_manager(
            "Zed Zero", annotated,
            shrinkage=settings.shrinkage, estimation=settings.estimation,
        )
        assert obs.early_picks > 0
        assert pytest.approx(sum(obs.early_position_share.values()), abs=1e-6) == 1.0


class TestEstimateParameters:
    def test_reach_mean_recovers_the_designed_reach(
        self, annotated: DraftHistory, settings: SimulationConfig
    ) -> None:
        """Qui reaches ~6 picks; Auto drafts on ADP. The estimate must order them."""
        estimates = {}
        for name in ("Qui Quarterback", "Auto Pilot"):
            obs = observe_manager(
                name, annotated,
                shrinkage=settings.shrinkage, estimation=settings.estimation,
            )
            estimates[name] = estimate_parameters(
                obs, settings.estimation, shrinkage=settings.shrinkage
            )
        qui = estimates["Qui Quarterback"]["reach_mean_picks"][0]
        auto = estimates["Auto Pilot"]["reach_mean_picks"][0]
        assert qui > auto
        assert qui > 0  # positive = reaches ahead of ADP
        assert abs(auto) < 2.0

    def test_predictability_is_highest_for_the_adp_follower(
        self, annotated: DraftHistory, settings: SimulationConfig
    ) -> None:
        values = {}
        for name in ("Zed Zero", "Rob Robust", "Qui Quarterback", "Auto Pilot"):
            obs = observe_manager(
                name, annotated,
                shrinkage=settings.shrinkage, estimation=settings.estimation,
            )
            params = estimate_parameters(
                obs, settings.estimation, shrinkage=settings.shrinkage
            )
            values[name] = params["predictability"][0]
        assert values["Auto Pilot"] == max(values.values())

    def test_unsupported_parameters_are_absent_rather_than_invented(
        self, annotated: DraftHistory, settings: SimulationConfig
    ) -> None:
        """A manager with no observations must yield no estimates at all."""
        obs = observe_manager(
            "Nobody At All", annotated,
            shrinkage=settings.shrinkage, estimation=settings.estimation,
        )
        assert estimate_parameters(
            obs, settings.estimation, shrinkage=settings.shrinkage
        ) == {}

    def test_every_estimate_carries_a_sample_size(
        self, annotated: DraftHistory, settings: SimulationConfig
    ) -> None:
        obs = observe_manager(
            "Rob Robust", annotated,
            shrinkage=settings.shrinkage, estimation=settings.estimation,
        )
        estimates = estimate_parameters(
            obs, settings.estimation, shrinkage=settings.shrinkage
        )
        assert estimates
        for key, (value, sample) in estimates.items():
            assert isinstance(value, float), key
            assert sample > 0, key


class TestShrinkage:
    def test_thin_history_stays_near_the_prior(
        self, annotated: DraftHistory, synthetic_league: League
    ) -> None:
        """With a huge prior strength, observations barely move the parameter.

        The target is the *blended* fallback with the manager's own archetype as
        the baseline level, which is what ``build_profile`` shrinks toward.
        """
        settings = SimulationConfig()
        settings.shrinkage = ShrinkageConfig(prior_strength=10_000.0)
        fallbacks = build_fallback_levels(annotated, settings=settings)
        manager = synthetic_league.managers[0]
        profile = build_profile(
            manager, annotated, settings=settings, fallbacks=fallbacks
        )
        expected = FallbackLevels(
            league=dict(fallbacks.league),
            platform=dict(fallbacks.platform),
            baseline=_archetype_parameter_map(profile.archetype),
            league_stats=fallbacks.league_stats,
            league_sample=fallbacks.league_sample,
            platform_sample=fallbacks.platform_sample,
        ).blended("rank_dependence", settings.shrinkage)
        assert abs(profile.get("rank_dependence") - expected) < 0.05

    def test_rich_history_follows_the_observation(
        self, annotated: DraftHistory, synthetic_league: League
    ) -> None:
        settings = SimulationConfig()
        settings.shrinkage = ShrinkageConfig(prior_strength=0.01)
        fallbacks = build_fallback_levels(annotated, settings=settings)
        manager = next(m for m in synthetic_league.managers if m.name == "Auto Pilot")
        profile = build_profile(
            manager, annotated, settings=settings, fallbacks=fallbacks
        )
        obs = observe_manager(
            "Auto Pilot", annotated,
            shrinkage=settings.shrinkage, estimation=settings.estimation,
        )
        observed = estimate_parameters(
            obs, settings.estimation, shrinkage=settings.shrinkage
        )["rank_dependence"][0]
        assert abs(profile.get("rank_dependence") - observed) < 0.05

    def test_manager_weight_rises_with_sample_size(self) -> None:
        shrinkage = ShrinkageConfig(prior_strength=10.0)
        assert shrinkage.manager_weight(0) == 0.0
        assert shrinkage.manager_weight(10) == pytest.approx(0.5)
        assert shrinkage.manager_weight(90) == pytest.approx(0.9)

    def test_season_weight_uses_the_season_prior(self) -> None:
        shrinkage = ShrinkageConfig(season_prior_strength=2.0)
        assert shrinkage.season_weight(0) == 0.0
        assert shrinkage.season_weight(2) == pytest.approx(0.5)

    def test_cluster_weight_rises_with_the_number_of_drafts(self) -> None:
        shrinkage = ShrinkageConfig(draft_prior_strength=1.0)
        assert shrinkage.cluster_weight(0) == 0.0
        assert shrinkage.cluster_weight(1) == pytest.approx(0.5)
        assert shrinkage.cluster_weight(2) == pytest.approx(2 / 3)
        assert shrinkage.cluster_weight(3) == pytest.approx(0.75)

    def test_one_draft_of_history_counts_for_less_than_three(
        self, annotated: DraftHistory, synthetic_league: League
    ) -> None:
        """Sixteen picks from one August are not sixteen independent observations.

        A brand-new league has exactly one draft on record, and treating its picks as
        an independent sample each had the model describing settled personalities off
        a single afternoon. The picks should still count — just for less.
        """
        settings = SimulationConfig()
        manager = next(m for m in synthetic_league.managers if m.name == "Auto Pilot")
        newest = max(d.season for d in annotated.drafts)
        thin = DraftHistory(
            drafts=[d for d in annotated.drafts if d.season == newest]
        )
        thin_profile = build_profile(manager, thin, settings=settings)
        full_profile = build_profile(manager, annotated, settings=settings)

        raw = estimate_parameters(
            observe_manager(
                manager.name, thin,
                shrinkage=settings.shrinkage, estimation=settings.estimation,
            ),
            settings.estimation, shrinkage=settings.shrinkage,
        )["rank_dependence"][1]
        # One draft, so the effective sample is half the picks observed…
        assert thin_profile.values["rank_dependence"].sample_size == pytest.approx(
            raw * 0.5
        )
        # …and three drafts of the same manager are believed more than one.
        assert (
            full_profile.values["rank_dependence"].manager_weight
            > thin_profile.values["rank_dependence"].manager_weight
        )

    def test_a_per_season_metric_is_not_discounted_twice(
        self, annotated: DraftHistory, synthetic_league: League
    ) -> None:
        """``first_qb_round`` already shrinks on seasons, so it skips the draft prior."""
        settings = SimulationConfig()
        manager = next(m for m in synthetic_league.managers if m.name == "Auto Pilot")
        profile = build_profile(manager, annotated, settings=settings)
        expected = estimate_parameters(
            observe_manager(
                manager.name, annotated,
                shrinkage=settings.shrinkage, estimation=settings.estimation,
            ),
            settings.estimation, shrinkage=settings.shrinkage,
        )["first_qb_round"][1]
        assert profile.values["first_qb_round"].sample_size == pytest.approx(expected)


class TestProvenance:
    def test_observed_parameters_are_marked_observed(
        self, profiles: dict[int, object]
    ) -> None:
        profile = _by_name(profiles)["Zed Zero"]
        assert profile.provenance("rank_dependence") is ProvenanceKind.OBSERVED

    def test_unmatched_manager_falls_back(
        self, annotated: DraftHistory, settings: SimulationConfig
    ) -> None:
        stranger = Manager(name="Brand New Guy", draft_slot=9)
        fallbacks = build_fallback_levels(annotated, settings=settings)
        profile = build_profile(
            stranger, annotated, settings=settings, fallbacks=fallbacks
        )
        assert profile.sample_picks == 0
        assert profile.provenance("rank_dependence") is not ProvenanceKind.OBSERVED

    def test_user_entered_preferences_are_marked(
        self, annotated: DraftHistory, settings: SimulationConfig
    ) -> None:
        manager = Manager(
            name="Zed Zero", draft_slot=1,
            preferences=ManagerPreferences(rookie_preference=0.95),
        )
        fallbacks = build_fallback_levels(annotated, settings=settings)
        profile = build_profile(
            manager, annotated, settings=settings, fallbacks=fallbacks
        )
        assert profile.provenance("rookie_rate") is ProvenanceKind.USER_ENTERED
        assert profile.get("rookie_rate") > 0.3

    def test_provenance_summary_covers_every_parameter(
        self, profiles: dict[int, object]
    ) -> None:
        profile = _by_name(profiles)["Rob Robust"]
        summary = profile.provenance_summary()
        assert sum(summary.values()) > 0


class TestArchetypeInference:
    """The label must reflect the plan each synthetic manager was given."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Zed Zero", Archetype.ZERO_RB),
            ("Rob Robust", Archetype.ROBUST_RB),
            ("Qui Quarterback", Archetype.EARLY_QB),
        ],
    )
    def test_positional_strategies_are_recovered(
        self,
        annotated: DraftHistory,
        settings: SimulationConfig,
        name: str,
        expected: Archetype,
    ) -> None:
        obs = observe_manager(
            name, annotated,
            shrinkage=settings.shrinkage, estimation=settings.estimation,
        )
        assert infer_archetype(obs, settings.estimation) is expected

    def test_a_manager_with_no_positional_signature_gets_a_rank_label(
        self, annotated: DraftHistory, settings: SimulationConfig
    ) -> None:
        """Auto Pilot spreads picks evenly, so the *rank* tests must be reached.

        Regression: the rank tests used to run first and swallowed every manager;
        now they are the fallback, and this asserts they are still reachable.
        """
        obs = observe_manager(
            "Auto Pilot", annotated,
            shrinkage=settings.shrinkage, estimation=settings.estimation,
        )
        assert infer_archetype(obs, settings.estimation) in {
            Archetype.AUTODRAFT,
            Archetype.RANK_FOLLOWER,
            Archetype.BALANCED,
        }

    def test_too_few_picks_stays_balanced(
        self, annotated: DraftHistory, settings: SimulationConfig
    ) -> None:
        """Below the minimum sample the model must not guess a strategy."""
        thin = annotated.before_season(2024)
        obs = observe_manager(
            "Zed Zero", thin,
            shrinkage=settings.shrinkage, estimation=settings.estimation,
        )
        if obs.weighted_picks < ARCHETYPE_MIN_PICKS:
            assert infer_archetype(obs, settings.estimation) is Archetype.BALANCED


class TestPositionalBias:
    def test_early_bias_differs_between_managers(
        self, profiles: dict[int, object]
    ) -> None:
        """Regression: every manager once received an identical early bias."""
        named = _by_name(profiles)
        zed = named["Zed Zero"].early_round_position_bias
        rob = named["Rob Robust"].early_round_position_bias
        assert zed != rob
        assert zed.get(Position.RB, 0.0) < 0.0   # zero-RB avoids backs early
        assert rob.get(Position.RB, 0.0) > 0.0   # robust-RB loads up on them

    def test_early_qb_manager_has_positive_qb_bias(
        self, profiles: dict[int, object]
    ) -> None:
        qui = _by_name(profiles)["Qui Quarterback"]
        assert qui.early_round_position_bias.get(Position.QB, 0.0) > 0.0

    def test_bias_is_clipped_to_the_configured_bound(
        self, annotated: DraftHistory, synthetic_league: League
    ) -> None:
        settings = SimulationConfig()
        settings.estimation = ProfileEstimationConfig(position_bias_clip=0.10)
        built = build_profiles(
            synthetic_league, annotated, settings=settings, annotate=False
        )
        for profile in built.values():
            for value in profile.position_bias.values():
                assert abs(value) <= 0.10 + 1e-9

    def test_first_qb_round_is_earlier_for_the_early_qb_manager(
        self, profiles: dict[int, object]
    ) -> None:
        named = _by_name(profiles)
        assert named["Qui Quarterback"].get("first_qb_round") < named["Zed Zero"].get(
            "first_qb_round"
        )


class TestPreferenceAdjustments:
    def test_preferred_position_raises_its_bias(
        self, annotated: DraftHistory, settings: SimulationConfig
    ) -> None:
        plain = Manager(name="Rob Robust", draft_slot=2)
        keen = Manager(
            name="Rob Robust", draft_slot=2,
            preferences=ManagerPreferences(preferred_positions=[Position.TE]),
        )
        fallbacks = build_fallback_levels(annotated, settings=settings)
        base = build_profile(plain, annotated, settings=settings, fallbacks=fallbacks)
        tweaked = build_profile(keen, annotated, settings=settings, fallbacks=fallbacks)
        assert tweaked.position_bias.get(Position.TE, 0.0) > base.position_bias.get(
            Position.TE, 0.0
        )

    def test_favorite_team_raises_the_homer_rate(
        self, annotated: DraftHistory, settings: SimulationConfig
    ) -> None:
        manager = Manager(
            name="Rob Robust", draft_slot=2,
            preferences=ManagerPreferences(favorite_nfl_team="GB"),
        )
        fallbacks = build_fallback_levels(annotated, settings=settings)
        profile = build_profile(
            manager, annotated, settings=settings, fallbacks=fallbacks
        )
        assert profile.get("favorite_team_rate") >= settings.estimation.favorite_team_min_share

    def test_user_entry_does_not_erase_observations(
        self, annotated: DraftHistory, settings: SimulationConfig
    ) -> None:
        """Manual entry is strong evidence, not gospel: it pulls, not overwrites."""
        manager = Manager(
            name="Auto Pilot", draft_slot=4,
            preferences=ManagerPreferences(rank_reliance=0.0),
        )
        fallbacks = build_fallback_levels(annotated, settings=settings)
        profile = build_profile(
            manager, annotated, settings=settings, fallbacks=fallbacks
        )
        # Auto Pilot's picks say "high rank dependence"; the user says zero. The
        # result must sit strictly between the two.
        assert 0.0 < profile.get("rank_dependence") < 0.85


class TestBuildProfiles:
    def test_one_profile_per_draft_slot(
        self, profiles: dict[int, object], synthetic_league: League
    ) -> None:
        assert set(profiles) == {m.draft_slot for m in synthetic_league.managers}

    def test_all_managers_matched_to_history(self, profiles: dict[int, object]) -> None:
        assert all(p.sample_picks > 0 for p in profiles.values())

    def test_unit_parameters_stay_in_range(self, profiles: dict[int, object]) -> None:
        from models.manager import UNIT_PARAMS

        for profile in profiles.values():
            for key in UNIT_PARAMS:
                assert 0.0 <= profile.get(key) <= 1.0, key

    def test_profiles_are_reproducible(
        self, annotated: DraftHistory, synthetic_league: League, settings: SimulationConfig
    ) -> None:
        first = build_profiles(
            synthetic_league, annotated, settings=settings, annotate=False
        )
        second = build_profiles(
            synthetic_league, annotated, settings=settings, annotate=False
        )
        for slot, profile in first.items():
            for key in profile.values:
                assert profile.get(key) == second[slot].get(key)

    def test_empty_history_still_produces_profiles(
        self, synthetic_league: League, settings: SimulationConfig
    ) -> None:
        built = build_profiles(
            synthetic_league, DraftHistory(), settings=settings
        )
        assert len(built) == len(synthetic_league.managers)
        assert all(p.sample_picks == 0 for p in built.values())

    def test_profiles_frame_has_a_row_per_manager(
        self, profiles: dict[int, object]
    ) -> None:
        frame = profiles_frame(profiles)
        assert len(frame) == len(profiles)


class TestLeagueAverageProfile:
    def test_pools_the_whole_league(
        self, annotated: DraftHistory, settings: SimulationConfig
    ) -> None:
        profile = league_average_profile(annotated, settings=settings)
        assert profile.sample_picks > 0
        assert profile.provenance("rank_dependence") is ProvenanceKind.LEAGUE_FALLBACK

    def test_empty_history_returns_the_baseline(
        self, settings: SimulationConfig
    ) -> None:
        profile = league_average_profile(DraftHistory(), settings=settings)
        assert profile.sample_picks == 0
