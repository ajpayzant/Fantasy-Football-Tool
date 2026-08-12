"""Shared fixtures.

The synthetic history built here is the workhorse of the opponent-model tests.
Each manager drafts to a *deliberately distinctive plan*, so a test can assert
that the model recovered the behaviour that was designed in. That is the only
way to test an estimator: against data whose right answer is known by
construction rather than by running the estimator and blessing its output.
"""

from __future__ import annotations

import pytest

from core.config import LeagueConfig, SimulationConfig
from core.enums import Archetype, Platform, Position
from models.draft import DraftHistory, HistoricalDraft, HistoricalPick
from models.league import League
from models.manager import Manager

SEASONS: tuple[int, ...] = (2023, 2024, 2025)
TEAM_COUNT: int = 4
ROUNDS: int = 10

# Each manager's positional script, round 1 → round 10. A manager may have one
# script per season; shorter lists are cycled. The scripts are exaggerated
# versions of real strategies so the signal is unambiguous at the small sample
# sizes these tests use.
PLANS: dict[str, tuple[list[Position], ...]] = {
    # Zero-RB: no running back until the middle rounds, receivers loaded early.
    "Zed Zero": (
        [
            Position.WR, Position.WR, Position.WR, Position.TE, Position.RB,
            Position.QB, Position.RB, Position.WR, Position.K, Position.DST,
        ],
    ),
    # Robust-RB: backs early and often.
    "Rob Robust": (
        [
            Position.RB, Position.RB, Position.RB, Position.WR, Position.WR,
            Position.QB, Position.TE, Position.RB, Position.K, Position.DST,
        ],
    ),
    # Early-QB: quarterback in the opening rounds, well before the league does.
    "Qui Quarterback": (
        [
            Position.WR, Position.QB, Position.RB, Position.RB, Position.WR,
            Position.TE, Position.WR, Position.RB, Position.K, Position.DST,
        ],
    ),
    # Autodraft-like: deliberately *no* positional signature. Whoever ADP says is
    # next, which means the opening rounds shuffle from season to season instead
    # of following a script. Quarterback and tight end always land in the middle
    # rounds, so no positional-strategy label can fire and the rank-based labels
    # — which this manager exists to exercise — are reachable.
    "Auto Pilot": (
        [
            Position.WR, Position.RB, Position.WR, Position.RB, Position.TE,
            Position.QB, Position.WR, Position.RB, Position.K, Position.DST,
        ],
        [
            Position.RB, Position.WR, Position.RB, Position.WR, Position.TE,
            Position.QB, Position.WR, Position.RB, Position.K, Position.DST,
        ],
        [
            Position.WR, Position.WR, Position.RB, Position.RB, Position.TE,
            Position.QB, Position.WR, Position.RB, Position.K, Position.DST,
        ],
    ),
}

# How far from ADP each manager drafts, and how consistently. "Auto Pilot" sits
# exactly on ADP with almost no variance, which is what makes it autodraft-like.
REACH_PROFILE: dict[str, tuple[float, float]] = {
    "Zed Zero": (4.0, 7.0),
    "Rob Robust": (2.0, 5.0),
    "Qui Quarterback": (6.0, 9.0),
    "Auto Pilot": (0.0, 1.0),
}

NFL_TEAMS_BY_POSITION: dict[Position, tuple[str, ...]] = {
    Position.QB: ("KC", "BUF", "CIN", "PHI"),
    Position.RB: ("SF", "DET", "ATL", "GB"),
    Position.WR: ("MIN", "MIA", "LV", "SEA"),
    Position.TE: ("BAL", "KC", "DET", "LAR"),
    Position.K: ("DAL", "NYG", "TB", "NO"),
    Position.DST: ("PIT", "CLE", "NYJ", "DEN"),
}


def _deterministic_jitter(index: int, spread: float) -> float:
    """A reproducible pseudo-random offset in roughly ``[-spread, +spread]``.

    A hand-rolled cycle rather than ``random`` so the fixture is identical on
    every machine and every run without seeding a global generator.
    """
    cycle = (-1.0, 0.4, 0.9, -0.6, 0.2, -0.3, 0.7, -0.9, 0.5, 0.1)
    return cycle[index % len(cycle)] * spread


def _plan_for(name: str, season: int) -> list[Position]:
    """A manager's script for one season, cycling when they have fewer than one each."""
    scripts = PLANS[name]
    return scripts[SEASONS.index(season) % len(scripts)]


def _build_draft(season: int) -> HistoricalDraft:
    """One season of a 4-team, 10-round snake draft following the plans above."""
    names = list(PLANS)
    picks: list[HistoricalPick] = []
    overall = 0
    for rnd in range(1, ROUNDS + 1):
        order = names if rnd % 2 else list(reversed(names))
        for name in order:
            overall += 1
            position = _plan_for(name, season)[rnd - 1]
            mean_reach, stdev = REACH_PROFILE[name]
            # ADP sits `mean_reach` picks later than the actual pick, so the
            # manager is reaching by that much on average, plus jitter.
            adp = float(overall) + mean_reach + _deterministic_jitter(overall + rnd, stdev)
            teams = NFL_TEAMS_BY_POSITION[position]
            picks.append(
                HistoricalPick(
                    season=season,
                    manager_name=name,
                    overall_pick=overall,
                    player_name=f"{position.value} {season}-{overall}",
                    position=position,
                    nfl_team=teams[(overall + rnd) % len(teams)],
                    round_number=rnd,
                    pick_in_round=order.index(name) + 1,
                    adp=max(1.0, adp),
                    platform_rank=max(1.0, float(overall) + _deterministic_jitter(overall, 2.0)),
                    is_rookie=(overall % 11 == 0),
                )
            )
    return HistoricalDraft(
        season=season,
        league_name="Synthetic Test League",
        platform=str(Platform.ESPN),
        team_count=TEAM_COUNT,
        rounds=ROUNDS,
        picks=picks,
    )


@pytest.fixture
def synthetic_history() -> DraftHistory:
    """Three seasons of drafts in which every manager follows a known plan."""
    return DraftHistory(drafts=[_build_draft(season) for season in SEASONS])


@pytest.fixture
def synthetic_league() -> League:
    """A 4-team league whose managers match the synthetic history by name."""
    config = LeagueConfig(
        name="Synthetic Test League",
        season=2026,
        platform=Platform.ESPN,
        team_count=TEAM_COUNT,
        rounds=ROUNDS,
        user_draft_slot=1,
    )
    managers = [
        Manager(name=name, draft_slot=slot, archetype=Archetype.BALANCED, is_user=(slot == 1))
        for slot, name in enumerate(PLANS, start=1)
    ]
    return League(config=config, managers=managers)


@pytest.fixture
def settings() -> SimulationConfig:
    """Default model settings, so a test that tweaks one is explicit about it."""
    return SimulationConfig()
