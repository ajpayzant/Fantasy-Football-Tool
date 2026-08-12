"""The fictional twelve-team sample league.

One league, twelve invented managers, each with a deliberately distinctive draft
plan (see :mod:`tests.fixtures.sample_league.drafts`). The plans exist so that the opponent
model has something real to recover: a demo in which every manager behaves
identically would make the profile page a wall of identical numbers and would
prove nothing about whether the estimator works.

The user takes over one of the twelve. Draft slot 6 by default — the middle of the
board, where the gap to your next pick is long enough that the wait-or-take
question the app is built around actually has two defensible answers. From slot 1
or 12 the back-to-back picks make most of those decisions for you.
"""

from __future__ import annotations

from core.config import LeagueConfig, RosterSettings, ScoringRules
from core.enums import (
    Archetype,
    DraftType,
    LeagueFormat,
    Platform,
    ScoringPreset,
    Slot,
)
from models.league import League
from models.manager import Manager, ManagerPreferences

from .names import MANAGER_NAMES

SAMPLE_LEAGUE_NAME = "Sample Dynasty of Dunces"
SAMPLE_SEASON = 2026
SAMPLE_TEAM_COUNT = 12
SAMPLE_ROUNDS = 16
SAMPLE_USER_SLOT = 6
"""Draft slot the user occupies by default. Mid-board on purpose — see module doc."""

HISTORY_SEASONS: tuple[int, ...] = (2023, 2024, 2025)
"""Seasons of draft history bundled with the sample league.

Three because that is the smallest number that lets the opponent model
distinguish a manager's consistent habit from one unusual draft, which is the
distinction its shrinkage machinery exists to make.
"""

SAMPLE_ROSTER = RosterSettings(
    slots={
        Slot.QB: 1, Slot.RB: 2, Slot.WR: 2, Slot.TE: 1,
        Slot.FLEX: 1, Slot.K: 1, Slot.DST: 1, Slot.BENCH: 7,
    }
)
"""A conventional 9-starter lineup. 9 starters + 7 bench = 16 rounds exactly, so a
completed sample draft leaves every team with a legal, full roster — a demo that
ends with unfillable seats reads as a bug in the app rather than the league."""


# ─────────────────────────────────────────────────────────────────────────────
# Managers
#
# Archetype here is the *label*, and it is the answer the opponent model is
# supposed to arrive at independently from the drafting behaviour in
# :mod:`tests.fixtures.sample_league.drafts`. It is deliberately not fed to the estimator.
# ─────────────────────────────────────────────────────────────────────────────
MANAGER_ARCHETYPES: tuple[Archetype, ...] = (
    Archetype.ZERO_RB,          # 1  Alicia Brandt
    Archetype.ROBUST_RB,        # 2  Dev Raghunathan
    Archetype.EARLY_QB,         # 3  Marcus Feld
    Archetype.ELITE_TE,         # 4  Priya Kaur
    Archetype.RANK_FOLLOWER,    # 5  Owen Castellanos
    Archetype.HERO_RB,          # 6  Nadia Oyelaran  (the user's seat)
    Archetype.LATE_QB,          # 7  Sam Whitlock
    Archetype.AUTODRAFT,        # 8  Tomas Iversen
    Archetype.ROOKIE_HEAVY,     # 9  Grace Lindqvist
    Archetype.HOMER,            # 10 Bishop Adeyemi
    Archetype.HIGH_VARIANCE,    # 11 Renata Cardozo
    Archetype.BALANCED,         # 12 Kyle Mahoney
)
"""One archetype per seat, chosen to cover every label ``infer_archetype`` can
return — so a user browsing the sample league sees the estimator's whole
vocabulary, and the test suite has a case for each.

:attr:`Archetype.BEST_PLAYER_AVAILABLE` is deliberately absent: no branch of
``infer_archetype`` returns it, so assigning it to a seat would guarantee that
seat's label could never be recovered and would make the demo look broken.
:attr:`Archetype.RANK_FOLLOWER` takes that seat instead."""

MANAGER_TEAM_NAMES: tuple[str, ...] = (
    "Brandt's Bandits", "Raghunathan Rangers", "Feld Marshals",
    "Kaur Corsairs", "Castellanos Comets", "Oyelaran Outlaws",
    "Whitlock Wolves", "Iversen Ironsides", "Lindqvist Lancers",
    "Adeyemi Admirals", "Cardozo Cyclones", "Mahoney Monarchs",
)

HOMER_FAVOURITE_TEAM = "KC"
"""The homer manager's team. Also drives their historical picks, so the
favourite-team detector has real evidence rather than a declared preference."""


def sample_managers(*, user_slot: int = SAMPLE_USER_SLOT) -> list[Manager]:
    """The twelve fictional managers, in draft-slot order.

    Only the homer carries a stated preference. The rest are left blank on purpose:
    the point of the sample league is to show what the model infers from *history*,
    and pre-filling every preference would hide that.
    """
    managers: list[Manager] = []
    for index, name in enumerate(MANAGER_NAMES):
        slot = index + 1
        preferences = ManagerPreferences()
        if MANAGER_ARCHETYPES[index] is Archetype.HOMER:
            preferences = ManagerPreferences(
                favorite_nfl_team=HOMER_FAVOURITE_TEAM,
                notes="Drafts their own team's players every year, without fail.",
            )
        managers.append(Manager(
            name=name,
            draft_slot=slot,
            team_name=MANAGER_TEAM_NAMES[index],
            is_user=(slot == int(user_slot)),
            archetype=MANAGER_ARCHETYPES[index],
            preferences=preferences,
        ))
    return managers


def sample_league_config(
    *,
    season: int = SAMPLE_SEASON,
    user_slot: int = SAMPLE_USER_SLOT,
    rounds: int = SAMPLE_ROUNDS,
) -> LeagueConfig:
    """Config for the sample league: 12-team, half-PPR, snake, redraft."""
    return LeagueConfig(
        name=SAMPLE_LEAGUE_NAME,
        season=int(season),
        platform=Platform.ESPN,
        team_count=SAMPLE_TEAM_COUNT,
        rounds=int(rounds),
        draft_type=DraftType.SNAKE,
        league_format=LeagueFormat.REDRAFT,
        scoring=ScoringRules.from_preset(ScoringPreset.HALF_PPR),
        roster=SAMPLE_ROSTER,
        user_draft_slot=int(user_slot),
        notes=(
            "SAMPLE LEAGUE — fictional managers and drafts, bundled so the app can "
            "be tried before importing your own league."
        ),
    )


def sample_league(
    *,
    season: int = SAMPLE_SEASON,
    user_slot: int = SAMPLE_USER_SLOT,
    rounds: int = SAMPLE_ROUNDS,
) -> League:
    """The assembled sample league: config plus twelve managers."""
    return League(
        config=sample_league_config(
            season=season, user_slot=user_slot, rounds=rounds
        ),
        managers=sample_managers(user_slot=user_slot),
    )


__all__ = [
    "sample_league", "sample_league_config", "sample_managers",
    "SAMPLE_LEAGUE_NAME", "SAMPLE_SEASON", "SAMPLE_TEAM_COUNT", "SAMPLE_ROUNDS",
    "SAMPLE_USER_SLOT", "SAMPLE_ROSTER", "HISTORY_SEASONS",
    "MANAGER_ARCHETYPES", "HOMER_FAVOURITE_TEAM",
]
