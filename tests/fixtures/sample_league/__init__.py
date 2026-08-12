"""Bundled fictional dataset: a player pool, a twelve-team league, three drafts.

**None of this is real.** The players do not exist, the managers do not exist, and
the projections are generated from a statistical shape rather than measured from
football. Everything here is labelled as sample data on the way in — see
:func:`sample_player_pool`, which sets ``is_sample_data`` on the imported pool —
so nothing downstream can mistake it for current NFL information.

It exists so the app is fully usable before a user has exported anything from
their own platform, and so the test suite has a realistic fixture. The generated
history is built from known per-manager plans, which makes it the only dataset that
can answer whether the opponent model actually recovers a tendency it was not told
about; ``tests/test_sample_data.py`` asserts it recovers all twelve.

The loaders return *frames* in the app's import format and then run them through
the real importers, rather than constructing model objects directly. Bypassing the
import path would mean the sample data exercised code the user's data never
touches, and vice versa.
"""

from __future__ import annotations

import pandas as pd

from models.draft import DraftHistory
from models.league import League
from models.player import PlayerPool
from services.importers import import_historical_drafts, import_player_pool

from .drafts import (
    DRAFT_PLANS,
    DRAFT_STYLES,
    HISTORY_SEED,
    plan_summary,
    sample_history_frame,
)
from .league import (
    HISTORY_SEASONS,
    HOMER_FAVOURITE_TEAM,
    MANAGER_ARCHETYPES,
    SAMPLE_LEAGUE_NAME,
    SAMPLE_ROSTER,
    SAMPLE_ROUNDS,
    SAMPLE_SEASON,
    SAMPLE_TEAM_COUNT,
    SAMPLE_USER_SLOT,
    sample_league,
    sample_league_config,
    sample_managers,
)
from .names import MANAGER_NAMES
from .players import (
    DEFAULT_PLAYER_COUNT,
    SAMPLE_SEED,
    sample_player_frame,
    sample_pool_summary,
)

SAMPLE_DATA_NOTICE = (
    "SAMPLE DATA — fictional players, managers and drafts, generated for "
    "demonstration. Not real NFL players, projections or draft results."
)
"""The one string every surface showing sample data must display.

Kept here rather than written per page so the labelling cannot drift out of sync
between the pages that show this data.
"""


def sample_player_pool(
    *, season: int = SAMPLE_SEASON, seed: int = SAMPLE_SEED
) -> PlayerPool:
    """The fictional player pool, through the real player-pool importer."""
    result = import_player_pool(
        sample_player_frame(seed=seed),
        source="sample data",
        season=int(season),
        is_sample_data=True,
    )
    return result.pool


def sample_draft_history(
    *,
    seasons: tuple[int, ...] = HISTORY_SEASONS,
    seed: int = HISTORY_SEED,
) -> DraftHistory:
    """Three seasons of fictional history, through the real draft importer."""
    result = import_historical_drafts(
        sample_history_frame(seasons=seasons, seed=seed),
        default_league_name=SAMPLE_LEAGUE_NAME,
        source_file="sample data",
    )
    return result.history


def sample_bundle(
    *, user_slot: int = SAMPLE_USER_SLOT
) -> tuple[League, PlayerPool, DraftHistory]:
    """Everything needed to start a sample draft, in one call.

    Returned as a tuple rather than a container class because the three pieces go
    to three different places: the league to the draft setup, the pool to the
    board, the history to the opponent model.
    """
    return (
        sample_league(user_slot=user_slot),
        sample_player_pool(),
        sample_draft_history(),
    )


def sample_history_summary() -> pd.DataFrame:
    """One row per sample manager: their designed plan, for display beside the
    profile the model inferred. See :func:`tests.fixtures.sample_league.drafts.plan_summary`."""
    return plan_summary()


__all__ = [
    # Loaders
    "sample_player_pool", "sample_draft_history", "sample_bundle",
    "sample_player_frame", "sample_history_frame", "sample_pool_summary",
    "sample_history_summary", "plan_summary",
    # League
    "sample_league", "sample_league_config", "sample_managers",
    # Labelling
    "SAMPLE_DATA_NOTICE",
    # Constants worth reaching for from outside
    "SAMPLE_LEAGUE_NAME", "SAMPLE_SEASON", "SAMPLE_TEAM_COUNT", "SAMPLE_ROUNDS",
    "SAMPLE_USER_SLOT", "SAMPLE_ROSTER", "SAMPLE_SEED", "HISTORY_SEED",
    "HISTORY_SEASONS", "DEFAULT_PLAYER_COUNT", "MANAGER_NAMES",
    "MANAGER_ARCHETYPES", "HOMER_FAVOURITE_TEAM", "DRAFT_PLANS", "DRAFT_STYLES",
]
