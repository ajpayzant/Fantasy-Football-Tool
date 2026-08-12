"""Draft types: every one offered is simulated, and a retired one still loads.

There used to be an ``AUCTION`` draft type. It was selectable, it validated, and it
modelled nothing — pick order fell through to plain ascending and a warning was
raised after the fact. It has been removed, and these tests exist for the two ways
that removal could go wrong.

The first is that it creeps back: someone adds an enum member for a format the
engine cannot simulate, and it becomes selectable again. :func:`test_every_draft_type_produces_a_real_order`
fails on any member ``round_slot_order`` does not genuinely handle.

The second is that removing it breaks a league already saved with the old value.
There is a real SQLite database in ``data/`` that may hold one, and a user whose
league fails to load has lost their setup, so the fallback is pinned here rather
than left to chance.
"""

from __future__ import annotations

import pytest

from core.config import LeagueConfig
from core.enums import DraftType
from core.validation import validate_league
from engine.draft_order import round_slot_order

TEAMS = 12
ROUNDS = 4


def _config(**overrides) -> LeagueConfig:
    return LeagueConfig(team_count=TEAMS, rounds=ROUNDS, **overrides)


# ─────────────────────────────────────────────────────────────────────────────
# Nothing on offer is a placeholder
# ─────────────────────────────────────────────────────────────────────────────
def test_auction_is_gone() -> None:
    """Not merely unused — absent, so it cannot be selected or stored again."""
    assert "auction" not in DraftType.values()
    assert not hasattr(DraftType, "AUCTION")


@pytest.mark.parametrize("draft_type", list(DraftType))
def test_every_draft_type_produces_a_real_order(draft_type: DraftType) -> None:
    """Each round must be a genuine permutation of the league's slots.

    This is the guard against another placeholder: a draft type that is accepted
    but unmodelled shows up here as an order that ignores the round entirely.
    """
    config = _config(draft_type=draft_type)
    expected = set(range(1, TEAMS + 1))
    for round_number in range(1, ROUNDS + 1):
        order = round_slot_order(config, round_number)
        assert len(order) == TEAMS, f"{draft_type} round {round_number}: {order}"
        assert set(order) == expected, f"{draft_type} round {round_number}: {order}"


def test_the_types_that_claim_to_reverse_actually_reverse() -> None:
    """Snake and linear must differ in even rounds, or one of them is a lie.

    ``AUCTION`` passed the permutation check above while being pure ascending in
    every round. This is the assertion that would have caught it.
    """
    snake = round_slot_order(_config(draft_type=DraftType.SNAKE), 2)
    linear = round_slot_order(_config(draft_type=DraftType.LINEAR), 2)
    assert snake == list(reversed(range(1, TEAMS + 1)))
    assert linear == list(range(1, TEAMS + 1))
    assert snake != linear


# ─────────────────────────────────────────────────────────────────────────────
# A league saved before the removal still opens
# ─────────────────────────────────────────────────────────────────────────────
def test_a_league_saved_as_auction_loads_as_a_snake() -> None:
    """Falls back rather than raising: losing a saved league is worse than a demotion.

    Snake specifically, because that is what the old auction branch did with the
    picks anyway — so this is the behaviour the user already had, not a new guess.
    """
    config = _config(draft_type="auction")
    assert config.draft_type is DraftType.SNAKE
    assert round_slot_order(config, 2) == list(reversed(range(1, TEAMS + 1)))


def test_a_legacy_auction_league_validates_without_complaint() -> None:
    """It is a snake league now, so there is nothing left to warn about.

    Full 16 rounds here, unlike the other tests: the shared ``ROUNDS`` of 4 cannot
    seat a default roster, and that error would mask the thing being checked.
    """
    report = validate_league(
        LeagueConfig(team_count=TEAMS, rounds=16, draft_type="auction")
    )
    assert report.ok, [issue.message for issue in report.errors]
    assert not any("auction" in issue.code for issue in report.warnings)


def test_an_unrecognised_draft_type_also_falls_back() -> None:
    """The same path protects any other value a hand-edited database might hold."""
    assert _config(draft_type="best_ball_mega_format").draft_type is DraftType.SNAKE
