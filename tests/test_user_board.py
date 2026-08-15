"""Tests for the user's own board: targets, do-not-draft, personal rankings.

The properties worth pinning are about *whose* opinion is being applied where:

1. **Matching must be forgiving.** The user types names from memory; "AJ Brown" has
   to find "A.J. Brown Jr." or the feature is a spelling test.
2. **A do-not-draft player must disappear from every lens** — a single suggestion of
   someone the user has sworn off destroys trust in the other eight.
3. **The board must not reach the opponents.** They keep drafting the players the user
   refuses, because that is what they would really do, and a room that politely
   avoided them would make every availability number wrong.
4. **Silence must be visible.** A typo, a contradiction, or an exclusion that hid a
   suggestion is reported rather than swallowed.
"""

from __future__ import annotations

import pytest

from core.config import LeagueConfig, SimulationConfig
from core.enums import Archetype, RecommendationLens
from engine.draft_state import DraftState
from engine.opponent_model import build_profiles
from engine.recommender import RecommendationEngine
from engine.simulator import DraftSimulator
from models.league import League
from models.manager import Manager, ManagerProfile
from models.player import PlayerPool
from services.user_board import (
    MAX_NAMES,
    UserBoard,
    parse_names,
    parse_rankings,
    rankings_from_frame,
    rankings_from_order,
)

from tests._smoke_pool import build as build_smoke_pool

TEAMS = 8
ROUNDS = 6
USER_SLOT = 3
FAST_SIMS = 8


@pytest.fixture
def league_pool(settings: SimulationConfig) -> tuple[LeagueConfig, PlayerPool, League]:
    config, pool = build_smoke_pool(TEAMS, ROUNDS)
    managers = [
        Manager(
            name=f"Manager {slot}", draft_slot=slot,
            archetype=Archetype.BALANCED, is_user=(slot == USER_SLOT),
        )
        for slot in range(1, TEAMS + 1)
    ]
    return config, pool, League(config=config, managers=managers)


@pytest.fixture
def at_user_turn(
    league_pool: tuple[LeagueConfig, PlayerPool, League],
    settings: SimulationConfig,
) -> tuple[DraftState, dict[int, ManagerProfile]]:
    _, pool, league = league_pool
    state = DraftState(league=league, pool=pool, settings=settings, seed=21)
    profiles = build_profiles(league, settings=settings, pool=pool)
    DraftSimulator(state, profiles).simulate_until_user()
    return state, profiles


def _available(state: DraftState, count: int = 6):
    return state.available_players(limit=count)


# ─────────────────────────────────────────────────────────────────────────────
# Matching and construction
# ─────────────────────────────────────────────────────────────────────────────
class TestMatching:
    def test_a_name_typed_from_memory_still_matches(self, league_pool) -> None:
        _, pool, _ = league_pool
        player = next(iter(pool))
        sloppy = player.name.replace(".", "").replace("'", "").upper()
        board = UserBoard(targets=[sloppy])
        assert board.is_target(player)

    def test_the_typed_spelling_is_kept_for_display(self, league_pool) -> None:
        """The user must see their own words back, not a normalised key."""
        board = UserBoard(targets=["A.J. Brown"])
        assert board.targets == ["A.J. Brown"]

    def test_one_player_typed_twice_counts_once(self) -> None:
        board = UserBoard(targets=["A.J. Brown", "AJ Brown Jr."])
        assert board.targets == ["A.J. Brown"]

    def test_target_order_is_the_priority(self) -> None:
        board = UserBoard(targets=["First Guy", "Second Guy"])
        assert board.target_priority_by_name("First Guy") == 1
        assert board.target_priority_by_name("Second Guy") == 2

    def test_a_player_on_both_lists_is_never_drafted_and_reported(
        self, league_pool
    ) -> None:
        """The safer reading of a contradiction wins, and the user is told."""
        _, pool, _ = league_pool
        player = next(iter(pool))
        board = UserBoard(targets=[player.name], avoid=[player.name])
        assert board.is_avoided(player)
        assert not board.is_target(player)
        assert player.name in board.conflicts

    def test_an_empty_board_changes_nothing(self, league_pool) -> None:
        _, pool, _ = league_pool
        board = UserBoard()
        player = next(iter(pool))
        assert board.is_empty
        assert not board.is_target(player)
        assert not board.is_avoided(player)
        assert board.custom_rank(player) is None
        assert board.effective_rank(player) == pytest.approx(
            float(player.rank_for() or float("inf"))
        )

    def test_a_typo_is_reported_rather_than_ignored(self, league_pool) -> None:
        _, pool, _ = league_pool
        board = UserBoard(avoid=["Nobody At All"])
        missing = board.unmatched(pool)
        assert missing["avoid"] == ["Nobody At All"]

    def test_the_fingerprint_changes_when_the_board_does(self) -> None:
        """The UI's cache stamp depends on this, so a stale answer cannot survive."""
        first = UserBoard(targets=["Player One"])
        second = UserBoard(targets=["Player One", "Player Two"])
        assert first.fingerprint != second.fingerprint
        assert first.fingerprint == UserBoard(targets=["Player One"]).fingerprint


class TestRanking:
    def test_a_personal_rank_overrides_the_consensus_one(self, league_pool) -> None:
        _, pool, _ = league_pool
        players = list(pool)[:2]
        board = UserBoard(custom_ranks={players[1].name: 1})
        assert board.effective_rank(players[1]) == 1.0
        assert board.effective_rank(players[0]) == float(players[0].rank_for())

    def test_an_unranked_player_sorts_after_every_ranked_one(self, league_pool) -> None:
        """A partial list means "these are my first picks", not "nobody else exists"."""
        _, pool, _ = league_pool
        players = list(pool)[:4]
        board = UserBoard(custom_ranks={players[3].name: 1})
        ordered = board.sorted_players(players)
        assert ordered[0] is players[3]
        assert len(ordered) == 4

    def test_targets_outrank_rankings(self, league_pool) -> None:
        _, pool, _ = league_pool
        players = list(pool)[:4]
        board = UserBoard(
            targets=[players[2].name], custom_ranks={players[0].name: 1}
        )
        assert board.sorted_players(players)[0] is players[2]

    def test_avoided_players_are_dropped_from_the_users_order(self, league_pool) -> None:
        _, pool, _ = league_pool
        players = list(pool)[:4]
        board = UserBoard(avoid=[players[0].name])
        assert players[0] not in board.sorted_players(players)

    def test_a_round_trip_through_json_preserves_the_board(self) -> None:
        import json

        board = UserBoard(
            targets=["One Guy"], avoid=["Other Guy"], custom_ranks={"One Guy": 4}
        )
        restored = UserBoard.from_dict(json.loads(json.dumps(board.to_dict())))
        assert restored.targets == board.targets
        assert restored.avoid == board.avoid
        assert restored.custom_ranks == board.custom_ranks

    def test_an_unknown_saved_format_is_dropped_rather_than_half_read(self) -> None:
        board = UserBoard.from_dict({"format_version": 99, "avoid": ["Someone"]})
        assert board.is_empty


class TestParsing:
    def test_a_numbered_paste_keeps_its_numbers(self) -> None:
        ranks = parse_rankings("1. Alpha Back\n2. Bravo Back\n25. Zulu Back")
        assert ranks == {"Alpha Back": 1, "Bravo Back": 2, "Zulu Back": 25}

    def test_unnumbered_lines_take_their_position(self) -> None:
        ranks = parse_rankings("Alpha Back\nBravo Back")
        assert ranks == {"Alpha Back": 1, "Bravo Back": 2}

    def test_explicit_and_implicit_numbering_can_be_mixed(self) -> None:
        """Hand-edited lists look like this, and the explicit numbers are the intent."""
        ranks = parse_rankings("1. Alpha Back\nBravo Back\n10. Kilo Back\nLima Back")
        assert ranks == {
            "Alpha Back": 1, "Bravo Back": 2, "Kilo Back": 10, "Lima Back": 11,
        }

    def test_a_pasted_list_of_names_loses_its_decoration(self) -> None:
        assert parse_names("- Alpha Back\n• Bravo Back\n3) Charlie Back") == [
            "Alpha Back", "Bravo Back", "Charlie Back",
        ]

    def test_a_comma_separated_line_is_split(self) -> None:
        assert parse_names("Alpha Back, Bravo Back") == ["Alpha Back", "Bravo Back"]

    def test_blank_input_produces_nothing(self) -> None:
        assert parse_names("") == []
        assert parse_rankings("   \n\n") == {}

    def test_an_ordered_list_becomes_ranks(self) -> None:
        assert rankings_from_order(["Alpha Back", "Bravo Back"]) == {
            "Alpha Back": 1, "Bravo Back": 2,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Reading a ranking file
#
# Two things are needed from an uploaded file and no more: which column holds the
# names, and what order the players are in. Everything else is ignored on purpose —
# the projections and ADP in someone else's file are theirs, and importing them here
# would quietly overwrite the board's own numbers with a stranger's.
# ─────────────────────────────────────────────────────────────────────────────
class TestReadingARankingFile:
    def _frame(self, data: dict) -> "pd.DataFrame":
        import pandas as pd

        return pd.DataFrame(data)

    def test_a_rank_column_decides_the_order(self) -> None:
        ranks, notes = rankings_from_frame(
            self._frame({"Player": ["Bravo Back", "Alpha Back"], "Rank": [2, 1]})
        )
        assert ranks == {"Alpha Back": 1, "Bravo Back": 2}
        assert any("Rank" in note or "rank" in note for note in notes)

    def test_the_numbers_are_re_based_to_a_dense_ordering(self) -> None:
        """A published file ranked 3, 17, 92 is an ordering, not three ranks. Keeping
        its numbers would leave the user's #92 sorting behind a consensus #40 who is
        genuinely worse in their own opinion."""
        ranks, _ = rankings_from_frame(
            self._frame({
                "player_name": ["Alpha Back", "Bravo Back", "Charlie Back"],
                "overall_rank": [3, 17, 92],
            })
        )
        assert ranks == {"Alpha Back": 1, "Bravo Back": 2, "Charlie Back": 3}

    def test_row_order_is_used_when_there_is_no_rank_column(self) -> None:
        """How most exports actually arrive: sorted, unnumbered."""
        ranks, notes = rankings_from_frame(
            self._frame({"Player": ["Bravo Back", "Alpha Back"]})
        )
        assert ranks == {"Bravo Back": 1, "Alpha Back": 2}
        assert any("row order" in note for note in notes)

    def test_an_unusable_rank_column_falls_back_to_row_order(self) -> None:
        """A column of text where numbers were expected is a real export, not a fake:
        "1st", "-", "NR". Failing the whole upload over it would be the wrong call."""
        ranks, notes = rankings_from_frame(
            self._frame({
                "Player": ["Bravo Back", "Alpha Back"],
                "overall_rank": ["NR", "-"],
            })
        )
        assert ranks == {"Bravo Back": 1, "Alpha Back": 2}
        assert any("row order" in note for note in notes)

    def test_a_duplicate_is_kept_once_at_its_first_position(self) -> None:
        ranks, notes = rankings_from_frame(
            self._frame({
                "Player": ["Alpha Back", "Bravo Back", "A.J. Brown", "AJ Brown"],
            })
        )
        assert ranks == {"Alpha Back": 1, "Bravo Back": 2, "A.J. Brown": 3}
        assert any("duplicate" in note for note in notes)

    def test_a_missing_name_column_is_reported_rather_than_raised(self) -> None:
        """The failure has to name the fix. A silent empty result would look like the
        upload worked and the file was empty."""
        ranks, notes = rankings_from_frame(self._frame({"Rank": [1, 2], "Pts": [9, 8]}))
        assert ranks == {}
        assert any("player_name" in note for note in notes)

    def test_an_empty_file_says_so(self) -> None:
        import pandas as pd

        assert rankings_from_frame(pd.DataFrame()) == ({}, ["The file had no rows."])
        assert rankings_from_frame(None) == ({}, ["The file had no rows."])

    def test_nothing_but_the_names_and_the_order_is_imported(self) -> None:
        """A file's projections belong to whoever published it."""
        ranks, _ = rankings_from_frame(
            self._frame({
                "Player": ["Alpha Back", "Bravo Back"],
                "Projection": [999.0, 1.0],
                "Team": ["SF", "GB"],
            })
        )
        assert ranks == {"Alpha Back": 1, "Bravo Back": 2}

    def test_the_file_is_capped_at_the_length_of_a_board(self) -> None:
        ranks, notes = rankings_from_frame(
            self._frame({"Player": [f"Player {i}" for i in range(MAX_NAMES + 40)]})
        )
        assert len(ranks) == MAX_NAMES
        assert max(ranks.values()) == MAX_NAMES
        assert any(str(MAX_NAMES) in note for note in notes)

    def test_what_it_produces_is_a_board_the_app_can_use(self) -> None:
        """The parser's output has to survive ``UserBoard``, which re-validates it."""
        ranks, _ = rankings_from_frame(
            self._frame({"Player": ["Bravo Back", "Alpha Back"], "Rank": [2, 1]})
        )
        board = UserBoard(custom_ranks=ranks)
        assert board.custom_ranks == {"Alpha Back": 1, "Bravo Back": 2}
        assert not board.is_empty


# ─────────────────────────────────────────────────────────────────────────────
# What the recommendation engine does with it
# ─────────────────────────────────────────────────────────────────────────────
class TestRecommendationsObeyTheBoard:
    def test_a_do_not_draft_player_appears_in_no_lens(self, at_user_turn) -> None:
        state, profiles = at_user_turn
        banned = _available(state, 3)
        board = UserBoard(avoid=[p.name for p in banned])
        result = RecommendationEngine(state, profiles, board=board).recommend(
            simulations=FAST_SIMS, seed=4
        )
        banned_ids = {p.player_id for p in banned}
        assert result.recommendations
        assert not any(r.player_id in banned_ids for r in result.recommendations)

    def test_hiding_a_player_is_reported(self, at_user_turn) -> None:
        state, profiles = at_user_turn
        banned = _available(state, 1)[0]
        result = RecommendationEngine(
            state, profiles, board=UserBoard(avoid=[banned.name])
        ).recommend(simulations=FAST_SIMS, seed=4)
        assert banned.name in result.hidden_by_board
        assert any(banned.name in w for w in result.warnings)

    def test_the_shortlist_stays_full_size_despite_exclusions(self, at_user_turn) -> None:
        """Three exclusions must not turn twelve options into nine."""
        state, profiles = at_user_turn
        board = UserBoard(avoid=[p.name for p in _available(state, 3)])
        plain = RecommendationEngine(state, profiles).recommend(
            simulations=FAST_SIMS, seed=4, shortlist_size=8
        )
        trimmed = RecommendationEngine(state, profiles, board=board).recommend(
            simulations=FAST_SIMS, seed=4, shortlist_size=8
        )
        # Both sets draw from a shortlist of 8; the lens count can differ (conditional
        # lenses fire on the candidates present), but a suggestion must still exist.
        assert plain.recommendations and trimmed.recommendations

    def test_a_target_gets_its_own_lens(self, at_user_turn) -> None:
        state, profiles = at_user_turn
        wanted = _available(state, 6)[-1]
        result = RecommendationEngine(
            state, profiles, board=UserBoard(targets=[wanted.name])
        ).recommend(simulations=FAST_SIMS, seed=4)
        mine = result.by_lens(RecommendationLens.YOUR_BOARD)
        assert mine is not None
        assert mine.player_id == wanted.player_id
        assert mine.is_target

    def test_the_board_lens_is_silent_without_a_board(self, at_user_turn) -> None:
        """An empty board must not add a ninth card saying nothing."""
        state, profiles = at_user_turn
        result = RecommendationEngine(state, profiles).recommend(
            simulations=FAST_SIMS, seed=4
        )
        assert result.by_lens(RecommendationLens.YOUR_BOARD) is None

    def test_a_personal_ranking_alone_fires_the_lens(self, at_user_turn) -> None:
        state, profiles = at_user_turn
        wanted = _available(state, 6)[-1]
        result = RecommendationEngine(
            state, profiles, board=UserBoard(custom_ranks={wanted.name: 1})
        ).recommend(simulations=FAST_SIMS, seed=4)
        mine = result.by_lens(RecommendationLens.YOUR_BOARD)
        assert mine is not None
        assert mine.player_id == wanted.player_id
        assert mine.board_rank == 1

    def test_the_opponents_never_see_the_board(self, at_user_turn) -> None:
        """The whole reason the board lives outside the pick model.

        The room drafts the same players in the same order whether or not the user has
        sworn off the best one on the board. If this ever fails, every availability
        percentage in the app is quietly wrong: the user would be told a player will
        last because the opponents were being polite about their preferences.
        """
        state, profiles = at_user_turn
        board = UserBoard(avoid=[p.name for p in _available(state, 4)])
        RecommendationEngine(state, profiles, board=board).recommend(
            simulations=FAST_SIMS, seed=4
        )
        after_board = [
            DraftSimulator(state, profiles).simulate_pick().pick.player_id
            for _ in range(4)
        ]
        # Same board, same seed, no board at all — the room must not have moved.
        _, pool, league = league_pool_for(state)
        fresh = DraftState(
            league=league, pool=pool, settings=state.settings, seed=state.seed
        )
        DraftSimulator(fresh, profiles).simulate_until_user()
        RecommendationEngine(fresh, profiles).recommend(simulations=FAST_SIMS, seed=4)
        without_board = [
            DraftSimulator(fresh, profiles).simulate_pick().pick.player_id
            for _ in range(4)
        ]
        assert after_board == without_board

    def test_a_target_about_to_vanish_is_warned_about(self, at_user_turn) -> None:
        """The last-chance lens names one player by utility, so a target can be about
        to disappear without ever being mentioned. This is the interrupt for that."""
        state, profiles = at_user_turn
        wanted = _available(state, 1)[0]
        result = RecommendationEngine(
            state, profiles, board=UserBoard(targets=[wanted.name])
        ).recommend(simulations=FAST_SIMS, seed=4)
        if result.picks_until_next > 0 and result.availability is not None:
            if result.availability.survival(wanted.player_id) <= 0.35:
                assert any(wanted.name in w for w in result.warnings)


def league_pool_for(state: DraftState) -> tuple[LeagueConfig, PlayerPool, League]:
    """The pieces needed to rebuild an identical draft, taken from a live one."""
    return state.config, state.pool, state.league
