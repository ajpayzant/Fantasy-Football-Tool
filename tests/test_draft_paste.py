"""Pasted draft boards → importable rows.

The samples below are shaped like what actually lands on the clipboard from each
platform's recap page, dashes and all. They are the point of these tests: a parser
for a format nobody specified is only as good as the range of real text it survives,
so each layout is tested on text with the platform's own punctuation rather than on
something convenient.

Every test here also asserts the result *reaches the importer*, because a frame that
parses cleanly and then fails validation has not imported anybody's league.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.enums import Position
from services.draft_paste import (
    LAYOUT_LABELS,
    detect_layout,
    parse_draft_board,
)
from services.importers import import_historical_drafts

# ─────────────────────────────────────────────────────────────────────────────
# Samples
# ─────────────────────────────────────────────────────────────────────────────
# ESPN's recap ordered by pick: a round header, then "n. Team — Player, POS TEAM".
ESPN_BY_ROUND = """Draft Recap
ROUND 1
1. Team Alpha — Ja'Marr Chase, WR CIN
2. Beta Ballers — Bijan Robinson, RB ATL
3. Gamma Squad — CeeDee Lamb, WR DAL
4. Delta Force — Breece Hall, RB NYJ
ROUND 2
5. Delta Force — Puka Nacua, WR LAR
6. Gamma Squad — Sam LaPorta, TE DET
7. Beta Ballers — Garrett Wilson, WR NYJ
8. Team Alpha — Jahmyr Gibbs, RB DET
"""

# The "by team" view: a block per manager, no pick numbers anywhere.
YAHOO_BY_TEAM = """Team Alpha
Ja'Marr Chase WR CIN
Jahmyr Gibbs RB DET
Travis Kelce TE KC
Beta Ballers
Bijan Robinson RB ATL
Garrett Wilson WR NYJ
Mike Evans WR TB
"""

# A board copied out of an HTML table: tab-separated, rounds down, teams across.
BOARD_GRID = (
    "Round\tTeam Alpha\tBeta Ballers\tGamma Squad\tDelta Force\n"
    "1\tJa'Marr Chase WR CIN\tBijan Robinson RB ATL\tCeeDee Lamb WR DAL\t"
    "Breece Hall RB NYJ\n"
    "2\tJahmyr Gibbs RB DET\tGarrett Wilson WR NYJ\tSam LaPorta TE DET\t"
    "Puka Nacua WR LAR\n"
)

# Round.pick notation, which is how most spreadsheets and Sleeper exports read.
DOT_NOTATION = """1.01 Team Alpha - Ja'Marr Chase WR CIN
1.02 Beta Ballers - Bijan Robinson RB ATL
2.01 Team Alpha - Jahmyr Gibbs RB DET
2.02 Beta Ballers - Puka Nacua WR LAR
"""


def _imports_cleanly(result, *, season: int = 2025) -> None:
    """The frame must survive the real importer, not just the parser."""
    outcome = import_historical_drafts(result.frame, default_season=season)
    assert outcome.ok, outcome.summary()
    assert outcome.rejected_rows == 0, outcome.report.rejected
    assert len(outcome.history.all_picks) == result.pick_count


# ─────────────────────────────────────────────────────────────────────────────
# Layout detection
# ─────────────────────────────────────────────────────────────────────────────
class TestLayoutDetection:
    """Naming the shape wrong is the failure that produces plausible nonsense."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            (ESPN_BY_ROUND, "round_blocks"),
            (YAHOO_BY_TEAM, "team_blocks"),
            (BOARD_GRID, "grid"),
            (DOT_NOTATION, "pick_list"),
        ],
    )
    def test_each_sample_is_recognised(self, text: str, expected: str) -> None:
        assert detect_layout(text) == expected

    def test_every_layout_has_a_label_for_the_ui(self) -> None:
        # The UI states which shape was assumed; an unlabelled layout would show a
        # bare identifier to the user.
        for text in (ESPN_BY_ROUND, YAHOO_BY_TEAM, BOARD_GRID, DOT_NOTATION):
            assert detect_layout(text) in LAYOUT_LABELS

    def test_nothing_recognises_as_nothing(self) -> None:
        assert detect_layout("") == ""
        assert detect_layout("   \n\n  ") == ""


# ─────────────────────────────────────────────────────────────────────────────
# Round blocks
# ─────────────────────────────────────────────────────────────────────────────
class TestRoundBlocks:
    def test_every_pick_is_read_with_its_round_and_owner(self) -> None:
        result = parse_draft_board(ESPN_BY_ROUND, season=2025)
        assert result.ok, result.report.summary()
        assert result.pick_count == 8
        assert result.managers == [
            "Beta Ballers", "Delta Force", "Gamma Squad", "Team Alpha",
        ]
        first = result.frame.iloc[0]
        assert first["player_name"] == "Ja'Marr Chase"
        assert first["manager_name"] == "Team Alpha"
        assert int(first["round"]) == 1
        assert int(first["overall_pick"]) == 1
        _imports_cleanly(result)

    def test_the_round_header_wins_over_the_line_count(self) -> None:
        # Round 2's picks are numbered 5-8 overall, so the round can only come from
        # the header. Deriving it from position in the file would say round 1.
        result = parse_draft_board(ESPN_BY_ROUND, season=2025)
        rounds = result.frame.set_index("player_name")["round"].astype(int)
        assert rounds["Puka Nacua"] == 2
        assert rounds["Jahmyr Gibbs"] == 2

    def test_position_and_team_come_off_the_player_name(self) -> None:
        result = parse_draft_board(ESPN_BY_ROUND, season=2025)
        row = result.frame.set_index("player_name").loc["Sam LaPorta"]
        assert row["position"] == str(Position.TE)
        assert row["nfl_team"] == "DET"
        # And the name is not left carrying them.
        assert "TE" not in str(row.name).split()

    def test_a_line_before_the_first_round_header_is_reported(self) -> None:
        # Forced, because one header is not enough evidence to detect this layout —
        # the point under test is that the round-block parser does not silently
        # attribute an orphan line to a round it cannot know.
        result = parse_draft_board(
            "1. Team Alpha — Ja'Marr Chase\nROUND 1\n2. Beta Ballers — Bijan Robinson",
            season=2025,
            layout="round_blocks",
        )
        assert result.pick_count == 1
        assert [u.line_number for u in result.unparsed] == [1]
        assert "before the first round header" in result.unparsed[0].reason


# ─────────────────────────────────────────────────────────────────────────────
# Team blocks
# ─────────────────────────────────────────────────────────────────────────────
class TestTeamBlocks:
    def test_the_block_header_owns_every_pick_beneath_it(self) -> None:
        result = parse_draft_board(YAHOO_BY_TEAM, season=2025)
        assert result.ok, result.report.summary()
        assert result.pick_count == 6
        owners = result.frame.set_index("player_name")["manager_name"]
        assert owners["Ja'Marr Chase"] == "Team Alpha"
        assert owners["Mike Evans"] == "Beta Ballers"
        _imports_cleanly(result)

    def test_rounds_come_from_position_within_the_block(self) -> None:
        result = parse_draft_board(YAHOO_BY_TEAM, season=2025)
        rounds = result.frame.set_index("player_name")["round"].astype(int)
        assert rounds["Ja'Marr Chase"] == 1
        assert rounds["Jahmyr Gibbs"] == 2
        assert rounds["Travis Kelce"] == 3

    def test_the_reconstruction_is_declared_rather_than_assumed_silently(self) -> None:
        # Nothing in a by-team paste says whether the draft snaked. Guessing is
        # unavoidable; not saying so is not.
        result = parse_draft_board(YAHOO_BY_TEAM, season=2025)
        assert any("reconstructed" in note for note in result.notes)
        assert any("snake" in note for note in result.notes)

    def test_a_snake_reconstruction_reverses_the_even_rounds(self) -> None:
        snaked = parse_draft_board(YAHOO_BY_TEAM, season=2025, snake=True)
        linear = parse_draft_board(YAHOO_BY_TEAM, season=2025, snake=False)
        snake_two = snaked.frame.set_index("player_name")["overall_pick"].astype(int)
        linear_two = linear.frame.set_index("player_name")["overall_pick"].astype(int)
        # Round 2, two teams: the second block picks first under a snake.
        assert snake_two["Garrett Wilson"] == 3
        assert snake_two["Jahmyr Gibbs"] == 4
        assert linear_two["Jahmyr Gibbs"] == 3
        assert linear_two["Garrett Wilson"] == 4

    def test_the_rows_come_back_in_draft_order_not_block_order(self) -> None:
        result = parse_draft_board(YAHOO_BY_TEAM, season=2025)
        overall = result.frame["overall_pick"].astype(int).tolist()
        assert overall == sorted(overall)


# ─────────────────────────────────────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────────────────────────────────────
class TestGrid:
    def test_a_column_is_a_team_and_a_row_is_a_round(self) -> None:
        result = parse_draft_board(BOARD_GRID, season=2025)
        assert result.ok, result.report.summary()
        assert result.pick_count == 8
        owners = result.frame.set_index("player_name")["manager_name"]
        assert owners["Ja'Marr Chase"] == "Team Alpha"
        assert owners["Puka Nacua"] == "Delta Force"
        _imports_cleanly(result)

    def test_even_rounds_of_a_snake_board_run_right_to_left(self) -> None:
        """The board's columns are teams, not pick positions.

        This is the one place where a column index is not a pick number: in a snake
        the leftmost team picks *last* in round two. Numbering by column would put
        half the draft in the wrong order and misstate every manager's reach, while
        still producing a frame that imports without complaint.
        """
        result = parse_draft_board(BOARD_GRID, season=2025, snake=True)
        overall = result.frame.set_index("player_name")["overall_pick"].astype(int)
        assert overall["Puka Nacua"] == 5      # Delta Force, rightmost column
        assert overall["Jahmyr Gibbs"] == 8    # Team Alpha, leftmost column

        linear = parse_draft_board(BOARD_GRID, season=2025, snake=False)
        straight = linear.frame.set_index("player_name")["overall_pick"].astype(int)
        assert straight["Jahmyr Gibbs"] == 5
        assert straight["Puka Nacua"] == 8

    def test_a_pick_number_already_in_the_cell_is_believed(self) -> None:
        # If the board prints its own numbering, that is the draft's answer and beats
        # anything derived from the geometry.
        text = (
            "Round\tTeam Alpha\tBeta Ballers\n"
            "1\t1.01 Ja'Marr Chase WR CIN\t1.02 Bijan Robinson RB ATL\n"
            "2\t2.02 Jahmyr Gibbs RB DET\t2.01 Puka Nacua WR LAR\n"
        )
        result = parse_draft_board(text, season=2025)
        in_round = result.frame.set_index("player_name")["pick_in_round"].astype(int)
        assert in_round["Jahmyr Gibbs"] == 2
        assert in_round["Puka Nacua"] == 1

    def test_a_space_aligned_board_is_read_as_a_board(self) -> None:
        # Not every copy carries tabs; a printable recap is aligned with spaces.
        text = (
            "Round    Team Alpha        Beta Ballers\n"
            "1        Ja'Marr Chase     Bijan Robinson\n"
            "2        Jahmyr Gibbs      Puka Nacua\n"
        )
        assert detect_layout(text) == "grid"
        result = parse_draft_board(text, season=2025)
        assert result.pick_count == 4
        assert result.managers == ["Beta Ballers", "Team Alpha"]

    def test_a_board_with_no_header_row_is_an_error_not_a_guess(self) -> None:
        """Without team names the columns belong to nobody, and inventing owners
        would attribute real picks to managers who never made them."""
        text = (
            "1\tJa'Marr Chase\tBijan Robinson\tCeeDee Lamb\n"
            "2\tJahmyr Gibbs\tGarrett Wilson\tSam LaPorta\n"
        )
        result = parse_draft_board(text, season=2025)
        assert not result.ok
        codes = {issue.code for issue in result.report.errors}
        assert "grid_no_header" in codes or "no_picks_found" in codes


# ─────────────────────────────────────────────────────────────────────────────
# Flat pick lists
# ─────────────────────────────────────────────────────────────────────────────
class TestPickList:
    def test_dot_notation_gives_the_round_and_the_slot(self) -> None:
        result = parse_draft_board(DOT_NOTATION, season=2025)
        assert result.ok, result.report.summary()
        row = result.frame.set_index("player_name").loc["Jahmyr Gibbs"]
        assert int(row["round"]) == 2
        assert int(row["pick_in_round"]) == 1
        assert int(row["overall_pick"]) == 3
        _imports_cleanly(result)

    def test_a_bare_overall_number_is_taken_as_the_overall_pick(self) -> None:
        result = parse_draft_board(
            "1. Team Alpha: Ja'Marr Chase\n"
            "2. Beta Ballers: Bijan Robinson\n"
            "3. Beta Ballers: Puka Nacua\n",
            season=2025,
        )
        assert result.frame["overall_pick"].astype(int).tolist() == [1, 2, 3]

    def test_a_line_with_no_separable_manager_is_reported_not_dropped(self) -> None:
        result = parse_draft_board(
            "1.01 Team Alpha - Ja'Marr Chase\n"
            "1.02 Bijan Robinson\n"
            "1.03 Beta Ballers - CeeDee Lamb\n",
            season=2025,
        )
        assert result.pick_count == 2
        assert [u.line_number for u in result.unparsed] == [2]
        assert "manager" in result.unparsed[0].reason
        # And the unreadable line is available as a table for the UI to show.
        assert list(result.unparsed_frame().columns) == ["line", "text", "why"]

    def test_a_keeper_marker_is_recorded_and_removed_from_the_name(self) -> None:
        result = parse_draft_board(
            "1.01 Team Alpha - Ja'Marr Chase WR CIN (keeper)\n"
            "1.02 Beta Ballers - Bijan Robinson RB ATL\n",
            season=2025,
        )
        row = result.frame.set_index("player_name").loc["Ja'Marr Chase"]
        assert bool(row["keeper_flag"]) is True
        assert bool(result.frame.set_index("player_name").loc[
            "Bijan Robinson", "keeper_flag"
        ]) is False


# ─────────────────────────────────────────────────────────────────────────────
# Names
# ─────────────────────────────────────────────────────────────────────────────
class TestNameHandling:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ("1.01 A - Ja'Marr Chase WR CIN", "Ja'Marr Chase"),
            ("1.01 A - A.J. Brown WR PHI", "A.J. Brown"),
            ("1.01 A - Marvin Harrison Jr. WR ARI", "Marvin Harrison Jr."),
            ("1.01 A - Kenneth Walker III RB SEA", "Kenneth Walker III"),
            ("1.01 A - Brian Robinson Jr. (RB - WSH)", "Brian Robinson Jr."),
            ("1.01 A - De'Von Achane, RB MIA", "De'Von Achane"),
        ],
    )
    def test_a_name_survives_having_its_position_stripped(
        self, line: str, expected: str
    ) -> None:
        """Trimming the trailing POS/TEAM must not start eating the surname.

        Suffixes are the hazard: ``Jr.`` and ``III`` sit exactly where a position
        token sits, so an over-eager trim silently renames the player and the pick
        then matches nobody on the board.
        """
        result = parse_draft_board(line, season=2025)
        assert result.frame["player_name"].tolist() == [expected]

    def test_a_defence_and_a_kicker_read_as_players(self) -> None:
        result = parse_draft_board(
            "1.01 A - Baltimore Ravens D/ST\n1.02 B - Harrison Butker K KC\n",
            season=2025,
        )
        names = result.frame["player_name"].tolist()
        assert "Baltimore Ravens" in names
        assert "Harrison Butker" in names
        positions = result.frame.set_index("player_name")["position"]
        assert positions["Harrison Butker"] == str(Position.K)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting what is wrong or assumed
# ─────────────────────────────────────────────────────────────────────────────
class TestReporting:
    def test_an_empty_paste_is_an_error_not_an_empty_success(self) -> None:
        result = parse_draft_board("", season=2025)
        assert not result.ok
        assert {i.code for i in result.report.errors} == {"empty_paste"}

    def test_prose_produces_an_error_rather_than_invented_picks(self) -> None:
        result = parse_draft_board(
            "I had a great draft this year and I think I won it outright.\n"
            "My league is very competitive.\n",
            season=2025,
        )
        assert not result.ok
        assert result.frame.empty

    def test_a_board_pasted_twice_is_flagged(self) -> None:
        result = parse_draft_board(ESPN_BY_ROUND + ESPN_BY_ROUND, season=2025)
        codes = {issue.code for issue in result.report.warnings}
        assert "duplicate_picks" in codes

    def test_a_truncated_recap_is_flagged(self) -> None:
        # The commonest real failure: the copy stops partway and the row count alone
        # looks perfectly healthy.
        result = parse_draft_board(
            "1. A: Ja'Marr Chase\n2. B: Bijan Robinson\n7. B: Puka Nacua\n",
            season=2025,
        )
        warnings = {issue.code: issue.message for issue in result.report.warnings}
        assert "missing_picks" in warnings
        assert "3" in warnings["missing_picks"]

    def test_names_are_snapped_onto_the_league_and_misses_are_named(self) -> None:
        result = parse_draft_board(
            ESPN_BY_ROUND,
            season=2025,
            manager_names=["team alpha", "Beta Ballers", "Someone Else"],
        )
        assert "team alpha" in result.frame["manager_name"].tolist()
        note = " ".join(result.notes)
        assert "Gamma Squad" in note and "Delta Force" in note

    def test_the_season_is_carried_onto_every_row(self) -> None:
        result = parse_draft_board(ESPN_BY_ROUND, season=2019, league_name="Dynasty")
        assert set(result.frame["season"]) == {2019}
        assert set(result.frame["league_name"]) == {"Dynasty"}

    def test_a_missing_season_is_called_out(self) -> None:
        # The importer requires one, so parsing without it has to say so rather than
        # produce rows that will be rejected later for no visible reason.
        result = parse_draft_board(ESPN_BY_ROUND)
        assert any("season" in note.lower() for note in result.notes)

    def test_a_forced_layout_overrides_detection(self) -> None:
        # The escape hatch for when detection is wrong: the user picks the shape.
        result = parse_draft_board(ESPN_BY_ROUND, season=2025, layout="pick_list")
        assert result.layout == "pick_list"
        assert result.pick_count == 8

    def test_the_frame_always_carries_the_importers_own_columns(self) -> None:
        from core.constants import HISTORICAL_IMPORT_COLUMNS

        result = parse_draft_board(ESPN_BY_ROUND, season=2025)
        assert list(result.frame.columns)[: len(HISTORICAL_IMPORT_COLUMNS)] == list(
            HISTORICAL_IMPORT_COLUMNS
        )

    def test_describe_says_what_happened_in_one_sentence(self) -> None:
        result = parse_draft_board(ESPN_BY_ROUND, season=2025)
        sentence = result.describe()
        assert "8 pick" in sentence
        assert LAYOUT_LABELS["round_blocks"] in sentence

    def test_nothing_here_raises_on_junk(self) -> None:
        """Every entry point takes user data, so an exception is a bug by contract."""
        for junk in ("", "\t\t\t", "—", "1.", "Round 1", "\x00\x01", "a" * 5000):
            result = parse_draft_board(junk, season=2025)
            assert isinstance(result.frame, pd.DataFrame)
            detect_layout(junk)
