"""Tests for historical feature engineering."""

from __future__ import annotations

from core.config import ProfileEstimationConfig, RosterSettings
from core.enums import Position
from engine.features import (
    annotate_draft,
    annotate_history,
    draft_phase,
    summarize_history,
)
from models.draft import DraftHistory, HistoricalDraft, HistoricalPick, Pick
from models.player import Player, PlayerPool, PoolMetadata


def _pick(overall: int, manager: str, position: Position, team: str, **kw) -> HistoricalPick:
    return HistoricalPick(
        season=2025,
        manager_name=manager,
        overall_pick=overall,
        player_name=f"P{overall}",
        position=position,
        nfl_team=team,
        **kw,
    )


class TestReachSign:
    """``adp_delta`` must be positive when a manager drafts ahead of ADP."""

    def test_taking_a_late_adp_player_early_is_a_reach(self) -> None:
        pick = Pick(
            overall_pick=5, round_number=1, pick_in_round=5, draft_slot=5,
            manager_name="X", player_id="p", player_name="P",
            position=Position.RB, adp_at_pick=100.0,
        )
        assert pick.adp_delta == 95.0
        assert pick.is_reach
        assert not pick.is_value

    def test_a_player_who_falls_past_adp_is_value(self) -> None:
        pick = Pick(
            overall_pick=100, round_number=9, pick_in_round=4, draft_slot=4,
            manager_name="X", player_id="p", player_name="P",
            position=Position.RB, adp_at_pick=5.0,
        )
        assert pick.adp_delta == -95.0
        assert pick.is_value
        assert not pick.is_reach

    def test_small_gaps_are_neither(self) -> None:
        pick = Pick(
            overall_pick=20, round_number=2, pick_in_round=8, draft_slot=8,
            manager_name="X", player_id="p", player_name="P",
            position=Position.RB, adp_at_pick=22.0,
        )
        assert not pick.is_reach
        assert not pick.is_value

    def test_reach_picks_matches_adp_delta(self) -> None:
        """A 20-pick reach — inside the plausibility clip, so it survives."""
        pick = _pick(5, "X", Position.RB, "SF", adp=25.0)
        annotate_draft(HistoricalDraft(season=2025, team_count=4, picks=[pick]))
        assert pick.adp_delta == 20.0
        assert pick.reach_picks == 20.0

    def test_keeper_picks_are_excluded_from_reach(self) -> None:
        """A keeper's pick number is bookkeeping, not a decision."""
        keeper = _pick(1, "X", Position.RB, "SF", adp=80.0, is_keeper=True)
        annotate_draft(HistoricalDraft(season=2025, team_count=4, picks=[keeper]))
        assert keeper.adp_delta is None

    def test_implausible_deltas_are_dropped_as_bad_data(self) -> None:
        config = ProfileEstimationConfig(reach_clip_picks=10.0)
        pick = _pick(1, "X", Position.RB, "SF", adp=99.0)
        annotate_draft(
            HistoricalDraft(season=2025, team_count=4, picks=[pick]), config=config
        )
        assert pick.adp_delta is None


class TestStackAndHandcuff:
    """A stack is QB↔pass-catcher; a handcuff is RB↔RB. Same team is not enough."""

    def test_qb_then_wr_same_team_is_a_stack(self) -> None:
        picks = [
            _pick(1, "A", Position.QB, "KC"),
            _pick(2, "A", Position.WR, "KC"),
        ]
        annotate_draft(HistoricalDraft(season=2025, team_count=1, picks=picks))
        assert picks[1].was_stack

    def test_wr_then_qb_same_team_is_a_stack(self) -> None:
        picks = [
            _pick(1, "A", Position.WR, "KC"),
            _pick(2, "A", Position.QB, "KC"),
        ]
        annotate_draft(HistoricalDraft(season=2025, team_count=1, picks=picks))
        assert picks[1].was_stack

    def test_qb_then_te_same_team_is_a_stack(self) -> None:
        picks = [
            _pick(1, "A", Position.QB, "BAL"),
            _pick(2, "A", Position.TE, "BAL"),
        ]
        annotate_draft(HistoricalDraft(season=2025, team_count=1, picks=picks))
        assert picks[1].was_stack

    def test_rb_then_te_same_team_is_not_a_stack(self) -> None:
        """Regression: any same-team pairing used to count as a stack."""
        picks = [
            _pick(1, "A", Position.RB, "SF"),
            _pick(2, "A", Position.TE, "SF"),
        ]
        annotate_draft(HistoricalDraft(season=2025, team_count=1, picks=picks))
        assert not picks[1].was_stack

    def test_rb_then_rb_same_team_is_a_handcuff(self) -> None:
        picks = [
            _pick(1, "A", Position.RB, "GB"),
            _pick(2, "A", Position.RB, "GB"),
        ]
        annotate_draft(HistoricalDraft(season=2025, team_count=1, picks=picks))
        assert picks[1].was_handcuff

    def test_wr_then_rb_same_team_is_not_a_handcuff(self) -> None:
        """Regression: any same-team RB used to count as a handcuff."""
        picks = [
            _pick(1, "A", Position.WR, "GB"),
            _pick(2, "A", Position.RB, "GB"),
        ]
        annotate_draft(HistoricalDraft(season=2025, team_count=1, picks=picks))
        assert not picks[1].was_handcuff

    def test_ownership_is_tracked_per_manager(self) -> None:
        """One manager's QB must not create a stack for another's receiver."""
        picks = [
            _pick(1, "A", Position.QB, "KC"),
            _pick(2, "B", Position.WR, "KC"),
        ]
        annotate_draft(HistoricalDraft(season=2025, team_count=2, picks=picks))
        assert not picks[1].was_stack


class TestRunDetection:
    def test_run_is_detected_within_the_window(self) -> None:
        config = ProfileEstimationConfig(run_window_picks=6, run_threshold_picks=2)
        picks = [
            _pick(1, "A", Position.RB, "SF"),
            _pick(2, "B", Position.RB, "DET"),
            _pick(3, "C", Position.RB, "ATL"),
        ]
        annotate_draft(
            HistoricalDraft(season=2025, team_count=3, picks=picks), config=config
        )
        assert picks[0].started_run
        assert not picks[0].continued_run
        assert not picks[1].continued_run  # only one RB in the window so far
        assert picks[2].continued_run      # two before it — a run

    def test_run_window_is_bounded(self) -> None:
        """A position taken long ago must fall out of the window."""
        config = ProfileEstimationConfig(run_window_picks=2, run_threshold_picks=1)
        picks = [
            _pick(1, "A", Position.RB, "SF"),
            _pick(2, "B", Position.WR, "MIN"),
            _pick(3, "C", Position.WR, "MIA"),
            _pick(4, "D", Position.RB, "GB"),
        ]
        annotate_draft(
            HistoricalDraft(season=2025, team_count=4, picks=picks), config=config
        )
        assert picks[3].started_run  # the round-1 RB is outside a 2-pick window


class TestNoLookAhead:
    def test_features_use_only_prior_picks(self) -> None:
        """The first pick of a draft can know nothing about what follows."""
        picks = [
            _pick(1, "A", Position.RB, "SF"),
            _pick(2, "A", Position.RB, "SF"),
        ]
        annotate_draft(HistoricalDraft(season=2025, team_count=1, picks=picks))
        assert picks[0].position_count_before == 0
        assert not picks[0].was_handcuff
        assert picks[1].position_count_before == 1
        assert picks[1].was_handcuff

    def test_before_season_excludes_the_reference_year(self) -> None:
        history = DraftHistory(
            drafts=[
                HistoricalDraft(season=2024, picks=[_pick(1, "A", Position.RB, "SF")]),
                HistoricalDraft(season=2025, picks=[_pick(1, "A", Position.WR, "MIN")]),
            ]
        )
        earlier = history.before_season(2025)
        assert [d.season for d in earlier.drafts] == [2024]


class TestPicksUntilNext:
    def test_gap_to_the_managers_next_pick(self) -> None:
        picks = [
            _pick(1, "A", Position.RB, "SF"),
            _pick(2, "B", Position.WR, "MIN"),
            _pick(3, "B", Position.WR, "MIA"),
            _pick(4, "A", Position.TE, "BAL"),
        ]
        annotate_draft(HistoricalDraft(season=2025, team_count=2, picks=picks))
        assert picks[0].picks_until_next == 2   # picks 2 and 3 intervene
        assert picks[1].picks_until_next == 0   # back-to-back
        assert picks[3].picks_until_next is None


class TestDraftPhase:
    def test_thirds(self) -> None:
        assert draft_phase(1, 30) == "early"
        assert draft_phase(15, 30) == "middle"
        assert draft_phase(29, 30) == "late"

    def test_degenerate_total(self) -> None:
        assert draft_phase(1, 0) == "early"


class TestFilledStartingSlot:
    def test_starters_fill_before_bench(self) -> None:
        roster = RosterSettings()
        picks = [_pick(i, "A", Position.QB, "KC") for i in range(1, 4)]
        annotate_draft(
            HistoricalDraft(season=2025, team_count=1, picks=picks), roster=roster
        )
        # The first QB fills the QB slot; with one QB seat the next does not.
        assert picks[0].filled_starting_slot
        assert not picks[2].filled_starting_slot


class TestSummarizeHistory:
    def test_early_share_uses_an_exact_denominator(self) -> None:
        """Regression: early share was approximated, giving every manager the
        same denominator and therefore identical early-round bias."""
        picks = [
            _pick(1, "A", Position.RB, "SF", round_number=1),
            _pick(2, "B", Position.RB, "DET", round_number=1),
            _pick(3, "A", Position.WR, "MIN", round_number=2),
            _pick(4, "B", Position.QB, "KC", round_number=5),
        ]
        history = DraftHistory(
            drafts=[HistoricalDraft(season=2025, team_count=2, picks=picks)]
        )
        stats = summarize_history(history, early_rounds=3)
        assert stats.early_rounds == 3
        # Three picks fall in rounds 1-3: two RB, one WR. The round-5 QB does not.
        assert stats.early_share_by_position[Position.RB] == 2 / 3
        assert stats.early_share_by_position[Position.WR] == 1 / 3
        assert Position.QB not in stats.early_share_by_position

    def test_keepers_are_excluded_from_league_aggregates(self) -> None:
        picks = [
            _pick(1, "A", Position.RB, "SF", round_number=1, is_keeper=True),
            _pick(2, "B", Position.WR, "MIN", round_number=1),
        ]
        history = DraftHistory(
            drafts=[HistoricalDraft(season=2025, team_count=2, picks=picks)]
        )
        stats = summarize_history(history)
        assert stats.pick_count == 1
        assert Position.RB not in stats.share_by_position

    def test_empty_history_is_safe(self) -> None:
        stats = summarize_history(DraftHistory(), early_rounds=4)
        assert stats.pick_count == 0
        assert stats.early_rounds == 4
        assert stats.early_share_by_position == {}


class TestAnnotateHistory:
    def test_annotates_every_draft_and_reports_totals(
        self, synthetic_history: DraftHistory
    ) -> None:
        stats = annotate_history(synthetic_history)
        assert stats.pick_count == len(synthetic_history.all_picks)
        assert stats.manager_count == 4
        assert all(p.draft_phase for p in synthetic_history.all_picks)

    def test_annotation_is_idempotent(self, synthetic_history: DraftHistory) -> None:
        first = annotate_history(synthetic_history)
        snapshot = [p.adp_delta for p in synthetic_history.all_picks]
        second = annotate_history(synthetic_history)
        assert [p.adp_delta for p in synthetic_history.all_picks] == snapshot
        assert first.pick_count == second.pick_count


class TestSeasonSpecificFieldsStayInTheirSeason:
    """Regression: last season's picks were priced off this season's board.

    ADP describes one August. Filling a 2025 pick from the 2026 board reported
    Ja'Marr Chase going at 4.7 in a draft where he went 46th, so every manager who
    took a player whose stock had moved looked like they reached forty picks — which
    collapsed their estimated predictability and made the simulator draft erratically
    on their behalf.
    """

    def _board(self, season: int | None) -> PlayerPool:
        return PlayerPool(
            [
                Player(
                    player_id="p1", name="P1", position=Position.RB,
                    overall_adp=4.7, platform_rank=5.0, projection=250.0, tier=1,
                )
            ],
            metadata=PoolMetadata(season=season),
        )

    def _draft(self, season: int, **kw) -> HistoricalDraft:
        pick = HistoricalPick(
            season=season, manager_name="A", overall_pick=46,
            player_name="P1", position=Position.RB, **kw,
        )
        return HistoricalDraft(season=season, picks=[pick])

    def test_a_same_season_board_still_fills_the_gaps(self) -> None:
        draft = self._draft(2026)
        annotate_draft(draft, pool=self._board(2026))
        assert draft.picks[0].adp == 4.7
        assert draft.picks[0].tier == 1

    def test_a_different_season_board_is_not_used(self) -> None:
        draft = self._draft(2025)
        annotate_draft(draft, pool=self._board(2026))
        assert draft.picks[0].adp is None
        assert draft.picks[0].adp_delta is None
        # Position and NFL team are facts about the player, not the August.
        assert draft.picks[0].position is Position.RB

    def test_an_adp_already_copied_off_the_wrong_board_is_dropped(self) -> None:
        """The bad values are in the database already; nothing else revisits them."""
        draft = self._draft(2025, adp=4.7, platform_rank=5.0)
        annotate_draft(draft, pool=self._board(2026))
        assert draft.picks[0].adp is None
        assert draft.picks[0].platform_rank is None

    def test_a_genuine_adp_from_the_users_file_survives(self) -> None:
        """A real 2025 ADP does not agree with a 2026 ADP to the decimal place."""
        draft = self._draft(2025, adp=42.0)
        annotate_draft(draft, pool=self._board(2026))
        assert draft.picks[0].adp == 42.0
        assert draft.picks[0].adp_delta == 42.0 - 46.0

    def test_a_board_with_no_season_is_trusted(self) -> None:
        """An uploaded spreadsheet often says nothing about which year it is."""
        draft = self._draft(2025)
        annotate_draft(draft, pool=self._board(None))
        assert draft.picks[0].adp == 4.7
