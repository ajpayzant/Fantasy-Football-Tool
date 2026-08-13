"""Whether an in-progress draft survives a refresh.

The bug: a draft lived only in ``st.session_state``, so it lasted exactly as long as
the browser tab's connection. Forty picks in, a stray Ctrl-R destroyed it silently —
no warning beforehand, and nothing to go back to afterwards.

The fix stores a *list of decisions* and replays them, rather than pickling the
``DraftState``. So the tests here are mostly about replay fidelity, and the sharpest
one is :func:`test_a_replayed_draft_is_indistinguishable_from_the_original`: if the
rebuilt rosters, clock and availability match the originals exactly, then everything
downstream — recommendations, survival odds, the pick model — sees the same draft it
would have seen without the refresh.

The other half is about *not* lying. Replay stops at the first pick it cannot make
rather than skipping it, because skipping shifts every later pick one slot earlier
and produces a plausible draft that never happened.
"""

from __future__ import annotations

import pytest

from core.config import SimulationConfig
from core.enums import Archetype
from engine.draft_state import DraftState
from engine.simulator import DraftSimulator
from models.league import Keeper, League
from models.manager import Manager
from services import draft_session
from services.draft_session import DraftSnapshot, SnapshotPick

from tests._smoke_pool import build as build_smoke_pool
from tests.conftest import PLANS, ROUNDS, TEAM_COUNT


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def league_and_pool():
    config, pool = build_smoke_pool(TEAM_COUNT, ROUNDS)
    managers = [
        Manager(name=name, draft_slot=slot, archetype=Archetype.BALANCED,
                is_user=(slot == 1))
        for slot, name in enumerate(PLANS, start=1)
    ]
    return League(config=config, managers=managers), pool


def _profiles(league: League) -> dict:
    """League-average profiles. The simulator refuses to run without one per slot."""
    from engine.opponent_model import build_profiles
    from models.draft import DraftHistory

    return build_profiles(league, DraftHistory(), settings=SimulationConfig())


def _played(league: League, pool, picks: int, *, seed: int = 11) -> DraftState:
    """A real draft advanced ``picks`` picks by the simulator."""
    draft = DraftState(league, pool, settings=SimulationConfig(), seed=seed)
    simulator = DraftSimulator(draft, _profiles(league))
    for _ in range(picks):
        if simulator.simulate_pick() is None:
            break
    return draft


def _fingerprint(draft: DraftState) -> dict:
    """Everything about a draft that a later decision depends on.

    Compared wholesale rather than field by field so a *new* piece of state added to
    ``DraftState`` later is covered by this test without anyone remembering to come
    back and extend it.
    """
    return {
        "picks": [
            (p.overall_pick, p.player_id, p.draft_slot, p.manager_name,
             str(p.position), str(p.assigned_slot), p.is_keeper, p.is_user_pick)
            for p in draft.picks
        ],
        "pick_index": draft.pick_index,
        "on_the_clock": draft.current_slot.overall_pick if draft.current_slot else None,
        "status": str(draft.status),
        "drafted": sorted(draft.drafted_ids),
        "available": [p.player_id for p in draft.available_players()],
        "rosters": {
            slot: (
                sorted(draft.roster(slot).all_player_ids())
                if hasattr(draft.roster(slot), "all_player_ids")
                else sorted(
                    [pid for ids in draft.roster(slot).lineup.values() for pid in ids]
                    + list(draft.roster(slot).bench)
                )
            )
            for slot in draft.league.slots_in_order()
        },
        "lineups": {
            slot: {
                str(s): list(ids)
                for s, ids in draft.roster(slot).lineup.items()
            }
            for slot in draft.league.slots_in_order()
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Replay fidelity — the claim the whole feature rests on
# ─────────────────────────────────────────────────────────────────────────────
def test_a_replayed_draft_is_indistinguishable_from_the_original(league_and_pool) -> None:
    """The central assertion: restore produces the same draft, not a summary of it.

    Rosters, assigned lineup slots, the clock, who is available and in what order —
    all of it, because the recommendation engine reads all of it. A restore that got
    the pick list right but the lineup assignment wrong would show the user the right
    board and the wrong advice.
    """
    league, pool = league_and_pool
    original = _played(league, pool, 17)
    before = _fingerprint(original)

    snapshot = draft_session.snapshot_from_draft(original)
    restored = draft_session.restore(snapshot, league, pool)

    assert restored.warnings == []
    assert restored.replayed == 17
    assert _fingerprint(restored.draft) == before


def test_the_snapshot_survives_json(league_and_pool) -> None:
    """It is stored as JSON in SQLite, so the round trip has to be lossless.

    Specifically: ``strategy_notes`` is keyed by integer draft slot, and JSON object
    keys are always strings. Without coercion on the way back the notes would attach
    to nothing and disappear silently.
    """
    import json

    league, pool = league_and_pool
    draft = _played(league, pool, 9)
    draft.note_strategy(2, "hoarding receivers")
    draft.note_strategy(2, "no QB yet")

    snapshot = draft_session.snapshot_from_draft(draft, league_id=4, source_id=7)
    revived = DraftSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))

    assert revived is not None
    assert revived.league_id == 4 and revived.source_id == 7
    assert [p.player_id for p in revived.picks] == [p.player_id for p in snapshot.picks]
    assert revived.strategy_notes == {2: ["hoarding receivers", "no QB yet"]}
    assert draft_session.restore(revived, league, pool).draft.strategy_notes(2) == [
        "hoarding receivers", "no QB yet"
    ]


def test_undo_still_works_after_a_restore(league_and_pool) -> None:
    """Replay goes through ``make_pick``, so the undo stack is rebuilt as a side effect.

    Worth pinning explicitly: a restore that produced the right board but an empty
    undo stack would take away the one control a user reaches for immediately after
    realising they mis-clicked, and it would look like the feature working.
    """
    league, pool = league_and_pool
    draft = _played(league, pool, 12)
    snapshot = draft_session.snapshot_from_draft(draft)

    restored = draft_session.restore(snapshot, league, pool).draft
    assert restored.can_undo

    last = restored.picks[-1]
    undone = restored.undo()
    assert undone is not None and undone.player_id == last.player_id
    assert restored.pick_index == 11
    assert restored.is_available(last.player_id)


def test_the_room_keeps_behaving_the_same_after_a_resume(league_and_pool) -> None:
    """The seed alone is not enough, and this test is why the RNG position is stored.

    ``DraftState`` holds one seeded generator and the simulator draws from it on every
    pick, so its *stream position* is part of the draft. Rebuilding from the seed
    restarts the stream: the picks already made are fine, but from the resume onward
    the opponents draw different numbers and behave like a different room — with
    nothing on screen to explain why, and the *Random seed* control on the start form
    quietly no longer meaning what it says.

    Deliberately routed through JSON, because that is the only path the app uses and
    ``random.getstate()`` is a tuple of tuples that JSON does not preserve.
    """
    import json

    league, pool = league_and_pool
    draft = _played(league, pool, 8, seed=4242)
    stored = json.loads(json.dumps(draft_session.snapshot_from_draft(draft).to_dict()))
    snapshot = DraftSnapshot.from_dict(stored)
    assert snapshot is not None

    restored = draft_session.restore(snapshot, league, pool).draft
    assert restored.seed == 4242

    expected = DraftSimulator(draft, _profiles(league)).simulate_pick()
    actual = DraftSimulator(restored, _profiles(league)).simulate_pick()
    assert expected is not None and actual is not None
    assert actual.pick.player_id == expected.pick.player_id

    # And it keeps holding, rather than agreeing once by luck.
    for _ in range(6):
        expected = DraftSimulator(draft, _profiles(league)).simulate_pick()
        actual = DraftSimulator(restored, _profiles(league)).simulate_pick()
        assert actual.pick.player_id == expected.pick.player_id


def test_a_snapshot_without_an_rng_position_still_restores(league_and_pool) -> None:
    """Older saved drafts have no stored stream position; they must not be refused.

    Falling back to the seed makes the room behave slightly differently from here.
    That is a far better outcome than declining to reopen a draft that is otherwise
    entirely intact, so it is not even reported to the user.
    """
    league, pool = league_and_pool
    snapshot = draft_session.snapshot_from_draft(_played(league, pool, 6))
    snapshot.rng_state = []

    restored = draft_session.restore(snapshot, league, pool)
    assert restored.replayed == 6
    assert restored.warnings == []


def test_an_unusable_rng_position_is_reported_rather_than_raised(league_and_pool) -> None:
    """A state string from another Python build must not take the draft down with it."""
    league, pool = league_and_pool
    snapshot = draft_session.snapshot_from_draft(_played(league, pool, 6))
    snapshot.rng_state = [3, [1, 2, 3], None]  # too short for the Mersenne Twister

    restored = draft_session.restore(snapshot, league, pool)
    assert restored.replayed == 6, "the picks are still real and must survive"
    assert any("random stream" in w for w in restored.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Keepers — the one class of pick that must not be replayed
# ─────────────────────────────────────────────────────────────────────────────
def test_keeper_picks_are_left_out_and_reapplied_by_the_draft_itself(league_and_pool) -> None:
    """Keepers are not decisions, so storing them would double-draft the player.

    ``DraftState`` commits them from the league's keeper list the moment the clock
    reaches them. A snapshot that included them would replay a pick for a player the
    rebuilt state had already taken, and ``make_pick`` rejects that — so the restore
    would stop at the first keeper instead of the first real problem.
    """
    league, pool = league_and_pool
    keeper_player = pool.require("RB1")
    league.keepers = [
        Keeper(
            manager_name=league.managers[0].name,
            player_name=keeper_player.name,
            keeper_round=1,
        )
    ]

    draft = _played(league, pool, 10)
    assert any(p.is_keeper for p in draft.picks), "fixture should produce a keeper pick"

    snapshot = draft_session.snapshot_from_draft(draft)
    assert keeper_player.player_id not in {p.player_id for p in snapshot.picks}

    restored = draft_session.restore(snapshot, league, pool)
    assert restored.warnings == []
    assert _fingerprint(restored.draft) == _fingerprint(draft)
    # And the keeper is still on the roster, put there by the draft rather than the
    # snapshot.
    assert keeper_player.player_id in restored.draft.drafted_ids


# ─────────────────────────────────────────────────────────────────────────────
# Refusing to guess
# ─────────────────────────────────────────────────────────────────────────────
def test_a_missing_player_stops_the_replay_instead_of_being_skipped(league_and_pool) -> None:
    """Skipping one pick would silently shift every pick after it.

    The result would be a draft that looks entirely coherent and never happened —
    wrong rosters, wrong picks, no error. Stopping leaves the user at the last pick
    that is definitely real and names the player it gave up on.
    """
    league, pool = league_and_pool
    draft = _played(league, pool, 12)
    snapshot = draft_session.snapshot_from_draft(draft)

    # A board that no longer has the player taken 6th — as if the user re-fetched and
    # somebody was removed.
    gone = snapshot.picks[5]
    trimmed = [p for p in pool if p.player_id != gone.player_id]
    from models.player import PlayerPool

    thin_pool = PlayerPool(trimmed, league=league.config, metadata=pool.metadata)

    restored = draft_session.restore(snapshot, league, thin_pool)
    assert restored.replayed == 5, "must stop at the gap, not draft around it"
    assert restored.draft.pick_index == 5
    assert len(restored.warnings) == 1
    message = restored.warnings[0]
    assert gone.player_name in message
    assert "not on the board" in message
    assert not restored.is_exact


def test_an_unknown_snapshot_format_is_discarded_rather_than_guessed_at() -> None:
    """Offering to resume and then rebuilding it wrongly is worse than refusing."""
    raw = {"format_version": 99, "picks": [{"overall_pick": 1, "player_id": "RB1"}]}
    assert DraftSnapshot.from_dict(raw) is None
    assert DraftSnapshot.from_dict(None) is None
    assert DraftSnapshot.from_dict("not a mapping") is None


def test_a_differently_shaped_league_is_not_offered_a_resume(league_and_pool) -> None:
    """Every pick number in a snapshot is relative to one team count and round count.

    Replaying a 4-team snapshot into a 12-team league would put real picks in slots
    that never existed, so the offer is withheld rather than the restore being left
    to fail informatively later.
    """
    league, pool = league_and_pool
    snapshot = draft_session.snapshot_from_draft(_played(league, pool, 6))

    assert draft_session.resumable(snapshot, league, pool)

    wider = league.config.with_(team_count=12)
    wider_league = League(
        config=wider,
        managers=[
            Manager(name=f"M{slot}", draft_slot=slot) for slot in range(1, 13)
        ],
    )
    assert not draft_session.resumable(snapshot, wider_league, pool)

    next_season = League(
        config=league.config.with_(season=2027), managers=league.managers
    )
    assert not draft_session.resumable(snapshot, next_season, pool)


def test_a_draft_with_no_picks_is_not_offered(league_and_pool) -> None:
    """*Start draft* reproduces it exactly, so an offer to restore nothing is noise."""
    league, pool = league_and_pool
    empty = DraftState(league, pool, settings=SimulationConfig(), seed=1)
    snapshot = draft_session.snapshot_from_draft(empty)

    assert snapshot.is_empty
    assert not draft_session.resumable(snapshot, league, pool)


def test_a_snapshot_longer_than_the_draft_order_is_truncated_with_a_warning(
    league_and_pool,
) -> None:
    """A shortened league must not silently drop the overflow."""
    league, pool = league_and_pool
    draft = _played(league, pool, 12)
    snapshot = draft_session.snapshot_from_draft(draft)
    short_league = League(
        config=league.config.with_(rounds=2), managers=league.managers
    )

    restored = draft_session.restore(snapshot, short_league, pool)
    assert restored.replayed == 8  # 4 teams x 2 rounds
    assert restored.warnings and "dropped" in restored.warnings[0]


# ─────────────────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point the whole database layer at a throwaway file.

    Without this the autosave writes into the user's real ``fantasy_mock_draft.db``
    while the suite runs, which would leave a fake resumable draft sitting in it.
    """
    from models import database

    database.dispose_engine()
    path = str(tmp_path / "resume.db")
    database.init_db(path)
    yield path
    database.dispose_engine()


def test_the_snapshot_round_trips_through_the_database(temp_db, league_and_pool) -> None:
    league, pool = league_and_pool
    draft = _played(league, pool, 14)

    draft_session.save_snapshot(
        draft_session.snapshot_from_draft(draft, league_id=1, source_id=2)
    )
    loaded = draft_session.load_snapshot()

    assert loaded is not None
    assert loaded.pick_count == 14
    assert loaded.league_id == 1 and loaded.source_id == 2
    assert draft_session.restore(loaded, league, pool).replayed == 14

    draft_session.clear_snapshot()
    assert draft_session.load_snapshot() is None


def test_autosave_keeps_only_the_latest_state(temp_db, league_and_pool) -> None:
    """One slot, not a history — "put me back where I was" has one answer.

    Also the check that a second autosave replaces rather than accumulates: this runs
    on every interaction, so a growing record would be a slow leak in a settings
    table read on every page load.
    """
    league, pool = league_and_pool
    draft = _played(league, pool, 5)
    draft_session.autosave(draft, league_id=1, source_id=2)
    assert draft_session.load_snapshot().pick_count == 5

    simulator = DraftSimulator(draft, _profiles(league))
    for _ in range(3):
        simulator.simulate_pick()
    draft_session.autosave(draft, league_id=1, source_id=2)

    reloaded = draft_session.load_snapshot()
    assert reloaded.pick_count == 8

    from models.database import session_scope
    from models.database import ApplicationSettingRow
    from sqlalchemy import select

    with session_scope(temp_db) as session:
        rows = session.execute(
            select(ApplicationSettingRow).where(
                ApplicationSettingRow.key == draft_session.RESUME_KEY
            )
        ).scalars().all()
    assert len(rows) == 1


def test_the_league_and_board_are_stored_so_a_cold_start_can_reload_them(
    temp_db, league_and_pool
) -> None:
    """The half that makes a refresh transparent rather than merely recoverable.

    Before this, the league and board were saved only if the user happened to press
    *Save to database* on Setup — so the common case, fetch data and draft straight
    away, had a saved pick list pointing at nothing.
    """
    league, pool = league_and_pool
    league_id, source_id = draft_session.persist_inputs(league, pool)
    assert league_id is not None and source_id is not None

    snapshot = draft_session.snapshot_from_draft(
        _played(league, pool, 7), league_id=league_id, source_id=source_id
    )
    revived = draft_session.rehydrate(snapshot)

    assert revived.ok, revived.reason
    assert revived.league.config.team_count == league.config.team_count
    assert len(revived.pool) == len(pool)
    # And the picks replay against the *reloaded* board, which is the real test —
    # player ids have to survive the database round trip for that to work.
    assert draft_session.restore(snapshot, revived.league, revived.pool).replayed == 7


def test_a_snapshot_pointing_at_a_deleted_league_says_so(temp_db, league_and_pool) -> None:
    """A missing league is reported, not raised: the page still has to render."""
    league, pool = league_and_pool
    snapshot = draft_session.snapshot_from_draft(
        _played(league, pool, 3), league_id=999, source_id=999
    )
    revived = draft_session.rehydrate(snapshot)
    assert not revived.ok
    assert "no longer in the database" in revived.reason


def test_a_snapshot_with_no_league_id_is_honest_about_it(temp_db) -> None:
    snapshot = DraftSnapshot(picks=[SnapshotPick(overall_pick=1, player_id="RB1")])
    revived = draft_session.rehydrate(snapshot)
    assert not revived.ok
    assert "does not name a stored league" in revived.reason


def test_a_corrupt_stored_snapshot_is_treated_as_absent(temp_db) -> None:
    """A resume feature that can crash the Draft Room is worse than no resume."""
    from services.repository import write_setting

    write_setting(draft_session.RESUME_KEY, {"picks": "not a list at all"})
    assert draft_session.load_snapshot() is None


def test_describe_age_reads_as_english(league_and_pool) -> None:
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)

    def at(hours: float) -> str:
        return draft_session.describe_age(
            DraftSnapshot(saved_at=(now - timedelta(hours=hours)).isoformat()), now=now
        )

    assert at(0.2) == "saved 12 min ago"
    assert at(5) == "saved 5 hours ago"
    assert at(72) == "saved 3.0 days ago"
    assert draft_session.describe_age(DraftSnapshot()) == "saved at an unknown time"
