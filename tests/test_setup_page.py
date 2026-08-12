"""Headless tests for the Setup page, the one route all data enters through.

Run through Streamlit's own harness, so these exercise the page as rendered rather
than the functions behind it. What they are for:

* **Nothing on the page can load fictional players.** That is a product
  requirement, and the kind that rots quietly — a helpful default or a re-added
  dropdown would break it without breaking anything else.
* **A live fetch actually seats a working league.** The claim "one click and you can
  draft" is either true end to end or it is marketing.
* **Refreshing the board does not discard a connected league.** The bug this guards
  is silent: the user's twelve real managers are replaced by generic ones and the
  page still looks fine.

The live fetch is stubbed with a *board built from the recorded payloads* rather
than a hand-made frame, so what the page receives is shaped exactly like what the
providers really return. No test here touches the network.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGES = Path(__file__).resolve().parents[1] / "ui" / "pages"
SETUP = str(PAGES / "1_setup.py")
PAYLOAD_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "live_payloads")
MANIFEST = os.path.join(PAYLOAD_DIR, "manifest.json")

TIMEOUT = 120

requires_payloads = pytest.mark.skipif(
    not os.path.exists(MANIFEST),
    reason="recorded payloads missing — run scripts/record_live_fixtures.py",
)


def _app() -> AppTest:
    """The Setup page as a first-time visitor sees it: nothing loaded."""
    from core.config import SimulationConfig
    from models.draft import DraftHistory
    from ui import state as ui_state

    app = AppTest.from_file(SETUP, default_timeout=TIMEOUT)
    app.session_state[ui_state.K_INITIALISED] = True
    app.session_state[ui_state.K_LEAGUE] = None
    app.session_state[ui_state.K_POOL] = None
    app.session_state[ui_state.K_HISTORY] = DraftHistory()
    app.session_state[ui_state.K_PROFILES] = {}
    app.session_state[ui_state.K_DRAFT] = None
    app.session_state[ui_state.K_SETTINGS] = SimulationConfig()
    app.session_state[ui_state.K_IS_SAMPLE] = False
    return app


def _text(app: AppTest) -> str:
    """Everything the page rendered as text, for content assertions."""
    parts: list[str] = []
    for collection in (
        app.markdown, app.caption, app.info, app.warning, app.error, app.success
    ):
        parts.extend(str(block.value) for block in collection)
    parts.extend(str(block.label) for block in app.button)
    return " ".join(parts)


def _assert_clean(app: AppTest) -> None:
    assert not app.exception, " | ".join(str(e.value) for e in app.exception)


# ─────────────────────────────────────────────────────────────────────────────
# The page renders, and offers no route to fictional data
# ─────────────────────────────────────────────────────────────────────────────
def test_setup_renders_with_nothing_loaded() -> None:
    """The first thing a new user sees must render without data behind it."""
    app = _app().run()
    _assert_clean(app)
    assert any("Fetch current player data" in b.label for b in app.button)


def test_setup_offers_no_sample_data_route() -> None:
    """No button, caption or adapter on this page can load generated players.

    Checked against the rendered page rather than the source, because the route
    that matters is the one a user can click.
    """
    app = _app().run()
    _assert_clean(app)
    lowered = _text(app).lower()
    for phrase in ("sample league", "sample data", "fictional", "demo data"):
        assert phrase not in lowered, f"Setup still offers '{phrase}'"

    from services.adapters import available_adapters

    assert "sample" not in available_adapters()


def test_every_scoring_format_is_offered() -> None:
    """The user picks the format, because ADP differs materially between them."""
    from core.enums import ScoringPreset

    app = _app().run()
    _assert_clean(app)
    scoring = app.selectbox(key="live_scoring")
    assert len(scoring.options) == len(list(ScoringPreset))


def test_espn_can_be_switched_off() -> None:
    """ESPN's 39 MB unfilterable payload can fail locally, so it must be optional."""
    app = _app().run()
    _assert_clean(app)
    assert app.checkbox(key="live_use_espn").value is True
    app.checkbox(key="live_use_espn").uncheck().run()
    _assert_clean(app)


# ─────────────────────────────────────────────────────────────────────────────
# A live fetch seats a real, drafting-ready league
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def recorded_board(monkeypatch: pytest.MonkeyPatch):
    """Stub ``build_live_board`` with a board resolved from the recorded payloads.

    Patched on :mod:`services.live` only, which is enough because ``AppTest``
    re-executes the page module on every run: the page's
    ``from services.live import build_live_board`` therefore resolves against the
    already-patched attribute rather than holding a stale reference.

    The stub does the real work — four providers, the real resolver, the real
    importer — against recorded payloads. Returning a hand-made frame instead would
    let the page pass while the join it depends on was broken.
    """
    if not os.path.exists(MANIFEST):
        pytest.skip("recorded payloads missing")

    import json

    from core.enums import Platform
    from services import live as live_module
    from services.importers import import_player_pool
    from services.providers import base as provider_base
    from services.providers.resolver import board_to_import_frame, resolve_board

    with open(MANIFEST, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    monkeypatch.setattr(provider_base, "cache_directory", lambda: PAYLOAD_DIR)
    monkeypatch.setattr(provider_base, "write_cache", lambda *a, **k: None)

    calls: list[dict] = []

    def _offline_board(**kwargs) -> live_module.LiveBoardResult:
        from services.providers import (
            ESPNProvider,
            FFCalculatorProvider,
            SleeperProvider,
            YahooProvider,
        )

        calls.append(dict(kwargs))
        season = int(manifest["season"])
        resolved = resolve_board(
            sleeper=SleeperProvider().fetch(ttl_seconds=-1),
            ffc=FFCalculatorProvider().fetch(
                scoring=manifest["scoring"], team_count=int(manifest["team_count"]),
                season=season, ttl_seconds=-1,
            ),
            espn=(
                ESPNProvider().fetch(season=season, scoring=manifest["scoring"], ttl_seconds=-1)
                if kwargs.get("use_espn", True) else None
            ),
            yahoo=(
                YahooProvider().fetch(player_limit=300, ttl_seconds=-1)
                if kwargs.get("use_yahoo", True) else None
            ),
            season=season,
            team_count=int(kwargs.get("team_count") or manifest["team_count"]),
            scoring_format=str(kwargs.get("scoring") or manifest["scoring"]),
        )
        imported = import_player_pool(
            board_to_import_frame(resolved.frame),
            league=kwargs.get("league"),
            source="live: recorded fixtures",
            season=season,
            platform=Platform.CUSTOM,
            is_sample_data=False,
        )
        result = live_module.LiveBoardResult(
            pool=imported.pool, board=resolved, season=season,
            scoring_format=str(kwargs.get("scoring") or manifest["scoring"]),
            source_status=resolved.source_status,
        )
        result.report.extend(resolved.report)
        result.report.extend(imported.report)
        return result

    monkeypatch.setattr(live_module, "build_live_board", _offline_board)
    # ``current_season`` reads Sleeper's state endpoint; served from the fixture.
    monkeypatch.setattr(live_module, "current_season", lambda: int(manifest["season"]))
    return calls


def _fetch(app: AppTest) -> AppTest:
    """Click the fetch button and run to completion."""
    app.button(key="fetch_live").click().run()
    return app


@requires_payloads
def test_fetching_loads_real_players_and_a_draftable_league(recorded_board) -> None:
    """One click, end to end: real players on the board and a league seated."""
    from ui import state as ui_state

    app = _fetch(_app().run())
    _assert_clean(app)

    pool = app.session_state[ui_state.K_POOL]
    league = app.session_state[ui_state.K_LEAGUE]
    assert pool is not None and len(pool) > 200
    assert pool.metadata.is_sample_data is False
    assert app.session_state[ui_state.K_IS_SAMPLE] is False
    assert league is not None
    assert len(league.managers) == league.config.team_count
    # Exactly one seat is the user's, or the draft cannot know whose turn it is.
    assert sum(1 for m in league.managers if m.is_user) == 1


@requires_payloads
def test_fetched_players_have_real_names(recorded_board) -> None:
    """The specific thing that was wrong before: generated names on the board.

    The generator produced names like "WR 2025-14". A real board's names contain no
    digits, which is a cheap and total discriminator.
    """
    from ui import state as ui_state

    app = _fetch(_app().run())
    pool = app.session_state[ui_state.K_POOL]
    names = [player.name for player in pool.players[:60]]
    assert not any(any(ch.isdigit() for ch in name) for name in names), names


@requires_payloads
def test_generic_opponents_are_labelled_by_tendency_not_named(recorded_board) -> None:
    """An opponent the model knows nothing about must not look like a real person.

    The archetype fallback names each seat for its slot and tendency. If a future
    change swaps in invented human names, this fails.
    """
    from ui import state as ui_state

    app = _fetch(_app().run())
    league = app.session_state[ui_state.K_LEAGUE]
    opponents = [m for m in league.managers if not m.is_user]
    assert opponents
    for manager in opponents:
        assert manager.name.startswith("Slot "), manager.name
        assert "tendency" in manager.name, manager.name


@requires_payloads
def test_the_page_says_the_opponents_are_generic(recorded_board) -> None:
    """Provenance the user can read, not just a flag in session state."""
    app = _fetch(_app().run())
    _assert_clean(app)
    lowered = _text(app).lower()
    assert "generic" in lowered
    assert "real players" in lowered


@requires_payloads
def test_no_sample_banner_after_a_live_fetch(recorded_board) -> None:
    """Real data must never carry the sample-data warning — it would train the user
    to ignore it."""
    app = _fetch(_app().run())
    warnings = " ".join(str(block.value).upper() for block in app.warning)
    assert "SAMPLE DATA" not in warnings


@requires_payloads
def test_the_selected_scoring_format_reaches_the_fetch(recorded_board) -> None:
    """Picking full PPR must actually change what is requested, not just the label."""
    from core.enums import ScoringPreset

    app = _app().run()
    app.selectbox(key="live_scoring").set_value(ScoringPreset.FULL_PPR).run()
    _fetch(app)
    _assert_clean(app)
    assert recorded_board, "the fetch was never called"
    assert str(recorded_board[-1]["scoring"]) == str(ScoringPreset.FULL_PPR)


@requires_payloads
def test_team_count_and_slot_reach_the_league(recorded_board) -> None:
    """League shape comes from the inputs, not from a default nobody chose."""
    from ui import state as ui_state

    app = _app().run()
    app.number_input(key="live_teams").set_value(10).run()
    app.number_input(key="live_rounds").set_value(16).run()
    app.number_input(key="live_slot").set_value(7).run()
    _fetch(app)
    _assert_clean(app)

    league = app.session_state[ui_state.K_LEAGUE]
    assert league.config.team_count == 10
    assert league.config.rounds == 16
    assert league.config.user_draft_slot == 7
    assert len(league.managers) == 10
    assert recorded_board[-1]["team_count"] == 10


@requires_payloads
def test_unticking_espn_skips_it(recorded_board) -> None:
    """The escape hatch for ESPN's oversized payload has to actually skip ESPN."""
    app = _app().run()
    app.checkbox(key="live_use_espn").uncheck().run()
    _fetch(app)
    _assert_clean(app)
    assert recorded_board[-1]["use_espn"] is False
    # And the board still works without it.
    from ui import state as ui_state

    assert len(app.session_state[ui_state.K_POOL]) > 200


# ─────────────────────────────────────────────────────────────────────────────
# A connected league survives a board refresh
# ─────────────────────────────────────────────────────────────────────────────
@requires_payloads
def test_refreshing_the_board_keeps_a_connected_league(recorded_board) -> None:
    """The silent bug: real managers replaced by generic ones on a data refresh.

    A user who has connected their league and then refreshes ADP would find the
    opponent model quietly reset to archetype priors, with nothing on screen saying
    so — the page would look exactly the same.
    """
    from core.config import LeagueConfig
    from core.enums import Platform
    from models.league import League
    from models.manager import Manager
    from ui import state as ui_state

    real = League(
        config=LeagueConfig(
            name="My Real League", season=2026, platform=Platform.SLEEPER,
            team_count=4, rounds=15, user_draft_slot=2,
        ),
        managers=[
            Manager(name="Alex", draft_slot=1),
            Manager(name="Sam", draft_slot=2, is_user=True),
            Manager(name="Jo", draft_slot=3),
            Manager(name="Kit", draft_slot=4),
        ],
    )
    app = _app()
    app.session_state[ui_state.K_LEAGUE] = real
    app.run()
    _fetch(app)
    _assert_clean(app)

    league = app.session_state[ui_state.K_LEAGUE]
    assert league.config.name == "My Real League"
    assert [m.name for m in league.managers] == ["Alex", "Sam", "Jo", "Kit"]
    # And the new board did land, rather than the guard skipping the whole update.
    assert len(app.session_state[ui_state.K_POOL]) > 200


# ─────────────────────────────────────────────────────────────────────────────
# Scoring: the preset you pick is the scoring you get
# ─────────────────────────────────────────────────────────────────────────────
def test_switching_preset_refills_the_per_event_values() -> None:
    """Picking a preset must not leave the previous preset's numbers behind.

    Streamlit pins a keyed widget to its session-state value and ignores ``value=``
    after the first run, so the per-event inputs in the advanced expander held the old
    figures. Selecting Full PPR then saving produced a league labelled **custom** that
    was still scoring half-PPR receptions — the user got neither preset, and nothing
    on the page said so.
    """
    from core.enums import ScoringPreset
    from ui import state as ui_state

    app = _app().run()
    app.selectbox(key="league_preset").set_value(ScoringPreset.FULL_PPR).run()
    app.button(key="save_league").click().run()
    _assert_clean(app)

    scoring = app.session_state[ui_state.K_LEAGUE].config.scoring
    assert scoring.preset is ScoringPreset.FULL_PPR
    assert scoring.reception == 1.0, (
        f"a full-PPR league is scoring {scoring.reception} per reception"
    )

    app.selectbox(key="league_preset").set_value(ScoringPreset.STANDARD).run()
    app.button(key="save_league").click().run()
    _assert_clean(app)

    scoring = app.session_state[ui_state.K_LEAGUE].config.scoring
    assert scoring.preset is ScoringPreset.STANDARD
    assert scoring.reception == 0.0


def test_editing_one_value_still_makes_the_league_custom() -> None:
    """The other half of the same behaviour: a real edit must survive the reset.

    Without this the fix above could pass by resetting the numbers on every run,
    which would make the advanced editor impossible to use.
    """
    from core.enums import ScoringPreset
    from ui import state as ui_state

    app = _app().run()
    app.number_input(key="scoring_reception").set_value(0.75).run()
    app.button(key="save_league").click().run()
    _assert_clean(app)

    scoring = app.session_state[ui_state.K_LEAGUE].config.scoring
    assert scoring.preset is ScoringPreset.CUSTOM
    assert scoring.reception == 0.75
    # Everything untouched keeps the preset's value rather than a default.
    assert scoring.pass_td == 4.0


# ─────────────────────────────────────────────────────────────────────────────
# Changing scoring rescores the board in place
# ─────────────────────────────────────────────────────────────────────────────
@requires_payloads
def test_changing_scoring_rescores_the_board_without_a_refetch(recorded_board) -> None:
    """Saving a new scoring preset must move the projections, offline.

    The page used to tell the user to press "Get current data" again — a network round
    trip to redo arithmetic on stats the app already had, which also swapped the whole
    board's ADP for whatever ESPN's had drifted to. Asserted through the rendered page
    because the instruction was the visible half of the bug.
    """
    from core.enums import ScoringPreset
    from ui import state as ui_state

    app = _fetch(_app().run())
    _assert_clean(app)
    pool = app.session_state[ui_state.K_POOL]
    before = {p.player_id: float(p.projection) for p in pool}
    fetches = len(recorded_board)

    app.selectbox(key="league_preset").set_value(ScoringPreset.FULL_PPR).run()
    app.button(key="save_league").click().run()
    _assert_clean(app)

    pool = app.session_state[ui_state.K_POOL]
    assert pool.league.scoring.preset is ScoringPreset.FULL_PPR
    moved = [
        p.name for p in pool
        if abs(float(p.projection) - before[p.player_id]) > 1.0
    ]
    assert len(moved) > 100, (
        f"only {len(moved)} projections moved when scoring changed to full PPR; "
        "the board is still scored under the old rules"
    )
    assert len(recorded_board) == fetches, "rescoring must not hit the fetch path"

    text = _text(app)
    assert "Rescored this board" in text
    assert "refetch and rescore" not in text
    assert "Get current data" not in text


# ─────────────────────────────────────────────────────────────────────────────
# Failure is reported, not hidden
# ─────────────────────────────────────────────────────────────────────────────
def test_a_total_fetch_failure_is_explained_on_the_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every source down must render a message naming the fallback, not a traceback."""
    from services import live as live_module
    from services.providers import base as provider_base
    from services.providers.base import FetchOutcome

    # Patched on ``base`` alone, and that is sufficient: every provider imports
    # ``fetch_json``, which resolves ``fetch_bytes`` from its own module globals at
    # call time. The cache is pointed at an empty directory so the stale-cache
    # fallback — which would otherwise serve a real payload and defeat the test —
    # finds nothing.
    monkeypatch.setattr(provider_base, "cache_directory", _empty_cache_dir)
    monkeypatch.setattr(
        provider_base, "fetch_bytes",
        lambda url, **kw: FetchOutcome(None, url, error="simulated outage"),
    )
    monkeypatch.setattr(live_module, "current_season", lambda: 2026)

    app = _fetch(_app().run())
    _assert_clean(app)
    errors = " ".join(str(block.value) for block in app.error).lower()
    assert "no live player board" in errors
    assert "import your own" in errors
    # Nothing half-loaded: a failed fetch must not seat a league with no players.
    from ui import state as ui_state

    assert app.session_state[ui_state.K_POOL] is None


def _empty_cache_dir() -> str:
    """A real, empty cache directory, so no stale payload can be served."""
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "fmd_empty_cache")
    os.makedirs(path, exist_ok=True)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# The draft order can be set by hand
# ─────────────────────────────────────────────────────────────────────────────
def _seated(draft_type=None, order=None, rounds: int = 4):
    """A four-team league with generated opponent labels and the user in slot 2."""
    from core.config import LeagueConfig
    from core.enums import DraftType, Platform
    from models.league import League
    from models.manager import Manager

    config = LeagueConfig(
        name="Order League", season=2026, platform=Platform.CUSTOM,
        team_count=4, rounds=rounds, user_draft_slot=2,
        draft_type=draft_type or DraftType.SNAKE,
        custom_round_order=order or {},
    )
    managers = [
        Manager(name="Slot 1 · Zero-RB tendency", draft_slot=1),
        Manager(name="You", draft_slot=2, is_user=True),
        Manager(name="Slot 3 · Hero-RB tendency", draft_slot=3),
        Manager(name="Dana", draft_slot=4),
    ]
    return League(config=config, managers=managers)


def _order_app(league) -> AppTest:
    from ui import state as ui_state

    app = _app()
    app.session_state[ui_state.K_LEAGUE] = league
    return app.run()


def _button(app: AppTest, label: str):
    """The button with this exact label, so a test names what a user would click."""
    for candidate in app.button:
        if str(candidate.label) == label:
            return candidate
    raise AssertionError(
        f"no button labelled {label!r}; present: {[str(b.label) for b in app.button]}"
    )


def test_drawing_the_order_at_random_reseats_everyone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The common real-league case: the order is drawn, not chosen.

    ``random.shuffle`` is replaced with a reversal so the expected seating is exact
    rather than "some permutation" — a test that only checked for a permutation would
    pass on an implementation that shuffled the names but not the seats.
    """
    from ui import state as ui_state

    monkeypatch.setattr("random.shuffle", lambda seq: seq.reverse())

    app = _order_app(_seated())
    _button(app, "Draw at random").click().run()
    _assert_clean(app)

    league = app.session_state[ui_state.K_LEAGUE]
    by_slot = {m.draft_slot: m.name for m in league.managers}
    assert sorted(by_slot) == [1, 2, 3, 4], "every seat must still be filled once"
    # Reversed: slot 1 → 4, slot 2 (the user) → 3, slot 3 → 2, slot 4 → 1.
    assert by_slot[1] == "Dana"
    assert by_slot[3] == "You"
    # The user's own seat has to follow their name, or the draft asks the wrong team
    # to pick.
    assert league.config.user_draft_slot == 3
    assert sum(1 for m in league.managers if m.is_user) == 1


def test_reseating_renumbers_generated_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A label that names a seat must name the seat it actually occupies.

    "Slot 1 · Zero-RB tendency" moved to seat 4 and still reading "Slot 1" is the
    same class of bug as the old "You" stuck on slot 1 — the tendency half is real
    and is kept, the seat number is not.
    """
    from ui import state as ui_state

    monkeypatch.setattr("random.shuffle", lambda seq: seq.reverse())

    app = _order_app(_seated())
    _button(app, "Draw at random").click().run()
    _assert_clean(app)

    league = app.session_state[ui_state.K_LEAGUE]
    by_slot = {m.draft_slot: m.name for m in league.managers}
    assert by_slot[4] == "Slot 4 · Zero-RB tendency", by_slot
    assert by_slot[2] == "Slot 2 · Hero-RB tendency", by_slot
    # A name the user typed is never rewritten.
    assert by_slot[1] == "Dana"


def test_the_order_preview_uses_the_engines_own_ordering() -> None:
    """The preview has to be the real order, not a re-implementation of snake."""
    from engine.draft_order import round_slot_order

    league = _seated()
    app = _order_app(league)
    _assert_clean(app)

    # Round 2 of a snake reverses, so the preview must lead with slot 4's manager.
    second = round_slot_order(league.config, 2)
    assert second[0] == 4
    frames = [frame.value for frame in app.dataframe]
    assert any(
        "Round 2" in list(getattr(frame, "index", []))
        and frame.iloc[1, 0] == "Dana"
        for frame in frames
        if hasattr(frame, "iloc")
    ), "no preview frame showed round 2 starting with slot 4"


def test_a_custom_draft_type_gets_an_editable_round_by_round_order() -> None:
    """Custom is unusable without an editor, and unsavable without a seeded order.

    ``core.validation`` errors when a custom league has no order for every round, so
    a league arriving here with an empty one must still render an order to edit
    rather than an empty grid or a traceback.
    """
    from core.enums import DraftType

    app = _order_app(_seated(draft_type=DraftType.CUSTOM))
    _assert_clean(app)
    text = _text(app)
    assert "Pick order, round by round" in text
    _button(app, "Apply pick order")


def test_a_custom_order_that_is_kept_is_shown_not_replaced() -> None:
    """A hand-built order must survive a page render untouched."""
    from core.enums import DraftType

    hand_built = {1: [2, 4, 1, 3], 2: [3, 1, 4, 2], 3: [1, 2, 3, 4], 4: [4, 3, 2, 1]}
    league = _seated(draft_type=DraftType.CUSTOM, order=hand_built)
    app = _order_app(league)
    _assert_clean(app)

    from engine.draft_order import round_slot_order

    assert round_slot_order(league.config, 1) == [2, 4, 1, 3]
    frames = [f.value for f in app.dataframe if hasattr(f.value, "iloc")]
    # The preview resolves slots to names: round 1 opens with slot 2, the user.
    assert any(
        "Round 1" in list(getattr(frame, "index", [])) and frame.iloc[0, 0] == "You"
        for frame in frames
    ), "the preview did not reflect the hand-built order"


def test_the_reversal_round_is_editable_for_a_reversal_draft() -> None:
    """Third-round reversal is a setting, not a constant — some leagues flip later."""
    from core.enums import DraftType
    from ui import state as ui_state

    app = _order_app(_seated(draft_type=DraftType.THIRD_ROUND_REVERSAL, rounds=6))
    _assert_clean(app)
    app.number_input(key="reversal_round_input").set_value(4).run()
    _button(app, "Apply reversal round").click().run()
    _assert_clean(app)

    assert app.session_state[ui_state.K_LEAGUE].config.reversal_round == 4


def test_the_draft_order_section_needs_a_league_first() -> None:
    """With nothing seated it must explain itself rather than render broken controls."""
    app = _app().run()
    _assert_clean(app)
    assert "Save league settings above first" in _text(app)


def test_connecting_without_a_league_id_is_a_warning_not_a_fetch() -> None:
    """The empty-input case, which is what a user hits first."""
    app = _app().run()
    app.button(key="connect_sleeper").click().run()
    _assert_clean(app)
    warnings = " ".join(str(block.value) for block in app.warning).lower()
    assert "sleeper league id" in warnings


# ─────────────────────────────────────────────────────────────────────────────
# Pasting a draft board — the ESPN/Yahoo route that replaces the false promise
# ─────────────────────────────────────────────────────────────────────────────
# The complaint these answer: "not sure how league sync works for ESPN and Yahoo".
# The honest answer is that it does not, and the page now says so and offers the
# route that does work. What follows pins both halves — the copy no longer promises
# a connect that does not exist, and the paste route really does seat history.
ESPN_RECAP = """ROUND 1
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


def _paste_app(text: str, *, league=None) -> AppTest:
    """The Setup page with a board pasted in, parsed but not yet imported."""
    from ui import state as ui_state

    app = _app()
    if league is not None:
        app.session_state[ui_state.K_LEAGUE] = league
    app.run()
    app.text_area(key="board_paste").set_value(text).run()
    return app


def test_the_page_no_longer_promises_an_espn_league_id_connect() -> None:
    """The specific false claim, asserted against the rendered page.

    It used to say a public ESPN league "imports with just its league ID". Nothing
    implemented that, so a user following it went looking for a field that does not
    exist and concluded the app was broken.
    """
    app = _app().run()
    _assert_clean(app)
    text = _text(app).lower()
    assert "imports with just its league id" not in text
    # And there is no ESPN league-ID input to type into.
    keys = {
        str(getattr(box, "label", "")).lower()
        for box in app.text_input
    }
    assert not any("espn" in key and "league" in key for key in keys), keys
    # What replaced it is named on the page.
    assert "paste" in text and "recap" in text


def test_pasting_a_recap_shows_what_it_read_before_importing_anything() -> None:
    """Nothing is imported by pasting — the reading is shown first.

    This is the whole safety property of the feature: misreading the layout produces
    picks that look completely normal and belong to the wrong managers, so the user
    has to be able to see the attribution before committing to it.
    """
    from ui import state as ui_state

    app = _paste_app(ESPN_RECAP)
    _assert_clean(app)
    assert not app.session_state[ui_state.K_HISTORY].drafts

    shown = _text(app)
    assert "8 pick" in shown
    assert any("Import these 8 picks" in str(b.label) for b in app.button)


def test_importing_the_pasted_board_seats_the_history() -> None:
    from ui import state as ui_state

    app = _paste_app(ESPN_RECAP)
    _button(app, "Import these 8 picks").click().run()
    _assert_clean(app)

    history = app.session_state[ui_state.K_HISTORY]
    assert len(history.all_picks) == 8
    picks = {p.player_name: p for p in history.all_picks}
    assert picks["Ja'Marr Chase"].overall_pick == 1
    # Round 2 must come from the header, not from the line's position in the file.
    assert picks["Puka Nacua"].round_number == 2
    # And the attribution, which is the part a misread layout gets wrong silently.
    assert picks["Ja'Marr Chase"].manager_name == "Team Alpha"
    assert picks["Jahmyr Gibbs"].manager_name == "Team Alpha"


def test_a_pasted_season_is_kept_separate_from_one_already_loaded() -> None:
    """Two seasons pasted one at a time have to accumulate, not overwrite.

    Several seasons is the difference between a profile that reflects a habit and one
    that reflects a single draft, and pasting them is inherently one at a time.
    """
    from ui import state as ui_state

    app = _paste_app(ESPN_RECAP)
    app.number_input(key="board_season").set_value(2024).run()
    _button(app, "Import these 8 picks").click().run()
    _assert_clean(app)
    assert {d.season for d in app.session_state[ui_state.K_HISTORY].drafts} == {2024}

    app.text_area(key="board_paste").set_value(ESPN_RECAP).run()
    app.number_input(key="board_season").set_value(2025).run()
    _button(app, "Import these 8 picks").click().run()
    _assert_clean(app)
    history = app.session_state[ui_state.K_HISTORY]
    assert {d.season for d in history.drafts} == {2024, 2025}
    assert len(history.all_picks) == 16


def test_a_connected_leagues_manager_names_are_used_for_the_paste() -> None:
    """A recap's team names are snapped onto the league's own where they agree.

    Without this the picks land under a second set of near-identical manager names
    and inform nobody's profile — the single most common reason a profile comes out
    empty, and silent.
    """
    from core.config import LeagueConfig
    from models.league import League
    from models.manager import Manager
    from ui import state as ui_state

    league = League(
        config=LeagueConfig(name="Real League", season=2026, team_count=4, rounds=2),
        managers=[
            Manager(name="team alpha", draft_slot=1, is_user=True),
            Manager(name="Beta Ballers", draft_slot=2),
            Manager(name="Gamma Squad", draft_slot=3),
            Manager(name="Delta Force", draft_slot=4),
        ],
    )
    app = _paste_app(ESPN_RECAP, league=league)
    _button(app, "Import these 8 picks").click().run()
    _assert_clean(app)

    history = app.session_state[ui_state.K_HISTORY]
    known = {m.key for m in league.managers}
    assert {p.manager_key for p in history.all_picks} <= known, (
        "a pasted name did not snap onto the league's own managers"
    )


def test_choosing_the_layout_by_hand_overrides_the_detection() -> None:
    """The escape hatch has to be reachable from the page, not just the parser."""
    app = _paste_app(ESPN_RECAP)
    layout = app.selectbox(key="board_layout")
    assert "Detect automatically" in [str(o) for o in layout.options]
    layout.set_value("pick_list").run()
    _assert_clean(app)
    assert any("Import these 8 picks" in str(b.label) for b in app.button)


def test_prose_pasted_by_mistake_says_so_and_offers_no_import() -> None:
    app = _paste_app("I had a really good draft this year and I think I won it.")
    _assert_clean(app)
    assert not any("Import these" in str(b.label) for b in app.button)
    assert app.error, "an unreadable paste must say so"


def test_lines_that_could_not_be_read_are_surfaced_not_dropped() -> None:
    app = _paste_app(
        "1.01 Team Alpha - Ja'Marr Chase WR CIN\n"
        "1.02 Bijan Robinson\n"
        "1.03 Beta Ballers - CeeDee Lamb WR DAL\n"
    )
    _assert_clean(app)
    assert "could not be read" in _text(app)
    assert any("Import these 2 picks" in str(b.label) for b in app.button)
