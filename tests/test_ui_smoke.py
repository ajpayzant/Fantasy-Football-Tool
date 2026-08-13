"""Smoke tests for the Streamlit UI.

These run each page through Streamlit's own test harness with real engine objects in
session state. They are not a substitute for looking at the app, and they assert very
little about layout — what they catch is the failure mode this UI is most prone to:
a page calling an engine attribute that does not exist, or exists with a different
shape. That is a runtime error in Streamlit, invisible until someone clicks the tab.

The pages are exercised with the synthetic bundle from
``tests/fixtures/sample_league`` — deliberately, not for want of real data. It is the
only dataset that comes with three seasons of *draft history*, which is what the
profile, draft-room and analysis pages actually render; the recorded live payloads
(used in ``test_live_providers.py`` and ``test_setup_page.py``) are a player board
with no drafts behind it. Both enter the app through the same accessors, so a page
that works against one works against the other.

The Setup page is the exception and is covered separately in ``test_setup_page.py``,
against the recorded real payloads, because it is where live data enters and the
thing worth asserting there is that no route to this fixture exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from engine.draft_state import DraftState
from engine.features import annotate_history
from engine.opponent_model import build_profiles
from engine.simulator import DraftSimulator

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "ui" / "pages"

# Every page this module actually runs, in one place so the coverage test below can
# check it against what ``app.py`` registers. A page in the app but not in here is a
# page whose imports nobody checks until a user clicks the tab.
COVERED_PAGES = {
    "1_setup.py", "2_player_pool.py", "3_manager_profiles.py", "4_draft_room.py",
    "5_simulations.py", "6_analysis.py", "7_settings.py",
}

# Generous: a page that builds profiles or runs rollouts does real work, and a
# timeout here would be a flaky failure rather than a finding.
TIMEOUT = 120


@pytest.fixture(scope="module")
def loaded():
    """League, pool, annotated history and profiles — built once, reused by every page."""
    from tests.fixtures.sample_league import sample_bundle

    league, pool, history = sample_bundle()
    pool.apply_league(league.config)
    annotate_history(history, pool=pool, roster=league.config.roster)
    profiles = build_profiles(league, history, pool=pool, annotate=False)
    return league, pool, history, profiles


def _app(page: str, loaded, *, with_draft: int = 0) -> AppTest:
    """A page primed with session state, as if the user had come via Setup.

    ``with_draft`` advances a real draft that many picks first, so pages that depend
    on a draft in progress are exercised against a board with history on it rather
    than an empty one.
    """
    from core.config import SimulationConfig
    from ui import state as ui_state

    league, pool, history, profiles = loaded
    app = AppTest.from_file(str(PAGES / page), default_timeout=TIMEOUT)
    app.session_state[ui_state.K_INITIALISED] = True
    app.session_state[ui_state.K_LEAGUE] = league
    app.session_state[ui_state.K_POOL] = pool
    app.session_state[ui_state.K_HISTORY] = history
    app.session_state[ui_state.K_PROFILES] = profiles
    app.session_state[ui_state.K_SETTINGS] = SimulationConfig()
    app.session_state[ui_state.K_IS_SAMPLE] = True
    app.session_state[ui_state.K_PROVENANCE] = ui_state.Provenance(
        pool_source="sample", history_source="sample", league_source="sample",
        is_sample=True,
    )

    draft = None
    if with_draft:
        draft = DraftState(league, pool, seed=11)
        simulator = DraftSimulator(draft, profiles)
        for _ in range(with_draft):
            if simulator.simulate_pick() is None:
                break
    app.session_state[ui_state.K_DRAFT] = draft
    return app


def _assert_clean(app: AppTest, page: str) -> None:
    """No unhandled exception rendered anywhere on the page."""
    assert not app.exception, (
        f"{page} raised: " + " | ".join(str(e.value) for e in app.exception)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Coverage: no page in the app may go unrun here
# ─────────────────────────────────────────────────────────────────────────────
def test_every_registered_page_is_covered_by_these_tests() -> None:
    """The pages ``app.py`` registers must all be run by this module.

    A page that nobody runs here is a page whose import list is unverified, and a bad
    import in Streamlit is invisible until someone clicks the tab — the exact failure
    these tests exist to catch. ``app.py`` is read rather than imported because it calls
    ``main()`` at module scope, and every page file on disk is checked too, so an
    orphaned page cannot hide either.
    """
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    registered = set(re.findall(r'"ui/pages/([^"]+\.py)"', source))
    assert registered, "no pages found in app.py — has _PAGE_FILES been renamed?"

    on_disk = {
        path.name for path in PAGES.glob("*.py") if not path.name.startswith("_")
    }
    assert registered <= COVERED_PAGES, (
        "registered in app.py but never run by these tests: "
        + ", ".join(sorted(registered - COVERED_PAGES))
    )
    assert on_disk <= COVERED_PAGES, (
        "page files that no test runs: " + ", ".join(sorted(on_disk - COVERED_PAGES))
    )
    assert COVERED_PAGES <= on_disk, (
        "listed as covered but no such file: "
        + ", ".join(sorted(COVERED_PAGES - on_disk))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Every page renders with data loaded
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("page", [
    "1_setup.py",
    "2_player_pool.py",
    "3_manager_profiles.py",
    "5_simulations.py",
    "6_analysis.py",
    "7_settings.py",
])
def test_page_renders_with_data_loaded(page: str, loaded) -> None:
    app = _app(page, loaded, with_draft=30).run()
    _assert_clean(app, page)


def test_draft_room_renders_and_recommends(loaded) -> None:
    """The draft room with a live board is the heaviest page: it runs rollouts and
    renders one card per recommendation lens, all against engine objects."""
    app = _app("4_draft_room.py", loaded, with_draft=30).run()
    _assert_clean(app, "4_draft_room.py")
    # A recommendation card names its lens in a markdown block; the availability
    # expander is only built when a report came back.
    text = " ".join(block.value for block in app.markdown)
    assert "Best Overall" in text or "Best Roster Fit" in text, (
        "no recommendation lens rendered"
    )


def test_the_board_is_above_the_recommendations(loaded) -> None:
    """Layout, asserted because it is the whole point of the redesign.

    The board and the shortlist have to render *before* the lenses: a drafter reads
    what has gone, then what is left, then what the model thinks. Position is checked
    by index rather than by eye so a later edit that moves a section back to the bottom
    fails here instead of silently regressing.
    """
    app = _app("4_draft_room.py", loaded, with_draft=30).run()
    _assert_clean(app, "4_draft_room.py")
    headings = [block.value for block in app.subheader]
    assert "The board" in headings, headings
    assert "Best players left" in headings, headings
    assert headings.index("The board") < headings.index("Best players left")
    assert headings.index("Best players left") < headings.index(
        "What to do with this pick"
    )


def test_every_board_view_renders(loaded) -> None:
    """All four tabs, including the grid's Styler, which only fails when rendered."""
    from ui import board_views

    league, pool, _history, profiles = loaded
    draft = DraftState(league, pool, seed=11)
    simulator = DraftSimulator(draft, profiles)
    for _ in range(30):
        simulator.simulate_pick()

    order = board_views.draft_order_frame(draft)
    assert len(order) == 30
    # Draft order by default: 1.01 at the top, read downwards. Asserted rather than
    # eyeballed because the earlier default was the reverse and it is an easy regression.
    assert order.iloc[0]["Overall"] == 1
    assert order.iloc[0]["Pick"] == "1.01"
    assert order.iloc[-1]["Overall"] == 30
    assert list(order["Overall"]) == sorted(order["Overall"])
    newest = board_views.draft_order_frame(draft, newest_first=True)
    assert newest.iloc[0]["Overall"] == 30

    grid, marks = board_views.snake_grid(draft, league)
    assert len(grid.columns) == league.config.team_count
    assert len(grid) == league.config.rounds
    # 30 picks in a 12-team league fills two rounds and six of the third.
    assert (marks.loc["R1"] != "").all()
    assert (marks.loc["R2"] != "").all()
    assert "__clock__" in marks.to_numpy().ravel().tolist()

    shape = board_views.roster_shape_frame(draft, league)
    assert len(shape) == league.config.team_count
    assert shape["Picks"].sum() == 30


def test_every_grid_cell_sets_a_text_colour(loaded) -> None:
    """Background without foreground is invisible under a dark browser theme.

    The app ships no theme, so Streamlit follows the browser's and renders table text
    white in dark mode. A cell style that sets only ``background-color`` then puts white
    text on a pale background. Both halves, on every cell, or the grid is unreadable for
    half the users.
    """
    from ui import board_views

    league, pool, _history, profiles = loaded
    draft = DraftState(league, pool, seed=11)
    simulator = DraftSimulator(draft, profiles)
    for _ in range(30):
        simulator.simulate_pick()

    _grid, marks = board_views.snake_grid(draft, league)
    styles = board_views.grid_styles(marks)
    cells = styles.to_numpy().ravel().tolist()
    assert cells, "no cells to style"
    for style in cells:
        assert "background-color:" in style, style
        assert "color:" in style.replace("background-color:", ""), style


def test_the_shortlist_honours_the_users_board(loaded) -> None:
    """A do-not-draft player must not appear in the best-players-left table."""
    from services.user_board import UserBoard
    from ui import board_views

    league, pool, _history, profiles = loaded
    draft = DraftState(league, pool, seed=11)
    available = draft.available_players(limit=5)
    banned = available[0].name

    frame = board_views.top_remaining_frame(
        draft, UserBoard(avoid=[banned], targets=[available[3].name]),
        count=10, order="My board",
    )
    assert banned not in set(frame["Player"])
    # The target is promoted above the three players ADP puts ahead of them.
    assert frame.iloc[0]["Player"] == available[3].name
    assert frame.iloc[0]["Mine"] == "🎯 1"


def test_draft_room_start_form_renders_without_a_draft(loaded) -> None:
    app = _app("4_draft_room.py", loaded).run()
    _assert_clean(app, "4_draft_room.py (no draft)")
    assert any("Start draft" in button.label for button in app.button)


@pytest.mark.parametrize("column_set", ["ADP by platform", "Ranks by platform"])
def test_player_pool_platform_views_render(column_set: str, loaded) -> None:
    """Both per-platform views, which the default render never reaches.

    They select columns by display name, so a rename in the frame breaks them silently
    — the table would simply come out short rather than raising.
    """
    app = _app("2_player_pool.py", loaded).run()
    _assert_clean(app, "2_player_pool.py")
    app.radio[0].set_value(column_set).run()
    _assert_clean(app, f"2_player_pool.py ({column_set})")
    shown = app.dataframe[-1].value if app.dataframe else None
    assert shown is not None
    expected = "Avg ADP" if column_set == "ADP by platform" else "Avg rank"
    assert any(
        expected in list(frame.columns) for frame in
        [d.value for d in app.dataframe if hasattr(d.value, "columns")]
    ), f"{expected} did not reach any table"


@pytest.fixture
def multi_platform(loaded):
    """The bundle with per-platform ADP and ranks on it.

    The synthetic pool carries none — every number came from one source — so a page that
    only ever renders it can drop the per-platform columns without any test noticing.
    The values are written straight onto the pool's players, which is where
    ``to_frame`` reads them from, so the page is exercised through its real path. A deep
    copy because the ``loaded`` fixture is module-scoped and shared.
    """
    import copy

    league, pool, history, profiles = loaded
    pool = copy.deepcopy(pool)
    for index, player in enumerate(pool):
        adp = float(index + 1)
        player.ffc_adp = adp
        player.espn_adp = adp + 2.0
        # Yahoo lists a short board, like the real one: every third player only. This is
        # what makes the average worth taking over "sources that have him" rather than
        # over all of them, and what the fully-covered filter has to notice.
        player.yahoo_adp = adp - 1.0 if index % 3 == 0 else None
        player.espn_rank = index + 1
        player.sleeper_rank = index + 5
    return league, pool, history, profiles


def _pool_app(multi_platform) -> AppTest:
    app = _app("2_player_pool.py", multi_platform)
    from ui import state as ui_state

    app.session_state[ui_state.K_POOL] = multi_platform[1]
    return app


def _pool_tables(app: AppTest) -> list:
    return [d.value for d in app.dataframe if hasattr(d.value, "columns")]


def test_every_platforms_own_adp_is_on_the_default_table(multi_platform) -> None:
    """The per-platform columns must be on the view the user lands on.

    They used to exist only under a "Columns" radio in the corner of the filter row,
    which is why the answer to "what does each site say" looked missing.
    """
    app = _pool_app(multi_platform).run()
    _assert_clean(app, "2_player_pool.py (multi-platform)")

    headers = {column for frame in _pool_tables(app) for column in frame.columns}
    for expected in ("FFC ADP", "ESPN ADP", "Yahoo ADP", "Avg ADP", "ADP"):
        assert expected in headers, f"{expected} is not on the default table"


def test_deselecting_a_platform_drops_its_column_and_leaves_the_average_alone(
    multi_platform,
) -> None:
    """The picker drives the columns *and* the average, which is the point of it."""
    app = _pool_app(multi_platform).run()
    assert "Yahoo" in app.multiselect(key="pool_platforms").value

    app.multiselect(key="pool_platforms").set_value(["FFC", "ESPN"]).run()
    _assert_clean(app, "2_player_pool.py (FFC + ESPN only)")

    headers = {column for frame in _pool_tables(app) for column in frame.columns}
    assert "Yahoo ADP" not in headers, "a deselected platform kept its column"
    assert "FFC ADP" in headers
    assert "ESPN ADP" in headers

    # FFC is index+1 and ESPN is index+3, so their mean is index+2 — computed from the
    # two selected sources only, with Yahoo's index no longer pulling it down.
    table = next(f for f in _pool_tables(app) if "Avg ADP" in f.columns)
    row = table.iloc[0]
    assert abs(float(row["Avg ADP"]) - (float(row["FFC ADP"]) + float(row["ESPN ADP"])) / 2) < 0.05


def test_the_fully_covered_filter_drops_players_a_platform_never_listed(
    multi_platform,
) -> None:
    """Yahoo has only every third player, so requiring all three has to cut the rest."""
    app = _pool_app(multi_platform).run()
    before = next(f for f in _pool_tables(app) if "Yahoo ADP" in f.columns)
    assert before["Yahoo ADP"].isna().any(), "the fixture should have gaps to filter on"

    app.checkbox(key="pool_complete_only").set_value(True).run()
    _assert_clean(app, "2_player_pool.py (fully covered)")

    after = next(f for f in _pool_tables(app) if "Yahoo ADP" in f.columns)
    assert not after["Yahoo ADP"].isna().any(), "a player with no Yahoo ADP survived"
    assert len(after) < len(before)


def test_the_room_can_be_built_by_hand(loaded) -> None:
    """The from-scratch route: a declared archetype has to reach the built profile."""
    from core.enums import Archetype

    app = _app("3_manager_profiles.py", loaded).run()
    _assert_clean(app, "3_manager_profiles.py")
    league = app.session_state["league"]
    slots = sorted(m.draft_slot for m in league.managers)
    target = slots[1]

    editor = app.session_state["room_editor"]
    assert editor is not None, "the room editor did not register in session state"

    # data_editor edits arrive as a patch dict keyed by row index, which is how a real
    # click reaches the page — building the frame directly would skip the page's own
    # reading of it.
    app.session_state["room_editor"] = {
        "edited_rows": {
            slots.index(target): {"How they draft": str(Archetype.ZERO_RB)},
        },
        "added_rows": [], "deleted_rows": [],
    }
    app = app.run()
    _assert_clean(app, "3_manager_profiles.py (edited room)")


# ─────────────────────────────────────────────────────────────────────────────
# Gating: a page with nothing loaded must explain itself, not crash
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("page", [
    "2_player_pool.py",
    "3_manager_profiles.py",
    "4_draft_room.py",
    "5_simulations.py",
    "6_analysis.py",
])
def test_page_blocks_cleanly_with_nothing_loaded(
    page: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The empty state is the first thing a new user sees on every page they open out
    of order, so it has to be a sentence rather than a traceback.

    ``load_snapshot`` is stubbed out because it reads the real autosave from the real
    database. With a draft saved there — which is the normal state of a machine the app
    has been used on — the Draft Room correctly rehydrates its league and board, the
    gate never fires, and this test fails for a reason that has nothing to do with the
    empty state. "Nothing loaded" has to mean nothing on disk either.
    """
    from core.config import SimulationConfig
    from models.draft import DraftHistory
    from services import draft_session
    from ui import state as ui_state

    monkeypatch.setattr(draft_session, "load_snapshot", lambda: None)

    app = AppTest.from_file(str(PAGES / page), default_timeout=TIMEOUT)
    app.session_state[ui_state.K_INITIALISED] = True
    app.session_state[ui_state.K_LEAGUE] = None
    app.session_state[ui_state.K_POOL] = None
    app.session_state[ui_state.K_HISTORY] = DraftHistory()
    app.session_state[ui_state.K_PROFILES] = {}
    app.session_state[ui_state.K_DRAFT] = None
    app.session_state[ui_state.K_SETTINGS] = SimulationConfig()
    app.session_state[ui_state.K_IS_SAMPLE] = False
    app.run()
    _assert_clean(app, f"{page} (empty)")
    messages = " ".join(block.value for block in app.info)
    assert "Setup" in messages, "the empty state should name the page to go to"


def test_settings_page_works_with_nothing_loaded() -> None:
    """Settings deliberately does not gate on data — the weights are editable before
    anything is imported."""
    from core.config import SimulationConfig
    from models.draft import DraftHistory
    from ui import state as ui_state

    app = AppTest.from_file(str(PAGES / "7_settings.py"), default_timeout=TIMEOUT)
    app.session_state[ui_state.K_INITIALISED] = True
    app.session_state[ui_state.K_HISTORY] = DraftHistory()
    app.session_state[ui_state.K_SETTINGS] = SimulationConfig()
    app.run()
    _assert_clean(app, "7_settings.py (empty)")
    assert any("Apply settings" in button.label for button in app.button)


# ─────────────────────────────────────────────────────────────────────────────
# The sample-data label really does appear on every page
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("page", [
    "1_setup.py",
    "2_player_pool.py",
    "3_manager_profiles.py",
    "4_draft_room.py",
    "5_simulations.py",
    "6_analysis.py",
])
def test_sample_data_is_labelled_on_every_page(page: str, loaded) -> None:
    """A product requirement, and the reason the banner is per-page rather than shown
    once at startup: a user who lands mid-session has not seen the startup notice."""
    app = _app(page, loaded, with_draft=30).run()
    _assert_clean(app, page)
    warnings = " ".join(block.value for block in app.warning)
    assert "SAMPLE" in warnings.upper(), f"{page} did not label the sample data"


# ─────────────────────────────────────────────────────────────────────────────
# Manager Profiles: the user can state what they know, and it reaches the model
# ─────────────────────────────────────────────────────────────────────────────
# The complaint these answer: the Build button only ever produced the canned
# archetypes, with no way to describe an actual leaguemate. The mechanism already
# existed in the engine (``ManagerPreferences`` → ``ProvenanceKind.USER_ENTERED``);
# what follows pins that the page now reaches it, and that a rebuild does not wipe it.
def _profiles_app(loaded) -> AppTest:
    return _app("3_manager_profiles.py", loaded).run()


def _first_slot(loaded) -> int:
    _, _, _, profiles = loaded
    return sorted(profiles)[0]


def test_stating_a_tendency_lands_on_the_league_and_the_profile(loaded) -> None:
    """A slider the user moves must change the parameter the simulator reads."""
    from core.enums import ProvenanceKind
    from ui import state as ui_state

    app = _profiles_app(loaded)
    _assert_clean(app, "3_manager_profiles.py")
    slot = _first_slot(loaded)

    app.checkbox(key=f"pref_on_{slot}_risk_tolerance").check().run()
    app.slider(key=f"pref_val_{slot}_risk_tolerance").set_value(0.95).run()
    _button = next(b for b in app.button if "Save what I know" in str(b.label))
    _button.click().run()
    _assert_clean(app, "3_manager_profiles.py (after save)")

    league = app.session_state[ui_state.K_LEAGUE]
    manager = league.manager_by_slot(slot)
    assert manager.preferences.risk_tolerance == pytest.approx(0.95)

    # And it reached the rebuilt profile, labelled as the user's statement rather
    # than as something observed.
    profile = app.session_state[ui_state.K_PROFILES][slot]
    assert profile.provenance("risk_preference") is ProvenanceKind.USER_ENTERED
    assert profile.get("risk_preference") > 0.5


def test_an_untouched_slider_states_nothing(loaded) -> None:
    """Saving without ticking anything must leave the model exactly as it was.

    The trap here is a slider defaulting to 0.5 and being saved as "this manager is
    average" — an assertion the user never made, indistinguishable to the engine
    from one they did.
    """
    from ui import state as ui_state

    app = _profiles_app(loaded)
    slot = _first_slot(loaded)
    before = app.session_state[ui_state.K_PROFILES][slot].get("risk_preference")

    next(b for b in app.button if "Save what I know" in str(b.label)).click().run()
    _assert_clean(app, "3_manager_profiles.py (empty save)")

    manager = app.session_state[ui_state.K_LEAGUE].manager_by_slot(slot)
    assert manager.preferences.risk_tolerance is None
    assert manager.preferences.rookie_preference is None
    after = app.session_state[ui_state.K_PROFILES][slot].get("risk_preference")
    assert after == pytest.approx(before)


def test_a_stated_strategy_overrides_the_inferred_archetype(loaded) -> None:
    """The user knowing how someone drafts has to beat the estimator's guess."""
    from core.enums import Archetype
    from ui import state as ui_state

    app = _profiles_app(loaded)
    slot = _first_slot(loaded)
    inferred = app.session_state[ui_state.K_PROFILES][slot].archetype

    target = next(
        a for a in Archetype if a is not inferred and a is not Archetype.BALANCED
    )
    strategy = next(
        box for box in app.selectbox
        if "(let the model infer it)" in [str(o) for o in box.options]
    )
    strategy.set_value(str(target)).run()
    next(b for b in app.button if "Save what I know" in str(b.label)).click().run()
    _assert_clean(app, "3_manager_profiles.py (strategy)")

    assert app.session_state[ui_state.K_PROFILES][slot].archetype is target


def test_stated_preferences_survive_pressing_build_profiles_again(loaded) -> None:
    """The whole point of storing these on the league: a rebuild must not wipe them.

    If they lived on the profile instead, "Build profiles" would silently discard
    everything the user had entered — and the page would look normal afterwards.
    """
    from core.enums import ProvenanceKind
    from ui import state as ui_state

    app = _profiles_app(loaded)
    slot = _first_slot(loaded)
    app.checkbox(key=f"pref_on_{slot}_rookie_preference").check().run()
    app.slider(key=f"pref_val_{slot}_rookie_preference").set_value(0.9).run()
    next(b for b in app.button if "Save what I know" in str(b.label)).click().run()
    _assert_clean(app, "3_manager_profiles.py (before rebuild)")

    next(b for b in app.button if str(b.label) == "Build profiles").click().run()
    _assert_clean(app, "3_manager_profiles.py (after rebuild)")

    manager = app.session_state[ui_state.K_LEAGUE].manager_by_slot(slot)
    assert manager.preferences.rookie_preference == pytest.approx(0.9)
    profile = app.session_state[ui_state.K_PROFILES][slot]
    assert profile.provenance("rookie_rate") is ProvenanceKind.USER_ENTERED


def test_a_player_name_that_matches_nobody_is_reported_back(loaded) -> None:
    """A typo must not silently do nothing — that is indistinguishable from working."""
    app = _profiles_app(loaded)
    boxes = [
        box for box in app.text_area
        if "always want" in str(box.label) or "will not touch" in str(box.label)
    ]
    assert boxes, "the named-player inputs are missing"
    boxes[0].set_value("Nobody Whatsoever").run()
    next(b for b in app.button if "Save what I know" in str(b.label)).click().run()
    _assert_clean(app, "3_manager_profiles.py (bad name)")

    shown = " ".join(str(b.value) for b in app.warning)
    assert "Nobody Whatsoever" in shown, shown


def test_forgetting_what_you_told_it_restores_the_model(loaded) -> None:
    from ui import state as ui_state

    app = _profiles_app(loaded)
    slot = _first_slot(loaded)
    app.checkbox(key=f"pref_on_{slot}_predictability").check().run()
    app.slider(key=f"pref_val_{slot}_predictability").set_value(0.05).run()
    next(b for b in app.button if "Save what I know" in str(b.label)).click().run()
    assert app.session_state[ui_state.K_LEAGUE].manager_by_slot(
        slot
    ).preferences.has_any

    next(b for b in app.button if "Forget what I told you" in str(b.label)).click().run()
    _assert_clean(app, "3_manager_profiles.py (forget)")
    assert not app.session_state[ui_state.K_LEAGUE].manager_by_slot(
        slot
    ).preferences.has_any
