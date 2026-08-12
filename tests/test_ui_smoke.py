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

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from engine.draft_state import DraftState
from engine.features import annotate_history
from engine.opponent_model import build_profiles
from engine.simulator import DraftSimulator

PAGES = Path(__file__).resolve().parents[1] / "ui" / "pages"

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


def test_draft_room_start_form_renders_without_a_draft(loaded) -> None:
    app = _app("4_draft_room.py", loaded).run()
    _assert_clean(app, "4_draft_room.py (no draft)")
    assert any("Start draft" in button.label for button in app.button)


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
def test_page_blocks_cleanly_with_nothing_loaded(page: str) -> None:
    """The empty state is the first thing a new user sees on every page they open out
    of order, so it has to be a sentence rather than a traceback."""
    from core.config import SimulationConfig
    from models.draft import DraftHistory
    from ui import state as ui_state

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
