"""The home page: does it tell you what to do next, and does the button go there?

The old landing page was accurate and useless — two columns of prose about what the
app does, then five Yes/No metrics. Nothing on it was clickable, so both a new user
and a returning one had to work out for themselves which sidebar entry came next.

These tests are about that one property. They assert the page offers *one* action
suited to how far along the session is, and that clicking it really navigates —
``st.switch_page`` is exercised for real here, so a stale path in ``_PAGE_FILES``
fails a test instead of failing on a user's click.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
TIMEOUT = 120


@pytest.fixture(scope="module")
def loaded():
    from engine.features import annotate_history
    from engine.opponent_model import build_profiles
    from tests.fixtures.sample_league import sample_bundle

    league, pool, history = sample_bundle()
    pool.apply_league(league.config)
    annotate_history(history, pool=pool, roster=league.config.roster)
    profiles = build_profiles(league, history, pool=pool, annotate=False)
    return league, pool, history, profiles


def _home(**session) -> AppTest:
    """The home page with session state primed to a chosen point in the journey."""
    from core.config import SimulationConfig
    from models.draft import DraftHistory
    from ui import state as ui_state

    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.session_state[ui_state.K_INITIALISED] = True
    app.session_state[ui_state.K_LEAGUE] = session.get("league")
    app.session_state[ui_state.K_POOL] = session.get("pool")
    app.session_state[ui_state.K_HISTORY] = session.get("history") or DraftHistory()
    app.session_state[ui_state.K_PROFILES] = session.get("profiles") or {}
    app.session_state[ui_state.K_DRAFT] = session.get("draft")
    app.session_state[ui_state.K_SETTINGS] = SimulationConfig()
    app.session_state[ui_state.K_IS_SAMPLE] = session.get("is_sample", False)
    return app.run()


def _text(app: AppTest) -> str:
    parts: list[str] = []
    for collection in (
        app.markdown, app.caption, app.info, app.warning, app.error, app.success
    ):
        parts.extend(str(block.value) for block in collection)
    return " ".join(parts)


def _labels(app: AppTest) -> list[str]:
    return [str(b.label) for b in app.button]


def _assert_clean(app: AppTest) -> None:
    assert not app.exception, " | ".join(str(e.value) for e in app.exception)


# ─────────────────────────────────────────────────────────────────────────────
# One action, and it is the right one
# ─────────────────────────────────────────────────────────────────────────────
def test_a_first_visit_offers_exactly_one_thing_to_do() -> None:
    """With nothing loaded there is one button, and it is the first real step.

    One and not several: a first-time user presented with a choice between four
    pages has to understand the app before they can start it.
    """
    app = _home()
    _assert_clean(app)
    assert _labels(app) == ["Go to Setup"]
    assert "Get current player data" in _text(app)


def test_the_first_button_actually_navigates_to_setup() -> None:
    """``st.switch_page`` for real, so a wrong path fails here and not on a click."""
    app = _home()
    app.button[0].click().run()
    _assert_clean(app)
    # The Setup page's own primary action is now on screen.
    assert any("Fetch current player data" in label for label in _labels(app))


def test_with_players_loaded_the_next_step_becomes_the_profiles(loaded) -> None:
    league, pool, _, _ = loaded
    app = _home(league=league, pool=pool, is_sample=True)
    _assert_clean(app)
    assert _labels(app)[0] == "Go to Manager Profiles"
    assert "Build the opponent models" in _text(app)


def test_with_profiles_built_the_next_step_becomes_the_draft_room(loaded) -> None:
    league, pool, history, profiles = loaded
    app = _home(
        league=league, pool=pool, history=history, profiles=profiles, is_sample=True
    )
    _assert_clean(app)
    assert _labels(app)[0] == "Go to the Draft Room"
    assert "Start a mock draft" in _text(app)


def test_a_draft_in_progress_offers_to_resume_it(loaded) -> None:
    """The returning-user case, which the Yes/No table made you work out yourself."""
    from engine.draft_state import DraftState
    from engine.simulator import DraftSimulator

    league, pool, history, profiles = loaded
    draft = DraftState(league, pool, seed=11)
    simulator = DraftSimulator(draft, profiles)
    for _ in range(6):
        if simulator.simulate_pick() is None:
            break

    app = _home(
        league=league, pool=pool, history=history, profiles=profiles, draft=draft,
        is_sample=True,
    )
    _assert_clean(app)
    labels = _labels(app)
    assert "Back to the Draft Room" in labels
    assert "Run simulations" in labels
    assert "Next:" not in _text(app)


def test_resuming_goes_to_the_draft_room(loaded) -> None:
    from engine.draft_state import DraftState

    league, pool, history, profiles = loaded
    app = _home(
        league=league, pool=pool, history=history, profiles=profiles,
        draft=DraftState(league, pool, seed=11), is_sample=True,
    )
    next(b for b in app.button if str(b.label) == "Back to the Draft Room").click().run()
    _assert_clean(app)
    # The Draft Room renders its own board rather than the home page's next step.
    assert "Next: Start a mock draft" not in _text(app)


# ─────────────────────────────────────────────────────────────────────────────
# Progress, and the one optional step that matters most
# ─────────────────────────────────────────────────────────────────────────────
def test_the_trail_names_every_stage_and_what_it_gives_you(loaded) -> None:
    """A checklist that only reports state does not explain why any of it matters."""
    import app as landing_module

    league, pool, _, _ = loaded
    app = _home(league=league, pool=pool, is_sample=True)
    _assert_clean(app)
    rendered = _text(app)
    for key, label in landing_module._PROGRESS_LABELS:
        assert label in rendered, f"the trail omits {label}"
        assert landing_module._TRAIL_HINTS[key] in rendered, f"no hint for {label}"


def test_importing_past_drafts_is_suggested_but_never_required(loaded) -> None:
    """History is the biggest upgrade to the model and still must not block anyone.

    Gating a new user at an optional import is how a tool earns a reputation for
    being hard to start, so this is a nudge — and it has to actually appear, or the
    single most valuable thing they could do stays invisible.
    """
    league, pool, _, _ = loaded
    app = _home(league=league, pool=pool, is_sample=True)
    _assert_clean(app)
    shown = _text(app)
    assert "import your past drafts" in shown.lower()
    # It is a suggestion, not the next step, and it does not replace the real one.
    assert _labels(app)[0] == "Go to Manager Profiles"


def test_the_nudge_goes_away_once_history_is_loaded(loaded) -> None:
    league, pool, history, profiles = loaded
    app = _home(
        league=league, pool=pool, history=history, profiles=profiles, is_sample=True
    )
    _assert_clean(app)
    assert "biggest single upgrade" not in _text(app)


def test_the_sample_data_warning_still_shows(loaded) -> None:
    """A fictional league must say so on the home page, redesign or not."""
    league, pool, history, profiles = loaded
    app = _home(
        league=league, pool=pool, history=history, profiles=profiles, is_sample=True
    )
    _assert_clean(app)
    assert "generated fiction" in _text(app)


# ─────────────────────────────────────────────────────────────────────────────
# The page table the landing page and the navigation share
# ─────────────────────────────────────────────────────────────────────────────
def test_every_registered_page_file_exists() -> None:
    """Both the sidebar and the landing page's buttons read this table.

    A path that is wrong here is a crash when the user clicks, not a visible
    mistake, so it is checked against the filesystem directly.
    """
    import app as landing_module

    for title, path in landing_module._PAGE_FILES.items():
        assert (ROOT / path).is_file(), f"{title} points at a missing file: {path}"
    assert set(landing_module._PAGE_ICONS) == set(landing_module._PAGE_FILES)


def test_every_step_points_at_a_registered_page() -> None:
    import app as landing_module

    for _, page_name, _, _, _ in landing_module._STEPS:
        assert page_name in landing_module._PAGE_FILES, page_name


def test_every_step_key_is_a_real_readiness_key() -> None:
    """The steps are keyed on readiness, so a renamed key would silently skip one."""
    import app as landing_module
    from ui import state as ui_state

    keys = set(ui_state.readiness())
    for step in landing_module._STEPS:
        assert step[0] in keys, step[0]
    for key, _ in landing_module._PROGRESS_LABELS:
        assert key in keys, key
    assert set(landing_module._TRAIL_HINTS) == {
        key for key, _ in landing_module._PROGRESS_LABELS
    }
