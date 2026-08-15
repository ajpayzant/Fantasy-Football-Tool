"""Fantasy Football Mock Draft Simulator — Streamlit entry point.

Run with::

    streamlit run app.py

This module wires up the app and nothing else: page registration, logging, the
database schema, and the landing page. All behaviour lives in ``engine/`` (which
never imports Streamlit) and all rendering in ``ui/`` and ``ui/pages/``.
"""

from __future__ import annotations

import logging

import streamlit as st

from models.database import init_db
from ui import components, state

LOGGER = logging.getLogger("fantasy_mock_draft.app")

PAGE_TITLE = "Fantasy Mock Draft Simulator"


def _configure_logging() -> None:
    """Log to the console at INFO. Called once per process.

    The engine logs the events that matter — profile estimation, rollout counts,
    imports — and a user reporting odd behaviour needs those visible rather than
    swallowed.
    """
    root = logging.getLogger("fantasy_mock_draft")
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(logging.INFO)


@st.cache_resource
def _bootstrap() -> str:
    """Create the schema once per process. Cached so reruns do not re-open it."""
    _configure_logging()
    path = init_db()
    LOGGER.info("Database ready at %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# The landing page
# ─────────────────────────────────────────────────────────────────────────────
# Each step is (key, page, title, why it matters, the label on the button). The
# order is the dependency order, so the first step whose key is not yet satisfied
# is the one to offer — which is why this is a table rather than a chain of ifs.
# "history" is deliberately absent as a *blocker*: the app works without it, it
# just models generic opponents, and stopping a new user at an optional import is
# how a tool acquires a reputation for being hard to start.
_STEPS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "players", "Setup", "Get current player data",
        "Real players, real ADP, real projections. One button, about ten seconds.",
        "Go to Setup",
    ),
    (
        "profiles", "Manager Profiles", "Build the opponent models",
        "Turns whatever is known about each manager into how they will draft. Without "
        "this the room drafts to archetype priors instead of to your league.",
        "Go to Manager Profiles",
    ),
    (
        "draft", "Draft Room", "Start a mock draft",
        "Live recommendations, survival odds for the players you want, and the room "
        "picking the way your league picks.",
        "Go to the Draft Room",
    ),
)

_PROGRESS_LABELS: tuple[tuple[str, str], ...] = (
    ("players", "Players"),
    ("league", "League"),
    ("history", "Past drafts"),
    ("profiles", "Profiles"),
    ("draft", "Mock draft"),
)

# What each step buys you, not whether it is done — a checklist that only reports
# state leaves the user to work out why any of it matters.
_TRAIL_HINTS: dict[str, str] = {
    "players": "real ADP and projections",
    "league": "teams, scoring, your seat",
    "history": "optional — makes opponents yours",
    "profiles": "how each manager drafts",
    "draft": "in progress",
}

# One place mapping a page title to its file, used both by the navigation below and
# by the landing page's buttons. Two lists would drift, and a stale path here is a
# crash on click rather than a visible mistake.
#
# The order here is the sidebar order. It is *not* the filename numbering for My Board,
# which was added last and so carries an 8: it belongs next to the Player Pool, because
# both answer "who is on the board and what do I think of them", and a user who has just
# imported a pool is exactly the user about to paste their own rankings over it.
_PAGE_FILES: dict[str, str] = {
    "Setup": "ui/pages/1_setup.py",
    "Player Pool": "ui/pages/2_player_pool.py",
    "My Board": "ui/pages/8_my_rankings.py",
    "Manager Profiles": "ui/pages/3_manager_profiles.py",
    "Draft Room": "ui/pages/4_draft_room.py",
    "Simulations": "ui/pages/5_simulations.py",
    "Analysis": "ui/pages/6_analysis.py",
    "Settings": "ui/pages/7_settings.py",
}
_PAGE_ICONS: dict[str, str] = {
    "Setup": "⚙️", "Player Pool": "📋", "My Board": "📝", "Manager Profiles": "🕵️",
    "Draft Room": "🎯", "Simulations": "🎲", "Analysis": "📊", "Settings": "🔧",
}


def _next_step(ready: dict[str, bool]) -> tuple[str, str, str, str, str] | None:
    """The first step not yet done, or ``None`` when everything is ready."""
    for step in _STEPS:
        if not ready[step[0]]:
            return step
    return None


def landing() -> None:
    """The home page: one thing to do next, and how far along you are.

    This used to be two columns of prose about what the app does and a row of
    Yes/No metrics. Both were true and neither was actionable — a new user read a
    feature list and then had to work out which sidebar entry to click, and a
    returning user had to re-derive where they had left off from five booleans. So
    the top of the page is now a single next action with a button that goes there,
    the explanation is still available but folded away, and the readiness row is
    a trail that says what each step *gives you* rather than whether it is done.
    """
    components.page_header(
        "🏈 Fantasy Football Mock Draft Simulator",
        "Practise your draft against opponents modelled on your league's own history.",
    )

    ready = state.readiness()
    step = _next_step(ready)

    # ── What to do next ──────────────────────────────────────────────────────
    if step is None:
        st.success(
            "**Everything is ready and a draft is in progress.** Pick up where you "
            "left off in the Draft Room.",
            icon="✅",
        )
        resume = st.columns([1, 1, 3])
        if resume[0].button("Back to the Draft Room", type="primary", width="stretch"):
            st.switch_page("ui/pages/4_draft_room.py")
        if resume[1].button("Run simulations", width="stretch"):
            st.switch_page("ui/pages/5_simulations.py")
    else:
        key, page_name, title, why, button_label = step
        with st.container(border=True):
            st.markdown(f"### Next: {title}")
            st.write(why)
            columns = st.columns([1, 2])
            if columns[0].button(button_label, type="primary", width="stretch"):
                st.switch_page(_PAGE_FILES[page_name])
            done = [label for k, label in _PROGRESS_LABELS if ready[k]]
            columns[1].caption(
                ("Already done: " + ", ".join(done) + ".") if done
                else "Nothing is loaded yet — this is the first step."
            )

    # ── Where you are ────────────────────────────────────────────────────────
    st.write("")
    trail = st.columns(len(_PROGRESS_LABELS))
    for column, (key, label) in zip(trail, _PROGRESS_LABELS):
        column.markdown(
            f"{'✅' if ready[key] else '⬜'} **{label}**<br>"
            f"<span style='color:#888;font-size:0.8em'>{_TRAIL_HINTS[key]}</span>",
            unsafe_allow_html=True,
        )

    if ready["players"] and not ready["history"]:
        # Not a blocker, so it is a nudge rather than a step: this is the single
        # change that moves the model from "generic opponents" to "your league".
        st.info(
            "**Optional, and the biggest single upgrade:** import your past drafts on "
            "**Setup → Draft history**. Paste your recap from ESPN, Yahoo or anywhere "
            "else, or connect a Sleeper league. Two or three seasons is enough for the "
            "model to tell a manager's habit from one unusual draft.",
            icon="💡",
        )

    # ── Reference, folded away ───────────────────────────────────────────────
    with st.expander("What this app does"):
        st.markdown(
            """
            - **Models each manager individually** from your league's past drafts —
              who reaches, who follows the rankings, who always takes a quarterback early.
            - **Simulates the picks between now and your next turn**, so "will he last?"
              gets an answer from the opponents who will actually be picking.
            - **Recommends through several lenses** — best available, best roster fit,
              last chance, best value — and tells you when they agree.
            - **Runs full drafts hundreds of times** to show the range of rosters you
              tend to end up with, not a single guess.
            """
        )
    with st.expander("Every page, and what it is for"):
        st.markdown(
            """
            1. **Setup** — fetch current rankings and ADP, connect your Sleeper league,
               paste a draft recap from any platform, or import your own files.
            2. **Player Pool** — the board: projections, ADP from each platform,
               and where every number came from.
            3. **Manager Profiles** — build the opponent models, see what was inferred,
               and tell the model what you know that the history does not show.
            4. **Draft Room** — run a live mock draft with recommendations at every turn.
            5. **Simulations** — run many drafts from any point and study the outcomes.
            6. **Analysis** — review a finished draft pick by pick.
            7. **Settings** — every weight and constant the model uses.
            """
        )

    if state.is_sample_data():
        st.divider()
        st.subheader("A note on this league")
        st.markdown(
            "The league currently loaded was saved as **sample data** in an earlier "
            "session: its players, managers and draft results are **generated fiction**. "
            "No projection here describes a real football player. Sample data can no "
            "longer be created — to draft against real players, use **Setup → Fetch "
            "current player data**, or connect your Sleeper league."
        )


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE, page_icon="🏈", layout="wide",
        initial_sidebar_state="expanded",
    )
    _bootstrap()
    state.ensure_initialised()

    pages = [st.Page(landing, title="Home", icon="🏈", default=True)]
    pages += [
        st.Page(path, title=title, icon=_PAGE_ICONS[title])
        for title, path in _PAGE_FILES.items()
    ]
    navigation = st.navigation(pages)
    navigation.run()
    # After, not before: the sidebar summarises session state, and a page that has
    # just loaded a league would otherwise show the previous contents until the
    # next interaction. Streamlit places it in the sidebar container either way.
    components.sidebar_status()


main()
