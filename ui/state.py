"""Session state: the single place the UI keeps the objects a draft needs.

Streamlit reruns the whole script on every interaction, so anything that must
outlive a click has to live in ``st.session_state``. Reaching into that dict
directly from seven pages would mean seven spellings of every key and no way to
tell which page invalidated what, so every read and write goes through the
accessors here.

Nothing in this module simulates, scores or decides anything — it holds objects
built by the engine and hands them back. That separation is deliberate: the engine
must stay runnable without Streamlit (see ``scripts/`` and the test suite, neither
of which imports this module).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import streamlit as st

from core.config import SimulationConfig
from engine.draft_state import DraftState
from models.draft import DraftHistory
from models.league import League
from models.manager import ManagerProfile
from models.player import PlayerPool

LOGGER = logging.getLogger("fantasy_mock_draft.ui")

# Session-state keys. Constants rather than literals so a typo is an ImportError
# at startup instead of a silently-empty page at runtime.
K_LEAGUE = "league"
K_POOL = "pool"
K_HISTORY = "history"
K_PROFILES = "profiles"
K_DRAFT = "draft_state"
K_SETTINGS = "settings"
K_IS_SAMPLE = "is_sample_data"
K_PROVENANCE = "provenance"
K_LAST_RECS = "last_recommendations"
K_LAST_AVAIL = "last_availability"
K_MC_REPORT = "monte_carlo_report"
K_INITIALISED = "initialised"


@dataclass(slots=True)
class Provenance:
    """Where the data currently loaded came from, shown verbatim in the sidebar.

    Tracked because the app deliberately ships fictional data, and a user who
    cannot tell at a glance which they are looking at is the failure mode the
    "never present sample data as real" constraint exists to prevent.
    """

    pool_source: str = "none"
    history_source: str = "none"
    league_source: str = "none"
    is_sample: bool = False
    notes: list[str] = field(default_factory=list)


def ensure_initialised() -> None:
    """Put the empty-but-valid defaults in place. Safe to call on every rerun."""
    if st.session_state.get(K_INITIALISED):
        return
    st.session_state.setdefault(K_LEAGUE, None)
    st.session_state.setdefault(K_POOL, None)
    st.session_state.setdefault(K_HISTORY, DraftHistory())
    st.session_state.setdefault(K_PROFILES, {})
    st.session_state.setdefault(K_DRAFT, None)
    st.session_state.setdefault(K_SETTINGS, SimulationConfig())
    st.session_state.setdefault(K_IS_SAMPLE, False)
    st.session_state.setdefault(K_PROVENANCE, Provenance())
    st.session_state[K_INITIALISED] = True
    LOGGER.info("UI session state initialised")


# ─────────────────────────────────────────────────────────────────────────────
# Readers
# ─────────────────────────────────────────────────────────────────────────────
def league() -> League | None:
    return st.session_state.get(K_LEAGUE)


def pool() -> PlayerPool | None:
    return st.session_state.get(K_POOL)


def history() -> DraftHistory:
    return st.session_state.get(K_HISTORY) or DraftHistory()


def profiles() -> dict[int, ManagerProfile]:
    """Draft slot → profile. Empty until the profiles page has been run."""
    return st.session_state.get(K_PROFILES) or {}


def draft() -> DraftState | None:
    return st.session_state.get(K_DRAFT)


def settings() -> SimulationConfig:
    return st.session_state.get(K_SETTINGS) or SimulationConfig()


def provenance() -> Provenance:
    return st.session_state.get(K_PROVENANCE) or Provenance()


def is_sample_data() -> bool:
    return bool(st.session_state.get(K_IS_SAMPLE))


# ─────────────────────────────────────────────────────────────────────────────
# Writers
#
# Each writer invalidates what its own change makes stale. Doing it here rather
# than at the call sites is the point: a page that loads a new player pool should
# not have to remember that the live draft was built against the old one.
# ─────────────────────────────────────────────────────────────────────────────
def set_league(value: League | None, *, source: str = "") -> None:
    st.session_state[K_LEAGUE] = value
    if source:
        provenance().league_source = source
    _clear_draft("the league changed")
    _clear_derived()


def set_pool(value: PlayerPool | None, *, source: str = "") -> None:
    st.session_state[K_POOL] = value
    if value is not None and value.metadata.is_sample_data:
        st.session_state[K_IS_SAMPLE] = True
    if source:
        provenance().pool_source = source
    _clear_draft("the player pool changed")
    _clear_derived()


def set_history(value: DraftHistory, *, source: str = "") -> None:
    st.session_state[K_HISTORY] = value
    if source:
        provenance().history_source = source
    # Profiles are estimated *from* history, so new history makes them stale.
    st.session_state[K_PROFILES] = {}
    _clear_derived()


def set_profiles(value: dict[int, ManagerProfile]) -> None:
    st.session_state[K_PROFILES] = dict(value)
    _clear_derived()


def set_draft(value: DraftState | None) -> None:
    st.session_state[K_DRAFT] = value
    _clear_derived()


def set_settings(value: SimulationConfig) -> None:
    st.session_state[K_SETTINGS] = value
    _clear_derived()


def mark_sample_data(flag: bool = True) -> None:
    st.session_state[K_IS_SAMPLE] = bool(flag)
    provenance().is_sample = bool(flag)


def _clear_draft(reason: str) -> None:
    if st.session_state.get(K_DRAFT) is not None:
        LOGGER.info("Discarding the in-progress draft: %s", reason)
        st.session_state[K_DRAFT] = None
        # The saved copy goes with it. A snapshot that outlived the draft it came
        # from would offer to restore picks made against a board or a league that no
        # longer exists — the resume offer checks for that, but leaving the record
        # behind means the check has to keep being right forever.
        discard_saved_draft(reason)


def discard_saved_draft(reason: str = "") -> None:
    """Forget the autosaved draft. Imported lazily so the UI can load headless.

    ``services.draft_session`` reaches the database, and this module is imported by
    every page at startup; a database that cannot be opened should not stop the app
    from rendering the page that explains why.
    """
    try:
        from services import draft_session

        draft_session.clear_snapshot()
    except Exception:  # pragma: no cover - never worth failing a render over
        LOGGER.exception("Could not clear the saved draft (%s)", reason or "no reason")


def _clear_derived() -> None:
    """Drop cached recommendations and rollouts.

    These are answers about a specific board at a specific pick. Showing one
    computed against a previous state would be worse than showing nothing,
    because the user cannot tell the difference.
    """
    for key in (K_LAST_RECS, K_LAST_AVAIL, K_MC_REPORT):
        st.session_state.pop(key, None)


# ─────────────────────────────────────────────────────────────────────────────
# Derived-value cache
# ─────────────────────────────────────────────────────────────────────────────
def cached(key: str, builder: Callable[[], Any], *, stamp: Any = None) -> Any:
    """Return a cached value, rebuilding when ``stamp`` differs from last time.

    ``st.cache_data`` is not usable here: the values are engine objects keyed by
    mutable draft state, which is neither hashable nor picklable. The stamp is
    normally the draft's pick index, so a value computed at pick 40 is not shown
    at pick 41.
    """
    holder = st.session_state.setdefault("_derived", {})
    entry = holder.get(key)
    if entry is not None and entry[0] == stamp:
        return entry[1]
    value = builder()
    holder[key] = (stamp, value)
    return value


def readiness() -> dict[str, bool]:
    """What is loaded, for the sidebar checklist and page-level gating."""
    return {
        "league": league() is not None,
        "players": pool() is not None and len(pool() or []) > 0,
        "history": bool(history().drafts),
        "profiles": bool(profiles()),
        "draft": draft() is not None,
    }


def blocking_reason(*, needs_draft: bool = False) -> str | None:
    """A sentence naming what the user must do first, or ``None`` when ready.

    Pages call this instead of raising: a half-rendered page with a traceback in
    the middle is worse for the user than one line telling them where to go.
    """
    ready = readiness()
    if not ready["league"]:
        return (
            "No league loaded yet. Start on **Setup** — *Fetch current player data* "
            "seats a league in one click, or connect your Sleeper league for your "
            "real managers."
        )
    if not ready["players"]:
        return (
            "No player pool loaded. Fetch current rankings and ADP on **Setup**, or "
            "import your own rankings file there."
        )
    if needs_draft and not ready["draft"]:
        return "No draft in progress. Start one on **Draft Room**."
    return None


__all__ = [
    "Provenance", "ensure_initialised", "league", "pool", "history", "profiles",
    "draft", "settings", "provenance", "is_sample_data", "set_league", "set_pool",
    "set_history", "set_profiles", "set_draft", "set_settings", "mark_sample_data",
    "cached", "readiness", "blocking_reason", "discard_saved_draft",
    "K_LAST_RECS", "K_LAST_AVAIL", "K_MC_REPORT",
]
