"""Saving an in-progress draft so a refresh does not throw it away.

Streamlit keeps a draft in ``st.session_state``, which lives exactly as long as the
browser tab's connection. A refresh, a laptop sleeping, a stray Ctrl-R forty picks
into a mock — all of them silently destroyed the draft. This module is the durable
half: after every pick the Draft Room writes a snapshot here, and on arrival it
offers to put the user back where they were.

**A snapshot is a list of decisions, not a pickled object.** That is the whole
design, and it is worth being explicit about why, because pickling ``DraftState``
would have been three lines:

* A pickle is a photograph of one version of the code. Renaming a field or adding a
  ``__slots__`` entry turns every saved draft into a traceback, and this app is
  actively being changed. Picks are ``(overall_pick, player_id)`` and will still be
  meaningful when everything around them has been rewritten.
* Replaying goes back through :meth:`DraftState.make_pick`, so rosters, the undo
  stack, availability and keeper resolution are rebuilt by the same code that built
  them the first time. A restored draft therefore cannot sit in a state a live
  draft could not have reached — which a pickle of a half-mutated object can.
* It is small. Two hundred picks is a few kilobytes, so writing one on every
  interaction costs nothing.

The pool and the league are *not* copied into the snapshot; it stores their database
ids and they are reloaded. A board is hundreds of players wide and duplicating it per
draft would make the snapshot bigger than the database it came from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Mapping

from core.config import SimulationConfig
from core.validation import ConfigurationError
from engine.draft_state import DraftState
from models.league import League
from models.player import PlayerPool
from services.repository import read_setting, write_setting

LOGGER = logging.getLogger("fantasy_mock_draft.draft_session")

RESUME_KEY = "resume_draft"
"""The ``application_settings`` key holding the snapshot.

One slot, not a history. The question this answers is "put me back where I was",
which has exactly one answer; a finished draft the user wants to keep goes through
``save_mock_draft`` on the Save tab, which is a different feature with a different
lifetime.
"""

SNAPSHOT_FORMAT = 1
"""Bumped if the snapshot layout changes incompatibly.

An unrecognised format is discarded rather than guessed at: offering to resume a
draft and then rebuilding it wrongly is worse than admitting it cannot be resumed.
"""


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class SnapshotPick:
    """One decision, and only the parts that cannot be recomputed.

    Everything else on :class:`models.draft.Pick` — the round, the manager, the ADP
    at the time, the roster slot it filled — is derived from the draft order and the
    board when the pick is replayed, so storing it would create two sources of truth
    that could disagree.
    """

    overall_pick: int
    player_id: str
    player_name: str = ""
    is_user_pick: bool = False
    was_manual_override: bool = False
    pick_probability: float | None = None
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_pick": int(self.overall_pick),
            "player_id": self.player_id,
            "player_name": self.player_name,
            "is_user_pick": bool(self.is_user_pick),
            "was_manual_override": bool(self.was_manual_override),
            "pick_probability": self.pick_probability,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SnapshotPick":
        probability = raw.get("pick_probability")
        return cls(
            overall_pick=int(raw.get("overall_pick") or 0),
            player_id=str(raw.get("player_id") or ""),
            player_name=str(raw.get("player_name") or ""),
            is_user_pick=bool(raw.get("is_user_pick")),
            was_manual_override=bool(raw.get("was_manual_override")),
            pick_probability=None if probability is None else float(probability),
            explanation=str(raw.get("explanation") or ""),
        )


@dataclass(slots=True)
class DraftSnapshot:
    """A resumable draft: which league, which board, which seat, which picks."""

    league_id: int | None = None
    source_id: int | None = None
    league_name: str = ""
    season: int | None = None
    team_count: int = 0
    rounds: int = 0
    seed: int | None = None
    user_slots: list[int] = field(default_factory=list)
    picks: list[SnapshotPick] = field(default_factory=list)
    strategy_notes: dict[int, list[str]] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    rng_state: list[Any] = field(default_factory=list)
    """Where the seeded generator had got to, as JSON-safe nesting.

    The seed alone is not enough. It fixes where the stream *starts*; the opponents'
    behaviour depends on where it has got to, and forty simulated picks in that is a
    long way from the start. Without this the room would quietly start behaving like
    a different room from the resume onward.
    """
    saved_at: str = ""
    format_version: int = SNAPSHOT_FORMAT

    @property
    def pick_count(self) -> int:
        return len(self.picks)

    @property
    def is_empty(self) -> bool:
        """A draft that has been started but has no picks yet.

        Not worth offering to resume: pressing *Start draft* again reproduces it
        exactly, and an offer to restore nothing reads like a bug.
        """
        return not self.picks

    def label(self) -> str:
        total = self.team_count * self.rounds
        where = f"{self.pick_count} of {total} picks" if total else f"{self.pick_count} picks"
        return f"{self.league_name or 'Draft'} ({self.season or '—'}) — {where}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": int(self.format_version),
            "league_id": self.league_id,
            "source_id": self.source_id,
            "league_name": self.league_name,
            "season": self.season,
            "team_count": int(self.team_count),
            "rounds": int(self.rounds),
            "seed": self.seed,
            "user_slots": [int(s) for s in self.user_slots],
            "picks": [p.to_dict() for p in self.picks],
            # JSON object keys are always strings, so the slot numbers come back as
            # strings and are coerced on the way in rather than trusted.
            "strategy_notes": {
                str(slot): list(notes) for slot, notes in self.strategy_notes.items()
            },
            "settings": dict(self.settings),
            "rng_state": list(self.rng_state),
            "saved_at": self.saved_at or _iso_now(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "DraftSnapshot | None":
        if not isinstance(raw, Mapping):
            return None
        version = int(raw.get("format_version") or 0)
        if version != SNAPSHOT_FORMAT:
            LOGGER.warning(
                "Ignoring a saved draft in format v%s (this build reads v%s)",
                version, SNAPSHOT_FORMAT,
            )
            return None
        notes: dict[int, list[str]] = {}
        for slot, values in (raw.get("strategy_notes") or {}).items():
            try:
                notes[int(slot)] = [str(v) for v in (values or [])]
            except (TypeError, ValueError):
                continue
        seed = raw.get("seed")
        season = raw.get("season")
        return cls(
            league_id=_opt_int(raw.get("league_id")),
            source_id=_opt_int(raw.get("source_id")),
            league_name=str(raw.get("league_name") or ""),
            season=None if season is None else _opt_int(season),
            team_count=int(raw.get("team_count") or 0),
            rounds=int(raw.get("rounds") or 0),
            seed=None if seed is None else _opt_int(seed),
            user_slots=[int(s) for s in (raw.get("user_slots") or [])],
            picks=[SnapshotPick.from_dict(p) for p in (raw.get("picks") or [])],
            strategy_notes=notes,
            settings=dict(raw.get("settings") or {}),
            rng_state=list(raw.get("rng_state") or []),
            saved_at=str(raw.get("saved_at") or ""),
            format_version=version,
        )

    def matches(self, league: League | None, pool: PlayerPool | None) -> bool:
        """Whether this snapshot describes the league and board currently loaded.

        Checked before a silent restore. The draft order and every pick number in
        the snapshot are relative to a specific team count and round count, so
        replaying into a differently-shaped league would put real picks in slots
        that never existed.
        """
        if league is None or pool is None:
            return False
        config = league.config
        if int(self.team_count) and int(self.team_count) != int(config.team_count):
            return False
        if int(self.rounds) and int(self.rounds) != int(config.rounds):
            return False
        if self.season is not None and config.season and int(self.season) != int(config.season):
            return False
        return True


def _opt_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rng_state_to_json(state: Any) -> list[Any]:
    """``random.getstate()`` as nested lists, since JSON has no tuples.

    The middle element is 625 integers — the Mersenne Twister's internal array. It is
    stored verbatim rather than summarised because there is nothing smaller that
    identifies a position in the stream.
    """
    try:
        version, internal, gauss = state
    except (TypeError, ValueError):
        return []
    return [int(version), [int(v) for v in internal], gauss]


def _rng_state_from_json(raw: Any) -> tuple | None:
    """Rebuild a ``setstate``-shaped tuple, or ``None`` if the record is unusable."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    version, internal, gauss = raw
    if not isinstance(internal, (list, tuple)):
        return None
    try:
        return (
            int(version),
            tuple(int(v) for v in internal),
            None if gauss is None else float(gauss),
        )
    except (TypeError, ValueError):
        return None


def snapshot_from_draft(
    draft: DraftState,
    *,
    league_id: int | None = None,
    source_id: int | None = None,
) -> DraftSnapshot:
    """Capture a live draft.

    Keeper picks are left out. They are not decisions — ``DraftState`` commits them
    automatically from the league's keeper list — so replaying them would try to
    draft a player the rebuilt state has already taken.
    """
    league = draft.league
    config = league.config
    notes = {
        slot: draft.strategy_notes(slot)
        for slot in league.slots_in_order()
        if draft.strategy_notes(slot)
    }
    return DraftSnapshot(
        league_id=_opt_int(league_id if league_id is not None else config.league_id),
        source_id=_opt_int(source_id),
        league_name=config.name,
        season=_opt_int(config.season),
        team_count=int(config.team_count),
        rounds=int(config.rounds),
        seed=draft.seed,
        user_slots=sorted(league.user_slots),
        picks=[
            SnapshotPick(
                overall_pick=pick.overall_pick,
                player_id=pick.player_id,
                player_name=pick.player_name,
                is_user_pick=bool(pick.is_user_pick),
                was_manual_override=bool(pick.was_manual_override),
                pick_probability=pick.pick_probability,
                explanation=pick.explanation,
            )
            for pick in draft.picks
            if not pick.is_keeper and pick.player_id
        ],
        strategy_notes=notes,
        settings=draft.settings.to_dict(),
        rng_state=_rng_state_to_json(draft.rng_state()),
        saved_at=_iso_now(),
    )


@dataclass(slots=True)
class RestoreResult:
    """A rebuilt draft, plus an honest account of anything that did not replay."""

    draft: DraftState
    replayed: int = 0
    expected: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def is_exact(self) -> bool:
        return self.replayed == self.expected and not self.warnings


def restore(
    snapshot: DraftSnapshot,
    league: League,
    pool: PlayerPool,
    settings: SimulationConfig | None = None,
) -> RestoreResult:
    """Rebuild a :class:`DraftState` by replaying the snapshot's picks.

    Replay **stops** at the first pick that cannot be made, rather than skipping it.
    Skipping would shift every later pick one slot earlier, producing a coherent
    looking draft that never happened — and the user would have no way to tell.
    Stopping leaves them at the last pick that is definitely real, and says which
    one it gave up on.
    """
    resolved = settings or SimulationConfig.from_dict(snapshot.settings or {})
    if snapshot.user_slots:
        league.set_user_slot(int(snapshot.user_slots[0]))
    draft = DraftState(league, pool, settings=resolved, seed=snapshot.seed)

    warnings: list[str] = []
    replayed = 0
    ordered = sorted(snapshot.picks, key=lambda p: p.overall_pick)
    for entry in ordered:
        slot = draft.current_slot
        if slot is None:
            warnings.append(
                f"The saved draft had {len(ordered)} picks but this league only has "
                f"{len(draft.order)} — the rest were dropped."
            )
            break
        if slot.overall_pick != entry.overall_pick:
            warnings.append(
                f"Restored {replayed} pick(s). The saved draft expected pick "
                f"{entry.overall_pick} next but this board is on "
                f"{slot.overall_pick}, so the rest was left off rather than "
                "guessed at — the league's keepers or draft order have changed."
            )
            break
        player = pool.get(entry.player_id)
        if player is None:
            warnings.append(
                f"Restored {replayed} pick(s), then stopped: "
                f"{entry.player_name or entry.player_id} (pick {entry.overall_pick}) "
                "is not on the board that is loaded now. Later picks were left off "
                "because dropping one would move every pick after it."
            )
            break
        try:
            draft.make_pick(
                player,
                is_user_pick=entry.is_user_pick,
                was_manual_override=entry.was_manual_override,
                pick_probability=entry.pick_probability,
                explanation=entry.explanation,
            )
        except ConfigurationError as error:
            warnings.append(
                f"Restored {replayed} pick(s), then stopped at pick "
                f"{entry.overall_pick}: {error}"
            )
            break
        replayed += 1

    for slot, notes in (snapshot.strategy_notes or {}).items():
        for note in notes:
            try:
                draft.note_strategy(int(slot), note)
            except (TypeError, ValueError):
                continue

    # Last, once the picks are in place: put the generator back where it was. Replay
    # goes through ``make_pick``, which draws nothing, so nothing here disturbs it.
    rng_state = _rng_state_from_json(snapshot.rng_state)
    if rng_state is not None and not draft.set_rng_state(rng_state):
        warnings.append(
            "The opponents' random stream could not be restored, so the room will "
            "make different (still plausible) picks from here than it would have. "
            "The picks already made are unaffected."
        )
    elif rng_state is None and snapshot.picks:
        LOGGER.info("Saved draft has no RNG position; reseeding from %s", snapshot.seed)

    LOGGER.info(
        "Restored draft: %s of %s pick(s), %s warning(s)",
        replayed, len(ordered), len(warnings),
    )
    return RestoreResult(
        draft=draft, replayed=replayed, expected=len(ordered), warnings=warnings
    )


# ─────────────────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────────────────
def save_snapshot(snapshot: DraftSnapshot) -> None:
    """Write the snapshot, replacing any previous one."""
    write_setting(RESUME_KEY, snapshot.to_dict())


def load_snapshot() -> DraftSnapshot | None:
    """The saved snapshot, or ``None`` when there is nothing resumable.

    Never raises. A corrupt or half-written record is treated as absent: the app
    still works without a resume offer, and a traceback on page load would take the
    whole Draft Room down over a feature that is meant to be a safety net.
    """
    try:
        raw = read_setting(RESUME_KEY)
    except Exception:  # pragma: no cover - a database failure must not block the page
        LOGGER.exception("Could not read the saved draft")
        return None
    try:
        return DraftSnapshot.from_dict(raw)
    except Exception:
        LOGGER.exception("Discarding an unreadable saved draft")
        return None


def clear_snapshot() -> None:
    """Forget the saved draft. Called when one is abandoned or completed."""
    try:
        write_setting(RESUME_KEY, None)
    except Exception:  # pragma: no cover
        LOGGER.exception("Could not clear the saved draft")


def autosave(
    draft: DraftState | None,
    *,
    league_id: int | None = None,
    source_id: int | None = None,
) -> DraftSnapshot | None:
    """Persist ``draft`` if there is anything worth persisting.

    Called once per page render rather than at each of the six places a pick can be
    made. Every one of those already ends in ``st.rerun()``, so the render that
    follows sees the mutation and saves it — one write per interaction, and no way
    for a new pick path to be added later and forget to save.
    """
    if draft is None:
        return None
    snapshot = snapshot_from_draft(draft, league_id=league_id, source_id=source_id)
    if snapshot.is_empty:
        return snapshot
    save_snapshot(snapshot)
    return snapshot


def with_sources(
    snapshot: DraftSnapshot, *, league_id: int | None, source_id: int | None
) -> DraftSnapshot:
    """The snapshot with its database ids filled in, keeping the rest untouched."""
    return replace(
        snapshot,
        league_id=_opt_int(league_id) if league_id is not None else snapshot.league_id,
        source_id=_opt_int(source_id) if source_id is not None else snapshot.source_id,
    )


def describe_age(snapshot: DraftSnapshot, *, now: datetime | None = None) -> str:
    """How long ago the snapshot was written, in words.

    Deliberately not routed through :mod:`core.freshness`: that module judges
    whether *data* is too old to trust, and a draft from last week is not less valid
    for being old — it is exactly as resumable as one from a minute ago. The only
    question here is whether the user recognises it as theirs.
    """
    from core import freshness as core_freshness

    hours = core_freshness.age_hours(snapshot.saved_at, now=now)
    if hours is None:
        return "saved at an unknown time"
    if hours < 0:
        return "saved in the future (check the clock)"
    if hours < 1 / 60:
        return "saved just now"
    if hours < 1:
        return f"saved {int(hours * 60)} min ago"
    if hours < 36:
        return f"saved {hours:.0f} hours ago"
    return f"saved {hours / 24:.1f} days ago"


def snapshot_frame_rows(snapshot: DraftSnapshot) -> list[dict[str, Any]]:
    """The snapshot's picks as plain rows, for a preview table before restoring."""
    return [
        {
            "Overall": p.overall_pick,
            "Player": p.player_name or p.player_id,
            "You": "★" if p.is_user_pick else "",
        }
        for p in sorted(snapshot.picks, key=lambda p: p.overall_pick)
    ]


def resumable(
    snapshot: DraftSnapshot | None, league: League | None, pool: PlayerPool | None
) -> bool:
    """Whether an offer to resume should be shown at all."""
    if snapshot is None or snapshot.is_empty:
        return False
    return snapshot.matches(league, pool)


# ─────────────────────────────────────────────────────────────────────────────
# Reconnecting a snapshot to the league and board it was drafted against
# ─────────────────────────────────────────────────────────────────────────────
def persist_inputs(
    league: League, pool: PlayerPool | None, *, is_sample: bool = False
) -> tuple[int | None, int | None]:
    """Save the league and board to the database, returning their ids.

    Called when a draft starts. Resuming needs the same league and the same board,
    and until now both were saved only if the user happened to press *Save to
    database* on Setup — so the common case, fetch data and draft immediately, had
    nothing to come back to. One save per draft start is cheap; per pick would not
    be, which is why the snapshot only stores the ids afterwards.
    """
    from models.database import session_scope
    from services.repository import save_league, save_player_pool

    league_id: int | None = None
    source_id: int | None = None
    try:
        with session_scope() as session:
            league_id = _opt_int(save_league(session, league))
            if pool is not None and len(pool):
                source_id = _opt_int(
                    save_player_pool(
                        session, pool,
                        source_kind="sample" if is_sample else "auto",
                    )
                )
    except Exception:
        # A draft the user can play but not resume is far better than no draft, so a
        # database problem here is logged and shrugged off rather than raised.
        LOGGER.exception("Could not persist the league/board for resume")
    return league_id, source_id


@dataclass(slots=True)
class Rehydrated:
    """The league and board a snapshot refers to, reloaded from the database."""

    league: League | None = None
    pool: PlayerPool | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.league is not None and self.pool is not None and len(self.pool) > 0


def rehydrate(snapshot: DraftSnapshot) -> Rehydrated:
    """Reload the league and board named by a snapshot's ids.

    This is what makes a refresh transparent rather than merely recoverable. Without
    it the Draft Room blocks on "no league loaded" and sends the user to Setup, and
    by the time they have fetched data again the session is a different one — the
    draft would be sitting in the database with nothing able to reach it.
    """
    from models.database import session_scope
    from services.repository import load_league, load_player_pool

    if snapshot.league_id is None:
        return Rehydrated(reason="the saved draft does not name a stored league")
    try:
        with session_scope() as session:
            league = load_league(session, int(snapshot.league_id))
            pool = (
                load_player_pool(
                    session, int(snapshot.source_id),
                    league.config if league is not None else None,
                )
                if snapshot.source_id is not None else None
            )
    except Exception:
        LOGGER.exception("Could not reload the league/board for a saved draft")
        return Rehydrated(reason="the stored league or board could not be read")
    if league is None:
        return Rehydrated(reason="the stored league is no longer in the database")
    if pool is None or not len(pool):
        return Rehydrated(
            league=league,
            reason="the stored player board is no longer in the database",
        )
    return Rehydrated(league=league, pool=pool)


__all__ = [
    "RESUME_KEY", "SNAPSHOT_FORMAT", "SnapshotPick", "DraftSnapshot",
    "RestoreResult", "snapshot_from_draft", "restore", "save_snapshot",
    "load_snapshot", "clear_snapshot", "autosave", "with_sources",
    "describe_age", "snapshot_frame_rows", "resumable",
    "persist_inputs", "Rehydrated", "rehydrate",
]
