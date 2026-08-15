"""The user's own board: players to target, players to never draft, own rankings.

Everything else in this app is an opinion the model holds. This is the one place the
*user's* opinion lives, and it is kept deliberately separate from
:class:`~models.manager.ManagerProfile` preferences, which describe what someone
*else* is expected to do. Mixing the two would be a real bug: a target list is not a
prediction, so it must never change how the eleven opponents draft. It changes only
what the app recommends to the person reading it.

Three lists, because they answer three different questions:

* **Targets** — "I want these players." An ordered list; the order is the priority.
* **Do not draft** — "I will not take these players, whatever the model says."
* **My rankings** — "here is my board", a full or partial ordering that overrides the
  platform's when the user has one.

Names, not player ids. The user types "AJ Brown" from memory before the board is even
loaded, and ids differ per source, so every lookup goes through
:func:`services.normalize.player_key` — the same matcher the importers use, which
folds punctuation, case and suffixes. The typed spelling is preserved for display so
the user always sees their own words back.

A board is treated as immutable: the UI builds a new one and stores it, rather than
mutating in place. That is what lets the lookups be built once in ``__post_init__``
instead of being recomputed for every candidate on every pick.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from models.player import Player, PlayerPool
from services.normalize import clean_text, player_key
from services.repository import read_setting, write_setting

LOGGER = logging.getLogger("fantasy_mock_draft.user_board")

BOARD_KEY = "user_board"
"""``application_settings`` key holding the saved board.

Stored per installation rather than per league. A user's convictions about players
are theirs, not their league's — the same do-not-draft list applies in every mock
they run, and re-typing it per league is exactly the friction that would stop the
feature being used.
"""

BOARD_FORMAT = 1

MAX_NAMES = 600
"""Cap per list. Deeper than any draft — a 12-team, 16-round league is 192 players —
and low enough that a pasted spreadsheet of the whole player universe cannot turn
every pick into a linear scan. Raised from 400 when uploading a ranking file became
possible: a published top-300 plus its bench tail runs past 400 with no user error
involved, and truncating a file the user chose to upload is worse than holding a few
hundred more names.
"""

_RANK_LINE_RE = re.compile(r"^\s*(\d{1,3})\s*[.):,\-\t]\s*(.+?)\s*$")
"""``12. Ja'Marr Chase`` / ``12, Chase`` / ``12) Chase`` — an explicit rank."""


def _normalise_names(raw: Iterable[Any]) -> list[str]:
    """Clean, de-duplicate and truncate a user-typed list, order preserved.

    Order is preserved because for the target list it *is* the priority. De-duplication
    is by match key, so typing both "AJ Brown" and "A.J. Brown" keeps the first
    spelling once rather than double-counting one player.
    """
    out: list[str] = []
    seen: set[str] = set()
    for entry in raw or ():
        name = clean_text(entry)
        if not name:
            continue
        key = player_key(name)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(name)
        if len(out) >= MAX_NAMES:
            break
    return out


@dataclass(slots=True)
class UserBoard:
    """One user's targets, do-not-draft list and personal rankings."""

    targets: list[str] = field(default_factory=list)
    """Wanted players, highest priority first."""
    avoid: list[str] = field(default_factory=list)
    """Players the user will not draft at any price."""
    custom_ranks: dict[str, int] = field(default_factory=dict)
    """Typed name → the user's own overall rank. Sparse: a partial list is fine."""
    conflicts: list[str] = field(default_factory=list)
    """Names that appeared on both lists, kept for the UI to report."""

    _target_keys: dict[str, int] = field(default_factory=dict, repr=False)
    _avoid_keys: set[str] = field(default_factory=set, repr=False)
    _rank_keys: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.targets = _normalise_names(self.targets)
        self.avoid = _normalise_names(self.avoid)
        # A name on both lists is a contradiction, and the do-not-draft side wins.
        # Guessing the other way round would recommend a player the user has said
        # they will not take, which is the more damaging of the two mistakes: a
        # missing recommendation is a nuisance, a forbidden one destroys trust in
        # every other suggestion on the page.
        avoid_keys = {player_key(name) for name in self.avoid}
        clashes = [name for name in self.targets if player_key(name) in avoid_keys]
        if clashes:
            self.conflicts = list(clashes)
            self.targets = [
                name for name in self.targets if player_key(name) not in avoid_keys
            ]
        self._target_keys = {
            player_key(name): index + 1 for index, name in enumerate(self.targets)
        }
        self._avoid_keys = avoid_keys
        ranks: dict[str, int] = {}
        clean_ranks: dict[str, int] = {}
        for name, rank in (self.custom_ranks or {}).items():
            typed = clean_text(name)
            key = player_key(typed)
            try:
                value = int(rank)
            except (TypeError, ValueError):
                continue
            if not key or value <= 0 or key in ranks:
                continue
            ranks[key] = value
            clean_ranks[typed] = value
        self.custom_ranks = clean_ranks
        self._rank_keys = ranks

    # -- queries ---------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return not (self.targets or self.avoid or self.custom_ranks)

    @property
    def fingerprint(self) -> str:
        """A short stable string that changes whenever the board does.

        Used in the UI's derived-value cache stamp: a recommendation computed under the
        previous board must not be shown after an edit, and the cache cannot see into
        the object to notice.
        """
        ranks = ",".join(f"{k}={v}" for k, v in sorted(self._rank_keys.items()))
        return f"{'|'.join(self._target_keys)}#{'|'.join(sorted(self._avoid_keys))}#{ranks}"

    def is_target(self, player: Player) -> bool:
        return player_key(player.name) in self._target_keys

    def target_priority(self, player: Player) -> int | None:
        """1 for the first name on the target list, 2 for the next, ``None`` if absent."""
        return self._target_keys.get(player_key(player.name))

    def target_priority_by_name(self, name: str) -> int | None:
        """Priority for a typed name, for callers holding a name and no ``Player``."""
        return self._target_keys.get(player_key(name))

    def is_avoided(self, player: Player) -> bool:
        return player_key(player.name) in self._avoid_keys

    def custom_rank(self, player: Player) -> int | None:
        return self._rank_keys.get(player_key(player.name))

    def effective_rank(self, player: Player) -> float:
        """The user's rank if they gave one, otherwise the board's own.

        Unranked players sort after every ranked one rather than being dropped: a
        partial personal ranking means "these are my first thirty picks", not "nobody
        else exists".
        """
        mine = self.custom_rank(player)
        if mine is not None:
            return float(mine)
        theirs = player.rank_for()
        return float(theirs) if theirs is not None else float("inf")

    def sort_key(self, player: Player) -> tuple[float, float, int]:
        """Ordering for the user's own board: targets first, then rank.

        Targets are lifted above the rank ordering on purpose. A user who typed a name
        into the target list has said something the board cannot know, and burying it
        at its ADP would make the list no different from the default one.

        Ranks stay numerically comparable — a personal #40 does not beat a consensus
        #1, because a partial list of five names is not a claim about the other 250
        players. The last element breaks the tie when two players hold the same number,
        in favour of the one the user ranked themselves: at equal rank, what they said
        is better information than what the platform said.
        """
        priority = self.target_priority(player)
        return (
            float(priority) if priority is not None else float("inf"),
            self.effective_rank(player),
            0 if self.custom_rank(player) is not None else 1,
        )

    def sorted_players(self, players: Iterable[Player]) -> list[Player]:
        """``players`` in the user's order, avoided names removed."""
        keep = [p for p in players if not self.is_avoided(p)]
        return sorted(keep, key=self.sort_key)

    def unmatched(self, pool: PlayerPool) -> dict[str, list[str]]:
        """Names on the board that match nobody in ``pool``, per list.

        Surfaced rather than silently ignored: a typo in a do-not-draft list is
        invisible otherwise, and the user would believe they were protected from a
        player the app has never heard of.
        """
        known = {player_key(p.name) for p in pool}
        out: dict[str, list[str]] = {}
        for label, names in (
            ("targets", self.targets),
            ("avoid", self.avoid),
            ("custom_ranks", list(self.custom_ranks)),
        ):
            missing = [n for n in names if player_key(n) not in known]
            if missing:
                out[label] = missing
        return out

    def describe(self) -> str:
        parts: list[str] = []
        if self.targets:
            parts.append(f"{len(self.targets)} target{'s' if len(self.targets) != 1 else ''}")
        if self.avoid:
            parts.append(f"{len(self.avoid)} to avoid")
        if self.custom_ranks:
            parts.append(f"{len(self.custom_ranks)} of your own rankings")
        return ", ".join(parts) if parts else "nothing set"

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": BOARD_FORMAT,
            "targets": list(self.targets),
            "avoid": list(self.avoid),
            "custom_ranks": {k: int(v) for k, v in self.custom_ranks.items()},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "UserBoard":
        if not raw:
            return cls()
        version = int(raw.get("format_version") or BOARD_FORMAT)
        if version != BOARD_FORMAT:
            # Unlike a draft snapshot, a board is cheap for the user to retype and
            # expensive to get wrong, so an unknown layout is dropped rather than
            # half-read.
            LOGGER.warning("Ignoring a saved board written in format %s", version)
            return cls()
        return cls(
            targets=list(raw.get("targets") or []),
            avoid=list(raw.get("avoid") or []),
            custom_ranks=dict(raw.get("custom_ranks") or {}),
        )


EMPTY_BOARD = UserBoard()
"""A board that changes nothing. Used as the default so every call site can assume
there is one, without ``if board is not None`` around each query."""


# ─────────────────────────────────────────────────────────────────────────────
# Parsing pasted text
# ─────────────────────────────────────────────────────────────────────────────
def parse_names(text: str) -> list[str]:
    """One name per line (or comma-separated), leading numbering discarded.

    Accepts what people actually paste: a numbered list copied off a rankings site, a
    bulleted list, or a comma-separated line typed by hand.
    """
    if not text:
        return []
    rows: list[str] = []
    for raw_line in str(text).replace("\r", "\n").split("\n"):
        line = raw_line.strip().lstrip("-•*").strip()
        if not line:
            continue
        parts = line.split(",") if ("," in line and not _RANK_LINE_RE.match(line)) else [line]
        for part in parts:
            candidate = part.strip()
            match = _RANK_LINE_RE.match(candidate)
            if match:
                candidate = match.group(2).strip()
            candidate = re.sub(r"^\d{1,3}\s+", "", candidate).strip()
            if candidate:
                rows.append(candidate)
    return _normalise_names(rows)


def parse_rankings(text: str) -> dict[str, int]:
    """A pasted ranking list → ``{name: rank}``.

    An explicit number on the line is trusted; a line without one takes its position
    in the list. Mixing the two is normal in a hand-edited list, and honouring the
    explicit numbers is what lets a user paste ranks 1-10, skip to 25, and get what
    they meant.
    """
    if not text:
        return {}
    out: dict[str, int] = {}
    used: set[int] = set()
    next_rank = 1
    for raw_line in str(text).replace("\r", "\n").split("\n"):
        line = raw_line.strip().lstrip("-•*").strip()
        if not line:
            continue
        match = _RANK_LINE_RE.match(line)
        if match:
            rank, name = int(match.group(1)), match.group(2).strip()
        else:
            rank, name = None, line
        name = clean_text(name)
        if not name or not player_key(name):
            continue
        if rank is None:
            while next_rank in used:
                next_rank += 1
            rank = next_rank
        used.add(rank)
        next_rank = max(next_rank, rank + 1)
        out.setdefault(name, rank)
        if len(out) >= MAX_NAMES:
            break
    return out


def rankings_from_order(names: Sequence[str]) -> dict[str, int]:
    """Turn an ordered list of names into ``{name: 1..n}``."""
    return {name: index + 1 for index, name in enumerate(_normalise_names(names))}


RANK_COLUMN_CANDIDATES: tuple[str, ...] = (
    "my_rank", "overall_rank", "position_rank", "platform_rank", "overall_adp",
)
"""Columns that could carry the user's own ordering, best first.

``overall_adp`` is last and is a deliberate compromise: a file exported from a
rankings site sometimes has no rank column at all, only the ADP it was sorted by.
Reading it as a rank is right for the *ordering*, which is the only thing this list
is used for — the numbers themselves are re-based to 1..n below.
"""


def rankings_from_frame(frame: Any) -> tuple[dict[str, int], list[str]]:
    """A ranking file → ``({name: 1..n}, notes)``.

    Only two things are needed: which column holds the names, and what order the
    players are in. Everything else in the file is ignored, because a personal ranking
    is an ordering and nothing else — projections and ADP in the same file belong to
    whoever published it, and importing them here would quietly overwrite the board's
    own numbers with a stranger's.

    Order comes from a rank column when the file has one, and from **row order**
    otherwise, which is how most exports arrive: sorted, unnumbered. Either way the
    result is re-based to a dense 1..n, so a file ranked 1-50 and a file ranked
    3, 17, 92 both come out as a clean ordering.

    ``notes`` explains what was read, for the page to show. A silent import here would
    be the worst outcome: the user would not know whether their file's third column or
    its row order decided their board.
    """
    from services.normalize import normalize_columns  # local: keeps import cost off boot

    notes: list[str] = []
    if frame is None or getattr(frame, "empty", True):
        return {}, ["The file had no rows."]
    normalised, _ = normalize_columns(frame)
    if "player_name" not in normalised.columns:
        return {}, [
            "No player-name column found. Name it `player_name`, `player` or `name` "
            "and re-upload."
        ]

    rank_column = next(
        (c for c in RANK_COLUMN_CANDIDATES if c in normalised.columns), None
    )
    working = normalised
    if rank_column is not None:
        numbers = pd.to_numeric(working[rank_column], errors="coerce")
        if numbers.notna().any():
            working = working.assign(_order=numbers).sort_values(
                "_order", na_position="last", kind="stable"
            )
            notes.append(f"Ordered by the file's `{rank_column}` column.")
        else:
            rank_column = None
    if rank_column is None:
        notes.append("No usable rank column, so the file's row order was used.")

    out: dict[str, int] = {}
    seen: set[str] = set()
    skipped = 0
    for raw_name in working["player_name"].tolist():
        name = clean_text(raw_name)
        key = player_key(name)
        if not name or not key:
            continue
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        out[name] = len(out) + 1
        if len(out) >= MAX_NAMES:
            break
    if skipped:
        notes.append(f"{skipped} duplicate name(s) kept once, at their first position.")
    if len(working) > MAX_NAMES:
        notes.append(
            f"Only the first {MAX_NAMES} players were kept — that is the cap on a "
            f"board, and it is deeper than any draft."
        )
    notes.append(f"Read {len(out)} ranked player(s).")
    return out, notes


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
def save_board(board: UserBoard) -> None:
    """Store the board, replacing whatever was there."""
    write_setting(BOARD_KEY, board.to_dict())


def load_board() -> UserBoard:
    """The saved board, or an empty one.

    Never raises, for the same reason :func:`services.draft_session.load_snapshot`
    does not: this is read on page load, and a database problem must not be able to
    take the page down with it.
    """
    try:
        raw = read_setting(BOARD_KEY)
    except Exception:  # pragma: no cover - a database failure must not block a page
        LOGGER.exception("Could not read the saved board")
        return UserBoard()
    try:
        return UserBoard.from_dict(raw)
    except Exception:
        LOGGER.exception("Discarding an unreadable saved board")
        return UserBoard()


def clear_board() -> None:
    try:
        write_setting(BOARD_KEY, None)
    except Exception:  # pragma: no cover
        LOGGER.exception("Could not clear the saved board")


__all__ = [
    "UserBoard", "EMPTY_BOARD", "BOARD_KEY", "BOARD_FORMAT", "MAX_NAMES",
    "parse_names", "parse_rankings", "rankings_from_order", "rankings_from_frame",
    "RANK_COLUMN_CANDIDATES",
    "save_board", "load_board", "clear_board",
]
