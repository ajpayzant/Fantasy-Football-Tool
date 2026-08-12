"""Turn a pasted draft board into importable rows, whatever shape it arrived in.

Every fantasy platform shows a draft recap, and every one of them lets you select
it and press Ctrl-C. None of them agree on what lands on the clipboard. This module
is the answer to "how do I get my ESPN or Yahoo league in here" that does not
require an API key, an OAuth application, or a cookie: paste the recap page.

Four layouts are recognised, which between them cover ESPN, Yahoo, NFL.com, CBS
and Fantrax recaps as well as anything hand-typed:

``round_blocks``
    A ``Round 3`` header followed by that round's picks, one per line. The dominant
    shape when you copy a recap ordered by pick.
``team_blocks``
    A manager's name on its own line, then that manager's picks. Round numbers are
    usually absent, so they come from the line's position in the block, and the
    overall pick is reconstructed from the draft order.
``grid``
    Rows are rounds and columns are teams — the literal draft board. Needs the
    header row of team names to be present, otherwise it is indistinguishable from
    a table of anything else.
``pick_list``
    One pick per line with no grouping at all: ``1.05 Alex — Bijan Robinson RB ATL``
    or ``17. Team Sharpe: Puka Nacua``.

The output is always a frame with :data:`core.constants.HISTORICAL_IMPORT_COLUMNS`
headers, so it feeds :func:`services.importers.import_historical_drafts` unchanged
and inherits its validation, its manager-spelling merge and its rejected-row
reporting. This module's job stops at "which text meant which pick".

Two rules hold throughout, both of them there because a draft board is user data
and the failure mode to avoid is silent:

* **Nothing raises and nothing is dropped in silence.** A line that cannot be read
  comes back in :attr:`PasteResult.unparsed` with the reason, for the UI to show.
* **A guess is labelled a guess.** :attr:`PasteResult.layout` says which shape was
  assumed and :attr:`PasteResult.notes` says what was inferred rather than read —
  the season, the team count, the draft direction. Getting the layout wrong is the
  one error that produces plausible-looking nonsense, so it is always stated.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from core.constants import HISTORICAL_IMPORT_COLUMNS
from core.enums import Position
from core.validation import ValidationReport
from services.normalize import (
    clean_text,
    normalize_manager_name,
    normalize_player_name,
    normalize_position,
    normalize_team,
)

LOGGER = logging.getLogger("fantasy_mock_draft.draft_paste")

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────
# Every separator a platform puts between a manager and a player, or a player and
# their team. Em dash and en dash are in here because copying from a browser
# produces them and a plain hyphen split would miss.
_SEPARATORS = ("—", "–", " - ", " | ", "\t", ":", ",")

_ROUND_HEADER_RE = re.compile(
    r"^\s*(?:round|rd\.?|r)\s*[:#]?\s*(\d{1,2})\s*(?:of\s*\d+)?\s*$",
    re.IGNORECASE,
)
# "1.05", "1.5", "12.11" — round.pick notation, the most compact and least
# ambiguous thing a board can carry.
_DOT_PICK_RE = re.compile(r"^\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*(?:[).:\-–—]|\s)\s*")
# "Pick 14", "#14", "14." at the start of a line — a bare overall pick number.
_OVERALL_PICK_RE = re.compile(
    r"^\s*(?:pick\s*)?#?\s*(\d{1,3})\s*(?:[).:\-–—]\s*|\s+)", re.IGNORECASE
)
# Trailing "(RB - ATL)", "RB ATL", "ATL RB", "RB, Atlanta" — the player's own
# position and team, which the importer stores but does not need.
_PAREN_TAIL_RE = re.compile(r"[(\[]([^)\]]{1,40})[)\]]\s*$")
_POSITION_TOKENS = {str(position).upper() for position in Position}
_POSITION_TOKENS.update({"D/ST", "DST", "DEF", "PK", "K", "WR/RB", "FLEX"})

# Text platforms add that is never part of a name. Checked as whole lines only, so
# a manager legitimately called "Keeper" is unaffected.
_NOISE_LINES = frozenset({
    "draft recap", "draft results", "draft board", "results", "recap", "round",
    "pick", "player", "team", "manager", "overall", "pos", "position", "nfl",
    "keeper", "keepers", "auction", "$", "view all", "full draft results",
    "by round", "by team", "printable", "expand", "collapse",
})

_KEEPER_MARKER_RE = re.compile(r"\(\s*(?:keeper|k)\s*\)\s*$", re.IGNORECASE)


@dataclass(slots=True)
class UnparsedLine:
    """A line the parser could not turn into a pick, and why."""

    line_number: int
    text: str
    reason: str


@dataclass(slots=True)
class PasteResult:
    """Rows ready for the historical importer, plus what had to be assumed."""

    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    report: ValidationReport = field(default_factory=ValidationReport)
    layout: str = ""
    notes: list[str] = field(default_factory=list)
    unparsed: list[UnparsedLine] = field(default_factory=list)
    managers: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.report.ok and not self.frame.empty

    @property
    def pick_count(self) -> int:
        return int(len(self.frame))

    def unparsed_frame(self) -> pd.DataFrame:
        """The unreadable lines as a table, for display or download."""
        return pd.DataFrame(
            [
                {"line": u.line_number, "text": u.text, "why": u.reason}
                for u in self.unparsed
            ],
            columns=["line", "text", "why"],
        )

    def describe(self) -> str:
        """One sentence naming the layout, the yield, and the assumptions."""
        if self.frame.empty:
            return "Nothing in the paste could be read as a draft pick."
        shape = LAYOUT_LABELS.get(self.layout, self.layout or "unrecognised layout")
        text = (
            f"Read {self.pick_count} pick(s) for {len(self.managers)} manager(s) "
            f"as {shape}."
        )
        if self.unparsed:
            text += f" {len(self.unparsed)} line(s) could not be read."
        return text


LAYOUT_LABELS: dict[str, str] = {
    "round_blocks": "rounds, each followed by its picks",
    "team_blocks": "one block per team",
    "grid": "a draft board grid (rounds down, teams across)",
    "pick_list": "a flat list of picks",
}


# ─────────────────────────────────────────────────────────────────────────────
# Line-level helpers
# ─────────────────────────────────────────────────────────────────────────────
_SPACE_RUN_RE = re.compile(r"[  ]{2,}")
_SPACE_ONLY_RE = re.compile(r"[  ]+")


def _visible_lines(text: str, *, tabify: bool = False) -> list[tuple[int, str]]:
    """Numbered non-blank lines, with platform chrome removed.

    Tabs are preserved rather than run through :func:`clean_text`, which collapses
    them — they are the only thing marking cell boundaries in a board copied out of
    a browser table, so losing them would make a grid unreadable.

    ``tabify`` additionally promotes runs of two or more spaces to tabs, for the
    boards that arrive space-aligned instead of tab-separated. It is off by default
    because it would split ``"Round  1"`` in two.

    Line numbers are the user's own 1-based line numbers so that a reported failure
    can be found in what they pasted.
    """
    out: list[tuple[int, str]] = []
    for index, raw in enumerate((text or "").splitlines(), start=1):
        line = str(raw).replace(" ", " ")
        line = _SPACE_RUN_RE.sub("\t", line) if tabify else line
        line = _SPACE_ONLY_RE.sub(" ", line).strip()
        if not line or not line.strip("\t "):
            continue
        if line.replace("\t", " ").strip().lower().strip(":#") in _NOISE_LINES:
            continue
        out.append((index, line))
    return out


def _strip_position_and_team(text: str) -> tuple[str, Position | None, str]:
    """Split ``"Bijan Robinson RB ATL"`` into the name, the position, the team.

    Written as a tail-trimming loop rather than one regex because the three
    orderings platforms use (``RB ATL``, ``ATL RB``, ``(RB - ATL)``) all reduce to
    the same operation: keep taking recognisable tokens off the end.
    """
    working = text.strip()
    position: Position | None = None
    team = ""

    tail = _PAREN_TAIL_RE.search(working)
    if tail:
        inside = tail.group(1)
        working = working[: tail.start()].strip()
        for token in re.split(r"[\s,/\-–—]+", inside):
            token = token.strip()
            if not token:
                continue
            if position is None and token.upper() in _POSITION_TOKENS:
                position = normalize_position(token)
            elif not team and normalize_team(token):
                team = normalize_team(token)

    # Up to three trailing tokens: covers "Name RB ATL" and the "ATL RB" order,
    # and stops before it can start eating surname tokens.
    for _ in range(3):
        parts = working.rsplit(None, 1)
        if len(parts) != 2:
            break
        head = parts[0].strip()
        token_clean = parts[1].strip(" ,;:|")
        if position is None and token_clean.upper() in _POSITION_TOKENS:
            position = normalize_position(token_clean)
            working = head
            continue
        if not team and len(token_clean) <= 4 and normalize_team(token_clean):
            # A two-to-four letter token that is a real NFL abbreviation. Length is
            # the guard: without it "Chase" would be tested and could collide.
            team = normalize_team(token_clean)
            working = head
            continue
        break

    return working.strip(" ,;:|-–—"), position, team


def _split_manager_and_player(text: str) -> tuple[str, str]:
    """Split a pick line into (manager, player) on the first real separator.

    Ambiguity is unavoidable here — ``Smith: Jones`` could be either way round —
    so the convention platforms actually use is applied: the manager comes first.
    """
    for separator in _SEPARATORS:
        if separator in text:
            left, _, right = text.partition(separator)
            left, right = left.strip(), right.strip()
            if left and right:
                return left, right
    return "", text.strip()


def _looks_like_a_person_only(text: str) -> bool:
    """True for a line that is plausibly just a team or manager name.

    Used to find the headers in a team-block layout. Deliberately strict: any pick
    number, any position token, any separator disqualifies it, because misreading a
    pick line as a header shifts every following pick onto the wrong manager.
    """
    if not text or len(text) > 40:
        return False
    if _DOT_PICK_RE.match(text) or _OVERALL_PICK_RE.match(text):
        return False
    if _ROUND_HEADER_RE.match(text):
        return False
    if any(separator in text for separator in ("—", "–", " - ", "\t", ":")):
        return False
    tokens = text.split()
    if not 1 <= len(tokens) <= 5:
        return False
    return not any(token.strip(".,").upper() in _POSITION_TOKENS for token in tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Layout detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_layout(text: str) -> str:
    """Name the shape of the paste. Returns one of :data:`LAYOUT_LABELS`' keys.

    Order matters. A grid is checked first because its rows contain tab-separated
    player names that would otherwise read as manager/player pairs; round headers
    next because they are unambiguous; team blocks last because their evidence
    (lines that are only a name) is the weakest.
    """
    lines = _visible_lines(text)
    if not lines:
        return ""

    bodies = [line for _, line in lines]
    # A grid is the one layout that needs ``tabify``, so it is tested against the
    # tabified reading: a space-aligned board has no tabs of its own but is still a
    # board. Three-plus cells per row is the threshold, because two could be any
    # "manager — player" line.
    gridded = [line for _, line in _visible_lines(text, tabify=True)]
    tabbed = [line for line in gridded if line.count("\t") >= 2]
    if len(tabbed) >= 2 and len(tabbed) >= len(bodies) * 0.5:
        return "grid"

    if sum(1 for line in bodies if _ROUND_HEADER_RE.match(line)) >= 2:
        return "round_blocks"

    dotted = sum(1 for line in bodies if _DOT_PICK_RE.match(line))
    if dotted >= max(2, len(bodies) * 0.3):
        return "pick_list"

    name_only = sum(1 for line in bodies if _looks_like_a_person_only(line))
    numbered = sum(1 for line in bodies if _OVERALL_PICK_RE.match(line))
    if name_only >= 2 and name_only <= len(bodies) * 0.5 and numbered < name_only:
        return "team_blocks"

    return "pick_list"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_draft_board(
    text: str,
    *,
    season: int | None = None,
    team_count: int | None = None,
    manager_names: Sequence[str] | None = None,
    layout: str | None = None,
    league_name: str = "",
    platform: Any = None,
    snake: bool = True,
) -> PasteResult:
    """Parse a pasted draft recap into rows for the historical importer.

    ``season`` is required by the importer; when it is absent a note says so and
    the caller is expected to supply it. ``team_count`` is only needed to
    reconstruct overall picks for layouts that do not carry them, and is inferred
    from the board when it can be.
    """
    result = PasteResult()
    if not (text or "").strip():
        result.report.error("empty_paste", "Nothing was pasted.")
        return result

    chosen = layout or detect_layout(text)
    result.layout = chosen
    if not chosen:
        result.report.error(
            "unrecognised_layout",
            "The paste has no lines that look like draft picks.",
        )
        return result

    parser = {
        "grid": _parse_grid,
        "round_blocks": _parse_round_blocks,
        "team_blocks": _parse_team_blocks,
        "pick_list": _parse_pick_list,
    }[chosen]
    picks = parser(text, result, team_count=team_count, snake=snake)

    if not picks:
        result.report.error(
            "no_picks_found",
            f"Read the paste as {LAYOUT_LABELS.get(chosen, chosen)}, but no line "
            "yielded both a manager and a player. If the layout is wrong, choose it "
            "by hand.",
        )
        return result

    if manager_names:
        _reconcile_managers(picks, manager_names, result)

    for pick in picks:
        pick["season"] = season
        pick["league_name"] = league_name
        pick["platform"] = str(platform) if platform is not None else ""

    if season is None:
        result.notes.append(
            "No season was set, so the rows carry none — the importer needs one."
        )

    result.managers = sorted({p["manager_name"] for p in picks if p["manager_name"]})
    result.frame = _as_frame(picks)
    _check_completeness(result, team_count=team_count)
    LOGGER.info(
        "Parsed %d picks from a pasted board as %s", len(picks), chosen,
    )
    return result


def _as_frame(picks: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Rows → a frame with the importer's own column names, in its own order."""
    frame = pd.DataFrame(list(picks))
    for column in HISTORICAL_IMPORT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    extra = [c for c in frame.columns if c not in HISTORICAL_IMPORT_COLUMNS]
    frame = frame[list(HISTORICAL_IMPORT_COLUMNS) + extra]
    # Draft order, not parse order — a team-block paste is read one manager at a
    # time, and a table the user is about to check should read like the draft did.
    order = pd.to_numeric(frame["overall_pick"], errors="coerce")
    return frame.assign(_order=order).sort_values(
        "_order", kind="stable", na_position="last"
    ).drop(columns="_order").reset_index(drop=True)


def _record(
    picks: list[dict[str, Any]],
    *,
    manager: str,
    player: str,
    round_number: int | None = None,
    pick_in_round: int | None = None,
    overall: int | None = None,
    position: Position | None = None,
    team: str = "",
    keeper: bool = False,
) -> None:
    picks.append({
        "manager_name": normalize_manager_name(manager),
        "player_name": normalize_player_name(player),
        "round": round_number,
        "pick_in_round": pick_in_round,
        "overall_pick": overall,
        "position": str(position) if position else "",
        "nfl_team": team,
        "keeper_flag": bool(keeper),
    })


def _clean_pick_text(text: str) -> tuple[str, bool]:
    """Strip a trailing ``(keeper)`` marker, reporting whether one was there."""
    if _KEEPER_MARKER_RE.search(text):
        return _KEEPER_MARKER_RE.sub("", text).strip(), True
    return text, False


# ─────────────────────────────────────────────────────────────────────────────
# Layout: round blocks
# ─────────────────────────────────────────────────────────────────────────────
def _parse_round_blocks(
    text: str, result: PasteResult, *, team_count: int | None, snake: bool
) -> list[dict[str, Any]]:
    picks: list[dict[str, Any]] = []
    current_round: int | None = None
    position_in_round = 0

    for number, line in _visible_lines(text):
        header = _ROUND_HEADER_RE.match(line)
        if header:
            current_round = int(header.group(1))
            position_in_round = 0
            continue
        if current_round is None:
            result.unparsed.append(
                UnparsedLine(number, line, "appears before the first round header")
            )
            continue

        body, keeper = _clean_pick_text(line)
        round_number, pick_in_round, overall, body = _take_pick_numbers(body)
        body = _after_numbers(body)
        manager, player_text = _split_manager_and_player(body)
        if not player_text:
            result.unparsed.append(UnparsedLine(number, line, "no player name found"))
            continue
        if not manager:
            result.unparsed.append(
                UnparsedLine(number, line, "no manager could be separated out")
            )
            continue

        position_in_round += 1
        player, pos, team = _strip_position_and_team(player_text)
        _record(
            picks, manager=manager, player=player,
            round_number=round_number or current_round,
            pick_in_round=pick_in_round or position_in_round,
            overall=overall, position=pos, team=team, keeper=keeper,
        )

    inferred = team_count or _infer_team_count(picks)
    _fill_overall_picks(picks, team_count=inferred, result=result)
    return picks


def _take_pick_numbers(
    body: str,
) -> tuple[int | None, int | None, int | None, str]:
    """Pull any leading pick numbering off a line, returning what it meant.

    ``1.05`` gives a round and a pick-in-round; a bare ``17`` gives an overall
    pick. Returned as ``(round, pick_in_round, overall, remainder)``.
    """
    dotted = _DOT_PICK_RE.match(body)
    if dotted:
        return int(dotted.group(1)), int(dotted.group(2)), None, body[dotted.end():]
    numbered = _OVERALL_PICK_RE.match(body)
    if numbered:
        return None, None, int(numbered.group(1)), body[numbered.end():]
    return None, None, None, body


def _after_numbers(body: str) -> str:
    """The remainder of a line once its numbering and the punctuation after it go.

    Separate from :func:`_take_pick_numbers` because the leading separator left
    behind is what would otherwise make ``"1.05\\tAlex\\tBijan"`` split into an
    empty manager and lose the row.
    """
    return body.strip(" \t|.:;-–—")


# ─────────────────────────────────────────────────────────────────────────────
# Layout: flat pick list
# ─────────────────────────────────────────────────────────────────────────────
def _parse_pick_list(
    text: str, result: PasteResult, *, team_count: int | None, snake: bool
) -> list[dict[str, Any]]:
    picks: list[dict[str, Any]] = []
    for number, line in _visible_lines(text):
        if _ROUND_HEADER_RE.match(line):
            continue
        body, keeper = _clean_pick_text(line)
        round_number, pick_in_round, overall, body = _take_pick_numbers(body)
        body = _after_numbers(body)
        manager, player_text = _split_manager_and_player(body)
        if not player_text:
            result.unparsed.append(UnparsedLine(number, line, "no player name found"))
            continue
        if not manager:
            result.unparsed.append(
                UnparsedLine(
                    number, line,
                    "could not tell the manager from the player — a dash, colon or "
                    "tab between them is what marks the split",
                )
            )
            continue
        player, pos, team = _strip_position_and_team(player_text)
        _record(
            picks, manager=manager, player=player, round_number=round_number,
            pick_in_round=pick_in_round, overall=overall, position=pos, team=team,
            keeper=keeper,
        )

    inferred = team_count or _infer_team_count(picks)
    _fill_overall_picks(picks, team_count=inferred, result=result)
    return picks


# ─────────────────────────────────────────────────────────────────────────────
# Layout: team blocks
# ─────────────────────────────────────────────────────────────────────────────
def _parse_team_blocks(
    text: str, result: PasteResult, *, team_count: int | None, snake: bool
) -> list[dict[str, Any]]:
    """One manager per block, their picks beneath in round order.

    The overall pick is not in the text, so it is reconstructed from the draft
    order. That reconstruction is the reason ``snake`` exists and the reason a note
    is added: a linear draft parsed as a snake puts half the picks in the wrong
    place, and nothing in the paste itself reveals which it was.
    """
    blocks: list[tuple[str, list[tuple[int, str]]]] = []
    for number, line in _visible_lines(text):
        if _ROUND_HEADER_RE.match(line):
            continue
        if _looks_like_a_person_only(line):
            blocks.append((line, []))
            continue
        if not blocks:
            result.unparsed.append(
                UnparsedLine(number, line, "appears before the first team name")
            )
            continue
        blocks[-1][1].append((number, line))

    blocks = [(name, rows) for name, rows in blocks if rows]
    if not blocks:
        return []

    picks: list[dict[str, Any]] = []
    slots = {name: index + 1 for index, (name, _) in enumerate(blocks)}
    teams = team_count or len(blocks)
    if team_count and team_count != len(blocks):
        result.notes.append(
            f"The paste has {len(blocks)} team block(s) but the league has "
            f"{team_count} teams, so the reconstructed pick numbers may be off."
        )

    for name, rows in blocks:
        slot = slots[name]
        for order, (number, line) in enumerate(rows, start=1):
            body, keeper = _clean_pick_text(line)
            round_number, pick_in_round, overall, body = _take_pick_numbers(body)
            body = _after_numbers(body)
            round_number = round_number or order
            # The manager is the block header, so anything before a separator on
            # this line is part of the player's own description, not a name.
            _, player_text = _split_manager_and_player(body)
            player_text = player_text or body
            player, pos, team_code = _strip_position_and_team(player_text)
            if not player:
                result.unparsed.append(
                    UnparsedLine(number, line, "no player name found")
                )
                continue
            if overall is None:
                pick_in_round = pick_in_round or _slot_in_round(
                    slot, round_number, teams, snake=snake
                )
                overall = (round_number - 1) * teams + pick_in_round
            _record(
                picks, manager=name, player=player, round_number=round_number,
                pick_in_round=pick_in_round, overall=overall, position=pos,
                team=team_code, keeper=keeper,
            )

    result.notes.append(
        "Team blocks carry no pick numbers, so the order was reconstructed by "
        "treating the blocks as draft slots 1..N in the order they appear"
        + (" and the draft as a snake." if snake else " and the draft as linear.")
    )
    return picks


def _slot_in_round(slot: int, round_number: int, teams: int, *, snake: bool) -> int:
    """Where a draft slot picks within a round."""
    if snake and round_number % 2 == 0:
        return teams - slot + 1
    return slot


# ─────────────────────────────────────────────────────────────────────────────
# Layout: grid
# ─────────────────────────────────────────────────────────────────────────────
def _looks_like_a_data_row(cells: Sequence[str]) -> bool:
    """True when a row of cells is picks rather than the board's header.

    Two signals, both of which a row of team names lacks: a leading cell that is a
    bare round number, and cells carrying position tokens. Without this test a
    headerless board reads its own first round as the list of managers — which
    imports without error and attributes every pick to a player's name.
    """
    if not cells:
        return False
    if re.fullmatch(r"\s*\d{1,2}\s*", cells[0] or ""):
        return True
    positions = sum(
        1 for cell in cells if cell and _strip_position_and_team(cell)[1] is not None
    )
    return positions >= 2


def _parse_grid(
    text: str, result: PasteResult, *, team_count: int | None, snake: bool
) -> list[dict[str, Any]]:
    """A literal draft board: a header row of teams, then one row per round.

    The header is what makes this readable at all, so its absence is an error
    rather than a guess — without team names every cell is an orphan.
    """
    rows = [
        (number, [clean_text(cell) for cell in line.split("\t")])
        for number, line in _visible_lines(text, tabify=True)
    ]
    rows = [(number, cells) for number, cells in rows if any(cells)]
    if len(rows) < 2:
        result.report.error(
            "grid_too_small", "A draft board needs a header row of team names and "
            "at least one round beneath it.",
        )
        return []

    header_number, header = rows[0]
    if _looks_like_a_data_row(header):
        result.report.error(
            "grid_no_header",
            f"Line {header_number} looks like picks rather than a row of team names. "
            "A board without its header cannot be read: the columns would have no "
            "owners, and guessing would credit real picks to the wrong managers. Add "
            "the team names as the first line, or paste the recap by round instead.",
        )
        return []
    # A leading corner cell ("Round", "" or "Rd") offsets every team by one.
    offset = 1 if header and (
        not header[0] or header[0].strip().lower().strip(":#") in {"round", "rd", "r"}
    ) else 0
    managers = [cell for cell in header[offset:] if cell]
    if len(managers) < 2:
        result.report.error(
            "grid_no_header",
            f"Line {header_number} was read as the header row of team names, but it "
            "does not contain at least two names. If the board has no header, the "
            "columns cannot be attributed to anyone.",
        )
        return []

    teams = len(managers)
    if team_count and team_count != teams:
        result.notes.append(
            f"The board has {teams} team column(s) against the league's {team_count}."
        )

    picks: list[dict[str, Any]] = []
    round_number = 0
    for number, cells in rows[1:]:
        leading = cells[0] if cells else ""
        stated = _ROUND_HEADER_RE.match(leading) or re.fullmatch(r"\s*(\d{1,2})\s*", leading)
        body = cells[offset:] if offset else cells
        if stated:
            round_number = int(stated.group(1))
            if offset == 0:
                body = cells[1:]
        else:
            round_number += 1

        if len(body) > teams:
            result.notes.append(
                f"Line {number} has {len(body)} cells for {teams} teams; the extras "
                "were ignored."
            )
        for column, cell in enumerate(body[:teams], start=1):
            if not cell:
                continue
            content, keeper = _clean_pick_text(cell)
            # A board often prints the pick number in the cell ("2.12 Puka Nacua").
            # Where it does, that is the draft's own answer and beats deriving one.
            cell_round, cell_in_round, cell_overall, content = _take_pick_numbers(content)
            content = _after_numbers(content)
            player, pos, team_code = _strip_position_and_team(content)
            if not player:
                result.unparsed.append(
                    UnparsedLine(number, cell, "cell held no readable player name")
                )
                continue
            # A column is a *team*, not a pick position: on a snake board every
            # even round is filled right to left, so the leftmost column picks last
            # in it. Numbering by column would put half the draft in the wrong order
            # and quietly misstate every manager's reach.
            in_round = cell_in_round or _slot_in_round(
                column, cell_round or round_number, teams, snake=snake
            )
            _record(
                picks, manager=managers[column - 1], player=player,
                round_number=cell_round or round_number, pick_in_round=in_round,
                overall=cell_overall
                or ((cell_round or round_number) - 1) * teams + in_round,
                position=pos, team=team_code, keeper=keeper,
            )

    result.notes.append(
        "Read as a draft board: the first row is team names, and each row below it "
        "is one round"
        + (
            " taken as a snake, so even rounds run right to left."
            if snake
            else " taken as a linear draft, so every round runs left to right."
        )
    )
    return picks


# ─────────────────────────────────────────────────────────────────────────────
# Filling in what the text did not say
# ─────────────────────────────────────────────────────────────────────────────
def _infer_team_count(picks: Sequence[dict[str, Any]]) -> int | None:
    """The team count implied by the picks, or ``None`` if nothing implies one.

    Distinct manager names are the strongest evidence and the most likely to be
    right; the widest pick-in-round seen is the fallback.
    """
    names = {p["manager_name"] for p in picks if p["manager_name"]}
    if len(names) >= 2:
        return len(names)
    widest = [p["pick_in_round"] for p in picks if p.get("pick_in_round")]
    return max(widest) if widest else None


def _fill_overall_picks(
    picks: list[dict[str, Any]], *, team_count: int | None, result: PasteResult
) -> None:
    """Derive the overall pick wherever the paste did not carry one.

    ``import_historical_drafts`` requires an overall pick on every row, so a board
    that only says "round 4, third line" still has to produce one. With a team
    count that is arithmetic; without one the rows fall back to their reading
    order, which is right for any board that was printed in pick order.
    """
    missing = [p for p in picks if not p.get("overall_pick")]
    if not missing:
        return

    if team_count:
        for pick in missing:
            round_number = pick.get("round") or 1
            in_round = pick.get("pick_in_round") or 1
            pick["overall_pick"] = (int(round_number) - 1) * int(team_count) + int(in_round)
        result.notes.append(
            f"{len(missing)} pick(s) had no overall number; it was computed from "
            f"round and position with {team_count} teams."
        )
        return

    for index, pick in enumerate(picks, start=1):
        if not pick.get("overall_pick"):
            pick["overall_pick"] = index
    result.notes.append(
        f"{len(missing)} pick(s) had no overall number and the team count could not "
        "be inferred, so the order they were pasted in was used."
    )


def _reconcile_managers(
    picks: list[dict[str, Any]], known: Sequence[str], result: PasteResult
) -> None:
    """Snap parsed names onto the league's own manager names where they agree.

    Matching is exact on the normalised name — the same rule the importer uses. No
    fuzzy matching: quietly deciding that "Mike B" is "Michael Brady" would move
    real picks onto the wrong person's profile, and a reported non-match costs the
    user one edit.
    """
    lookup = {normalize_manager_name(name).lower(): name for name in known if name}
    if not lookup:
        return
    matched = 0
    unknown: set[str] = set()
    for pick in picks:
        name = pick["manager_name"]
        target = lookup.get(name.lower())
        if target:
            pick["manager_name"] = target
            matched += 1
        elif name:
            unknown.add(name)
    if unknown:
        result.notes.append(
            "These names in the paste match no manager in your league, so their "
            "picks will not inform anyone's profile until the names agree: "
            + ", ".join(sorted(unknown))
        )
    if matched:
        result.notes.append(f"{matched} pick(s) matched a manager already in your league.")


def _check_completeness(result: PasteResult, *, team_count: int | None) -> None:
    """Warn about the ways a paste is usually short without looking wrong.

    A recap copied from a browser very often loses its last screen, or a duplicate
    overall pick appears because two blocks were pasted twice. Neither is visible
    in a row count.
    """
    frame = result.frame
    if frame.empty:
        return
    overall = pd.to_numeric(frame["overall_pick"], errors="coerce").dropna()
    if overall.empty:
        return

    duplicated = overall[overall.duplicated()].astype(int).unique().tolist()
    if duplicated:
        result.report.warn(
            "duplicate_picks",
            "These overall pick numbers appear more than once, which usually means "
            "part of the board was pasted twice or the layout was misread: "
            + ", ".join(str(value) for value in sorted(duplicated)[:12])
            + ("…" if len(duplicated) > 12 else ""),
        )

    expected = int(overall.max())
    gaps = sorted(set(range(1, expected + 1)) - set(overall.astype(int).tolist()))
    if gaps:
        result.report.warn(
            "missing_picks",
            f"{len(gaps)} pick number(s) between 1 and {expected} are absent — "
            "most often the tail of a recap that did not all get copied. Missing "
            f"picks: {', '.join(str(g) for g in gaps[:12])}"
            + ("…" if len(gaps) > 12 else ""),
        )

    if team_count and result.managers and len(result.managers) != team_count:
        result.report.warn(
            "manager_count_mismatch",
            f"The paste yielded {len(result.managers)} manager(s) but the league has "
            f"{team_count} teams. Spelling differences count as separate managers.",
        )


__all__ = [
    "PasteResult",
    "UnparsedLine",
    "LAYOUT_LABELS",
    "detect_layout",
    "parse_draft_board",
]
