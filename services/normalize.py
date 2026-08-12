"""Normalisation helpers shared by every importer.

Uploaded spreadsheets differ in column naming, name punctuation, team
abbreviations and position labels. Everything here is deterministic and pure so
the same file always normalises the same way, and so tests can assert on it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from core.constants import (
    HISTORICAL_IMPORT_COLUMNS,
    NAME_SUFFIXES,
    NFL_TEAMS,
    PLAYER_IMPORT_COLUMNS,
    POSITION_ALIASES,
    TEAM_ALIASES,
)
from core.enums import Position

# ─────────────────────────────────────────────────────────────────────────────
# Column header aliases
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_ALIASES: dict[str, str] = {
    # Identity
    "player": "player_name", "name": "player_name", "playername": "player_name",
    "full_name": "player_name", "fullname": "player_name", "player_full_name": "player_name",
    "pos": "position", "position_": "position", "player_position": "position",
    "team": "nfl_team", "tm": "nfl_team", "nfl": "nfl_team", "pro_team": "nfl_team",
    "nflteam": "nfl_team", "team_abbr": "nfl_team", "nfl_abbr": "nfl_team",
    # Draft slots
    "manager": "manager_name", "owner": "manager_name", "owner_name": "manager_name",
    "team_owner": "manager_name", "drafted_by": "manager_name", "fantasy_manager": "manager_name",
    "team_name": "manager_name",
    "overall": "overall_pick", "pick": "overall_pick", "pick_no": "overall_pick",
    "pick_number": "overall_pick", "overall_selection": "overall_pick",
    "draft_pick": "overall_pick", "pick_overall": "overall_pick", "no": "overall_pick",
    "rnd": "round", "round_no": "round", "round_number": "round", "draft_round": "round",
    "pick_in_rnd": "pick_in_round", "round_pick": "pick_in_round",
    "pick_slot": "pick_in_round", "slot": "pick_in_round",
    "year": "season", "draft_year": "season", "season_year": "season",
    "league": "league_name", "league_title": "league_name",
    "site": "platform", "source": "platform", "host": "platform",
    "date": "draft_date", "drafted_at": "draft_date",
    # Values
    "average_draft_position": "adp", "avg_pick": "adp", "adp_overall": "adp",
    "consensus_adp": "adp", "sleeper_adp": "sleeper_rank",
    # ``espn_adp``, ``yahoo_adp``, ``espn_rank`` and friends are *not* aliased onto
    # ``platform_adp``/``platform_rank`` any more: they are canonical columns in their
    # own right (see ``PLAYER_IMPORT_COLUMNS``), because folding two platforms into
    # one column meant the second one silently became ``platform_adp_2`` and vanished.
    # ``import_player_pool`` fills ``platform_adp`` from them when it is absent, so a
    # file carrying only an ESPN column still drives the engine's platform lens.
    "espn_average_draft_position": "espn_adp", "espn_average_pick": "espn_adp",
    "yahoo_average_draft_position": "yahoo_adp", "yahoo_average_pick": "yahoo_adp",
    "espn_ranking": "espn_rank", "yahoo_ranking": "yahoo_rank",
    "ffc_average_draft_position": "ffc_adp", "calculator_adp": "ffc_adp",
    "rank": "overall_rank", "overall_ranking": "overall_rank", "ovr_rank": "overall_rank",
    "ecr": "overall_rank", "consensus_rank": "overall_rank",
    "pos_rank": "position_rank", "positional_rank": "position_rank",
    "site_rank": "platform_rank", "default_rank": "platform_rank",
    "proj": "projection", "projected_points": "projection", "fpts": "projection",
    "points": "projection", "fantasy_points": "projection", "proj_pts": "projection",
    "high": "ceiling", "best_case": "ceiling", "upside": "ceiling",
    "low": "floor", "worst_case": "floor", "downside": "floor",
    "sd": "adp_stdev", "stdev": "adp_stdev", "std_dev": "adp_stdev",
    "adp_sd": "adp_stdev", "adp_std": "adp_stdev",
    "earliest": "min_pick", "best_pick": "min_pick",
    "latest": "max_pick", "worst_pick": "max_pick",
    "bye": "bye_week", "bye_wk": "bye_week",
    "exp": "experience", "years_exp": "experience", "yoe": "experience",
    "keeper": "keeper_flag", "is_keeper": "keeper_flag", "kept": "keeper_flag",
    "rookie": "rookie_flag", "is_rookie": "rookie_flag", "rk": "rookie_flag",
    "injury": "injury_status", "status": "injury_status", "inj": "injury_status",
    "risk": "risk_score", "vor": "value_over_replacement", "vorp": "value_over_replacement",
    "tier_number": "tier", "tier_group": "tier",
}
"""Loose header → canonical column name. Keys are already snake-cased."""

KNOWN_COLUMNS: frozenset[str] = frozenset(
    HISTORICAL_IMPORT_COLUMNS + PLAYER_IMPORT_COLUMNS + ("round", "keeper_flag")
)

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_SPACE_RE = re.compile(r"\s+")
_HEADER_RE = re.compile(r"[^a-z0-9]+")


def snake_header(raw: Any) -> str:
    """Convert an arbitrary spreadsheet header into a snake_case token."""
    text = unicodedata.normalize("NFKD", str(raw)).strip().lower()
    text = _HEADER_RE.sub("_", text).strip("_")
    return text


def canonical_column(raw: Any) -> str:
    """Map a header to its canonical name, leaving unknown headers snake-cased."""
    token = snake_header(raw)
    if token in KNOWN_COLUMNS:
        return token
    return COLUMN_ALIASES.get(token, token)


def normalize_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename every column to its canonical form.

    Returns the renamed copy and a ``{original: canonical}`` map for the UI so
    the user can see how their headers were interpreted. Duplicate canonical
    names keep the first occurrence and suffix the rest, so no data is lost.
    """
    mapping: dict[str, str] = {}
    seen: dict[str, int] = {}
    new_names: list[str] = []
    for original in frame.columns:
        canonical = canonical_column(original)
        count = seen.get(canonical, 0)
        seen[canonical] = count + 1
        final = canonical if count == 0 else f"{canonical}_{count + 1}"
        mapping[str(original)] = final
        new_names.append(final)
    out = frame.copy()
    out.columns = new_names
    return out, mapping


# ─────────────────────────────────────────────────────────────────────────────
# Value normalisation
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(value: Any) -> str:
    """Trim, collapse whitespace, and drop pandas nulls to an empty string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return _SPACE_RE.sub(" ", str(value).replace(" ", " ")).strip()


def normalize_player_name(value: Any) -> str:
    """Tidy a display name: collapse spacing, fix casing of ALL-CAPS entries.

    ``"SMITH, JOHN"`` → ``"John Smith"``. Names already in mixed case keep their
    capitalisation so ``"A.J. Brown"`` and ``"DeVonta Smith"`` survive intact.
    """
    text = clean_text(value)
    if not text:
        return ""
    if "," in text and text.count(",") == 1:
        last, first = (part.strip() for part in text.split(","))
        if first and last:
            # A generational suffix belongs at the end, not between the names:
            # "MAHOMES, PATRICK JR." → "Patrick Mahomes Jr."
            first_tokens = first.split(" ")
            suffix = ""
            if len(first_tokens) > 1 and first_tokens[-1].strip(".").lower() in NAME_SUFFIXES:
                suffix = first_tokens.pop()
                first = " ".join(first_tokens)
            text = f"{first} {last} {suffix}".strip()
    if text.isupper() or text.islower():
        text = " ".join(_title_token(t) for t in text.split(" "))
    return text


def _title_token(token: str) -> str:
    """Title-case a token, preserving internal punctuation (``a.j.`` → ``A.J.``)."""
    if not token:
        return token
    if "." in token:
        return ".".join(part.capitalize() for part in token.split("."))
    if "-" in token:
        return "-".join(part.capitalize() for part in token.split("-"))
    if "'" in token:
        head, _, tail = token.partition("'")
        return f"{head.capitalize()}'{tail.capitalize()}"
    return token.capitalize()


def player_key(name: Any, position: Position | str | None = None) -> str:
    """Stable join key: lowercase, punctuation-free, suffix-stripped name.

    Position is appended when known so a WR and a DST sharing a city name do not
    collide. ``"A.J. Brown Jr."`` and ``"AJ Brown"`` produce the same key.
    """
    text = unicodedata.normalize("NFKD", clean_text(name)).encode("ascii", "ignore").decode()
    text = _PUNCT_RE.sub("", text.lower())
    tokens = [t for t in text.split() if t and t not in NAME_SUFFIXES]
    base = "".join(tokens)
    if position is None:
        return base
    coerced = Position.coerce(position, None)
    return f"{base}_{coerced}".lower() if coerced else base


def normalize_team(value: Any) -> str:
    """Map any team spelling to a canonical NFL abbreviation (``FA`` if unknown)."""
    text = clean_text(value).upper().replace(".", "")
    if not text:
        return "FA"
    if text in NFL_TEAMS:
        return text
    if text in TEAM_ALIASES:
        return TEAM_ALIASES[text]
    stripped = _PUNCT_RE.sub("", text.lower()).upper()
    if stripped in NFL_TEAMS:
        return stripped
    return TEAM_ALIASES.get(stripped, "FA")


def normalize_position(value: Any) -> Position | None:
    """Map a loose position label to a :class:`Position`, or ``None`` if unknown."""
    text = clean_text(value).upper().replace(".", "")
    if not text:
        return None
    # Platform files often suffix the position rank: "WR12", "RB3".
    match = re.match(r"^([A-Z/]+)\s*\d*$", text)
    if match:
        text = match.group(1)
    alias = POSITION_ALIASES.get(text)
    if alias:
        return Position.coerce(alias, None)
    return Position.coerce(text, None)


def normalize_manager_name(value: Any) -> str:
    """Tidy a manager/owner name without forcing a case convention."""
    text = clean_text(value)
    if text.isupper() and len(text) > 3:
        text = " ".join(_title_token(t) for t in text.split(" "))
    return text


def normalize_season(value: Any) -> int | None:
    """Parse a season year, accepting ``2024``, ``'24``, and ``2024-25``."""
    text = clean_text(value)
    if not text:
        return None
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return int(match.group(0))
    digits = re.sub(r"\D", "", text)
    if len(digits) == 2:
        # Two-digit years are assumed to be 2000s: fantasy football data
        # predating 2000 is not a realistic input for this tool.
        return 2000 + int(digits)
    return None


def normalize_injury_status(value: Any) -> str:
    """Map platform injury codes to the canonical label set."""
    text = clean_text(value).lower()
    if not text or text in {"a", "active", "healthy", "ok", "-"}:
        return "Healthy"
    if text.startswith("q") or "question" in text:
        return "Questionable"
    if text.startswith("d") or "doubt" in text:
        return "Doubtful"
    if text in {"o", "out"} or "out" == text:
        return "Out"
    if "ir" in text or "injured reserve" in text:
        return "IR"
    if "pup" in text:
        return "IR"
    if "susp" in text:
        return "Suspended"
    return "Healthy"


def dedupe_names(names: Iterable[str]) -> dict[str, str]:
    """Group manager spellings that normalise to the same key.

    Returns ``{original: canonical}`` where the canonical spelling is the most
    common (ties broken by first appearance) — so "Mike", "mike" and "MIKE "
    collapse onto one manager without the user editing the file.
    """
    from collections import Counter

    from models.manager import normalize_manager_key

    buckets: dict[str, Counter] = {}
    order: dict[str, int] = {}
    for index, name in enumerate(names):
        cleaned = normalize_manager_name(name)
        if not cleaned:
            continue
        key = normalize_manager_key(cleaned)
        buckets.setdefault(key, Counter())[cleaned] += 1
        order.setdefault(cleaned, index)

    canonical: dict[str, str] = {}
    for key, counter in buckets.items():
        best = max(counter.items(), key=lambda kv: (kv[1], -order[kv[0]]))[0]
        for spelling in counter:
            canonical[spelling] = best
    return canonical


def coerce_frame_columns(
    frame: pd.DataFrame, wanted: Sequence[str]
) -> pd.DataFrame:
    """Return a copy containing every ``wanted`` column, adding missing ones as NA."""
    out = frame.copy()
    for column in wanted:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def describe_column_mapping(mapping: Mapping[str, str]) -> pd.DataFrame:
    """Tabulate how headers were interpreted, for display after an import."""
    rows = [
        {
            "your_column": original,
            "read_as": canonical,
            "recognised": canonical in KNOWN_COLUMNS,
        }
        for original, canonical in mapping.items()
    ]
    return pd.DataFrame(rows)


__all__ = [
    "COLUMN_ALIASES", "KNOWN_COLUMNS", "snake_header", "canonical_column",
    "normalize_columns", "clean_text", "normalize_player_name", "player_key",
    "normalize_team", "normalize_position", "normalize_manager_name",
    "normalize_season", "normalize_injury_status", "dedupe_names",
    "coerce_frame_columns", "describe_column_mapping",
]
