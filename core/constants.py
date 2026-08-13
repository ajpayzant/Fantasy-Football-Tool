"""Static reference data: slot eligibility, team maps, name-cleaning tables.

Nothing here depends on league configuration — league-specific values live in
:mod:`core.config`.
"""

from __future__ import annotations

from .enums import Position, Slot

APP_NAME = "League-Aware Fantasy Mock Draft"
SCHEMA_VERSION = 3
"""Bumped whenever the SQLite schema changes; stored in ``application_settings``.

v2 added the provenance columns on ``players`` — the stored stat line a projection
can be rescored from, and the per-platform ADP/rank columns that a save-and-reload
used to silently drop.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Slot eligibility — the single source of truth for "can player X fill slot Y".
# Deliberately data-driven so no code path assumes 1QB/2RB/2WR/1TE.
# ─────────────────────────────────────────────────────────────────────────────
SLOT_ELIGIBILITY: dict[Slot, frozenset[Position]] = {
    Slot.QB: frozenset({Position.QB}),
    Slot.RB: frozenset({Position.RB}),
    Slot.WR: frozenset({Position.WR}),
    Slot.TE: frozenset({Position.TE}),
    Slot.FLEX: frozenset({Position.RB, Position.WR, Position.TE}),
    Slot.WR_RB_FLEX: frozenset({Position.RB, Position.WR}),
    Slot.WR_TE_FLEX: frozenset({Position.WR, Position.TE}),
    Slot.SUPERFLEX: frozenset({Position.QB, Position.RB, Position.WR, Position.TE}),
    Slot.K: frozenset({Position.K}),
    Slot.DST: frozenset({Position.DST}),
    Slot.BENCH: frozenset(Position),
    Slot.IR: frozenset(Position),
}

STARTING_SLOTS: tuple[Slot, ...] = (
    Slot.QB, Slot.RB, Slot.WR, Slot.TE,
    Slot.FLEX, Slot.WR_RB_FLEX, Slot.WR_TE_FLEX, Slot.SUPERFLEX,
    Slot.K, Slot.DST,
)
"""Slots that count toward a starting lineup (excludes BN / IR)."""

# Order matters: when auto-assigning a roster we fill the most restrictive
# slots first so a flex-eligible player is never wasted on a dedicated slot.
SLOT_FILL_PRIORITY: tuple[Slot, ...] = (
    Slot.QB, Slot.RB, Slot.WR, Slot.TE, Slot.K, Slot.DST,
    Slot.WR_RB_FLEX, Slot.WR_TE_FLEX, Slot.FLEX, Slot.SUPERFLEX,
)

# ─────────────────────────────────────────────────────────────────────────────
# NFL teams
# ─────────────────────────────────────────────────────────────────────────────
NFL_TEAMS: tuple[str, ...] = (
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
)

TEAM_ALIASES: dict[str, str] = {
    # Relocations / historical abbreviations
    "OAK": "LV", "RAI": "LV", "LVR": "LV", "RAIDERS": "LV",
    "SD": "LAC", "SDG": "LAC", "CHARGERS": "LAC",
    "STL": "LAR", "RAM": "LAR", "RAMS": "LAR",
    "WSH": "WAS", "WFT": "WAS", "COMMANDERS": "WAS", "REDSKINS": "WAS",
    # Common alternate abbreviations across platforms
    "JAC": "JAX", "JAGUARS": "JAX",
    "GNB": "GB", "PACKERS": "GB",
    "KAN": "KC", "CHIEFS": "KC",
    "NWE": "NE", "PATRIOTS": "NE",
    "NOR": "NO", "SAINTS": "NO",
    "SFO": "SF", "49ERS": "SF", "NINERS": "SF",
    "TAM": "TB", "BUCCANEERS": "TB", "BUCS": "TB",
    "ARZ": "ARI", "CARDINALS": "ARI",
    "BLT": "BAL", "RAVENS": "BAL",
    "CLV": "CLE", "BROWNS": "CLE",
    "HST": "HOU", "TEXANS": "HOU",
    "LA": "LAR",
    # Free agents / unknown
    "FA": "FA", "FREE AGENT": "FA", "NONE": "FA", "": "FA",
}

POSITION_ALIASES: dict[str, str] = {
    "QB": "QB", "Q B": "QB", "QUARTERBACK": "QB",
    "RB": "RB", "HB": "RB", "FB": "RB", "RUNNINGBACK": "RB", "RUNNING BACK": "RB",
    "WR": "WR", "WIDERECEIVER": "WR", "WIDE RECEIVER": "WR",
    "TE": "TE", "TIGHTEND": "TE", "TIGHT END": "TE",
    "K": "K", "PK": "K", "KICKER": "K",
    "DST": "DST", "DEF": "DST", "D/ST": "DST", "DEFENSE": "DST",
    "DEF/ST": "DST", "TEAM DEFENSE": "DST", "D": "DST", "DL": "DST",
}

NAME_SUFFIXES: frozenset[str] = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})
"""Stripped during name normalisation so "A.J. Brown Jr." matches "AJ Brown"."""

# ─────────────────────────────────────────────────────────────────────────────
# Positional replacement baselines (starters consumed league-wide before a
# position's value flattens). Used for value-over-replacement when the player
# file does not supply VOR directly. Scaled by team count at runtime.
# ─────────────────────────────────────────────────────────────────────────────
REPLACEMENT_RANK_PER_TEAM: dict[Position, float] = {
    Position.QB: 1.2,
    Position.RB: 2.6,
    Position.WR: 3.2,
    Position.TE: 1.2,
    Position.K: 1.0,
    Position.DST: 1.0,
}

POSITION_COLORS: dict[str, str] = {
    "QB": "#C7522A",
    "RB": "#2E7D6F",
    "WR": "#3A6EA5",
    "TE": "#8A6BBE",
    "K": "#7A7A7A",
    "DST": "#4F5D2F",
}
"""Colour-blind-safe-ish position palette for the draft board."""

# ─────────────────────────────────────────────────────────────────────────────
# Import templates — canonical column sets exposed as downloadable CSVs.
# ─────────────────────────────────────────────────────────────────────────────
HISTORICAL_IMPORT_COLUMNS: tuple[str, ...] = (
    "season", "league_name", "platform", "manager_name", "round",
    "pick_in_round", "overall_pick", "player_name", "position", "nfl_team",
    "adp", "platform_rank", "projection", "tier", "keeper_flag",
    "rookie_flag", "draft_date",
)
HISTORICAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "season", "manager_name", "overall_pick", "player_name",
)

PLAYER_IMPORT_COLUMNS: tuple[str, ...] = (
    "player_name", "position", "nfl_team", "bye_week", "experience",
    "rookie_flag", "injury_status", "projection", "overall_rank",
    "position_rank", "platform_rank", "overall_adp", "platform_adp",
    "adp_stdev", "min_pick", "max_pick", "tier", "ceiling", "floor",
    "risk_score", "value_over_replacement", "notes",
    # Per-platform columns. These must be listed here and not only in the alias
    # table, because ``canonical_column`` consults this set first — which is exactly
    # what stops ``espn_adp`` and ``yahoo_adp`` from both collapsing onto
    # ``platform_adp``. A user's own file with an "ESPN ADP" header lands here too.
    "ffc_adp", "espn_adp", "espn_rank", "yahoo_adp", "yahoo_rank", "sleeper_rank",
    "adp_source_count", "adp_disagreement", "adp_stdev_is_estimated",
    # Where the projection came from, and what it is made of. ``stat_totals`` is the
    # machine-readable version of ``projection_detail``: a JSON stat line keyed by the
    # canonical names in :mod:`core.stats`, which is what allows a projection to be
    # rescored when scoring rules change instead of refetched.
    "projection_source", "projection_detail", "stat_totals",
)
PLAYER_REQUIRED_COLUMNS: tuple[str, ...] = ("player_name", "position")

SAMPLE_DATA_BANNER = (
    "SAMPLE DATA — fictional players and drafts for demonstration only. "
    "Not real NFL players, rankings, or ADP."
)
