"""A projected stat line, and the one place points are computed from one.

Why this module exists
----------------------
Projections used to be scored once, at fetch time, inside the ESPN provider — and
only the resulting *points* survived, alongside a formatted string for display. That
had one visible consequence and one invisible one.

The visible one: changing your scoring rules could not change your projections. The
Setup page had to tell you to re-download everything from ESPN to see half-PPR
become full PPR, which is a network round trip to recompute arithmetic the app
already had all the inputs for.

The invisible one: the moment a *second* source of projections exists — a user's own
upload — there would be two scoring implementations, and they would drift. A league
with a 6-point passing TD would then value ESPN's Josh Allen and an uploaded Josh
Allen differently for no reason a user could ever discover.

So a stat line is stored as *stats*, and points are derived from it on demand, here.

The stat vocabulary
-------------------
Keys are canonical names (:data:`STAT_FIELDS`), not any provider's stat ids. Three
reasons: ESPN's ids are undocumented integers that mean nothing in a CSV a user
opens; the same vocabulary has to describe an uploaded projection, which arrives
with human column headers; and a second provider could be added without teaching
every consumer a new numbering.

A stat line is a plain ``dict[str, float]`` holding only what is known, rather than a
dataclass with a field per stat. Uploads supply subsets — plenty of projection sheets
carry rushing and receiving but not two-point conversions — and a dict keeps "this
sheet did not mention it" cheap to represent. For *scoring*, an absent stat and a
zero stat are the same thing (:func:`get`). For *display* they are not, which is why
:func:`describe` omits absent keys instead of printing zeros.

What is deliberately not scored
-------------------------------
The per-game yardage bonuses (300-yard passing games and friends) are not applied.
Projections are season totals, and how many individual games cleared a threshold
cannot be recovered from a season sum. Their defaults are 0.0, so nothing is silently
dropped unless a user turns one on, and :func:`unscorable_rules` names them so the UI
can say so out loud rather than quietly under-reporting.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

from core.config import ScoringRules
from core.enums import Position

# ─────────────────────────────────────────────────────────────────────────────
# The vocabulary
# ─────────────────────────────────────────────────────────────────────────────
# Grouped by the block of the game they describe. Order is the display order.
PASSING_FIELDS: tuple[str, ...] = (
    "pass_attempts", "pass_completions", "pass_yards", "pass_td",
    "interceptions", "pass_2pt", "pass_td_40_plus",
)
RUSHING_FIELDS: tuple[str, ...] = (
    "rush_attempts", "rush_yards", "rush_td", "rush_2pt", "rush_td_40_plus",
)
RECEIVING_FIELDS: tuple[str, ...] = (
    "targets", "receptions", "rec_yards", "rec_td", "rec_2pt", "rec_td_40_plus",
)
MISC_FIELDS: tuple[str, ...] = ("fumbles_lost", "games")
KICKING_FIELDS: tuple[str, ...] = (
    "fg_made_0_39", "fg_made_40_49", "fg_made_50_plus", "fg_made", "fg_attempted",
    "xp_made",
)
# ``dst_pa_games_*`` hold the expected *number of games* the defence finishes in that
# points-allowed band, not a points total. They are the largest single component of
# real defence scoring, which is why they are carried per band rather than collapsed.
DST_FIELDS: tuple[str, ...] = (
    "dst_sacks", "dst_interceptions", "dst_fumbles_recovered", "dst_safeties",
    "dst_touchdowns", "dst_points_allowed", "dst_yards_allowed",
    "dst_pa_games_0", "dst_pa_games_1_6", "dst_pa_games_7_13", "dst_pa_games_14_17",
    "dst_pa_games_18_21", "dst_pa_games_22_27", "dst_pa_games_28_34",
    "dst_pa_games_35_plus",
)

STAT_FIELDS: tuple[str, ...] = (
    PASSING_FIELDS + RUSHING_FIELDS + RECEIVING_FIELDS + MISC_FIELDS
    + KICKING_FIELDS + DST_FIELDS
)
STAT_FIELD_SET = frozenset(STAT_FIELDS)

# Each points-allowed band paired with the ScoringRules attribute that prices it.
DST_PA_BANDS: tuple[tuple[str, str], ...] = (
    ("dst_pa_games_0", "dst_points_allowed_0"),
    ("dst_pa_games_1_6", "dst_points_allowed_1_6"),
    ("dst_pa_games_7_13", "dst_points_allowed_7_13"),
    ("dst_pa_games_14_17", "dst_points_allowed_14_17"),
    ("dst_pa_games_18_21", "dst_points_allowed_18_21"),
    ("dst_pa_games_22_27", "dst_points_allowed_22_27"),
    ("dst_pa_games_28_34", "dst_points_allowed_28_34"),
    ("dst_pa_games_35_plus", "dst_points_allowed_35_plus"),
)

SKILL_POSITIONS = frozenset({Position.QB, Position.RB, Position.WR, Position.TE})

# Labels for display. Only these are shown, in this order.
STAT_LABELS: dict[str, str] = {
    "pass_attempts": "Pass attempts", "pass_completions": "Completions",
    "pass_yards": "Pass yards", "pass_td": "Pass TD", "interceptions": "Interceptions",
    "pass_2pt": "Pass 2PT", "pass_td_40_plus": "Pass TD 40+",
    "rush_attempts": "Carries", "rush_yards": "Rush yards", "rush_td": "Rush TD",
    "rush_2pt": "Rush 2PT", "rush_td_40_plus": "Rush TD 40+",
    "targets": "Targets", "receptions": "Receptions", "rec_yards": "Rec yards",
    "rec_td": "Rec TD", "rec_2pt": "Rec 2PT", "rec_td_40_plus": "Rec TD 40+",
    "fumbles_lost": "Fumbles lost", "games": "Games",
    "fg_made_0_39": "FG made 0-39", "fg_made_40_49": "FG made 40-49",
    "fg_made_50_plus": "FG made 50+", "fg_made": "FG made",
    "fg_attempted": "FG attempted", "xp_made": "XP made",
    "dst_sacks": "Sacks", "dst_interceptions": "Interceptions",
    "dst_fumbles_recovered": "Fumbles recovered", "dst_safeties": "Safeties",
    "dst_touchdowns": "Touchdowns", "dst_points_allowed": "Points allowed",
    "dst_yards_allowed": "Yards allowed",
    "dst_pa_games_0": "Games allowing 0", "dst_pa_games_1_6": "Games allowing 1-6",
    "dst_pa_games_7_13": "Games allowing 7-13",
    "dst_pa_games_14_17": "Games allowing 14-17",
    "dst_pa_games_18_21": "Games allowing 18-21",
    "dst_pa_games_22_27": "Games allowing 22-27",
    "dst_pa_games_28_34": "Games allowing 28-34",
    "dst_pa_games_35_plus": "Games allowing 35+",
}

# Which stats each position's score is built from, so an upload can be told it gave a
# quarterback nothing but kicking stats.
FIELDS_FOR_POSITION: dict[Position, frozenset[str]] = {
    Position.QB: frozenset(PASSING_FIELDS + RUSHING_FIELDS + RECEIVING_FIELDS + MISC_FIELDS),
    Position.RB: frozenset(RUSHING_FIELDS + RECEIVING_FIELDS + PASSING_FIELDS + MISC_FIELDS),
    Position.WR: frozenset(RECEIVING_FIELDS + RUSHING_FIELDS + PASSING_FIELDS + MISC_FIELDS),
    Position.TE: frozenset(RECEIVING_FIELDS + RUSHING_FIELDS + MISC_FIELDS),
    Position.K: frozenset(KICKING_FIELDS + MISC_FIELDS),
    Position.DST: frozenset(DST_FIELDS + MISC_FIELDS),
}

# ─────────────────────────────────────────────────────────────────────────────
# Accepting a stat line typed or exported by a human
# ─────────────────────────────────────────────────────────────────────────────
# Every spelling here was chosen because it appears on a real projection export or is
# the obvious thing someone would type. Matching is done after lowercasing and
# collapsing punctuation to underscores, so "Rec. Yds" and "rec yds" land together and
# only genuinely different words need an entry.
_ALIASES: dict[str, str] = {
    # Passing
    "pass_att": "pass_attempts", "att": "pass_attempts", "attempts": "pass_attempts",
    "pa": "pass_attempts", "cmp": "pass_completions", "comp": "pass_completions",
    "completions": "pass_completions", "pass_cmp": "pass_completions",
    "pass_yds": "pass_yards", "passing_yards": "pass_yards", "py": "pass_yards",
    "pyds": "pass_yards", "pass_tds": "pass_td", "passing_td": "pass_td",
    "passing_tds": "pass_td", "ptd": "pass_td", "td_pass": "pass_td",
    "int": "interceptions", "ints": "interceptions", "interception": "interceptions",
    "int_thrown": "interceptions", "pass_int": "interceptions",
    "pass_2pc": "pass_2pt", "two_pt_pass": "pass_2pt",
    # Rushing
    "rush_att": "rush_attempts", "carries": "rush_attempts", "car": "rush_attempts",
    "rushing_attempts": "rush_attempts", "ra": "rush_attempts",
    "rush_yds": "rush_yards", "rushing_yards": "rush_yards", "ry": "rush_yards",
    "ryds": "rush_yards", "rush_tds": "rush_td", "rushing_td": "rush_td",
    "rushing_tds": "rush_td", "rtd": "rush_td",
    # Receiving
    "tgt": "targets", "tgts": "targets", "target": "targets",
    "rec": "receptions", "recs": "receptions", "catches": "receptions",
    "reception": "receptions", "rec_yds": "rec_yards", "receiving_yards": "rec_yards",
    "recv_yards": "rec_yards", "reyds": "rec_yards", "rec_tds": "rec_td",
    "receiving_td": "rec_td", "receiving_tds": "rec_td", "retd": "rec_td",
    # Misc
    "fum": "fumbles_lost", "fumbles": "fumbles_lost", "fum_lost": "fumbles_lost",
    "fumbles_l": "fumbles_lost", "fl": "fumbles_lost",
    "g": "games", "gp": "games", "games_played": "games",
    # Kicking
    "fgm": "fg_made", "fg": "fg_made", "fga": "fg_attempted",
    "fgm_0_39": "fg_made_0_39", "fgm_40_49": "fg_made_40_49",
    "fgm_50": "fg_made_50_plus", "fgm_50_plus": "fg_made_50_plus",
    "fg_50": "fg_made_50_plus", "xpm": "xp_made", "xp": "xp_made",
    "pat": "xp_made", "pat_made": "xp_made",
    # Defence
    "sacks": "dst_sacks", "sack": "dst_sacks", "sk": "dst_sacks",
    "def_sacks": "dst_sacks", "def_int": "dst_interceptions",
    "def_interceptions": "dst_interceptions", "fumbles_recovered": "dst_fumbles_recovered",
    "fr": "dst_fumbles_recovered", "def_fr": "dst_fumbles_recovered",
    "safeties": "dst_safeties", "sfty": "dst_safeties",
    "def_td": "dst_touchdowns", "def_tds": "dst_touchdowns",
    "defensive_td": "dst_touchdowns", "points_allowed": "dst_points_allowed",
    "pa_total": "dst_points_allowed", "yards_allowed": "dst_yards_allowed",
    "ya": "dst_yards_allowed",
}

_PUNCTUATION = re.compile(r"[^a-z0-9]+")


def canonical_field(name: object) -> str | None:
    """Map an arbitrary column header onto a canonical stat name, or ``None``.

    ``None`` means "not a stat this app scores". Callers report those back to the
    user rather than dropping them silently — a column the app ignored is exactly
    what makes an uploaded projection come out lower than the user expected.
    """
    text = _PUNCTUATION.sub("_", str(name or "").strip().lower()).strip("_")
    if not text:
        return None
    if text in STAT_FIELD_SET:
        return text
    return _ALIASES.get(text)


def normalise(raw: Mapping[object, object] | None) -> dict[str, float]:
    """Coerce an arbitrary mapping into a canonical stat line.

    Unrecognised keys are dropped and non-numeric values are skipped; use
    :func:`unrecognised_fields` first when the caller needs to report them. Zeroes
    are kept when explicitly given, because "projected for zero carries" is a real
    statement about a wide receiver and differs from silence.
    """
    if not raw:
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        field = canonical_field(key)
        if field is None:
            continue
        number = _as_float(value)
        if number is None:
            continue
        out[field] = number
    return out


def unrecognised_fields(raw: Mapping[object, object] | None) -> list[str]:
    """Keys :func:`normalise` would discard, in the order given."""
    if not raw:
        return []
    return [str(key) for key in raw if canonical_field(key) is None]


def _as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None
    # NaN never survives: ESPN emits it for stats it tracks but has not projected,
    # and it would poison every sum it touched.
    return None if number != number else number


def get(stats: Mapping[str, float] | None, field: str) -> float:
    """A stat's value, treating absent and unparseable as ``0.0``."""
    if not stats:
        return 0.0
    return _as_float(stats.get(field)) or 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
def score(
    stats: Mapping[str, float] | None,
    position: Position | None,
    scoring: ScoringRules,
) -> float | None:
    """Points for a stat line under ``scoring``, or ``None`` if unscorable.

    ``None`` rather than ``0.0`` is deliberate and load-bearing: the player pool
    treats an absent projection as something to derive from draft position and says
    so on the row, whereas a confident zero would rank a real player below every
    imputed one. So a stat line with no yardage, no receptions and no scores is not
    a player projected to do nothing — it is a player nobody projected.
    """
    if not stats:
        return None
    if position is Position.DST:
        return _score_dst(stats, scoring)
    if position is Position.K:
        return _score_kicker(stats, scoring)
    if position in SKILL_POSITIONS:
        return _score_skill(stats, position, scoring)
    return None


def _score_skill(
    stats: Mapping[str, float], position: Position, scoring: ScoringRules
) -> float | None:
    pass_yards = get(stats, "pass_yards")
    rush_yards = get(stats, "rush_yards")
    rec_yards = get(stats, "rec_yards")
    receptions = get(stats, "receptions")
    pass_td = get(stats, "pass_td")
    rush_td = get(stats, "rush_td")
    rec_td = get(stats, "rec_td")
    if not any((pass_yards, rush_yards, rec_yards, receptions,
                pass_td + rush_td + rec_td)):
        return None

    total = 0.0
    if scoring.pass_yards_per_point:
        total += pass_yards / scoring.pass_yards_per_point
    total += pass_td * scoring.pass_td
    total += get(stats, "interceptions") * scoring.interception
    total += get(stats, "pass_2pt") * scoring.pass_2pt

    if scoring.rush_yards_per_point:
        total += rush_yards / scoring.rush_yards_per_point
    total += rush_td * scoring.rush_td

    if scoring.rec_yards_per_point:
        total += rec_yards / scoring.rec_yards_per_point
    total += rec_td * scoring.rec_td
    total += receptions * scoring.reception_value(position)

    total += (get(stats, "rush_2pt") + get(stats, "rec_2pt")) * scoring.rush_rec_2pt
    total += get(stats, "fumbles_lost") * scoring.fumble_lost

    if scoring.bonus_long_td_40_plus:
        total += (
            get(stats, "pass_td_40_plus")
            + get(stats, "rush_td_40_plus")
            + get(stats, "rec_td_40_plus")
        ) * scoring.bonus_long_td_40_plus

    return float(total)


def _score_kicker(stats: Mapping[str, float], scoring: ScoringRules) -> float | None:
    # Prefer the distance buckets over the plain total: they sum to it, and keeping
    # them separate is what lets a league that pays more for long kicks be scored
    # correctly if that ever becomes a rule.
    made = (
        get(stats, "fg_made_0_39")
        + get(stats, "fg_made_40_49")
        + get(stats, "fg_made_50_plus")
    )
    if not made:
        made = get(stats, "fg_made")
    extra_points = get(stats, "xp_made")
    if not made and not extra_points:
        return None
    return float(made * scoring.kick_fg_made + extra_points * scoring.kick_xp_made)


def _score_dst(stats: Mapping[str, float], scoring: ScoringRules) -> float | None:
    sacks = get(stats, "dst_sacks")
    interceptions = get(stats, "dst_interceptions")
    fumbles = get(stats, "dst_fumbles_recovered")
    touchdowns = get(stats, "dst_touchdowns")
    bands = {field: get(stats, field) for field, _ in DST_PA_BANDS}
    if not any((sacks, interceptions, fumbles, touchdowns)) and not any(bands.values()):
        return None

    total = (
        sacks * scoring.dst_sack
        + interceptions * scoring.dst_interception
        + fumbles * scoring.dst_fumble_recovery
        + touchdowns * scoring.dst_touchdown
        + get(stats, "dst_safeties") * scoring.dst_safety
    )
    # Each band holds the expected number of games finishing in it, so the
    # contribution is games × the band's per-game value.
    for field, rule in DST_PA_BANDS:
        total += bands[field] * float(getattr(scoring, rule, 0.0))
    return float(total)


def unscorable_rules(scoring: ScoringRules) -> list[str]:
    """Scoring rules the user has switched on that a season total cannot express.

    Returned so the UI can name them. Every one is a per-game threshold: a season
    projection of 4,200 passing yards does not say how many individual games cleared
    300, so applying the bonus would mean inventing a game log.
    """
    per_game = (
        ("bonus_pass_300_yards", "300-yard passing games"),
        ("bonus_pass_400_yards", "400-yard passing games"),
        ("bonus_rush_100_yards", "100-yard rushing games"),
        ("bonus_rush_200_yards", "200-yard rushing games"),
        ("bonus_rec_100_yards", "100-yard receiving games"),
        ("bonus_rec_200_yards", "200-yard receiving games"),
    )
    return [label for attribute, label in per_game if float(getattr(scoring, attribute, 0.0))]


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────
def relevant_fields(position: Position | None) -> tuple[str, ...]:
    """The stats worth showing for a position, in display order."""
    allowed = FIELDS_FOR_POSITION.get(position) if position else None
    if allowed is None:
        return STAT_FIELDS
    return tuple(field for field in STAT_FIELDS if field in allowed)


def labelled(
    stats: Mapping[str, float] | None, position: Position | None = None
) -> dict[str, float]:
    """``{display label: value}`` for the non-zero stats relevant to the position.

    Zeroes are dropped here even though :func:`normalise` keeps them, because a stat
    line reading "0 pass yards · 0 pass TD" for a running back is noise the reader
    has to filter themselves.
    """
    if not stats:
        return {}
    out: dict[str, float] = {}
    for field in relevant_fields(position):
        value = get(stats, field)
        if value:
            out[STAT_LABELS.get(field, field)] = round(value, 1)
    return out


def describe(
    stats: Mapping[str, float] | None, position: Position | None = None
) -> str:
    """One short line naming what a projection is made of.

    Games are excluded: they are context for the other numbers rather than a
    projection of production, and they crowd out the stats being explained.
    """
    line = labelled(stats, position)
    parts = [
        f"{value:,.0f} {label.lower()}" if abs(value) >= 10 else f"{value:.1f} {label.lower()}"
        for label, value in line.items()
        if label != "Games"
    ]
    return " · ".join(parts)


def is_empty(stats: Mapping[str, float] | None) -> bool:
    """True when nothing here could contribute to a score."""
    return not any(get(stats, field) for field in STAT_FIELDS) if stats else True


def merge(
    base: Mapping[str, float] | None, override: Mapping[str, float] | None
) -> dict[str, float]:
    """``base`` with ``override``'s stats replacing it field by field.

    Field-by-field rather than wholesale so a user who uploads only receiving stats
    for a running back does not silently erase his rushing projection.
    """
    out = dict(base or {})
    for field, value in (override or {}).items():
        if field in STAT_FIELD_SET:
            number = _as_float(value)
            if number is not None:
                out[field] = number
    return out


def coverage(
    stats: Mapping[str, float] | None, position: Position | None
) -> tuple[int, int]:
    """``(stats supplied, stats this position scores on)`` — for reporting quality.

    Lets an importer say "this row gave 3 of the 17 stats a running back scores on"
    instead of accepting a projection built from almost nothing without comment.
    """
    fields = relevant_fields(position)
    supplied = sum(1 for field in fields if get(stats, field))
    return supplied, len(fields)


def to_frame_value(stats: Mapping[str, float] | None) -> str:
    """Encode a stat line for a DataFrame cell / CSV column.

    JSON rather than a bespoke format because these round-trip through pandas, CSV
    export and SQLite, and a user may well open the CSV and read this cell.
    """
    import json

    if not stats:
        return ""
    # Six decimals, not three: providers project fractional stats, and the coarser
    # rounding shifted a quarterback's season by ~0.002 points on the round trip.
    # Nobody would notice that in a projection, but it turns "the stored line scores
    # identically to the fetched one" into an approximate claim, and that claim is
    # the whole point of storing the line.
    ordered = {field: round(float(stats[field]), 6)
               for field in STAT_FIELDS if field in stats}
    return json.dumps(ordered, separators=(",", ":")) if ordered else ""


def from_frame_value(value: object) -> dict[str, float]:
    """Decode whatever :func:`to_frame_value` wrote, tolerating junk.

    Tolerant because this reads back user-editable CSV and a decade-old database
    row: a malformed cell costs one player's stat detail, and must not fail a load.
    """
    import json

    if value is None or isinstance(value, float) and value != value:
        return {}
    if isinstance(value, Mapping):
        return normalise(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "{}"}:
        return {}
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return normalise(parsed) if isinstance(parsed, Mapping) else {}


def summarise_fields(fields: Iterable[str]) -> str:
    """Human list of canonical field names, for validation messages."""
    labels = [STAT_LABELS.get(field, field) for field in fields]
    return ", ".join(labels)


__all__ = [
    "STAT_FIELDS", "STAT_FIELD_SET", "STAT_LABELS", "DST_PA_BANDS",
    "PASSING_FIELDS", "RUSHING_FIELDS", "RECEIVING_FIELDS", "MISC_FIELDS",
    "KICKING_FIELDS", "DST_FIELDS", "FIELDS_FOR_POSITION",
    "canonical_field", "normalise", "unrecognised_fields", "get", "score",
    "unscorable_rules", "relevant_fields", "labelled", "describe", "is_empty",
    "merge", "coverage", "to_frame_value", "from_frame_value", "summarise_fields",
]
