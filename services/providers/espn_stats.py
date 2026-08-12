"""Decode ESPN's projected stat line and score it under the league's own rules.

ESPN's player payload carries a real season projection, but reading it takes care
on two counts, and both are the reason this module exists rather than a one-liner
in :mod:`services.providers.espn`.

**Finding the projection.** Each player's ``stats`` list holds dozens of entries
spanning several seasons and every week. The season projection is the single entry
where ``statSourceId == 1`` (projected, not actual), ``statSplitTypeId == 0``
(season, not weekly), ``scoringPeriodId == 0`` and ``seasonId`` is the season
asked for — equivalently ``id == f"10{season}"``. Every other entry in that list is
either a weekly split (``appliedTotal`` is ``None`` for all of them, pre-season) or
a prior season.

**Not trusting ``appliedTotal``.** That field is ESPN's own points total under an
unspecified default ruleset, and it is *wrong* often enough to be unusable:

* Kicker Brandon Aubrey comes back at **5,237.8**, because ESPN's default ruleset
  scores the field-goal *yardage* stats (214/215/216 — made, missed and attempted
  FG yards, 1,371 + 206 = 1,577 in his case) as though they were points.
* Tight end Trey McBride comes back at **5.36**, which is exactly his projected
  receiving-touchdown count (stat 43) and nothing else — a top-five TE valued at
  the weight of one flex bench player.

So this module ignores ``appliedTotal`` and scores the raw projected stat totals
with :class:`~core.config.ScoringRules`. That is not merely a workaround: it means
the projection respects *the user's* league — half-PPR vs full PPR vs TE premium,
their field-goal value, their points-allowed tiers — rather than ESPN's fixed one.

**How the stat ids below were established.** They are derived from the payload's
own internal arithmetic and cross-checked against known NFL season totals, not
transcribed from an undocumented list:

* ``1 + 2 == 0`` — completions plus incompletions equals attempts (Josh Allen:
  340.06 + 168.75 == 508.81).
* ``74 + 77 + 80 == 83`` — field goals made from 0-39, 40-49 and 50+ sum to total
  field goals made (Aubrey: 7.03 + 9.23 + 19.23 == 35.48). ``86 + 88 == 87`` does
  the same for extra points.
* ``129..136`` — the points-allowed buckets sum to exactly **17.0** for every
  defence, i.e. one entry per game of the season.
* ``120 / 210 == 126`` and ``127 / 210 == 137`` — total points and yards allowed
  divided by games equals the published per-game figures.
* Scaled to 32 teams, stat **99** totals ~1,343 sacks against an NFL 2024 actual of
  1,290; stat **105** (~70 defensive touchdowns) matches the sum of its own
  sub-types 101-104 (~66).

Anything not verified this way is deliberately absent. An unscored stat costs a few
points of accuracy; a misidentified one silently corrupts every projection.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

from core.config import ScoringRules
from core.enums import Position

LOGGER = logging.getLogger("fantasy_mock_draft.providers.espn_stats")

# Which ``stats`` entry is the season projection.
PROJECTED_SOURCE_ID = 1
SEASON_SPLIT_TYPE_ID = 0

# ── Offence ──────────────────────────────────────────────────────────────────
PASS_ATTEMPTS = 0
PASS_COMPLETIONS = 1
PASS_INCOMPLETIONS = 2
PASS_YARDS = 3
PASS_TD = 4
PASS_TD_40_PLUS = 15
PASS_2PT = 19
INTERCEPTIONS = 20

RUSH_ATTEMPTS = 23
RUSH_YARDS = 24
RUSH_TD = 25
RUSH_2PT = 26
RUSH_TD_40_PLUS = 35

REC_YARDS = 42
REC_TD = 43
REC_2PT = 44
REC_TD_40_PLUS = 45
RECEPTIONS = 53
TARGETS = 58

FUMBLES_LOST = 72

# ── Kicking ──────────────────────────────────────────────────────────────────
FG_MADE_0_39 = 74
FG_MADE_40_49 = 77
FG_MADE_50_PLUS = 80
FG_MADE = 83
FG_ATTEMPTED = 84
XP_MADE = 86
XP_ATTEMPTED = 87

# ── Defence / special teams ──────────────────────────────────────────────────
DST_INTERCEPTIONS = 95
DST_FUMBLES_RECOVERED = 96
DST_SAFETIES = 98
DST_SACKS = 99
DST_TOUCHDOWNS = 105
DST_POINTS_ALLOWED = 120
DST_YARDS_ALLOWED = 127

# Points-allowed buckets, in the order ScoringRules names them. These sum to the
# number of games, which is the check that identifies them.
DST_PA_BUCKETS: tuple[tuple[int, str], ...] = (
    (129, "dst_points_allowed_0"),
    (130, "dst_points_allowed_1_6"),
    (131, "dst_points_allowed_7_13"),
    (132, "dst_points_allowed_14_17"),
    (133, "dst_points_allowed_18_21"),
    (134, "dst_points_allowed_22_27"),
    (135, "dst_points_allowed_28_34"),
    (136, "dst_points_allowed_35_plus"),
)

GAMES = 210

# Positions scored from the offensive stat block. K and DST use their own.
SKILL_POSITIONS = frozenset({Position.QB, Position.RB, Position.WR, Position.TE})


def season_projection_stats(
    player: Mapping[str, Any], season: int
) -> dict[int, float] | None:
    """Return ``{stat_id: projected_total}`` for a season, or ``None``.

    ``None`` means this player carries no season projection — which is normal for
    the deep-roster players ESPN lists but nobody drafts, so callers treat it as an
    absent value rather than an error.
    """
    entries = player.get("stats")
    if not isinstance(entries, list):
        return None
    wanted_id = f"10{int(season)}"
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("id") != wanted_id:
            continue
        if entry.get("statSourceId") != PROJECTED_SOURCE_ID:
            continue
        if entry.get("statSplitTypeId") != SEASON_SPLIT_TYPE_ID:
            continue
        raw = entry.get("stats")
        if not isinstance(raw, dict) or not raw:
            return None
        stats: dict[int, float] = {}
        for key, value in raw.items():
            try:
                stats[int(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return stats or None
    return None


def _get(stats: Mapping[int, float], stat_id: int) -> float:
    value = stats.get(stat_id)
    if value is None:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    # ESPN emits NaN for stats it tracks but has no projection for.
    return 0.0 if parsed != parsed else parsed


def project_points(
    stats: Mapping[int, float] | None,
    position: Position,
    scoring: ScoringRules,
) -> float | None:
    """Score a projected stat line under ``scoring``. ``None`` if unscoreable.

    The stat blocks are disjoint by position (offence 0-73, kicking 74-88, defence
    89-137), so the position only selects which block to read — it never has to
    disambiguate an id.
    """
    if not stats:
        return None

    if position is Position.DST:
        return _project_dst(stats, scoring)
    if position is Position.K:
        return _project_kicker(stats, scoring)
    if position in SKILL_POSITIONS:
        return _project_skill(stats, position, scoring)
    return None


def _project_skill(
    stats: Mapping[int, float], position: Position, scoring: ScoringRules
) -> float | None:
    pass_yards = _get(stats, PASS_YARDS)
    rush_yards = _get(stats, RUSH_YARDS)
    rec_yards = _get(stats, REC_YARDS)
    receptions = _get(stats, RECEPTIONS)
    touchdowns = (
        _get(stats, PASS_TD) + _get(stats, RUSH_TD) + _get(stats, REC_TD)
    )
    # A stat line with no yardage and no scores carries no information — better an
    # absent projection the pool can flag than a confident 0.0.
    if not any((pass_yards, rush_yards, rec_yards, receptions, touchdowns)):
        return None

    total = 0.0
    if scoring.pass_yards_per_point:
        total += pass_yards / scoring.pass_yards_per_point
    total += _get(stats, PASS_TD) * scoring.pass_td
    total += _get(stats, INTERCEPTIONS) * scoring.interception
    total += _get(stats, PASS_2PT) * scoring.pass_2pt

    if scoring.rush_yards_per_point:
        total += rush_yards / scoring.rush_yards_per_point
    total += _get(stats, RUSH_TD) * scoring.rush_td

    if scoring.rec_yards_per_point:
        total += rec_yards / scoring.rec_yards_per_point
    total += _get(stats, REC_TD) * scoring.rec_td
    total += receptions * scoring.reception_value(position)

    total += (
        _get(stats, RUSH_2PT) + _get(stats, REC_2PT)
    ) * scoring.rush_rec_2pt
    total += _get(stats, FUMBLES_LOST) * scoring.fumble_lost

    if scoring.bonus_long_td_40_plus:
        long_tds = (
            _get(stats, PASS_TD_40_PLUS)
            + _get(stats, RUSH_TD_40_PLUS)
            + _get(stats, REC_TD_40_PLUS)
        )
        total += long_tds * scoring.bonus_long_td_40_plus

    # The per-game yardage bonuses (300-yard passing games and friends) are
    # deliberately not applied: ESPN publishes season totals, and how many
    # individual games cleared a threshold cannot be recovered from a season sum.
    # Their defaults are 0.0, so nothing is silently dropped unless a user turns
    # one on — which the Settings page says.
    return float(total)


def _project_kicker(
    stats: Mapping[int, float], scoring: ScoringRules
) -> float | None:
    # Prefer the distance buckets, which sum to the published total (74+77+80==83)
    # and let a league that pays more for long kicks be scored correctly later.
    made = (
        _get(stats, FG_MADE_0_39)
        + _get(stats, FG_MADE_40_49)
        + _get(stats, FG_MADE_50_PLUS)
    )
    if not made:
        made = _get(stats, FG_MADE)
    extra_points = _get(stats, XP_MADE)
    if not made and not extra_points:
        return None
    return float(made * scoring.kick_fg_made + extra_points * scoring.kick_xp_made)


def _project_dst(
    stats: Mapping[int, float], scoring: ScoringRules
) -> float | None:
    sacks = _get(stats, DST_SACKS)
    interceptions = _get(stats, DST_INTERCEPTIONS)
    fumbles = _get(stats, DST_FUMBLES_RECOVERED)
    touchdowns = _get(stats, DST_TOUCHDOWNS)
    buckets = {name: _get(stats, key) for key, name in DST_PA_BUCKETS}
    if not any((sacks, interceptions, fumbles, touchdowns)) and not any(buckets.values()):
        return None

    total = (
        sacks * scoring.dst_sack
        + interceptions * scoring.dst_interception
        + fumbles * scoring.dst_fumble_recovery
        + touchdowns * scoring.dst_touchdown
        + _get(stats, DST_SAFETIES) * scoring.dst_safety
    )
    # Points-allowed tiers are the largest single component of real defence
    # scoring, and each bucket holds the expected *number of games* finishing in
    # that band, so the contribution is games × the band's value.
    for name, games in buckets.items():
        total += games * float(getattr(scoring, name, 0.0))
    return float(total)


def projected_stat_line(
    stats: Mapping[int, float] | None, position: Position
) -> dict[str, float]:
    """Human-readable projected stats, for showing *why* a projection is what it is.

    Keys are display labels; only the stats relevant to the position are returned,
    and only where ESPN projected something.
    """
    if not stats:
        return {}
    if position is Position.DST:
        wanted = [
            ("Sacks", DST_SACKS),
            ("Interceptions", DST_INTERCEPTIONS),
            ("Fumbles recovered", DST_FUMBLES_RECOVERED),
            ("Touchdowns", DST_TOUCHDOWNS),
            ("Points allowed", DST_POINTS_ALLOWED),
            ("Yards allowed", DST_YARDS_ALLOWED),
        ]
    elif position is Position.K:
        wanted = [
            ("FG made", FG_MADE),
            ("FG attempted", FG_ATTEMPTED),
            ("FG made 50+", FG_MADE_50_PLUS),
            ("XP made", XP_MADE),
        ]
    else:
        wanted = [
            ("Pass yards", PASS_YARDS),
            ("Pass TD", PASS_TD),
            ("Interceptions", INTERCEPTIONS),
            ("Carries", RUSH_ATTEMPTS),
            ("Rush yards", RUSH_YARDS),
            ("Rush TD", RUSH_TD),
            ("Targets", TARGETS),
            ("Receptions", RECEPTIONS),
            ("Rec yards", REC_YARDS),
            ("Rec TD", REC_TD),
            ("Fumbles lost", FUMBLES_LOST),
        ]
    line: dict[str, float] = {}
    for label, stat_id in wanted:
        value = _get(stats, stat_id)
        if value:
            line[label] = round(value, 1)
    games = _get(stats, GAMES)
    if games:
        line["Games"] = round(games, 1)
    return line


def verify_stat_map(records: Iterable[Mapping[str, Any]], season: int) -> list[str]:
    """Check the identities documented above against a real payload.

    Returns a list of failure descriptions — empty means the map still holds.
    Exists so a change in ESPN's payload shows up as a test failure naming the
    identity that broke, rather than as quietly wrong projections.
    """
    problems: list[str] = []
    checked = {"passing": 0, "kicking": 0, "defence": 0}

    for record in records:
        player = record.get("player", record) if isinstance(record, dict) else None
        if not isinstance(player, dict):
            continue
        stats = season_projection_stats(player, season)
        if not stats:
            continue
        name = str(player.get("fullName") or player.get("id") or "?")

        attempts = _get(stats, PASS_ATTEMPTS)
        if attempts > 1:
            checked["passing"] += 1
            parts = _get(stats, PASS_COMPLETIONS) + _get(stats, PASS_INCOMPLETIONS)
            if abs(parts - attempts) > 0.5:
                problems.append(
                    f"{name}: completions + incompletions ({parts:.2f}) != "
                    f"attempts ({attempts:.2f}) — ids 1/2/0 have moved."
                )

        total_fg = _get(stats, FG_MADE)
        if total_fg > 1:
            checked["kicking"] += 1
            buckets = (
                _get(stats, FG_MADE_0_39)
                + _get(stats, FG_MADE_40_49)
                + _get(stats, FG_MADE_50_PLUS)
            )
            if abs(buckets - total_fg) > 0.5:
                problems.append(
                    f"{name}: field goals by distance ({buckets:.2f}) != total field "
                    f"goals made ({total_fg:.2f}) — ids 74/77/80/83 have moved."
                )

        games = _get(stats, GAMES)
        pa_games = sum(_get(stats, key) for key, _ in DST_PA_BUCKETS)
        if pa_games > 1:
            checked["defence"] += 1
            if games and abs(pa_games - games) > 0.5:
                problems.append(
                    f"{name}: points-allowed buckets cover {pa_games:.2f} games but "
                    f"the season is {games:.2f} — ids 129-136 have moved."
                )

    for block, count in checked.items():
        if not count:
            problems.append(
                f"No player exercised the {block} identity check — the payload no "
                "longer carries that stat block, so the map is unverified."
            )
    return problems


__all__ = [
    "season_projection_stats",
    "project_points",
    "projected_stat_line",
    "verify_stat_map",
    "PROJECTED_SOURCE_ID",
    "SEASON_SPLIT_TYPE_ID",
    "DST_PA_BUCKETS",
]
