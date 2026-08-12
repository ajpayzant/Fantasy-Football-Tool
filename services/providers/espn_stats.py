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

from core import stats as core_stats
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
SKILL_POSITIONS = core_stats.SKILL_POSITIONS


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


# ESPN stat id → the canonical name in :mod:`core.stats`. This table is the entire
# ESPN-specific part of scoring: everything past it is provider-agnostic, so ESPN's
# projections and a user's uploaded ones go through exactly one scorer and cannot
# drift apart. Ids absent from this table are absent on purpose — see the module
# docstring on why an unidentified stat is worse than a missing one.
STAT_ID_TO_FIELD: dict[int, str] = {
    PASS_ATTEMPTS: "pass_attempts",
    PASS_COMPLETIONS: "pass_completions",
    PASS_YARDS: "pass_yards",
    PASS_TD: "pass_td",
    PASS_TD_40_PLUS: "pass_td_40_plus",
    PASS_2PT: "pass_2pt",
    INTERCEPTIONS: "interceptions",
    RUSH_ATTEMPTS: "rush_attempts",
    RUSH_YARDS: "rush_yards",
    RUSH_TD: "rush_td",
    RUSH_2PT: "rush_2pt",
    RUSH_TD_40_PLUS: "rush_td_40_plus",
    TARGETS: "targets",
    RECEPTIONS: "receptions",
    REC_YARDS: "rec_yards",
    REC_TD: "rec_td",
    REC_2PT: "rec_2pt",
    REC_TD_40_PLUS: "rec_td_40_plus",
    FUMBLES_LOST: "fumbles_lost",
    GAMES: "games",
    FG_MADE_0_39: "fg_made_0_39",
    FG_MADE_40_49: "fg_made_40_49",
    FG_MADE_50_PLUS: "fg_made_50_plus",
    FG_MADE: "fg_made",
    FG_ATTEMPTED: "fg_attempted",
    XP_MADE: "xp_made",
    DST_SACKS: "dst_sacks",
    DST_INTERCEPTIONS: "dst_interceptions",
    DST_FUMBLES_RECOVERED: "dst_fumbles_recovered",
    DST_SAFETIES: "dst_safeties",
    DST_TOUCHDOWNS: "dst_touchdowns",
    DST_POINTS_ALLOWED: "dst_points_allowed",
    DST_YARDS_ALLOWED: "dst_yards_allowed",
    **{stat_id: f"dst_pa_games_{rule.removeprefix('dst_points_allowed_')}"
       for stat_id, rule in DST_PA_BUCKETS},
}


def to_stat_line(
    stats: Mapping[int, float] | None, position: Position | None = None
) -> dict[str, float]:
    """Translate ESPN's ``{stat_id: total}`` into a canonical stat line.

    Only the stats the position is scored on are kept. ESPN's blocks are disjoint by
    position (offence 0-73, kicking 74-88, defence 89-137), so this filter is not
    needed to disambiguate ids — it is there so a stored stat line does not carry a
    defence's yards-allowed on a wide receiver and invite someone to score it.
    """
    if not stats:
        return {}
    allowed = core_stats.FIELDS_FOR_POSITION.get(position) if position else None
    line: dict[str, float] = {}
    for stat_id, field in STAT_ID_TO_FIELD.items():
        if allowed is not None and field not in allowed:
            continue
        value = _get(stats, stat_id)
        if value:
            # Not rounded. ESPN projects fractional stats (340.06 completions), and
            # rounding here moved Josh Allen's season by 0.002 points — harmless in
            # isolation, but it makes "does the refactor score identically?" an
            # approximate question, and that check is worth keeping exact.
            line[field] = value
    return line


def project_points(
    stats: Mapping[int, float] | None,
    position: Position,
    scoring: ScoringRules,
) -> float | None:
    """Score an ESPN projected stat line under ``scoring``. ``None`` if unscorable.

    Kept as a convenience for callers holding raw ESPN ids. The arithmetic lives in
    :func:`core.stats.score`; this only translates. Callers that will need to
    *re*-score later — anything that persists a player — should store
    :func:`to_stat_line` and score from that instead, so a change of scoring rules
    does not require going back to the network.
    """
    return core_stats.score(to_stat_line(stats, position), position, scoring)


def projected_stat_line(
    stats: Mapping[int, float] | None, position: Position | None
) -> dict[str, float]:
    """``{display label: value}`` for a raw ESPN stat block.

    Delegates so the labels a user reads are the same ones an uploaded projection
    gets. Callers with a stored canonical line should use :func:`core.stats.labelled`
    directly; this exists for the fetch path, which still starts from ESPN ids.
    """
    return core_stats.labelled(to_stat_line(stats, position), position)


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
    "to_stat_line",
    "project_points",
    "projected_stat_line",
    "verify_stat_map",
    "STAT_ID_TO_FIELD",
    "PROJECTED_SOURCE_ID",
    "SEASON_SPLIT_TYPE_ID",
    "DST_PA_BUCKETS",
]
