"""Player records and the in-memory player pool used by the engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from core import freshness as core_freshness
from core import stats as core_stats
from core.config import LeagueConfig, RosterSettings, ScoringRules
from core.constants import SLOT_ELIGIBILITY, SLOT_FILL_PRIORITY
from core.enums import InjuryStatus, Position, ProjectionMode, RankingSource, Slot

# Injury statuses that make a player undraftable-ish; scored as a penalty, not
# a hard filter, because managers do stash injured players.
INJURY_PENALTY: dict[InjuryStatus, float] = {
    InjuryStatus.HEALTHY: 0.00,
    InjuryStatus.QUESTIONABLE: 0.05,
    InjuryStatus.DOUBTFUL: 0.20,
    InjuryStatus.OUT: 0.45,
    InjuryStatus.PUP: 0.55,
    InjuryStatus.IR: 0.80,
    InjuryStatus.SUSPENDED: 0.60,
}

# Half-width of the outcome band, in standard deviations. 1.28σ is the 10th/90th
# percentile of a normal, so ceiling and floor bracket an 80% interval — wide enough
# to be informative, narrow enough not to rest on the tails, where the normal
# assumption fits fantasy scoring worst.
_BAND_Z = 1.28

# Risk weights. Relative disagreement leads because it is the only component measured
# from real data rather than read off a status label.
_RISK_WEIGHT_DISAGREEMENT = 0.65
_RISK_WEIGHT_INJURY = 0.25
_RISK_WEIGHT_ROOKIE = 0.10

# The σ/ADP ratio treated as maximum disagreement. FFC's published spreads run at
# roughly 0.20 of ADP across the board (see ``resolver.ESTIMATED_STDEV_FRACTION``),
# so twice that is a player the room genuinely cannot place. Normalising against a
# fixed ratio rather than the pool's own worst case keeps a risk score comparable
# between a 200-player board and a 400-player one.
_DISAGREEMENT_AT_FULL_RISK = 0.40

# The two ``projection_source`` strings this app writes itself. Named, because
# :meth:`PlayerPool.rescore` has to be able to tell its own placeholder from a line a
# provider or a user wrote: it may replace the former and must never overwrite the
# latter. Before they were named, a projection scored from a stat line kept whichever
# of these was set at import time and went on claiming to be an estimate.
IMPUTED_PROJECTION_SOURCE = (
    "Estimated from draft position — no real projection was supplied for this player"
)
GENERIC_PROJECTION_SOURCE = "Supplied by your source"
STAT_LINE_PROJECTION_SOURCE = (
    "Computed from this player's projected stat line under your league's scoring rules"
)


def _read_curve(slots: np.ndarray, curve: np.ndarray, at: float) -> float:
    """Read a monotone-decreasing curve at ``at``, extrapolating past either end.

    ``np.interp`` clamps outside its range, which would hand the best player at a
    position a ceiling identical to their projection — no upside at all for exactly
    the players whose upside decides a draft. Past each end the local slope of the two
    nearest points is continued instead, so the consensus best QB still has room above
    him and the last player on the board still has room below.
    """
    if len(curve) == 0:
        return 0.0
    if len(curve) == 1:
        return float(curve[0])
    if at < 0.0:
        slope = float(curve[0] - curve[1])  # positive: the curve falls as slot rises
        return float(curve[0] + slope * -at)
    last = len(curve) - 1
    if at > last:
        slope = float(curve[last - 1] - curve[last])
        return float(curve[last] - slope * (at - last))
    return float(np.interp(at, slots, curve))


@dataclass(slots=True)
class Player:
    """A draftable player.

    Only ``player_id``, ``name`` and ``position`` are guaranteed. Every ranking
    or projection field may be ``None``; :class:`PlayerPool` imputes what the
    engine needs and records that it did so.
    """

    player_id: str
    name: str
    position: Position
    nfl_team: str = "FA"
    bye_week: int | None = None
    experience: int | None = None
    is_rookie: bool = False
    injury_status: InjuryStatus = InjuryStatus.HEALTHY
    suspended: bool = False
    projection: float | None = None
    position_projection: float | None = None
    overall_rank: float | None = None
    position_rank: int | None = None
    platform_rank: float | None = None
    overall_adp: float | None = None
    platform_adp: float | None = None
    adp_stdev: float | None = None
    min_pick: int | None = None
    max_pick: int | None = None
    tier: int | None = None
    ceiling: float | None = None
    floor: float | None = None
    risk_score: float | None = None
    expected_points: float | None = None
    replacement_points: float | None = None
    value_over_replacement: float | None = None
    eligible_slots: tuple[Slot, ...] = ()
    notes: str = ""
    source: str = ""

    # -- provenance ------------------------------------------------------
    # Each platform's own number, kept beside the consensus rather than blended into
    # it. Any of these may be None: no source covers every player, and a blank cell
    # is the honest answer for "what does Yahoo think of this player".
    ffc_adp: float | None = None
    espn_adp: float | None = None
    espn_rank: float | None = None
    yahoo_adp: float | None = None
    yahoo_rank: float | None = None
    sleeper_rank: float | None = None
    adp_source_count: int | None = None
    adp_disagreement: float | None = None
    # True when the spread was estimated from draft position rather than measured
    # from real mock drafts. Ceiling, floor and risk are all derived from the spread,
    # so this flag governs how much any of them should be trusted.
    adp_stdev_is_estimated: bool = False
    # Free text, shown verbatim in the UI. Empty means "not established", which the
    # pool fills in when it imputes.
    projection_source: str = ""
    projection_detail: str = ""
    tier_source: str = ""
    outcome_band_source: str = ""

    # The projected stat line this player's points were computed from, keyed by the
    # canonical names in :mod:`core.stats`. Empty when no source supplied stats.
    #
    # This is what makes a projection re-derivable. ``projection`` is the answer under
    # one particular set of scoring rules; these are the inputs, so changing from
    # half-PPR to full PPR is arithmetic the app can do offline rather than a reason to
    # re-download the season from ESPN.
    stat_totals: dict[str, float] = field(default_factory=dict)
    # True when ``projection`` was estimated from draft position rather than supplied.
    # Needed for rescoring, not just for display: imputed projections are anchored to
    # the range of the real ones, so when the real ones move to a new scoring scale the
    # imputed ones have to be thrown away and re-derived, and this is how they are
    # found. Matching on the wording of ``projection_source`` would work until someone
    # rephrased it.
    projection_imputed: bool = False

    def __post_init__(self) -> None:
        self.position = Position.coerce(self.position, Position.RB) or Position.RB
        self.injury_status = (
            InjuryStatus.coerce(self.injury_status, InjuryStatus.HEALTHY)
            or InjuryStatus.HEALTHY
        )

    # -- derived helpers -------------------------------------------------
    @property
    def display(self) -> str:
        return f"{self.name} ({self.position} - {self.nfl_team})"

    @property
    def injury_penalty(self) -> float:
        """0-1 penalty magnitude from injury / suspension status."""
        base = INJURY_PENALTY.get(self.injury_status, 0.0)
        if self.suspended:
            base = max(base, INJURY_PENALTY[InjuryStatus.SUSPENDED])
        return base

    @property
    def is_available_flag(self) -> bool:
        """False only for statuses that make a player unusable all season."""
        return self.injury_status not in (InjuryStatus.IR,) and not self.suspended

    @property
    def upside(self) -> float:
        """Ceiling above projection, in points. 0 when no ceiling supplied."""
        if self.ceiling is None or self.projection is None:
            return 0.0
        return max(0.0, float(self.ceiling) - float(self.projection))

    @property
    def downside(self) -> float:
        if self.floor is None or self.projection is None:
            return 0.0
        return max(0.0, float(self.projection) - float(self.floor))

    def fills_slot(self, slot: Slot) -> bool:
        return self.position in SLOT_ELIGIBILITY.get(slot, frozenset())

    def adp_for(self, prefer_platform: bool = True) -> float | None:
        """Best available ADP, preferring platform-specific when present."""
        order = (
            (self.platform_adp, self.overall_adp) if prefer_platform
            else (self.overall_adp, self.platform_adp)
        )
        for value in order:
            if value is not None and not pd.isna(value):
                return float(value)
        return None

    def rank_for(self, prefer_platform: bool = True) -> float | None:
        order = (
            (self.platform_rank, self.overall_rank) if prefer_platform
            else (self.overall_rank, self.platform_rank)
        )
        for value in order:
            if value is not None and not pd.isna(value):
                return float(value)
        return None

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        raw = asdict(self)
        raw["position"] = str(self.position)
        raw["injury_status"] = str(self.injury_status)
        raw["eligible_slots"] = [str(s) for s in self.eligible_slots]
        return raw


def _s(count: int) -> str:
    """"" or "s" — these counts are read by a person, and "1 estimates" reads badly."""
    return "" if count == 1 else "s"


@dataclass(slots=True)
class RescoreOutcome:
    """What :meth:`PlayerPool.rescore` actually managed to do.

    Returned rather than logged because the honest answer is usually partial — ESPN
    does not publish a stat line for every player it ranks — and a user changing
    their scoring rules deserves to be told how much of the board really moved.
    """

    rescored: int = 0
    """Players whose points were recomputed from their own stat line."""
    reimputed: int = 0
    """Players whose projection was an estimate and was re-derived on the new scale."""
    no_stat_line: int = 0
    """Players left untouched: a real projection with no stats behind it to redo."""
    unscorable_rules: list[str] = field(default_factory=list)
    """Scoring rules that cannot be applied to season totals at all."""

    @property
    def changed(self) -> int:
        return self.rescored + self.reimputed

    def describe(self) -> str:
        if not self.changed:
            return "No projections could be rescored — no stored stat lines to work from."
        bits = [
            f"{self.rescored} projection{_s(self.rescored)} recomputed from "
            f"{'its' if self.rescored == 1 else 'their'} stat line"
            f"{_s(self.rescored)}"
        ]
        if self.reimputed:
            bits.append(
                f"{self.reimputed} estimate{_s(self.reimputed)} re-derived on the "
                "new scale"
            )
        if self.no_stat_line:
            bits.append(
                f"{self.no_stat_line} left as-is (a projection with no stat line "
                "behind it cannot be rescored)"
            )
        return "; ".join(bits) + "."


@dataclass(slots=True)
class ProjectionUpdate:
    """One player's projection as supplied by the user, already matched to the board.

    ``player_id`` is resolved by the importer, not here: matching a spreadsheet name
    to a board name needs the alias and suffix handling that lives in
    :mod:`services.normalize`, and this module deliberately depends on nothing below
    :mod:`core`. By the time an update reaches the pool the hard part is done.

    ``stat_totals`` and ``points`` are both optional but at least one is present.
    Stats are strictly better: they survive a scoring change (see
    :meth:`PlayerPool.rescore`), a points total cannot.
    """

    player_id: str
    name: str = ""
    stat_totals: dict[str, float] = field(default_factory=dict)
    points: float | None = None


@dataclass(slots=True)
class ProjectionOutcome:
    """What :meth:`PlayerPool.apply_projections` did, in terms a user can check.

    Counts rather than a boolean because an upload is almost never total: a sheet
    covers 200 of 400 players, gives stats for some and a points total for others, and
    names a handful the board has never heard of. Reporting that is the difference
    between a user trusting the board and wondering why their sleeper did not move.
    """

    from_stats: int = 0
    """Players whose projection now comes from stats you uploaded."""
    from_points: int = 0
    """Players given a points total with no stats behind it — frozen at these rules."""
    blended: int = 0
    """Players whose projection is an average of yours and the board's."""
    skipped_had_real: int = 0
    """Fill-gaps only: players left alone because a real projection was already there."""
    partial_merge: int = 0
    """Players where stats your file did not mention were kept from the old projection."""
    unmatched_ids: list[str] = field(default_factory=list)
    """Updates naming a player the pool no longer holds — a stale match list."""
    rescore: RescoreOutcome | None = None
    """The rescore that turned the new stat lines into points."""

    @property
    def applied(self) -> int:
        return self.from_stats + self.from_points

    def describe(self) -> str:
        bits = []
        if self.from_stats:
            bits.append(
                f"{self.from_stats} projection{_s(self.from_stats)} built from your "
                f"stat line{_s(self.from_stats)} and scored under your league rules"
            )
        if self.from_points:
            bits.append(
                f"{self.from_points} taken as your points total"
                f"{_s(self.from_points)} as-is"
            )
        if self.blended:
            bits.append(f"{self.blended} averaged with what was already there")
        if self.skipped_had_real:
            bits.append(
                f"{self.skipped_had_real} left alone (a real projection was already "
                "on the board)"
            )
        if not bits:
            # Reached when every row matched a player and none of them were used, which
            # is not the same as "nothing matched" and must not be reported as if it were.
            return "Nothing was applied — no uploaded row matched a player on the board."
        return "; ".join(bits) + "."


@dataclass(slots=True)
class PoolMetadata:
    """Provenance for a loaded player pool — surfaced verbatim in the UI."""

    source: str = "unknown"
    imported_at: str = ""
    """When the *data* was retrieved, not when this object was built.

    For a live board this is the oldest contributing source's fetch time, so a board
    assembled partly from an expired cache dates itself from the cache rather than
    from the moment the user pressed the button.
    """
    timestamp_basis: str = core_freshness.FETCHED
    """Whether :attr:`imported_at` is when the data was *fetched* or merely *loaded*.

    A fetched board's timestamp is the age of the data. An uploaded file's is the age
    of the upload and nothing more — a spreadsheet handed over a minute ago can hold
    numbers from last August, and no column in it says so. Kept apart so the warning
    can be accurate about which of the two it is measuring.
    """
    season: int | None = None
    platform: str | None = None
    player_count: int = 0
    is_sample_data: bool = False
    missing_fields: dict[str, int] = field(default_factory=dict)
    imputed_fields: dict[str, int] = field(default_factory=dict)
    notes: str = ""

    def freshness(
        self, *, expected_season: int | None = None
    ) -> core_freshness.FreshnessVerdict:
        """How old this pool is, and whether it is even the right season.

        Delegated rather than decided here so the sidebar, the Setup page and the
        Draft Room cannot disagree about what "stale" means. Pass the season being
        drafted to have a last-season board reported as wrong rather than merely old.
        """
        return core_freshness.assess(
            self.imported_at,
            season=self.season,
            expected_season=expected_season,
            basis=self.timestamp_basis,
        )

    def describe(self) -> str:
        bits = [f"{self.player_count} players", f"source: {self.source}"]
        if self.season:
            bits.append(f"season {self.season}")
        if self.platform:
            bits.append(f"platform {self.platform}")
        if self.imported_at:
            # The age, not just the timestamp: a raw ISO string asks the reader to
            # do the subtraction, and the whole reason this line exists is that
            # nobody does.
            age = self.freshness().age_label()
            bits.append(
                f"loaded {age.replace(' old', ' ago')}"
                if self.timestamp_basis == core_freshness.IMPORTED
                else f"data {age}"
            )
        else:
            bits.append("age unknown")
        if self.is_sample_data:
            bits.insert(0, "SAMPLE DATA")
        return " • ".join(bits)


class PlayerPool:
    """Indexed collection of players with league-aware derived values.

    Construction imputes missing ADP / rank / projection fields so the engine
    always has a usable ordering, and records what was imputed in
    :attr:`metadata` rather than hiding it.
    """

    __slots__ = (
        "_players", "_by_id", "_by_name", "metadata", "league",
        "_draft_order_hint", "_projection_rank_cache", "_vor_rank_cache",
    )

    def __init__(
        self,
        players: Sequence[Player],
        *,
        league: LeagueConfig | None = None,
        metadata: PoolMetadata | None = None,
    ) -> None:
        self._players: list[Player] = list(players)
        self.league = league
        self.metadata = metadata or PoolMetadata(player_count=len(self._players))
        self.metadata.player_count = len(self._players)
        self._projection_rank_cache: dict[str, float] | None = None
        self._vor_rank_cache: dict[str, float] | None = None
        self._reindex()
        if league is not None:
            self.apply_league(league)
        else:
            self._impute_core_fields()
        self._draft_order_hint = self._compute_order_hint()

    # -- indexing --------------------------------------------------------
    def _reindex(self) -> None:
        self._by_id = {p.player_id: p for p in self._players}
        self._by_name: dict[str, Player] = {}
        for player in self._players:
            self._by_name.setdefault(_name_key(player.name), player)

    def __len__(self) -> int:
        return len(self._players)

    def __iter__(self) -> Iterator[Player]:
        return iter(self._players)

    def __contains__(self, key: object) -> bool:
        return str(key) in self._by_id or _name_key(str(key)) in self._by_name

    @property
    def players(self) -> list[Player]:
        return list(self._players)

    def get(self, key: str) -> Player | None:
        """Fetch by player id, falling back to a normalised name match."""
        if key in self._by_id:
            return self._by_id[key]
        return self._by_name.get(_name_key(key))

    def require(self, key: str) -> Player:
        player = self.get(key)
        if player is None:
            raise KeyError(f"Unknown player: {key!r}")
        return player

    def ids(self) -> list[str]:
        return [p.player_id for p in self._players]

    def by_position(self, position: Position) -> list[Player]:
        return [p for p in self._players if p.position is position]

    # -- league-aware derivation ----------------------------------------
    def apply_league(self, league: LeagueConfig) -> None:
        """Recompute eligibility, projections and VOR for a specific league."""
        self.league = league
        self._impute_core_fields()
        roster = league.roster
        for player in self._players:
            player.eligible_slots = tuple(
                slot for slot in roster.slots
                if player.fills_slot(slot) and slot is not Slot.IR
            )
        self._compute_value_over_replacement(league)
        self._invalidate_caches()
        self._draft_order_hint = self._compute_order_hint()

    def rescore(self, scoring: ScoringRules) -> RescoreOutcome:
        """Recompute every projection from its stored stat line under ``scoring``.

        This is what :attr:`Player.stat_totals` exists for. Before it, changing a
        league from half-PPR to full PPR left every projection on the board scored
        under the old rules, and the only remedy the app could offer was "download the
        season from ESPN again" — a network round trip to recompute arithmetic from
        numbers already in hand, and one that quietly returns *different* ADP because
        ESPN's has moved in the meantime.

        Three groups of players, handled differently on purpose, and tested in that
        order because the first case must win over the second:

        * **Has a scorable stat line.** Rescored. This is the whole point.
        * **Projection was estimated from draft position.** Thrown away and re-derived,
          not rescored — an imputed projection is a point on a curve fitted to the range
          of the *real* projections, so it is only meaningful on the scale those are on.
          A 6-point-passing-TD league moves the real quarterbacks up, and an estimate
          left behind on the old scale would rank a nobody above them.
        * **Real projection, no stat line.** Left exactly as it was. A user's CSV that
          supplied only a points total is their number, and guessing at the stats behind
          it to convert it would be inventing data. The return value says how many.

        Derived tiers and outcome bands are cleared pool-wide and re-derived, because
        both are read off the *position's* projection curve: one player moving changes
        the curve every player at that position is placed on. Tiers and bands a source
        supplied are kept — they are identifiable by an empty ``tier_source`` /
        ``outcome_band_source``, which is only ever written when this app derived them.
        """
        outcome = RescoreOutcome(
            unscorable_rules=core_stats.unscorable_rules(scoring)
        )
        for player in self._players:
            points = (
                core_stats.score(player.stat_totals, player.position, scoring)
                if player.stat_totals else None
            )
            if points is not None:
                # Stats are checked before the imputed flag, and the order is the whole
                # correctness of this branch: a player who had no projection and has
                # since been given a real stat line is no longer an estimate, and
                # reading the flag first would throw those stats away.
                player.projection = round(float(points), 1)
                player.expected_points = player.projection
                player.projection_detail = core_stats.describe(
                    player.stat_totals, player.position
                )
                if player.projection_imputed or player.projection_source in {
                    "", GENERIC_PROJECTION_SOURCE
                }:
                    # Only ever replaces this app's own placeholder text. A provider's
                    # or an upload's own line says something the app does not know and
                    # is left exactly as written — but "estimated from draft position"
                    # is now a lie about a number computed from real stats.
                    player.projection_source = STAT_LINE_PROJECTION_SOURCE
                player.projection_imputed = False
                outcome.rescored += 1
                continue
            if player.projection_imputed:
                player.projection = None
                player.expected_points = None
                player.projection_source = ""
                outcome.reimputed += 1
                continue
            # Either no stat line at all, or one that scores to nothing under these
            # rules — a defence in a league that scores no defensive stats, say.
            # Keeping the old number is wrong, but so is a confident zero, so leave it
            # and report it rather than choosing between two wrong answers silently.
            outcome.no_stat_line += 1

        if not outcome.changed:
            return outcome
        self._rederive_from_projections()
        return outcome

    def _rederive_from_projections(self) -> None:
        """Throw away everything read off the projection curve and derive it again.

        Called whenever projections move, by a scoring change or by an upload. Tiers
        and outcome bands a *source* supplied are kept — they are identifiable by an
        empty ``tier_source`` / ``outcome_band_source``, which is only ever written
        when this app derived the value itself.
        """
        for player in self._players:
            if player.tier_source:
                player.tier = None
                player.tier_source = ""
            if player.outcome_band_source:
                # All three, not just the ones this app filled last time: ceiling, floor
                # and risk are one statement about a player, and half of it on the old
                # scoring scale would be worse than re-deriving all of it.
                player.ceiling = None
                player.floor = None
                player.risk_score = None
                player.outcome_band_source = ""

        if self.league is not None:
            self.apply_league(self.league)
        else:
            self._impute_core_fields()
            self._invalidate_caches()
            self._draft_order_hint = self._compute_order_hint()

    def apply_projections(
        self,
        updates: Sequence[ProjectionUpdate],
        *,
        scoring: ScoringRules,
        mode: ProjectionMode = ProjectionMode.REPLACE,
        source: str = "your uploaded projections",
    ) -> ProjectionOutcome:
        """Overlay user-supplied projections onto this board and re-derive everything.

        A stat line and a points total are handled very differently on purpose:

        * **Stats** are merged into the player's stored line field by field and then
          scored by :meth:`rescore` under ``scoring``, so they behave exactly like a
          fetched projection — including surviving a later scoring change. Merging
          field by field means a sheet carrying only receiving numbers does not erase
          a running back's rushing projection; the count is reported so the UI can say
          the projection is a hybrid rather than leaving the user to guess.
        * **A points total** is taken as given and the player's stat line is *cleared*.
          Keeping the old stats would mean the next scoring change silently threw the
          user's own number away and went back to the provider's — the projection on
          screen would not be the one they uploaded, and nothing would say so.

        ``mode`` decides what happens where the board already has a projection: the
        user's number wins (``REPLACE``), the two are averaged (``BLEND``), or the
        upload only fills players whose projection was estimated or absent
        (``FILL_GAPS``).
        """
        outcome = ProjectionOutcome()
        touched = False
        for update in updates:
            player = self._by_id.get(update.player_id)
            if player is None:
                outcome.unmatched_ids.append(update.player_id or update.name)
                continue
            # An estimate is not a real projection, so nothing here treats it as one:
            # it is not worth blending against and it never blocks a fill-gaps upload.
            had_real = player.projection is not None and not player.projection_imputed
            if mode is ProjectionMode.FILL_GAPS and had_real:
                outcome.skipped_had_real += 1
                continue

            if update.stat_totals:
                base = player.stat_totals
                if mode is ProjectionMode.BLEND and had_real and base:
                    merged = _blend_stats(base, update.stat_totals)
                    outcome.blended += 1
                else:
                    merged = core_stats.merge(base, update.stat_totals)
                kept = set(merged) - set(update.stat_totals)
                if kept:
                    outcome.partial_merge += 1
                player.stat_totals = merged
                player.projection_imputed = False
                player.projection_source = (
                    f"From {source}, scored under your league's rules"
                    if not kept
                    else f"From {source}, with stats your file did not give kept from "
                         "the previous projection, scored under your league's rules"
                )
                outcome.from_stats += 1
                touched = True
                continue

            if update.points is None:
                continue
            points = float(update.points)
            if mode is ProjectionMode.BLEND and had_real:
                points = (points + float(player.projection or 0.0)) / 2.0
                outcome.blended += 1
            player.projection = round(points, 1)
            player.expected_points = player.projection
            player.projection_imputed = False
            # Cleared deliberately — see the docstring. The detail line goes with it,
            # because it described stats that no longer explain this number.
            player.stat_totals = {}
            player.projection_detail = ""
            player.projection_source = (
                f"A points total from {source} — a scoring change cannot rescore this, "
                "because the file gave points rather than stats"
            )
            outcome.from_points += 1
            touched = True

        if not touched:
            return outcome

        # Rescore turns the stat lines just written into points, re-derives the
        # estimated tail on the new scale, and re-derives tiers, bands and VOR. It
        # leaves points-only players exactly as set above: no stat line and not
        # imputed is its "left as-is" case.
        outcome.rescore = self.rescore(scoring)
        if not outcome.rescore.changed:
            # Nothing had a stat line, so rescore returned early without re-deriving —
            # but points-only uploads still moved the board out from under the tiers.
            self._rederive_from_projections()
        return outcome

    def _impute_core_fields(self) -> None:
        """Fill missing rank/ADP/projection fields from whatever is present."""
        imputed: dict[str, int] = {}
        missing: dict[str, int] = {}

        def bump(store: dict[str, int], key: str) -> None:
            store[key] = store.get(key, 0) + 1

        # 1. Overall rank: from ADP, else platform rank, else projection order.
        needs_rank = [p for p in self._players if p.overall_rank is None]
        if needs_rank:
            projection_order = sorted(
                self._players,
                key=lambda p: (-(p.projection or -1e9), p.name),
            )
            fallback_rank = {p.player_id: i + 1 for i, p in enumerate(projection_order)}
            for player in needs_rank:
                for candidate in (player.overall_adp, player.platform_adp,
                                  player.platform_rank):
                    if candidate is not None:
                        player.overall_rank = float(candidate)
                        break
                else:
                    player.overall_rank = float(fallback_rank[player.player_id])
                bump(imputed, "overall_rank")

        # 2. ADP: from rank when absent (rank is a reasonable ADP proxy).
        for player in self._players:
            if player.overall_adp is None:
                source = player.platform_adp or player.overall_rank or player.platform_rank
                if source is not None:
                    player.overall_adp = float(source)
                    bump(imputed, "overall_adp")
                else:
                    bump(missing, "overall_adp")
            if player.platform_adp is None:
                player.platform_adp = player.overall_adp
                bump(imputed, "platform_adp")
            if player.platform_rank is None:
                player.platform_rank = player.overall_rank
                bump(imputed, "platform_rank")

        # 3. Position rank from position-sorted overall rank.
        by_position: dict[Position, list[Player]] = {}
        for player in self._players:
            by_position.setdefault(player.position, []).append(player)
        for position, group in by_position.items():
            group.sort(key=lambda p: (p.overall_rank if p.overall_rank is not None else 1e9,
                                      p.name))
            for index, player in enumerate(group, start=1):
                if player.position_rank is None:
                    player.position_rank = index
                    bump(imputed, "position_rank")

        # 4. Projection: monotone decreasing proxy from overall rank when absent.
        have_projection = [p.projection for p in self._players if p.projection is not None]
        if have_projection:
            top = float(max(have_projection))
            bottom = float(min(have_projection))
        else:
            top, bottom = 320.0, 20.0
        n = max(1, len(self._players))
        for player in self._players:
            if player.projection is None:
                rank = float(player.overall_rank or n)
                fraction = min(1.0, max(0.0, (rank - 1.0) / max(1.0, n - 1.0)))
                # Convex decay: elite players separate more than late-round ones.
                player.projection = bottom + (top - bottom) * (1.0 - fraction) ** 1.8
                bump(imputed, "projection")
                player.projection_imputed = True
                # Said plainly, because this is a restatement of draft position rather
                # than an opinion about the player. A user comparing two projections
                # needs to know when one of them is really just an ADP.
                player.projection_source = IMPUTED_PROJECTION_SOURCE
            else:
                # Cleared, not left alone: a player who was imputed earlier and has
                # since been given a real projection is no longer an estimate, and a
                # stale flag would make :meth:`rescore` discard the real number.
                player.projection_imputed = False
                if not player.projection_source:
                    player.projection_source = GENERIC_PROJECTION_SOURCE
            if player.expected_points is None:
                player.expected_points = player.projection
            if player.adp_stdev is None:
                bump(missing, "adp_stdev")
            if player.bye_week is None:
                bump(missing, "bye_week")

        self._assign_tiers(imputed)
        self._derive_outcome_bands(imputed)

        self.metadata.imputed_fields = imputed
        self.metadata.missing_fields = missing

    def _assign_tiers(self, imputed: dict[str, int] | None = None) -> None:
        """Derive tiers from projection gaps for players lacking an explicit tier.

        Within a position, players are ordered by projection and a new tier starts
        wherever the drop to the next player exceeds the mean gap plus one standard
        deviation of all gaps at that position. So a tier break is a gap that is
        genuinely unusual *for that position*, which is why QB tiers and WR tiers come
        out different sizes rather than being forced to a fixed count.
        """
        by_position: dict[Position, list[Player]] = {}
        for player in self._players:
            by_position.setdefault(player.position, []).append(player)

        for position, group in by_position.items():
            missing = [p for p in group if p.tier is None]
            if not missing:
                continue
            group.sort(key=lambda p: -(p.projection or 0.0))
            values = np.array([float(p.projection or 0.0) for p in group])
            if len(values) < 2:
                for player in group:
                    player.tier = player.tier or 1
                continue
            gaps = np.diff(values) * -1.0  # positive where value drops
            threshold = float(np.mean(gaps) + np.std(gaps)) if len(gaps) else 0.0
            label = (
                f"Projection gap breakpoints at {position}: a new tier starts where the "
                f"drop to the next player exceeds {threshold:.1f} points "
                f"(mean gap + 1σ)"
            )
            tier = 1
            group[0].tier = group[0].tier or tier
            for index in range(1, len(group)):
                if threshold > 0 and gaps[index - 1] > threshold:
                    tier += 1
                if group[index].tier is None:
                    group[index].tier = tier
            for player in missing:
                player.tier_source = label
                if imputed is not None:
                    imputed["tier"] = imputed.get("tier", 0) + 1

    def _derive_outcome_bands(self, imputed: dict[str, int] | None = None) -> None:
        """Fill ceiling, floor and risk for players whose source did not supply them.

        **What these numbers are.** The one real distribution the board has is how much
        drafters disagree about where a player belongs: Fantasy Football Calculator
        publishes the standard deviation of each player's pick across real mock drafts.
        So ceiling and floor are the *range of value implied by that disagreement* —
        if this player went as early as the room's optimists take them, they would be a
        player of roughly this calibre; as late as the pessimists, roughly that.

        **What they are not.** They are not a forecast of the player's season. Draft
        position and outcome are different quantities: the room agreeing on a player
        says nothing about whether he tears an ACL. So a consensus first-rounder gets a
        narrow band here — correctly, because ADP disagreement is near zero for him and
        this is the only thing being measured. A genuine outcome interval would need
        years of projections paired with actual results, which no free source publishes.
        Every row says which of the two it is, in :attr:`outcome_band_source`.

        **How the mapping works**, per position:

        1. Sort the position's projections descending. That is the empirical "the Nth
           best player at this position is worth this much" curve, in the league's own
           scoring units, and it is monotone by construction.
        2. Locate each player on it at their own projection rank — so reading the curve
           at their own slot returns their own projection exactly, and the band
           brackets the projection rather than floating away from it.
        3. Convert their pick spread (σ, in overall picks) into a displacement in
           positional ranks, using how densely that position is drafted. Two rounds of
           disagreement moves a WR much further down the WR curve than it moves a TE
           down the TE curve, because there are more WRs in between.
        4. Read the curve at ∓1.28σ — the 10th and 90th percentiles of a normal.

        ``min_pick``/``max_pick`` are deliberately *not* used for the spread. They are
        the extremes over every recorded mock draft, so their range grows with the
        number of drafts sampled; treating it as a percentile interval overstated
        disagreement by three to four times. σ is the right statistic and FFC
        publishes it directly.
        """
        by_position: dict[Position, list[Player]] = {}
        for player in self._players:
            by_position.setdefault(player.position, []).append(player)

        for position, group in by_position.items():
            needs = [
                p for p in group
                if p.ceiling is None or p.floor is None or p.risk_score is None
            ]
            if not needs:
                continue

            # Step 1-2: the projection curve, and each player's own slot on it.
            ordered = sorted(group, key=lambda p: (-(p.projection or 0.0), p.name))
            curve = np.array([float(p.projection or 0.0) for p in ordered])
            slots = np.arange(len(curve), dtype=float)
            slot_of = {p.player_id: float(i) for i, p in enumerate(ordered)}

            # Step 3: picks → positional ranks, over the picks this position occupies.
            picks = [
                float(p.overall_adp) for p in group if p.overall_adp is not None
            ]
            pick_span = (max(picks) - min(picks)) if len(picks) > 1 else 0.0
            ranks_per_pick = (len(curve) - 1) / pick_span if pick_span > 0 else 0.0

            for player in needs:
                slot = slot_of[player.player_id]
                displacement = (
                    float(player.adp_stdev or 0.0) * _BAND_Z * ranks_per_pick
                )
                disagreement = self._disagreement_ratio(player)

                if player.ceiling is None:
                    player.ceiling = round(
                        _read_curve(slots, curve, slot - displacement), 1
                    )
                    if imputed is not None:
                        imputed["ceiling"] = imputed.get("ceiling", 0) + 1
                if player.floor is None:
                    player.floor = round(
                        max(0.0, _read_curve(slots, curve, slot + displacement)), 1
                    )
                    if imputed is not None:
                        imputed["floor"] = imputed.get("floor", 0) + 1
                if player.risk_score is None:
                    player.risk_score = self._risk_score(player, disagreement)
                    if imputed is not None:
                        imputed["risk_score"] = imputed.get("risk_score", 0) + 1

                measured = (
                    "an estimated" if player.adp_stdev_is_estimated else "a measured"
                )
                player.outcome_band_source = (
                    f"The value range implied by {measured} draft-pick spread of "
                    f"±{float(player.adp_stdev or 0.0):.1f} picks — about "
                    f"{displacement:.1f} {position} places either way on this "
                    f"position's projection curve. How much the room disagrees about "
                    f"him, not a forecast of his season."
                )

    def _disagreement_ratio(self, player: Player) -> float:
        """Draft-pick spread as a fraction of the player's own ADP.

        Scale-free on purpose: three picks of spread is a wide disagreement at pick 5
        and none at all at pick 150, so the raw σ cannot be compared across the board
        while σ/ADP can.
        """
        adp = player.overall_adp
        if adp is None or float(adp) <= 0 or player.adp_stdev is None:
            return 0.0
        return max(0.0, float(player.adp_stdev) / float(adp))

    def _risk_score(self, player: Player, disagreement: float) -> float:
        """0-1: how much less predictable this player is than others.

        Deliberately *not* the width of the outcome band, which is dominated by the
        per-position baseline and so would make every healthy running back look
        equally risky. This scores what actually separates one player from their
        positional peers: how much the draft room disagrees about them, whether they
        are hurt, and whether they have an NFL season on record at all.
        """
        spread = min(1.0, disagreement / _DISAGREEMENT_AT_FULL_RISK)
        score = (
            _RISK_WEIGHT_DISAGREEMENT * spread
            + _RISK_WEIGHT_INJURY * player.injury_penalty
            + (_RISK_WEIGHT_ROOKIE if player.is_rookie else 0.0)
        )
        return float(round(min(1.0, max(0.0, score)), 3))

    def _compute_value_over_replacement(self, league: LeagueConfig) -> None:
        """VOR = projection minus the projection of the positional replacement."""
        by_position: dict[Position, list[Player]] = {}
        for player in self._players:
            by_position.setdefault(player.position, []).append(player)

        for position, group in by_position.items():
            group.sort(key=lambda p: -(p.projection or 0.0))
            cutoff = int(round(league.replacement_rank(position)))
            index = min(max(0, cutoff - 1), len(group) - 1)
            replacement = float(group[index].projection or 0.0)
            # Always recompute: VOR is league-format dependent, so a value
            # carried in from an import would be wrong for this league.
            for player in group:
                player.replacement_points = replacement
                player.value_over_replacement = float(
                    (player.projection or 0.0) - replacement
                )

    def _compute_order_hint(self) -> dict[str, float]:
        """Cached blended ordering value per player, lower = drafted earlier."""
        hint: dict[str, float] = {}
        for player in self._players:
            adp = player.adp_for(prefer_platform=True)
            rank = player.rank_for(prefer_platform=True)
            candidates = [v for v in (adp, rank) if v is not None]
            hint[player.player_id] = float(np.mean(candidates)) if candidates else 1e6
        return hint

    def order_value(self, player: Player) -> float:
        return self._draft_order_hint.get(player.player_id, 1e6)

    # -- ranking sources -------------------------------------------------
    def ranking_value(
        self,
        player: Player,
        source: RankingSource,
        blend_weights: dict[str, float] | None = None,
    ) -> float:
        """Return a 'lower is better' ordering value for the chosen source."""
        n = max(1, len(self._players))
        if source is RankingSource.PLATFORM_ADP:
            return float(player.platform_adp or player.overall_adp or n)
        if source is RankingSource.OVERALL_ADP:
            return float(player.overall_adp or player.platform_adp or n)
        if source in (RankingSource.EXPERT_CONSENSUS, RankingSource.PERSONAL):
            return float(player.overall_rank or n)
        if source is RankingSource.PROJECTION:
            return self._projection_rank(player)
        if source is RankingSource.LEAGUE_ADJUSTED:
            return self._vor_rank(player)
        # BLEND
        weights = blend_weights or {"platform_adp": 0.45, "overall_adp": 0.25,
                                    "projection": 0.30}
        parts: list[tuple[float, float]] = []
        if weights.get("platform_adp"):
            parts.append((float(player.platform_adp or player.overall_adp or n),
                          weights["platform_adp"]))
        if weights.get("overall_adp"):
            parts.append((float(player.overall_adp or player.platform_adp or n),
                          weights["overall_adp"]))
        if weights.get("projection"):
            parts.append((self._projection_rank(player), weights["projection"]))
        if weights.get("league_adjusted"):
            parts.append((self._vor_rank(player), weights["league_adjusted"]))
        total = sum(w for _, w in parts)
        if total <= 0:
            return float(player.overall_adp or n)
        return float(sum(v * w for v, w in parts) / total)

    def _projection_rank(self, player: Player) -> float:
        """1-based rank by projection (cached; invalidated on league change)."""
        if self._projection_rank_cache is None:
            order = sorted(self._players, key=lambda p: -(p.projection or 0.0))
            self._projection_rank_cache = {
                p.player_id: float(i + 1) for i, p in enumerate(order)
            }
        return self._projection_rank_cache.get(player.player_id, float(len(self._players)))

    def _vor_rank(self, player: Player) -> float:
        """1-based rank by value over replacement (cached)."""
        if self._vor_rank_cache is None:
            order = sorted(
                self._players, key=lambda p: -(p.value_over_replacement or 0.0)
            )
            self._vor_rank_cache = {
                p.player_id: float(i + 1) for i, p in enumerate(order)
            }
        return self._vor_rank_cache.get(player.player_id, float(len(self._players)))

    def projection_percentile(self, player: Player) -> float:
        """Projection standing as 0-1, where 1.0 is the highest-projected player.

        A percentile rather than a raw point total so the pick model's weights
        mean the same thing whether a file carries season points, per-game
        points, or an arbitrary index.
        """
        if player.projection is None:
            return 0.0
        return self._rank_to_percentile(self._projection_rank(player))

    def vor_percentile(self, player: Player) -> float:
        """Value-over-replacement standing as 0-1, 1.0 being the most valuable."""
        if player.value_over_replacement is None:
            return 0.0
        return self._rank_to_percentile(self._vor_rank(player))

    def _rank_to_percentile(self, rank: float) -> float:
        """Convert a 1-based rank into a 0-1 score where rank 1 scores 1.0."""
        n = max(1, len(self._players))
        if n == 1:
            return 1.0
        return float(max(0.0, min(1.0, (float(n) - float(rank)) / float(n - 1))))

    def _invalidate_caches(self) -> None:
        self._projection_rank_cache = None
        self._vor_rank_cache = None

    # -- frames ----------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """Tabular view for the UI and exports."""
        rows = []
        for player in self._players:
            rows.append({
                "player_id": player.player_id,
                "player_name": player.name,
                "position": str(player.position),
                "nfl_team": player.nfl_team,
                "bye_week": player.bye_week,
                "tier": player.tier,
                "projection": player.projection,
                "overall_rank": player.overall_rank,
                "position_rank": player.position_rank,
                "platform_rank": player.platform_rank,
                "overall_adp": player.overall_adp,
                "platform_adp": player.platform_adp,
                "adp_stdev": player.adp_stdev,
                "value_over_replacement": player.value_over_replacement,
                "ceiling": player.ceiling,
                "floor": player.floor,
                "risk_score": player.risk_score,
                "is_rookie": player.is_rookie,
                "injury_status": str(player.injury_status),
                "experience": player.experience,
                "ffc_adp": player.ffc_adp,
                "espn_adp": player.espn_adp,
                "espn_rank": player.espn_rank,
                "yahoo_adp": player.yahoo_adp,
                "yahoo_rank": player.yahoo_rank,
                "sleeper_rank": player.sleeper_rank,
                "adp_source_count": player.adp_source_count,
                "adp_disagreement": player.adp_disagreement,
                "adp_stdev_is_estimated": player.adp_stdev_is_estimated,
                "projection_source": player.projection_source,
                "projection_detail": player.projection_detail,
                "stat_totals": core_stats.to_frame_value(player.stat_totals),
                "projection_imputed": player.projection_imputed,
                "tier_source": player.tier_source,
                "outcome_band_source": player.outcome_band_source,
                "notes": player.notes,
            })
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values("overall_adp", na_position="last").reset_index(drop=True)
        return frame

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        league: LeagueConfig | None = None,
        metadata: PoolMetadata | None = None,
    ) -> "PlayerPool":
        """Build a pool from an already-normalised frame."""
        players = [player_from_row(row) for _, row in frame.iterrows()]
        return cls(players, league=league, metadata=metadata)


def _blend_stats(
    base: dict[str, float], override: dict[str, float]
) -> dict[str, float]:
    """Average two stat lines field by field, keeping fields only one side has.

    The stat line is averaged rather than the points, so the stored line still explains
    the projection and a later scoring change rescores the blend instead of discarding
    it. Scoring is linear in every stat, so averaging the stats and averaging the points
    give the same answer anyway — this way just stays honest about where it came from.

    A field only one side mentions is taken as-is rather than halved: silence is not a
    projection of zero, and halving one source's 1,200 rushing yards because the other
    sheet had no rushing column would invent a number neither source claims.
    """
    out = dict(base)
    for field_name, value in override.items():
        if field_name not in core_stats.STAT_FIELD_SET:
            continue
        if field_name in out:
            out[field_name] = (float(out[field_name]) + float(value)) / 2.0
        else:
            out[field_name] = float(value)
    return out


def _name_key(name: str) -> str:
    """Loose key for name lookups (case/punctuation-insensitive)."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def player_from_row(row: Any) -> Player:
    """Build a :class:`Player` from a normalised frame row / mapping."""
    from core.validation import to_bool, to_float, to_int

    def val(key: str, default: Any = None) -> Any:
        try:
            value = row[key]
        except (KeyError, IndexError, TypeError):
            return default
        if value is None:
            return default
        try:
            if pd.isna(value):
                return default
        except (TypeError, ValueError):
            pass
        return value

    name = str(val("player_name", val("name", "Unknown"))).strip()
    position = Position.coerce(val("position"), Position.RB) or Position.RB
    player_id = str(val("player_id") or f"{_name_key(name)}_{position}")

    return Player(
        player_id=player_id,
        name=name,
        position=position,
        nfl_team=str(val("nfl_team", "FA") or "FA"),
        bye_week=to_int(val("bye_week"), None),
        experience=to_int(val("experience"), None),
        is_rookie=to_bool(val("rookie_flag", val("is_rookie")), False),
        injury_status=InjuryStatus.coerce(val("injury_status"), InjuryStatus.HEALTHY)
        or InjuryStatus.HEALTHY,
        suspended=to_bool(val("suspended"), False),
        projection=to_float(val("projection"), None),
        overall_rank=to_float(val("overall_rank"), None),
        position_rank=to_int(val("position_rank"), None),
        platform_rank=to_float(val("platform_rank"), None),
        overall_adp=to_float(val("overall_adp", val("adp")), None),
        platform_adp=to_float(val("platform_adp"), None),
        adp_stdev=to_float(val("adp_stdev"), None),
        min_pick=to_int(val("min_pick"), None),
        max_pick=to_int(val("max_pick"), None),
        tier=to_int(val("tier"), None),
        ceiling=to_float(val("ceiling"), None),
        floor=to_float(val("floor"), None),
        risk_score=to_float(val("risk_score"), None),
        value_over_replacement=to_float(val("value_over_replacement"), None),
        notes=str(val("notes", "") or ""),
        source=str(val("source", "") or ""),
        ffc_adp=to_float(val("ffc_adp"), None),
        espn_adp=to_float(val("espn_adp"), None),
        espn_rank=to_float(val("espn_rank"), None),
        yahoo_adp=to_float(val("yahoo_adp"), None),
        yahoo_rank=to_float(val("yahoo_rank"), None),
        sleeper_rank=to_float(val("sleeper_rank"), None),
        adp_source_count=to_int(val("adp_source_count"), None),
        adp_disagreement=to_float(val("adp_disagreement"), None),
        adp_stdev_is_estimated=to_bool(val("adp_stdev_is_estimated"), False),
        projection_source=str(val("projection_source", "") or ""),
        projection_detail=str(val("projection_detail", "") or ""),
        stat_totals=core_stats.from_frame_value(val("stat_totals")),
        projection_imputed=to_bool(val("projection_imputed"), False),
        tier_source=str(val("tier_source", "") or ""),
        outcome_band_source=str(val("outcome_band_source", "") or ""),
    )


def available_slot_for(
    position: Position, filled: dict[Slot, int], roster: RosterSettings
) -> Slot | None:
    """First open starting slot ``position`` can fill, most-restrictive first.

    ``filled`` maps slot → how many seats of that slot type are already used.
    Returns ``None`` when every eligible starting slot is full (bench territory).
    """
    for slot in SLOT_FILL_PRIORITY:
        seats = roster.count(slot)
        if not seats:
            continue
        if position not in SLOT_ELIGIBILITY.get(slot, frozenset()):
            continue
        if filled.get(slot, 0) < seats:
            return slot
    return None
