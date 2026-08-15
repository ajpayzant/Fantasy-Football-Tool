"""Typed configuration objects.

Everything the simulation needs is expressed here as plain dataclasses so the
engine can be exercised from tests without Streamlit or a database. Model
weights live in :class:`ModelWeights` — there are no magic numbers buried in
the engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from .constants import (
    BENCH_DEPTH_ALLOWANCE,
    REPLACEMENT_RANK_PER_TEAM,
    SLOT_ELIGIBILITY,
    SLOT_FILL_PRIORITY,
    STARTING_SLOTS,
)
from .enums import (
    Archetype,
    DraftType,
    LeagueFormat,
    Platform,
    Position,
    RankingSource,
    ScoringPreset,
    Slot,
)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ScoringRules:
    """Explicit per-event scoring values.

    Presets expand into these fields so downstream code never branches on a
    preset name. Fractional values are supported throughout.
    """

    # Passing
    pass_yards_per_point: float = 25.0
    pass_td: float = 4.0
    interception: float = -2.0
    pass_2pt: float = 2.0
    # Rushing
    rush_yards_per_point: float = 10.0
    rush_td: float = 6.0
    # Receiving
    rec_yards_per_point: float = 10.0
    rec_td: float = 6.0
    reception: float = 0.0
    te_premium_reception_bonus: float = 0.0
    # Misc
    fumble_lost: float = -2.0
    rush_rec_2pt: float = 2.0
    # Bonuses (yardage thresholds → extra points)
    bonus_pass_300_yards: float = 0.0
    bonus_pass_400_yards: float = 0.0
    bonus_rush_100_yards: float = 0.0
    bonus_rush_200_yards: float = 0.0
    bonus_rec_100_yards: float = 0.0
    bonus_rec_200_yards: float = 0.0
    bonus_long_td_40_plus: float = 0.0
    # Kicking
    kick_fg_made: float = 3.0
    kick_xp_made: float = 1.0
    # Defence / special teams
    dst_sack: float = 1.0
    dst_interception: float = 2.0
    dst_fumble_recovery: float = 2.0
    dst_touchdown: float = 6.0
    dst_safety: float = 2.0
    # Points-allowed tiers, in points per game finishing in that band. These are
    # the largest single component of real defence scoring — a defence scored on
    # sacks and turnovers alone projects roughly 25% low — so they are explicit
    # rather than folded into a single "shutout" bonus. Defaults are ESPN's.
    dst_points_allowed_0: float = 5.0
    dst_points_allowed_1_6: float = 4.0
    dst_points_allowed_7_13: float = 3.0
    dst_points_allowed_14_17: float = 1.0
    dst_points_allowed_18_21: float = 0.0
    dst_points_allowed_22_27: float = -1.0
    dst_points_allowed_28_34: float = -3.0
    dst_points_allowed_35_plus: float = -5.0
    preset: ScoringPreset = ScoringPreset.HALF_PPR

    @classmethod
    def from_preset(cls, preset: ScoringPreset | str, **overrides: Any) -> "ScoringRules":
        """Build rules from a named preset, then apply explicit overrides."""
        preset = ScoringPreset.coerce(preset, ScoringPreset.CUSTOM) or ScoringPreset.CUSTOM
        base: dict[str, Any] = {"preset": preset}
        if preset is ScoringPreset.STANDARD:
            base["reception"] = 0.0
        elif preset is ScoringPreset.HALF_PPR:
            base["reception"] = 0.5
        elif preset is ScoringPreset.FULL_PPR:
            base["reception"] = 1.0
        elif preset is ScoringPreset.TE_PREMIUM:
            base.update(reception=1.0, te_premium_reception_bonus=0.5)
        base.update(overrides)
        return cls(**base)

    def reception_value(self, position: Position) -> float:
        """Points per catch for a position, including any TE premium."""
        bonus = self.te_premium_reception_bonus if position is Position.TE else 0.0
        return self.reception + bonus

    @property
    def is_ppr(self) -> bool:
        return self.reception > 0

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        raw = asdict(self)
        raw["preset"] = str(self.preset)
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ScoringRules":
        raw = dict(raw or {})
        # ``dst_shutout`` was this field's name before the points-allowed tiers
        # existed. Migrated rather than dropped, because the filter below discards
        # unknown keys silently and a league saved earlier would lose the value.
        if "dst_shutout" in raw and "dst_points_allowed_0" not in raw:
            raw["dst_points_allowed_0"] = raw["dst_shutout"]
        known = {f for f in cls.__slots__}
        clean = {k: v for k, v in raw.items() if k in known}
        if "preset" in clean:
            clean["preset"] = ScoringPreset.coerce(clean["preset"], ScoringPreset.CUSTOM)
        return cls(**clean)


# ─────────────────────────────────────────────────────────────────────────────
# Roster
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class RosterSettings:
    """Slot counts plus positional limits.

    ``slots`` maps a :class:`Slot` to how many of that slot each team carries.
    Any slot may be zero or absent. Nothing assumes a "standard" lineup.
    """

    slots: dict[Slot, int] = field(default_factory=lambda: {
        Slot.QB: 1, Slot.RB: 2, Slot.WR: 2, Slot.TE: 1,
        Slot.FLEX: 1, Slot.K: 1, Slot.DST: 1, Slot.BENCH: 7,
    })
    position_max: dict[Position, int] = field(default_factory=dict)
    position_min: dict[Position, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.slots = {
            Slot.coerce(k, None) or k: int(v)
            for k, v in self.slots.items() if int(v) > 0
        }
        self.position_max = {
            Position.coerce(k, None) or k: int(v) for k, v in self.position_max.items()
        }
        self.position_min = {
            Position.coerce(k, None) or k: int(v) for k, v in self.position_min.items()
        }

    def count(self, slot: Slot) -> int:
        return int(self.slots.get(slot, 0))

    @property
    def starting_slots(self) -> dict[Slot, int]:
        return {s: n for s, n in self.slots.items() if s in STARTING_SLOTS and n > 0}

    @property
    def starters_total(self) -> int:
        return sum(self.starting_slots.values())

    @property
    def bench_total(self) -> int:
        return self.count(Slot.BENCH)

    @property
    def ir_total(self) -> int:
        return self.count(Slot.IR)

    @property
    def roster_size(self) -> int:
        """Draftable roster size (IR slots are not drafted into)."""
        return self.starters_total + self.bench_total

    @property
    def is_superflex(self) -> bool:
        return self.count(Slot.SUPERFLEX) > 0

    @property
    def is_two_qb(self) -> bool:
        return self.count(Slot.QB) >= 2

    def ordered_starting_slots(self) -> list[Slot]:
        """Starting slots expanded to one entry per seat, most-restrictive first."""
        out: list[Slot] = []
        for slot in SLOT_FILL_PRIORITY:
            out.extend([slot] * self.count(slot))
        # Any starting slot not in the priority list (future additions).
        for slot, n in self.starting_slots.items():
            if slot not in SLOT_FILL_PRIORITY:
                out.extend([slot] * n)
        return out

    def starting_demand(self) -> dict[Position, float]:
        """Expected starters demanded per position, splitting flex slots evenly."""
        demand: dict[Position, float] = {p: 0.0 for p in Position}
        for slot, n in self.starting_slots.items():
            eligible = SLOT_ELIGIBILITY.get(slot, frozenset())
            if not eligible:
                continue
            share = n / len(eligible)
            for pos in eligible:
                demand[pos] += share
        return demand

    def positional_seats(self) -> dict[Position, float]:
        """Starting seats per position, flex seats shared by how much each needs one.

        Differs from :meth:`starting_demand` in how a flex seat is split. Splitting it
        evenly says a RB/WR/TE flex is one-third a tight-end seat, which no real lineup
        behaves like: the seat goes to whichever position the lineup already leans on,
        so it is shared in proportion to the *dedicated* seats each position holds. In
        a 2RB/2WR/1TE lineup that is 40/40/20 rather than 33/33/33, which is the
        difference between treating a second tight end as a starter and treating him as
        the backup he is.

        A superflex seat goes wholly to the quarterback, for the reason given in
        :meth:`useful_depth`. Slots nobody is eligible for contribute nothing.
        """
        dedicated: dict[Position, float] = {p: 0.0 for p in Position}
        shared: list[tuple[frozenset[Position], int]] = []
        for slot, count in self.starting_slots.items():
            eligible = SLOT_ELIGIBILITY.get(slot, frozenset())
            if not eligible:
                continue
            if slot is Slot.SUPERFLEX:
                dedicated[Position.QB] += count
            elif len(eligible) == 1:
                dedicated[next(iter(eligible))] += count
            else:
                shared.append((eligible, count))
        seats = dict(dedicated)
        for eligible, count in shared:
            weights = {p: dedicated.get(p, 0.0) for p in eligible}
            total = sum(weights.values())
            if total <= 0:
                # Nobody eligible holds a dedicated seat, so the seat really is open
                # to all of them equally — a flex-only lineup, which is legal.
                for position in eligible:
                    seats[position] = seats.get(position, 0.0) + count / len(eligible)
                continue
            for position, weight in weights.items():
                seats[position] = seats.get(position, 0.0) + count * weight / total
        return seats

    def slot_fit(self, position: Position, slot: Slot) -> float:
        """How naturally ``position`` fills ``slot``, 0-1.

        1.0 for a seat dedicated to the position, and for any seat the position has
        as strong a claim on as anyone else. Below 1.0 for a shared seat the position
        is only the *third* choice for: a RB/WR/TE flex in a 2RB/2WR/1TE lineup is a
        running back or receiver seat that a tight end may occupy, not a tight-end
        seat, and treating the two as identical is what let a manager take his second
        tight end in round five and have the model score it as filling a starter.

        Weighted by dedicated seats, so a TE-premium lineup — which really does start
        two tight ends — returns 1.0 for the same seat without a special case. A
        superflex returns 1.0 for everyone eligible, because it is a start-anyone seat
        by construction.
        """
        eligible = SLOT_ELIGIBILITY.get(slot, frozenset())
        if position not in eligible:
            return 0.0
        if len(eligible) == 1 or slot is Slot.SUPERFLEX:
            return 1.0
        dedicated = {
            p: sum(
                count for s, count in self.starting_slots.items()
                if SLOT_ELIGIBILITY.get(s, frozenset()) == frozenset({p})
            )
            for p in eligible
        }
        best = max(dedicated.values())
        if best <= 0:
            # No position eligible for this seat has a dedicated one, so none of them
            # has a better claim than the others.
            return 1.0
        return min(1.0, dedicated[position] / float(best))

    def useful_depth(self, position: Position) -> int:
        """How many of ``position`` a roster can actually use, starters included.

        The ceiling a sane drafter respects. A one-QB lineup can start one
        quarterback, so the second is a backup and the third is a wasted pick — and
        without a number saying so, nothing in the pick model does: roster need still
        scores a backup at half a starter, and a manager with an early-QB tendency
        spends it three times because the tendency has no idea the seat is taken.

        Seats come from :meth:`positional_seats`, so a TE-premium or WR-heavy lineup
        shifts the ceiling without a special case. Superflex is the one exception: its
        seat counts whole toward the quarterback rather than being shared, because that
        is what fills it in practice, and sharing it would have the model penalising the
        second quarterback in the one format that demands two.

        On top of the seats sits :data:`core.constants.BENCH_DEPTH_ALLOWANCE`, scaled
        by how much bench the league carries per starter — a league with three bench
        seats has no room for the depth a seven-seat bench has, and the ceiling should
        say so.

        Deliberately not a hard cap: it feeds a graded penalty, so an extraordinary
        player still gets taken. It is the point past which a pick needs a reason
        beyond "he was next on my list".
        """
        seats = self.positional_seats().get(position, 0.0)
        if seats <= 0.0:
            return 0
        depth_room = min(1.0, self.bench_total / max(1, self.starters_total))
        allowance = BENCH_DEPTH_ALLOWANCE.get(position, 1) * depth_room
        # floor(x + 0.5) rather than round(), which rounds halves to even and would
        # make an exactly-borderline ceiling depend on whether it landed on 2.5 or 3.5.
        return max(1, int(math.floor(seats + allowance + 0.5)))

    def max_for(self, position: Position) -> int | None:
        return self.position_max.get(position)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots": {str(s): n for s, n in self.slots.items()},
            "position_max": {str(p): n for p, n in self.position_max.items()},
            "position_min": {str(p): n for p, n in self.position_min.items()},
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RosterSettings":
        raw = raw or {}
        return cls(
            slots={Slot.coerce(k, Slot.BENCH): int(v)
                   for k, v in (raw.get("slots") or {}).items()},
            position_max={Position.coerce(k, Position.RB): int(v)
                          for k, v in (raw.get("position_max") or {}).items()},
            position_min={Position.coerce(k, Position.RB): int(v)
                          for k, v in (raw.get("position_min") or {}).items()},
        )


# ─────────────────────────────────────────────────────────────────────────────
# League
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class LeagueConfig:
    """Complete definition of a league's rules and draft structure."""

    name: str = "My League"
    season: int = 2026
    platform: Platform = Platform.ESPN
    team_count: int = 12
    rounds: int = 16
    draft_type: DraftType = DraftType.SNAKE
    league_format: LeagueFormat = LeagueFormat.REDRAFT
    scoring: ScoringRules = field(default_factory=ScoringRules)
    roster: RosterSettings = field(default_factory=RosterSettings)
    user_draft_slot: int = 1
    draft_date: str | None = None
    reversal_round: int = 3
    """Round at which a third-round-reversal draft flips (configurable)."""
    custom_round_order: dict[int, list[int]] = field(default_factory=dict)
    """Round number → explicit list of draft slots, for ``DraftType.CUSTOM``."""
    notes: str = ""
    league_id: int | None = None

    def __post_init__(self) -> None:
        self.platform = Platform.coerce(self.platform, Platform.CUSTOM) or Platform.CUSTOM
        self.draft_type = DraftType.coerce(self.draft_type, DraftType.SNAKE) or DraftType.SNAKE
        self.league_format = (
            LeagueFormat.coerce(self.league_format, LeagueFormat.REDRAFT) or LeagueFormat.REDRAFT
        )
        self.team_count = int(self.team_count)
        self.rounds = int(self.rounds)
        self.user_draft_slot = int(self.user_draft_slot)
        self.custom_round_order = {
            int(k): [int(s) for s in v] for k, v in (self.custom_round_order or {}).items()
        }

    @property
    def total_picks(self) -> int:
        return self.team_count * self.rounds

    @property
    def allows_keepers(self) -> bool:
        return self.league_format in (LeagueFormat.KEEPER, LeagueFormat.DYNASTY)

    def replacement_rank(self, position: Position) -> float:
        """Approximate positional rank at which value hits replacement level."""
        per_team = REPLACEMENT_RANK_PER_TEAM.get(position, 1.0)
        if position is Position.QB and self.roster.is_superflex:
            per_team = 1.9
        if position is Position.QB and self.roster.is_two_qb:
            per_team = 2.2
        return max(1.0, per_team * self.team_count)

    def with_(self, **changes: Any) -> "LeagueConfig":
        """Return a modified copy (configs are treated as immutable in the engine)."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "season": self.season,
            "platform": str(self.platform),
            "team_count": self.team_count,
            "rounds": self.rounds,
            "draft_type": str(self.draft_type),
            "league_format": str(self.league_format),
            "scoring": self.scoring.to_dict(),
            "roster": self.roster.to_dict(),
            "user_draft_slot": self.user_draft_slot,
            "draft_date": self.draft_date,
            "reversal_round": self.reversal_round,
            "custom_round_order": {str(k): v for k, v in self.custom_round_order.items()},
            "notes": self.notes,
            "league_id": self.league_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LeagueConfig":
        raw = dict(raw or {})
        return cls(
            name=raw.get("name", "My League"),
            season=int(raw.get("season", 2026)),
            platform=raw.get("platform", Platform.ESPN),
            team_count=int(raw.get("team_count", 12)),
            rounds=int(raw.get("rounds", 16)),
            draft_type=raw.get("draft_type", DraftType.SNAKE),
            league_format=raw.get("league_format", LeagueFormat.REDRAFT),
            scoring=ScoringRules.from_dict(raw.get("scoring") or {}),
            roster=RosterSettings.from_dict(raw.get("roster") or {}),
            user_draft_slot=int(raw.get("user_draft_slot", 1)),
            draft_date=raw.get("draft_date"),
            reversal_round=int(raw.get("reversal_round", 3)),
            custom_round_order={int(k): v for k, v in (raw.get("custom_round_order") or {}).items()},
            notes=raw.get("notes", ""),
            league_id=raw.get("league_id"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model weights
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ModelWeights:
    """Weights for every term in the candidate utility score.

    Utility is computed on a roughly comparable scale per component (most
    components are normalised to ~[-1, 1] or [0, 1] before weighting), so these
    numbers are directly interpretable as relative importance.
    """

    adp: float = 1.00
    projection: float = 0.30
    """Weight on raw projected points, as a percentile of the draftable board.

    Deliberately the smaller half of the board-value pair. Raw points are not
    comparable across positions — on a real board the top sixteen quarterbacks average
    a 0.93 projection percentile against 0.71 for receivers, purely because passing
    yardage scores more points than anything else — so this term is a standing +0.1
    utility bonus on every quarterback in the file. Measured on a live board it was
    enough to make a *backup* quarterback the highest-utility player available in round
    six, which is how simulated teams ended up with three of them.

    It is not zero, because within a position it is the most direct statement of how
    good a player is, and because the cross-position skew is precisely what
    ``value_over_replacement`` exists to correct.
    """
    value_over_replacement: float = 1.00
    """Weight on points above the last startable player at the same position.

    The cross-position measure, and the larger half of the board-value pair by a wide
    margin. A quarterback's 300 points and a receiver's 230 are not comparable until
    both are measured against what the position's replacement gives you free, and this
    is the only term that does that: on the same board it puts a fringe starting
    quarterback at a 0.65 percentile where raw projection put him at 0.96, above every
    receiver alive.

    It carries the weight the ``tier`` term used to hold, because it is the term that
    said the same thing properly. A tier was a coarse restatement of the projection
    curve — derived from projection gaps, then read back as a strength signal — and
    deleting it without moving its weight anywhere left every *non*-board term
    relatively louder, which measurably worsened the thing it was meant to fix:
    quarterback doubles inside the first seven rounds went from 10.8% of simulated
    teams to 17.5%. Reassigning the weight here put them back to 11.7% and took tight
    end doubles to 5.0%.
    """
    roster_need: float = 0.70
    positional_scarcity: float = 0.35
    manager_position_preference: float = 0.60
    round_specific_preference: float = 0.40
    platform_rank_dependence: float = 0.50
    rookie_preference: float = 0.25
    favorite_team_preference: float = 0.20
    named_player_preference: float = 0.45
    """Pull toward (or away from) players the user named for this manager.

    Set above the positional and team preferences because it is far more specific:
    "he always takes Kupp" is a statement about one player, and a weight that only
    matched a positional lean would be drowned out by board value on every pick.
    Zero for every manager the user has said nothing about, so it changes no
    simulation until someone fills it in.
    """
    repeat_player_affinity: float = 0.25
    """Pull toward a player this manager has drafted in previous seasons.

    Below ``named_player_preference`` deliberately. That weight acts on something the
    user *stated*; this one acts on something inferred from a draft history, where
    re-drafting the same player is also consistent with him simply having been the best
    available at that slot each year. Zero for every manager with no repeat picks on
    record, so it changes nothing until a history is imported.
    """
    stack: float = 0.15
    handcuff: float = 0.15
    positional_run: float = 0.30
    expected_availability: float = 0.35
    strategy: float = 0.45
    randomness: float = 0.10
    injury_penalty: float = 0.60
    roster_imbalance_penalty: float = 1.50
    """Cost of a pick that strands a starting slot, per seat irrecoverably lost.

    Set above the widest realistic board-value gap between two candidates on
    purpose. This term only engages once a manager has no spare picks left, and at
    that point the choice is between a slightly better bench player and starting
    the season with an empty seat that scores zero every week. No board edge is
    worth that, so the penalty has to dominate rather than merely compete: at 0.30
    it lost to a 0.6 value edge and rosters finished with unfilled K/DST seats.
    """
    positional_limit_penalty: float = 0.80
    positional_surplus_penalty: float = 1.20
    """Cost of one body more at a position than the lineup can use, per body.

    Above the widest board-value edge between two candidates so that it decides the
    pick rather than merely joining the argument, and below
    ``roster_imbalance_penalty`` because stranding a starting seat is worse than
    wasting a bench one. Zero for every pick inside a position's useful depth, so it
    changes nothing about a normally shaped roster: what it stops is the 30% of
    simulated teams that finished a one-quarterback draft holding three quarterbacks,
    and the fifth of them that drafted two kickers.
    """
    bench_before_starters_penalty: float = 0.90
    """Cost of adding depth while the starting lineup still has holes, at its worst.

    Graded by how much of the lineup is still empty, so it is near full strength in the
    early rounds and gone by the time the starters are set. Set below the surplus
    penalty because it describes a question of *order* rather than of waste — a second
    tight end in round five is a bad pick, not an impossible one, and an outstanding
    player should still be able to outbid it, which at this weight he can.
    """
    premature_kicker_penalty: float = 1.50
    """Cost of taking a kicker or defence early, at its full round-one strength.

    The one thing every drafter in every league agrees on. Nothing else in the model
    expresses it: a kicker's ADP is the only thing holding him back, and on a blended
    board that ADP can read as round nine in a ten-team league, so a manager whose
    starters were full would take one in round five about once every four drafts. The
    penalty fades to nothing over the last few rounds, where taking one is correct, and
    the roster-imbalance penalty still guarantees the seat gets filled before the end.
    """

    def to_dict(self) -> dict[str, float]:
        return {f: float(getattr(self, f)) for f in self.__slots__}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelWeights":
        clean = {k: float(v) for k, v in (raw or {}).items() if k in cls.__slots__}
        return cls(**clean)

    def scaled(self, factor: float) -> "ModelWeights":
        return ModelWeights(**{k: v * factor for k, v in self.to_dict().items()})


@dataclass(slots=True)
class ShrinkageConfig:
    """Controls hierarchical blending of manager / league / platform / baseline.

    ``manager_weight = n / (n + prior_strength)`` — a manager with
    ``prior_strength`` observations gets 50% personalisation. The remainder is
    split across the fallback levels by the ratios below.
    """

    prior_strength: float = 24.0
    """Pseudo-count for manager-level shrinkage (in picks)."""
    season_prior_strength: float = 2.0
    """Pseudo-count in *seasons*, for metrics observed once per draft rather than
    once per pick (e.g. the round of a manager's first quarterback)."""
    draft_prior_strength: float = 1.0
    """Pseudo-count in *drafts*, discounting a per-pick sample drawn from few drafts.

    ``prior_strength`` treats sixteen picks as sixteen independent observations, and
    from a single draft they are nothing of the kind: one manager who spent a bad
    August chasing quarterbacks produces sixteen correlated picks that read as a
    settled personality. A new league with one year of history would have its
    managers modelled at 40% personalisation off that one afternoon. This shrinks the
    effective sample by ``drafts / (drafts + 1)`` — one draft counts half, two
    two-thirds, three three-quarters — so a single season still moves the model, just
    not as far as three seasons of the same tendency would."""
    league_share: float = 0.45
    platform_share: float = 0.30
    baseline_share: float = 0.25
    recency_half_life_seasons: float = 1.5
    """Half-life of the exponential recency decay applied to historical picks."""
    min_picks_for_observed: int = 8
    """Below this many picks a metric is labelled inferred rather than observed."""

    def fallback_shares(self) -> tuple[float, float, float]:
        total = self.league_share + self.platform_share + self.baseline_share
        if total <= 0:
            return (0.0, 0.0, 1.0)
        return (
            self.league_share / total,
            self.platform_share / total,
            self.baseline_share / total,
        )

    def manager_weight(self, sample_size: float) -> float:
        n = max(0.0, float(sample_size))
        return n / (n + max(1e-9, self.prior_strength))

    def season_weight(self, seasons: float) -> float:
        """Shrinkage weight for a per-season observation (e.g. first-QB round)."""
        n = max(0.0, float(seasons))
        return n / (n + max(1e-9, self.season_prior_strength))

    def cluster_weight(self, drafts: float) -> float:
        """How much of a per-pick sample to believe, given how many drafts it spans.

        Applied to the sample *size* rather than to the estimate, so it compounds with
        :meth:`manager_weight` instead of competing with it. See
        :attr:`draft_prior_strength`.
        """
        n = max(0.0, float(drafts))
        if n <= 0:
            return 0.0
        return n / (n + max(1e-9, self.draft_prior_strength))

    def recency_decay(self, seasons_ago: float) -> float:
        """Exponential weight for a pick made ``seasons_ago`` seasons back."""
        if self.recency_half_life_seasons <= 0:
            return 1.0
        return float(0.5 ** (max(0.0, seasons_ago) / self.recency_half_life_seasons))

    def to_dict(self) -> dict[str, float]:
        return {f: float(getattr(self, f)) for f in self.__slots__}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ShrinkageConfig":
        clean = {k: float(v) for k, v in (raw or {}).items() if k in cls.__slots__}
        if "min_picks_for_observed" in clean:
            clean["min_picks_for_observed"] = int(clean["min_picks_for_observed"])
        return cls(**clean)


@dataclass(slots=True)
class ProfileEstimationConfig:
    """Scales that convert raw historical statistics into model parameters.

    Every number here is an *anchor*: the observed statistic is divided by (or
    exponentially damped against) the anchor so the resulting parameter lands on
    the same 0-1 scale the archetype priors use. Anchors are configurable so a
    user whose league behaves unusually can recalibrate without touching code.
    """

    reach_clip_picks: float = 60.0
    """|ADP − pick| beyond this is treated as bad data and dropped."""
    rank_delta_scale_picks: float = 18.0
    """Mean |rank − pick| at which ``rank_dependence`` decays to 1/e (~0.37)."""
    predictability_scale_picks: float = 14.0
    """Reach standard deviation at which ``predictability`` decays to 1/e."""
    fill_rate_anchor: float = 0.62
    """Share of picks that fill an open starting slot for a league-average
    manager. Maps to ``need_dependence`` = 0.5."""
    run_continue_anchor: float = 0.34
    """Share of picks continuing a positional run for a league-average manager.
    Maps to ``run_chase`` = 0.5."""
    upside_anchor: float = 0.5
    """Share of picks with above-median ceiling that maps to ``risk_preference``
    = 0.5. Only used when the historical file carries ceiling data."""
    position_bias_timing_scale: float = 1.5
    """Utility units per unit of normalised draft-position lean vs the league."""
    position_bias_share_scale: float = 2.0
    """Utility units per unit of positional pick-share lean vs the league."""
    position_bias_clip: float = 0.60
    """Maximum magnitude of any derived positional bias, in utility units."""
    early_rounds: int = 3
    """Rounds counted as 'early' for the early-round positional bias."""
    favorite_team_min_share: float = 0.15
    """Minimum share of picks before a team counts as a favourite."""
    user_preference_weight: float = 0.60
    """How strongly a value the user typed in pulls a parameter toward it. 1.0
    would make user input an outright override."""
    run_window_picks: int = 6
    """Look-back window (picks) used when labelling historical positional runs."""
    run_threshold_picks: int = 2
    """Same-position picks inside the window that constitute a run."""

    def to_dict(self) -> dict[str, float]:
        return {f: float(getattr(self, f)) for f in self.__slots__}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ProfileEstimationConfig":
        clean = {k: float(v) for k, v in (raw or {}).items() if k in cls.__slots__}
        for key in ("early_rounds", "run_window_picks", "run_threshold_picks"):
            if key in clean:
                clean[key] = int(clean[key])
        return cls(**clean)


@dataclass(slots=True)
class SimulationConfig:
    """Knobs for the pick model and Monte Carlo layer."""

    weights: ModelWeights = field(default_factory=ModelWeights)
    shrinkage: ShrinkageConfig = field(default_factory=ShrinkageConfig)
    estimation: ProfileEstimationConfig = field(default_factory=ProfileEstimationConfig)
    ranking_source: RankingSource = RankingSource.BLEND
    blend_weights: dict[str, float] = field(default_factory=lambda: {
        "platform_adp": 0.45, "overall_adp": 0.25, "projection": 0.30,
    })
    base_temperature: float = 0.55
    """Softmax temperature at average predictability."""
    temperature_range: tuple[float, float] = (0.22, 1.35)
    """Clamp for per-manager temperature (predictable → volatile)."""
    early_round_temperature: float = 0.55
    """Temperature multiplier in round 1, easing to 1.0 by ``early_round_rounds``.

    Rounds 1-3 of a real draft barely deviate from consensus: the crowd's top twelve
    are the top twelve, and a manager who deviates does it by one or two slots, not by
    ten. Later on the same manager is genuinely making it up — the board is flat, the
    rankings disagree with each other, and personal preference is most of the decision.
    One temperature for the whole draft cannot say both things, so the early rounds get
    a colder one.
    """
    early_round_rounds: int = 4
    """Round by which the early-round temperature discount is fully gone."""
    candidate_pool_size: int = 40
    """How many top-utility players enter the softmax. Keeps sims fast."""
    adp_sigma_floor: float = 6.0
    """Minimum ADP standard deviation when a player file supplies none."""
    adp_sigma_round_growth: float = 1.6
    """Extra ADP sigma per round — later rounds are noisier."""
    run_windows: tuple[int, ...] = (3, 6, 12)
    """Look-back windows (in picks) used to detect positional runs."""
    availability_simulations: int = 120
    """Rollouts used when estimating survival to the user's next pick."""
    monte_carlo_default_runs: int = 200
    reach_scale_picks: float = 12.0
    """ADP delta (picks) treated as one unit of 'reach' when scoring utility."""
    random_seed: int | None = None

    def temperature_for(
        self, predictability: float, round_number: int | None = None
    ) -> float:
        """Map predictability in [0, 1] to a softmax temperature.

        Higher predictability → lower temperature → more deterministic picks.

        ``round_number`` applies the early-round discount described on
        :attr:`early_round_temperature`, interpolating linearly back to no discount by
        :attr:`early_round_rounds`. Omitting it asks for the manager's baseline
        temperature, which is what the settings preview and the profile displays want.

        The discount is applied *after* the range clamp, deliberately: the clamp bounds
        how volatile a manager can be over a draft, and a very predictable manager in
        round 1 is allowed to be more deterministic than that floor.
        """
        lo, hi = self.temperature_range
        p = min(1.0, max(0.0, float(predictability)))
        # Predictability 0.5 reproduces base_temperature.
        if p >= 0.5:
            t = lo + (self.base_temperature - lo) * ((1.0 - p) / 0.5)
        else:
            t = self.base_temperature + (hi - self.base_temperature) * ((0.5 - p) / 0.5)
        return float(min(hi, max(lo, t))) * self.round_temperature_factor(round_number)

    def round_temperature_factor(self, round_number: int | None) -> float:
        """How much of the early-round temperature discount applies in this round."""
        if round_number is None:
            return 1.0
        rounds = int(self.early_round_rounds)
        start = float(self.early_round_temperature)
        r = max(1, int(round_number))
        if rounds <= 1 or r >= rounds:
            return 1.0
        progress = float(r - 1) / float(rounds - 1)
        return float(start + (1.0 - start) * progress)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.to_dict(),
            "shrinkage": self.shrinkage.to_dict(),
            "estimation": self.estimation.to_dict(),
            "ranking_source": str(self.ranking_source),
            "blend_weights": dict(self.blend_weights),
            "base_temperature": self.base_temperature,
            "temperature_range": list(self.temperature_range),
            "early_round_temperature": self.early_round_temperature,
            "early_round_rounds": self.early_round_rounds,
            "candidate_pool_size": self.candidate_pool_size,
            "adp_sigma_floor": self.adp_sigma_floor,
            "adp_sigma_round_growth": self.adp_sigma_round_growth,
            "run_windows": list(self.run_windows),
            "availability_simulations": self.availability_simulations,
            "monte_carlo_default_runs": self.monte_carlo_default_runs,
            "reach_scale_picks": self.reach_scale_picks,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SimulationConfig":
        raw = dict(raw or {})
        cfg = cls(
            weights=ModelWeights.from_dict(raw.get("weights") or {}),
            shrinkage=ShrinkageConfig.from_dict(raw.get("shrinkage") or {}),
            estimation=ProfileEstimationConfig.from_dict(raw.get("estimation") or {}),
            ranking_source=RankingSource.coerce(
                raw.get("ranking_source"), RankingSource.BLEND) or RankingSource.BLEND,
            blend_weights=dict(raw.get("blend_weights") or {
                "platform_adp": 0.45, "overall_adp": 0.25, "projection": 0.30}),
        )
        for key in ("base_temperature", "early_round_temperature", "adp_sigma_floor",
                    "adp_sigma_round_growth", "reach_scale_picks"):
            if key in raw:
                setattr(cfg, key, float(raw[key]))
        for key in ("early_round_rounds", "candidate_pool_size",
                    "availability_simulations", "monte_carlo_default_runs"):
            if key in raw:
                setattr(cfg, key, int(raw[key]))
        if raw.get("temperature_range"):
            lo, hi = raw["temperature_range"]
            cfg.temperature_range = (float(lo), float(hi))
        if raw.get("run_windows"):
            cfg.run_windows = tuple(int(w) for w in raw["run_windows"])
        if raw.get("random_seed") is not None:
            cfg.random_seed = int(raw["random_seed"])
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Archetype parameter table
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ArchetypeParams:
    """Starting model parameters for a fallback archetype.

    These are *priors*, not rules: they seed a profile which the probabilistic
    engine then blends with roster need, runs, and ADP.
    """

    label: str
    description: str
    position_bias: dict[Position, float] = field(default_factory=dict)
    """Additive utility bias per position, in utility units."""
    early_round_position_bias: dict[Position, float] = field(default_factory=dict)
    """Extra bias applied in rounds 1-3 only."""
    first_qb_round: float = 8.0
    first_te_round: float = 8.0
    reach_mean_picks: float = 0.0
    """Positive = drafts ahead of ADP on average."""
    reach_stdev_picks: float = 9.0
    rank_dependence: float = 0.5
    """0 = ignores platform ranks, 1 = drafts strictly by rank."""
    need_dependence: float = 0.5
    rookie_rate: float = 0.10
    stack_rate: float = 0.08
    handcuff_rate: float = 0.06
    favorite_team_rate: float = 0.05
    run_chase: float = 0.5
    """0 = disciplined against positional runs, 1 = chases them."""
    predictability: float = 0.5
    risk_preference: float = 0.5
    """0 = floor-seeking, 1 = ceiling-seeking."""


def _pos(**kwargs: float) -> dict[Position, float]:
    return {Position[k]: float(v) for k, v in kwargs.items()}


ARCHETYPE_PARAMS: dict[Archetype, ArchetypeParams] = {
    Archetype.BALANCED: ArchetypeParams(
        label="Balanced",
        description="Drafts near consensus, fills needs without extreme lean.",
    ),
    Archetype.BEST_PLAYER_AVAILABLE: ArchetypeParams(
        label="Best Player Available",
        description="Follows value over need; rarely reaches.",
        need_dependence=0.22, rank_dependence=0.72, reach_mean_picks=-1.5,
        reach_stdev_picks=6.0, predictability=0.72,
    ),
    Archetype.ZERO_RB: ArchetypeParams(
        label="Zero RB",
        description="Avoids RB early, loads WR/TE, attacks RB from the middle rounds.",
        position_bias=_pos(RB=-0.25, WR=0.30, TE=0.15),
        early_round_position_bias=_pos(RB=-0.95, WR=0.60),
        need_dependence=0.35, predictability=0.62, risk_preference=0.68,
    ),
    Archetype.HERO_RB: ArchetypeParams(
        label="Hero RB",
        description="One elite RB early, then WR-heavy for several rounds.",
        early_round_position_bias=_pos(RB=0.35, WR=0.25),
        position_bias=_pos(RB=-0.10, WR=0.18),
        predictability=0.58, risk_preference=0.55,
    ),
    Archetype.ROBUST_RB: ArchetypeParams(
        label="Robust RB",
        description="Prioritises multiple early running backs.",
        position_bias=_pos(RB=0.32),
        early_round_position_bias=_pos(RB=0.85, WR=-0.20),
        need_dependence=0.6, predictability=0.62, risk_preference=0.38,
    ),
    Archetype.EARLY_QB: ArchetypeParams(
        label="Early QB",
        description="Takes a quarterback well before league average.",
        position_bias=_pos(QB=0.45), first_qb_round=4.0,
        reach_mean_picks=4.0, predictability=0.55,
    ),
    Archetype.LATE_QB: ArchetypeParams(
        label="Late-Round QB",
        description="Waits on quarterback, often the last starter drafted.",
        position_bias=_pos(QB=-0.35), first_qb_round=11.0,
        early_round_position_bias=_pos(QB=-1.10), predictability=0.65,
    ),
    Archetype.ELITE_TE: ArchetypeParams(
        label="Elite TE",
        description="Pays up for a top tight end inside the first few rounds.",
        position_bias=_pos(TE=0.40), first_te_round=3.0,
        early_round_position_bias=_pos(TE=0.75), reach_mean_picks=3.0,
    ),
    Archetype.ROOKIE_HEAVY: ArchetypeParams(
        label="Rookie Heavy",
        description="Reaches for first-year players and upside profiles.",
        rookie_rate=0.34, reach_mean_picks=6.0, reach_stdev_picks=13.0,
        risk_preference=0.80, predictability=0.40,
    ),
    Archetype.RANK_FOLLOWER: ArchetypeParams(
        label="Platform Rank Follower",
        description="Drafts almost strictly off the platform's default list.",
        rank_dependence=0.95, need_dependence=0.28, reach_mean_picks=-0.5,
        reach_stdev_picks=3.5, predictability=0.90,
    ),
    Archetype.HIGH_VARIANCE: ArchetypeParams(
        label="High Variance",
        description="Unpredictable; frequent large reaches in both directions.",
        reach_stdev_picks=20.0, predictability=0.15, risk_preference=0.75,
        rank_dependence=0.30, run_chase=0.7,
    ),
    Archetype.HOMER: ArchetypeParams(
        label="Favorite-Team Homer",
        description="Repeatedly drafts players from one favourite NFL team.",
        favorite_team_rate=0.30, reach_mean_picks=5.0, predictability=0.42,
    ),
    Archetype.AUTODRAFT: ArchetypeParams(
        label="Autodraft-Like",
        description="Effectively drafts the default queue in order.",
        rank_dependence=1.0, need_dependence=0.45, reach_stdev_picks=2.0,
        predictability=0.96, run_chase=0.15, rookie_rate=0.05,
    ),
    Archetype.CUSTOM: ArchetypeParams(
        label="Custom",
        description="User-defined parameters only.",
    ),
}


def archetype_params(archetype: Archetype | str | None) -> ArchetypeParams:
    """Look up archetype priors, defaulting to Balanced."""
    key = Archetype.coerce(archetype, Archetype.BALANCED) or Archetype.BALANCED
    return ARCHETYPE_PARAMS.get(key, ARCHETYPE_PARAMS[Archetype.BALANCED])


# ─────────────────────────────────────────────────────────────────────────────
# Paths / app settings
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class AppPaths:
    """Filesystem locations, resolved relative to the package root."""

    root: str = ""
    database: str = ""
    cache: str = ""
    exports: str = ""

    @classmethod
    def default(cls) -> "AppPaths":
        import os

        # There is deliberately no ``sample_data`` path here any more: the synthetic
        # league moved to ``tests/fixtures/sample_league`` and the app has no route
        # to it, so a path pointing at a directory that no longer exists would only
        # invite something to start reading from it again.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return cls(
            root=root,
            database=os.path.join(root, "data", "fantasy_mock_draft.db"),
            cache=os.path.join(root, "data", "cache"),
            exports=os.path.join(root, "data", "exports"),
        )


DEFAULT_PATHS = AppPaths.default()


def positions_in_use(roster: RosterSettings) -> set[Position]:
    """Positions worth drafting given the league's slot configuration."""
    used: set[Position] = set()
    for slot, n in roster.starting_slots.items():
        if n > 0:
            used |= set(SLOT_ELIGIBILITY.get(slot, frozenset()))
    if not used:
        used = {Position.QB, Position.RB, Position.WR, Position.TE}
    return used
