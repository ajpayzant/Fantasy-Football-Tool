"""Pick recommendations for the user, from several deliberately different angles.

The engine answers one question — "who should I take?" — eight ways, because there
is no single correct answer to it. A user at pick 3.04 genuinely faces a trade-off
between the best player left, the one who fits their roster, and the one who will
not survive the twelve picks until their next turn. Collapsing that into one
ranking hides the decision instead of informing it.

Each :class:`RecommendationLens` gets its own scoring rule and its own reason
text. The lenses share one expensive input: a single
:class:`~engine.simulator.AvailabilityReport` from the Monte Carlo rollouts. That
is the whole reason this module exists separately from :mod:`engine.pick_model` —
the pick model scores candidates with a cheap closed-form survival estimate
because it runs inside every AI pick, while the user's own recommendation can
afford to roll the board forward a hundred times and get the real answer.

**What "value" means here.** Every lens scores in *utility* units from the pick
model, using the user's own manager profile. The user is modelled as a manager
like any other, so the recommendation is calibrated to the same board the
opponents are picking from rather than to an abstract best-player list.

Nothing here imports Streamlit. The UI renders :class:`Recommendation` objects; it
does not compute them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from core.enums import Position, RecommendationLens, RiskBand
from engine.draft_state import DraftState
from engine.pick_model import (
    PickContext,
    RosterView,
    ScoredCandidate,
    context_for,
    pick_probabilities,
    score_candidates,
)
from engine.simulator import (
    AvailabilityReport,
    simulate_availability,
    upcoming_position_pressure,
)
from models.manager import ManagerProfile, baseline_profile
from models.player import Player

LOGGER = logging.getLogger("fantasy_mock_draft.recommender")

# ─────────────────────────────────────────────────────────────────────────────
# Lens tuning. Every number that shapes a recommendation lives here with its
# reasoning, rather than inline where it would read as arbitrary.
# ─────────────────────────────────────────────────────────────────────────────
LAST_CHANCE_SURVIVAL: float = 0.35
"""Survival at or below which a player counts as "now or never".

Set below the coin-flip band on purpose: a player with a 45% chance of lasting is
a real risk but not a reason to abandon your board, and labelling him "last
chance" would make the strongest label on the page fire almost every pick.
"""

SAFE_TO_WAIT_SURVIVAL: float = 0.80
"""Survival at or above which waiting a round is a reasonable plan."""

SCARCITY_PRESSURE_PICKS: float = 1.5
"""Expected opponent picks at a position before your turn that counts as a run.

Below roughly this, positional demand ahead of you is noise; above it, the
position is genuinely being drained and the tier you want may not return.
"""

UPSIDE_FLOOR_PENALTY: float = 0.35
"""How much a wide downside discounts a high-ceiling pick.

Upside is not free: the same volatility that produces the ceiling produces the
floor. At 0.35 a boom/bust player still wins the upside lens over a safe one with
the same ceiling, but a player whose downside is twice his upside does not.
"""

VALUE_LENS_SPAN_PICKS: float = 18.0
"""ADP fall (in picks) treated as one full unit of "value" in the value lens."""

ALTERNATIVE_MIN_UTILITY_SHARE: float = 0.80
"""Floor on a strategic alternative's utility, as a share of the best candidate.

Prevents the lens from recommending a genuinely worse player merely because he is
positionally different. An alternative must be a real option, not a curiosity.
"""

SHORTLIST_SIZE: int = 12
"""Candidates carried into the availability rollouts and the lens scoring."""

MIN_CONFIDENCE_SAMPLE: int = 20
"""Rollouts below which survival numbers are reported as indicative only."""

BYE_STACK_WARNING: int = 4
"""Starters sharing one bye week before it is worth warning about.

Three is normal and unavoidable in a 9-starter lineup; four means a week where
half the lineup is out.
"""


@dataclass(slots=True)
class Recommendation:
    """One lens's answer, with the reasoning that produced it."""

    lens: RecommendationLens
    player: Player
    score: float
    """The lens's own score. Comparable *within* a lens, not across lenses."""
    utility: float
    """The pick model's utility for this player, for cross-lens comparison."""
    survival: float
    risk_band: RiskBand
    headline: str
    """One line, written for the user: what this pick is and why."""
    detail: list[str] = field(default_factory=list)
    """Supporting bullets — component contributions, roster fit, availability."""
    components: dict[str, float] = field(default_factory=dict)
    is_consensus: bool = False
    """True when more than one lens picked this player. Set by the engine."""

    @property
    def player_id(self) -> str:
        return self.player.player_id

    @property
    def lens_label(self) -> str:
        return str(self.lens).replace("_", " ").title()

    def to_dict(self) -> dict[str, Any]:
        return {
            "lens": str(self.lens),
            "lens_label": self.lens_label,
            "player_id": self.player.player_id,
            "player_name": self.player.name,
            "position": str(self.player.position),
            "nfl_team": self.player.nfl_team,
            "bye_week": self.player.bye_week,
            "score": round(float(self.score), 4),
            "utility": round(float(self.utility), 4),
            "survival": round(float(self.survival), 4),
            "risk_band": str(self.risk_band),
            "headline": self.headline,
            "detail": list(self.detail),
            "is_consensus": self.is_consensus,
        }


@dataclass(slots=True)
class RecommendationSet:
    """Every lens's answer for one pick, plus the shared context behind them."""

    overall_pick: int
    round_number: int
    draft_slot: int
    recommendations: list[Recommendation] = field(default_factory=list)
    availability: AvailabilityReport | None = None
    pressure: dict[Position, float] = field(default_factory=dict)
    picks_until_next: int = 0
    roster_summary: str = ""
    warnings: list[str] = field(default_factory=list)
    """Things the user should know regardless of which lens they follow."""
    elapsed_seconds: float = 0.0

    def by_lens(self, lens: RecommendationLens) -> Recommendation | None:
        for rec in self.recommendations:
            if rec.lens is lens:
                return rec
        return None

    @property
    def primary(self) -> Recommendation | None:
        """The headline suggestion: best overall, or the first lens that fired."""
        return (
            self.by_lens(RecommendationLens.BEST_OVERALL)
            or (self.recommendations[0] if self.recommendations else None)
        )

    @property
    def consensus_players(self) -> list[Player]:
        """Players more than one lens agreed on, most-agreed first.

        Agreement across lenses is the strongest signal the engine produces: a
        player who is simultaneously the best available, the best fit, and about
        to disappear is not a close call.
        """
        counts: dict[str, int] = {}
        players: dict[str, Player] = {}
        for rec in self.recommendations:
            counts[rec.player_id] = counts.get(rec.player_id, 0) + 1
            players[rec.player_id] = rec.player
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        return [players[pid] for pid, n in ranked if n > 1]

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame([r.to_dict() for r in self.recommendations])


# ─────────────────────────────────────────────────────────────────────────────
# Lens scoring helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fit_score(candidate: ScoredCandidate, view: RosterView) -> float:
    """How well this player fits the roster's *unfilled* needs.

    Reads the need and scarcity components the pick model already computed rather
    than recomputing them, so the fit lens cannot drift out of agreement with the
    utility that drives every other manager's behaviour.
    """
    components = candidate.components
    fit = float(components.get("roster_need", 0.0))
    fit += float(components.get("positional_scarcity", 0.0)) * 0.5
    fit += float(components.get("roster_imbalance_penalty", 0.0))
    if view.fills_starting_slot(candidate.player.position):
        fit += 0.35
    return fit


def _value_score(candidate: ScoredCandidate, overall_pick: int) -> float:
    """How far past his ADP the player has fallen, in units of ~18 picks.

    Uses the raw ADP gap rather than the pick model's ``adp_value`` component
    because that component is already multiplied by a configured weight; the value
    lens wants the unweighted fall so a user can see "he is 22 picks past ADP".
    """
    adp = candidate.player.adp_for()
    if adp is None:
        return 0.0
    return float(overall_pick - adp) / VALUE_LENS_SPAN_PICKS


def _upside_score(candidate: ScoredCandidate, pool_span: float) -> float:
    """Ceiling above projection, penalised by the downside that comes with it."""
    player = candidate.player
    if pool_span <= 0:
        return 0.0
    upside = float(player.upside) / pool_span
    downside = float(player.downside) / pool_span
    return upside - UPSIDE_FLOOR_PENALTY * downside


def _safety_score(candidate: ScoredCandidate) -> float:
    """Reliability: high floor, healthy, established.

    Note this is *roster* safety — will this player produce — and is unrelated to
    availability. A player can be the safest pick on the board and also certain to
    be gone by your next turn; the two are reported separately because they lead
    to opposite actions.
    """
    player = candidate.player
    projection = float(player.projection or 0.0)
    floor_share = (
        float(player.floor) / projection if player.floor and projection > 0 else 0.0
    )
    score = floor_share
    score -= float(player.injury_penalty)
    score -= float(player.risk_score or 0.0)
    if player.is_rookie:
        # Rookies carry genuine outcome variance regardless of their projection,
        # so the safety lens discounts them rather than ranking them on floor
        # alone — a rookie's "floor" is the least trustworthy number on his line.
        score -= 0.20
    if player.suspended:
        score -= 0.50
    return score


def _describe_availability(entry_survival: float, picks_until_next: int) -> str:
    if picks_until_next <= 0:
        return "you pick again immediately"
    pct = f"{entry_survival:.0%}"
    if entry_survival >= SAFE_TO_WAIT_SURVIVAL:
        return f"{pct} chance he lasts your next {picks_until_next} picks — you can wait"
    if entry_survival <= LAST_CHANCE_SURVIVAL:
        return f"only a {pct} chance he lasts {picks_until_next} more picks"
    return f"{pct} chance he is still there in {picks_until_next} picks"


def _reason_bullets(
    candidate: ScoredCandidate,
    survival: float,
    picks_until_next: int,
    view: RosterView,
) -> list[str]:
    """Shared supporting detail every lens shows beneath its headline."""
    player = candidate.player
    bullets: list[str] = []
    adp = player.adp_for()
    if adp is not None:
        bullets.append(f"ADP {adp:.1f}")
    if player.projection is not None:
        bullets.append(f"projected {player.projection:.1f}")
    if player.tier is not None:
        bullets.append(f"tier {int(player.tier)}")
    slot = view.starting_slot_for(player.position)
    bullets.append(
        f"fills your open {str(slot).upper()} seat" if slot
        else "would go to your bench"
    )
    bullets.append(_describe_availability(survival, picks_until_next))
    reasons = candidate.top_reasons(3)
    if reasons:
        bullets.append(
            "model drivers: "
            + ", ".join(f"{k.replace('_', ' ')} {v:+.2f}" for k, v in reasons)
        )
    return bullets


# ─────────────────────────────────────────────────────────────────────────────
# The engine
# ─────────────────────────────────────────────────────────────────────────────
class RecommendationEngine:
    """Builds the eight-lens recommendation set for whoever is on the clock.

    Stateless between calls apart from the profiles it was constructed with:
    :meth:`recommend` reads the draft state and returns a fresh set, so the UI can
    call it after every board change without invalidating anything.
    """

    __slots__ = ("state", "profiles")

    def __init__(
        self, state: DraftState, profiles: Mapping[int, ManagerProfile]
    ) -> None:
        self.state = state
        self.profiles = dict(profiles)

    # -- context ---------------------------------------------------------
    def _profile_for(self, draft_slot: int) -> ManagerProfile:
        """The profile to recommend with, substituting a baseline if there is none.

        Unlike the simulator, a missing profile for the slot *being advised* is
        recoverable and common: a brand-new user has no draft history to model. A
        baseline profile is the honest stand-in, and the recommendation is for a
        human who is about to override it anyway.

        The substitute is stored back into ``self.profiles`` rather than used
        locally, because the availability rollouts run through the simulator — which
        deliberately raises on a missing profile — and would otherwise reject the
        very slot this method just accommodated. Storing it also keeps the rollouts
        modelling the user with the same profile the recommendation is scored under.
        """
        slot = int(draft_slot)
        profile = self.profiles.get(slot)
        if profile is None:
            LOGGER.info(
                "No profile for slot %s; recommending with a baseline profile", slot
            )
            profile = baseline_profile(f"Slot {slot}")
            self.profiles[slot] = profile
        return profile

    def _context(self, draft_slot: int) -> PickContext:
        return context_for(
            self.state, self._profile_for(draft_slot), draft_slot=int(draft_slot)
        )

    def _shortlist(
        self, context: PickContext, shortlist_size: int
    ) -> list[ScoredCandidate]:
        ranked = score_candidates(context)
        if not ranked:
            return []
        temperature = self.state.settings.temperature_for(
            float(context.profile.get("predictability"))
        )
        pick_probabilities(ranked, temperature)
        return ranked[:shortlist_size]

    # -- the public call --------------------------------------------------
    def recommend(
        self,
        *,
        draft_slot: int | None = None,
        simulations: int | None = None,
        shortlist_size: int = SHORTLIST_SIZE,
        availability: AvailabilityReport | None = None,
        seed: int | None = None,
    ) -> RecommendationSet:
        """Score every lens for the current pick.

        ``availability`` lets a caller reuse rollouts it already ran — the UI does
        this so the availability table and the recommendations cannot disagree.
        When omitted, rollouts are run here.
        """
        started = time.perf_counter()
        slot = self.state.current_slot
        if slot is None:
            return RecommendationSet(
                overall_pick=len(self.state.order), round_number=0, draft_slot=0,
                warnings=["The draft is complete — there is nothing to recommend."],
            )
        target = int(draft_slot if draft_slot is not None else slot.draft_slot)
        context = self._context(target)
        shortlist = self._shortlist(context, max(1, int(shortlist_size)))
        if not shortlist:
            return RecommendationSet(
                overall_pick=int(slot.overall_pick),
                round_number=int(slot.round_number),
                draft_slot=target,
                warnings=["No draftable players remain."],
            )

        report = availability if availability is not None else simulate_availability(
            self.state,
            self.profiles,
            draft_slot=target,
            simulations=simulations,
            extra_players=[c.player for c in shortlist],
            seed=seed,
        )
        pressure = upcoming_position_pressure(
            self.state, self.profiles, draft_slot=target
        )
        result = RecommendationSet(
            overall_pick=int(slot.overall_pick),
            round_number=int(slot.round_number),
            draft_slot=target,
            availability=report,
            pressure=pressure,
            picks_until_next=int(report.picks_until_next),
            roster_summary=self._roster_summary(context),
        )
        result.recommendations = self._build_lenses(context, shortlist, report, pressure)
        self._mark_consensus(result)
        result.warnings = self._warnings(context, shortlist, report, pressure)
        result.elapsed_seconds = time.perf_counter() - started
        LOGGER.info(
            "Recommendations for pick %s: %d lenses in %.2fs",
            slot.label, len(result.recommendations), result.elapsed_seconds,
        )
        return result

    # -- lenses ----------------------------------------------------------
    def _build_lenses(
        self,
        context: PickContext,
        shortlist: Sequence[ScoredCandidate],
        report: AvailabilityReport,
        pressure: Mapping[Position, float],
    ) -> list[Recommendation]:
        view = context.view
        gap = int(report.picks_until_next)
        pool_span = self._projection_span()

        def survival_of(candidate: ScoredCandidate) -> float:
            return report.survival(candidate.player_id)

        def build(
            lens: RecommendationLens,
            candidate: ScoredCandidate,
            score: float,
            headline: str,
            extra: Iterable[str] = (),
        ) -> Recommendation:
            detail = _reason_bullets(candidate, survival_of(candidate), gap, view)
            detail.extend(extra)
            return Recommendation(
                lens=lens,
                player=candidate.player,
                score=float(score),
                utility=float(candidate.utility),
                survival=survival_of(candidate),
                risk_band=report.band(candidate.player_id),
                headline=headline,
                detail=detail,
                components=dict(candidate.components),
            )

        out: list[Recommendation] = []
        best_utility = max(c.utility for c in shortlist)

        # 1. BEST_OVERALL — the pick model's own top choice, unmodified. This is
        #    the anchor the other lenses are alternatives *to*.
        best = max(shortlist, key=lambda c: c.utility)
        out.append(build(
            RecommendationLens.BEST_OVERALL, best, best.utility,
            f"{best.player.name} is the strongest all-round pick on the board "
            f"({best.probability:.0%} of managers in your spot would take him)",
        ))

        # 2. BEST_FIT — need and scarcity, not raw value.
        fit = max(shortlist, key=lambda c: _fit_score(c, view))
        fit_value = _fit_score(fit, view)
        out.append(build(
            RecommendationLens.BEST_FIT, fit, fit_value,
            f"{fit.player.name} does the most for this roster's shape",
            extra=[
                f"{view.open_starter_count} starting seats still unfilled",
            ],
        ))

        # 3. BEST_VALUE — furthest past ADP. Only fires on an actual faller.
        value = max(shortlist, key=lambda c: _value_score(c, context.overall_pick))
        value_score = _value_score(value, context.overall_pick)
        if value_score > 0:
            adp = value.player.adp_for() or 0.0
            fall = context.overall_pick - adp
            out.append(build(
                RecommendationLens.BEST_VALUE, value, value_score,
                f"{value.player.name} has fallen {fall:.0f} picks past his ADP "
                f"of {adp:.0f}",
            ))

        # 4. SAFEST — highest floor, healthiest, least variance.
        safe = max(shortlist, key=_safety_score)
        out.append(build(
            RecommendationLens.SAFEST, safe, _safety_score(safe),
            f"{safe.player.name} is the most reliable producer here",
            extra=[
                f"floor {safe.player.floor:.1f}" if safe.player.floor
                else "no published floor — ranked on health and experience",
            ],
        ))

        # 5. HIGHEST_UPSIDE — ceiling above projection, discounted by downside.
        upside = max(shortlist, key=lambda c: _upside_score(c, pool_span))
        if _upside_score(upside, pool_span) > 0:
            out.append(build(
                RecommendationLens.HIGHEST_UPSIDE, upside,
                _upside_score(upside, pool_span),
                f"{upside.player.name} has the widest path to a league-winning "
                f"season ({upside.player.upside:+.0f} over his projection)",
            ))

        # 6. SCARCITY — the position about to be drained, if any is.
        scarce_position = self._scarcest_position(pressure, view)
        if scarce_position is not None:
            at_position = [
                c for c in shortlist if c.player.position is scarce_position
            ]
            if at_position:
                scarce = max(at_position, key=lambda c: c.utility)
                demand = float(pressure.get(scarce_position, 0.0))
                out.append(build(
                    RecommendationLens.SCARCITY, scarce, demand,
                    f"{scarce.player.name} pre-empts a run: "
                    f"{str(scarce_position).upper()} is drying up, with about "
                    f"{demand:.1f} of the next {gap} picks projected to go there",
                    extra=[
                        f"{view.remaining_at(scarce_position)} "
                        f"{str(scarce_position).upper()}s left on the board",
                    ],
                ))

        # 7. LAST_CHANCE — a player you will not see again, ranked by what you
        #    would lose. Utility × the odds he is gone: a marginal player who is
        #    certain to vanish is not a crisis.
        endangered = [
            c for c in shortlist
            if report.survival(c.player_id) <= LAST_CHANCE_SURVIVAL
        ]
        if endangered and gap > 0:
            last = max(
                endangered,
                key=lambda c: c.utility * (1.0 - report.survival(c.player_id)),
            )
            entry = report.get(last.player_id)
            taker = entry.likeliest_taker if entry else None
            out.append(build(
                RecommendationLens.LAST_CHANCE, last,
                last.utility * (1.0 - report.survival(last.player_id)),
                f"{last.player.name} will almost certainly be gone — take him now "
                f"or move on",
                extra=(
                    [f"most often taken by {taker} in the rollouts"] if taker else []
                ),
            ))

        # 8. ALTERNATIVE — the best candidate at a *different* position from the
        #    headline pick, so the user always sees the road not taken. Held to a
        #    utility floor so it is a real option rather than a token contrast.
        floor = best_utility * ALTERNATIVE_MIN_UTILITY_SHARE
        alternatives = [
            c for c in shortlist
            if c.player.position is not best.player.position and c.utility >= floor
        ]
        if alternatives:
            alternative = max(alternatives, key=lambda c: c.utility)
            out.append(build(
                RecommendationLens.ALTERNATIVE, alternative, alternative.utility,
                f"If you would rather not take a "
                f"{str(best.player.position).upper()} here, "
                f"{alternative.player.name} is the best {str(alternative.player.position).upper()} "
                f"on the board",
            ))
        return out

    # -- supporting analysis ---------------------------------------------
    def _projection_span(self) -> float:
        """Spread of projections across the board, used to normalise upside.

        Falls back to 1.0 on a pool with no projections so the upside lens scores
        zero for everyone rather than dividing by zero — which correctly means
        "this file cannot answer the upside question".
        """
        available = self.state.available_players(limit=64)
        values = [
            float(p.projection) for p in available if p.projection is not None
        ]
        if len(values) < 2:
            return 1.0
        span = max(values) - min(values)
        return span if span > 0 else 1.0

    def _scarcest_position(
        self, pressure: Mapping[Position, float], view: RosterView
    ) -> Position | None:
        """The position most worth pre-empting: high demand, thin supply, needed.

        Restricted to positions the user actually has an open starting seat for.
        Warning a user that kickers are drying up when they already have one is
        noise, and the scarcity lens is the one most likely to be acted on
        impulsively.
        """
        best: Position | None = None
        best_demand = SCARCITY_PRESSURE_PICKS
        for position, demand in pressure.items():
            if float(demand) <= best_demand:
                continue
            if not view.fills_starting_slot(position):
                continue
            if view.remaining_at(position) <= 0:
                continue
            best, best_demand = position, float(demand)
        return best

    def _roster_summary(self, context: PickContext) -> str:
        view = context.view
        counts = context.roster.position_counts()
        shape = ", ".join(
            f"{count}{str(position).upper()}"
            for position, count in sorted(counts.items(), key=lambda kv: str(kv[0]))
        ) or "empty"
        return (
            f"{len(context.roster)} drafted ({shape}); "
            f"{view.open_starter_count} starting seats open, "
            f"{context.picks_left_for_manager} picks remaining"
        )

    def _mark_consensus(self, result: RecommendationSet) -> None:
        counts: dict[str, int] = {}
        for rec in result.recommendations:
            counts[rec.player_id] = counts.get(rec.player_id, 0) + 1
        for rec in result.recommendations:
            rec.is_consensus = counts.get(rec.player_id, 0) > 1

    def _warnings(
        self,
        context: PickContext,
        shortlist: Sequence[ScoredCandidate],
        report: AvailabilityReport,
        pressure: Mapping[Position, float],
    ) -> list[str]:
        """Things true regardless of which lens the user follows."""
        warnings: list[str] = []
        view = context.view
        unmet = view.open_starter_count
        picks_left = context.picks_left_for_manager
        if unmet > picks_left:
            warnings.append(
                f"You have {unmet} starting seats to fill and only {picks_left} "
                f"picks left — you cannot field a full lineup unless you start "
                f"filling them now."
            )
        if 0 < report.simulations < MIN_CONFIDENCE_SAMPLE:
            warnings.append(
                f"Availability is based on only {report.simulations} rollouts; "
                "treat the percentages as indicative."
            )
        endangered = [
            c for c in shortlist
            if report.survival(c.player_id) <= LAST_CHANCE_SURVIVAL
        ]
        if len(endangered) >= max(3, len(shortlist) // 2):
            warnings.append(
                f"{len(endangered)} of your top {len(shortlist)} targets are "
                f"unlikely to survive {report.picks_until_next} picks — this is a "
                "cliff, not a normal wait."
            )
        # Bye-week stacking, computed from the pool rather than the roster: the
        # roster stores ids and positions only, deliberately, so it works without
        # the pool. Counting starters only — a bench bye is not a lineup problem.
        byes: dict[int, int] = {}
        for pid in context.roster.starters():
            player = self.state.pool.get(pid)
            if player is not None and player.bye_week:
                byes[int(player.bye_week)] = byes.get(int(player.bye_week), 0) + 1
        for week, count in sorted(byes.items()):
            if count >= BYE_STACK_WARNING:
                warnings.append(
                    f"{count} of your projected starters share a week {week} bye."
                )
        return warnings


def recommend_for(
    state: DraftState,
    profiles: Mapping[int, ManagerProfile],
    **kwargs: Any,
) -> RecommendationSet:
    """Convenience wrapper: build an engine and ask it once.

    The engine is cheap to construct, so a caller that recommends only occasionally
    need not hold one.
    """
    return RecommendationEngine(state, profiles).recommend(**kwargs)


__all__ = [
    "Recommendation", "RecommendationSet", "RecommendationEngine", "recommend_for",
    "LAST_CHANCE_SURVIVAL", "SAFE_TO_WAIT_SURVIVAL", "SCARCITY_PRESSURE_PICKS",
    "SHORTLIST_SIZE",
]
