"""Is the top of the board realistic? Early-round distributions for the saved league.

Prints the pick-1 shortlist with probabilities, then the share of simulated picks in
each round that landed inside the consensus ADP order — the number the first two rounds
are supposed to hold high and the last ten are supposed to let go.

    python scripts/diagnose_early_rounds.py [--drafts 40]
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from engine.draft_state import DraftState
from engine.features import annotate_history
from engine.opponent_model import build_profiles
from engine.pick_model import context_for, pick_probabilities, score_candidates
from engine.simulator import DraftSimulator
from scripts.diagnose_picks import load_world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    league, pool, history = load_world()
    if history.drafts:
        annotate_history(history, pool=pool, roster=league.config.roster)
    profiles = build_profiles(league, history, pool=pool, annotate=False)

    state = DraftState(league, pool, seed=args.seed)
    sim = DraftSimulator(state, profiles)
    slot = state.current_slot
    profile = sim.profile_for(slot.draft_slot)
    context = context_for(state, profile, rng=sim.rng)
    ranked = pick_probabilities(
        score_candidates(context),
        state.settings.temperature_for(
            float(profile.get("predictability")), context.round_number
        ),
    )
    print(f"PICK 1 -- {profile.manager_name} "
          f"(predictability {profile.get('predictability'):.2f}, "
          f"temperature {state.settings.temperature_for(float(profile.get('predictability')), 1):.3f})")
    for candidate in ranked[:10]:
        player = candidate.player
        adp = player.adp_for()
        print(f"  p={candidate.probability:.3f} u={candidate.utility:+.2f} "
              f"{player.name:<20} {str(player.position):<3} "
              f"adp {adp if adp is None else round(adp, 1):>5} | {candidate.explain(4)}")

    top5 = sorted(
        (p for p in pool if p.adp_for() is not None), key=lambda p: p.adp_for()
    )[:5]
    ids = {p.player_id for p in top5}
    share = sum(c.probability for c in ranked if c.player.player_id in ids)
    print(f"  consensus ADP top 5 ({', '.join(p.name for p in top5)}): {share:.3f}")

    # Full drafts: how far from ADP does each round drift?
    by_round: dict[int, list[float]] = defaultdict(list)
    worst: dict[int, tuple[float, str]] = {}
    for run in range(args.drafts):
        state = DraftState(league, pool, seed=args.seed + run)
        sim = DraftSimulator(state, profiles)
        while state.current_slot is not None:
            made = sim.simulate_pick()
            if made is None:
                break
            player = pool.get(made.pick.player_id)
            adp = player.adp_for() if player else None
            if adp is None:
                continue
            rnd = int(made.pick.round_number)
            reach = float(adp) - float(made.pick.overall_pick)
            by_round[rnd].append(abs(reach))
            if reach > worst.get(rnd, (0.0, ""))[0]:
                worst[rnd] = (reach, f"{player.name} adp {adp:.1f} at "
                                     f"{made.pick.overall_pick}")

    print(f"\n{args.drafts} drafts -- |pick - ADP| by round")
    for rnd in sorted(by_round):
        values = by_round[rnd]
        reach, who = worst.get(rnd, (0.0, "-"))
        print(f"  r{rnd:<2} median {statistics.median(values):5.1f} "
              f"p90 {statistics.quantiles(values, n=10)[8] if len(values) > 9 else max(values):6.1f} "
              f"| worst reach {reach:5.1f} {who}")


if __name__ == "__main__":
    main()
