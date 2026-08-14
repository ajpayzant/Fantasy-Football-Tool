"""Why did the AI take *that*? A reach report over a full simulated draft.

Loads the saved league, board and history out of the app's own database, runs whole
drafts, and lists the picks that came furthest ahead of ADP with the utility terms that
drove them. Run it, read the top of the table, and the cause is either one term dominating
every reach or it is the softmax being too flat.

    python scripts/diagnose_picks.py [--drafts 5] [--seed 7]
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter

from engine.draft_state import DraftState
from engine.features import annotate_history
from engine.opponent_model import build_profiles
from engine.pick_model import context_for, pick_probabilities, score_candidates
from engine.simulator import DraftSimulator
from models.database import session_scope
from services import repository


def load_world():
    with session_scope() as session:
        leagues = repository.list_leagues(session)
        if not leagues:
            raise SystemExit("no league saved; connect one on the Setup page first")
        league_id = leagues[0]["league_id"]
        league = repository.load_league(session, league_id)
        history = repository.load_history(session, league_id)
        sources = repository.list_player_sources(session)
        if not sources:
            raise SystemExit("no player board saved; import one on the Player pool page")
        pool = repository.load_player_pool(
            session, sources[0]["source_id"], league.config
        )
    return league, pool, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    league, pool, history = load_world()
    print(f"league: {league.config.name} | {league.config.team_count} teams "
          f"| {league.config.rounds} rounds | pool {len(pool)}")
    print(f"history: {len(history.drafts)} draft(s), {len(history.all_picks)} picks")
    # Mirror the Manager profiles page: annotate against the board, then build.
    # Passing the pool is what lets the season check in ``_backfill_from_pool`` run,
    # so a diagnosis without it measures a model the app never uses.
    if history.drafts:
        annotate_history(history, pool=pool, roster=league.config.roster)
    profiles = build_profiles(league, history, pool=pool, annotate=False)
    for slot, profile in sorted(profiles.items())[:3]:
        print(f"  slot {slot} {profile.manager_name}: "
              f"picks={profile.sample_picks:.1f} seasons={sorted(profile.seasons_seen)} "
              f"predictability={profile.get('predictability'):.2f} "
              f"weight(pred)={profile.values['predictability'].manager_weight:.2f}")

    reaches: list[tuple[float, str]] = []
    fallers: list[tuple[float, str]] = []
    term_blame: Counter[str] = Counter()
    top_prob: list[float] = []
    spreads: list[tuple[float, float]] = []
    temperatures: list[float] = []

    for run in range(args.drafts):
        state = DraftState(league, pool, seed=args.seed + run)
        sim = DraftSimulator(state, profiles)
        while state.current_slot is not None:
            slot = state.current_slot
            profile = sim.profile_for(slot.draft_slot)
            context = context_for(state, profile, rng=sim.rng)
            ranked = pick_probabilities(
                score_candidates(context),
                state.settings.temperature_for(float(profile.get("predictability"))),
            )
            top_prob.append(ranked[0].probability if ranked else 0.0)
            if ranked:
                utilities = [c.utility for c in ranked]
                best = max(utilities)
                spreads.append((best - statistics.median(utilities),
                                best - min(utilities)))
                temperatures.append(
                    state.settings.temperature_for(
                        float(profile.get("predictability"))
                    )
                )
            made = sim.simulate_pick()
            if made is None:
                break
            player = pool.get(made.pick.player_id)
            adp = player.adp_for() if player else None
            if adp is None:
                continue
            delta = float(adp) - float(slot.overall_pick)   # + = reached past ADP
            chosen = next(
                (c for c in ranked if c.player_id == made.pick.player_id), None
            )
            terms = chosen.explain(4) if chosen else ""
            prob = chosen.probability if chosen else 0.0
            board_rank = 1 + sum(
                1 for c in ranked
                if (c.player.adp_for() or 9e9) < float(adp)
            )
            line = (f"r{run} pick {slot.overall_pick:>3} "
                    f"{player.name:<22} {player.position!s:<3} adp {float(adp):>5.1f} "
                    f"delta {delta:>+6.1f} p={prob:.2f} best-avail-ahead={board_rank - 1:>2} "
                    f"| {terms}")
            if delta > 0:
                reaches.append((delta, line))
                if chosen:
                    for name, value in chosen.top_reasons(2):
                        if value > 0:
                            term_blame[name] += 1
            else:
                fallers.append((-delta, line))

    print(f"\ntop-utility player's probability: mean {statistics.mean(top_prob):.3f} "
          f"median {statistics.median(top_prob):.3f}")
    print(f"utility spread best-median {statistics.median(s for s, _ in spreads):.3f} "
          f"| best-worst {statistics.median(w for _, w in spreads):.3f} "
          f"| temperature {statistics.median(temperatures):.2f}")
    print(f"\n=== biggest reaches (drafted N picks before ADP) ===")
    for delta, line in sorted(reaches, reverse=True)[: args.top]:
        print(line)
    print(f"\n=== terms most often driving a reach ===")
    for name, count in term_blame.most_common(8):
        print(f"  {name:<32} {count}")
    print(f"\n=== biggest fallers (still on the board long past ADP) ===")
    for delta, line in sorted(fallers, reverse=True)[:10]:
        print(line)


if __name__ == "__main__":
    main()
