"""Drive a whole draft through the pick model and report what each manager did.

A manual harness, not a test: it prints positional signatures and unfilled
starting seats across several seeds so the model's behaviour can be eyeballed
against the plans the synthetic managers were designed with.
"""
from __future__ import annotations

import random
import sys
from collections import Counter

sys.path.insert(0, ".")

from core.config import SimulationConfig
from core.enums import Archetype
from engine.draft_state import DraftState
from engine.features import annotate_history
from engine.opponent_model import build_profiles
from engine.pick_model import choose_player, context_for
from models.league import League
from models.manager import Manager
from tests._smoke_pool import build as build_pool
from tests.conftest import PLANS, ROUNDS, SEASONS, TEAM_COUNT, _build_draft
from models.draft import DraftHistory


def run(seed: int) -> tuple[dict[str, int], dict[str, list[str]]]:
    config, pool = build_pool(TEAM_COUNT, ROUNDS)
    history = DraftHistory(drafts=[_build_draft(s) for s in SEASONS])
    annotate_history(history)
    managers = [
        Manager(name=name, draft_slot=slot, archetype=Archetype.BALANCED,
                is_user=False)
        for slot, name in enumerate(PLANS, start=1)
    ]
    league = League(config=config, managers=managers)
    settings = SimulationConfig()
    profiles = build_profiles(league, history, settings=settings, annotate=False)

    state = DraftState(league=league, pool=pool, settings=settings, seed=seed)
    rng = random.Random(seed)
    while not state.is_complete:
        slot = state.current_slot
        profile = profiles[slot.draft_slot]
        context = context_for(state, profile, rng=rng)
        chosen = choose_player(context)
        if chosen is None:
            break
        state.make_pick(chosen.player.player_id)

    open_starters: dict[str, int] = {}
    opening: dict[str, list[str]] = {}
    for manager in managers:
        roster = state.roster(manager.draft_slot)
        open_starters[manager.name] = sum(roster.open_starting_slots().values())
        picks = state.picks_by_slot(manager.draft_slot)
        opening[manager.name] = [p.position.value for p in picks]
    return open_starters, opening


def main() -> None:
    counts: Counter[tuple[str, str]] = Counter()
    for seed in (7, 11, 23, 42, 99):
        open_starters, opening = run(seed)
        short = {n[:3]: v for n, v in open_starters.items()}
        print(f"seed={seed:<3} open_starters={short}")
        for name, positions in opening.items():
            print(f"    {name:<16} first3={positions[:3]}  all={positions}")
            counts[(name, "".join(positions[:3]))] += 1
        print()
    print("first-three signatures across seeds:")
    for (name, sig), n in sorted(counts.items()):
        print(f"    {name:<16} {sig:<12} x{n}")


if __name__ == "__main__":
    main()
