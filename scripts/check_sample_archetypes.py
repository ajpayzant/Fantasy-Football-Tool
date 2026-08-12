"""Does the opponent model recover the archetype each sample manager was built as?

The sample history is generated from known plans (``tests.fixtures.sample_league.drafts``), so
this is the one check that says whether the demo actually demonstrates anything:
it runs the real importer, the real feature annotation and the real estimator, and
compares the inferred label against the designed one.

Run: ``PYTHONIOENCODING=utf-8 python -m scripts.check_sample_archetypes``
"""

from __future__ import annotations

import sys

from core.enums import Position
from tests.fixtures.sample_league.drafts import sample_history_frame
from tests.fixtures.sample_league.league import MANAGER_ARCHETYPES, sample_league
from tests.fixtures.sample_league.players import sample_player_frame
from engine.features import annotate_history
from engine.opponent_model import build_profiles, observe_manager
from services.importers import import_historical_drafts, import_player_pool


def main() -> int:
    league = sample_league()
    pool = import_player_pool(sample_player_frame()).pool
    history = import_historical_drafts(sample_history_frame()).history
    annotate_history(history, pool=pool, roster=league.config.roster)
    profiles = build_profiles(league, history, pool=pool, annotate=False)

    hits = 0
    for manager, designed in zip(league.managers, MANAGER_ARCHETYPES):
        profile = profiles.get(manager.draft_slot)
        inferred = profile.archetype if profile else None
        ok = inferred is designed
        hits += ok
        obs = observe_manager(manager.name, history)
        erb = obs.early_position_share.get(Position.RB, 0.0)
        team = max(obs.team_share.values()) if obs.team_share else 0.0
        qb = obs.first_round_by_position.get(Position.QB)
        te = obs.first_round_by_position.get(Position.TE)
        print(
            f"{manager.name:<18} {str(designed):<22} -> {str(inferred):<22} "
            f"inv {obs.rank_inversions.mean:6.2f} eRB {erb:.2f} "
            f"rook {obs.rookie_rate.mean:.2f} tm {team:.2f} "
            f"qb {qb.mean if qb else -1:5.2f} te {te.mean if te else -1:5.2f} "
            f"{'OK' if ok else 'x'}"
        )
    print(f"\n{hits}/{len(league.managers)} recovered")
    return 0 if hits == len(league.managers) else 1


if __name__ == "__main__":
    sys.exit(main())
