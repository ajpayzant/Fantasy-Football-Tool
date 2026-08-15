"""Build a small ADP-realistic pool for smoke tests."""
from core.config import LeagueConfig, RosterSettings
from core.enums import Platform, Position, Slot
from models.player import Player, PlayerPool, PoolMetadata

# A plausible positional shape for the first ~120 picks of a real draft:
# RB/WR dominate early, QBs trickle in from the 3rd round, K/DST at the end.
SHAPE = (
    [Position.RB, Position.WR, Position.WR, Position.RB] * 3          # picks 1-12
    + [Position.WR, Position.RB, Position.TE, Position.WR] * 2        # 13-20
    + [Position.QB, Position.RB, Position.WR, Position.TE]            # 21-24
    + [Position.WR, Position.RB, Position.QB, Position.WR] * 3        # 25-36
    + [Position.TE, Position.RB, Position.WR, Position.QB] * 3        # 37-48
    + [Position.WR, Position.RB, Position.TE, Position.QB] * 4        # 49-64
    + [Position.RB, Position.WR, Position.QB, Position.TE] * 4        # 65-80
    + [Position.K, Position.DST] * 6                                  # 81-92
    + [Position.WR, Position.RB, Position.K, Position.DST] * 5        # 93-112
)

def build(team_count: int = 4, rounds: int = 10) -> tuple[LeagueConfig, PlayerPool]:
    roster = RosterSettings(slots={
        Slot.QB: 1, Slot.RB: 2, Slot.WR: 2, Slot.TE: 1,
        Slot.FLEX: 1, Slot.K: 1, Slot.DST: 1, Slot.BENCH: 1,
    })
    config = LeagueConfig(
        name="Smoke League", season=2026, platform=Platform.ESPN,
        team_count=team_count, rounds=rounds, roster=roster,
    )
    teams = ("KC", "SF", "BUF", "GB", "MIN", "DAL", "PHI", "BAL")
    per: dict[Position, int] = {}
    players = []
    # ADP has to be expressed in *this* league's pick numbers. SHAPE describes the
    # order players come off the board, so its index is stretched onto the draft's
    # real length: in a 40-pick draft the kicker cluster two-thirds down the shape
    # belongs near pick 27, not pick 81. Left unscaled, every kicker looks like a
    # 40-pick reach at every pick in the draft and no roster ever fills its K seat.
    total_picks = max(1, team_count * rounds)
    scale = float(total_picks) / float(len(SHAPE))
    for index, position in enumerate(SHAPE, start=1):
        per[position] = per.get(position, 0) + 1
        adp = max(1.0, round(index * scale, 1))
        players.append(Player(
            player_id=f"{position.value}{per[position]}",
            name=f"{position.value} Player {per[position]}",
            position=position, nfl_team=teams[index % len(teams)],
            overall_adp=adp, platform_adp=adp,
            overall_rank=float(index), platform_rank=float(index),
            projection=max(1.0, 320.0 - index * 1.6),
        ))
    pool = PlayerPool(players, league=config,
                      metadata=PoolMetadata(source="smoke test", is_sample_data=True))
    return config, pool
