"""Reusable UI pieces shared across pages.

Anything that appears on more than one page lives here, so the seven pages stay
short enough to read and a change to (say) how a player row is formatted happens
once. Every function renders and returns nothing, except the few that return the
user's choice.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

from core.enums import Archetype, Position, RiskBand
from models.player import Player, PlayerPool

from . import state

POSITION_COLOURS: dict[str, str] = {
    "QB": "#e05263", "RB": "#3fa45b", "WR": "#3d7ecf",
    "TE": "#e08a3c", "K": "#8a7fb5", "DST": "#6b7280",
}
"""One colour per position, used consistently on every chart and badge."""

RISK_ICONS: dict[str, str] = {
    "safe": "🟢", "likely": "🟡", "coin_flip": "🟠", "unlikely": "🔴", "gone": "⚫",
}


K_FLASH = "_flash_messages"


def flash(message: str, kind: str = "success") -> None:
    """Queue a message to show after the next rerun.

    Needed because ``st.rerun`` discards everything already rendered: a page that
    loads data has to rerun so the sidebar reflects it, which would otherwise throw
    away the confirmation of what was just loaded.
    """
    st.session_state.setdefault(K_FLASH, []).append((kind, message))


def render_flashes() -> None:
    """Show and clear anything queued by :func:`flash`."""
    for kind, message in st.session_state.pop(K_FLASH, []):
        {"success": st.success, "info": st.info,
         "warning": st.warning, "error": st.error}.get(kind, st.info)(message)


def page_header(title: str, subtitle: str = "") -> None:
    """Title, optional subtitle, the sample-data banner, and any queued messages."""
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    sample_banner()
    render_flashes()


def sample_banner() -> None:
    """The unmissable notice that the loaded data is not real.

    Rendered at the top of every page rather than once at startup, because a user
    who lands mid-session on the Draft Room has not seen the startup notice, and
    "do not present sample data as current real-world data" has to hold on the page
    they are actually looking at.

    No route through the app loads fictional data any more — the synthetic league
    now lives in ``tests/fixtures/sample_league`` and is reachable only from the
    test suite. This banner stays because the flag it reads is also set by any pool
    loaded from the database that was saved as sample data in an earlier session,
    and because a pool that *claims* to be synthetic must always say so.
    """
    if not state.is_sample_data():
        return
    st.warning(
        "⚠️ SAMPLE DATA — this pool is flagged as fictional and is not real NFL "
        "data. Fetch current data on **Setup** to replace it.",
        icon="⚠️",
    )


def blocked(reason: str) -> None:
    """Explain what is missing and stop rendering the rest of the page."""
    st.info(reason)
    st.stop()


def require(*, needs_draft: bool = False) -> None:
    """Gate a page on the data it needs, naming the page that supplies it."""
    reason = state.blocking_reason(needs_draft=needs_draft)
    if reason:
        blocked(reason)


def sidebar_status() -> None:
    """The persistent left-hand panel: what is loaded, and where it came from."""
    ready = state.readiness()
    with st.sidebar:
        st.subheader("Session")
        if state.is_sample_data():
            st.error("SAMPLE DATA loaded — fictional players", icon="⚠️")

        labels = {
            "league": "League", "players": "Player pool", "history": "Draft history",
            "profiles": "Manager profiles", "draft": "Draft in progress",
        }
        for key, label in labels.items():
            st.write(f"{'✅' if ready[key] else '⬜'} {label}")

        league = state.league()
        if league is not None:
            st.divider()
            st.caption(
                f"**{league.config.name}** · {league.config.team_count} teams · "
                f"{league.config.rounds} rounds · {league.config.scoring.preset}"
            )
        pool = state.pool()
        if pool is not None:
            st.caption(pool.metadata.describe())

        history = state.history()
        if history.drafts:
            seasons = ", ".join(str(d.season) for d in history.drafts)
            st.caption(f"History: {len(history.all_picks)} picks ({seasons})")

        draft = state.draft()
        if draft is not None:
            st.divider()
            st.caption(draft.summary_line())


# ─────────────────────────────────────────────────────────────────────────────
# Player rendering
# ─────────────────────────────────────────────────────────────────────────────
def position_badge(position: Position | str) -> str:
    """Coloured position tag as inline HTML."""
    text = str(position).upper()
    colour = POSITION_COLOURS.get(text, "#6b7280")
    return (
        f"<span style='background:{colour};color:white;padding:1px 6px;"
        f"border-radius:4px;font-size:0.75rem;font-weight:600'>{text}</span>"
    )


def player_line(player: Player, *, extra: str = "") -> str:
    """``Name · POS · TEAM`` plus whatever the caller wants appended."""
    bits = [f"**{player.name}**", str(player.position), player.nfl_team or "FA"]
    if player.bye_week:
        bits.append(f"bye {player.bye_week}")
    line = " · ".join(bits)
    return f"{line} — {extra}" if extra else line


def player_frame(
    players: Sequence[Player],
    *,
    pool: PlayerPool | None = None,
    survival: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Players as a display table, with optional survival odds joined on.

    Deliberately a small, fixed set of columns. The full player record has 30+
    fields and dumping all of them produces a table nobody reads; the ones here
    are the ones a draft decision actually turns on.
    """
    rows = []
    for player in players:
        row = {
            "Player": player.name,
            "Pos": str(player.position),
            "Team": player.nfl_team or "FA",
            "ADP": player.overall_adp,
            "Rank": player.platform_rank or player.overall_rank,
            "Proj": player.projection,
            "Tier": player.tier,
            "Bye": player.bye_week,
        }
        if pool is not None:
            row["VOR"] = player.value_over_replacement
        if survival is not None:
            row["Survives"] = survival.get(player.player_id)
        rows.append(row)
    return pd.DataFrame(rows)


def risk_label(band: RiskBand | str) -> str:
    """Icon plus words, e.g. ``🟠 Coin Flip``."""
    key = str(band)
    return f"{RISK_ICONS.get(key, '⚪')} {key.replace('_', ' ').title()}"


def survival_bar(survival: float) -> str:
    """A ten-cell text bar. Reads at a glance in a table cell, unlike a number."""
    filled = int(round(max(0.0, min(1.0, survival)) * 10))
    return "█" * filled + "░" * (10 - filled) + f" {survival:.0%}"


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
def position_bar_chart(counts: Mapping[Position | str, float], title: str) -> None:
    """Horizontal bar chart of a per-position quantity, position-coloured."""
    if not counts:
        st.caption("No data yet.")
        return
    import plotly.express as express

    frame = pd.DataFrame(
        {"Position": [str(k) for k in counts], "Value": list(counts.values())}
    ).sort_values("Value", ascending=True)
    figure = express.bar(
        frame, x="Value", y="Position", orientation="h", title=title,
        color="Position", color_discrete_map=POSITION_COLOURS,
    )
    figure.update_layout(showlegend=False, height=260, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(figure, width="stretch")


def histogram(values: Sequence[float], title: str, x_label: str) -> None:
    """Distribution of a simulated quantity, with the mean marked.

    The mean line is the point: a Monte Carlo result shown as a bare number
    invites reading it as a prediction, and the spread is the actual finding.
    """
    if not values:
        st.caption("No simulations yet.")
        return
    import plotly.express as express

    figure = express.histogram(
        pd.DataFrame({x_label: list(values)}), x=x_label, nbins=30, title=title
    )
    mean = sum(values) / len(values)
    figure.add_vline(
        x=mean, line_dash="dash", line_color="#e05263",
        annotation_text=f"mean {mean:.1f}",
    )
    figure.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
    st.plotly_chart(figure, width="stretch")


def archetype_caption(archetype: Archetype) -> str:
    """One plain sentence per archetype, so a label is never bare jargon."""
    text = {
        Archetype.ZERO_RB: "Loads up on receivers early and waits on running backs.",
        Archetype.ROBUST_RB: "Takes running backs early and often.",
        Archetype.HERO_RB: "One early back, then receivers.",
        Archetype.EARLY_QB: "Takes a quarterback well before the room does.",
        Archetype.LATE_QB: "Waits on quarterback, often taking two late.",
        Archetype.ELITE_TE: "Pays up for a tight end in the early rounds.",
        Archetype.ROOKIE_HEAVY: "Reaches for rookies ahead of their ADP.",
        Archetype.HOMER: "Over-drafts one NFL team's players.",
        Archetype.HIGH_VARIANCE: "Unpredictable — reaches a long way, inconsistently.",
        Archetype.AUTODRAFT: "Takes whoever the ranking list says is next.",
        Archetype.RANK_FOLLOWER: "Broadly follows the rankings, with small deviations.",
        Archetype.BALANCED: "No strong signature; drafts close to the room's average.",
        Archetype.BEST_PLAYER_AVAILABLE: "Takes the best player left, ignoring roster fit.",
        Archetype.CUSTOM: "Described by you rather than inferred from history.",
    }
    return text.get(archetype, "")


def metric_row(items: Iterable[tuple[str, object, str]]) -> None:
    """A row of ``st.metric`` cells from ``(label, value, help)`` triples."""
    entries = list(items)
    if not entries:
        return
    for column, (label, value, helper) in zip(st.columns(len(entries)), entries):
        column.metric(label, value, help=helper or None)


def download_frame(frame: pd.DataFrame, label: str, file_name: str) -> None:
    """CSV export button. Every table the app builds is exportable."""
    if frame.empty:
        return
    st.download_button(
        label, frame.to_csv(index=False).encode("utf-8"),
        file_name=file_name, mime="text/csv",
    )


__all__ = [
    "POSITION_COLOURS", "RISK_ICONS", "flash", "render_flashes",
    "page_header", "sample_banner", "blocked",
    "require", "sidebar_status", "position_badge", "player_line", "player_frame",
    "risk_label", "survival_bar", "position_bar_chart", "histogram",
    "archetype_caption", "metric_row", "download_frame",
]
