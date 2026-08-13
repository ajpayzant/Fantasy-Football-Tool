"""How old the loaded data is, and whether that is old enough to say something.

Every part of the app that wants to talk about staleness comes here, for the same
reason every part that scores a stat line goes to :mod:`core.stats`: otherwise the
sidebar, the Setup page and the Draft Room each invent their own idea of "old" and
the user gets three different answers about one board.

What actually goes stale, in the order it bites:

* **Injury and roster status.** A player ruled out on Saturday is still healthy on a
  Friday board. This moves in hours during the season, and it is the change most
  likely to make a recommendation actively wrong rather than merely dated.
* **ADP.** Moves over days as the mock-draft pool turns over, faster after news.
  A board a week old will have a rookie two rounds off where the room has him.
* **The season itself.** A board from last season is not old data, it is the wrong
  data — retired players, no rookies, ADP for a field that no longer exists. So a
  season mismatch is reported as its own failure and not as an age at all.

Ages are deliberately coarse. The honest claim is "this is about a day old, ADP has
probably moved", not a number of minutes: the fetch timestamp is when the payload
was retrieved, and the provider computed it from drafts spread over days before
that. A precise-looking age would overstate what is known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from core.enums import StrEnum

# The boundaries. Chosen against what moves rather than round numbers:
#
# 12 hours matches the provider cache TTL, so anything inside it is as fresh as the
# app ever intends to be and saying "stale" would just be noise.
FRESH_HOURS = 12.0
# Two days is where ADP drift becomes visible at the round level and where a full
# injury-report cycle has certainly happened.
AGING_HOURS = 48.0
# A week is long enough that the board predates news the user has already read, and
# drafting off it is a mistake rather than a compromise.
STALE_HOURS = 24.0 * 7


class Freshness(StrEnum):
    """How much trust the age of the data has earned."""

    FRESH = "fresh"
    AGING = "aging"
    STALE = "stale"
    VERY_STALE = "very_stale"
    WRONG_SEASON = "wrong_season"
    UNKNOWN = "unknown"

    @property
    def is_concerning(self) -> bool:
        """Whether this warrants an interruption rather than a caption.

        ``UNKNOWN`` counts. Data that cannot say when it was fetched is not
        data known to be fresh, and the whole point of this module is to stop the
        app presenting an unlabelled board as a current one.
        """
        return self in {
            Freshness.STALE, Freshness.VERY_STALE,
            Freshness.WRONG_SEASON, Freshness.UNKNOWN,
        }


#: What a timestamp is the time *of*. A fetched board's timestamp is the age of the
#: data itself. An uploaded file's is only the age of the upload — nothing can see how
#: old the numbers in someone's spreadsheet are, and reporting an upload from an hour
#: ago as "fresh data" would be exactly the overstatement this module exists to remove.
FETCHED = "fetched"
IMPORTED = "imported"


@dataclass(slots=True)
class FreshnessVerdict:
    """A judgement about one dataset's age, ready to render.

    ``headline`` is a single sentence safe to put in a banner. ``advice`` names the
    action that fixes it, which is the part a warning is useless without.
    """

    level: Freshness = Freshness.UNKNOWN
    age_hours: float | None = None
    fetched_at: str = ""
    season: int | None = None
    expected_season: int | None = None
    basis: str = FETCHED
    reasons: list[str] = field(default_factory=list)

    @property
    def is_concerning(self) -> bool:
        return self.level.is_concerning

    def age_label(self) -> str:
        """The age in the coarsest unit that is still informative."""
        if self.age_hours is None:
            return "age unknown"
        hours = self.age_hours
        if hours < 0:
            # A fetch timestamp in the future means a clock disagreement, not
            # negative age. Reporting "-3 hours old" would look like a bug in the
            # data rather than in the clock, so it is named for what it is.
            return "timestamped in the future"
        if hours < 1:
            return f"{int(hours * 60)} min old"
        if hours < 36:
            return f"{hours:.0f} hours old"
        return f"{hours / 24:.1f} days old"

    @property
    def is_upload(self) -> bool:
        return self.basis == IMPORTED

    def headline(self) -> str:
        if self.level is Freshness.WRONG_SEASON:
            return (
                f"This board is for the {self.season} season, not "
                f"{self.expected_season}."
            )
        if self.level is Freshness.UNKNOWN:
            return "This board does not say when it was loaded."
        if self.is_upload:
            return f"This board was uploaded {self.age_label().replace(' old', ' ago')}."
        return f"This board is {self.age_label()}."

    def advice(self) -> str:
        if self.level is Freshness.WRONG_SEASON:
            return (
                "Fetch again for the current season on Setup — last season's board "
                "has no rookies, keeps retired players, and its ADP describes a "
                "field that no longer exists."
            )
        if self.level is Freshness.UNKNOWN:
            return (
                "Fetch on Setup to get a board with a timestamp on it. Until then "
                "nothing here can tell you whether the injury statuses are current."
            )
        if self.is_upload and self.level is not Freshness.FRESH:
            # Worth saying every time for an upload: the age shown is the age of the
            # upload, and a file uploaded ten minutes ago can hold numbers from
            # last August. Nothing in the file says, so nothing here pretends to.
            return (
                "That is how long ago the file was loaded, not how old the numbers "
                "in it are — nothing in an uploaded file says when it was produced. "
                "Re-upload a current export, or fetch live data on Setup."
            )
        if self.level is Freshness.VERY_STALE:
            return (
                "Fetch again on Setup before you draft off it. At this age it "
                "predates news you have probably already seen, and ADP will be a "
                "round or more out on the players who have moved most."
            )
        if self.level is Freshness.STALE:
            return (
                "Worth re-fetching on Setup. ADP has moved and at least one full "
                "injury-report cycle has happened since this was retrieved."
            )
        if self.level is Freshness.AGING:
            return (
                "Fine for planning. Re-fetch before a live draft — injury status is "
                "the field most likely to have changed."
            )
        if self.is_upload:
            return (
                "Just loaded. Note that nothing in an uploaded file says how old its "
                "own numbers are."
            )
        return "Current."

    def describe(self) -> str:
        """Headline and advice as one line, for a caption or a log."""
        return f"{self.headline()} {self.advice()}"


def parse_timestamp(raw: object) -> datetime | None:
    """Read an ISO-8601 timestamp, returning ``None`` for anything unreadable.

    Timestamps reach this from three places that can all disagree: a provider's
    ``fetched_at``, a ``PoolMetadata`` round-tripped through SQLite, and whatever a
    user's own file happened to contain. A bad one is treated as "no timestamp",
    which the caller reports as :attr:`Freshness.UNKNOWN` — never as fresh.

    A timestamp with no zone is read as UTC, because that is what everything in
    this app writes; guessing local time would shift the age by the user's offset.
    """
    if isinstance(raw, datetime):
        moment = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return None
        # ``fromisoformat`` handles a trailing Z from 3.11 on, so this is a fallback
        # for older interpreters — the same reason ``core.enums`` hand-rolls StrEnum.
        # The Z form is covered by a behaviour test either way, so removing this on a
        # newer interpreter would not fail anything; it is kept for the older one.
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def age_hours(fetched_at: object, *, now: datetime | None = None) -> float | None:
    """Hours between ``fetched_at`` and now, or ``None`` if it cannot be read."""
    moment = parse_timestamp(fetched_at)
    if moment is None:
        return None
    reference = now.astimezone(timezone.utc) if now is not None else datetime.now(
        timezone.utc
    )
    return (reference - moment).total_seconds() / 3600.0


def classify(hours: float | None) -> Freshness:
    """Turn an age in hours into a level. Kept separate so it can be tested alone."""
    if hours is None:
        return Freshness.UNKNOWN
    # A clock skew of a few minutes is ordinary and should not be reported at all;
    # a timestamp genuinely in the future is a dataset that cannot be trusted to
    # date itself, which is exactly what UNKNOWN means.
    if hours < -0.25:
        return Freshness.UNKNOWN
    if hours <= FRESH_HOURS:
        return Freshness.FRESH
    if hours <= AGING_HOURS:
        return Freshness.AGING
    if hours <= STALE_HOURS:
        return Freshness.STALE
    return Freshness.VERY_STALE


def assess(
    fetched_at: object,
    *,
    season: int | None = None,
    expected_season: int | None = None,
    basis: str = FETCHED,
    now: datetime | None = None,
) -> FreshnessVerdict:
    """Judge one dataset's age, and whether it is even the right season.

    ``expected_season`` is the season the user is drafting. When it is supplied and
    differs from ``season``, that decides the verdict on its own: a board can be
    ten minutes old and still be last year's, and the age is the less important
    fact about it by a wide margin.
    """
    hours = age_hours(fetched_at, now=now)
    moment = parse_timestamp(fetched_at)
    verdict = FreshnessVerdict(
        level=classify(hours),
        age_hours=hours,
        fetched_at=moment.isoformat() if moment is not None else "",
        season=season,
        expected_season=expected_season,
        basis=basis,
    )
    if (
        season is not None
        and expected_season is not None
        and int(season) != int(expected_season)
    ):
        verdict.level = Freshness.WRONG_SEASON
        verdict.reasons.append(
            f"season {season} data loaded while drafting {expected_season}"
        )
        return verdict
    if verdict.level is Freshness.UNKNOWN:
        verdict.reasons.append("no readable fetch timestamp")
    elif verdict.level is not Freshness.FRESH:
        verdict.reasons.append(f"fetched {verdict.age_label()}")
    return verdict


def worst(verdicts: object) -> FreshnessVerdict:
    """The most concerning of several verdicts, for a board built from many sources.

    A board is only as current as its stalest contributing source: averaging the
    ages would let three fresh sources hide one that failed and fell back to a
    week-old cache, which is precisely the case worth reporting.
    """
    ordering = {
        Freshness.FRESH: 0,
        Freshness.AGING: 1,
        Freshness.UNKNOWN: 2,
        Freshness.STALE: 3,
        Freshness.VERY_STALE: 4,
        Freshness.WRONG_SEASON: 5,
    }
    items = [v for v in (verdicts or []) if isinstance(v, FreshnessVerdict)]
    if not items:
        return FreshnessVerdict()
    return max(items, key=lambda v: (ordering.get(v.level, 2), v.age_hours or 0.0))


def stale_after(hours: float | None = None) -> timedelta:
    """The window past which data is treated as stale, as a timedelta."""
    return timedelta(hours=STALE_HOURS if hours is None else hours)


__all__ = [
    "FRESH_HOURS",
    "AGING_HOURS",
    "STALE_HOURS",
    "FETCHED",
    "IMPORTED",
    "Freshness",
    "FreshnessVerdict",
    "parse_timestamp",
    "age_hours",
    "classify",
    "assess",
    "worst",
    "stale_after",
]
