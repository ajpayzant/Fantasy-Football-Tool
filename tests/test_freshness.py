"""Whether the app can tell how old its data is, and says so when it matters.

The failure this file exists to prevent is not a crash. It is a board that looks
current, is three days old, and gets drafted off — and the reason that was possible
is that every timestamp in the pipeline was easy to set to "now":

* A cache hit stamped itself with the time of the *call*. Serving a three-day-old
  cached copy is the normal, deliberate behaviour when a source is unreachable, so
  the one case where staleness mattered most was the one case guaranteed to report
  itself as fresh.
* Saving a pool to the database stamped the row with the time of the *save*, so a
  board fetched last week and saved today came back tomorrow looking brand new.
* A board built from four sources took the first timestamp that answered, letting
  one live source date a board mostly assembled from expired cache.

Each of those has a test here. The rest pin the judgement itself: thresholds, the
distinction between a fetched board's age and an uploaded file's, and the rule that
data which cannot say when it was loaded is never reported as fresh.
"""

from __future__ import annotations

import gzip
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core import freshness
from core.freshness import Freshness
from core.validation import ValidationReport
from services.providers import base as provider_base
from services.providers.base import ProviderResult

NOW = datetime(2026, 8, 12, 18, 0, 0, tzinfo=timezone.utc)


def _ago(hours: float) -> str:
    return (NOW - timedelta(hours=hours)).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# The judgement itself
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("hours", "expected"),
    [
        (0.0, Freshness.FRESH),
        (11.9, Freshness.FRESH),
        (12.0, Freshness.FRESH),        # the cache TTL is inside "fresh" by design
        (12.1, Freshness.AGING),
        (47.9, Freshness.AGING),
        (48.1, Freshness.STALE),
        (24 * 7 - 1, Freshness.STALE),
        (24 * 7 + 1, Freshness.VERY_STALE),
        (24 * 40, Freshness.VERY_STALE),
    ],
)
def test_the_thresholds_are_where_the_data_actually_changes(
    hours: float, expected: Freshness
) -> None:
    """Boundaries pinned, because they are the whole content of the warning.

    Not arbitrary: 12 hours is the provider cache TTL, so anything inside it is as
    fresh as the app ever intends to be; 48 hours is a full injury-report cycle and
    visible ADP drift; a week predates news the user has already read.
    """
    assert freshness.classify(hours) is expected


def test_no_timestamp_is_never_reported_as_fresh() -> None:
    """The default has to be "unknown", not "fine".

    A pool with no timestamp is the case an omission produces, so if that read as
    fresh then every future code path that forgot to stamp a board would silently
    present it as current.
    """
    for missing in ("", None, "not a date", "2026-13-45", float("nan")):
        assert freshness.assess(missing, now=NOW).level is Freshness.UNKNOWN


def test_unknown_age_is_treated_as_a_problem_and_aging_is_not() -> None:
    """What interrupts the user, and what only gets a caption.

    Both halves matter. Warning on a 20-hour-old board trains the user to dismiss
    the banner; staying quiet about a board with no timestamp at all defeats it.
    """
    assert freshness.assess("", now=NOW).is_concerning
    assert not freshness.assess(_ago(20), now=NOW).is_concerning
    assert freshness.assess(_ago(20), now=NOW).level is Freshness.AGING
    assert freshness.assess(_ago(80), now=NOW).is_concerning


def test_a_timestamp_in_the_future_is_unknown_not_negative_age() -> None:
    """A clock disagreement must not read as impossibly fresh data.

    A small skew is ordinary and tolerated; a timestamp genuinely ahead of now means
    the data cannot date itself, which is what UNKNOWN means.
    """
    slight = freshness.assess(_ago(-0.1), now=NOW)
    assert slight.level is Freshness.FRESH

    ahead = freshness.assess(_ago(-48), now=NOW)
    assert ahead.level is Freshness.UNKNOWN
    assert "future" in ahead.age_label()


def test_timestamps_are_read_in_the_formats_this_app_writes() -> None:
    """Three writers, three spellings, one meaning.

    ``fetch_bytes`` writes an offset-aware ISO string, SQLite hands back a naive one,
    and a Z suffix arrives from anything that touched JSON. A naive timestamp is read
    as UTC, not local time — guessing local would shift every age by the user's
    offset and quietly re-band boards near a threshold.
    """
    aware = freshness.parse_timestamp("2026-08-12T06:00:00+00:00")
    naive = freshness.parse_timestamp("2026-08-12T06:00:00")
    zulu = freshness.parse_timestamp("2026-08-12T06:00:00Z")
    assert aware == naive == zulu

    assert freshness.age_hours("2026-08-12T06:00:00", now=NOW) == pytest.approx(12.0)


def test_last_seasons_board_is_wrong_rather_than_old() -> None:
    """Season beats age, and reports itself differently.

    A board fetched two minutes ago for last season is not stale data, it is data
    about a different set of players: no rookies, retired players still listed, ADP
    for a field that no longer exists. Reporting it as "2 minutes old" would be true
    and useless.
    """
    verdict = freshness.assess(
        _ago(0.03), season=2025, expected_season=2026, now=NOW
    )
    assert verdict.level is Freshness.WRONG_SEASON
    assert verdict.is_concerning
    assert "2025" in verdict.headline() and "2026" in verdict.headline()
    assert "rookies" in verdict.advice()


def test_a_matching_season_does_not_trip_the_season_check() -> None:
    verdict = freshness.assess(_ago(1), season=2026, expected_season=2026, now=NOW)
    assert verdict.level is Freshness.FRESH


def test_an_unknown_season_is_not_guessed_at() -> None:
    """No expected season means no season verdict — silence, not a guess.

    An uploaded file often has no season column, and inferring one from the wall
    clock would flag a perfectly good board as the wrong year every spring, when the
    fantasy season has not rolled over yet.
    """
    assert freshness.assess(_ago(1), season=None, expected_season=2026, now=NOW).level \
        is Freshness.FRESH
    assert freshness.assess(_ago(1), season=2025, expected_season=None, now=NOW).level \
        is Freshness.FRESH


def test_every_level_names_something_the_user_can_do() -> None:
    """A staleness warning with no action attached is just an interruption."""
    seen = set()
    for hours in (1.0, 20.0, 80.0, 24 * 30):
        verdict = freshness.assess(_ago(hours), now=NOW)
        seen.add(verdict.level)
        assert verdict.advice().strip(), verdict.level
        assert verdict.headline().strip()
    assert seen == {
        Freshness.FRESH, Freshness.AGING, Freshness.STALE, Freshness.VERY_STALE
    }, "the loop above has to actually reach every level it claims to cover"

    # The two that can only be reached another way, and the only two where the fix is
    # not obvious from the headline alone.
    wrong = freshness.assess(_ago(1), season=2020, expected_season=2026, now=NOW)
    assert "Setup" in wrong.advice()
    assert "Setup" in freshness.assess("", now=NOW).advice()


def test_the_age_is_stated_in_a_unit_that_does_not_overstate_it() -> None:
    """Coarse on purpose.

    A fetch timestamp is when the payload was retrieved; the provider computed the
    ADP in it from drafts spread over days before that. "13 hours old" is honest;
    "12h 47m 3s old" claims a precision the number does not have.
    """
    assert freshness.assess(_ago(0.5), now=NOW).age_label() == "30 min old"
    assert freshness.assess(_ago(13), now=NOW).age_label() == "13 hours old"
    assert freshness.assess(_ago(72), now=NOW).age_label() == "3.0 days old"


def test_an_uploads_age_is_the_age_of_the_upload_not_of_its_contents() -> None:
    """The one thing that cannot be measured, said out loud.

    A spreadsheet handed over a minute ago can hold projections from last August, and
    no column in it says so. Reporting an upload the same way as a fetch would put a
    reassuring "0 min old" on numbers of completely unknown vintage.
    """
    fetched = freshness.assess(_ago(60), basis=freshness.FETCHED, now=NOW)
    uploaded = freshness.assess(_ago(60), basis=freshness.IMPORTED, now=NOW)

    assert fetched.level is uploaded.level is Freshness.STALE
    assert "uploaded" in uploaded.headline()
    assert "not how old the numbers in it are" in uploaded.advice()
    assert "not how old" not in fetched.advice()

    just_now = freshness.assess(_ago(0.01), basis=freshness.IMPORTED, now=NOW)
    assert just_now.level is Freshness.FRESH
    assert not just_now.is_concerning     # a file loaded seconds ago is not a warning
    assert "nothing in an uploaded file says" in just_now.advice()


def test_a_board_is_as_old_as_its_oldest_source() -> None:
    """``worst`` picks the stalest, because averaging hides the case worth reporting.

    Three live sources and one that fell back to a week-old cache is a board with a
    week-old column on it. A mean age would report that as fresh.
    """
    verdicts = [
        freshness.assess(_ago(1), now=NOW),
        freshness.assess(_ago(2), now=NOW),
        freshness.assess(_ago(24 * 9), now=NOW),
    ]
    assert freshness.worst(verdicts).level is Freshness.VERY_STALE
    assert freshness.worst([]).level is Freshness.UNKNOWN
    # A season mismatch outranks any age.
    verdicts.append(freshness.assess(_ago(0.1), season=2020, expected_season=2026, now=NOW))
    assert freshness.worst(verdicts).level is Freshness.WRONG_SEASON


# ─────────────────────────────────────────────────────────────────────────────
# The fetch layer: a cached read must date itself from the cache
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def cache_dir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> str:
    directory = tmp_path / "cache"
    directory.mkdir()
    monkeypatch.setattr(provider_base, "cache_directory", lambda: str(directory))
    return str(directory)


def _seed_cache(key: str, payload: bytes, *, age_hours: float) -> str:
    path = provider_base._cache_path(key)
    with gzip.open(path, "wb") as handle:
        handle.write(payload)
    stamp = time.time() - age_hours * 3600
    os.utime(path, (stamp, stamp))
    return path


@pytest.fixture
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    def _blocked(*args, **kwargs):
        raise AssertionError("this test must not reach the network")

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)


def test_a_cache_hit_is_dated_from_the_cache_not_from_the_call(
    cache_dir: str, _no_network: None
) -> None:
    """The bug this whole feature rests on.

    ``fetched_at`` used to be ``now()`` on every cache hit, so the timestamp that
    every staleness check reads was the one value guaranteed to say "just now". A
    board served from an eight-hour-old cache reported itself as freshly fetched.
    """
    _seed_cache("aged_key", b'{"ok": true}', age_hours=8.0)

    outcome = provider_base.fetch_bytes(
        "https://example.invalid/data.json", cache_key="aged_key", ttl_seconds=-1
    )

    assert outcome.ok and outcome.from_cache
    assert freshness.age_hours(outcome.fetched_at) == pytest.approx(8.0, abs=0.05)
    assert outcome.cache_age_seconds == pytest.approx(8 * 3600, abs=60)


def test_an_unreachable_source_serving_expired_cache_says_so(
    cache_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Falling back to stale data is right. Doing it silently is not.

    ``ok`` is True on this path, so every provider treats it as a success. Before
    ``stale_fallback`` existed the only trace was an ``error`` string that nothing
    reads on the success path, which meant a week-old board and no notice.
    """
    _seed_cache("expired_key", b'{"ok": true}', age_hours=72.0)

    import urllib.request

    def _fail(*args, **kwargs):
        raise TimeoutError("simulated outage")

    monkeypatch.setattr(urllib.request, "urlopen", _fail)

    outcome = provider_base.fetch_bytes(
        "https://example.invalid/data.json",
        cache_key="expired_key",
        ttl_seconds=3600,      # the cache is well past this
        retries=1,
    )

    assert outcome.ok, "stale data beats no data — the fallback itself is correct"
    assert outcome.stale_fallback
    assert freshness.age_hours(outcome.fetched_at) == pytest.approx(72.0, abs=0.1)
    assert "stale" in outcome.error


def test_a_live_fetch_is_dated_now(cache_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The other half: a genuine network read is not accidentally back-dated."""

    class _Response:
        headers: dict[str, str] = {}

        def read(self) -> bytes:
            return b'{"ok": true}'

        def __enter__(self):
            return self

        def __exit__(self, *args) -> bool:
            return False

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Response())

    outcome = provider_base.fetch_bytes(
        "https://example.invalid/data.json", cache_key="live_key", force_refresh=True
    )

    assert outcome.ok and not outcome.from_cache and not outcome.stale_fallback
    assert freshness.age_hours(outcome.fetched_at) == pytest.approx(0.0, abs=0.05)


def test_an_expired_cache_reads_differently_from_a_deliberate_one() -> None:
    """Two cache hits, two meanings, and the label has to tell them apart.

    "cached 40 minutes ago" is the app choosing not to re-request. "expired cache,
    72 hours old" is the app unable to reach the source. Only the second is a
    problem, and collapsing both into "cached" hides it.
    """
    ordinary = ProviderResult(
        pd.DataFrame([{"a": 1}]), "Test", fetched_at="2026-08-12T17:20:00+00:00",
        from_cache=True, cache_age_seconds=40 * 60,
    )
    expired = ProviderResult(
        pd.DataFrame([{"a": 1}]), "Test", fetched_at="2026-08-09T18:00:00+00:00",
        from_cache=True, cache_age_seconds=72 * 3600, stale_fallback=True,
    )

    assert "cached 40 min ago" == ordinary.freshness_label()
    assert "could not be reached" in expired.freshness_label()
    assert "72.0 hours" in expired.freshness_label()
    assert ordinary.age_hours == pytest.approx(40 / 60)
    assert expired.age_hours == pytest.approx(72.0)


# ─────────────────────────────────────────────────────────────────────────────
# The resolver: a stale source that "succeeded" is still reported
# ─────────────────────────────────────────────────────────────────────────────
def _board_frame(names: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "player_name": name, "position": "WR", "nfl_team": "NE",
            "ffc_adp": float(index + 1), "ffc_stdev": 3.0,
        }
        for index, name in enumerate(names)
    ])


def test_a_source_that_answered_from_an_expired_cache_is_named_on_the_report() -> None:
    """It contributed columns, so its age is on the board whether or not it "worked".

    Reported as a warning rather than an error because the board is still usable —
    the point is that the user can see which column is three days old and decide.
    """
    from services.providers.resolver import resolve_board

    ffc = ProviderResult(
        _board_frame(("Alpha One", "Bravo Two", "Charlie Three")),
        "Fantasy Football Calculator",
        fetched_at=(datetime.now(timezone.utc) - timedelta(hours=72)).isoformat(),
        from_cache=True, cache_age_seconds=72 * 3600, stale_fallback=True,
        report=ValidationReport(),
    )
    board = resolve_board(ffc=ffc, season=2026, team_count=12)

    assert board.ok, "the board is still built — stale data beats no data"
    messages = " ".join(issue.message for issue in board.report.warnings)
    assert "Fantasy Football Calculator" in messages
    assert "could not be reached" in messages
    assert "3.0 days" in messages

    status = board.source_status["ffc"]
    assert status["ok"] is True
    assert status["stale_fallback"] is True
    assert status["age_hours"] == pytest.approx(72.0, abs=0.1)


def test_an_ordinary_cache_hit_is_not_warned_about() -> None:
    """Otherwise every normal run carries a warning and the real one gets ignored."""
    from services.providers.resolver import resolve_board

    ffc = ProviderResult(
        _board_frame(("Alpha One", "Bravo Two")),
        "Fantasy Football Calculator",
        fetched_at=(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
        from_cache=True, cache_age_seconds=2 * 3600,
        report=ValidationReport(),
    )
    board = resolve_board(ffc=ffc, season=2026, team_count=12)

    messages = " ".join(issue.message for issue in board.report.warnings)
    assert "could not be reached" not in messages
    assert board.source_status["ffc"]["stale_fallback"] is False


def test_a_paged_source_is_as_old_as_its_oldest_page() -> None:
    """Yahoo is fetched a page at a time, and the pages can be different ages.

    Each page has its own cache entry, so a refetch that hits three cached pages and
    one live one used to be stamped with whichever page happened to come last. The
    ranks on the assembled board are as old as the oldest page in it.
    """
    from services.providers import yahoo as yahoo_module

    # The oldest page is neither the first nor the last, so neither min() nor a
    # first-wins nor a last-wins rule can pass this by accident.
    page_ages = [2.0, 96.0, 1.0]
    calls: list[str] = []

    def _yahoo_page(index: int) -> dict:
        return {"fantasy_content": {"league": [
            {"league_key": "nfl.l.public"},
            {"players": {"count": 1, "0": {"player": [
                [
                    {"player_id": str(index)},
                    {"full": f"Player {index}"},
                    {"display_position": "WR"},
                    {"editorial_team_abbr": "NE"},
                ],
                {"average_pick": str(index * 25 + 1)},
            ]}}},
        ]}}

    def _fake_fetch_json(url, **kwargs):
        index = len(calls)
        calls.append(url)
        if index >= len(page_ages):
            return {}, provider_base.FetchOutcome(
                payload=None, url=url, error="past end of list"
            )
        hours = page_ages[index]
        return _yahoo_page(index), provider_base.FetchOutcome(
            payload=b"{}",
            fetched_at=(datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(),
            from_cache=True,
            cache_age_seconds=hours * 3600,
            # The oldest page is the one that fell back, which is the realistic
            # shape: the page that could not be refreshed is the page that is old.
            stale_fallback=hours == max(page_ages),
            url=url,
        )

    original = yahoo_module.fetch_json
    try:
        yahoo_module.fetch_json = _fake_fetch_json
        result = yahoo_module.YahooProvider().fetch(player_limit=100)
    finally:
        yahoo_module.fetch_json = original

    assert result.ok and result.row_count == 3
    assert freshness.age_hours(result.fetched_at) == pytest.approx(96.0, abs=0.2)
    assert result.age_hours == pytest.approx(96.0, abs=0.2)
    assert result.stale_fallback, "one page came from an expired cache"


def test_a_board_reports_the_age_of_its_oldest_source_not_its_first() -> None:
    """Four sources, four timestamps, one board — and the board is the oldest.

    ``build_live_board`` used to take the first source that answered. With FFC live
    and Sleeper served from a four-day-old cache — which is what an outage produces,
    automatically — the board was stamped as minutes old with four-day-old names,
    teams and injury statuses on it.
    """
    from services import live as live_module

    sleeper_frame = pd.DataFrame([
        {"player_name": name, "position": "WR", "nfl_team": "NE",
         "sleeper_id": str(index), "sleeper_search_rank": float(index + 1)}
        for index, name in enumerate(("Alpha One", "Bravo Two", "Charlie Three"))
    ])

    def _result(frame: pd.DataFrame, source: str, hours: float) -> ProviderResult:
        return ProviderResult(
            frame, source,
            fetched_at=(datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(),
            from_cache=True, cache_age_seconds=hours * 3600,
            report=ValidationReport(),
        )

    class _FakeSleeper:
        def fetch(self, **kwargs):
            return _result(sleeper_frame, "Sleeper", 96.0)      # the oldest

    class _FakeFFC:
        def fetch(self, **kwargs):
            return _result(_board_frame(
                ("Alpha One", "Bravo Two", "Charlie Three")
            ), "Fantasy Football Calculator", 0.05)             # live

    original = (live_module.SleeperProvider, live_module.FFCalculatorProvider)
    try:
        live_module.SleeperProvider = _FakeSleeper
        live_module.FFCalculatorProvider = _FakeFFC
        result = live_module.build_live_board(
            season=2026, use_espn=False, use_yahoo=False
        )
    finally:
        live_module.SleeperProvider, live_module.FFCalculatorProvider = original

    assert result.ok
    assert freshness.age_hours(result.fetched_at) == pytest.approx(96.0, abs=0.2)
    assert result.freshness().level is Freshness.STALE
    assert result.pool is not None
    assert result.pool.metadata.freshness().level is Freshness.STALE, (
        "the pool the rest of the app reads has to carry the same age as the board"
    )


# ─────────────────────────────────────────────────────────────────────────────
# The pool: what it says about itself, and what survives the database
# ─────────────────────────────────────────────────────────────────────────────
def _pool_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"player_name": name, "position": "WR", "overall_adp": float(index + 1)}
        for index, name in enumerate(("Alpha One", "Bravo Two", "Charlie Three"))
    ])


def test_an_uploaded_pool_is_stamped_and_flagged_as_an_upload() -> None:
    """Every pool can say how old it is, and which kind of "old" that is.

    A pool with no timestamp would read as UNKNOWN and fire the warning on a file the
    user handed over five seconds ago, which is how a banner gets trained away.
    """
    from services.importers import import_player_pool

    result = import_player_pool(_pool_frame(), source="mine.csv")

    assert result.pool is not None
    metadata = result.pool.metadata
    assert metadata.imported_at, "an unstamped pool cannot be judged at all"
    assert metadata.timestamp_basis == freshness.IMPORTED
    verdict = metadata.freshness()
    assert verdict.level is Freshness.FRESH
    assert not verdict.is_concerning
    assert "uploaded" in verdict.headline()


def test_a_fetched_pool_keeps_the_fetch_time_it_was_given() -> None:
    """The live path supplies the real retrieval time and it is not overwritten."""
    from services.importers import import_player_pool

    fetched = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    result = import_player_pool(
        _pool_frame(), source="live: Sleeper", imported_at=fetched
    )

    assert result.pool is not None
    metadata = result.pool.metadata
    assert metadata.timestamp_basis == freshness.FETCHED
    assert metadata.freshness().age_hours == pytest.approx(30.0, abs=0.1)
    assert metadata.freshness().level is Freshness.AGING


def test_the_pool_summary_states_an_age_rather_than_a_timestamp() -> None:
    """The sidebar line has to be readable at a glance.

    A raw ISO string asks the reader to do the subtraction, and the reason this line
    exists at all is that nobody does.
    """
    from services.importers import import_player_pool

    fetched = (datetime.now(timezone.utc) - timedelta(hours=50)).isoformat()
    result = import_player_pool(
        _pool_frame(), source="live: Sleeper", imported_at=fetched
    )

    assert result.pool is not None
    described = result.pool.metadata.describe()
    assert "2.1 days old" in described
    assert fetched not in described


def test_a_pool_from_last_season_reports_the_season_not_the_age() -> None:
    """Minutes old and still the wrong board — the pool has to say which it is."""
    from services.importers import import_player_pool

    result = import_player_pool(
        _pool_frame(), source="live: Sleeper", season=2025,
        imported_at=datetime.now(timezone.utc).isoformat(),
    )

    assert result.pool is not None
    assert result.pool.metadata.freshness(expected_season=2026).level \
        is Freshness.WRONG_SEASON
    # No expected season supplied means no complaint: most callers do not know one.
    assert result.pool.metadata.freshness().level is Freshness.FRESH


def test_saving_a_pool_does_not_reset_how_old_it_is(tmp_path) -> None:
    """A save is not a fetch.

    The row used to be stamped with ``utcnow()``, so a board fetched three days ago
    and saved today came back tomorrow claiming to be brand new — the staleness
    warning would have been permanently disarmed for anyone who saves their league.
    """
    from models.database import init_db, session_scope
    from services.importers import import_player_pool
    from services.repository import load_player_pool, save_player_pool

    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    fetched = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
    imported = import_player_pool(
        _pool_frame(), source="live: Sleeper", season=2026, imported_at=fetched
    )
    assert imported.pool is not None

    with session_scope(db_path) as session:
        source_id = save_player_pool(session, imported.pool, source_kind="api")

    with session_scope(db_path) as session:
        reloaded = load_player_pool(session, source_id)

    assert reloaded is not None
    verdict = reloaded.metadata.freshness()
    assert verdict.age_hours == pytest.approx(72.0, abs=0.2)
    assert verdict.level is Freshness.STALE
    assert reloaded.metadata.timestamp_basis == freshness.FETCHED


def test_an_uploads_basis_survives_the_database(tmp_path) -> None:
    """Otherwise a reloaded upload claims its numbers are as fresh as the load was."""
    from models.database import init_db, session_scope
    from services.importers import import_player_pool
    from services.repository import load_player_pool, save_player_pool

    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    imported = import_player_pool(_pool_frame(), source="mine.csv", season=2026)
    assert imported.pool is not None

    with session_scope(db_path) as session:
        source_id = save_player_pool(session, imported.pool, source_kind="upload")
    with session_scope(db_path) as session:
        reloaded = load_player_pool(session, source_id)

    assert reloaded is not None
    assert reloaded.metadata.timestamp_basis == freshness.IMPORTED
    assert "uploaded" in reloaded.metadata.freshness().headline()


# ─────────────────────────────────────────────────────────────────────────────
# The banner: does a stale board actually say so on the page
# ─────────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")
TIMEOUT = 120


def _aged_pool(hours: float, *, season: int = 2026, basis: str = freshness.FETCHED):
    """A real pool, aged by hand, so the banner is driven by the same field the app uses."""
    from services.importers import import_player_pool

    stamp = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    result = import_player_pool(_pool_frame(), source="live: Sleeper", season=season,
                                imported_at=stamp)
    assert result.pool is not None
    result.pool.metadata.timestamp_basis = basis
    return result.pool


def _page(pool, *, league=None):
    from streamlit.testing.v1 import AppTest

    from core.config import SimulationConfig
    from models.draft import DraftHistory
    from ui import state as ui_state

    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.session_state[ui_state.K_INITIALISED] = True
    app.session_state[ui_state.K_LEAGUE] = league
    app.session_state[ui_state.K_POOL] = pool
    app.session_state[ui_state.K_HISTORY] = DraftHistory()
    app.session_state[ui_state.K_PROFILES] = {}
    app.session_state[ui_state.K_DRAFT] = None
    app.session_state[ui_state.K_SETTINGS] = SimulationConfig()
    app.session_state[ui_state.K_IS_SAMPLE] = False
    app = app.run()
    assert not app.exception, " | ".join(str(e.value) for e in app.exception)
    return app


#: A phrase only the aging banner produces. The sidebar prints the pool's age in a
#: caption too, so a test that matches on the age alone cannot tell the two apart.
_BANNER_ADVICE = "Fine for planning."


def _warnings(app) -> str:
    return " ".join(str(block.value) for block in app.warning)


def _captions(app) -> str:
    return " ".join(str(block.value) for block in app.caption)


def test_a_stale_board_says_so_on_the_page() -> None:
    """The end of the chain. Everything else is machinery for this one sentence.

    Rendered from ``page_header``, so it appears on every page rather than only on
    Setup: the page where stale ADP does damage is the Draft Room, and someone who
    left the app open overnight never goes back to Setup to be told.
    """
    app = _page(_aged_pool(96.0))
    warnings = _warnings(app)
    assert "4.0 days old" in warnings
    assert "Setup" in warnings, "a warning with no action attached is an interruption"


def test_a_fresh_board_is_not_warned_about() -> None:
    """The other half, and the reason the thresholds exist.

    A banner on every run is a banner nobody reads, which would make the stale case
    invisible again by a different route.
    """
    app = _page(_aged_pool(2.0))
    assert "old" not in _warnings(app)
    assert _BANNER_ADVICE not in _captions(app)


def test_a_day_old_board_gets_a_caption_rather_than_a_warning() -> None:
    """Between 12 and 48 hours the honest answer is "fine, but re-fetch before drafting".

    Matched on the banner's own advice sentence rather than on the age. The sidebar
    already prints the pool's age in a caption of its own, so asserting "20 hours old"
    appears somewhere passes whether or not this banner rendered at all.
    """
    app = _page(_aged_pool(20.0))
    captions = _captions(app)
    assert "20 hours old" not in _warnings(app)
    assert _BANNER_ADVICE in captions
    assert "This board is 20 hours old." in captions


def test_last_seasons_board_is_called_out_on_the_page() -> None:
    """Reported against the league's season, which is the only place the answer lives."""
    from core.config import LeagueConfig
    from models.league import League

    league = League(config=LeagueConfig(name="Test", season=2026, team_count=12))
    app = _page(_aged_pool(1.0, season=2025), league=league)
    text = _warnings(app) + " " + " ".join(str(e.value) for e in app.error)
    assert "2025" in text and "2026" in text
    assert "rookies" in text


def test_an_upload_is_not_described_as_freshly_fetched_data() -> None:
    """A file loaded three days ago is three days since *loading*, and says so.

    Without the distinction the banner would claim to know the age of numbers that
    nothing in the file dates.
    """
    app = _page(_aged_pool(96.0, basis=freshness.IMPORTED))
    warnings = _warnings(app)
    assert "uploaded 4.0 days ago" in warnings
    assert "not how old the numbers in it are" in warnings
