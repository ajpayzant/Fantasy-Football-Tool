# League-Aware Fantasy Football Mock Draft Simulator

A redraft snake-draft mock simulator whose opponents are modelled on **your league's own
draft history** rather than on generic ADP. It answers the question a normal mock draft
tool cannot: *will this player still be there in two picks, given who is actually
picking?*

Everything runs locally: a SQLite file, no account, no server. The player board is built
from **current public rankings and ADP** — Sleeper, Fantasy Football Calculator, ESPN and
Yahoo — and no fictional player or manager is reachable from the app.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Python 3.11 or newer** is required — `core/enums.py` and `core/freshness.py` use
`enum.StrEnum`, which does not exist before 3.11. On 3.10 the app dies on its first import
with `ImportError: cannot import name 'StrEnum'`.

On Windows, `launch_mock_draft.bat` (or `Fantasy Mock Draft.vbs`, which runs it without a
console window) starts the app on **port 8502** and opens the browser, and does nothing but
re-open the browser if it is already running. The port lives in that launcher rather than
in `.streamlit/config.toml`, deliberately — see the deployment note below.

Then, in the app: **Setup → "Fetch current player data"** (one button). That pulls live
rankings and ADP for the format you pick, joins the four sources into one board, and seats
generic opponents so you can draft immediately. Connect your Sleeper league ID on the same
page to replace those with your real managers and their past drafts — which is the whole
point of the tool.

To run the tests:

```bash
python -m pytest tests -q          # 575 tests, ~100s, no network
```

On Windows, prefix commands with `PYTHONIOENCODING=utf-8` — the console is cp1252 and
some output contains non-ASCII characters.

### Deploying to Streamlit Community Cloud

Main file path is `app.py`, and under **Advanced settings** pick **Python 3.11 or newer**
(3.12 or 3.13 is fine) for the reason above.

**Do not commit `.streamlit/config.toml` with a `server.port` in it.** Cloud assigns the
port itself; an app that binds to its own port instead never answers the health check, and
Cloud reports a bare *"Error running app"* with nothing useful in the log. That file is
gitignored here for exactly that reason, and the local port lives in
`launch_mock_draft.bat`. If you want a bare `streamlit run app.py` to default to 8502
locally, create the file yourself — it stays out of the repo:

```toml
[server]
port = 8502
```

Nothing else about the app is deployment-specific: it writes its SQLite file into `data/`
(which is why that directory is committed with a `.gitkeep`), reads no secrets, and needs
no environment variables.

---

## Where the data comes from

Four public, unauthenticated endpoints, each behind its own provider. **No provider
raises**: a timeout, a 404, a moved endpoint or a changed payload shape becomes a
validation error on the result, and the board is built from whatever else answered.

| Source | What it contributes | Notes |
|---|---|---|
| **Sleeper** | The player universe and the identity crosswalk (`espn_id`, `yahoo_id`), plus team, bye, experience, injury status | The identity spine. Its ids are `NaN` for team defences, which is why defences join on team code instead. |
| **Fantasy Football Calculator** | The primary ADP, and the only source publishing `stdev` / `high` / `low` | Mock-draft consensus, filtered by scoring format and team count. |
| **ESPN** | ESPN's own positional/overall ranks and its ADP | Kept in separate columns: ESPN's ADP is the average pick *in ESPN leagues*, a different signal from FFC's mock consensus, and the board labels it as such. Its endpoint cannot be filtered — it returns the whole player database (~39 MB, ~11.5k records) regardless — so **Setup has an "Include ESPN" toggle** for machines where that is too slow. |
| **Yahoo** | Yahoo's draft-analysis ADP and percent-drafted | Paged 25 at a time, so ~12 requests. |

Design decisions worth knowing before editing this layer:

- **Providers are dumb; the resolver joins.** Each provider shapes exactly one source into
  a DataFrame. `services/providers/resolver.py` does all cross-source reconciliation and is
  the only place that knows about more than one source.
- **No fuzzy name matching, by design.** A wrong join silently attributes one player's ADP
  to another, which is worse than a missing value. Joins go through the Sleeper id
  crosswalk, or through team code for defences — where four incompatible conventions
  ("Houston Texans", "Houston Defense", "Texans D/ST", "Texans") are normalised explicitly.
- **The consensus is not diluted by absence.** A player only one source carries gets that
  source's number, not an average dragged toward the sources that never listed them.
- **Requests are honest and cached.** The User-Agent identifies the tool
  (`fantasy-mock-draft/1.0 (personal draft tool; +local use)`) rather than impersonating a
  browser, payloads are cached on disk for 12 hours so a rerun does not re-request them,
  and a stale cache is served if the network fails. The cache is listed and clearable on
  **Setup → Saved leagues**.
- **The app is not dependent on any of them.** With every source down, the board is empty
  and the page says so and points at file import; with only ESPN and Yahoo down, the board
  is still complete.

---

## What it actually does

**1. It estimates a profile per manager from their past picks.** For each manager it
measures how far they reach past ADP, how much they follow the platform ranking, how
often they join a positional run, their rookie appetite, their NFL-team loyalty, their
round-by-round positional habits, and how predictable they are overall. Each of these is
**shrunk toward the league average in proportion to how much history exists** — a
statistic from four picks and one from forty produce the same kind of number, so every
parameter carries a provenance label (`observed`, `league fallback`, `baseline`) that is
shown in the UI. Recent seasons are weighted more heavily than old ones.

**2. It turns those profiles into a probability distribution over each pick.** Every
available player is scored against ~20 weighted features (ADP, projection, tier, VOR,
roster need, positional scarcity, the manager's own positional lean, run-chasing, stack
and handcuff preference, injury penalty…), then the scores are converted to pick
probabilities with a temperature derived from that manager's estimated predictability. A
predictable manager concentrates on their top-scoring player; an erratic one spreads out.

**3. It simulates forward from the current pick.** Rolling the next *k* picks forward
hundreds of times gives each player a survival probability — the number that decides
whether you take a player now or wait — plus who is most likely to take them and when
the position runs dry.

**4. It recommends through eight lenses,** each optimising for something different (best
overall, best roster fit, best value vs ADP, safest, highest upside, scarcity, last
chance, strategic alternative), and flags when several agree. Every recommendation
carries the engine's own headline, detail, and score breakdown — the reasoning shown is
the reasoning used, not a caption written by the UI.

**5. It runs whole drafts many times** from any point, so the output is the *distribution*
of rosters you tend to end up with rather than one draft that happened to go well.

---

## The pages

| Page | What it is for |
|---|---|
| **Home** | What the app is, and a readiness checklist. |
| **1 · Setup** | Fetch current rankings and ADP (one button, format selectable), connect a Sleeper league to pull your real managers and past drafts, import your own files (league settings / player pool / draft history / keepers), or reload a saved league. Every rejected row is shown with its reason. |
| **2 · Player Pool** | The board, with provenance: which columns you supplied, which were **estimated** because they were missing, and how VOR was derived from your league's shape. Filters, sorting, CSV export, position/tier charts. |
| **3 · Manager Profiles** | Every modelled number next to where it came from, per manager, plus the raw pre-shrinkage statistics and every pick they have made. Archetype labels are presented as inferred summaries, not as ground truth — a real league has no answer key. |
| **4 · Draft Room** | The live mock. Status bar, advance one pick / advance to your turn, likely next picks, positional pressure, one card per recommendation lens, survival table with "who takes them", manual override for any pick, board / rosters / runs / save tabs. |
| **5 · Simulations** | Monte Carlo from the current pick, with an optional "best available by ranking" baseline to compare against. Starter-points distribution with 5th/50th/95th percentiles, player frequency, roster shape vs the starting lineup's demand, unfilled slots. |
| **6 · Analysis** | Review the live draft or any saved one: full board, your picks with reach/fall narrative, where the value went across the room, and every final roster with unfilled-slot and bye-week warnings. |
| **7 · Settings** | Every weight and constant the model uses, each with a help string describing its *behavioural* effect. Includes a live temperature-curve preview. Applying settings discards cached results, because they were computed under the old weights. |

---

## Honesty mechanisms

These exist because a simulator that cannot be checked is indistinguishable from one that
makes numbers up.

**The archetype answer key — kept, but moved out of the app.** Twelve synthetic managers
were each generated from a fixed, known strategy. The estimator never sees those plans; it
reads only the resulting picks. `scripts/check_sample_archetypes.py` recovers all 12:

```bash
python -m scripts.check_sample_archetypes
# ...
# 12/12 recovered
```

Six of the twelve share a neutral opening (their tell is in rookie preference, team
loyalty, reach spread, or the late rounds), so `tests/fixtures/sample_league/drafts.py`
also exports `DESIGNED_TELL`: one sentence on what each manager was built to demonstrate.
This is the evidence that the opponent model measures something real rather than merely
producing plausible numbers — but it lives under `tests/` and **the app has no route to
it**. Against a real league there is no ground truth, so the Manager Profiles page shows
inferred labels as inferred rather than printing a comparison table implying an answer key.

**No fictional data is reachable from the app.** The synthetic league is a test fixture:
the `sample` adapter is not in `available_adapters()`, and a headless test asserts the
rendered Setup page contains no route to it. The sample-data banner and
`state.is_sample_data()` remain, because a league *saved* as sample data in an earlier
session can still be reloaded from the local database and must still be labelled.

**Live data is labelled by source, and by what the number means.** ESPN's ADP is the
average pick in ESPN's own leagues, not the mock-draft consensus FFC publishes; the two are
kept in separate columns and the board carries an `espn_adp_population` warning saying so.
Where the spread (`adp_stdev`) had to be estimated rather than published, the rows are
counted and flagged.

**Offline tests run against recorded real payloads.** `scripts/record_live_fixtures.py`
writes trimmed copies of what the four endpoints actually returned into
`tests/fixtures/live_payloads/` (16 files, ~2.7 MB, trimmed by row count only — every field
of every kept record is verbatim). The tests then point the provider cache at those files
and block `urllib.request.urlopen`, so every provider runs its whole real path — cache
lookup, decode, shaping, report construction — with a network call being a test failure.
Hand-written fixtures were rejected deliberately: they would encode what the shape was
*assumed* to be, and the assumptions are exactly what needs pinning down.

**Estimated values are marked as estimated.** The Player Pool page distinguishes
`metadata.missing_fields` (you did not supply it) from `metadata.imputed_fields` (we filled
it in), because a projection the app invented should not look like one you provided.

**Nothing is swallowed.** Importers return a `ValidationReport` with errors, warnings, and
info; the UI renders all three severities and shows rejected rows in a table with reasons
attached. Rejected pick attempts (draft over, player gone, reserved keeper) surface as
errors rather than silently doing nothing.

---

## Layout

```
app.py                  Streamlit entry point: page registration, logging, schema init
core/                   Enums, configuration dataclasses, validation, constants   (2,685 loc)
  config.py               SimulationConfig, ModelWeights, ShrinkageConfig,
                          ProfileEstimationConfig, RosterSettings, ScoringRules
  validation.py           validate_league / ValidationReport / ConfigurationError
engine/                 All simulation logic — never imports Streamlit           (5,404 loc)
  draft_order.py          Snake / linear / third-round-reversal / custom orders
  draft_state.py          The mutable board: picks, rosters, availability
  features.py             Annotates picks with context (rank inversions, runs, fills)
  opponent_model.py       Observation → shrinkage → ManagerProfile → archetype
  pick_model.py           Feature scoring and the softmax over candidates
  simulator.py            DraftSimulator, availability rollouts, monte_carlo_draft
  recommender.py          The eight lenses and their explanations
models/                 SQLAlchemy schema and the domain objects                 (3,732 loc)
services/               Import, normalisation, adapters, live providers, storage  (5,148 loc)
  live.py                 build_live_board: fetch → resolve → import → league
  providers/              One module per source, plus the resolver               (4,079 loc)
    base.py                 Fetch, disk cache, ProviderResult, failed_result
    sleeper.py              Player universe + the espn_id/yahoo_id crosswalk
    ffcalculator.py         Primary ADP, with published stdev/high/low
    espn.py / yahoo.py      Platform ranks and platform-population ADP
    leagues.py              Real league connectors (Sleeper ID; ESPN/Yahoo notes)
    resolver.py             The cross-source join and consensus merge
ui/                     Streamlit only — imports the engine, never the reverse    (6,233 loc)
  state.py                Typed session-state accessors with cache invalidation
  components.py           Shared rendering: banners, gating, charts, flash messages
  pages/1..7_*.py         The seven pages
scripts/                Developer diagnostics, none of them imported by the app
  record_live_fixtures.py   Re-record the provider payloads the offline tests replay
  check_sample_archetypes.py  Recover all 12 designed archetypes from picks alone
  diagnose_picks.py         Reach report: which utility term drove each early pick
  diagnose_early_rounds.py  Round-by-round spread of picks against consensus ADP
tests/                  575 tests, no network                                   (10,230 loc)
  fixtures/sample_league/  The synthetic league — test-only, unreachable from app (1,443 loc)
  fixtures/live_payloads/  Recorded real provider responses, trimmed  (17 files, 2.8 MB)
```

**The one architectural rule:** the engine never imports Streamlit. `ui/` depends on
`engine/`; `engine/` does not know the UI exists. This is what lets the whole simulation
run under pytest and from `scripts/`, and it is enforced in practice by the test suite
running without a Streamlit runtime.

**Two Streamlit-specific consequences worth knowing before editing the UI:**

- `st.cache_data` is unusable for engine objects — they are unhashable, unpicklable, and
  keyed by mutable draft state. `ui/state.py` provides `cached(key, builder, stamp)`
  instead, stamped on the draft's pick index so a result computed two picks ago is never
  displayed as if it applied to the current board.
- `st.rerun()` discards output already rendered this pass, so a success message written
  before a rerun vanishes. Confirmations go through `components.flash()` and are drained
  by `render_flashes()` on the next pass.

---

## Importing your own league

Fetching current data is the fast route; importing is the route to *your* league. Setup's
**Connect** tab takes a Sleeper league ID and pulls the real managers plus every past draft
Sleeper has, which is what turns generic opponents into modelled ones. ESPN and Yahoo
leagues require private-league authentication that this tool deliberately does not
automate — the page prints the export steps instead, and the files land through the tabs
below.

Column names are matched through a large alias table
(`services/normalize.py`), so exports from ESPN, Yahoo, Sleeper, NFL, CBS and Underdog
generally land without editing. Templates are downloadable in the UI; the canonical
columns are:

**Player pool** — `player_name`, `position` required; `nfl_team`, `bye_week`, `experience`,
`rookie_flag`, `injury_status`, `projection`, `overall_rank`, `position_rank`,
`platform_rank`, `overall_adp`, `platform_adp`, `adp_stdev`, `min_pick`, `max_pick`,
`tier`, `ceiling`, `floor`, `risk_score`, `value_over_replacement`, `notes` optional.

**Draft history** — `season`, `manager_name`, `player_name` and a pick position
(`overall_pick`, or `round` + `pick_in_round`) required; `position`, `nfl_team`, `adp`,
`platform_rank`, `projection`, `tier`, `keeper_flag`, `rookie_flag`, `platform`,
`league_name`, `draft_date` optional.

**Keepers** — `manager_name`, `player_name`, `keeper_round` / `overall_pick`,
`removes_pick`.

Missing optional columns are estimated and flagged, not silently invented. Manager names
are normalised and cross-checked against the league roster, and both unmatched and missing
names are reported — a spelling mismatch there is the most common reason a manager ends up
modelled on the league average.

Data is persisted to a local SQLite file (`data/fantasy_mock_draft.db`, path shown on
Settings → System). Deleting it resets the app to empty.

---

## Current state, and what is left

**Working end to end**, verified by the test suite and by booting the app:

- Redraft snake drafts, including linear, third-round reversal, and custom round orders.
- Live fetch → four-source join → validation → player pool → draft, from one button.
- Sleeper league connection: real managers and every past draft Sleeper exposes.
- Import → validation → profile estimation → live draft → recommendations → Monte Carlo →
  post-draft analysis, all from the UI.
- Persistence for leagues, player pools, draft history, manager profiles, saved mocks and
  simulation runs.
- 575 tests passing with no network access, including 95 headless page tests via
  `streamlit.testing.v1.AppTest` that render every page with real engine objects, and 44
  provider/resolver tests run against recorded real payloads.

**Deliberately not implemented** (the spec prioritised a reliable redraft snake experience
over partial advanced features):

- **Auction drafts.** `DraftType.AUCTION` exists in the enum, and `core/validation.py`
  emits a warning saying so: the data model supports it but the engine treats the league as
  a snake draft (`engine/draft_order.py` gives it a linear order). Nothing pretends
  otherwise, but note this is a warning rather than a hard rejection — an auction league
  will run, as a snake.
- **Keeper / dynasty leagues** are *partially* supported: keepers can be imported and they
  correctly remove picks from the board, but no keeper-specific valuation (contract value,
  future-pick trading, rookie-draft logic) exists. The Setup page says so on screen.
- **Best-ball / weekly lineup optimisation.** Starter projections assume the best legal
  lineup, not a week-by-week optimisation.

**Known rough edges worth a decision:**

1. Settings are session-scoped, not persisted. A browser reload returns to the defaults in
   `core/config.py`. The `settings` table exists and `read_setting`/`write_setting` are
   implemented, so wiring this up is small.
2. Monte Carlo runs are single-threaded. A 500-run full-draft simulation takes real time.
   The engine is pure Python with no shared mutable state across runs, so this parallelises
   cleanly if it becomes annoying.
3. The recommendation lenses are not weighted against each other — the app shows all eight
   and flags agreement rather than producing one ranked answer. That was deliberate, but it
   is the main thing to reconsider if the page feels like too much to read at pick speed.

---

## The synthetic league (test-only)

`tests/fixtures/sample_league/` — **not reachable from the app.** It exists because the
opponent model needs ground truth to be checkable, and real leagues do not come with any.

260 fictional players (83 WR, 68 RB, 34 QB, 31 TE, 22 K, 22 DST; 48 rookies), 12 managers,
three seasons (2023–2025) and 576 historical picks. 12-team half-PPR redraft, 16 rounds,
snake, QB/RB/RB/WR/WR/TE/FLEX/K/DST plus 7 bench; the user sits in slot 6.

Each manager was generated to one of 12 archetypes — `zero_rb`, `robust_rb`, `hero_rb`,
`early_qb`, `late_round_qb`, `elite_te`, `rookie_heavy`, `platform_rank_follower`,
`favorite_team_homer`, `high_variance`, `autodraft_like`, `balanced` — and the estimator
recovers all 12 from picks alone. That is the fixture's whole purpose.

To use it deliberately (in a test or a script), the adapter must be registered by hand:

```python
from services.adapters import SampleDataAdapter, register_adapter, unregister_adapter
from tests.fixtures.sample_league.players import sample_player_frame

register_adapter("sample", SampleDataAdapter(loader=sample_player_frame))
# ... anything read through it is flagged is_sample_data=True and carries a
# "fictional" notice on its validation report ...
unregister_adapter("sample")
```

No number in it describes a real football player. Real players come from **Setup → Fetch
current player data**.
