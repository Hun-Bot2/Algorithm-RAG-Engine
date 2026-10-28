# Data flow and the October 20 investigation

Audit scope: read-only inspection of the current local learner/catalog/explanation databases,
actual source and committed solution file. No import, learner correction, recommendation generation
or database migration was executed for this investigation. The follow-up
retrieved public schedule metadata and used disposable databases for integration/browser checks.

## 1. Input contracts: O is not universal proof of independence

| Input | Interpretation in this implementation |
|---|---|
| Daily `leetcode/YY-MM-DD.md`, platform equivalents or `daily/` | `parseDailyLog` resolves supported URLs/exact catalog titles. A resolved `O` explicitly maps to `independent,hints=false`; `X` to failed with unknown hints; `-` creates no attempt. This is the convention's self-declaration, not a verified assessment of how the user solved it. |
| Schema-2 Markdown | `parseMarkdownAttempt`: O defaults to no-hints, explicit hints must remain consistent; practice is default. Schema-1 independent requires explicit no-hints evidence. |
| Legacy Git trailers/files | `classify`: independent + explicit no-hints is independent; accepted/solved/unmarked commits remain `reported_completion`. |
| Web/CLI attempt | Event validator checks explicit outcome/hints and activity; review/transfer metadata is distinct. A platform-imported AC is only reported completion. |

Code: [daily-log.mjs](../hosted/mastery/daily-log.mjs) `parseDailyLog` L15–70;
[markdown-attempt.mjs](../hosted/mastery/markdown-attempt.mjs) L51–70;
[git-convention.mjs](../hosted/mastery/git-convention.mjs) `classify` L57–81;
[events.mjs](../hosted/core/events.mjs) payload validation.

The current dirty daily parser also accepts an optional `시간` column/self-reported minutes.
That is uncommitted work, not part of `a5fcc72`'s original three-column change. The October 20
import was a three-column table and stored minutes=null. Timers and manual minutes are distinct.

## 2. Git → canonical evidence → catalog → derived state

1. `registerRepository` stores explicit path/branch. `createGitSync` polls registered repositories;
   `ingestRepository` reads **committed blobs**, never current uncommitted note edits.
2. Import starts after the append-only watermark. Non-ancestor/rewritten history stops until an
   explicit reviewed anchor; errors must not silently advance successful evidence.
3. Daily paths bypass single-problem path inference. The parser checks filename/date/header/table,
   resolves identities against the active verified catalog, and flags ambiguous/unresolved rows.
   It supports the pasted `[|Title](URL)|O|memo|` variant and does not inspect detailed code sections.
4. Observations and watermark commit atomically. Stable repo+canonical-problem+daily-date tokens
   deduplicate edits; changed metadata under an imported attempt token is a conflict, not an update.
   Provenance retains commit SHA, path/blob hash, row/date/catalog and semantic signature.
5. `readLedger` reads core events **and** side ledgers, applies cutoff/high-water bounds and
   supersession, then `attemptStream` emits the common chronological attempt projection.
   Looking only at `events` would miss valid Git evidence.
6. Catalog construction joins native problem metadata with versioned concept mappings. Native
   tags → reviewed vocabulary mappings → `problem_concepts`; trusted statuses are `approved` and
   `ai_reviewed`. Notes and solution code do not classify the current learner's algorithm.
7. Calibration, concept counters, review replay and progression are derived views of those attempts.
   The exact counting predicates and reset rules are in [ALGORITHMS](ALGORITHMS.md).

Sources: [git-ingest.mjs](../hosted/mastery/git-ingest.mjs) L39–92,95–126;
[ledger.mjs](../hosted/mastery/ledger.mjs) L10–49;
[build.py](../research/mastery_catalog/build.py) `load_concept_map`/`build`;
[catalog.mjs](../hosted/mastery/catalog.mjs) L13–56.

## 3. Today's three problems: verified joins and outcomes

The user identifies the source as the
[DSA Quest](https://leetcode.com/quest/data-structures-and-algorithms-quest/).
That live URL exposed only a loading shell during this audit. The trace below is verified
against the actual committed problem URLs and the local catalog; Quest membership/order itself
was not independently established and is not a runtime classification source.

Solution repository: `~/algorithm-solution`; commit
**`db710c8fba4513c12a222bae0c8fcf114c4cc076`**, file `leetcode/26-10-20.md`, rows 5–7.
Imported at **2026-10-20 05:48:57 UTC / 14:48:57 KST**. Watermark sequence 1 records observed=3.
Each observation has `source=git`, `outcome=independent`, `hints=false`, `activity_type=practice`,
no supersession and date token `daily-2026-10-20`.

| Canonical LeetCode key (prefix `leetcode:`) | Native difficulty | Trusted catalog concepts | Contribution |
|---|---|---|---|
| `how-many-numbers-are-smaller-than-the-current-number` | `lc_level=1` (Easy) | arrays-hashing, sorting (`ai_reviewed`) | +1 arrays Easy; +1 sorting Easy |
| `find-all-numbers-disappeared-in-an-array` | `lc_level=1` (Easy) | arrays-hashing (`ai_reviewed`) | +1 arrays Easy |
| `longer-contiguous-segments-of-ones-than-zeros` | `lc_level=1` (Easy) | string-processing (`ai_reviewed`) | +1 string Easy; no arrays-hashing credit |

All three keys exist, are active/eligible/free-at-collection, and need no alias rewriting.
The first problem has Hash Table/Sorting/etc. tags; one separate arrays-hashing mapping via
Counting Sort is abstained, but its Hash Table mapping is AI-reviewed, so arrays credit remains.
The third has native tag **String**, not an arrays-hashing mapping. A shared Quest path does not
make every problem share one engine concept. No problem identity or difficulty failure was found.

Catalog: `private-learning/mastery/catalog-v4/catalog-v4.sqlite3`, SHA-256
`27a4ca622de50bfa6b86eb7d789abcc856bf40dd8fd511001e91edba20cffde4`, matching the catalog switch
and receipt. There are **zero matching core `events` rows**, but **three matching active
`git_observations`**; `attemptStream` sees all three. This distinction is essential.

Daily `occurred_at` is `2026-10-19T15:00:00Z`, i.e. **October 20 midnight KST**. The parser/importer
uses a date boundary, not an actual solve timestamp. This is not an off-by-one local-date loss.
The stored records meet the software's independence predicate; actual no-hint solving remains
user-reported and cannot be independently proved from O, code or Git alone.

## 4. Why the quoted 0/3 differs from the current result

| Stored journal snapshot | Evidence visible at preparation | arrays-hashing counters |
|---|---|---|
| `journal-3254daace4501f33de8dfaf32f76180db8e5a12a`, **2026-10-18**, cutoff `05:09:16.272Z` | high-water: Git observations=0; attempts=0 | Easy **0**, variations 0; Medium 0 |
| `journal-49c84d6cb219da16675cde8da84a49597107be4f`, **2026-10-20**, cutoff `05:48:58.005Z` | high-water: Git observations=3; attempts=3 | Easy **2**, variations **2**; Medium 0 |

Read-only replay through the actual `readLedger(...,{cutoff,highWater})` and
`planProgression(...,{evidenceOnly:true})` reproduced **both** stored count sets exactly.
The current replay additionally gives string-processing Easy=1/variations=1. The three Easy
attempts cannot produce Medium credit. Calibration sees n=3, posterior Easy alpha=6.2/beta=.8,
mean=.885714/SD=.112486, so `insufficient_evidence` is expected until six observations.

The old reason selected `leetcode:best-poker-hand` with **0/3**; the new reason selects
`leetcode:buddy-strings` with **2/3**. The local explanation ledger's frozen input references the
**October 20 snapshot and 2/3 text**, not the quoted zero text.

**Conclusion:** current data ingestion/counting is working for these records. The quote exactly
matches an older valid snapshot. Viewing history, an old tab or an earlier copied example could
explain the observation, but the exact UI path cannot be proven retrospectively without its
screen state/time. Do not label this a confirmed current stale-cache bug or claim all solves
were excluded. Old snapshots are immutable historical evidence and should not be deleted.

## 5. Recommendation snapshots → two LLM explanations

`journal.mjs::view` L65–75 derives an evidence-version key and reuses or saves an immutable plan.
CLI `plan` is different: it returns the first frozen plan for that date even after new evidence.
The journal keeps old revisions in history and selects a new active revision when evidence changes.

`explanationEvidence` L19–25 freezes only the selected problem, date/snapshot/catalog/policy,
kind/stage/difficulty/concepts, original engine reason and review metadata. **The user's memo,
full written explanation and code are not sent.** Canonical JSON and a versioned system prompt
are identical for concurrent Gemini/NVIDIA calls. LLMs never increment counters or pick problems.

The separate `private-learning/explanations/research.sqlite3` stores inputs, per-provider
attempts/results and optional actual-cost receipts. Cache identity includes learner partition,
evidence/input, prompt, provider/model and generation parameters. Successful responses survive
process restart; failed responses are also retained. Only explicit `explain` generation or
`--retry-failed` can dispatch; journal GETs/server startup do not call models. A changed model or
parameter set is a separate generation identity, not an overwritten old result.

For this checkout, records show the old Gemini/NVIDIA HTTP failures, later Gemini success,
NVIDIA HTTP-200 truncation, and NVIDIA success after its model-specific thinking switch. Usage
and failures are preserved independently. Safe output/evidence-ID checks are **not semantic
validation**. Keys and raw provider error bodies are excluded; no live API call was made here.
See [explanations.mjs](../hosted/mastery/explanations.mjs) `explain`, `presented`, `withExplanations`
and [explanation-store.mjs](../hosted/mastery/explanation-store.mjs).

## Reproduce safely

Use SQLite `mode=ro` and `DatabaseSync(...,{readOnly:true})`; do **not** invoke the normal CLI
for a forensic read because its database helper can initialize/migrate and journal views can
save a new snapshot. Select original `canonical` payloads from `git_observations`, catalog
`problems/problem_concepts/problem_tags`, and `recommendation_snapshots.payload`. Feed a
read-only `.prepare().bind().all()/first()` adapter to `readLedger`, then call pure
`planProgression` with the stored cutoff/high-water. This audit did exactly that; it neither
re-ingested commits nor modified the learner's evidence to make the count match.


## Calendar dates and the later duration-edit conflict

Read-only evidence from the running local sync endpoint at **2026-10-20 16:51 KST** reports
three `markdown_attempt_conflict:leetcode:...` flags. The original three-column rows were imported
from sibling solutions commit **`db710c8fba4513c12a222bae0c8fcf114c4cc076`**. Sibling commit
**`8b864b2`** (`fix : add solve time`, Oct 20 16:36:11 KST) adds a `시간` column and `10m`
to the same three daily attempt tokens. `markdownAttemptSignature` includes `self_minutes`, so
null → 10 changes semantics. `git-ingest.mjs::ingestRepository` intentionally skips the conflicting
version instead of replacing evidence or adding three new solves. Current persisted minutes remain
null. These hashes belong to **algorithm-solution**, not this engine's Git history.

The new `syncDiagnostic` explains exactly that safeguard, naming problems/repository and keeping
other parser flag/file/line details separate. The durations have **not** been applied. An explicit
append-only correction UX for imported Git attempts remains unavailable; the user must decide
whether a source edit is a correction or a new attempt. Recommending a new daily ID merely to
silence the warning would incorrectly create solve evidence, so this work does not do that.

All three `occurred_at` values remain `2026-10-19T15:00:00.000000Z` = **Oct 20 00:00 KST**;
`recorded_at` is Oct 20 14:48:57 KST. The importer uses the daily file's civil date at KST midnight,
not the Git commit clock and not a measured solve clock. Calendar actual entries use
`workspaceEvents → localDate(occurred_at)`; recommendations use `local_date` and future reviews
use `due_date`. Thus the same problem can legitimately appear as an actual solve on Oct 20 and a
**복습 예정** on Oct 21. Both preserve provenance and belong to different calendar layers.

The reported original screen was not captured, so its exact cause is **unconfirmed**. No stored
one-day timestamp shift was found. Current projections and rendered synthetic Chrome tests verify
Oct 20 midnight placement, Oct 20/21 UTC 15:00 boundaries, and separation of scheduled reviews.
The UI now labels review plans explicitly and explains date-only imported timestamps. It does not
subtract a day or alter stored data.

## Public contest metadata and local personalization

`공개 일정 동기화 → POST /api/contests/sync → local.mjs contestSync → syncProvider(discovery=true)`:

- Codeforces official JSON / AtCoder official upcoming HTML / CLIST mirror JSON → normalize
  strict official identity and ISO timestamp, duration, optional rated range and publication time.
- Per-platform owner-scoped cache lease → six-hour cache; stale-on-failure previous successful
  items; five-minute retry suppression; source publication age checked independently. Empty
  successful calendars are distinct from failed retrieval. No page-refresh upstream call.
- `GET /api/state → cachedProviders → readLedger + activeLocalCatalog → contestDiscoveryFit`:
  predictions use only local canonical attempts, rating observations and native catalog difficulties.
  No learner ID, attempt history, code or notes are included in upstream public GET requests.
- State → `workspaceEvents` gives a KST civil date from the actual `start_at`; list and calendar
  use the same events. The modal exposes duration, official URL, last successful fetch and upstream
  update time, stale status, conditional classification and evidence. Weekday events stay weekdays.

Original snapshots and explanation input hashes are not changed by these modules. Viewing the
calendar or contest details never dispatches Gemini/NVIDIA. Discovery cache JSON is replaceable
metadata, separate from append-only learning evidence; no schema migration was introduced.
