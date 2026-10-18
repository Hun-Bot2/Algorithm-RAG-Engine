# Local Algorithm Mastery Journal

## Start

Run from the repository root. The CLI and local server now resolve the same default
SQLite and catalog paths regardless of the current directory.

```sh
# First setup only, if no catalog is already selected. Uses the existing verified catalog.
node hosted/cli.mjs catalog use --version catalog-v4
node hosted/local.mjs
# Open http://127.0.0.1:8787
```

The first command applies the existing SQL migrations to your chosen database and
records the catalog selection. Implementation tests did not run it against your
real database. Existing catalog selections remain respected; no automatic switch.
The journal opens by default. Today/Progress/Contests, timers and backup controls
remain available. A catalog must be selected before journal recommendations exist.
The server is loopback-only; there is no deployment or external account requirement.

Both defaults use `private-learning/hosted.sqlite3`. For a different existing store,
use the same explicit absolute path on both entry points:

```sh
node hosted/cli.mjs journal --db /absolute/path/learner.sqlite3
LOCAL_DATABASE=/absolute/path/learner.sqlite3 node hosted/local.mjs
```

Earlier `npm run dev` launches from `hosted/` could have resolved a different relative
SQLite path. Those files are preserved; no automatic merge or migration of another
file is attempted. Select the intended file explicitly and use the full backup UI
for recovery. Backup v2 preserves snapshots as inactive archive rows in SQLite, not active recommendations;
a restored journal shows historical archived recommendations and derives fresh active snapshots from restored evidence.

## Record outcomes

Use “풀이 결과 기록” on a journal recommendation. O requires independent/no-hint
solving. X and assisted attempts enter priority review. Minutes are optional,
self-reported, and separate from explicitly confirmed timers. Unknown time is never
zero. Every attempt enters review, including reported completion (unverified).

```sh
node hosted/cli.mjs journal
node hosted/cli.mjs attempt --problem leetcode:two-sum --outcome O --hints none --minutes 24
node hosted/cli.mjs attempt --problem leetcode:valid-parentheses --outcome X --hints used
node hosted/cli.mjs attempt --problem codeforces:4A --outcome independent --hints none --activity transfer --from leetcode:two-sum
node hosted/cli.mjs attempt --problem leetcode:two-sum --outcome independent --hints none --activity review
```

These are syntax examples, not a claim that those example problems form a valid
transfer relation. The progression policy independently checks stage evidence,
source Medium, concepts, novelty and retention. Semantic-only candidates cannot
satisfy a concept transfer gate. The existing `plan` command retains its frozen
legacy daily-plan contract; `journal` is the curriculum-driven current view.

Optional focus preference:

```sh
node hosted/cli.mjs prefs --mastery '{"target_concepts":["arrays-hashing"]}'
```

## Optional local Git ingestion

```sh
node hosted/cli.mjs repo add --id solutions --path /absolute/path/algorithm-solutions --branch main
node hosted/cli.mjs repo ingest --id solutions
```

No GitHub push is required. Existing path/trailer conventions and rewrite-history
stops are retained; see `specs/mastery-system/runbook.md`. Missing outcome/hint
metadata remains reported completion, never independently solved. Git attempts feed
the same review/progression ledger as manual attempts. Repeat commits are idempotent.

## Policy and reproducibility

`hosted/mastery/progression-pattern.v1.json` defaults to three distinct Easy
variations, two Medium variations, new CF/AtCoder transfer, and an independent
retention review at least seven days later. Counts/variation thresholds are
configurable positive values; one Medium is supported. Changes require versioned
configuration review. Distinct problem IDs alone do not prove variation: native
concept/tag signatures are an explicitly provisional variation proxy.
`interview-curriculum.v1.json` supplies authored ordering/prerequisites, not proven
pedagogy. Native difficulty caps are conservative configuration, not cross-platform
rating equivalences. Hard/advanced mastery is not inferred from Easy/Medium success.

Failed or assisted attempts reset problem review to next day. Priority adds failure
urgency; unbounded overdue aging ensures old normal reviews can outrank recent
failures. The full queue remains visible even when the daily budget defers items.
An independent recovery lowers problem priority but preserves historical failures.
Concept remediation uses another eligible problem; novel transfer must use a new
canonical target after failure. A known-problem review never becomes novel transfer.

The journal composes the existing planner's reviews/calibration with progression
slots. It replaces ordinary legacy exploit/explore slots with stage-supported
practice, rather than adding another engine. On contest-rhythm days the journal
still offers curriculum/review; existing Today/Contests and CLI plan preserve
contest scheduling. Priority reviews may consume the budget; no unsupported filler
is invented when remaining candidates or minutes are insufficient.

Recommendations use immutable existing SQLite snapshots, keyed by date, relevant
attempt/rating evidence, preferences, config and implementation hash. New outcomes
create new revisions; timer-only actions do not rotate a problem. Snapshots contain
catalog/policy/config/code versions, event cutoff and ledger high-water marks.
Timeline corrections show the old record as superseded while active progression
uses only its successor. Mastery changes are evidence-prefix projections; labels
may change after catalog/config changes, so the associated versioned snapshot is
the historical recommendation record. No Markdown runtime database or LLM context
journal is introduced.

Core and planner versions are now deterministic-v3 / planner-v2 / mastery-policy-v2.
The journal is local-only; the Worker returns an explicit unavailable response
unless an appropriate journal service is injected. No private catalog is packaged
into or uploaded to Cloudflare.

## Limits and learning evaluation

These checks validate mechanics, not learning gains. Real learner history was empty
at audit, so no actual transfer success, retention improvement or educational
effectiveness can be claimed. Existing weekly reports expose outcome, review,
transfer and coverage denominators when evidence arrives. AI-reviewed concepts,
metadata variation signatures, user-reported independence and semantic candidates
remain weaker than judge-verified solution/algorithm evidence. Missing candidates
or classifications stay visible; no automatic graduation or invented matches.

Old clients that reject extra version1 payload fields need upgrading before reading
new optional `self_minutes`. The new reader accepts old records/backups. Confirmed
same-day timer time is labeled as not attributable to a specific attempt.

## Automatic solution reflection

While `node hosted/local.mjs` is running, registered solution repositories are
polled every five seconds through the existing Git ingester. The visible idle
Journal checks the local evidence revision every five seconds and refreshes only
when it changes (normally within about 5–10 seconds). An open outcome form is not
replaced; date filters survive refresh. The server must remain running.

First-time registration is unchanged:

```sh
node hosted/cli.mjs repo add --id solutions --path ../algorithm-solution --branch main
```

After that, write solutions and make local commits in your editor. No push or
repeated `repo ingest` command is required. Saving an uncommitted file does not
record an attempt or infer an outcome. O/X/hint metadata uses the existing commit
convention; missing metadata stays unverified. Rewritten history stops ingestion
and surfaces a warning, requiring explicit investigation/re-anchoring rather than
silently replaying history. No Git hooks or background OS service are installed.

## Markdown-only solution entry

Copy `../algorithm-solution/templates/solution.md` into a recognized problem path,
for example `leetcode/two-sum.md` or `leetcode/two-sum/README.md`. Replace the
attempt and outcome placeholders. No TXT file or Outcome commit trailer is
required. The detailed Korean guide is `templates/MARKDOWN-GUIDE.md` in that repo.

```md
---
learning_schema: 2
attempt: 26-10-18-01
outcome: O
---
# Solution
```

Use O for independent success and X for failure or non-independent solving.
No hints field is needed. Omitted activity means practice; add `activity: review`
for a review attempt. Old schema 1 records retain their original contract. Optional minutes
must be measured; omit unknown time. Use a new attempt token only for a real new
attempt. Prose-only edits retain one observation. Changed outcome metadata under
an existing token is flagged, never used to overwrite history. Existing explicit
commit outcome/hints trailers retain precedence. Invalid marked blocks are
skipped, with warnings exposed through local sync status. Templates are ignored.

With the solutions repository registered and the local journal running, ordinary
local commits are read on the existing five-second poll; the idle visible journal
then refreshes. Push is optional. A push from another computer must first reach
this local checkout; this service does not fetch, pull, or install remote hooks.
Previously processed commits are not silently rescanned. No real learner import
or repository registration was performed while implementing this feature.
