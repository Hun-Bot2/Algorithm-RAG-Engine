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

### Solutions repository convention cleanup

Active sibling README, Markdown guide, common solution template, optional commit
message template and push checker share schema 2. Previous scaffolding is preserved
under templates/legacy/ and is not an active format. The duplicate empty
leetcode/solution.md was moved there to avoid a fake problem identity. The
standalone checker ships a snapshot of the exact parser with only its problem-key
import replaced; parser changes must synchronize both copies. No real attempts,
registration, background service startup or push was performed.

## 적응형 progression-v2 (2026-10-29)

T8 합성 재생과 독립 AI 검토 후 일지의 기본 추천을 v2로 전환했습니다. `plan` 명령의 기존 일별 동결 정책은 그대로입니다.

```sh
node hosted/cli.mjs journal                      # v2; 네트워크 호출 없음
node hosted/cli.mjs journal --progression v1     # 이전 정책으로 조회
PROGRESSION_VERSION=v1 node hosted/local.mjs     # 로컬 화면도 이전 정책으로 실행
```

날짜별 Markdown 형식은 바꾸지 않습니다. 같은 문제의 같은 서울 날짜 기록은 마지막 결과 하나만 씁니다. 첫 날짜는 능력 관측,
다음 날짜부터는 실제 결과와 간격을 가진 복습 관측입니다. 복습 표시와 메모는 능력 근거가 아니며 시간은 보여주기만 합니다.
새 개념은 authored prior에서 시작하고, 해당 개념의 첫 능력 관측 이후 다른 개념의 이력으로 수준별 prior를 보정합니다.

추천의 `action`은 기초로 돌아가기(step_down), 현재 수준 다지기(consolidate), 다음 수준 도전(step_up), 상위 수준 탐색(stretch),
다른 플랫폼 전이(transfer)입니다. `estimate.p`와 `interval`은 모형의 추정 평균과 구간이며 실측 정답률이 아닙니다.
`benefit_proxy`는 추천 순서를 위한 가정적 점수이며 확률이나 학습 효과가 아닙니다. 구성요소 `D/I/C/F/K`는 도전 수준 적합도,
추정 불확실성 감소, 새 태그 변형, 미해결 후속 연습, 현재 개념의 연속성입니다. 가중치와 `config_sha256`도 함께 기록합니다.

`readiness`에는 준비도와 선행 조건 면제가, `coverage`에는 개념 기반 후보 부족이 표시됩니다. 제목 유사도로 대체하지 않습니다.
실패 후속 연습은 같은 수준 이하의 해결로 닫히거나 14일 뒤 만료됩니다. 이 기간과 가중치는 경험적으로 최적화되지 않았습니다.
복습과 progression은 별도 레인이어서 제안 시간 합계가 하루 예산을 넘을 수 있으며 그 경우 안내가 표시됩니다.

기존 기록과 과거 스냅샷을 수정하지 않습니다. 새 버전·설정·근거에 대해 새 일지 스냅샷이 생성됩니다. 이전 AI 설명은 과거
스냅샷에 남고, 새 추천의 설명은 별도 명시적 `explain` 실행 전에는 생성하지 않습니다. 주간 표시는 completion rate (policy v3)로
정정했지만 기존 `isr` JSON 키와 통계 계산은 호환성을 위해 유지합니다. 주간 복습/보정 통계는 기존 정의이며 v2 성능 검증이 아닙니다.

검증 기록: `specs/adaptive-progression/validation.md`. 실제 학습자 DB는 테스트에 사용하지 않았습니다.
