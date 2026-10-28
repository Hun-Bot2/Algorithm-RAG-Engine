# Issues, corrections and remaining uncertainty

Evidence levels: **committed correction** = inspected code/diff + durable artifact/test;
**uncommitted correction** = present checkout + tests/local receipts, no commit to cite;
**review finding** = recorded development review, not an independently reproduced historical
incident; **open limit** = current code/data boundary. None implies deployed-service proof.
Git baseline and current architecture are in [README](README.md).

## Historical corrections with source evidence

| Problem / root cause | Correction and provenance | Verification / remaining boundary |
|---|---|---|
| Date-named multi-problem logs were unsupported by the old per-path importer. The old LC path resolver strips a leading numeric prefix: `leetcode/26-10-20.md` could become `leetcode:10-20` rather than three problem IDs. | **Committed `a5fcc72`**: reject daily paths in `keyFromPath`; parse the table against the active catalog before legacy inference; stable daily attempt tokens and per-row provenance. | Inspected parent resolver and commit diff; `mastery-daily.test.mjs` covers three-row import, malformed pasted links, conflicts and memo-edit idempotency. No evidence of a wrong date-derived ID in this learner's actual three observations. Historical corruption is a possible old behavior, not observed here. |
| Legacy trailers were awkward for per-problem evidence and could not describe several distinct row outcomes in one commit. | **`41c7d20`**, structured Markdown attempts; **`c50239d`**, schema-2 O/X and defaults; **`0a57502`** records repository convention. This is an explicit evidence-contract evolution. | Parser/ingestion tests reject contradictory hints and preserve missing outcomes; do not retrofit real independence into old unmarked commits. |
| Registered solution commits were not automatically ingested by a running local web server. | **`82b892f`** introduces `createGitSync`, five-second committed-history polling and visible evidence-revision refresh. | Commit adds `mastery-git-sync.test.mjs` plus local-server test; rewrite stop/error isolation covered. Dirty note changes still require committing; external working trees are never auto-committed. |
| UI mission/progress did not expose a unified chronological Git/manual/review projection; failures could be hard to revisit. | **`df3ef8a`**, catalog-backed journal, immutable evidence revisions and universal review, reused progression/calibration rather than creating a second learner engine. | `mastery-journal.test.mjs`, `mastery-review.test.mjs`, `mastery-progression.test.mjs`; failure, correction, delayed review, Hard-vs-Medium regression boundaries exercised. These are supported behaviors, not proof of learner benefit. |
| AtCoder Problems may point original tasks at their latest ADT reuse; URLs/contest identity drift. | **`e80bb89`**, `refresh.py::canonical_atcoder_contests` prefers the task-prefix contest, else earliest non-ADT membership; immutable catalog-v3 supersedes v2. | `test_adt_reuse_does_not_move_canonical_contest`; `catalog-v3-refresh.json` and current helper. Native task IDs stay intact; no frozen-label rewrite. |
| Incremental title re-encoding can differ ~1e-7 with batch padding; text-only reuse could change unchanged-node neighbors. | **`e80bb89`**, `EmbeddingCache` keys model revision + `(problem_key,text)` before shared-text fallback. | `test_unchanged_nodes_keep_their_own_vectors_bit_for_bit`; `refresh-replay-acceptance.v1.json` records byte-identical full/replay neighbors. This audit inspected those hashes/tests; it did not re-encode the model. |
| Publisher enum interpretation differs: Hugging Face CodeContests uses AtCoder=5/Aizu=6, unlike original protobuf labels. | **`2b02146`**, `codecontests.py::source_names` uses pinned sidecar ClassLabel names rather than guessing integer meaning. | `test_sidecar_required_and_hf_enum` rerun; aggregate source receipts inspected. Source IDs without native correspondence remain unresolved. |
| Japanese titles formed language-driven neighborhoods under an English encoder. | **`6580d21`**, compare raw/centroid/translation/multilingual encoders with random controls; select translated-title C2 by a fixed rule; preserve full-v1. | Rehashed evaluation and reapplied decision: C2 gain .489983; recomputed full-map per-node overlaps and coordinate hashes. Translation noise, missing algorithm labels and unsupported scripts remain. |
| Stale README/workflow assumptions described an active Baekjoon batch that no longer represented the product direction. | **`cee364d`** removes obsolete Actions; **`f249dfb`** removes batch code, preserved as `legacy-pipeline-final`. | Git history/tag source inspected. There is no scheduled recommendation service proven by retained historical documentation. |

### Research review findings: corrected within their introducing changes

[IMPLEMENTATION-REVIEW.md](../research/unified_map/IMPLEMENTATION-REVIEW.md) records actual
2026-10-17 findings. **`3679a61`** already contains their corrected form; the Git snapshot does
not independently preserve every earlier broken intermediate, so these are not invented separate
fix commits:

- Input hashes alone did not bind exact A1 text/key selection; builders/reduction now bind catalog,
  receipt, variant digest and key order. Fixture source-binding tests cover metadata/hash drift.
- Sampled 2D masks initially differed from full-ranking duplicate/candidate-group masks; both now
  share the exclusions. Otherwise projection comparisons would mix protocols.
- Zero-Jaccard deterministic ties could earn a tag rank in RRF. `_rrf` now filters positive scores
  before fusion and writes component support. A zero-score tag-only neighbor remains **no evidence**.
- `sys.path` pollution loaded the wrong `test_evaluate`; package-qualified imports/relevant discovery
  were corrected. Some algorithm-system tests still require directory-based discovery today.

**`302a717`** adds full-map projection/validation. `full_project.py` documents that this recorded
sklearn t-SNE PCA initialization gives identical repeats regardless of seed; random-init runs
are needed to evaluate seed sensitivity. This is a measurement correction, not a proof that t-SNE
is generally seed-independent.

## Explanation integration: present locally, not in HEAD

The explanation modules, ADR 004 and related UI changes remain **uncommitted** in this checkout.
Do not attribute the following corrections to `a5fcc72` or invent later commit hashes.

| Observed problem | Current correction | Evidence and limit |
|---|---|---|
| Single-provider JSON fallback could not retain paired generation, retries or usage. | Concurrent Gemini/NVIDIA, same frozen input/prompt, append-only per-provider SQLite records, independent caches, usage coverage and side-by-side UI. | Actual source and `mastery-explanations.test.mjs` / CLI tests; a separate research ledger must be backed up independently. |
| Reported Gemini 404 on old default. | Default changed from 2.5 to 3.5 Flash-Lite; env overrides remain explicit. | Local ledger shows old failure then new-model success; exact historical 404 cause was not stored. The documented account/model restriction is a plausible explanation, not independently proved for that old request. |
| Reported NVIDIA 403 despite correct model/key. | Safe error classification now records known categories without raw error bodies; retry later reached HTTP 200. | Original stored error only has HTTP status. **Historical 403 root cause remains unknown**; the audit did not recontact NVIDIA or assume changing SDK solves authorization. |
| Later NVIDIA HTTP 200 still failed with `finish_reason:length`, 512 output tokens. | For the exact Nemotron model, `chat_template_kwargs.enable_thinking=false`; preserve the shared cap and record this parameter/cache identity. | Ledger shows truncated attempt followed by HTTP-200/stop success; parameter regression test. This supports the correction, not a claim that thinking caused the earlier 403. |
| Need to preserve printed answers on restart without new billing attempts. | Persisted results are read on startup/GET; only explicit generation dispatches. Successes survive `--retry-failed`. | `mastery-explanation-restart.test.mjs` actually terminates/restarts synthetic server processes; records/usage remain identical. This is stronger than an in-memory cache test. |

## Today's count: resolved observation, unresolved screen provenance

See [DATA_FLOW](DATA_FLOW.md) for exact IDs, rows and SQL/replay evidence. **All three are
persisted; two map to arrays-hashing, one to string-processing.** Current counts are 2/3 and 1/3,
not 0/3. The quote matches the October 18 zero-attempt snapshot; the October 20 LLM input already
contains 2/3. There is no basis to repair/delete these records. Which tab/history/example produced
the user's quoted string is **unresolved**, not a confirmed current UI-cache defect.

## Current limits and source-backed issues not fixed by this audit

- **Policy wording vs code:** progression says “not recently recommended,” but `priorExposure`
  excludes all prior plans through the date, unlike the base planner's finite cooldown. Confirmed
  source mismatch; no demonstrated learner harm or patch claimed.
- **Thin learner evidence:** today's calibration n=3 is insufficient; 3 Easy/2 Medium gates,
  metadata variation signatures, relatedness weights and retention timing lack outcome validation.
  Easy/Medium next choice is lexicographic. No learned personalization model is implemented.
- **Self-declarations:** daily O implies independent/no-hints by contract, not verified solving.
  Assisted ACs must not be assumed independent. The importer does not infer hints from memo/code.
- **Classification coverage:** trusted AI-reviewed concepts can be wrong. The AtCoder report is a
  450-item sample, not a full-label run: 136 labelled, 3 proposed-only, 56 abstained, 255 unknown;
  current catalog has only 211 distinct AtCoder keys with trusted concepts. These sums/statuses
  were checked against receipts/current SQL, not called classification accuracy.
- **Research label sparsity:** 44 positives in 3,647 historical pair records; 761 abstentions.
  Conditional metrics and neighbor preservation cannot establish full-catalog recommendation
  relevance. Sparse tags and unresolved publisher IDs prevent fully matched algorithm comparisons.
- **Known rights/source gaps:** LC statements/derived legacy vectors remain quarantined; admitted
  publisher text does not resolve native identity or authorize public redistribution. p03510 remains
  unresolved; abc084_c retains English/Japanese constraint and sample flags (preservation verifier).
- **Explanation completeness:** raw study explanations are not inputs; evidence-ID/JSON checks do
  not prove prose factuality. Stable prompt version must be managed deliberately. Unknown costs,
  timeouts and interrupted attempts remain unknown; external billing is not inferred from usage.
- **UI/version distinction:** current working-tree calendar workspace differs from older docs saying
  journal is the default. The follow-up browser tests validate current synthetic rendering; original stale-screen provenance remains open.

## Initial read-only audit verification (before follow-up implementation)

- Read-only database replay: October 18 **0** attempts → arrays 0; October 20 **3** attempts →
  arrays 2/variations 2, strings 1/variation 1. Catalog hash verified against receipt/switch.
- `node --test hosted/test/*.test.mjs`: **169 passed, 0 failed** on the observed working tree,
  with loopback permission; synthetic stores/providers only. This is not a release/remote-CI result.
- Eight targeted Python modules (catalog build/refresh/archive/relatedness/AtCoder concepts,
  reduction, full-quality, language metrics): **63 tests passed**. Algorithm-system discovery:
  `.venv/bin/python -B -m unittest discover -s research/algorithm_system -v`: **51 passed**.
  An initial package-style invocation failed to import local `compress`/`codecontests`; rerunning
  with the repository's documented discovery form resolved the invocation error. No source edit.
- `.venv/bin/python -B -m research.unified_map.verify private-learning/unified-map/map-v2`:
  **24,297 nodes, 501 preserved files, all checks passed**. It reads hashes/metadata only.
- Independently recomputed six title compression and nine statement compression Top10 overlaps,
  six full-map per-node means, 3,647 label totals, and language-selection decision; artifact hashes
  matched. Encoder training/inference, projection fitting, old timings and human labels were not rerun.
- During the initial read-only phase, only the four files in this directory were authored. Existing dirty code/workspace
  work was preserved. No real provider request, ingestion, data regeneration, learner correction,
  schedule, deployment or Git commit was performed.

## Follow-up fixes and investigation (2026-10-20, uncommitted)

These changes are present in the working tree, **not historical commits**. The hashes below for
source-note edits are in the sibling `algorithm-solution` repository.

| Symptom / observation | Confirmed root cause and affected code | Applied change / verification |
|---|---|---|
| Generic warning asks to check Markdown format and duplicate IDs. Local endpoint actually reports three `markdown_attempt_conflict` flags. | `db710c8` imported three attempts with null minutes; sibling `8b864b2` adds 10m to the same daily IDs. `markdownAttemptSignature` includes minutes, and `ingestRepository` correctly preserves the original semantic evidence. `app.mjs::pollLocalEvidence` collapsed every flag into an inaccurate generic warning. | New `workspace-data.mjs::syncDiagnostic` names problems/repository, explains changed time/result/hints, preserves independent solves, and distinguishes parser file/line flags and blocked history. Regression tests verify these messages. **No silent duration update**; the underlying conflict remains a legitimate review item. |
| Oct 20 solutions reportedly appear Oct 21. | Stored instants are Oct 19 15:00Z = Oct 20 midnight KST. `recorded_at` is Oct 20; daily title/date and canonical IDs agree. Future review entries are independently dated Oct 21. No actual timestamp shift was reproduced; original screenshot/version is unavailable. | Preserve timestamps. `workspace.mjs` explicitly labels review plans **복습 예정**, explains date-only midnight, and shows KST times in 24-hour format. Pure tests cover UTC 14:59:59.999/15:00 and offset timestamps; Chrome verifies the Oct 20 00:00 actual record, actual-vs-plan layers and real weekday contest dates. **Original reported screen cause remains unconfirmed**, not claimed as a fixed timezone defect. |
| `arrays-hashing: 0/3 Easy and 0/2 Medium` after three O records. | Read-only replay: Oct 18 snapshot has zero attempts; Oct 20 snapshot has 3 independent/no-hints practice observations, two trusted arrays-hashing mappings and one string-processing mapping. All three are LC Easy. | No policy/data fix necessary. Existing current evidence is **2/3 arrays Easy**, **0/2 arrays Medium**, plus one string Easy. Historical frozen explanations remain valid for their snapshots. Latest replay after implementation gives identical counts/catalog hash. Exact source of the old quoted screen remains unknown. |
| AtCoder synchronization rejects available catalog response. | `providers.mjs` bounded all calendars at 5,000 rows; live AtCoder Problems metadata has **6,582**. | Compatible metadata cap increased to 20,000 (4MB response bound retained). New discovery reads the official upcoming table. Regression fixture with 6,582 entries passes; live new parser reads 7 upcoming contests. |
| Need verified upcoming schedules across three platforms without LeetCode scraper or new secrets. | Local public adapters were disabled, old calendar path covered only CF/AtCoder, and LC had no admitted adapter. | `contest-discovery.mjs`, `provider-service.mjs`, `local.mjs`, `core/app.mjs`: explicit same-origin local sync, owner-scoped durable cache, source identity checks, official URLs, deduplication, six-hour TTL, five-minute failure suppression and stale fallback. Live CF/AtCoder/LC-mirror retrieval succeeds with 3/7/2 observed records. Mirror is correctly stale by publication time. No statement/API scraping added. |
| Fresh source can misleadingly expose an error string. | `cachedProviders` used `envelope.error ?? 'stale'`, so a fresh envelope's explicit null error became `stale`. | Fallback error now depends on computed freshness; successful fresh records return null error. This is a confirmed code-path defect, not a historically proven user incident. |
| Need personalized contest labels without fabricated difficulty/rating. | Existing weekend proposals use format-level heuristics and activity alone is not readiness evidence. Upcoming problems do not have published difficulties. | New independent `contestDiscoveryFit` reuses local platform calibration with six-distinct-independent/practice gate and a disclosed historical native opening-problem proxy. Recommended/stretch/insufficient branches tested; repeated, assisted, other-platform, stale and missing-demand cases remain insufficient. Existing problem selection and old live/virtual policy unchanged. New thresholds have **no measured predictive validity**. |

### Explicitly unresolved / intentionally preserved

- The three source `10m` edits are not applied to imported evidence. There is no explicit Git-observation
  correction UI yet. Changing IDs to suppress conflicts would create false new solves. Existing old
  timestamps/outcomes/null durations are preserved; an explicit append-only correction workflow is future work.
- Sync warnings are process-local diagnostics and persist across idle ticks until a later inspected commit;
  a restart is not proof that a conflict was resolved. This work fixes wording, not diagnostic durability.
- Original Oct 21 screen and original `0/3` screen provenance are unavailable. Current stored data and
  current rendered projection do not reproduce lost solves or a one-day shift.
- LeetCode uses an attributed community CLIST/ContestDeck mirror. Successful HTTP retrieval does not
  establish direct official verification or current publication; six-hour-old publications stay stale.
  Official registration eligibility is unknown, including rated/unrated distinctions; consult each URL.
- Contest fit is an exploratory **opening-problem practice goal**, not official rating/difficulty prediction
  or validated educational benefit. Sparse native catalog membership can keep even well-practiced users
  in insufficient evidence. The current learner's three LC solves are insufficient for skill estimation.
- No hosted deployment, background schedule, live notification, real Gemini/NVIDIA call or private learner
  correction was performed. The existing server must be restarted to load backend changes; no running
  service was stopped as part of this work.

### Follow-up validation receipts

- `node --test hosted/test/*.test.mjs`: **192 passed**, 0 failed (local loopback permission).
  Includes source normalization, metadata bounds, cache/failure/retry/owner isolation, restart integration,
  same-origin checks, classification gates, timestamp boundaries, original engine and explanation tests.
- `python3 -m unittest discover -s tests -v`: **74 passed**. Scientific artifacts were not regenerated.
- `SMOKE_OUTPUT_DIR=/tmp/recommendation-contest-browser node hosted/workspace-browser-smoke.mjs`:
  **3 scenarios passed** (both explanations succeed / partial failure / both fail). Rendered widths
  1440/834/390 have no horizontal overflow; provider results keep desktop side-by-side/mobile stacked,
  unchanged cache usage and two synthetic LLM attempts per scenario. Contest UI performs exactly three
  synthetic source calls, opens details with **Oct 21 23:35 KST / 120 min**, preserves weekday placement,
  marks weekends, and renders Oct 20 midnight attempts correctly. Screenshots were actually inspected,
  including `contest-details-desktop.png` and `populated-contests-390.png`.
- Browser receipt: `/tmp/recommendation-contest-browser/receipt.json`; full Node log:
  `/tmp/recommendation-final-node-tests.log`; live public metadata receipt:
  `/tmp/contest-discovery-live-receipt.json`. These are local temporary receipts, not deployed-runtime proof.
- Re-ran read-only progression replay (`/tmp/recommendation-readonly-audit.mjs`) and research metric audit
  (`/tmp/recommendation-metrics-audit.py`): unchanged 2-array/1-string counts, artifact hashes/neighbor
  overlaps/label totals/C2 selection agree with ALGORITHMS. These temporary investigation helpers are
  not repository tools; use the source modules/artifact paths cited in the documents to reproduce.
- `.venv/bin/python -B -m research.unified_map.verify private-learning/unified-map/map-v2`:
  **24,297 nodes, 501 preserved files, all checks passed**. `git diff --check` passed.

Browser harness issues corrected during this validation: select the contest tab before clicking its sync
button; use the intended nonempty synthetic fetcher; assert the UI's actual Korean numeric date format.
These are test-harness corrections, not invented historical product bugs. Product timestamp formatting
was made explicit with `hourCycle: h23` so a 23:35 KST contest is unambiguous.
