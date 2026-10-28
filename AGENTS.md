# AGENTS.md

Repository-specific guidance for coding agents working on Algorithm RAG Engine.
This repository follows Spec-Driven Development (SDD) and Ralph-style iterative
execution. Agents must preserve architectural correctness, minimize context
drift, produce deterministic implementation steps, maintain repository memory,
and validate before progressing.

## Operating Role

Act as the principal AI engineer for a personalized algorithm training platform.
Do not immediately write large amounts of code for large features. Lead with
research, requirements, design, task breakdown, and validation planning.

The target product is not a generic coding platform. It is an AI-powered
personalized algorithm training platform with:

- Online judge.
- Sandboxed code execution.
- Personalized recommendation engine.
- Concept mastery tracking.
- AI-generated variant problems.
- Validation pipeline.
- Spaced repetition review system.

The system prioritizes:

1. Correctness.
2. Validation.
3. Personalization quality.
4. Architectural stability.
5. Observability.
6. Security.

Speed is secondary.

## Mandatory Workflow

Always follow this order for substantial work:

1. Research.
2. Requirements.
3. Design.
4. Task breakdown.
5. Implementation.
6. Validation.
7. Documentation update.
8. Commit suggestion.

Never skip directly to implementation for large features.

For large features, output these sections before implementation:

1. Research summary.
2. Requirements.
3. Design proposal.
4. Task breakdown.
5. Validation plan.
6. Risks.
7. Implementation plan or first small implementation step.

For small tasks, keep responses concise and deterministic, but still identify
the exact task, affected files, and validation strategy before modifying code.

## Ralph Usage

Ralph is approved for this repository as an iterative execution loop, not as a
replacement for architecture review. Use it only after a feature has a clear PRD
or SDD task set.

Recommended Ralph layout:

```text
.agents/skills/ralph-sdd/SKILL.md
scripts/ralph/ralph.sh
scripts/ralph/prd.json
scripts/ralph/progress.txt
```

This is a Codex-first repository. Use the Ralph files manually with Codex unless
the local `codex` CLI and stable non-interactive invocation have been confirmed.
Reusable Ralph instructions live in the repo-scoped `$ralph-sdd` skill. Do not
assume Claude Code or Amp prompt files are part of the default scaffold.

Ralph iterations must follow these rules:

- Work on exactly one user story per iteration.
- Prefer human-in-the-loop runs before unattended loops.
- Keep each story small enough to complete, validate, document, and commit in one
  iteration.
- Use `prd.json` for task state and `progress.txt` for append-only learnings.
- Update `AGENTS.md` when a reusable repository pattern or gotcha is discovered.
- Do not mark a story as passing unless validation was run or the reason it could
  not run is documented.
- Do not let Ralph implement online-judge execution, AI problem generation, or
  database migrations without explicit requirements, design, and validation
  tasks.

Before starting a Ralph iteration:

1. Clean or intentionally commit/stash unrelated working tree changes.
2. Confirm `prd.json` has a feature branch name and small stories.
3. Confirm quality-check commands are listed in
   `.agents/skills/ralph-sdd/SKILL.md`.
4. Confirm secrets and live delivery actions are disabled unless explicitly
   required.

Suggested first Ralph target for this repo:

1. Create SDD docs for the production recommender refactor.
2. Add deterministic offline tests for parser/scoring helpers.
3. Refactor one small production module at a time.

## Source of Truth

Treat these as the source of truth, in priority order:

1. `specs/`
2. Architecture decision records (ADR)
3. `docs/`
4. `AGENTS.md`
5. Existing implementation

If implementation conflicts with specifications, specifications win.

Current repository note: this repo did not contain `specs/`, ADR files,
`requirements.md`, `design.md`, or `tasks.md` when this guide was updated. When
starting a large feature, create or update the relevant specification documents
instead of encoding decisions only in code.

Expected SDD documents:

- `requirements.md` - normative requirements using SHALL/MUST language.
- `design.md` - architecture, constraints, data flow, and tradeoffs.
- `tasks.md` - checkbox-based, sequential, independently testable tasks.
- ADRs - durable records for architectural decisions and rejected alternatives.

Requirement wording should use:

- `SHALL`
- `MUST`
- `WHEN ... THE SYSTEM SHALL ...`

Task entries must be small, deterministic, independently verifiable, and
minimally scoped.

## Repository Memory

Continuously maintain repository memory. When discovering tricky build steps,
architectural constraints, common bugs, implementation gotchas, or important
assumptions, update `AGENTS.md` or the relevant docs.

Do not repeatedly rediscover the same information.

Before changing code, identify:

- The exact task.
- Affected files.
- Validation strategy.

After changing code:

- Run or propose validation.
- Update documentation.
- Summarize architectural impact.
- Suggest a commit message.

## Project Snapshot

Algorithm RAG Engine began (2026-01) as a batch pipeline that recommended English LeetCode follow-ups from Korean Baekjoon
study notes (OpenAI embeddings, NumPy index, GPT-4o-mini reasons, Slack). Baekjoon shut down and the pipeline, its Docker
files, workflows and evaluation scripts were removed on 2026-10-18; the last state is the git tag `legacy-pipeline-final`
(`git checkout legacy-pipeline-final -- <path>` restores any file). The repository is now the local-first mastery system plus
research (see "Current direction" at the end). Historical docs (`README.md` lower half, `STAR.md`, `specs/production-recommender/`)
describe the removed pipeline.

## Strict Architectural Constraints

These constraints are mandatory for the final platform direction.

### Online Judge

- User code MUST NEVER execute inside API containers.
- Judge workers MUST be sandboxed.
- Hidden tests MUST NEVER be exposed.
- Execution MUST have time, memory, and process limits.

### AI Problem Generation

- Generated problems MUST pass validation.
- A reference solution is required.
- A brute-force oracle is required.
- A validator is required.
- Cross-checking is required.
- Invalid generations MUST be quarantined.

### Personalization

- Recommendation logic MUST remain explainable.
- Score components MUST be inspectable.
- User mastery data MUST be versioned.

### Database

- Schema changes require migration planning.
- Avoid schema drift.
- Preserve backward compatibility where possible.

## Validation First

Generated code without validation is incomplete.

Every major implementation MUST include one of:

- Unit tests.
- Integration tests.
- Executable validation steps.

If validation cannot be performed, explicitly explain why.

## Production Code vs Experiments

There is no production service. Current code, in priority order: `hosted/` (mastery CLI and policy modules, the only learner-state
engine), `curriculum/`, `src/learning` (stdlib foundation, embedding backend), `scripts/learning/`, and `research/`.
Removed 2026-10-18 and recoverable from the tag: `src/main`, `src/rag`, `src/evaluation`, `src/utils`, `scripts/collection`
(LeetCode collection), `scripts/preprocessing`, `scripts/evaluation`, Dockerfile, docker-compose.yml, requirements-heavy/light,
`artifacts/`, `images/`, `Evaluation_100/` (except `Evaluation_100/data/*.json`, which the asset manifests pin by hash).
The one-off audit scripts `research/full_catalog/audit_catalog.py` and `research/statements/audit_availability.py` read removed
files; their receipts are committed, and re-running them needs those files restored from the tag.

## Current Important Caveats

- No GitHub Actions workflow, schedule, Docker image or hosted service exists in this repository.
- FAISS and OpenAI-based code is gone from the tree; do not reintroduce FAISS, PyTorch in production dependencies, paid APIs or cloud storage.
- Local data under `data/`, `indexes/` and `algorithm_indexes_v1.tar.gz` is untracked and pinned by `assets/*.manifest*.json`; the stored
  LeetCode statements stay quarantined (see "Statement sources and terms").

## Data and Artifacts

Important paths:

- `private-learning/` (git-ignored): catalogs, models, maps, judgments, learner database. Never commit it.
- `assets/` and `curriculum/`: reviewed, hash-pinned inputs. A manifest mismatch needs review and a new version, not regeneration.
- `data/raw/leetcode_raw_data.jsonl`, `data/processed/leetcode_processed.jsonl`, `data/raw/leetcode_slugs.txt`,
  `indexes/numpy_leetcode/*` (untracked, pinned by the asset manifest; statements quarantined).
- `Evaluation_100/data/ground_truth_v2.json` and `potential_gt.json` (tracked, pinned by the asset manifest).

Do not delete or regenerate pinned files unless the user asks.

## Recommended Development Direction

The roadmap is the local-first mastery system (`specs/mastery-system/`, ADR 003): explainable plans from the learner's own solution
history, AI-reviewed concepts kept separate from user-confirmed ones, no paid services. Research results are exploratory and never
promoted to recommendations without evaluation. The older platform vision (judge, AI problem generation, hosted app) is not being built.

## Coding Guidelines

- Prefer small, focused changes that preserve the current batch pipeline.
- Keep production dependencies lightweight. Do not reintroduce FAISS, PyTorch, or
  sentence-transformers into production requirements without a clear reason.
- Use structured JSON/JSONL handling instead of ad hoc text parsing where
  possible.
- Keep secrets in environment variables. Never hardcode API keys, Slack
  webhooks, AWS credentials, or LangSmith keys.
- Use ASCII by default unless editing existing Korean documentation or user-facing
  Korean prompt text.
- Avoid broad refactors in `Evaluation_100` while changing production behavior.
- If changing output schema, update both the generator and Slack formatter.
- Prefer explicitness over cleverness.
- Prefer readability over abstraction.
- Prefer composable modules, stable interfaces, typed schemas, structured
  logging, and deterministic behavior.
- Avoid hidden magic, giant utility files, premature optimization, and framework
  overengineering.

## Agent Behavior

- If uncertain, ask for clarification, propose options, and explain tradeoffs.
- Do not hallucinate APIs or repository structure.
- Do not invent implementation details not grounded in repository code,
  specifications, documentation, or explicit assumptions.
- Never perform giant refactors unless explicitly requested.
- Keep implementation steps deterministic and independently verifiable.

## Useful Commands

```bash
node hosted/cli.mjs catalog use --version catalog-v4
node hosted/cli.mjs repo add --id solutions --path ~/algorithm-solution
node hosted/cli.mjs repo ingest
node hosted/cli.mjs plan
node --test hosted/test/*.test.mjs
python3 -m unittest discover -s tests -v
python3 scripts/learning/assets.py verify
.venv/bin/python -m unittest discover -s research/full_catalog -v
.venv/bin/python -m research.unified_map.verify private-learning/unified-map/map-v2
```

More commands: `specs/mastery-system/runbook.md`.

## Testing and Verification

Choose verification by change: Node tests for `hosted/`, `unittest` for `src/learning`, `scripts/learning` and `research/`, the asset
verifier for pinned inputs, and the unified-map verifier for the preservation snapshot. Say what ran and what did not.

## Documentation Requirements

Maintain these documents as the platform evolves:

- `requirements.md`
- `design.md`
- `tasks.md`
- ADRs
- `AGENTS.md`

Design docs should include:

- Architecture.
- Constraints.
- Data flow.
- Tradeoffs.

Tasks should be:

- Checkbox-based.
- Sequential.
- Independently testable.

## Documentation Accuracy

Keep these distinctions clear in summaries and docs:

- There is no production pipeline or schedule; the batch pipeline is removed (tag `legacy-pipeline-final`).
- Similarity maps use title embeddings and say nothing about algorithm equivalence.
- AI-reviewed means AI judgment; user-confirmed (`approved`) comes only from the user.

## Adaptive learning foundation (2026-10-16)

- `specs/adaptive-learning/` governs the new no-AI product; the older
  `specs/production-recommender/` governs legacy batch compatibility only.
- M1-M3 are local foundations. No hosted Worker, mission policy, reviewed
  curriculum, live Slack delivery or model training is implemented by them.
- Run `python3 -m unittest discover -s tests -v` and
  `python3 scripts/learning/assets.py verify` for the foundation changes.
- `src/learning` uses only the standard library. Never import the legacy jobs
  to use event storage. SQLite `executescript` commits implicitly: migrations
  execute complete statements inside an explicit transaction instead.
- Timers require explicit start, stop and confirmation. Open/unconfirmed intervals
  have unavailable elapsed evidence; never infer time from submission timestamps.
- Existing `docs/` is ignored; put new governing SDD/ADRs under `specs/`.
  `improve.md` and `data/raw/leetcode_slugs.txt` were absent in this checkout.
- Do not regenerate asset manifests automatically; a mismatch needs review and a
  new version. Historical rights and precise embedding-run provenance are unknown.
- Private event databases/exports belong outside release/static assets. Local
  learner IDs are partitions, not authentication; Access/D1 remains a release gate.


## Hosted pilot and research integration (2026-10-16)

- M4-M7 local implementations now live under `curriculum/`, `hosted/`, and
  `specs/adaptive-learning/`. The earlier M1-M3 boundary above is historical;
  hosting, live Slack activation and optional model fitting remain gated.
- User explicitly accepts Luna drafts followed by independent Sol review. Record
  these as AI-reviewed algorithm judgments, never human gold or learning outcomes.
- Generate the allowlisted module with `python3 scripts/learning/build_hosted.py`.
  Only `hosted/public/` is a static asset directory; never upload repository roots.
- Use Node with built-in SQLite (26.8.1 verified), `npm ci` in hosted/, then
  `node --test hosted/test/*.test.mjs` from root. Loopback tests need socket access.
  Real-browser and workerd smoke commands are in deployment.md; Node tests alone
  are not browser or deployed-runtime proof.
- Wrangler 4.151.0 uses Miniflare 5; runtime tests use its exported
  convertV4MiniflareOptions helper. Set task-local Wrangler log/config paths when
  the global preferences folder is outside the sandbox. Do not repurpose HOME.
- Freeze at 05:40 KST; scheduled != posted != observed phone alert. Ambiguous
  accepted sends stay unknown and must not be blindly retried. Restores set a
  durable delivery-reconciliation pause; archives never restore a ready outbox.
- Codeforces keys use the native form codeforces:4A in both seed and imports.
  Confirmed timer intervals are distinct from manual contest minutes; do not add
  them into one elapsed total because the evidence can overlap.
- The initial 180 judgments have sparse query coverage: only six multi-candidate
  groups after graph-overlap exclusion. Conditional perfect scores cannot justify
  model promotion. No training labels or deployed quota/30-day proof exist yet.
- Preserve original asset hashes and unrelated Ralph changes. Optional encoders
  load only explicit hash-verified local directories; no server/download fallback.

## Optional OCI and vector diagnostics (2026-10-16)

- OCI is an optional stdlib offline batch worker, not a mission/Slack dependency.
  Package only the explicit allowlist in scripts/learning/package_oci.py. The
  systemd timer stays disabled pending actual Ubuntu cgroup/manual-run checks.
- Private vector experiments live in research/vector_geometry/, outside hosted
  release assets. Use its research requirements, never production dependencies.
  Notebook execution needs a local .venv kernel and loopback socket permission.
- Compare production raw squared-L2 separately from cosine compression geometry.
  Use Euclidean neighborhoods for t-SNE/UMAP coordinates: their arbitrary origin
  makes coordinate cosine similarity inappropriate. Projection plots and neighbor
  preservation do not establish algorithmic relationships or learner outcomes.
- Catalog collection is a separate stdlib CLI: scripts/learning/collect_catalogs.py.
  Both source responses and normalized metadata remain under private-learning/;
  public source/hash/missingness receipts are in assets/cross_platform_sources.v1.json.
- AtCoder ABC/ARC shared tasks use native arc IDs: legacy abc049_c, abc086_c,
  abc095_c map to arc065_a, arc089_a, arc096_a. Official-source mappings live in
  assets/problem_identity_aliases.v1.json. Offline coverage resolves them, but
  hosted URLs/import reconciliation remains a release gate; never silently rewrite
  frozen labels or historical learner keys. The catalog APIs contain no statements.
- MiniLM was explicitly authorized and executed on 2026-10-16. Its pinned
  allowlisted downloader is scripts/learning/download_minilm.py; research results
  are under research/minilm_pilot/. Keep weights outside Git and hosted assets.
  Model manifests must be outside the model directory because the offline backend
  verifies exact directory membership. Compare only equal-input embedding spaces.
- The first MiniLM run uses 90 authored English annotations, not the full API
  catalog/statements. Only three similarity query groups support conditional
  ranking; observed nearest neighbors include narrative/technique mismatches.
  No quality promotion or inference service follows from a successful model run.

## Statement sources and terms (2026-10-16)

- Read research/full_catalog/terms-review.v1.json before any statement or LeetCode
  fetch. LeetCode terms forbid crawling/scraping and robots.txt disallows /graphql
  and /api/; the stored 3,053 LeetCode statements came from /graphql. Codeforces
  robots.txt limits scraping tools to /api/ (no statements there). AtCoder task
  pages are robots-allowed but opt out of AI-training crawlers. No collection is
  approved; the CF/AtCoder catalogs have titles/metadata only.
- 2026-10-16 decision: all statement collection rejected/skipped. Stored LeetCode
  `content`/`content_clean` and vectors derived from them are quarantined: keep the
  files byte-identical (production still uses them) but never read them in new
  research. Full-catalog work follows specs/adaptive-learning/full-catalog-similarity.md.
- Full-catalog Phase 2: `python3 research/full_catalog/catalog.py --private-output
  private-learning/full-catalog/<new> --receipt <new>.json`. Outputs refuse to
  overwrite, and the script fails if input hashes differ from audit.json. LeetCode
  statement fields are dropped while parsing via object_pairs_hook. Run its tests with
  `python3 -m unittest discover -s research/full_catalog -v`. Kenkoooo AtCoder task IDs
  are already unique across reused contests (ADT and ABC/ARC share task IDs), so
  contest reuse is not a duplicate.
- research/full_catalog/retrieve.py needs numpy, so run its tests with .venv/bin/python.
  numpy 2.1.3 + Accelerate emits spurious divide/overflow RuntimeWarnings on finite
  float32 matmul. dense_cosine suppresses them with errstate, and topk rejects
  non-finite blocks. Do not "fix" this by switching to float64.
- Phase 3-5 methods must take rows and text from research/full_catalog/inputs.py
  (variants A1, A1-all, A2) so that inputs stay matched. baselines.py imports its
  sibling modules by name, so run it as a script or through discover -s research/full_catalog.
- research/full_catalog/embed.py loads MiniLM only through the hash-verified
  LocalEmbeddingBackend. Its outputs are private-learning/full-catalog/embeddings-v1/.
  CPU with 1 thread is bitwise-reproducible here. Titles are short (max 48 tokens
  in A2), so B/C statement variants stay unmeasured by decision D1, not by truncation.
- The judged pool (research/full_catalog/pool.v1.json) is frozen. Never regenerate it
  after labeling starts. A new pool needs a new version, and labels must reference
  pair_id. Unjudged pairs are not negatives.
- Labeling via subagents: label validators must reject empty rationales for labeled
  pairs (v1 did not). Self-reported `basis: recognized_problem` is unreliable across
  agents. `deceptive` is the least reliable field; adjudicators needed a stricter
  definition. Agents can misread batch fields, so spot-check their claims against the files.

## Statement-first research priority (2026-10-16, supersedes next-work priority)

- Follow specs/statement-similarity/ for new research. A1/A2 title/tag runs and
  3,647 pair records remain frozen preliminary baselines; do not optimize those
  baselines or start PCA/SVD before a permitted full-statement 384D baseline.
- Availability/feasibility report: research/statements/AVAILABILITY-AND-FEASIBILITY.md.
  The deterministic audit inspects aggregate availability only, including stored
  quarantined content. It does not release that content for new retrieval work.
- Current acquisition decisions are in research/statements/acquisition-review.v2.json.
  Prior terms-review.v1.json stays historical and byte-identical. Do not equate
  robots allowance, model-inference use or a code-repository license with verified
  rights to collect/reuse every statement. CodeContests explicitly licenses non-code
  CC BY 4.0 but also retains separate third-party conditions; preserve both facts.
- The frozen full-catalog labels contain 3,647 distinct pair records: 2,886 non-null
  judgments and 761 abstentions. Reuse their evidence/basis faithfully; do not treat
  abstentions as negatives or relabel title-pooled judgments as statement-grounded.

## Full-catalog research continuation (2026-10-16)

- `specs/algorithm-system/` extends statement-first research. Publisher non-code
  dataset imports are admitted privately under the scoped decision in
  `research/algorithm_system/CHECKPOINT2.md`; the earlier acquisition review stays
  historical. LeetCode text remains quarantined. No direct platform scraping or
  public statement redistribution is authorized by that decision.
- Hugging Face CodeContests revision `802411c3010cb00d1b05bad57ca77365a3c699d6`
  uses source ClassLabel AtCoder=5/Aizu=6, unlike the original protobuf 6/7.
  Bare Parquet int64 values require the pinned, hash-checked dataset_infos.json;
  never infer the enum from prose or reuse the protobuf mapping.
- `research/algorithm_system/verify_capture.py` reconstructs selected statement
  columns from hashed local HTTP ranges without network. Whole Parquet shards,
  solution archives, private tests, and generated tests are unnecessary.
- The CodeNet small description archive lacks metadata/native URLs in this
  snapshot. Its pNNNNN IDs remain unresolved, not guessed AtCoder task IDs.
  AtCoder CodeContests rows likewise retain dataset identities when IDs are absent.
- Similarity judgments are not difficulty-ordering labels. Native percentile
  alignment is provisional, not an equality of cross-platform learner effort.
- Run the new scientific suite with `.venv/bin/python -m unittest discover -s
  research/algorithm_system -v`. Compression requires the complete hash-bound
  384D chunk baseline. Neighbor preservation alone cannot promote a model.

## Local verification pilot (2026-10-16)

- `research/verification_workbench/server.py serve` opens the loopback-only
  identity/relationship workbench; its frozen 40-pair pilot and SQLite evidence
  live under `private-learning/verification-workbench/pilot-v1/`.
- Role and independence are self-attestations, not authenticated human proof.
  Two independent human reviews plus a third adjudicator gate accepted overlays;
  AI reviews, title candidates and direct assertions never resolve source IDs.
- Reveal mode logs exposure and blocks later independent claims for that pair.
  Browser forms clear across reviewers/problems; stale responses are discarded.
  Corrections suspend accepted overlays that referenced an earlier review.
- Workbench tests use authored fixtures: `python3 -m unittest discover -s
  research/verification_workbench -v`; two tests require loopback socket access.
  Browser test writes MUST use the disposable synthetic server, not the real pilot.
  Real statement screenshots and exports stay private. Production import is absent.
- User-requested AI-first reviews remain role=ai/independent=false; their private
  40-pair report is under `private-learning/verification-workbench/ai-review-v1/`.
  View AI draft exposes only AI+own records, keeps ranks hidden, and persists an
  exposure lock; assisted human judgments are allowed but cannot satisfy the
  independent-human verification gates. AI-only import validates exact pilot
  coverage before writing and is resumable/idempotent. Never call these drafts gold.

## Bounded identity investigation (2026-10-17)

- Preserve numeric problem-title prefixes when stripping publisher/contest
  wrappers: “3 Steps” and “01 Unbalanced” are complete titles, not task labels.
- `research/identity_resolution/` validates manually authored primary-source
  comparison notes and emits only AI identity proposals. Cached web observations
  are not live response receipts or bulk statement reuse permission. Native task
  IDs may differ from displayed task letters; compare canonical links.
- Identity agreement does not establish full statement fidelity. Keep language
  discrepancies and inaccessible sources explicit; never infer an unavailable
  statement from a title or editorial alone.

## Unified similarity research (2026-10-17)

- `research/unified_map/` links all native catalog nodes and a separate admitted
  statement source index. Semantic title neighbors, statement neighbors and
  algorithm-tag proxies are separate channels; missing tags remain unknown.
- Use package-qualified imports for new research helpers. Inserting another
  research directory at sys.path[0] can make unittest discover the wrong test module.
- Hybrid RRF MUST NOT grant algorithm evidence to zero-Jaccard rank ties. Inspect
  component support; tag consistency of a tag ranker is circular, not validation.
- Native map-v2 and reduction-v2 are the reviewed runs; v1 remains preliminary.
  2D UMAP/t-SNE use a disclosed stratified sample, not all-catalog relevance proof.

## Full-catalog interactive map (2026-10-17)

- Spec: specs/unified-similarity-map/interactive-map.md. Code: research/unified_map/full_{embed,project,quality,html,verify}.py
  and full_map_template.html. Outputs are private (private-learning/unified-map/full-v1/), including interactive.html,
  which embeds titles/tags/URLs. Never commit it; only receipts under research/unified_map/full-receipts/ and
  point-only figures are committed.
- research/full_catalog/retrieve.py is hash-pinned by research/unified_map/PRESERVATION.v1.json. Do not edit it or other pinned
  files. Run `.venv/bin/python -m research.unified_map.verify private-learning/unified-map/map-v2` before finishing; put new
  retrieval modes in new modules (the any-platform mode lives in full_quality.py).
- scikit-learn t-SNE with init="pca" ignores random_state: seed repeats give a bit-identical layout. Measure stability with
  --tsne-init random. UMAP/t-SNE run with one thread for reproducibility (t-SNE ~3 min, UMAP ~30 s for 24,297 nodes).
- Tokenizer unknown-share alone overstates embedding reliability: Cyrillic titles tokenize into known word pieces. A non-Latin
  title is capped at "degraded". Latin-script romanized Japanese titles cannot be detected and stay "reliable".
- Compare neighbor tag agreement with a platform-matched random baseline, not random pairs: cross-platform concept sets differ,
  so an all-pairs baseline is unfair.
- Port 8765 is the verification workbench (private statements). Preview static HTML on another loopback port from a directory
  that contains only the HTML; never stop servers you did not start.
- Tests: `.venv/bin/python -m unittest research.unified_map.test_full_embed research.unified_map.test_full_quality
  research.unified_map.test_full_html`. Spurious Accelerate FP warnings appear in float32 matmuls; wrap test calls in np.errstate.

## Language-bias study (2026-10-17)

- Spec specs/unified-similarity-map/language-bias.md; code and report in research/language_bias/. The selected representation (C2) embeds
  English-translated titles with the same pinned MiniLM; full-v2 holds its vectors, neighbors, maps and page. full-v1 is preserved: do not
  modify it (research.language_bias.verify proves its hashes). Never judge a language fix by low homophily or platform mixing alone: random vectors
  score best on both. Use neighbor consistency, translation retrieval, English-quality retention and the random control.
- Models live under private-learning/models/ with the manifest OUTSIDE the directory; the verifier checks exact membership, so move any CLI
  `.cache` folder out (huggingface-cli leaves one inside --local-dir). The translator ships only pickle weights: it is quarantined and converted once
  with torch weights-only loading and the network blocked; never load that .bin any other way. Only user-approved models may be fetched, and a
  downloader must refuse everything else; get approval before any package install (sentencepiece 0.2.1 is the only addition, research .venv only).
- Marian (opus-mt) needs sentencepiece; its output ends with a period that real titles lack, so the translator strips it. The model reads Japanese
  only: Russian and Korean titles stay untranslated and still cluster by language. Machine translation can fail badly on numbered tutorial titles and
  place names; those nodes are tiered degraded.
- Helper methods named `run` in a unittest.TestCase shadow TestCase.run and break the suite; name them otherwise. scikit-learn's centroid-style
  fixes need a minimum group size: a one-node group's centroid is the node itself (zero vector).
- An interrupted tool call may already have run part of its command (an edit and a delete ran before the interrupt): check the state afterwards.

## Mastery system (implemented locally, 2026-10-17)

- Spec: specs/mastery-system/ (design.md section 15 lists deviations; runbook.md has commands). Code:
  hosted/mastery/*.mjs, hosted/cli.mjs, migrations 007-008, research/mastery_catalog/. ADR 003 accepted:
  local SQLite is canonical, the JS core is the only learner-state engine, hosting is optional.
- Tests: `node --test hosted/test/*.test.mjs`; `.venv/bin/python -m unittest research.mastery_catalog.test_build
  research.mastery_catalog.test_refresh research.mastery_catalog.test_archive research.mastery_catalog.test_relatedness`.
  The CLI test builds a fixture catalog with system python3 (`-I`) and needs git.
- hosted/core/events.mjs requires exact payload fields and events has CHECK constraints: add evidence as
  append-only side tables (hosted/mastery/side-ledger.mjs contract), read only through ledger.mjs. Snapshots
  store per-table sequence high-water marks. Node-only modules (git-ingest, catalog, views, cli) must never be
  imported by core/ or worker.mjs; `wrangler deploy --dry-run` confirms the bundle.
- AtCoder Problems: problem-models `difficulty` is raw (unclipped); all models share one discrimination;
  agc010_a = -10000 is a sentinel. problems.json points shared ABC tasks at the latest ADT reuse, so derive
  canonical contests from contest-problem.json. MiniLM batch padding makes identical texts differ by ~1e-7:
  reuse vectors by (problem_key, text) to keep neighbors byte-identical across refreshes.
- Ingestion never rewrites a stored outcome (reported_completion stays reported_completion). Since policy v3 (ADR 005, 2026-10-28)
  the policy counts every completion as solved; see "Solved-only outcomes" below.
  Difficulty stays per platform; selection uses mean minus one SD while history is sparse.
- AI judgments: two different model roles, blind; agreement -> ai_reviewed, else abstained; never gold.
  Two Claude judges share training memory, so their agreement is not independent verification.
- Catalog versions are immutable: catalog-v2 is superseded by catalog-v3 (ADT URL fix). Do not edit or
  delete private-learning/mastery/catalog-v*; make a new version.

## AtCoder concept classification (2026-10-17)

- Spec specs/mastery-system/atcoder-concepts.md, result atcoder-concepts-report.md. Only a 450-problem random sample was judged (cost: about
  3,800 tokens per labeled problem, 5M projected for ABC/ARC/AGC); other problems stay unclassified. Judge batches mix blind decoys (invented
  tasks, swapped titles); judge files and truth.json live under private-learning/mastery/judgments/ and stay out of Git.
- ai_reviewed needs both judges plus at least one `recognized`; title-inference-only agreement is `proposed` (untrusted). Only the `confirm`
  command writes approved/rejected. catalog-v4 is the offline replay with the new rows; never edit v1-v3.
- related-v1 has a semantic-only fallback: unclassified candidates from a title channel, reliable titles on both sides, label
  "Semantic title neighbor...", -1.5 penalty, never credited to a concept. Before estimating cost for a subagent run, state the agent count and
  total tokens up front: the user stopped a 10-agent wave that had been described only as "5M tokens".

## Current direction (2026-10-18)

- Baekjoon has shut down: the legacy batch pipeline (`src/main/generate_index_map.py`, Docker, S3, OpenAI, Slack) has no input and no
  automation. Its code stays as history; do not revive it or add paid-API or cloud dependencies for it.
- The product direction is the local-first mastery system (`hosted/`, `specs/mastery-system/`, ADR 003): LeetCode, Codeforces and AtCoder
  only, no paid APIs, no hosting. Minimize cost and storage; ask before adding anything that costs money or large disk.
- Solutions live in the separate private repo `algorithm-solution` (layout `leetcode/`, `codeforces/`, `atcoder/`; commit trailers
  `Problem:`, `Outcome:`, `Hints:`). Its own Actions only check naming and trailers. Pushing never produces recommendations: run
  `node hosted/cli.mjs repo ingest` and `plan` locally.
- This repository is not a portfolio or a commercial service. Do not add features for external users.

## Local mastery journal (2026-10-18)

- Local runtime/CLI share repository-root defaults for learner SQLite and catalog
  paths. Never initialize or migrate the real learner DB while testing; use
  :memory: or a temporary path and synthetic learner IDs.
- `mastery/journal.mjs` composes the existing planner and versioned progression;
  journal handlers are injected only by the local runtime, not into Worker code.
  Core review replay schedules every attempted problem; failures reset to next-day priority.
  (Policy v3: hints are not tracked and imported completion counts as solved; ADR 005.)
- Journal snapshots fingerprint relevant evidence/config/code. Timer-only events
  must not rotate the current recommendation. Corrections remain visible as
  superseded history; do not emit initial unknown-state changes for every concept.
- Backup-v2 recommendation archives are inert `archive-...` snapshot rows with
  wrapper metadata. Unwrap them for historical display only; never activate them
  as current plans. Export flattens wrappers, preventing recursive nesting.
- Novel transfer cannot reuse an attempted canonical key; known-problem review is
  retention evidence, not a new transfer. Hard evidence stays separate from Easy
  and Medium gates. Metadata variation is a proxy, not proven algorithm equivalence.
- Browser writes require an isolated `browser-smoke` learner and disposable store.
  Set SMOKE_OUTPUT_DIR to a fresh private path; preserve earlier evidence receipts.


## Daily study-log ingestion

- Platform/date Markdown and `daily/YY-MM-DD.md` are multi-problem tables, never path-derived problem IDs.
- Resolve plain titles exactly within the platform or validate supported URLs against the active catalog; ambiguous/missing identities are flagged, not guessed.
- O/X are explicit row evidence. `-` creates no attempt. Date/problem tokens prevent memo edits from counting twice; outcome edits are flagged conflicts.
- Use `repo ingest` then `journal` to see evidence-refreshed recommendations; `plan` is frozen per date. Test with temporary DBs, preserving the real learner DB.

## Dual-model recommendation explanations

- `hosted/mastery/explanations.mjs` is a Node-only presentation layer. Production `explain` concurrently requests Gemini and NVIDIA from one frozen input/prompt; journal/web reads never dispatch.
- Keep planner, journal, evidence and immutable snapshots unchanged. AI prose is not verified grounding; output checks cannot establish truth.
- `explanation-store.mjs` owns a separate, append-only private SQLite research ledger beside the learner DB. It is durable, not disposable cache; back it up separately with SQLite online backup (WAL-aware). v1 JSON is preserved, never promoted into attempt evidence.
- Cache per learner/input/provider/model/prompt/parameters. Both provider roles remain visible when configuration is missing. Explicit `--retry-failed` appends lineage and preserves successful peers; cache hits never add tokens/requests.
- Preserve provider-native usage and null missingness. Estimated cost uses explicit versioned rates; actual costs require a separately sourced receipt. Unknown/crashed attempts are not free or confirmed failures.
- `explanation-usage` reports aggregates; `explanation-records` is private. `explanation-otel` is an offline redacted projection, with no external export by default. No LangChain/Grafana/service dependencies.
- Browser smoke: `node hosted/explanation-browser-smoke.mjs` uses isolated synthetic learners, mocked APIs, its own Chrome profile and temporary loopback service. It proves UI behavior only. Never inspect or print real keys for validation.
- Nemotron 3.5 Lightning short prose uses model-specific `chat_template_kwargs.enable_thinking=false`; otherwise its 512-token cap can end with `finish_reason:length`. Persist this option in parameters/cache identity and leave Gemini caches intact. A successful model-list GET does not prove inference permission; classify HTTP errors without exposing raw bodies.

## Calendar workspace (2026-10-20)

- `hosted/public/workspace-data.mjs` is presentation-only; actual attempts use `occurred_at`, plans use their explicit local date. Exclude superseded records from aggregates while retaining them in problem detail history.
- Policy timezone is fixed at Asia/Seoul; no user timezone preference exists. Calendar forms explicitly convert KST and preserve unchanged correction timestamps byte-for-byte.
- `local.mjs` has a static allowlist: include new browser modules there. Keep Node-only learner/policy modules out of browser imports.
- New UI smoke: `SMOKE_OUTPUT_DIR=<fresh private directory> node hosted/workspace-browser-smoke.mjs` uses disposable synthetic learners and mocked explanation providers. It does not establish live contest sync or paid inference availability.
- Contest source availability follows `specs/contest-discovery/` and the actual capability flag; distinguish official sources, the LeetCode community mirror and manual records. Never infer schedules or eligibility from titles.

- Calendar date clicks select the agenda; overflow opens a day dialog. Both modes cap entries at three. Group same-day/problem recommendation revisions for presentation, preserving reasons and current cached explanations in details.
- Preserve supplied `contest.recommendation` evidence; the UI fallback must not infer skill from title keywords or platform activity counts. Keep source/sync state separate from suitability.
- Explicit dated reviews use optional `preferences.review_targets` and catalog-enriched `journal.review_selections`; `review add|remove` never creates attempts. Reload older local readers before writing the additive preference key. Uncommitted Markdown with blank outcomes remains outside committed-Git solve ingestion.

## Daily duration units (2026-10-23)

- Daily summary `시간` accepts integer `m`/`s` up to 24h, blank/`-` unavailable, explicit zero preserved. Seconds use optional JSON `self_seconds`; never round into integer `self_minutes` or set missing minutes to zero. Minute/second values are mutually exclusive.
- Keep optional seconds in unified ledger/journal, corrected history and backups. Weekly exact `self_reported_seconds_total` includes both units; legacy `self_reported_minutes` remains the minute-unit subtotal. Confirmed timer evidence stays separate.
- Already-running local servers must restart to pick up new event/ledger validators. The general browser smoke contains fixed October contest fixtures that expire; use targeted duration checks or revise fixtures before claiming a complete browser pass.

## Weekly LeetCode mock contest (2026-10-28)

- Spec specs/mock-contest/, code hosted/mastery/mock-contest.mjs (pure, Worker-safe), CLI `mock-contest create|show`. Stored as
  `recommendation_snapshots` kind `mock_contest`: one immutable set per date, no attempts. Policy `mock-contest-v2` has a date and a
  solving time (default 120m) only. Mock contests never read or adjust to official contest schedules (user decision; v1 slot placement removed).
- No title text or title embeddings in selection (user decision: inaccurate). Ranking uses trusted weak concepts, a tag-set diversity rule
  and a seeded FNV hash. Composition targets 60% conservative expected solves: an unvalidated heuristic.
- catalog-v4 has 61 eligible LeetCode rows with no native tags (JavaScript 30-days, Pandas, newer SQL). They are not algorithm problems;
  the mock-contest pool requires a native tag. The daily planner does not filter them yet.
- The journal reads every snapshot kind as plan history and iterates `slots`. Any new snapshot kind without `slots` must be filtered out
  there, or `/api/journal` breaks. Mock contests feed daily plans only as exposures (`mockExposurePlans`).
- Backup import wraps every snapshot as an inert `archive-...` row. Restored mock contests show as archived history and are not exposures.
- Daily recommendations are mostly `journal-*` snapshot kinds, not `plan`. Code that collects exposures must read every kind
  (`snapshotExposures`), not only `kind='plan'`.

## Solved-only outcomes (2026-10-28, ADR 005)

- User decision: whether help was used is a matter of personal honesty and is not tracked. Policy `mastery-policy-v3` treats
  `independent`, `assisted` and `reported_completion` as solved, `failed` as unsolved. Daily-log `1` therefore counts.
- The projection lives only in `attemptStream` (hosted/mastery/ledger.mjs): consumers see `outcome:'independent', hints:false` for
  solved; the stored outcome stays in `recorded_outcome`/`recorded_hints`. Never re-read raw outcomes in policy code, and never
  rewrite stored rows. Ingestion tests assert `recorded_*`.
- UI and forms offer solved / unsolved only (contract test in ui-contract.test.mjs). Validators still require `hints:false` for a
  stored `independent` event, so forms send `false` for solved and `null` for unsolved.
- Changing the policy rotates journal plans (code hash). Run `explain` again for new recommendations.


## Plan carry-over in the calendar (2026-10-28)

- Unsolved system plans from the last 7 days show on today as carried (`workspaceEvents` in hosted/public/workspace-data.mjs). This is
  presentation only: never write carried items into snapshots or feed them to the planner. Any attempt on/after the plan date ends it.
- `workspaceEvents` takes `{today}`; tests with past-dated fixtures must pass it, or results change with the real date.
- Carried plans are explained from their stored snapshot: `withExplanations(journal,explanations,db)` and `explain` pass `snapshotPlan`.
  `pendingPlanEntries` in explanations.mjs mirrors the calendar rule; change both together. Running servers need a restart to pick it up.
- hosted/workspace-browser-smoke.mjs uses fixture dates relative to today (next Wed/Fri/Sun, yesterday); never reintroduce fixed dates.

## Comparison viewer maps (2026-10-28)

- research/recommendation_inputs/maps.py draws t-SNE maps in the comparison viewer (port 8789, `comparison-viewer` in .claude/launch.json).
  A method gets a map only if recomputing top-k from its vectors reproduces the stored ranking exactly; rows follow strategy_all's key order.
- Map positions are layout only (about 50% top-10 preservation). Numbers and lines come from the stored 384D ranking. Only curriculum-idea
  scopes have vectors; the LeetCode title/tag and publisher scopes show no map. The title method is a baseline, not a recommendation input.
- hosted/local-comparison.mjs allowlists `/comparison/api/map`; the 8787 server must restart to proxy it.
- Maps have a same-platform mode (`/api/map?platform=`): layout and top-10 are recomputed within that platform from the verified vectors.

## Research library (2026-10-28)

- `research/library.v1.json` is the curated catalog (folders → files) behind the "연구 자료" view; `hosted/local-library.mjs` serves it
  locally only, by catalog id (never a requested path), from research/ or private-learning/, refusing symlinks that leave the repo.
- HTML files are served with a `sandbox allow-scripts` CSP and shown in a sandboxed iframe: no network (notebook MathJax won't render).
  The desktop Browser pane's screenshots show sandboxed iframes as blank; check them in headless Chrome instead.
- Add new figures by editing the catalog (title, note on what it does not show, source report). Private files stay uncommitted.

## Participation-first contests (2026-10-28)

- User decision: contests show no suitability verdict and no stale-cache badge. `contestSuitability` and server `contest.recommendation`
  remain in data/tests but are not displayed. Provenance and stale fallback live under the collapsed "일정 출처" section of the details.

## Desktop readiness (2026-10-29)

- User desktop-product instructions authorize Windows/macOS distribution direction; the older personal-only audience is historical. Framework and catalog distribution remain undecided. See specs/desktop-readiness/status.md and data-safety.md.
- Convert filesystem file URLs with fileURLToPath/pathToFileURL. CLI roots and catalog SQLite URI paths have regressions for spaces, Korean, # and %; real Windows execution remains unverified.
- hosted/runtime/database.mjs owns the SQLite adapter. Test helpers re-export it for compatibility. Existing nonempty DBs receive a SQLite VACUUM INTO snapshot before pending migrations; newer schemas and receipt drift are rejected. Stop concurrent writers before upgrading. No automatic relocation, restore or backup deletion.
- Full Node tests require loopback sockets; EPERM in restricted runs is environmental. Final 2026-10-29 regression: Node 254/254 and Python 74/74 passed; no real learner DB was used.

## Recommendation audit and adaptive progression (2026-10-29)

- Audit: research/recommendation_audit/REPORT.md (scripts + receipts; reads the private catalog read-only). The journal shows planner reviews
  plus one progression item; v1 Easy/Medium picks are alphabetical, and several concepts stall on catalog coverage.
- Daily Markdown rows always ingest as `practice`, so anything needing a declared activity type (v1 transfer/retention, review-stage growth)
  is unreachable from the user's real input. The 복습 column is a user review request, never evidence; memos are never evidence.
- Accepted direction: progression-v2 (ADR 006, specs/adaptive-progression/). Implement T1-T9 in order behind journal option
  `progressionVersion` (default v2 after T8 behavioral AI-review sign-off; v1 selectable); do not modify core/policy.mjs, calibration.mjs or progression.mjs (v1 must stay
  byte-identical). D2-D4, D6 and the 14-day follow-up are provisional: never describe them as optimised.
- No independent relevance labels and only 9 real attempts exist: do not report precision, calibration or learning effects as measured.

- Progression-v2 T1-T9 implemented 2026-10-29. Fresh concepts retain authored priors (user clarification); updates begin after own LC evidence. Never use future ratings in evidence-prefix transitions. Review and progression lanes are independent; CLI journal --progression v1 and local PROGRESSION_VERSION=v1 preserve old output. T8 is synthetic behavioral evidence, not human gold, calibration or learning benefit.
