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

Algorithm RAG Engine recommends English LeetCode follow-up problems from Korean
Baekjoon study notes. The current production path is a batch pipeline:

1. Detect changed Baekjoon Markdown/MDX study files.
2. Parse problem metadata.
3. Embed the source problem with OpenAI `text-embedding-3-small`.
4. Search the local NumPy LeetCode index.
5. Generate a Korean recommendation reason with GPT-4o-mini.
6. Write `artifacts/recommendation_map.json`.
7. Send the result to Slack through a lightweight Slack job.

Read the docs before making architectural changes:

- `README.md` - project overview and current pipeline narrative.
- `STAR.md` - portfolio summary, accurate caveats, file inventory.
- `improve.md` - production improvement roadmap.
- `docs/leetcode_update.md` - LeetCode data update strategy.
- `docs/embedding_visualization.md` - embedding visualization notes.
- `Evaluation_100/docs/*.md` - evaluation and benchmark context.

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

Treat these as the current production entry points:

- `src/main/generate_index_map.py` - heavy recommendation generation job.
- `src/main/slack_bot_daily_review.py` - light Slack delivery job.
- `Dockerfile` - multi-stage heavy/light image.
- `docker-compose.yml` - local/container orchestration.
- `.github/workflows/daily_algorithm_pipeline.yml` - pipeline orchestration.

Treat these as legacy or experimental unless the task explicitly targets them:

- `src/rag/faiss_recommendation_engine.py` - older FAISS-based recommender.
- `scripts/preprocessing/build_faiss_index.py`
- `scripts/preprocessing/build_production_faiss_index.py`
- `Evaluation_100/**` - evaluation research, benchmark scripts, notebooks, images.

Do not claim that `Evaluation_100` is the exact deployed production code path.
It is useful for ideas such as hybrid scoring and GPT reranking, but production
currently runs through `src/main/generate_index_map.py`.

## Current Important Caveats

- The main production search path uses NumPy, not FAISS.
- FAISS files and scripts remain for migration history and experiments.
- The workflow schedule is documented historically, but the checked-in
  `.github/workflows/daily_algorithm_pipeline.yml` currently uses manual
  `workflow_dispatch`; verify before saying it runs daily.
- `generate_index_map.py` relies on OpenAI API access and a NumPy index.
- `artifacts/recommendation_map.json` is generated output and may contain sample
  recommendations; do not use it as proof of quality without evaluation.
- LangSmith and Slack feedback helper modules exist, but feedback is not yet in
  the main production ranking loop.

## Data and Artifacts

Important paths:

- `data/raw/leetcode_raw_data.jsonl` - raw LeetCode records.
- `data/processed/leetcode_processed.jsonl` - processed LeetCode records.
- `data/raw/leetcode_slugs.txt` - slug manifest for duplicate detection.
- `indexes/numpy_leetcode/vectors.npy` - current production vector index.
- `indexes/numpy_leetcode/metadata.json` - current production metadata.
- `artifacts/recommendation_map.json` - generated recommendation map.
- `Evaluation_100/data/ground_truth_v2.json` - 100-pair evaluation ground truth.

Large data, indexes, generated images, and artifacts may be intentionally stored
outside Git or synced through S3 in workflows. Do not delete or regenerate them
unless the user asks.

## Recommended Development Direction

The current documented roadmap prioritizes recommendation quality before product
UI:

1. Refactor the recommender into reusable production modules under `src/rag/`.
2. Parse Baekjoon MDX into structured algorithm profiles.
3. Add hybrid scoring with explicit score components.
4. Add GPT reranking as a decision layer.
5. Make evaluation call the same recommender code path used by production.
6. Add API, database, notification routing, feedback loop, and observability.

When making changes, keep new code aligned with that direction instead of adding
more one-off scripts.

The final platform direction is:

> A production-style personalized algorithm training platform with validated
> AI-generated problems, secure sandboxed judging, concept-level
> personalization, spaced repetition learning, explainable recommendations, and
> feedback-driven adaptation.

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

Install production-heavy dependencies:

```bash
pip install -r requirements-heavy.txt
```

Install light Slack dependencies:

```bash
pip install -r requirements-light.txt
```

Run the heavy generator locally, assuming required environment variables and
index paths are configured:

```bash
python3 src/main/generate_index_map.py
```

Run the Slack delivery job locally:

```bash
python3 src/main/slack_bot_daily_review.py
```

Check for new LeetCode problems without GraphQL:

```bash
python3 scripts/collection/leetcode_check_new_fast.py
```

Collect LeetCode details when new problems exist:

```bash
python3 scripts/collection/leetcode_data_collection.py
```

Run retrieval evaluation:

```bash
python3 scripts/evaluation/evaluate_retrieval.py
```

## Testing and Verification

There is no single documented test command for the whole repository. Choose
verification based on the change:

- Parser/scoring changes: add or run focused unit-style checks if available.
- Generator changes: verify JSON output shape and no missing required fields.
- Slack changes: validate Block Kit payload shape without sending live messages
  unless the user approves real delivery.
- Workflow/Docker changes: inspect commands and paths carefully; local execution
  may require API keys, S3 data, or Docker credentials.
- Evaluation changes: report Recall@K/MRR changes against the documented
  100-pair ground truth when possible.

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

- "Current production" means the NumPy-based heavy/light Slack pipeline.
- "Evaluation_100" means experimental benchmark work.
- "FAISS" means legacy or alternate indexing unless a task explicitly revives it.
- "Daily schedule" should be described as disabled/manual unless the workflow is
  changed and verified.

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
