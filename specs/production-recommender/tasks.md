# Production Recommender Tasks

This task list implements the production recommender refactor from
`requirements.md` and `design.md`.

Rules:

- Complete tasks sequentially.
- Keep each task independently testable.
- Prefer offline fixture tests before production behavior changes.
- Preserve the existing `recommendation_map.json` contract unless a task
  explicitly updates and validates all consumers.
- Do not run live OpenAI, Slack, S3, Docker push, or network-dependent checks
  unless the task explicitly requires it and approval is granted.

## Phase 1: Validation Scaffolding

- [ ] T001: Add test fixture directory for production recommender examples.
  - Scope: Create small Markdown/MDX fixtures and a sample Slack-compatible
    recommendation artifact.
  - Validation: `test -d tests/fixtures/production_recommender`

- [ ] T002: Add a minimal offline validation command for recommender-related
  tests.
  - Scope: Document or add a command that can run without OpenAI, Slack, S3, or
    Docker.
  - Validation: Run the documented command and confirm it exits successfully.

- [ ] T003: Add artifact-shape validation for the current
  `recommendation_map.json` contract.
  - Scope: Validate top-level problem entries and recommendation item fields:
    `id`, `title`, `tags`, `recommendations`, `difficulty`, `similarity`, and
    `ai_comment`.
  - Validation: Run the artifact-shape validation against a fixture artifact.

## Phase 2: Schemas

- [ ] T004: Add `src/rag/schemas.py` with standard-library typed schemas.
  - Scope: Define source problem, index record, score breakdown, recommendation
    item, and recommendation result contracts.
  - Validation: Run schema serialization tests proving current artifact fields
    are preserved.

- [ ] T005: Add schema tests for additive score fields.
  - Scope: Verify fields such as `dense_score`, `tag_score`, `final_score`, and
    `reason_ko` can be represented without breaking current JSON consumers.
  - Validation: Run schema tests for backward-compatible JSON output.

## Phase 3: Source Discovery and Parsing

- [ ] T006: Extract changed-file discovery into `src/rag/problem_parser.py`
  without changing behavior.
  - Scope: Move or wrap current GitPython discovery and all-files fallback.
  - Validation: Run fixture tests for changed-file and fallback discovery.

- [ ] T007: Extract Markdown/MDX frontmatter parsing into
  `src/rag/problem_parser.py`.
  - Scope: Preserve current `id`, `title`, `tags`, exclusion, and
    `embedding_text` behavior.
  - Validation: Run parser tests for valid frontmatter, missing `id`, excluded
    IDs, and tag parsing.

- [ ] T008: Add parser tests for deterministic behavior on malformed files.
  - Scope: Verify malformed or missing frontmatter files are skipped without
    crashing.
  - Validation: Run parser edge-case tests.

## Phase 4: NumPy Retrieval

- [ ] T009: Extract NumPy index loading into `src/rag/retriever.py`.
  - Scope: Load `vectors.npy` and `metadata.json`; validate both exist and have
    compatible lengths.
  - Validation: Run retriever tests using a tiny fixture index.

- [ ] T010: Extract dense candidate search into `src/rag/retriever.py`.
  - Scope: Preserve squared L2 behavior and expose raw dense distance.
  - Validation: Run deterministic nearest-neighbor tests against a tiny fixture
    index.

- [ ] T011: Add deterministic retrieval mode.
  - Scope: Provide sorted top-k retrieval for tests and future evaluation while
    preserving configurable random-pool behavior for production if enabled.
  - Validation: Run repeated retrieval tests and confirm identical output.

## Phase 5: Scoring and Filtering

- [ ] T012: Extract duplicate and near-title filtering into `src/rag/scoring.py`.
  - Scope: Preserve current title-cleaning behavior and duplicate title removal.
  - Validation: Run scoring tests for exact title, near title, and duplicate
    recommendation cases.

- [ ] T013: Add inspectable score component structure.
  - Scope: Attach dense distance or dense score to recommendation items without
    removing the existing `similarity` field.
  - Validation: Run score component tests and artifact-shape validation.

- [ ] T014: Add extension points for future hybrid scoring.
  - Scope: Keep inactive placeholders or pure functions for tag, pattern,
    difficulty, and feedback scores without changing production ranking.
  - Validation: Run unit tests proving default final ordering remains unchanged.

## Phase 6: Explanation

- [ ] T015: Extract explanation generation into `src/rag/explainer.py`.
  - Scope: Preserve GPT-4o-mini prompt behavior and the existing Korean fallback
    strings.
  - Validation: Run explainer tests with fake LLM clients and no OpenAI calls.

- [ ] T016: Add fallback tests for missing API key and LLM exceptions.
  - Scope: Verify recommendation generation can continue when explanations are
    unavailable.
  - Validation: Run offline fallback tests.

## Phase 7: Recommender Orchestration

- [ ] T017: Add `src/rag/recommender.py` orchestration without changing output
  schema.
  - Scope: Compose parser output, retriever candidates, scoring, and explainer
    into recommendation results.
  - Validation: Run end-to-end offline recommender test with fake embeddings and
    fixture index.

- [ ] T018: Add `recommend_many` support for multiple source problems.
  - Scope: Return a recommendation map keyed by source problem ID.
  - Validation: Run multi-source fixture test and artifact-shape validation.

- [ ] T019: Add deterministic mode to the recommender interface.
  - Scope: Thread deterministic retrieval behavior through the orchestration
    API.
  - Validation: Run repeated end-to-end fixture tests and confirm identical JSON.

## Phase 8: Heavy Job Integration

- [ ] T020: Convert `src/main/generate_index_map.py` into a thin wrapper around
  the production recommender.
  - Scope: Preserve environment variables, output path, artifact filename, exit
    codes, and current Slack-compatible JSON fields.
  - Validation: Run offline wrapper test or dry-run fixture validation without
    live OpenAI calls.

- [ ] T021: Add artifact compatibility validation for generated output.
  - Scope: Compare generated fixture output against required artifact fields.
  - Validation: Run artifact compatibility validation.

## Phase 9: Slack Compatibility

- [ ] T022: Add Slack formatter tests for the existing artifact contract.
  - Scope: Validate `DailyReviewBot.build_slack_block` using fixture artifacts
    without sending messages.
  - Validation: Run Slack formatter tests offline.

- [ ] T023: Add compatibility tests for additive recommendation fields.
  - Scope: Verify Slack ignores or safely handles optional fields such as
    `final_score`, `dense_score`, and `reason_ko`.
  - Validation: Run Slack formatter tests with additive-field fixtures.

## Phase 10: Evaluation Hook

- [ ] T024: Add a deterministic evaluation adapter for the production
  recommender.
  - Scope: Allow evaluation scripts to call the production recommender without
    using `Evaluation_100` as production code.
  - Validation: Run adapter tests with fixture data.

- [ ] T025: Document how to run Recall@K/MRR evaluation when retrieval behavior
  changes.
  - Scope: Add instructions for benchmark execution and required environment.
  - Validation: Confirm documentation names the ground truth file and
    deterministic mode.

## Phase 11: Observability and Documentation

- [ ] T026: Add parseable logs for recommender stages.
  - Scope: Log source count, parsed count, index path, model name, candidate
    count, scoring mode, deterministic mode, fallback count, and artifact path.
  - Validation: Run fixture path and assert expected log messages are emitted.

- [ ] T027: Update docs after behavior-preserving refactor is complete.
  - Scope: Update README or relevant docs to describe the module structure and
    clarify production vs `Evaluation_100`.
  - Validation: Review docs and run Ralph scaffold validation.

## Phase 12: Deferred Quality Improvements

- [ ] T028: Design hybrid scoring before changing production ranking.
  - Scope: Create or update a spec for dense, tag, pattern, difficulty, and
    feedback score weighting.
  - Validation: Design review only; no production ranking changes.

- [ ] T029: Design GPT reranking before adding it to production.
  - Scope: Specify strict JSON output, fallback behavior, cost controls, and
    evaluation gates.
  - Validation: Design review only; no production reranking changes.

- [ ] T030: Design structured source profiles before parsing full MDX bodies.
  - Scope: Specify deterministic extraction, optional LLM extraction, schema,
    fallback, and artifact storage.
  - Validation: Design review only; no production parser changes.
