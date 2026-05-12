# Production Recommender Requirements

## Purpose

This specification defines requirements for refactoring the current production
recommendation path into reusable modules without changing the externally
observable batch workflow.

The production recommender maps Korean Baekjoon study records to English
LeetCode follow-up problems, produces explainable recommendation artifacts, and
preserves compatibility with Slack delivery.

## Scope

This specification covers:

- Source problem discovery and parsing.
- Source problem representation.
- LeetCode index loading and retrieval.
- Recommendation scoring and filtering.
- Korean explanation generation.
- Recommendation artifact output.
- Compatibility with the existing Slack light job.
- Validation and observability requirements for the recommender path.

This specification does not cover online judge execution, AI-generated problem
creation, database schema design, frontend UI, or notification channels beyond
preserving the existing Slack artifact contract.

## Functional Requirements

### Source Problem Discovery

- The system SHALL support detecting changed Markdown and MDX files from the
  configured study repository path.
- WHEN Git history is unavailable or unreadable, THE SYSTEM SHALL provide a
  deterministic fallback path for discovering eligible source files.
- The system MUST allow repository root and target subdirectory configuration
  through environment variables or explicit parameters.
- The system MUST exclude non-problem documents such as `algorithm`, `intro`,
  `overview`, `readme`, and `template` records.

### Source Problem Parsing

- The system SHALL parse source problem frontmatter fields including `id`,
  `title`, and `tags`.
- The system MUST ignore source files that do not contain a stable `id`.
- The system SHALL preserve the current minimal query behavior based on title and
  tags until a structured profile extractor is designed and validated.
- WHEN richer MDX parsing is introduced, THE SYSTEM SHALL keep deterministic
  parsing separate from any LLM-assisted extraction.
- WHEN LLM-assisted extraction is used, THE SYSTEM SHALL provide a fallback path
  that does not block recommendation generation on malformed LLM output.

### Source Problem Representation

- The system SHALL represent a source problem with a typed or documented schema.
- The source problem schema MUST include source `id`, `title`, `tags`, and
  embedding text.
- The source problem schema SHOULD allow future fields such as
  `problem_summary`, `solution_summary`, `algorithm_pattern`,
  `data_structures`, and `complexity` without breaking existing consumers.

### LeetCode Index Loading

- The production recommender SHALL use the NumPy LeetCode index as the primary
  production retrieval source.
- The system MUST load vectors from `vectors.npy` and metadata from
  `metadata.json`, or from explicitly configured equivalent paths.
- The system MUST fail clearly when index files are missing, unreadable, or have
  incompatible lengths.
- The system MUST NOT reintroduce FAISS, PyTorch, or sentence-transformers into
  the production runtime without a separate design decision.

### Retrieval

- The system SHALL embed source problem text with the configured embedding model.
- The default production embedding model SHALL remain
  `text-embedding-3-small` until a benchmark-backed change is approved.
- The system SHALL retrieve candidate LeetCode problems from the NumPy index.
- The system MUST expose the raw dense distance or dense similarity used for each
  candidate.
- WHEN randomized candidate selection is used for diversity, THE SYSTEM SHALL
  make the random behavior explicit and configurable.
- The system MUST support deterministic retrieval mode for tests and evaluation.

### Scoring and Filtering

- Recommendation logic MUST remain explainable.
- Score components MUST be inspectable for every recommendation item.
- The system SHALL avoid recommending exact title duplicates or near-identical
  title matches to the source problem.
- The system SHALL avoid duplicate recommendation titles within a single source
  problem result.
- The system SHOULD support future hybrid scoring components including dense
  similarity, tag overlap, algorithm pattern overlap, difficulty match, and
  feedback prior.
- WHEN hybrid scoring is introduced, THE SYSTEM SHALL output each score
  component and the final score.
- WHEN feedback-derived score components are introduced, THE SYSTEM SHALL include
  the feedback data version used to compute the score.

### Explanation Generation

- The system SHALL generate a concise Korean explanation for each recommendation
  when an LLM is configured.
- The explanation MUST focus on algorithmic follow-up value, not generic praise.
- WHEN the LLM call fails or is not configured, THE SYSTEM SHALL return a stable
  fallback explanation instead of failing the entire recommendation run.
- The system MUST keep explanation generation separate from retrieval scoring
  until GPT reranking is explicitly designed and validated.

### Artifact Output

- The system SHALL write a recommendation map JSON artifact.
- The default artifact filename SHALL remain `recommendation_map.json`.
- The artifact MUST preserve Slack-compatible top-level problem entries with:
  `id`, `title`, `tags`, and `recommendations`.
- Each recommendation item MUST preserve the current Slack-compatible fields:
  `id`, `title`, `difficulty`, `similarity`, and `ai_comment`.
- The artifact SHOULD allow additive fields such as `url`, `slug`,
  `dense_score`, `tag_score`, `pattern_score`, `difficulty_score`,
  `feedback_score`, `final_score`, and `reason_ko`.
- The system MUST NOT remove or rename existing artifact fields without updating
  and validating the Slack light job.

### Slack Compatibility

- The recommender SHALL preserve compatibility with
  `src/main/slack_bot_daily_review.py`.
- WHEN the recommendation artifact schema changes, THE SYSTEM SHALL update the
  Slack formatter in the same story or provide a backward-compatible adapter.
- The Slack formatter MUST continue to handle missing or empty recommendations.
- The system MUST NOT send live Slack messages during recommender validation
  unless explicitly approved.

### Evaluation Compatibility

- The production recommender SHALL be callable from evaluation code once the
  refactor is complete.
- Evaluation MUST be able to run in deterministic mode.
- Evaluation SHOULD report Recall@K and MRR against the documented 100-pair
  ground truth when retrieval behavior changes.
- The system MUST distinguish production logic from `Evaluation_100`
  experimental code in documentation and code ownership.

### Observability and Errors

- The system SHALL use structured or consistently parseable logs for each major
  stage: source discovery, parsing, embedding, retrieval, scoring, explanation,
  and artifact writing.
- The system MUST report missing configuration, missing index files, and invalid
  artifact output as explicit errors.
- The system SHOULD record enough metadata to debug recommendation quality,
  including model name, index path, candidate count, and scoring mode.
- The system MUST NOT log secrets such as OpenAI API keys, Slack webhook URLs, or
  AWS credentials.

## Non-Functional Requirements

- The refactor MUST preserve the current heavy/light batch architecture.
- The refactor MUST keep production dependencies lightweight.
- The refactor MUST be implemented through small, independently verifiable
  tasks.
- The refactor MUST include tests or executable validation steps for behavior
  changes.
- The refactor SHOULD prefer typed schemas or documented dataclass-style
  structures over unstructured dictionaries at module boundaries.
- The recommender MUST remain usable without live Slack delivery.
- The recommender MUST avoid live OpenAI calls in unit tests unless explicitly
  marked as integration tests and approved.

## Backward Compatibility Requirements

- Existing Docker and workflow paths SHOULD continue to work unless changed by a
  documented design decision.
- Existing artifact consumers MUST continue to read current artifact fields.
- Existing local development paths SHOULD be preserved or replaced with explicit
  configuration.
- Existing `artifacts/recommendation_map.json` shape SHALL remain a valid sample
  for Slack formatting tests.

## Out of Scope for This Spec Iteration

- Implementing the recommender refactor.
- Adding GPT reranking to production.
- Adding database-backed feedback.
- Adding FastAPI or frontend surfaces.
- Adding online judge execution.
- Adding AI-generated problem validation.
