# Production Recommender Design

## Overview

This design refactors the current production recommender into reusable modules
while preserving the existing heavy/light batch workflow.

The first refactor target is not to change recommendation behavior. The target
is to create stable module boundaries so generation, Slack delivery, and future
evaluation can call the same production path with deterministic validation.

## Architecture

Current production entry point:

```text
src/main/generate_index_map.py
```

Target production structure:

```text
src/rag/
  schemas.py          # typed source/recommendation/index schemas
  problem_parser.py   # Baekjoon Markdown/MDX discovery and parsing
  retriever.py        # NumPy index loading and dense retrieval
  scoring.py          # filtering and inspectable score components
  explainer.py        # Korean LLM explanation and fallback explanation
  recommender.py      # orchestration API used by jobs and evaluation

src/main/
  generate_index_map.py        # heavy job wrapper around src/rag/recommender.py
  slack_bot_daily_review.py    # light job, artifact consumer
```

Initial orchestration interface:

```text
AlgorithmRecommender.recommend_many(source_problems, top_k, deterministic)
```

The heavy job remains responsible for environment loading, repository path
selection, and artifact writing. The reusable recommender modules own parsing,
retrieval, scoring, explanation, and result construction.

## Current Production Path

The current path is NumPy-based:

```text
GitPython changed-file detection
  -> frontmatter parser
  -> OpenAI text-embedding-3-small
  -> indexes/numpy_leetcode vectors.npy + metadata.json
  -> squared L2 distance candidate search
  -> duplicate/title filtering
  -> GPT-4o-mini Korean explanation
  -> artifacts/recommendation_map.json
  -> Slack light job
```

FAISS files and `Evaluation_100` scripts are not the deployed production path.
They may inform design decisions, but production behavior must be verified
against `src/main/generate_index_map.py` and `src/main/slack_bot_daily_review.py`.

## Data Flow

1. Source discovery receives `repo_root` and `target_subdir`.
2. Discovery returns changed Markdown/MDX files or a deterministic fallback file
   list when Git history is unavailable.
3. Parsing converts files into `SourceProblem` records with `id`, `title`,
   `tags`, and `embedding_text`.
4. Index loading reads NumPy vectors and metadata from the configured index path.
5. Embedding creates one query vector per source problem.
6. Retrieval returns candidate records with dense distances or similarities.
7. Scoring filters duplicates and exact/near title matches.
8. Explanation adds a Korean `ai_comment` or stable fallback text.
9. Result construction emits a Slack-compatible `RecommendationMap`.
10. Artifact writing serializes `recommendation_map.json`.
11. Slack delivery reads the artifact and formats Block Kit messages.

## Schemas

The refactor should introduce explicit schemas before behavior changes.

Recommended schema boundaries:

```text
SourceProblem
  id: str
  title: str
  tags: list[str]
  embedding_text: str

IndexRecord
  id: str
  title: str
  difficulty: str
  slug: optional str
  tags: optional list[str]

ScoreBreakdown
  dense_score or dense_distance: float
  tag_score: optional float
  pattern_score: optional float
  difficulty_score: optional float
  feedback_score: optional float
  final_score: optional float

RecommendationItem
  id: str
  title: str
  difficulty: str
  similarity: float
  ai_comment: str
  score_components: optional ScoreBreakdown

RecommendationResult
  id: str
  title: str
  tags: list[str]
  recommendations: list[RecommendationItem]
```

The artifact must preserve current fields. New inspectable scoring fields should
be additive.

## Constraints

- The heavy/light batch architecture must remain intact.
- Production retrieval must continue to use the NumPy index until a separate
  benchmark-backed design changes it.
- FAISS, PyTorch, and sentence-transformers must not be reintroduced into
  production requirements as part of this refactor.
- Slack artifact compatibility must be preserved.
- Unit tests must not require live OpenAI, Slack, S3, Docker, or network access.
- Randomized candidate selection must be configurable, and deterministic mode
  must exist for tests and evaluation.
- Secrets must not be logged.
- The design must not treat `Evaluation_100` as the deployed implementation.

## Module Responsibilities

### `schemas.py`

Owns typed data contracts for source problems, index records, score breakdowns,
recommendation items, and recommendation maps.

### `problem_parser.py`

Owns changed-file discovery and Markdown/MDX frontmatter parsing. It should
start with the current deterministic frontmatter behavior and leave richer MDX
body parsing for a later validated task.

### `retriever.py`

Owns NumPy index loading, index shape validation, query embedding integration,
and candidate retrieval. It should expose raw dense distance or similarity.

### `scoring.py`

Owns title duplicate filtering and score normalization. Hybrid scoring can be
added later behind explicit score components.

### `explainer.py`

Owns GPT-4o-mini Korean explanation generation and stable fallback explanation.
It must not decide ranking until GPT reranking is separately designed.

### `recommender.py`

Owns orchestration. It should be callable by the heavy job and future evaluation
code without depending on Slack delivery.

### `generate_index_map.py`

Becomes a thin production job wrapper. It should keep environment compatibility,
call the recommender, and write `recommendation_map.json`.

### `slack_bot_daily_review.py`

Remains the artifact consumer. It should only change if artifact fields change
or a backward-compatible adapter is needed.

## Tradeoffs

### Keep Current Behavior First

Refactoring before changing ranking limits risk. It delays quality improvements
such as hybrid scoring, but it creates a testable path for those changes.

### Typed Schemas vs Existing Dictionaries

Typed schemas add initial implementation cost but reduce artifact drift and make
evaluation integration safer. The current dictionary shape should remain the JSON
boundary for compatibility.

### NumPy Retrieval vs FAISS

NumPy keeps the production runtime lightweight and matches the documented current
pipeline. FAISS may be useful experimentally, but adding it back increases image
size and dependency complexity.

### Random Diversity vs Deterministic Evaluation

Random selection from a candidate pool can improve recommendation variety, but it
weakens repeatability. The design keeps diversity configurable and requires
deterministic mode for tests and metrics.

### Explanation vs Reranking

GPT explanations are currently a presentation layer. GPT reranking may improve
precision, but it should be added as a separate stage with strict JSON parsing,
fallback behavior, and evaluation.

## Validation Strategy

Validation should be layered and offline-first:

1. Schema tests verify serialization keeps Slack-compatible fields.
2. Parser tests use small fixture Markdown/MDX files with frontmatter edge cases.
3. Retriever tests use a tiny in-memory NumPy index and deterministic embeddings.
4. Scoring tests verify duplicate filtering and score component visibility.
5. Explainer tests verify fallback behavior without live OpenAI calls.
6. Artifact tests verify `recommendation_map.json` shape.
7. Slack formatter tests verify existing artifact compatibility without sending
   real Slack messages.
8. Evaluation integration should later run deterministic Recall@K/MRR checks
   against the 100-pair ground truth when ranking behavior changes.

Required validation for each implementation task should be smaller than the full
pipeline and tied to the module being changed.

## Observability

The recommender should emit consistently parseable logs for:

- source discovery count
- parsed source problem count
- index path and index record count
- embedding model name
- candidate count
- scoring mode
- deterministic/random mode
- explanation fallback count
- artifact output path

Logs must not include API keys, webhook URLs, AWS credentials, or full private
study-note contents.

## Migration Plan

1. Add schemas and offline tests without changing production behavior.
2. Extract parsing into `problem_parser.py` and keep the heavy job output stable.
3. Extract NumPy index loading and retrieval into `retriever.py`.
4. Extract duplicate filtering and score mapping into `scoring.py`.
5. Extract explanation fallback behavior into `explainer.py`.
6. Introduce `recommender.py` orchestration.
7. Convert `generate_index_map.py` into a thin wrapper.
8. Add artifact compatibility tests for `slack_bot_daily_review.py`.
9. Connect evaluation to the production recommender in deterministic mode.

Each step should be separately validated before moving to the next step.

## Affected Files

Expected future implementation files:

- `src/rag/schemas.py`
- `src/rag/problem_parser.py`
- `src/rag/retriever.py`
- `src/rag/scoring.py`
- `src/rag/explainer.py`
- `src/rag/recommender.py`
- `src/main/generate_index_map.py`
- `src/main/slack_bot_daily_review.py`
- focused tests or validation scripts

No production Python file is changed by this design story.

## Open Questions

- Should schemas use `dataclasses`, `TypedDict`, or Pydantic? The current
  dependency set does not require Pydantic, so standard-library schemas are the
  conservative first choice.
- Should deterministic mode preserve top-k sorted nearest neighbors or seeded
  random selection from the candidate pool? Tests should prefer sorted nearest
  neighbors.
- Should future `slug` fields come from metadata when available instead of being
  generated from title in the Slack job? This should be additive.
