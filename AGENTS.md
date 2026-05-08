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
scripts/ralph/ralph.sh
scripts/ralph/CODEX.md
scripts/ralph/prd.json
scripts/ralph/progress.txt
```

This is a Codex-first repository. Use the Ralph files manually with Codex unless
the local `codex` CLI and stable non-interactive invocation have been confirmed.
Do not assume Claude Code or Amp prompt files are part of the default scaffold.

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
3. Confirm quality-check commands are listed in `scripts/ralph/CODEX.md`.
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
