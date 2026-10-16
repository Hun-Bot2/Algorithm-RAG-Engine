# Adaptive learning requirements v1

## Scope and precedence

This specification adds a private adaptive-learning application. It supersedes
mandatory embeddings/LLM requirements in `../production-recommender/` only for
this new product path. The legacy batch application and artifact contract remain
unchanged. M1-M3 establish foundations; later milestones are not released.

## Core application

- The system MUST operate without OpenAI credentials, LLM servers, embedding
  inference, or a running personal computer after hosted deployment.
- Daily missions, timers, history, prerequisite planning, review scheduling,
  analytics and notifications SHALL have deterministic no-AI implementations.
- The default weekday budget SHALL be configurable, initially 60 minutes.
- The system SHALL use platform-qualified identifiers and canonical links,
  native difficulty, explicit access state, provenance and source freshness.
- Relationships MUST remain separate from adaptive decisions. Approved directed
  prerequisite edges MUST form a DAG. Unknown access MUST NOT imply free access.
- Imported acceptance SHALL mean reported completion, never independent mastery.
- Evidence states SHALL be unknown, learning, demonstrated, transfer-evidenced.
- Review intervals SHALL initially be 1, 3, 7, 14, 30, 60 days. Failed or assisted
  review SHALL return to next-day remediation. Practice and review are separate.
- Missions SHALL record candidate exposure, selection components, policy version,
  event cutoff and grounded reasons. Missing outcomes MUST remain missing.

## M2: asset and research boundaries

- Original datasets, vectors and evaluation files MUST remain byte-identical.
- The registry SHALL record hashes, missing files, provenance, rights status,
  permitted uses and release exclusion. Unknown rights MUST fail closed.
- Historical 100 pairs and 300 candidate links MUST NOT become training truth.
- Embedding manifests SHALL include namespace, revision/license references,
  dimensions, normalization, preprocessing/input hashes, ordered keys and metric.
  Unavailable historical fields MUST be null with an explicit boundary.
- Incompatible spaces MUST NOT be compared or averaged. Missing models SHALL
  remove optional evidence without triggering a remote fallback.
- Optional annotation drafts MUST NOT write approved relationships or outcomes.

## M3: learner evidence

- Events SHALL have schema version, globally unique event ID, learner ID,
  occurrence and recorded UTC timestamps, type, provenance and typed payload.
- Storage SHALL be append-only; corrections SHALL supersede one earlier event
  belonging to the same learner and type, retaining original evidence.
- Duplicate identical IDs SHALL be idempotent; conflicting reuse MUST fail.
- Reads/exports SHALL require a learner ID. Hosted authentication MUST supply
  this identity; a caller-provided ID alone is not hosted authorization.
- Timers SHALL derive only from explicit start/stop/confirmation actions.
  Open or unconfirmed intervals MUST NOT count toward study time. Submissions
  MUST NOT imply duration. Confirmed overlapping intervals MUST NOT double count.
- SQL migration SHALL be versioned, transactional and reject schema drift.
- Export/import SHALL be versioned, owner-scoped, validated and atomic. Restore
  SHALL reproduce replay order and derived history; repeated import is a no-op.

## Hosted release and future gates

- Worker static assets + D1 + Access SHALL host the private application. Python
  SHALL handle offline data and research. No model weights/full-vector scans in
  Workers. Local SQLite alone does not meet unattended delivery requirements.
- Approximately 90 independently reviewed problems and 180 held-out relationship
  judgments SHALL precede curriculum release. No fabricated review approvals.
- Today/Progress/Contests SHALL support manual evidence, timer recovery and stale
  provider fallbacks. Contest suitability overrides the configurable weekly rhythm.
- Freeze missions at 05:40 Asia/Seoul; target Slack delivery 06:00-06:10. Use a
  scoped bot, durable daily uniqueness, five-minute recovery and reconciliation
  of ambiguous sends. Track scheduled/posted/unknown/missed separately. Phone
  alerts are a distinct measurement. No scheduled-message metadata parameter.
- Codeforces ingestion SHALL be cached, incremental and serialized; AtCoder
  community data SHALL expose freshness/manual fallback. LeetCode scraping and
  restricted-content redistribution remain disabled pending rights review.
- Schedule activation, public registration, live sends, paid inference and model
  training MUST remain disabled until their individual release gates pass.
- Target recurring cost is $0 within verified free-tier limits, not a guarantee.

## Research gates

Compare equal-input metadata, TF-IDF, eligible legacy vectors, unchanged compact
encoders, fusion, then a small feature scorer before encoder fine-tuning.
MiniLM-L6-v2 and multilingual-e5-small are candidates, not installed dependencies.
Pin revisions, license, input formatting and resource measurements before use.
Sentence Transformers batch jobs are preferred; Ollama is optional local-only
annotation tooling; vLLM is deferred. Neither is required by production.

Initial 180 reviewed judgments are evaluation-only. An encoder pilot requires a
reviewed planning checkpoint of about 1,000 adjudicated training judgments over
300 distinct problems, platform-pair/family diversity, learning curves, authorized
inputs and owned-compute budget. Do not train from scratch or historical pairs.
Freeze problem/family-disjoint splits, loss/config/seeds and early stopping;
no training pair may contain a held-out problem. Report Recall@10, nDCG@5,
directional precision/accuracy, subgroup failures, ablations, grouped uncertainty,
CPU latency, memory, duration and cost. Promotion requires >=0.03 absolute held-out
nDCG@5 gain over the strongest simpler deployable baseline, no eligibility
regression and acceptable subgroups. Fine-tuning must also beat its base encoder.
Inconclusive evidence retains the simpler baseline. Relationship quality does not
prove learning improvement; prospective transfer/delayed re-solves and longitudinal
exposure/outcome evidence are separate gates for personalization research.
