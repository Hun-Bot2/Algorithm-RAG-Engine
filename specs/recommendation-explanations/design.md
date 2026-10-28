# Dual-model production design

## Research and architecture

The previous v1 module sequentially fell back from Gemini to NVIDIA and saved disposable JSON.
The production v2 path uses Promise.all on exactly two provider roles, a canonical frozen input
string and one versioned system prompt; transport formatting differs, instructions do not.
Journal/planner/evidence source files remain unchanged. A wrapper decorates copied views.
Generation stays explicit (`explain`); web reads only query locally persisted results.

A Node built-in SQLite research store at `<learner DB directory>/explanations/research.sqlite3`
is canonical for explanation records. No new service/dependency or learner DB migration.
SQLite schema version 1 uses a transactional, additive initial migration, foreign keys,
strict tables, WAL, busy timeout and append-only triggers on frozen snapshots, attempts,
and results. Input snapshots include learner partition locally only, evidence version/hash,
prompt and exact instructions/input. Attempt rows precede dispatch and results are appended
independently, before the other provider finishes. A crash may leave a started attempt with
unknown outcome; an unexpired lease suppresses duplicate dispatch across processes. Expired
attempts remain unknown and can be retried explicitly. No inference of nonbilling after timeout.

Cache identity: learner + frozen recommendation evidence (including snapshot/version) + prompt
version + provider + model + generation parameters. Adding/changing one model preserves the
other cache. Latest successful validated result wins; failures are durable and require explicit
`--retry-failed`. Every retry has its own attempt ID and retry_of pointer. Web cache reads create
no rows. Legacy v1 JSON is preserved but not imported: it lacks research/usage provenance.

## Costs, usage and privacy

Preserve allowlisted provider-native token fields and normalize input/output/total/cached/reasoning
counts without inventing absent values. Actual billed cost is null unless explicitly supplied by
the provider with currency and source. Estimated cost uses optional user-configured, per-model
USD/million input/output rates (and reasoning rate where known); it is never presented as a bill.
No default free/zero-cost assumption. Report coverage for missing tokens, latency and prices.
Aggregate every dispatch once, including invalid responses, HTTP failures and retries, exclude
cache hits, and separate unresolved started records from completed failures.

Provider input is the existing allowlisted recommendation facts; source logs/code/paths and
learner IDs are excluded. Local records retain narrow input and explanation text for reproducibility.
Upstream error bodies/headers are not stored; safe status categories and HTTP code suffice.
Credentials are never persisted. Observability is a derived, opt-in redacted export, not durable
truth. See observability.md. SQLite retention is indefinite by default; backup this research file
separately using SQLite online backup or after all processes stop; never delete it as cache.

## Validation and limits

Verify simultaneous dispatch, byte-identical input/system prompt, independent partial failure,
retry accounting, cross-service cache/leases, stale input/model isolation, append-only records,
missing usage/cost, invalid outputs and snapshot preservation. Model output shape/evidence IDs
are validated, not semantic truth or educational quality. Live endpoints, account quotas/billing
and Korean quality require configured keys and are separate from mock-based regression proof.

## Nemotron short-output transport correction

For `nvidia/nemotron-3.5-lightning-30b-a3b` specifically, send
`chat_template_kwargs: {enable_thinking: false}`. A live retry reached HTTP 200 but
exhausted the 512-token output cap (`finish_reason: length`); with the template switch
it produced a complete, structurally valid JSON explanation. The API/SDK choice was
not the blocker in this retry. The earlier 403 cause cannot be determined retrospectively.
The switch is model-specific, persisted as an actual generation parameter and included
in that provider's cache identity. Shared input, prompt, temperature and token cap remain
unchanged; Gemini's parameters/cache are unchanged. See NVIDIA's
[official template parameter example](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b).

HTTP failures now retain an allowlisted category (authorization failure, insufficient
scope, permission denial, regional restriction, gateway denial, quota, model missing,
or unspecified). Raw upstream messages are inspected transiently for classification,
never persisted or rendered; no arbitrary provider error text is treated as safe.
