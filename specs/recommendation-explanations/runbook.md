# Dual-model Korean explanations (default production path)

Every new explanation uses **Gemini and NVIDIA NIM concurrently** with identical frozen
recommendation JSON and versioned instructions. There is no experiment/single-model mode.
Selection/ranking and immutable learner recommendation snapshots remain unchanged. Requests
are made by explicit `explain` execution; ordinary web/journal reads use persisted results.

## Setup (Node 26; no package install)

Create ignored `private-learning/explanation.env` and restrict it with `chmod 600`:

```dotenv
GEMINI_API_KEY=your-key
GEMINI_EXPLANATION_MODEL=gemini-3.5-flash-lite
NVIDIA_API_KEY=your-key
NVIDIA_EXPLANATION_MODEL=exact-model-id-available-to-your-account
```

Both providers are required for full dual generation. Missing settings display that
provider as `설정 필요`, without blocking the other or inventing a request. Never paste
keys in chat. Account quotas/credits/billing eligibility must be checked by the operator;
no unlimited-free or zero-charge guarantee is made. Gemini free-tier inputs may be used
for product improvement. Only allowlisted policy facts are transmitted, not study logs/code.

From the repository root:

```bash
node hosted/cli.mjs repo ingest
node --env-file=private-learning/explanation.env hosted/cli.mjs explain
node --env-file=private-learning/explanation.env hosted/local.mjs
# Open http://127.0.0.1:8787, expand 추천 이유 보기 and AI 설명 사용량.
```

New generation processes at most five current recommendation slots sequentially. The two
providers for each slot start concurrently; each request has a 15-second timeout. The same
input string, evidence ID, instructions, prompt version, temperature and output-token cap
are used; Gemini's native JSON schema constrains the same output requested from NVIDIA.
Configured provider/model and returned provider model version are recorded separately.

The web shows two columns on desktop, stacked on mobile: validated success, failure,
missing config, not generated, in progress or unknown result. Original engine evidence
remains available. If both explanations are unusable, a deterministic Korean template is
shown. Validation is structural/evidence-ID checking, not a semantic truth guarantee.
Historical recommendations retain their original evidence; generation targets current slots.

## Cache and retries

Each provider/model result is cached independently by learner partition, frozen input,
recommendation evidence/snapshot version, prompt version and generation parameters.
Changing one model preserves the other result. Keys are not cache identity and can rotate.
For NVIDIA, keep its model setting when viewing previously stored responses without keys.

Successes are never regenerated on page refresh or repeated `explain`. Failures also stay
recorded: explicitly retry only failed/expired-unknown requests with:

```bash
node --env-file=private-learning/explanation.env hosted/cli.mjs explain --retry-failed
```

Every retry appends an attempt with `retry_of`; successful peers are not called again.
There is no automatic retry/backoff. Honor provider rate limits before explicit retries.
A live attempt lease suppresses duplicate concurrent invocations. After a process crash,
started attempts remain unknown, not zero-token/free/definitely failed. Explicit retry is
allowed only after the lease (timeout + 30 seconds); the original remains recorded.

## Durable research records and privacy

Canonical store: `private-learning/explanations/research.sqlite3` (beside the learner DB
when `--db` changes its directory). Schema v1 is created transactionally with append-only
inputs/attempts/results/cost receipts; no learner DB migration is involved. There is no TTL
or automatic deletion. **This is a research ledger, not disposable cache.** v1 JSON files
are preserved but not imported because they lack dual-response/attempt provenance.

Records include exact narrow input JSON/hash, prompt/instructions/schema, provider/model,
parameters, all attempts including failures/retries, returned prose/raw text and evidence
IDs, validation, safe errors/status, timestamps, latency, native and normalized usage, cost
estimates and separate actual receipt provenance. Keys, raw provider error bodies, filesystem
paths, source code and full learning histories are excluded. Records are private, not assets.

The learner's normal SQLite backup/export does not include this separate ledger. Back up
both. Use SQLite online backup for a running ledger (Python example below), or stop all
processes before copying it. Do not copy only the main SQLite file while WAL writers run.
A new ledger copy at the same relative path remains usable with matching model settings.

```bash
python3 - <<'PY'
import sqlite3
from pathlib import Path
source = Path('private-learning/explanations/research.sqlite3')
target = Path('private-learning/explanations-backup.sqlite3')
if target.exists():
    raise SystemExit('refuse overwrite')
with sqlite3.connect(f'file:{source}?mode=ro', uri=True) as src:
    with sqlite3.connect(target) as dst:
        src.backup(dst)
target.chmod(0o600)
PY
```

## Usage and cost reports

```bash
node hosted/cli.mjs explanation-usage
node hosted/cli.mjs explanation-records   # private: includes input/prose
```

Reports and the web usage table group every dispatch attempt (retries included) by provider
and requested model: successes/failures/unresolved, input/output/total/reasoning tokens,
latency sum/mean, estimated and actual USD cost. Cache hits add neither attempts nor usage.
Each total has coverage counts: null means unavailable, a partial sum is only known coverage.
A stored attempt is a dispatch intent recorded immediately before fetch; a crash can leave
remote execution/acceptance unknown. No inference of provider receipt or billing is made.

Native usage fields are retained even if output validation fails. Missing token fields are
not inferred from text or total counts. Gemini candidate tokens and thought tokens remain
separate; NVIDIA completion counts may already include reasoning tokens.

Optional per-model estimates (no default rates): add one-line JSON to the private env file:

```dotenv
EXPLANATION_PRICES_JSON='{"gemini/gemini-3.5-flash-lite":{"input_per_million":0.30,"output_per_million":2.50,"source":"operator-verified rate and date"}}'
```

Verify current prices yourself before using this illustrative entry. Rates are USD/million.
Gemini estimates require input/candidate/thought counts, and include thoughts at the configured
output rate; NVIDIA uses completion tokens once. Nominal estimates exclude caching discounts,
other fees and tier adjustments. Rates/provenance are frozen per result; updating configuration
does not rewrite historical estimates. Actual cost stays unknown because these standard API
responses do not report the invoice. To reconcile a known invoice, write a private JSON array:

```json
[{"receipt_id":"invoice-line-1","attempt_id":"recorded-attempt-uuid","actual_usd":0.001,"source":"user-supplied invoice identifier"}]
```

```bash
node hosted/cli.mjs explanation-costs --path private-learning/cost-receipts.json
```

Receipts are append-only, owner-checked and idempotent. A new receipt ID can correct an earlier
receipt; reports use the latest for that attempt without deleting prior provenance. These are
user-supplied billing records, not independently verified costs.

## Optional observability and validation

`explanation-otel --out private-learning/new-redacted-otel.json` produces an offline redacted
OTLP payload. It does not contact Langfuse. See [assessment](observability.md) for current
Langfuse endpoints/compatibility, privacy and the decision against added infrastructure.

```bash
node --test hosted/test/*.test.mjs
node hosted/explanation-browser-smoke.mjs # installed Chrome + loopback permissions; synthetic only
```

Real provider endpoints, Korean quality, account quotas/prices and Langfuse ingestion need a
separate configured trial; automated fixtures never establish these facts.

## Model access troubleshooting (2026-10-20)

The default is now `gemini-3.5-flash-lite`. An explicit env-file setting overrides it:
update `GEMINI_EXPLANATION_MODEL` there if it still names 2.5. Google's current
[model lifecycle notice](https://ai.google.dev/gemini-api/docs/deprecations) limits 2.5
access to prior active users; it does not state that the stable 2.5 models are shut down.
A 404 alone cannot prove the specific cause for a given account. Confirm supported model
IDs/capabilities in AI Studio or Google's model-list API for your project.

The NVIDIA model ID `nvidia/nemotron-3.5-lightning-30b-a3b` and our chat-completions URL
match [NVIDIA's official example](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b).
A 403 alone does not distinguish key scopes, account/model access, regional restrictions
or a gateway denial. Log into Build, select that model, and check the generated API example
and key access there. Do not post keys or raw error bodies to chat. We do not auto-replace a
model or erase failed attempts to conceal an access problem.

After changing model/key settings, run `explain --retry-failed` with the env file, then
restart the local server with that same file and reload the page. Existing failed records
are not automatically retried; new model IDs get a separate cache identity and record.

## Restarting the server without regenerating explanations

After explanation results have been persisted, restart only the local server:

```bash
node --env-file=private-learning/explanation.env hosted/local.mjs
```

Do not run `explain` as a server-startup prerequisite. Startup, page refresh and repeated
journal GETs read existing SQLite results without dispatching either API, including when
one provider previously failed. Preserve the same learner DB directory, learner identity,
provider/model settings and prompt/evidence versions to display the same cached pair.
Different recommendation evidence or model/prompt settings have separate cache identities;
merely finding a cache miss in the web does not trigger generation. Even an explicit
`explain --retry-failed` reuses successful results and only retries failed/unknown peers.

Process-level verification: `node --test hosted/test/mastery-explanation-restart.test.mjs`
starts an isolated synthetic server, seeds paired mock responses, stops the process,
starts a new process on the same files, and reads the journal three times. Both success
and partial-failure cases verify identical records, identical usage and zero external
calls after restart. This needs loopback sockets and uses no real credentials.

A later live retry on this checkout returned NVIDIA HTTP 200, not 403. It initially
failed output validation because the 512-token cap truncated the response. For this
Nemotron model the adapter now explicitly disables thinking, retaining the shared output
cap and storing the model-specific switch in attempt parameters/cache identity. The next
call completed successfully. This does not retrospectively establish why the earlier
403 occurred, and does not mean that any 403 can be fixed with generation options.
