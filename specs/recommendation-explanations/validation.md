# Dual-explanation validation — 2026-10-20

Final full command: `node --test hosted/test/*.test.mjs`.
Result: **162 passed, 0 failed, 0 skipped**. Loopback access was permitted for this final
run; the initial restricted run's two socket failures were environment EPERM errors.
Node syntax and `git diff --check` passed.

Eleven explanation regression tests plus the production CLI end-to-end test exercise:

- Simultaneous calls, byte-identical frozen input/instructions and equivalent parameters.
- Per-provider result commits while the other is pending; cross-instance lease deduplication.
- Both success, partial failure, both failure, timeout, malformed JSON, wrong evidence IDs,
  incomplete output and unavailable usage. Provider errors are sanitized.
- Missing configuration without fake requests; per-learner/input/model caches; explicit
  retry lineage and no repeat dispatch/accounting for cache reads.
- Native token preservation, missingness/coverage, pricing provenance, append-only invoice
  receipts and ownership checks, unknown interrupted attempts, append-only SQL triggers.
- Original snapshot IDs, selected problem/order and stored snapshot bytes unchanged.
- Offline OTLP redaction and CLI reporting from durable local records.

Actual installed Chrome/CDP command: `node hosted/explanation-browser-smoke.mjs`.
Three isolated synthetic cases passed: dual success, partial failure and both failed.
Each verified desktop columns, mobile stacking without page overflow, original evidence,
correct success/failure/fallback states and no new provider calls on reload. Screenshots
were generated and partial-failure desktop output was visually inspected. Fixture generation
used two mock dispatches per case; no real account or learner was used.

Private synthetic browser artifacts:
`/var/folders/90/cs35h9p976g0zgmt05xtrt4m0000gn/T/explanation-browser-aXWNgu/receipts/`.
The machine-readable validation receipt includes hashes of this run's test log/browser
receipt and implementation files. Temporary artifacts may expire; the browser smoke is
self-contained and rerunnable without keys. The browser test starts/stops only its own
Chrome process, loopback server and disposable profiles/stores.

The actual learner DB, catalog, pinned research files and recommendation policy modules
were not modified. No new package, service, LangChain, Grafana or external database was
introduced. No real Gemini/NVIDIA request, invoice verification, real Korean-quality
assessment, live Langfuse import, deployment or notification was performed. OTLP export
is a local projection; its live Langfuse acceptance remains unverified.

## Follow-up: reported Gemini 404 / NVIDIA 403

The Gemini default is now `gemini-3.5-flash-lite`, based on Google's current lifecycle
notice restricting 2.5 access for new users. NVIDIA's reported model ID/endpoint matches
its official Build example; a 403's exact cause remains unknown without an account-side
check. Cached errors now receive safe, provider/status-specific UI guidance without
rewriting attempts. Env-file overrides must be updated separately by the operator.

Validation after this follow-up: 12 explanation tests plus 13 CLI/UI-contract tests passed;
UI syntax and whitespace checks passed. No real keys were inspected and no live API call
was performed. The earlier validation.v2.json/source hashes document the prior baseline,
not this follow-up source revision. Prior live failure attempts remain untouched.

## Follow-up: actual server-process restart

Added `mastery-explanation-restart.test.mjs` and its synthetic child-process helper.
Two tests passed with loopback permission: paired success and partial failure. Each
terminates the first server process, starts a different PID on the same learner/research
SQLite files, and performs three journal GETs. Stored answer/status pairs, records and
aggregate token/request counts are identical, with zero external API calls after restart.
The initial sandbox run was blocked by listen EPERM; the permitted rerun passed both.
No production-code change or real learner data mutation was required for this follow-up.
Suggested commit: `test(mastery): verify explanation reuse across server restarts`.

## Follow-up: live NVIDIA diagnosis and output-cap correction

A read-only NVIDIA model-list request returned HTTP 200 and included the configured model.
This alone was not treated as proof of inference permission. The explicit failed-explanation
retry then returned HTTP 200 with `finish_reason: length` at the shared output cap. Gemini's
already successful record was a cache hit. After disabling thinking for this specific
Nemotron endpoint, the NVIDIA retry returned HTTP 200 / stop and passed the existing
JSON/evidence-ID/Korean output checks. This is structural validation, not a semantic
quality evaluation. All generation attempts and their native usage remain in the private
local research ledger. Keys, error bodies and generated learner-related prose were not
printed. The cause of the historical 403 remains unknown.

Another explicit command reused both successful cache entries and produced an identical
aggregate usage report: no additional dispatches/tokens. Final full Node regression suite
passed **167/167**, including process-restart tests, safe-error classification and the
Nemotron parameter/cache regression. The earlier validation.v2.json remains a historical
baseline receipt. No production-policy source file was modified.
Suggested commit: `fix(mastery): complete short Nemotron explanations without thinking`.
