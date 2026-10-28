# Dual production explanations: phases

- [x] 1. Specify local schema/migration/retention and implement durable frozen inputs plus append-only per-provider attempts/results.
- [x] 2. Concurrent adapters with equivalent versioned prompts; independent caches, safe errors, explicit retry lineage and usage/cost missingness.
- [x] 3. Wire production CLI/journal presentation; show two result columns and original evidence.
- [x] 4. Add provider/model aggregate usage reporting and assess optional redacted Langfuse export.
- [x] 5. Regression tests, existing suite, syntax checks, documentation and validation boundaries.

## Final validation

See validation.md and validation.v2.json for reproducible receipts and scope boundaries.
Proposed commit: `feat(mastery): persist and compare dual-model explanations`.
No commit, deployment, real-provider request or external trace upload was performed.

- [x] Follow up on reported 404/403: update current Gemini default and safe cached-error
  guidance, document account/model checks, and run 25 focused regression tests.

- [x] Verify stored successful/failed explanations survive actual server-process restarts
  without new provider dispatches, token counts or record mutations; document startup-only command.

- [x] Diagnose current NVIDIA requests with safe error categories; verify live HTTP 200,
  fix truncated Nemotron JSON via model-specific thinking=false, preserve the Gemini cache,
  confirm no new usage on repeat, and pass all 167 Node regression tests.
