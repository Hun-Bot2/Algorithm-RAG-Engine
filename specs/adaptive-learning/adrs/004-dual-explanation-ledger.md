# ADR 004: Dual-model presentation with durable local attempts

Status: accepted for the user-requested implementation, 2026-10-20.

Context: The user requires Gemini and NVIDIA explanations for each new recommendation,
side-by-side comparison, durable records, request/token/cost aggregates and optional
observability. Earlier single-result JSON fallback cannot preserve these observations.

Decision: Make dual concurrent generation the only production explanation path, operating
on the existing frozen recommendation evidence. Preserve the no-AI learner engine and
its ranking/selection. Persist narrow inputs, provider-specific attempts/results and cost
receipts in an independent Node built-in SQLite file beside the learner DB. Use append-only
records and per-provider cache/leases. Generation remains explicit; web reads do not dispatch.

Existing data migration: no learner schema migration; create research schema v1 atomically
on first write. Earlier JSON files remain byte-identical but are not imported as research
receipts. Future changes require transactional, versioned additive migration planning.
Back up the explanation ledger separately using SQLite online backup; it is not covered by
learner event backup and is never disposable cache. There is no automated deletion.

Alternatives rejected: replacing original recommendation snapshots with prose (changes
canonical evidence); one-provider fallback (loses paired observations); external Langfuse
as authority (privacy, availability and extra infrastructure); LangChain/Grafana/full SDK
stack (unnecessary dependencies for two direct fetch calls and local reports).

Consequences: two configured provider accounts/models and potentially two API charges per
new explanation; explicit retries remain separately accounted. Missing provider config
is visible, not silently converted to a single-model policy. Explain failures never alter
learner state. Local privacy/provenance records are richer and require separate retention
and backup. Langfuse remains an opt-in redacted OTLP projection; no new service is deployed.

See specs/recommendation-explanations/ for schema, commands, provider contract and validation.
