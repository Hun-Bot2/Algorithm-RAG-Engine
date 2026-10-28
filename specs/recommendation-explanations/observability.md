# Optional Langfuse assessment (2026-10-20)

Decision: feasible as an optional external projection. Do not self-host Langfuse or add
LangChain, Grafana, collectors, SDK packages or a background exporter for this feature.
The local SQLite ledger and local usage UI/CLI already satisfy durable comparison and
accounting needs. A cloud account is optional; no account or external export is activated.

Official sources:
- [Direct OpenTelemetry ingestion](https://langfuse.com/integrations/native/opentelemetry)
- [Getting started, including plain HTTP/OTLP](https://langfuse.com/docs/observability/get-started)
- [Version and ingestion compatibility](https://langfuse.com/docs/compatibility)
- [API lifecycle](https://langfuse.com/docs/api-and-data-platform/features/public-api)

Langfuse accepts OTLP at `<region base>/api/public/otel/v1/traces`, with HTTP Basic
public/secret-key credentials. Current docs require `x-langfuse-ingestion-version: 4`
for direct OTLP v4 ingestion. The legacy `/api/public/ingestion` event endpoint is being
retired on Cloud on 2026-11-16; do not build a new integration against it.

Implementation: `explanation-otel --out <new-private-path>` writes a dependency-free
OTLP/JSON export of completed attempts only. It never sends data anywhere. The projection
contains deterministic pseudonymous trace/span IDs, provider/model, prompt version,
validation/status/error category, timestamps and numeric usage/cost fields. It excludes
input/evidence hashes, problem/learner IDs, titles, original reasons, prompts, generated
prose, raw responses, API keys and paths. IDs permit linking paired calls without exposing
the local learner partition. Metrics still disclose operational activity times/volume.

If an operator chooses to connect Langfuse, inspect this JSON first and explicitly POST
it to the project's chosen regional OTLP endpoint using the documented Basic credentials,
JSON content type and ingestion-version header. Keep credentials in a private process
configuration, never the payload or command history. Treat export failures as independent
of local generation; no local record is removed, mutated, or marked undelivered by default.
Deterministic span IDs aid reconciliation, but exactly-once remote ingestion is unverified.
No live Langfuse compatibility or account-limit claim follows from local OTLP shape tests.

The simple OTLP projection is suitable for timing/error/usage comparison. Full prompt
inspection and online evaluations would send learner-related content and require a new
explicit privacy decision. Do not infer semantic explanation quality from token/latency
metrics. Further automation is justified only after an actual operational need and a
successful redacted import trial; durable local records must remain sufficient on their own.
