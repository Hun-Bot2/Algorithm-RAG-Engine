# Dual-model recommendation explanations (normative)

- The production explanation generation path SHALL concurrently attempt Gemini and NVIDIA NIM from the same immutable recommendation evidence and versioned instructions. There SHALL NOT be a single-model default or experiment switch.
- Missing credentials/model configuration SHALL be shown independently as unavailable, without suppressing the configured provider. Neither SHALL fabricate an API attempt.
- Ranking, selection, learner state, evidence generation and frozen journal snapshots MUST remain unchanged. Web reads SHALL NOT initiate generation.
- Provider/model/prompt/evidence/learner-specific successful results SHALL be cached independently; failure records SHALL remain visible until explicit retry. Each actual dispatch including retries SHALL have a new durable attempt ID; cache reads SHALL NOT add attempts or tokens.
- Local SQLite research records SHALL preserve input snapshot/hash, prompt version/instructions, generation parameters, provider/requested and returned model, provider response ID, generated/raw explanation, cited evidence IDs, validation result, usage with native fields/missingness, timestamps, latency, safe errors and cost provenance. Failed or interrupted attempts MUST NOT be erased.
- Both explanations and original engine evidence SHALL be shown in the existing web UI. Success, failure, missing configuration, in-progress/unknown, and deterministic fallback SHALL be distinguished. One provider failure MUST NOT discard the other's result.
- Aggregates SHALL group requests, success/failures, unknown/pending requests, latency, tokens and estimated/actual cost by provider and model with coverage. Missing values MUST NOT become zeros.
- Provider inputs SHALL exclude code, full study logs, paths, credentials and learner identifiers. Optional observability MUST exclude evidence/prose/learning history by default and MUST NOT be required for local retention.
- No LangChain, Grafana, new service, package or external database SHALL be required. Langfuse feasibility SHALL be documented against official current interfaces.
