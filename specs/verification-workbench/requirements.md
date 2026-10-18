# Verification workbench requirements

- The system SHALL preserve frozen catalogs, statements, judgments, retrievals,
  and production behavior. It SHALL run locally without external services.
- A pilot SHALL contain 40 deterministic, stratified pairs from the existing
  unjudged queue (or all available pairs when fewer), with immutable source hashes,
  stable pair IDs, seed, sampling rule and actual strata counts.
- The browser SHALL display admitted private publisher descriptions as plain text,
  source identities, completeness limitations, and identity candidates. Title
  matching SHALL never resolve a native identity automatically.
- Each relationship review SHALL independently record graded similarity,
  prerequisite direction, transfer relevance, relative difficulty, evidence and
  rationale. Unknown SHALL remain distinct from negative.
- Reviewer identity and human/AI role SHALL be explicit self-reports. The system
  MUST NOT claim authenticated human identity or independent expertise from these.
- Other reviewers' relationship labels and retrieval ranks SHALL remain hidden
  until the current reviewer has submitted an independent human review.
- Verification SHALL require two distinct independently declared human reviewers
  plus a third human adjudicator. AI reviews and direct identity assertions SHALL
  never count as verification or alter source/native catalog identities.
- Accepted identity decisions SHALL require two agreeing native-ID proposals,
  canonical URL evidence matching the native catalog, and an adjudicator decision.
  The evidence is recorded for human assessment, not fetched or validated as proof
  of content equivalence by software.
- Records SHALL be append-only in private SQLite schema v1, hash-chained, exported
  losslessly, and idempotent by content. No existing database migration is needed.
- The local server SHALL bind loopback, reject untrusted Host/Origin and invalid
  JSON, bound request size, require a per-process write token, and serve only an
  explicit static allowlist. No statement export to public artifacts or deployment.
