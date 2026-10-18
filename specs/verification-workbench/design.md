# Design and validation

Research: the frozen 1,000-pair queue has zero overlapping historical judgments.
AtCoder dataset identities are unresolved; source tags/difficulty remain missing.
A 40-pair pilot is a workflow/annotation-guide test, not sufficient evidence for
model promotion or supervised difficulty calibration.

Use standard-library Python for deterministic pilot preparation, a local HTTP
server, and a new private SQLite store. Serve static HTML/JS without a build tool.
The browser receives only pilot problems, never a generic filesystem endpoint.
The stored pilot binds SHA256s of the source queue, private corpus and native
catalog. Startup refuses drift. Candidate lookup normalizes dataset title wrappers
and uses lexical title evidence, with source/status kept as proposals only.

Schema v1: metadata table binds schema and pilot hash; append-only records table
stores immutable canonical JSON, record hash, previous hash and insertion sequence.
Transactions serialize writes. Update/delete triggers reject edits. Corrections
are subsequent reviews/adjudications; exports preserve all records and the chain.
Exported verified decisions are a separate reviewed overlay and never mutate
frozen identities, relevance files, curricula or production policies.

Relationship, identity and adjudication forms share a validator; the server never
trusts UI validation. Before acceptance, adjudications must reference two matching
subjects and distinct human-independent reviewers, and use a third adjudicator.
Identity accept decisions agree with both native-ID proposals and canonical URL
references. Relationship disagreements require explicit adjudication; agreement
alone is not a gold label. Review independence and human role are self-attestations.

Validation: deterministic sampling and source hashes; title wrappers/candidate
missingness; invalid dimensions/evidence/roles; unknown vs negative; blind views;
AI and duplicate reviewer exclusion; wrong subject/decision/canonical URL;
transactional idempotency; append-only triggers/hash-chain/tamper checks; export
round-trip; HTTP Host/Origin/token/path/body rejection; actual local UI smoke.
No automated system can establish reviewer expertise or verify semantic evidence.
