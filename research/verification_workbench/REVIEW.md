# Independent workflow review

The `test_core.py` corpus, identities and review records are synthetic fixtures.
They are validation inputs, not human pilot judgments or evidence of reviewer
expertise. This review accepts the implementation for a scoped local pilot. It
does not establish deployed behavior, publisher rights, or semantic correctness
of recorded evidence.

The core freezes queue/corpus/catalog byte hashes and stable pair IDs, keeps title
matches as unverified candidates, records four relationship dimensions with an
explicit unknown state, hides other reviews and retrieval metadata before a
self-declared independent review, and requires two distinct self-declared human
reviewers plus a third human adjudicator before an accepted overlay is exported.
Identity decisions also require the same native catalog key and canonical URL
evidence. SQLite insert transactions, update/delete triggers, content idempotency,
and a hash chain protect ordinary append-only use. Export includes the complete
record history, adjudicator exposure history and the audit head. A declared
adjudicator view writes an exposure receipt and blocks a later independent review
claim by that same reviewer on that pair. Idempotent replay of a pre-exposure
review remains valid. A later correction suspends a prior accepted overlay until
the corrected evidence is adjudicated. A fresh-store restore reproduces records,
timestamps, exposures and audit. No source or production identity is changed.

Limits: reviewer names, roles, independence and evidence are self-reported. A
person can reuse another self-reported reviewer name to retrieve that review;
the local API has no authentication. The blind view is therefore a workflow aid,
not access control or proof of independent human judgment. A database owner can
alter data and triggers directly; the hash chain detects payload tampering when
`audit()` runs but is not a remote notarization mechanism. No real pilot labels
or accepted identities have been produced by these tests.

The frozen pilot receipt records 40 pairs, 74 distinct problem keys and 37
unresolved native identities. Title lookup supplies at least one unverified
candidate for 36 of those 37 keys. The receipt records zero review records and
40 pairs pending human relationship work. Candidate presence is no identity
verification. The pilot receipt is `pilot-receipt.v1.json`.

Validation command: `python3 -m unittest discover -s research/verification_workbench -v`.
The 10 core fixture tests pass in this review environment. Its two loopback HTTP
tests could not bind a socket under this agent's sandbox (`PermissionError`);
the parent agent reports 12 passing tests with socket access. The synthetic
suite checks deterministic sampling and drift refusal, candidate
status, all relationship dimensions, relationship and subject-scoped identity
blindness, reviewer/role/independence gates, identity URL and subject gates,
idempotency, append-only triggers, chain detection, correction invalidation,
and export/restore. HTTP tests cover Host, Origin, write token, static allowlist
and request-body rejection.

The private browser-smoke receipt reports a real read-only pilot view, synthetic
two-reviewer/third-adjudicator flow, no browser exceptions, and no horizontal
overflow at 390 px. The captured desktop screenshot visibly contains both
pilot statement panels and their source-completeness notices. The updated smoke
script waits for both statements and the completeness notice before capture.
The code's `#records .record-card` blind-view assertion
targets the actual record class and checks that reviewer 2 sees no reviewer 1
label before submitting. The browser smoke writes only to its synthetic fixture
store; the real pilot stays at zero records. Actual independent human review
remains an open external gate.

## AI-first review extension — 2026-10-17

The user requested AI drafts before judging them. The imported batch has 40 AI
relationship records, no human records/adjudications, and no accepted overlays.
The full private publisher-description packets bind 80 endpoint content hashes.
Root review corrected four reversed difficulty enums before import (positions
1,2,3,23). Initial/corrected artifacts are private; the first initial snapshot was
reconstructed afterward and is explicitly marked as such in its receipt.

Independent reasoning-agent audit found no concrete gate bug in AI-only import
or assisted viewing. It rechecked the two DP/parity sketches at pilot positions
28 and 34 against the provided statements. This is a spot check, not mathematical
certification of all 40 solution sketches. Six pairs retain all-unknown judgments;
several others include partial derivations and provisional dimensions for the user
to accept or correct. No AI precision, difficulty calibration, or gold claim follows.

Root validation: 16 workbench tests pass, including both loopback HTTP tests.
The AI-assisted Chrome test uses actual-pilot read-only access plus disposable
synthetic human submissions. It verifies AI+own visibility, other-human hiding,
assisted save, no accepted decision, persistence across reload, and mobile width.
AI explanation cards now display prose and separate dimensions, with references
and raw JSON in expandable sections. See `validation.ai-first.v1.json`.
