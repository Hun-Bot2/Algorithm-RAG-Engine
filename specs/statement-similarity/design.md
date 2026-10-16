# Full-statement similarity — design and first implementation boundary

## Research summary

The existing 24,297-row source catalog combines 3,053 historical LeetCode records
and 21,244 external metadata-only records. Prior A1/A2 title/tag results and the
frozen pool are preliminary baselines. Existing LeetCode content is quarantined;
the new audit may inspect field/length/structural availability, without using the
text for retrieval or silently clearing rights.

## Design proposal

Use a new, versioned research namespace under research/statements/ and private
captures under private-learning/statements/. Preserve all existing baselines and
source decisions as historical evidence. A per-source acquisition registry records
URL/revision, policy evidence/hash/date, data license scope and third-party terms,
allowed uses, acquisition decision and unresolved conditions. A source must pass
both access and intended-use review before its collector is implemented/activated.
A user's authorization does not grant platform-owned rights.

The pipeline is acquisition → deterministic extraction/validation → original tags
+ reviewed shared taxonomy → duplicate identities/exact hashes/review queue →
frozen matched eligible corpus → original 384D MiniLM, TF-IDF, tag and hybrid
ranking → existing-label applicability and new blind review pool → comparison →
optional PCA/SVD. No metadata-only experiment fills a blocked statement stage.

Private records retain raw text/HTML and structured sections. Public reports
contain IDs, provenance, missingness and aggregates, never statement text without
redistribution clearance. Preserve mathematical notation; record image-dependent,
Japanese-only, truncated, parse-failed, not-provided and rights-pending states.
Never infer public samples from judge test sets or synthesize absent constraints.

The prior crosswalk remains a draft baseline. Create a new shared taxonomy with
stable concept IDs rather than treating broad CF tags as equivalent to every
narrow LC technique. AtCoder lacks algorithm tags in the current source; separately
reviewed annotations may fill gaps but must remain distinguishable from native tags.

## Validation plan

First step: deterministic offline availability audit and before/after hashes for
all baseline files; independently reconcile label counts with existing receipts.
Then inspect primary source policies and dataset-specific licenses. Only after a
source is cleared, implement its bounded opt-in collector with fresh directories,
serial per-host rate gates, size/time limits, stop-on-403/429/challenge/login and
hash-verified replay. Test extraction with authored fixtures before a live canary.
Do not equate parser fixtures or cached policies with successful collection.

## Risks and open gates

The major current risk is source clearance, not vector-search speed. Public API
metadata is not statement availability. Community data may state a data license
while retaining separate third-party conditions; report that distinction without
claiming all reuse is either permitted or unlawful. Licensing budgets remain $0
unless separately approved. No paid license purchase or contact is initiated.
The old 256-token model can discard constraints; measure this before retrieval.
Existing title-pooled judgments may favor old methods and lack statement evidence.
