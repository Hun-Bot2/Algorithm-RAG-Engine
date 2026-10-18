# Independent implementation review

2026-10-17. AI source review of `build.py`, `reduction.py`, and `evaluate.py`.
This document is provisional while implementation agents finish; final validation
and disposition are recorded below when available.

## Findings raised

1. Build initially checked vector/key hashes without checking the catalog or exact
   A1 text digest. This allowed old vectors to be combined with changed metadata.
   The updated builder now checks catalog hash, variant input digest, exact selected
   key sequence and vector/key hashes against the frozen embedding receipt.
2. Initial tag eligibility admitted nonempty raw tags with no usable equivalent
   normalized concepts. Updated builder filters to nonempty direct equivalent
   concepts at confidence >= .85. Missing tags remain explicit in node metadata.
3. An initial A1-all reference named a nonexistent word-TFIDF run. Updated builder
   references and verifies the actual frozen char-TFIDF top50 artifact.
4. Test discovery exposed `sys.path` pollution: `test_evaluate` resolved to the
   algorithm-system tests. Package-qualified imports and discovery fixes were
   requested; the initial full-suite invocation failed before running tests.
5. Evaluation initially lacked manifest input checks. Updated wrapper checks output
   hashes; taxonomy pinning, explicit method allowlisting and source hashes were
   requested to prevent unmanifested methods or changed taxonomy entering results.
6. Reduction initially recorded current input hashes without validating the frozen
   A1 receipt. Receipt binding and exact selected-key/text verification were
   requested, alongside sampled coordinate row mapping and saved original neighbors.
7. Full-catalog Top10 uses representative/candidate-group exclusions; initial sample
   neighbors used platform/self only. Consistent sample masks or an explicit separate
   diagnostic policy were requested.
8. Tag retrieval fills top50 with deterministic key-sorted zero-overlap ties. Rank
   fusion therefore may grant tag-rank contribution despite no tag overlap. Raw
   score sidecars and explicit reporting or positive-score filtering were requested.

## Supported implementation choices

The map preserves every catalog row while limiting retrieval to eligible cohorts.
Title and statement vector spaces are not averaged. Statement availability is
metadata only, with approved overlay provenance separate from source identities;
p03510 stays unresolved and abc084_c flags are retained. Shared retrieval masks
self, representative duplicates and candidate groups, ranks deterministically,
checks finite blocks and computes exhaustive constituent top50 lists.

Hybrid fusion is explicitly bounded to the constituent top50 union, not claimed
as exhaustive full-rank RRF. Full semantic evaluation and the identical matched
semantic/tag/hybrid cohort are separate. The reused evaluator preserves unknowns,
reports judged coverage and precision bounds, and qualifies common-judged nDCG.
Its descriptive tag agreement is circular for tag-based methods. Native difficulty
scales remain separate. These properties do not establish human relevance,
classification truth, learner benefit, or a deployment outcome.

## Final source disposition and validation

Reinspection confirms the raised implementation blockers were addressed: package
imports no longer break discovery; A1 catalog/text/key/vector receipt binding is
implemented in both build and reduction; evaluation pins the taxonomy and records
source hashes while allowlisting the four intended methods. Reduction saves sample
indices/keys and original 384D Top10, and uses the shared representative/candidate
mask in both sampled reference and 2D neighborhoods. Hybrid excludes zero-Jaccard
taxonomy candidates from fusion and saves modality scores/ranks/support in its
sidecar. Tag-only zero-overlap ties are explicitly documented as no evidence.

Independent command `.venv/bin/python -m unittest discover -s research/unified_map -v`
passes all 8 tests (1.083 seconds on the reviewed source). These are authored
fixtures, not proof of an executed full dataset run. No remaining blocking source
finding was identified within this review scope. Parent-run dataset receipts,
preservation verification and notebook execution must be assessed separately.

Remaining limitations: bounded fusion is not exhaustive full-rank RRF; omitted
constituent rank in a sidecar means outside its top50, not known zero similarity.
The wrapper relies on the generated manifest listing each consumed artifact and
may silently compare fewer methods if an expected method file is absent. Strict
required-method/manifest-membership checks would harden malformed-input handling.
The existing tests do not exercise every hash-tampering rejection branch. Sparse
historical label support, unavailable statement/classification coverage and
lack of independent human outcome evidence remain scientific limits, not fixed by
passing tests.

## Executed final package review

The final `map-v2`, `reduction-v2`, statement evaluation and `package-v1` receipts
were inspected after execution. REPORT.md's catalog/platform/modality counts,
7,113 linked native identities (also 7,113 distinct keys), 1,020 identified records
outside the catalog and 1,291 unresolved source identities match their artifacts.
Its matched Top10 coverage, conditional precision, bootstrap intervals, two-query
common-positive nDCG support, 1–3 judged statement slots, reduction preservation
values and 500-pair queues match the evaluation receipts. The report preserves
unknowns and makes no algorithmic-superiority or learner-outcome claim. Excluded
native nodes are represented as metadata without native-map recommendations.

`statement_evaluate.py` applies the approved mappings only to in-memory metadata;
source corpus bytes and frozen rankings are not rewritten. It validates source
corpus/approval binding and each reused top50 hash. `package.py` retains publisher
source identities separately, marks catalog membership explicitly and does not
merge unresolved identities. The currently approved source records have no
preexisting quality flags lost by the overlay; abc084_c's two flags are preserved.

Independent checks verified all four package reference hashes and the source-index
hash. The final notebook has three executed code cells and no error outputs.
Independent unified-map suite rerun passes 11 tests (1.052 seconds). The parent's
edge-boundary and 501-file preservation verification are separately reported
checks, not repeated by this source reviewer. No blocking finding remains.

One nonblocking provenance limitation remains: the statement evaluation receipt
records source, ranking-manifest, approval and label hashes but lacks the native
evaluation receipt's taxonomy/evaluator code hashes. Preserve the current source
commit with that receipt; future reruns should include those hashes explicitly.

## Supplemental reduction relevance diagnostics

Reviewed `reduction_evaluate.py`, its unknown-label fixture and the executed
`quality-diagnostics.json`. Missing/abstained pairs remain outside the judged
precision denominator, directed labels are preserved, and linear full-catalog
versus sampled-2D cohorts are identified separately. REPORT.md correctly quotes
Original384/SVD256/PCA256 judged counts 597/596/580, their conditional positive
fractions 1.84%/1.85%/2.07%, tag agreement near .086, and only 2–31 judged slots for
sampled 2D. It explicitly rejects improvement inference from these differing
selected judged subsets. The new focused fixture passes independently.

No blocking scientific finding was identified. Supplement provenance could be
hardened by recording label/node/key hashes and requiring every consumed neighbor
file to be listed in the reduction receipt instead of unrestricted globbing;
the executed result uses the reviewed unchanged inputs. The supplement supplies
sparse conditional diagnostics, not new algorithm labels or promotion evidence.
