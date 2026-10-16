# Full-statement algorithmic similarity — requirements v1

Status: current research priority, requested 2026-10-16. Supersedes the next-work
priority in adaptive-learning/full-catalog-similarity.md, without invalidating its
completed baselines, frozen labels, source restrictions or recorded provenance.

1. The system SHALL first audit statement, constraints, public samples, language,
   tag, identity and provenance availability and report acquisition feasibility
   separately for LeetCode, Codeforces and AtCoder.
2. Collection SHALL occur only for sources whose access and intended use have been
   verified. Official APIs, undocumented endpoints, community data and first-party
   licensing SHALL be distinguished. HTTP accessibility, robots allowance and a
   repository code license MUST NOT be presented as statement reuse permission.
3. The system MUST NOT bypass authentication, Premium, robots restrictions,
   challenges, blocked clients or anti-bot responses. It MUST NOT substitute
   mirrors to conceal unresolved upstream restrictions or fabricate missing text.
4. Records SHALL retain native ID, aliases, canonical/source URL, publisher and
   source revision, acquisition basis, raw/content hashes, language and detection
   basis, parser version, collection status, source-specific limits and allowed
   uses. Statement, constraints, input/output, public samples and notes SHALL be
   separate fields with explicit missingness. Raw/source captures remain private.
5. All original tags SHALL be preserved. Shared concepts SHALL record mapping type
   (equivalent, broader, narrower, related, unmapped), evidence, reviewer/version
   and annotation provenance. Empty/unavailable AtCoder tags are not negatives.
6. Exact duplicate evidence SHALL distinguish raw bytes, normalized statement and
   shared native identity. Ambiguous cross-platform duplicates SHALL enter a
   human-review queue; title equality alone MUST NOT cause merging. Duplicate
   families SHALL be excluded from independent train/test or query leakage.
7. Full-statement MiniLM, statement TF-IDF, tag similarity and hybrid tag/semantic
   methods SHALL use the same frozen eligible IDs/candidate sets. Eligibility
   requires allowed use and actual language/field completeness; source/platform
   missingness and tag-only subset selection SHALL be reported explicitly.
8. MiniLM truncation and coverage SHALL be measured. A 256-token prefix is not a
   full-statement representation. Chunking/aggregation, section choices and query
   roles SHALL be frozen before scoring; solutions/editorials and hidden tests
   MUST NOT enter retrieval text.
9. Existing judgments SHALL be reused only with explicit applicability/basis and
   missingness. Additional candidates SHALL include the new statement methods,
   disagreements, hard negatives and random controls, with blinded review. The
   3,647 frozen pair records and all prior artifacts MUST remain byte-identical.
10. PCA/SVD SHALL be deferred until the original 384D full-statement baseline is
    established. No baseline optimization, model training, promotion, paid data
    purchase or production-index changes are authorized by this reprioritization.
