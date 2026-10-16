# Checkpoint 3 — executed publisher-description similarity baseline

The matched corpus is **9,424 rows: 8,101 Codeforces and 1,323 AtCoder**.
All 8,097 resolved native IDs are Codeforces; 7,077 match the existing native
catalog. AtCoder publisher identities are unresolved and remain dataset-keyed.
LeetCode text is not admitted. Thus this is an available two-platform publisher
cohort, not full native-catalog or three-platform statement coverage.

[The run receipt](statement-baseline.v1.json) binds the corpus, pinned model,
embedding matrices, and exact Top10/20/50 outputs. Word TF-IDF, character TF-IDF,
MiniLM title, title+description prefix, and original-token chunk pooling all use
identical query/candidate rows. All 9,424 queries are retrieved cross-platform.
The 1,007 exact normalized-content duplicate groups (2,050 rows) remain stored;
identical content is excluded from recommendations. Ambiguous semantic matches
are not removed or merged.

There are 4,410,241 content tokens and 22,034 disjoint content windows. Median
length is 441 tokens, maximum 2,755; **8,106/9,424** descriptions exceed 254 tokens.
The chunk method covers every original token; publisher section completeness is
unverified. Actual baseline runtime is **336.60 seconds**, process peak RSS
**1,608,040,448 bytes (1.50 GiB)** on the local machine. This is not a claim of
fitness for the user's 1 GiB OCI instance. MiniLM encoding took 3.46 seconds for
titles, 64.80 seconds for prefixes, and 146.14 seconds for complete chunk pooling.
TF-IDF fit/retrieval took 10.05 seconds (word) and 96.41 seconds (character).

## Evaluation and review dataset

[Evaluation receipt and pending review dataset](evaluation.v1.json) includes
K=10/20/50, explicit judged denominators, query-bootstrap intervals where defined,
precision bounds, hubness, platform diversity, and missing diagnostic coverage.
All methods have **zero judged retrieved slots**. Precision and nDCG are null;
precision bounds are [0,1]. This is missing evidence, not evidence of poor or good
retrieval. AtCoder's missing native identity prevents linkage to prior labels.

The pending human-review dataset has **1,000 unlabelled pairs**, drawn from a
seeded maximum 100 queries per platform and balanced across available method,
platform, difficulty, and topic strata. Difficulty/topic are unavailable for
cross-platform endpoints here; no artificial strata are invented. A separate
**200-pair** queue records top-10 agreement across methods as possible semantic
overlap, explicitly unverified. No human judgments were manufactured.

Top10 maximum candidate indegrees: title MiniLM 1,414; prefix MiniLM 425;
chunk MiniLM 484; character TF-IDF 1,316; word TF-IDF 2,436. These concentration
statistics do not establish relevance. Candidate platform diversity is necessarily
one per query in this two-platform cross-retrieval cohort.

## Blocked comparisons

AtCoder tags are unknown and LeetCode statements quarantined. Cross-platform tag,
hierarchical taxonomy, and tag+semantic comparisons therefore have no matched
eligible cohort. Their status is explicit in the receipt; missing tags are not
negatives. The taxonomy and generic scoring helpers remain available for a future
properly annotated cohort. No model, weight, or production policy is promoted.
