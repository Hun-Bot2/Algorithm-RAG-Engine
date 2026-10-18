# Statement availability and acquisition feasibility

**Reviewed: 2026-10-16. Decision: statement-first research; acquisition clearance is the next gate.**

The title-only and title+tag runs are preserved preliminary baselines. No additional
baseline tuning, retrieval run, model promotion or dimensionality reduction was
performed for this report. The requested target remains full-statement algorithmic
similarity with a shared, provenance-preserving tag taxonomy across all three
platforms. Missing statements will not be replaced by titles or generated text.

## 1. What is actually available locally

| Platform | Catalog records | Statement availability | Constraints / samples | Original algorithm tags |
|---|---:|---|---|---|
| LeetCode | 3,053 | 3,053 nonempty historical HTML `content`; processed `content_clean` also populated. Quarantined for research. Completeness is not yet validated. | No separately persisted constraint/sample fields. Material may be embedded in HTML; marker counts are only heuristics, not extracted-section coverage. | 2,992 tagged; 61 empty. 72 distinct tags. |
| Codeforces | 11,457 | 0 statements; `statement_status=not_provided`. | 0 separately stored constraints/samples. | 11,277 tagged; 180 empty. 38 distinct tags. |
| AtCoder | 9,787 | 0 statements; `statement_status=not_provided`. | 0 separately stored constraints/samples. | All 9,787 unavailable/null. No native algorithm-tag evidence in this dataset. |

The LeetCode source files lack persisted paid-status, per-statement language,
content hash, section completeness and collection-status fields. Historical file
hashes and collection-code attribution exist; they do not establish current access
rights or complete original acquisition provenance. Existing index metadata has
matching tags for all 3,053 slugs. Script-based language checks are heuristics,
not verified English-language or mathematical-completeness judgments.

All seven legacy `Evaluation_100/data/*.jsonl` files have 100 populated `content`
records each. The three LeetCode variants cover the same 100 IDs/slugs already in
the 3,053-record catalog, so they add **zero unique LeetCode problems**. The other
four are Baekjoon material outside the requested three-platform corpus. Cleaned
and LLM-refined `embedding_text` fields are derived representations, not additional
original statements. None stores constraints/samples as separate fields.

The external catalog has provenance at the **run/response** level in its existing
manifest, including URLs, retrieval timestamps and hashes. That provenance is not
yet copied into each normalized problem record. Full measured field counts,
marker definitions, caveats and source hashes are in
[availability-audit.v1.json](availability-audit.v1.json).

## 2. Access channels and current acquisition decisions

“Technically reachable,” “robots-allowed,” “publisher-licensed,” and “cleared for
this project's intended reuse” are distinct findings. The decisions below are
project acquisition gates, not a determination that every possible research use
is lawful or unlawful. An AI-training opt-out is not automatically a prohibition
on all fixed-model inference, and absence of a robots rule is not reuse permission.

| Platform | Official APIs | Other channels | Decision for new statement collection |
|---|---|---|---|
| LeetCode | No documented public full-statement API was identified in this review. | Historical collector uses undocumented `/graphql`; public problem pages and third-party mirrors also exist. | **Do not collect directly.** Current terms explicitly prohibit crawling/scraping; robots disallows `/graphql` and `/api/`. Existing statements stay quarantined, byte-identical. Seek a rights-cleared export/license; do not refresh Premium status by scraping. |
| Codeforces | Documented `problemset.problems` exposes metadata/tags, not statements; at most one API call per two seconds. | HTML statement pages; publisher-distributed research datasets containing Codeforces material. | **Not cleared for direct bulk collection.** `/contest` is disallowed; named scraper clients are limited to `/api/`, and named AI crawlers are blocked. The wildcard group does not explicitly disallow `/problemset/problem/`; this fact does not resolve the prior rejection or intended-use/license question. No custom identity is used to evade these signals. |
| AtCoder | No official public full-statement API was identified. AtCoder Problems is an unofficial community metadata source, not a statement API. | Official task HTML and a first-party licensed dataset program. | **Needs an appropriate use basis/license.** Task paths are not disallowed for the wildcard group, but current terms reserve rights in service text. The official dataset program offers Japanese/English statements via a commercial contract; no license is held. |

Primary sources checked:

- [LeetCode terms](https://leetcode.com/terms/), especially Use of Services restrictions and Content;
  [robots](https://leetcode.com/robots.txt).
- [Codeforces API](https://codeforces.com/apiHelp),
  [Problem object](https://codeforces.com/apiHelp/objects#Problem),
  [terms](https://codeforces.com/terms), [robots](https://codeforces.com/robots.txt).
- [AtCoder terms](https://atcoder.jp/tos?lang=en), Intellectual Property Rights;
  [robots](https://atcoder.jp/robots.txt);
  [AtCoder Problems documentation](https://github.com/kenkoooo/AtCoderProblems/blob/master/doc/api.md).
- [AtCoder's first-party dataset program](https://datasets.atcoder.jp/) expressly
  advertises licensed statements. Its current page targets companies and asks for
  a budget of at least $100,000. This is a verified licensing route, **not** a free
  dataset or a purchase recommendation. No payment or contact was initiated.

All three robots files were fetched for this review: HTTP 200, with hashes matching
the prior review. Only policy documents were requested; no statement canary was
fetched. Machine-readable evidence and exact hashes are in
[acquisition-review.v2.json](acquisition-review.v2.json).

## 3. Published datasets: availability is real, clearance needs scope review

| Candidate | What the publisher documents | Remaining issue / next action |
|---|---|---|
| [CodeContests](https://github.com/google-deepmind/code_contests/blob/main/README.md#license) | Contains descriptions from Codeforces and AtCoder/CodeNet among other sources. Its README explicitly licenses non-code material CC BY 4.0, separately from Apache-licensed code. | Strongest documented candidate for a dataset import. Retain attribution and its explicit separate-third-party-terms/no-rights-representation notice. That notice leaves upstream scope unresolved under the project's prior clearance gate. Verify original-source/use scope before enabling import; LeetCode is not among its listed sources. |
| [Project CodeNet](https://github.com/IBM/Project_CodeNet#problem-descriptions) | Documents HTML problem descriptions and original Aizu/AtCoder sources. | Verify the actual dataset's terms and the description archive's scope; the tools repository's Apache license alone is insufficient. Stable mapping from dataset IDs to native task IDs must also be checked. |
| [APPS](https://github.com/hendrycks/apps) | Publisher links a separate problem dataset; repository includes training/evaluation code. | Repository MIT licensing does not by itself establish permission for every original statement. Inspect dataset-specific terms/source inventory before use. |
| [TACO](https://github.com/FlagOpen/TACO) | Publisher describes a multi-source benchmark, including APPS/CodeContests and the target platforms. | Preserve original-source conditions and distinguish its dataset license claim from source-owner clearance. Do not treat it as an automatic cure for blocked collection. |

These sources were investigated, **not downloaded**. We have not concluded that
all community dataset use is prohibited. We have also not marked any of these
sources as unconditionally cleared merely because a reputable publisher distributes
it or a repository displays an open-source badge. The previous source rejection
record remains preserved; this new review reopens investigation rather than
silently rewriting that decision.

## 4. What will be collected after clearance

Implement the collector only for the source/use scope that passes review. Each
private record will retain native platform/task ID and aliases, canonical URL,
publisher/dataset revision, license and permitted-use evidence, source language,
raw SHA-256, normalized content SHA-256, parser version, retrieval time and status.
Legend, constraints, input/output, public samples and notes will be separate; absent
fields, image dependencies and parse/language failures stay explicit. Preserve math
markup. Do not load solutions/editorials or mistake hidden judge cases for samples.

A run will be opt-in, serial, rate-capped and immutable, with bounded responses,
stop-on-login/Premium/403/429/challenge and hash-verified offline replay. No proxy,
identity rotation, alternate endpoint or mirror will bypass a restriction. Rate
limits belong to each actual source; the Codeforces API limit does not grant HTML
scraping permission. Redistribution remains disabled unless specifically allowed.

## 5. Tag normalization, duplicates and the subsequent experiment

The existing `tag_crosswalk.v1.json` remains an **agent draft baseline**. Its
entries already distinguish equivalent/narrower/related, but it is not a complete
three-platform taxonomy. Do not equate CF `matrices` with LC `Matrix`, infer every
narrow data structure from CF `data structures`, or treat missing AtCoder tags as
an empty ground-truth concept set. The new taxonomy will preserve raw tags and
reviewed mappings, including broad/narrow/related/unmapped relationships and source.
New inferred/curated AtCoder tags must be separately labeled annotations.

Exact duplicate detection will distinguish identical bytes, normalized statement
hashes and native task reuse. Same titles and similar wording generate a
**human-review queue**, not automatic merges. Duplicate groups must be considered
in corpus membership and leakage controls.

Once statements are available, freeze the same eligible IDs and section/language
policy for statement MiniLM, statement TF-IDF, tags and hybrids. Report a statement-
complete cohort and the more restricted tag-covered cohort separately. MiniLM's
256-token limit requires measured coverage and a frozen chunk/aggregation policy;
a prefix-only encoding cannot be called a full-statement baseline.

Reuse applicable existing judgments with their original metadata-based evidence
and abstentions intact. Create an additional blinded pool from the statement
methods, their disagreements, hard negatives and random controls. **Only after the
original 384D statement baseline is established will PCA/SVD be tested.**

## 6. Preservation and immediate next task

[BASELINE-PRESERVATION.v1.json](BASELINE-PRESERVATION.v1.json) hashes **191 protected
files** covering existing research runs, receipts, private artifacts, raw catalogs,
source data, vectors and judgments. Before/after hashes match. The frozen pool has
3,647 distinct pair records: **2,886 non-null judgments and 761 unjudged/abstained**.
Their original evidence, provenance and flags are unchanged; nulls are not negatives.
No records have been discarded, upgraded to statement-grounded judgments or relabeled.

**The next task is source clearance/acquisition, not another metadata experiment.**
No new statement source currently passes the project's documented access-and-use
gate. Therefore no live statement collector was activated or bulk statement data
requested. A platform authorization or a dataset-specific clearance decision is
needed before that conditional implementation step can proceed. User approval
alone cannot grant third-party rights. We can prepare permission-request drafts,
but will not contact platform owners or purchase data without explicit instruction.

The new governing sequence is in [requirements](../../specs/statement-similarity/requirements.md),
[design](../../specs/statement-similarity/design.md) and
[tasks](../../specs/statement-similarity/tasks.md). The older metadata experiment
specification and terms-review decision remain historical baseline artifacts.
