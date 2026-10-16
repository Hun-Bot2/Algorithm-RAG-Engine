# Full-catalog similarity research: Phase 1 catalog audit

Date: 2026-10-16. This audit is offline only. It made no network calls, ran no model inference and changed no input data.
Machine-readable receipt: [audit.json](audit.json), produced by
`python3 research/full_catalog/audit_catalog.py --output research/full_catalog/audit.json`.
Two runs produced byte-identical output.

## Inventory

| Platform | Source (all private, untracked) | Records | Statement present | Tags present | Native difficulty |
|---|---|---:|---:|---:|---:|
| LeetCode | `data/raw/leetcode_raw_data.jsonl` + `data/processed/leetcode_processed.jsonl` (collected Jan 2026) | 3,053 | 3,053 (HTML, English) | 2,992 (61 empty) | 3,053 |
| Codeforces | `private-learning/catalogs/2026-10-16-api-v1-reviewed-identities/` | 11,457 | **0** | 11,277 (180 empty) | 11,152 (305 null) |
| AtCoder | same snapshot (kenkoooo community dataset) | 9,787 | **0** | **0** | **0** |
| **Total** | | **24,297** | **3,053 (12.6%)** | | |

The often-quoted figure of 21,244 is **Codeforces + AtCoder only**. It does not include LeetCode. Every record in the 21,244 has `statement_status: not_provided`.

Other stored text:
- `curriculum/annotations.v1.json` and the MiniLM pilot hold 90 authored English annotations. They are not statements, so they are excluded from statement-based experiments.
- `indexes/numpy_leetcode/` holds the same 3,053 LeetCode keys, embedded with OpenAI vectors. That vector space cannot be compared with MiniLM.

## LeetCode Premium (paid-only) trace

- Filter: `scripts/collection/leetcode_data_collection.py` requests `isPaidOnly` in the GraphQL `questionList` query. It then skips any problem where `if prob['isPaidOnly']:` is true, before it fetches details. It also skips problems whose detail response has empty `content`.
- The `isPaidOnly` flag is **not persisted**. Nothing in the repository stores a problem-list snapshot or a skip log, and `data/raw/leetcode_slugs.txt` is absent.
- Evidence from the stored data:
  - All 3,053 records have non-empty statements.
  - 0 records contain premium-gate text. One match on the word "premium" (`find-churn-risk-customers`) is a SQL plan name, not a gate.
  - IDs run from 1 to 4,195, with no duplicate IDs or slugs.
- **Conclusion:** the stored records are consistent with free-only collection. However, the paid status of the **1,142 IDs that are absent below the maximum ID is unknown**. The cause could be paid-only status, empty content, a fetch failure or the run's cutoff. This cannot be resolved offline. Resolving it needs one fetch of LeetCode's public problem-list metadata (slug and `isPaidOnly` only, no statements).
- The current LeetCode collection is about 9 months older than the CF/AtCoder snapshot. Problems added since January 2026 are missing.

## Eligibility and data-quality findings

1. **Statements:** only LeetCode has statements. Therefore experiments B (title + statement) and C (chunked statements) **can currently run only on the 3,053 LeetCode problems**. Cross-platform B/C retrieval has no eligible target set.
2. **Experiment A (title-only)** can cover all 24,297 records. Titles are a weak and unequal signal:
   - AtCoder has 2,218 non-Latin titles (Japanese); Codeforces has 42 (Russian).
   - MiniLM-L6-v2 is English-only, so these titles need either an explicit exclusion or a separate flag.
   - AtCoder also includes non-problem entries, such as tutorial and APG4b pages (e.g. `1.00.はじめに`) and a visualizer (`ビジュアライザ`).
3. **Non-algorithmic LeetCode problems:** 94 Database, 6 Concurrency and 4 Shell problems. They should be excluded from algorithm-similarity evaluation, but kept in the catalog with a recorded reason.
4. **Exact duplicates:**
   - No repeated platform IDs exist on any platform.
   - LeetCode has 0 duplicate titles and 0 exact normalized-statement duplicates.
   - CF has 435 duplicate-title groups (1,102 rows); AtCoder has 407 (960 rows). Examples are generic names such as "Game", plus Div1/Div2 mirrors and ABC/ARC shared tasks.
   - A shared title is **not** duplicate evidence. Without statements, CF and AtCoder duplicates cannot be confirmed beyond structural rules (shared-task aliases, adjacent-contest mirrors) and manual review.
5. **Provenance:**
   - CF/AtCoder: the snapshot has raw-response hashes and replay receipts (`assets/cross_platform_sources.v1.json`).
   - LeetCode: the JSONL files have no retrieval timestamps or response hashes. `audit.json` now pins their SHA-256 as the frozen research input.

## Consequences for the research plan

Without new data, Phases 3–5 reduce to:
- A: full-catalog title-only retrieval across 3 platforms (weak, with language imbalance);
- B/C: LeetCode-only, used as a same-platform diagnostic.

None of these measures cross-platform statement similarity. Before Phase 2 runs, there are two data decisions to make:

1. **Statement source for Codeforces and AtCoder.** Statement scraping was explicitly kept out of scope in the corpus spec (`specs/adaptive-learning/cross-platform-corpus.md`). Any collection needs a rights and terms review, rate limits and a clear decision about a bounded subset versus the full catalog.
2. **LeetCode list refresh.** One metadata-only fetch would record `isPaidOnly` for all problems and settle the Premium question.

No deduplication, embedding or retrieval artifacts have been produced yet.

Caveat: the stored `id` is the GraphQL `questionId`. It can differ from the frontend
problem number, so the 1,142 absent IDs are an upper-bound signal, not a count of
missing problems.

Next step: the draft [statement-collection spec](../../specs/adaptive-learning/statement-collection.md)
(terms review gate, bounded subset, metadata-only LeetCode refresh). Nothing has been fetched.

## Update 2026-10-16: decisions

The user approved the terms review. Codeforces and LeetCode collection, and the
LeetCode `isPaidOnly` refresh, are rejected; AtCoder statements are skipped. The
LeetCode paid status of absent IDs therefore stays **unknown**. The 3,053 stored
statements are quarantined (kept, not used by new research). Phases 2-5 continue
on metadata only: [full-catalog-similarity.md](../../specs/adaptive-learning/full-catalog-similarity.md).
