# Full-catalog similarity research on existing metadata — spec v1

Status: approved direction (2026-10-16). Implementation is research-only and local.
Inputs:
- [Phase 1 audit](../../research/full_catalog/AUDIT.md)
- [terms review](../../research/full_catalog/terms-review.v1.json): Codeforces and
  LeetCode collection rejected; AtCoder statements skipped by the user.
- [statement-collection.md](statement-collection.md): closed, not implemented.

This spec replaces the statement-based A/B/C plan in the original research brief.
No new network collection is part of it.

## 1. Research summary

| Platform | Records | Title | Tags | Difficulty | Statement usable |
|---|---:|---|---|---|---|
| LeetCode | 3,053 | yes | 2,992 | Easy/Medium/Hard | **no (D1)** |
| Codeforces | 11,457 | yes (42 Russian) | 11,277 | rating, 11,152 | no (never collected) |
| AtCoder | 9,787 | yes (2,218 Japanese) | none | none | no (never collected) |

The only signal common to all three platforms is the **title**. Tags exist for
LeetCode and Codeforces, but the two platforms use different vocabularies. Existing
judgments are `research/relationship-judgments.v1.json`: 180 AI-reviewed
(Luna draft, Sol review) pairs, 78 of them cross-platform, plus the 90-annotation
MiniLM pilot. These are evaluation-only and are not human gold labels.

The research question becomes: *how much cross-platform algorithmic
relationship can metadata alone recover, and where does it fail?* A negative
answer is an acceptable, publishable outcome. It would steer the architecture
toward curated concept annotations instead of text embeddings.

## 2. Decisions

- **D1 — stored LeetCode statements (3,053; `data/raw`, `data/processed`).**
  Quarantine. Do not delete.
  - The files stay byte-identical, so existing production jobs and their S3-synced
    index keep working, and audit hashes stay valid. The research does not delete
    data (repository rule).
  - New research MUST NOT read `content` or `content_clean`. It also MUST NOT
    read vectors derived from them (`indexes/numpy_leetcode/vectors.npy`, FAISS
    indexes) as experiment inputs. Earlier results (vector_geometry, Evaluation_100)
    stay as historical records, annotated with the provenance caveat. They are not reproduced.
  - The current production recommender still embeds the source note and searches
    statement-derived LeetCode vectors. Whether to rebuild, retire or keep that
    index is a **separate production decision**, outside this research spec. It is
    tracked as an open task, not decided here.
  - Rationale: the statements were obtained through `/graphql`. The current LeetCode
    terms forbid scraping and robots.txt disallows that path. Deleting the files is
    irreversible and would break production without being asked for. Continued
    research use would compound the issue.
- **D2 — LeetCode identity metadata** (slug, title, difficulty, tags).
  - Usable for research as already-held short factual metadata, with the provenance
    caveat recorded in every report. It MUST NOT be re-collected.
  - The user can veto this. Without D2, LeetCode drops out and the study becomes
    Codeforces↔AtCoder only, which has no shared tag vocabulary.
- **D3 — English-only encoder.** MiniLM-L6-v2 is used unchanged. Titles containing
  Cyrillic, kana or CJK characters (see `NON_LATIN` in `audit_catalog.py`) are
  `ineligible_language` for MiniLM runs. They remain in TF-IDF char-n-gram
  diagnostics and in every count. No translation is performed.

## 3. Requirements

### Phase 2 — identity and duplicates

- P2-1. THE SYSTEM SHALL produce `catalog.v1.jsonl`, with one row per source record
  (24,297). Each row carries the native key, platform, provenance pointer,
  `eligibility` (`eligible` or a reason code) and `representative_key`. No source
  record SHALL be removed.
- P2-2. Exact duplicates SHALL be limited to repeated platform keys (0 found) and
  reviewed alias mappings (`assets/problem_identity_aliases.v1.json`).
- P2-3. Same-title groups and the following structural patterns SHALL be emitted as
  `duplicate_candidate` groups, with evidence fields:
  - Codeforces Div1/Div2 mirrors: same normalized title, adjacent contest IDs and a
    different index;
  - AtCoder shared tasks: same title across concurrent ABC/ARC contests.

  Candidates SHALL NOT change eligibility. WHEN a candidate is confirmed by
  official-source evidence or manual review, it MAY become an exact duplicate in a
  new mapping version. That evidence SHALL be recorded.
- P2-4. Eligibility exclusions SHALL be rule-based and versioned
  (`research/full_catalog/exclusions.v1.json`), with a reason code:
  - LeetCode `Database`/`Shell`/`Concurrency`;
  - Codeforces' own `*special` tag (used instead of a hand-written contest list);
  - AtCoder APG4b tutorial, the `practice` contest, AHC heuristic contests and
    visualizer pages.

  The exclusion report SHALL count each reason per platform, and a row with several
  reasons SHALL list all of them. Non-Latin titles are an `encoder_flags` entry
  (`non_latin_title`), not an exclusion. MiniLM runs skip flagged rows (D3).

### Phase 3 — embeddings (metadata variants)

- P3-1. Variants replace the statement-based B/C:
  - **A1 title-only:** all eligible rows, all platforms.
  - **A2 structured metadata** `"{title}. Topics: {tags}."`: LeetCode and Codeforces
    rows with tags only. Difficulty is excluded from the text so that it stays
    available as a separate score component.
  - **B/C (statement):** not measurable. Reported as such, with the reason; no proxy text.
- P3-2. Use the pinned local model (`private-learning/models/minilm-1110a243…`) with
  offline flags, normalized float32 vectors, a fixed batch size and a fixed input order
  (sorted key).
- P3-3. Record:
  - token-length distribution and truncation rate (expected about 0 for titles);
  - encode throughput;
  - peak RSS;
  - library versions, model hash, input hash and vector SHA-256.

  Vectors stay in `private-learning/full-catalog/`. Committed receipts hold hashes only.

### Phase 4 — exhaustive retrieval

- P4-1. Exact blockwise cosine (query blocks of ≤ 1,024 × the full target matrix),
  never materializing N×N. Cross-platform: each query against all eligible rows of the
  other two platforms. Same-platform: a separate optional diagnostic.
- P4-2. Exclude self, rows in the same exact-duplicate group, and rows in the
  query's `duplicate_candidate` group. The candidate exclusion is reported separately,
  so its effect is visible.
- P4-3. Ranking is score descending, with ties broken by key ascending. Save Top-10,
  Top-20 and Top-50 (key, score, rank) for every method. Re-running SHALL reproduce
  byte-identical results files.

### Phase 5 — evaluation

- P5-1. Baselines on exactly the same rows and inputs as each embedding variant:
  - TF-IDF word uni+bigram;
  - TF-IDF char 3–5-gram;
  - for A2 only: tag-Jaccard. The two platforms' tags are mapped through a committed,
    reviewed CF↔LC tag crosswalk. Pairs with no crosswalk are reported as unmatched.
  - A random baseline at the same K.
- P5-2. Judged pool v2: the union of Top-10 lists from all methods for a fixed,
  stratified query sample (seeded hash; strata by platform pair and difficulty band;
  ≤ 150 queries).
  - Graded 0–3 (`unrelated`, `partially_related`, `relevant`, `strongly_relevant`),
    plus a `deceptive` flag for title/narrative matches with a different algorithm.
  - The label source is recorded per pair: `ai_reviewed` (draft plus independent
    review, the existing practice) or `human`.
  - Labelers see only metadata and public URLs. The quarantined statements are not shown.
- P5-3. Metrics are reported only over judged pairs:
  - Precision@K and nDCG@K, using condensed lists (unjudged pairs removed) and also
    with unjudged pairs treated as missing;
  - judged-coverage@K for every method.

  Unjudged pairs are NEVER counted as negatives. Full-catalog estimates are labelled
  as estimates and given with a bootstrap CI over queries. The existing 180
  judgments are reported separately as a fixed set.
- P5-4. Diagnostics:
  - tag-consistency@K (LeetCode↔Codeforces through the crosswalk; reported as
    circular for A2 and tag-Jaccard);
  - difficulty-gap distribution;
  - per-platform-pair hubness (k-occurrence skew);
  - hard-negative error analysis on `deceptive` pairs;
  - failure examples per stratum.
- P5-5. No model is promoted. The output is a comparative report and an architecture
  proposal (Phase 6) grounded in these numbers.

## 4. Design

```text
research/full_catalog/
  audit_catalog.py              (Phase 1, exists)
  catalog.py                    P2: build catalog.v1 + duplicate/exclusion report
  exclusions.v1.json            committed rule lists (contest IDs, prefixes, tags)
  tag_crosswalk.v1.json         committed CF<->LC tag map (reviewed)
  embed.py                      P3: A1/A2 encoding + resource receipt (.venv)
  retrieve.py                   P4: blockwise exact top-K (numpy only)
  baselines.py                  P5: TF-IDF/Jaccard/random on identical rows
  pool.py, evaluate.py          P5: judged-pool sampling and metrics
  test_*.py                     fixture tests (no private data needed)
  REPORT.md, full_catalog.ipynb, *.png, receipts/*.json   (aggregates only)
private-learning/full-catalog/  vectors, top-K files, catalog rows (ignored)
```

Every step reads only the hashed inputs: the CF/AtCoder snapshot and the
LeetCode metadata fields (D2). Each script refuses to run if any input hash differs
from `audit.json`. No step imports production code or reads statement fields. A
test asserts that `content`/`content_clean` are never accessed.

Memory: 24k × 384 float32 is about 37 MB. A 1,024 × 24k score block is about
100 MB. This fits easily on this machine (24 GB). TF-IDF uses sparse matrices with
the same blockwise top-K.

## 5. Task breakdown

- [x] 1. Exclusion rules and CF<->LC tag crosswalk drafted (`exclusions.v1.json`, `tag_crosswalk.v1.json`). Status: agent draft, pending user review.
- [x] 2. `catalog.py` and 14 fixture tests: catalog.v1, duplicate candidates, exclusion report. Two runs produced byte-identical output.
  Receipt: `research/full_catalog/receipts/catalog.v1.json`.
  - Eligible rows: 23,454 of 24,297. MiniLM-eligible rows: 21,297.
  - Exclusions: 104 LeetCode, 560 Codeforces and 179 AtCoder.
  - Duplicate candidates: 842 same-title groups and 379 cross-platform exact-title matches (evidence only). There are 0 exact duplicates.
  - Only 4 Codeforces same-title groups fall in adjacent contests. The one inspected (505B/506D) has different indices, so it is likely an easy/hard pair rather than a mirror. This needs review; it is not merged.
- [x] 3. `retrieve.py` and 9 fixture tests: brute-force equality across modes, block sizes and candidate settings; exact boundary ties; exclusions; non-finite guard; prefix outputs.
  Full-scale synthetic check (23,454 eligible keys, random 384-d vectors, not an embedding):
  - 6.2 s and about 520 MB peak RSS at block size 1,024;
  - 0 mismatches against a float64 brute force on 500 sampled queries;
  - 0 same-platform leaks;
  - byte-identical output on rerun.
- [x] 4. `inputs.py` (the shared row and text builder for every method) and `baselines.py`, with 10 fixture tests. Cross-platform runs completed and are byte-identical on rerun. Receipts: `receipts/baselines-{A1,A1-all,A2}-cross.v1.json`.
  - A1 (21,297 rows): TF-IDF word and char, random.
  - A1-all (23,454 rows): TF-IDF char, random.
  - A2 (13,590 LeetCode/Codeforces rows): TF-IDF word and char, tag-Jaccard, random.

  Each run took ≤ 20 s and about 1 GB peak RSS.

  Observed limits, to report in Phase 5:
  - 7 A1 titles have no word tokens.
  - 403 A2 rows have an empty concept set: LeetCode tags that are only unmapped, such as Array or Hash Table. Their lists are pure key-order ties.
  - Tag-Jaccard produces large score-1.0 tie blocks. Its within-tie order is key order, not relevance, so evaluate.py must report tie-block size per query.
- [x] 5. `embed.py` with 5 fixture tests (38 total in research/full_catalog).
  - Model: pinned MiniLM, hash-verified, offline, CPU, 1 thread, batch 64.
  - A1 (21,297 titles): 3–33 tokens (p95 10), 0 truncated, 1,538 texts/s, about 0.88 GB peak RSS.
  - A2 (13,590 rows): 8–48 tokens (p95 28), 0 truncated, 648 texts/s, about 0.95 GB peak RSS.
  - Both encodes replayed to byte-identical vectors with no warnings.
  - Receipts: `receipts/embeddings-{A1,A2}.v1.json`. The vectors stay private.
- [x] 6. Cross-platform P4 has been run for every method:
  - MiniLM: `receipts/retrieval-minilm-{A1,A2}-cross.v1.json`. Each run took ≤ 5 s and was byte-identical on rerun.
  - Baselines: see task 4.

  The same-platform diagnostic has not been run; it is optional. Top-K lists stay private.
- [ ] 7. `pool.py`: freeze the query sample and pooled pairs (committed key list); labeling round.
- [ ] 8. `evaluate.py`: judged-pool metrics, CIs, diagnostics; notebook and plots.
- [ ] 9. REPORT.md and architecture proposal; update AGENTS.md and tasks.md.

## 6. Validation plan

- `.venv/bin/python -m unittest discover -s research/full_catalog -v`: fixture-only, no private data, no sockets. Retrieval tests need numpy; the catalog tests are stdlib-only.
- `python3 -m unittest discover -s tests -v` and `python3 scripts/learning/assets.py verify` still pass.
- Determinism: every script run twice gives identical output hashes.
- Brute-force cross-check of blockwise top-K on a 500-row random sample per method.
- Hash check that `data/`, `indexes/`, `assets/` and catalog snapshots are unchanged.
- Committed artifacts are scanned so that no `content`, `content_clean` or `full_statement` text appears.

## 7. Risks

- Titles are a weak signal: expect generic titles ("Game", "Two Arrays") and narrative
  titles that are unrelated to the algorithm. This is the main expected finding, not a defect.
- Language exclusion removes about 23% of AtCoder rows. The bias is reported per stratum.
- The tag crosswalk is a judgment artifact, so it is reviewed and versioned.
  Tag-based metrics are partly circular.
- AI-reviewed labels are not human gold, and pool bias favors methods that
  contributed to the pool. Report judged coverage and do not over-interpret small differences.
- D2 provenance: LeetCode metadata also came via `/graphql`. If the user vetoes D2,
  the plan reduces to Codeforces↔AtCoder.

## 8. Open items (not decided here)

- Production index built from quarantined statements (`indexes/numpy_leetcode`,
  S3 copy, `generate_index_map.py`): keep, rebuild from metadata, or retire.
- AtCoder difficulty via the documented AtCoder Problems `problem-models` endpoint:
  metadata only, needs its own small review if wanted.
