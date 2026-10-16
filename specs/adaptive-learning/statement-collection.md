# Bounded statement collection and LeetCode paid-status refresh — spec v1 (draft)

Status: **closed — rejected 2026-10-16** (user approved the terms-review
recommendations; see `research/full_catalog/terms-review.v1.json`). Superseded by
[full-catalog-similarity.md](full-catalog-similarity.md). Kept as a record.
Original status: draft, not approved. No fetch, scrape or model run is authorized by this
document. Implementation starts only after the Terms Review Gate (section 4) is
recorded as passed and the user approves the bounded subset (section 5).

Context: [Phase 1 audit](../../research/full_catalog/AUDIT.md) found that the
21,244 Codeforces/AtCoder records contain 0 statements, while 3,053 LeetCode
records do. Because of this, the title+statement (B) and chunked-statement (C)
experiments currently have no cross-platform targets. The audit also found that
`isPaidOnly` was used as a filter but never persisted, which leaves 1,142 absent
LeetCode IDs with unknown status. This spec extends
[cross-platform-corpus.md](cross-platform-corpus.md), which put statement
ingestion out of scope until a separate source and coverage design existed.

## 1. Research summary

- Codeforces: the documented API (`problemset.problems`) returns metadata only.
  Statements are available only as HTML pages at
  `https://codeforces.com/problemset/problem/{contestId}/{index}` (or under `/contest/`).
  The API documents one call per two seconds. HTML pages have no documented rate,
  so this spec sets a stricter one.
- AtCoder: the kenkoooo dataset is unofficial and contains metadata only. Statements
  are available as HTML at the task URLs already stored in the snapshot
  (`https://atcoder.jp/contests/{contest}/tasks/{task}`). The English text, where it
  exists, is served from the same page with `?lang=en`. Older and some
  non-ABC/ARC/AGC contests may be Japanese-only. This must be measured, not assumed.
- LeetCode: `scripts/collection/leetcode_data_collection.py` uses the unofficial
  GraphQL `questionList` query, which returns `isPaidOnly` per problem without
  statements. `leetcode_check_new_fast.py` instead reads a third-party mirror
  (`noworneverev/leetcode-api` on GitHub). Its provenance is secondary and its update
  time is unverified.
- No public statement source here carries a redistribution license. Being reachable
  over the network does not grant rights to store or redistribute the content
  (same principle as `cross-platform-corpus.md`).

## 2. Requirements

### 2.1 General

- R1. THE SYSTEM SHALL NOT perform any network request for statements or
  LeetCode metadata until the gate record (`research/full_catalog/terms-review.v1.json`)
  shows every item in section 4 as `passed` and names the approving user and date.
- R2. Collection SHALL be opt-in per run (`--fetch`), standard-library only, and
  write to a new run directory under the git-ignored `private-learning/statements/`.
  Existing run directories MUST NOT be overwritten.
- R3. Raw response bytes, URL, retrieval timestamp, HTTP status, final URL after
  redirects, byte count and SHA-256 SHALL be persisted per request. Normalized output
  SHALL be reproducible byte-for-byte from the raw captures through offline replay.
- R4. Statements, raw HTML and derived text MUST remain private. They MUST NOT be
  committed to Git, copied into `hosted/public/`, OCI bundles, release assets or
  published reports. Committed receipts MAY contain only counts, hashes, IDs and
  missingness.
- R5. THE SYSTEM SHALL NOT invent, translate, summarize or LLM-rewrite a statement.
  A missing, Japanese-only, image-only or unparsable statement SHALL be recorded
  with an explicit status. It SHALL NOT be filled in.
- R6. Problem keys SHALL use the existing native forms (`codeforces:4A`,
  `atcoder:arc065_a`). Alias mappings in `assets/problem_identity_aliases.v1.json`
  SHALL be read, not rewritten.

### 2.2 Rate limiting and failure handling

- R7. Requests SHALL be strictly serial, with at least **10 seconds** between
  request starts per host. A run SHALL be capped at **1,200 requests per host**
  (about 3.5 hours). The interval and cap MAY be tightened, but never loosened,
  without amending this spec.
- R8. WHEN a response is HTTP 403, 429 or 503, contains a CAPTCHA or anti-bot
  challenge, or redirects to a login page, THE SYSTEM SHALL stop that host's run
  immediately, persist the response as evidence and exit non-zero. It MUST NOT
  retry, rotate identity, add cookies or attempt any bypass.
- R9. Other errors (timeout, 5xx other than 503, parse failure) SHALL be recorded
  per item. They SHALL NOT be retried within the same run. A later run MAY retry
  only the items recorded as failed.
- R10. Requests SHALL carry a fixed, honest User-Agent that identifies the project
  (pattern of `catalog_sources.bounded_fetch`). They SHALL use no credentials or
  session cookies, a 30-second timeout and a 5 MiB response cap.
- R11. WHERE a host's `robots.txt` disallows the path, THE SYSTEM SHALL skip it.
  The `robots.txt` bytes and hash in effect at run start SHALL be stored in the run receipt.

### 2.3 Statement normalization

- R12. Extraction SHALL be deterministic and parser-versioned. It SHALL produce the
  following separated fields: `legend`, `input_spec`, `output_spec`, `samples`,
  `notes`, `constraints`, plus a `statement_text` built by concatenating them in that
  fixed order. Field boundaries come from documented page structure.
  WHEN the structure is not recognized, THE SYSTEM SHALL emit
  `statement_status: parse_failed` instead of guessing.
- R13. Math markup (Codeforces `$$$…$$$`, AtCoder `<var>`/KaTeX) SHALL be kept as
  source text, with a recorded normalization rule. Images SHALL be counted
  (`image_count`) and their `src` recorded, but not downloaded.
- R14. Each record SHALL include `language` (`en`, `ja`, `ru`, `mixed`) detected by a
  documented character-class rule. AtCoder records SHALL include `en_available: bool`.
  Experiment B/C eligibility requires `language == "en"`.
- R15. Each record SHALL include `content_sha256` of normalized `statement_text`, for
  exact-duplicate detection in Phase 2. Equal hashes are duplicate evidence, but
  they do not trigger an automatic merge.

### 2.4 LeetCode paid-status refresh (metadata only)

- R16. The refresh SHALL request only list fields: `questionId`, `questionFrontendId`,
  `title`, `titleSlug`, `difficulty`, `isPaidOnly`. It MUST NOT request `content`,
  hints, solutions or code snippets.
- R17. Pages SHALL be fetched serially, at least 5 seconds apart, with a cap of
  100 pages. Each page SHALL be persisted raw. WHEN `totalNum` is inconsistent
  across pages, or a page is short before the end, the run SHALL fail. It SHALL NOT
  produce a partial "complete" list.
- R18. The refresh SHALL write a reconciliation receipt covering:
  - every stored record (expected `isPaidOnly == false`);
  - every one of the 1,142 absent IDs, classified as `paid`, `free_not_collected`,
    `not_in_list` or `id_mismatch`;
  - every problem published after the January 2026 collection.

  A stored record that is now `isPaidOnly == true` SHALL be reported. It SHALL NOT
  be deleted.
- R19. Existing `data/raw/` and `data/processed/` files, `indexes/` and production
  jobs MUST remain byte-identical. The refresh SHALL NOT append to
  `leetcode_raw_data.jsonl`. Collecting new free statements is a separate, later decision.
- R20. The third-party mirror MAY be used only as an offline cross-check, with its
  commit SHA recorded. It MUST NOT be used as the primary paid-status source.

## 3. Design

```text
research/full_catalog/terms-review.v1.json      (committed gate record, R1)
research/full_catalog/subset.v1.json            (committed selected keys + selection hash)
src/learning/statement_sources.py               (pure parsers + bounded transport)
scripts/learning/collect_statements.py          (CLI: --plan | --fetch | --replay)
scripts/learning/refresh_leetcode_paid.py       (CLI: --fetch | --replay)
private-learning/statements/<run>/              (ignored)
    manifest.json   robots/<host>.txt   raw/<key>.html   statements.jsonl   failures.jsonl
private-learning/leetcode-list/<run>/           (ignored)
    manifest.json   raw/page-NNN.json   list.jsonl   reconciliation.json
research/full_catalog/statement-coverage.v1.json (committed aggregate receipt only)
```

- Transport: reuse the shape of `catalog_sources.bounded_fetch` (stdlib `urllib`, size
  cap, timeout), and add per-host interval, cap and stop conditions (R7–R11). Keep it
  as a new function so the existing metadata collector is unchanged.
- `--plan` performs no network calls. It prints the exact URL list, per-host request
  count and minimum duration. The plan hash goes into the gate record, so approval
  binds to an exact URL set.
- Parsers are pure functions from `(bytes, parser_version)` to a record or a failure.
  They are tested only against fixtures. The fixtures are small hand-written HTML
  imitating the page structure, not copied statements.
- Replay re-parses `raw/` with hash verification and must reproduce
  `statements.jsonl` exactly (same contract as catalog replay).
- Downstream (Phase 2–5) reads `statements.jsonl` together with the metadata
  snapshot. It never reads `raw/`.

Tradeoffs:
- Serial requests at 10 s are slow (about 3.5 h per host for the cap). This is
  deliberate: correctness and politeness outrank speed, and the subset is bounded.
- Structured field extraction costs more parser work than raw text, but it allows
  ablations such as legend-only versus with samples. It also prevents sample I/O
  from dominating similarity.
- Japanese-only AtCoder tasks are kept as records but made ineligible for English
  MiniLM. Coverage bias therefore has to be reported, not hidden.

## 4. Terms Review Gate (all pending)

The 2026-10-16 read-only review is recorded in
[terms-review.v1.json](../../research/full_catalog/terms-review.v1.json), with all items
`pending_user_approval`. The agent recommends:

- LeetCode (T5/T6): reject. The terms forbid scraping, and robots.txt disallows `/graphql`.
  This also affects the provenance of the existing 3,053 statements.
- Codeforces (T1/T2): reject. robots.txt restricts scraping tools to `/api/`, and
  `/contest` is disallowed.
- AtCoder (T3/T4): user decision. Task pages are not disallowed, but there is an AI-training opt-out signal.

Each item records: source URL, retrieval date, the relevant clause in the
reviewer's own words, a decision and the reviewer. The review is done by the user
or by the agent reading the pages, with the user approving. The agent MUST NOT
mark items passed on its own authority.

| # | Item | Status |
|---|---|---|
| T1 | Codeforces terms/rules on automated access and storing problem content for private research | pending |
| T2 | Codeforces `robots.txt` for `/problemset/problem/` and `/contest/*/problem/` | pending |
| T3 | AtCoder Terms of Use on automated access, copying and storing task statements | pending |
| T4 | AtCoder `robots.txt` for `/contests/*/tasks/*` | pending |
| T5 | LeetCode Terms of Service on automated access to `/graphql` list metadata | pending |
| T6 | LeetCode `robots.txt` for `/graphql` | pending |
| T7 | Confirm private, non-redistributed, research-only use is acceptable for each source; otherwise drop that source | pending |
| T8 | Confirm results that quote no statement text may be committed (aggregate receipts only) | pending |

WHEN any item is `rejected` or unclear for a source, that source SHALL be
removed from scope. Experiments then report it as unavailable. No workaround (for
example, mirrors of statements) is allowed.

## 5. Bounded subset selection (proposal for approval)

A deterministic, committed selection, made before any fetch:

- S1. All 30 curriculum seed problems (10 CF, 20 AtCoder via native or alias keys).
- S2. Codeforces: 1,000 problems, stratified by rating band (null, <1200, 1200–1599,
  1600–1999, 2000–2399, ≥2400) proportional to catalog share, with each band ≥ 50
  where available. Within each band, order by SHA-256(`"cf-v1:" + key`) and take the
  first N. Exclude the 42 non-Latin-title problems. Exclude problems from contests
  that are clearly non-standard (April Fools, Kotlin Heroes language-restricted,
  testing rounds) using a committed contest-ID list.
- S3. AtCoder: 1,000 tasks from ABC/ARC/AGC only, stratified by contest type, ordered by
  SHA-256(`"at-v1:" + key`). Exclude APG4b, tutorials, `practice*`, heuristic (AHC)
  and visualizer entries. Japanese-only results are recorded, not replaced.
- S4. Structural duplicate candidates found in the audit (adjacent-contest
  same-title CF pairs, ABC/ARC shared tasks) are added to the subset so that Phase 2
  can test them with statement evidence. These additions are capped at 200 per platform.

Selection inputs: the metadata snapshot hash, this spec version and the rules
above. The output is `subset.v1.json` with `selection_sha256`. Expected scale is about
2,250 HTML requests at most, roughly 3.5 h per host. This subset supports B/C
comparisons against all 3,053 LeetCode statements. It does not support claims
about the full CF/AtCoder catalogs, which stay title-only (experiment A).

## 6. Task breakdown

- [ ] 1. Record the terms review (T1–T8) in `terms-review.v1.json`; user approves or rejects per source.
- [ ] 2. Implement `--plan` subset selection, and commit `subset.v1.json` plus its plan hash (no network).
- [ ] 3. Implement pure CF/AtCoder statement parsers, and fixture tests for parse success, `parse_failed`, language detection, math/image handling and hashing.
- [ ] 4. Implement bounded transport with stop conditions, and fixture tests for 403/429/503/CAPTCHA/login redirect stop, robots skip, size cap and no retry.
- [ ] 5. Implement replay, and test byte-identical re-normalization and hash-mismatch failure.
- [ ] 6. Implement the LeetCode list refresh plus reconciliation, with fixture tests for paging consistency, short pages and absent-ID classification.
- [ ] 7. With explicit per-run user approval: run the LeetCode refresh, then review the reconciliation.
- [ ] 8. With explicit per-run user approval: run a 20-item pilot per host, then inspect parse quality and stop-condition behavior.
- [ ] 9. With explicit user approval of the pilot: run the full bounded subset, then replay, then commit only `statement-coverage.v1.json`.
- [ ] 10. Update AUDIT.md, AGENTS.md and the Phase 2–5 plan with the measured eligible subset.

## 7. Validation plan

- Unit: `python3 -m unittest tests.learning.test_statement_sources tests.learning.test_leetcode_paid_refresh -v`
  (fixtures only, no sockets). Existing `python3 -m unittest discover -s tests -v`
  must still pass.
- Integrity:
  - replay equality;
  - SHA-256 of `data/raw`, `data/processed`, `indexes/` and catalog snapshots unchanged before and after;
  - `git status` shows no private files;
  - `scripts/learning/check_release.py` still passes.
- Pilot review: manual comparison of 20 parsed records per host against the live
  page by the user, with field-boundary errors counted. If more than 2 of 20 records
  per host have field errors, the parser is fixed before the full run.
- Coverage receipt:
  - per-platform counts by `statement_status`, `language` and `en_available`;
  - token-length distribution (computed later with the MiniLM tokenizer);
  - image-dependent count;
  - failures by reason.

## 8. Risks

- Terms may prohibit automated access or storage. That source is dropped (section 4),
  and the research reports title-only results for it.
- HTML structure drift and anti-bot pages are handled by the stop conditions and
  `parse_failed`. The parser never guesses.
- Selection bias: excluding Japanese-only, image-heavy and non-standard contests
  skews the eligible set. It must be reported per stratum.
- Statement similarity can be dominated by sample I/O and boilerplate. Field
  separation allows ablation, and this is a Phase 3 design concern.
- LeetCode list pages may change paid status over time. The receipt reflects only the
  retrieval date.
- Private data leakage is prevented through `.gitignore` coverage of
  `private-learning/` and a test that committed receipts contain no `statement_text` fields.

## 9. Open decisions for the user

1. Approve or adjust the subset sizes (1,000 + 1,000 + seeds + up to 200 + 200 duplicate candidates).
2. Approve the rate settings (10 s HTML / 5 s LeetCode list, per-host caps).
3. Choose whether the agent performs the terms review reading, as a non-collecting
   read of the terms and robots pages for user approval, or whether the user does it.
4. Decide whether newly found free LeetCode problems should later get statements
   (a separate spec revision).
