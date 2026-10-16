# Cross-platform corpus ingestion — first implementation

## Research summary

The previous vector_geometry experiment used only the preserved 3,053-row LeetCode index. It did not measure cross-platform retrieval. Existing hosted adapters synchronize contests and submissions, not problem catalogs.

Codeforces documents anonymous `problemset.problems` with Problem and ProblemStatistics objects and a maximum one call per two seconds. AtCoder Problems documents the unofficial `resources/problems.json` dataset and requests more than one second between accesses. These catalogs do not provide full problem statements. Public API availability is not a blanket redistribution license.

Sources checked 2026-10-16:
- https://codeforces.com/apiHelp
- https://codeforces.com/apiHelp/methods#problemset.problems
- https://codeforces.com/apiHelp/objects#Problem
- https://github.com/kenkoooo/AtCoderProblems/blob/master/doc/api.md

## Requirements

- The collector SHALL fetch documented public metadata endpoints with no credentials, automatic model downloads or inference.
- Requests SHALL be serialized with at least 2.1 seconds between starts, bounded time and response size; errors SHALL be retained as errors, not empty successful catalogs.
- Raw response bytes, source URL, retrieval timestamp, HTTP status, hash and normalized snapshot SHALL remain private and immutable per run.
- Normalization SHALL preserve platform-qualified identity, native difficulty and absent fields. AtCoder tags/rating and both platforms' statements MUST NOT be invented.
- Existing LeetCode assets, curriculum, evaluation labels, production bundles and learner data MUST remain unchanged.
- API discovery/catalog presence SHALL NOT automatically approve relationships or enable production recommendations.

## Design proposal

A standard-library Python offline collector writes fresh run directories below ignored `private-learning/`. A reusable parser normalizes Codeforces contestId+index and AtCoder task IDs; malformed/duplicate identities fail or are explicitly excluded and counted. A receipt distinguishes fetched, failed and missing coverage. Offline replay parses saved bytes without another API call. The seed's 30 external problems provide an explicit coverage denominator, not a semantic quality benchmark. Research documentation may retain aggregate receipts; downloaded metadata remains outside hosted/static assets and Git.

## Task breakdown and implementation plan

1. Implement parsers and bounded transport/CLI.
2. Test identities, missingness, schema failure, duplicate protection and replay.
3. Fetch the two endpoints once, inspect actual shapes and preserve responses.
4. Validate unique IDs, coverage of selected curriculum problems and input hashes.
5. Record measured counts and failures; identify the next common-input/encoder gate.

## Validation plan

Use deterministic fixture tests for parser/transport failures and a local replay of actual downloaded data. Verify the saved byte hashes before parsing and report catalog size, missing fields, rejected records and curriculum matches. Live HTTP success is distinct from fixture success. Keep historical hashes intact.

## Risks and next decision

Endpoints may be unavailable, return anti-bot HTML or change schema; preserve bounded evidence and do not bypass access restrictions. Missing AtCoder concepts cannot be filled with guesses. Full-catalog title/tag comparison has unequal information and platform imbalance. A meaningful next retrieval experiment needs a frozen common input representation and one shared encoder for all platforms. Existing OpenAI vectors cannot be compared with unrelated encoder outputs. New model downloads/inference and full-statement collection remain separate review gates; finish metadata ingestion before asking for that concrete decision.

## Implementation and validation receipt

The stdlib collector is implemented in `src/learning/catalog_sources.py` with the
CLI `scripts/learning/collect_catalogs.py`. On 2026-10-16 both public endpoints
returned HTTP 200: 11,457 Codeforces and 9,787 AtCoder records. Thirteen focused
tests passed. Raw replay reproduced normalized bytes exactly. Detailed counts,
missingness and known integration limits are in `research/cross_platform/INGESTION.md`.

The live join exposed three incorrect legacy AtCoder curriculum identifiers.
`assets/problem_identity_aliases.v1.json` records official-source-verified native
IDs. Coverage reports keep 17 direct matches and 3 alias matches separate. Raw
IDs, historical assets, frozen judgments and current hosted catalog are unchanged.
A versioned hosted identity/URL/import reconciliation is required before M8 release;
do not report this offline mapping as a deployed correction.
