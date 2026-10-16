# Cross-platform catalog ingestion receipt

2026-10-16: two live HTTP 200 responses were downloaded from documented public metadata endpoints. No embedding inference or statement scraping was performed.

| Platform | Records | Missing native difficulty | Missing tags | Missing statements |
|---|---:|---:|---:|---:|
| codeforces | 11,457 | 305 | 0 | 11,457 |
| atcoder | 9,787 | 9,787 | 9,787 | 9,787 |

All 21,244 normalized keys are unique. Neither parser rejected records in these responses. Retrieval time is recorded, but the upstream dataset update time is unknown. A new retrieval does not prove source freshness or complete platform coverage. Codeforces represents the public problemset endpoint; AtCoder is an unofficial community catalog.

## Curriculum identity checks

- Codeforces: 10/10 native ID matches.
- AtCoder: 17/20 direct ID matches; 3/20 resolve through official-source-verified legacy mappings; 0 unresolved after explicit mapping.
- `abc049_c → arc065_a` (Daydream), `abc086_c → arc089_a` (Traveling), `abc095_c → arc096_a` (Half and Half).
- The downloaded native IDs are preserved. The original curriculum and held-out labels are unchanged. The mapping registry is a dataset integration input; it does not yet migrate hosted problem keys, correct existing generated URLs, or reconcile submission imports. Those three legacy identities must be integrated before hosted pilot activation.

## Provenance and validation

- [Source registry](../../assets/cross_platform_sources.v1.json) contains URLs, retrieval timestamps, HTTP statuses, raw and normalized hashes, missingness, and direct versus alias-resolved coverage.
- [Identity mapping](../../assets/problem_identity_aliases.v1.json) records official task links and evidence.
- Raw responses and normalized metadata stay in ignored `private-learning/catalogs/`; they are not uploaded with the dashboard or checked into Git.
- Saved response bytes were replayed without network calls; normalized output bytes matched exactly and source retrieval times were retained.
- All preserved historical files listed in the original source registry retained their hashes.
- Focused parser/transport/identity tests: 13 passed. Live API downloads are separate evidence from fixture tests.

## Reproduce

```bash
python3 scripts/learning/collect_catalogs.py --fetch --output private-learning/catalogs/NEW-RUN
python3 scripts/learning/collect_catalogs.py --replay private-learning/catalogs/NEW-RUN --output private-learning/catalogs/NEW-REPLAY
python3 -m unittest tests.learning.test_catalog_sources -v
```

The output path must be new. Fetches are opt-in and serialized with a 2.1-second minimum between request starts. Responses are bounded to 20 MiB and 30-second socket timeout; partial failures are preserved and return a failing CLI status. API title/tag metadata is not a substitute for full problem statements.

## Next experiment gate

A fair semantic comparison requires one frozen input representation and a shared encoder across platforms. The old OpenAI-space vectors cannot be combined with a new local model space. Start with the reviewed 90-problem pilot and equal English annotations; a larger statement corpus requires a separate source/coverage review. Optional model download approval is still pending. No claims about cross-platform retrieval quality follow from catalog presence.

Update: the user authorized MiniLM download and local inference on 2026-10-16.
The [90-annotation pilot](../minilm_pilot/REPORT.md) is now executed. This does not
change the metadata-only scope of the downloaded catalogs or authorize statements
as a complete, reviewed corpus.
