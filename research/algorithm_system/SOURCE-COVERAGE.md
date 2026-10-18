# Source integration and coverage

This report records aggregate source availability, identity joins, and provenance
for the current algorithm-system research corpus. The counts describe distinct
acquisition receipts and cohorts; they are **not additive** and do not form a
single count of eligible, unique, rights-cleared problems. See the machine-readable
[coverage receipt](source-coverage.v1.json) and the aggregate-only
[CodeNet receipt](codenet-source.v1.json). Raw statements and private source
responses remain under `private-learning/`.

## CodeContests statement cohort

The pinned Google DeepMind CodeContests snapshot is revision
`802411c3010cb00d1b05bad57ca77365a3c699d6`. The receipt contains 9,424 selected
statement rows: 8,101 labeled Codeforces and 1,323 labeled AtCoder. Of the
Codeforces rows, 8,097 carry explicit contest/index IDs and 4 remain unresolved.
Of the AtCoder rows, none contains an explicit AtCoder task URL, so all 1,323
remain unresolved. The importer does not infer task identity from titles. Among
the 8,097 explicit Codeforces IDs, 7,077 join the frozen metadata catalog;
1,020 do not. These figures were independently recounted from the private
normalized corpus and the pinned shard receipt.

The import read 41 Parquet shards through 260 bounded byte ranges, transferring
16,209,154 bytes. It selected named statement, public-sample, and metadata
columns; it did not download full shards or read solution, private-test, or
generated-test columns. The offline replay receipt confirms 9,424 output records,
zero network calls, and output SHA-256
`369f072ebef82d7c7fc12b881ae8a9f5bef4753b4b141c566d69cc153d8295b6`. The
schema sidecar's SHA-256 is
`6289cbfb168f2bde925e27bdada4cdaed65d7a4b9c060c75946504584f3cbdc0`.

Attribution is to Google DeepMind CodeContests. Its README grants CC BY 4.0 to
non-code materials, while explicitly preserving separate third-party terms and
making no representation about rights to those third-party materials. The
snapshot includes Codeforces and CodeNet-origin materials, so the publisher grant
is not treated as blanket permission from those original platforms. The dataset
source and pinned revision are recorded in
[codecontests-source.v1.json](codecontests-source.v1.json); consult its source
[license notice](https://github.com/google-deepmind/code_contests/blob/main/README.md#license)
for attribution and third-party caveats.

## CodeNet descriptions

The private import contains 3,999 descriptions. Its source archive is pinned to
IBM Project CodeNet commit `55c323b527ab3d6510d55edff188bc0a0d7bc5e5`; archive
SHA-256 is
`8b631ae168ba84dce69c7d8e1b6c632256e0155858c2664c6493a5f001b45fdd` and the
pinned Git blob check passed. No metadata CSV was present. All 3,999 rows have
unknown platform/native identity and zero join to the current catalog. The
aggregate receipt attributes the dataset under CDLA-Permissive-2.0, separately
from the Apache-2.0 tools-repository license, and retains the caveat that original
problem-description rights and terms are not resolved by the dataset-level grant.
The [Linux Foundation announcement](https://www.linuxfoundation.org/press/press-release/enabling-easier-collaboration-on-open-data-for-ai-and-ml-with-cdla-permissive-2-0)
provides the cited dataset-license evidence. Only the description archive was
imported; the full submissions archive was not acquired. The public receipt
contains aggregate metadata and hashes, not descriptions.

## Platform metadata and LeetCode boundary

The frozen full catalog contains 24,297 records: 11,457 Codeforces, 9,787
AtCoder, and 3,053 LeetCode. Codeforces metadata came from its documented
`problemset.problems` API. AtCoder metadata came from the community-maintained
Kenkoooo/AtCoder Problems endpoint, not an official AtCoder catalog. The saved
metadata responses were HTTP 200 and have retrieval timestamps and hashes in the
private catalog manifest. The repository collector spaces request starts by at
least 2.1 seconds; this is client-side pacing, not a statement of either
provider's published quota. Those endpoints supply metadata, not full problem
statements, and no Codeforces or AtCoder problem page was fetched directly for
this integration.

The 3,053 historical LeetCode statement records remain quarantined and are
excluded from new statement research and any eligible-source total. The legacy
collector read `isPaidOnly` but did not persist its value per row, so row-level
Premium status is unknown. No new LeetCode scrape or Premium refresh was run.

## Duplicate review and counting limits

The private 9,424-row CodeContests statement-run duplicate receipt reports 1,007
exact-content groups covering 2,050 rows. It reports zero normalized cross-platform
title-equality groups queued for human review. The audit deletes no records; an
exact-content group is a review signal and does not itself establish same-platform
identity or original-source equivalence.

No aggregate eligible total is reported. CodeContests rows overlap source
catalogs, CodeNet has unresolved identity, LeetCode statements are quarantined,
and source rights/attribution remain scoped as described above. Summing these
cohorts would overstate unique eligible coverage. No full solution archives,
hidden tests, or direct platform problem-page statements were acquired in this
integration.
