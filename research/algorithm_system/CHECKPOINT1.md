# Checkpoint 1 — reviewed source and taxonomy foundation

The frozen catalog has **24,297 unique IDs**: LeetCode 3,053, Codeforces 11,457,
AtCoder 9,787. Existing retrieval runs and all 3,647 pair records are preserved.

| Platform | Original tags | Native difficulty evidence | Statements in current catalog |
|---|---|---|---|
| LeetCode | 2,992 nonempty; 72 distinct | Easy 792 / Medium 1,538 / Hard 723 | 3,053 historical HTML records, quarantined |
| Codeforces | 11,277 nonempty; 38 distinct | 11,152 ratings; 305 missing | 0; official API is metadata only |
| AtCoder | 9,787 unavailable | Newly fetched community estimates: 4,867 values matched by exact task ID; 4,920 without values | 0; community catalog is metadata only |

Premium exclusion is **not row-verifiable**: the original collector filters
`isPaidOnly`, but the field/skip log was not retained. No new LeetCode fetch or
Premium access was attempted. Native categories and original URLs are preserved.
The source universe is the frozen public problemset snapshot, not proof of all
current platform problems or private/Gym coverage.

The versioned [taxonomy](taxonomy.v1.json) covers all observed LeetCode/Codeforces
tags and retains equivalent/broad/narrow/related/unmapped distinctions, confidence,
original tags and separate ancestors. AtCoder missing tags stay unavailable.
[Independent review](checkpoint1-review.md) corrected non-entailing hierarchy edges
and broad/specific mappings before acceptance. Fourteen focused tests passed.

The documented AtCoder Problems difficulty endpoint returned HTTP 200. Of 5,175
source keys, 5,151 match the 9,787-row catalog: 4,867 values and 284 nulls; 4,636
catalog IDs have no source entry. There are 948 negative estimates, preserved raw.
These are community estimates, not Codeforces-equivalent ratings. See the
[source receipt](atcoder-difficulty-source.v1.json) for exact source/hash/date.

## Decision to proceed

The latest instruction explicitly requests eligible publisher datasets. Independent
review accepts a **scoped private CodeContests non-code import** under its express
CC BY 4.0 grant, keeping attribution and separate third-party conditions. IBM's
CodeNet **dataset** grant is identified as CDLA-Permissive-2.0 by the Linux Foundation
release statement; the small publisher description archive can be investigated
without its submission-code dump. These findings update the earlier conservative
project clearance judgment; the earlier report and policy record remain unchanged.
This is not unrestricted original-platform permission or redistribution approval.

Proceed to CP2: statement-only acquisition, actual schema/ID validation, duplicate
analysis and within-source/topic-conditioned quantile baselines. Do not infer
AtCoder native task IDs from titles, or fit difficulty calibration to the existing
semantic-only evaluation labels. No approval pause or production change.
