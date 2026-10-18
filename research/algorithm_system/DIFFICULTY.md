# Native difficulty distributions and rank diagnostics

This checkpoint describes native source distributions. It does not fit a model
that converts one platform's difficulty into another platform's scale. The
machine-readable result is [calibration.v1.json](calibration.v1.json), produced
by [calibration.py](calibration.py) from the frozen metadata catalog,
AtCoder Problems snapshot, reviewed taxonomy and existing label artifacts.
Input SHA-256 values are recorded in the result. No statement text was read.
The distribution figure is [difficulty-cdf.v1.svg](difficulty-cdf.v1.svg).

## Coverage and provenance

| Source | Catalog rows | Native values observed | Missing | Source scale |
|---|---:|---:|---:|---|
| LeetCode | 3,053 | 3,053 | 0 | Easy / Medium / Hard categories |
| Codeforces | 11,457 | 11,152 | 305 | Native rating points |
| AtCoder | 9,787 | 4,867 | 4,920 | AtCoder Problems community estimates |

AtCoder's 4,920 unavailable values comprise 284 matched records with a null
estimate and 4,636 catalog task IDs absent from the fetched model dataset. The
4,867 observed estimates include 948 negative values; these were retained. The
dataset is documented by [AtCoder Problems](https://github.com/kenkoooo/AtCoderProblems/blob/master/doc/api.md)
and is not an AtCoder-native rating. Its raw response and hash remain in the
private snapshot identified by [the source receipt](atcoder-difficulty-source.v1.json).
LeetCode's missing paid-only flags remain unavailable; there was no LeetCode
refresh.

## Within-source empirical distributions

For each observed native value, the empirical CDF records the tie interval
`[count(values < x) / n, count(values <= x) / n]` and its midpoint. This keeps
ties visible. LeetCode categories use their published order Easy, Medium, Hard.
The full native-value CDFs are in the JSON receipt.

The table shows selected native-value percentiles and 95% percentile intervals
from 500 row-level bootstrap samples (seed 17). LeetCode's results are category
landmarks; the other two remain in their own native numeric units.

| Source | p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|
| LeetCode | Easy [Easy, Easy] | Easy [Easy, Medium] | Medium [Medium, Medium] | Medium [Medium, Hard] | Hard [Hard, Hard] |
| Codeforces | 900 [800, 900] | 1,300 [1,200, 1,300] | 1,800 [1,800, 1,800] | 2,400 [2,400, 2,400] | 2,900 [2,900, 2,900] |
| AtCoder Problems | -581 [-636, -514] | 276 [210, 344] | 1,310 [1,261, 1,355] | 2,282 [2,236, 2,333] | 3,094 [3,034, 3,136] |

Bootstrap intervals describe resampling variation in these catalog rows. They
do not include uncertainty about the source's rating process, catalog coverage,
or representativeness. Similar percentile positions do not mean equal ability.

## Topic-conditioned rank diagnostics

Only taxonomy mappings marked `equivalent` with confidence at least 0.85 define
primary topic cohorts. Inherited taxonomy ancestors are analyzed separately;
they are not counted as observed native tags. AtCoder has no native tags in this
catalog, so its topic-conditioned comparison is unavailable. Cohorts with fewer
than 20 observed native difficulties are marked below the planning minimum.

The diagnostic compares each row's global midrank with its midrank among rows
carrying that primary concept or ancestor. It reports absolute rank shifts, not
prediction errors. There are 48 LeetCode and 35 Codeforces primary-concept
cohorts with at least 20 observed values, plus 14 and 8 ancestor cohorts,
respectively. For example, the median absolute global-to-topic shift is about
0.274 for LeetCode `segment-tree` (n=65) and 0.378 for Codeforces `fft` (n=116).
These are planning diagnostics: they can reflect cohort composition, source tag
coverage, and selection. They do not establish that topic-conditioned ranks are
more accurate or that the taxonomy is difficulty ground truth.

## Existing evaluation labels and future split plan

All 3,647 existing AI-reviewed pair records remain evaluation-only. Their final
labels are 2,443 unrelated, 399 partially related, 40 relevant, 4 strongly
relevant, and 761 abstentions. The 44 positive relevance judgments are semantic
algorithm-similarity labels, not difficulty orderings; none was used to fit or
select a difficulty mapping. Forty-three cross-platform positive pairs have
native values on both endpoints, which still supplies no difficulty truth.

Every endpoint of these pairs is reserved as `legacy_eval_only`. The stable
seed-17 split plan hashes keys for all 24,297 catalog records and partitions
them into 2,940 evaluation-only records, 14,930 calibration-planned records,
3,207 model-selection-planned records, and 3,220 final-planned records. It uses
catalog `representative_key` identity groups and does not merge unverified
same-title candidates. The JSON receipt stores each partition's key-set hash,
counts and platform counts; the partitions are disjoint and cover the catalog.

The plan is problem-group disjoint, not platform-source-disjoint. Since each
platform defines a different native score source, platform-held-out validation
and any future cross-source calibration design require a separate review. No
supervised fit has been performed, and no difficulty score conclusion is made
from these descriptive distributions.
