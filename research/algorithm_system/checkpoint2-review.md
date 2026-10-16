# Checkpoint 2 independent review

Decision: **accept the descriptive difficulty analysis; statement ingestion is
conditional on importer validation**. The calibration contains no fitted
cross-platform difficulty conversion and does not promote AI-reviewed semantic
labels into difficulty truth.

## Pinned CodeContests source encoding

The locally saved Hugging Face `dataset_infos.json` has SHA-256
`6289cbfb168f2bde925e27bdada4cdaed65d7a4b9c060c75946504584f3cbdc0`.
Its `source` ClassLabel names are, in order, `UNKNOWN_SOURCE`, `CODECHEF`,
`CODEFORCES`, `HACKEREARTH`, `CODEJAM`, `ATCODER`, `AIZU`. Thus the pinned HF
builder's integer encoding is Codeforces 2, AtCoder 5, Aizu 6. The original
DeepMind `contest_problem.proto` instead reserves enum value 5 and assigns
AtCoder 6, Aizu 7. A Parquet `int64` field has no ClassLabel metadata, so an
importer must validate and use the pinned HF sidecar; it must not decode HF
Parquet integers as protobuf enum values. Record this encoding distinction in
the receipt. The HF dataset card lists 13,328 training rows; the current
3.76k-row viewer is explicitly a first-5GB preview and is not the full count.

## Calibration checks

- Input hashes in `calibration.v1.json` match the current frozen catalog,
  AtCoder model response, and corrected taxonomy. Its three source cohorts
  cover all 24,297 catalog keys once, with no recorded partition overlap.
- Native availability is explicit: LeetCode 3,053/3,053 ordinal levels;
  Codeforces 11,152/11,457 ratings; AtCoder Problems 4,867/9,787 estimates.
  The AtCoder data retain negative values; source distributions and units stay
  separate. Tie-aware midrank CDFs, nearest-rank percentiles, and 500 seeded
  bootstrap resamples are descriptive catalog diagnostics.
- Topic cohorts use reviewed equivalent primary concepts with confidence at
  least 0.85, preserve ancestor groups separately, and suppress detailed output
  below 20 observations. AtCoder topic analysis is unavailable because native
  tags are absent. Topic rank shifts are selection diagnostics, not prediction
  errors or calibration ground truth.
- The 3,647 frozen pair labels are counted as 2,886 non-null judgments and
  761 abstentions. Only 44 labels are strong (2/3), and these are semantic
  relevance judgments; there are zero difficulty-ordering labels. They remain
  evaluation-only. Future supervised calibration needs distinct difficulty
  ordering/appropriateness judgments and a prospective validation design.

The calibration script currently writes `calibration.v1.json` unconditionally.
Treat this output as a versioned research artifact: subsequent reruns should
write a new version or refuse overwrite once frozen. Before any model fit,
re-audit title/statement duplicate families because the current prospective
split groups only reviewed native-identity aliases.

Focused calibration tests passed. A concurrent full research test run had one
unrelated CodeNet fixture failure because `test_private_import_records_hashes_counts_and_license_caveat`
passed a dummy source commit to an importer that now requires its pinned commit;
reconcile that fixture before accepting ingestion validation.

Sources: [DeepMind protobuf](https://github.com/google-deepmind/code_contests/blob/main/contest_problem.proto),
[HF dataset card](https://huggingface.co/datasets/deepmind/code_contests),
[DeepMind dataset README](https://github.com/google-deepmind/code_contests/blob/main/README.md).
