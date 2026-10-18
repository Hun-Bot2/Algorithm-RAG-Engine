# Checkpoint 1 independent review

Decision: **accept the taxonomy and native difficulty foundation after the focused
corrections below**. This review does not approve a cross-platform difficulty
conversion or claim a statement-complete research corpus.

- Verified taxonomy source counts against the frozen metadata catalog: 72
  LeetCode tags, 38 Codeforces tags, and unavailable AtCoder native tags.
  `None`, observed empty lists, unknown future labels, and the Codeforces
  `*special` marker remain distinct.
- Corrected Codeforces `string suffix structures`: the source tag names a
  family, so it maps to the family concept rather than asserting the narrower
  suffix-array technique. `brute force` now maps to its direct concept. Related
  mappings cannot imply ancestors. Removed hierarchy edges that inferred search
  from recursion, arrays from two pointers, or optimization from generic methods.
  Original native tags and observed document frequencies remain unchanged.
- The AtCoder Problems `problem-models.json` join uses exact native task IDs.
  The recorded response has 5,175 model keys, 5,151 matches to the 9,787-row
  catalog, 4,867 non-null estimates, 284 matched nulls, and 948 negative values.
  Native negative values are retained; the empirical CDF computes only within
  present AtCoder estimates and retains ties. The 305 null Codeforces ratings
  in this catalog were verified and are now computed from the input snapshot,
  rather than hardcoded in future receipts.
- AtCoder Problems difficulty is a community estimate. Within-source empirical
  ranks do not make it equivalent to Codeforces ratings or LeetCode levels.
  Any later cross-platform comparison must state its cohort, missingness,
  calibration assumptions, and validation evidence. Missing LeetCode paid
  status remains unavailable and no LeetCode refresh was performed.
- The published CodeContests README grants CC BY 4.0 for its non-code materials
  and separately warns that third-party material may have other conditions.
  A scoped private research import from a pinned publisher dataset is reasonable
  with attribution and the caveat preserved; this does not authorize direct
  source-host scraping or establish every upstream owner's grant. Its proto has
  exact Codeforces `cf_contest_id` and `cf_index`, but no native AtCoder task ID.
  The Hugging Face card disagrees with the official proto on AtCoder/Aizu enum
  numbers and with its current viewer on row counts: inspect pinned Arrow schema
  and measured data before decoding or claiming coverage.
- IBM Project CodeNet describes anonymized `pNNNNN` problem IDs. Its GitHub
  publisher archive `doc/problem_descriptions.tar.gz` is 3.33 MB. The Linux
  Foundation's 2021 announcement identifies Project CodeNet as an IBM dataset
  released under CDLA Permissive 2.0; this is separate from the tools repository's
  Apache license. Use original CodeNet IDs unless an explicit native task URL or
  mapping is observed; title similarity is review evidence only.

Validation: `python3 -m unittest discover -s research/algorithm_system -v`
passed 14 tests after corrections. No statement text was downloaded for this
review, and no production module was edited.

Primary sources: [CodeContests license and inventory](https://github.com/google-deepmind/code_contests/blob/main/README.md#license),
[official schema](https://github.com/google-deepmind/code_contests/blob/main/contest_problem.proto),
[HF dataset card](https://huggingface.co/datasets/deepmind/code_contests),
[CodeNet README](https://github.com/IBM/Project_CodeNet/blob/main/README.md),
[publisher archive](https://github.com/IBM/Project_CodeNet/blob/main/doc/problem_descriptions.tar.gz),
[Linux Foundation CodeNet release statement](https://www.linuxfoundation.org/press/press-release/enabling-easier-collaboration-on-open-data-for-ai-and-ml-with-cdla-permissive-2-0).
