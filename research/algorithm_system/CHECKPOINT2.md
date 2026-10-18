# Checkpoint 2 — difficulty baseline and statement integration

The native-difficulty baseline is executed and independently reviewed in
[checkpoint2-review.md](checkpoint2-review.md). [DIFFICULTY.md](DIFFICULTY.md)
and [calibration.v1.json](calibration.v1.json) preserve counts, input hashes,
tie intervals, seeded bootstrap uncertainty, topic strata, and prospective splits.

There are **zero difficulty-ordering labels**. The 44 strong existing similarity
judgments are not difficulty comparisons. No monotonic/ordinal mapping is fitted;
percentile alignment is descriptive, provisional, and not equivalent learner effort.
All 3,647 historical label records remain evaluation-only. The proposed key-group
split needs duplicate-family adjudication before any future supervised fitting.

## Revised publisher-data admission decision

The earlier conservative statement audit is preserved. This research admits a
**private, attributed, non-code publisher-dataset import**, following independent
source review. This does not establish permission to scrape original platforms,
release a public statement mirror, or automatically admit quarantined LeetCode text.

* [Google DeepMind CodeContests](https://github.com/google-deepmind/code_contests#license)
  explicitly grants CC BY 4.0 for non-code data while retaining third-party
  conditions and disclaimers. The publisher-organization Hugging Face snapshot is
  pinned to `802411c3010cb00d1b05bad57ca77365a3c699d6`. Only description,
  public-sample, identity, and metadata columns are selected with bounded HTTP
  ranges. Solutions, private tests, and generated tests are excluded.
* The pinned Hugging Face `dataset_infos.json` source encoding differs from the
  original protobuf: **AtCoder=5, Aizu=6** in this snapshot. Its SHA256 is
  `6289cbfb168f2bde925e27bdada4cdaed65d7a4b9c060c75946504584f3cbdc0`.
  Neither schema nor native identity is guessed from titles.
* [IBM Project CodeNet](https://github.com/IBM/Project_CodeNet) provides a small
  description archive. Its dataset grant is CDLA-Permissive-2.0, distinct from the
  tools repository's Apache license; the [Linux Foundation announcement](https://www.linuxfoundation.org/press/press-release/enabling-easier-collaboration-on-open-data-for-ai-and-ml-with-cdla-permissive-2-0)
  provides explicit IBM attribution for that dataset-license decision. The full
  submissions archive is not acquired.

The actual CodeNet import contains **3,999 unique descriptions**. No metadata CSV
or canonical task URLs are present in this small archive, so all platform/native
identities remain unresolved and **zero** rows join the current native catalog.
They are preserved privately, excluded from verified cross-platform experiments,
and never guessed to be AtCoder from their titles. Archive SHA256:
`8b631ae168ba84dce69c7d8e1b6c632256e0155858c2664c6493a5f001b45fdd`.
The matching pinned Git blob is `22a8342ab304d69edd537be86e38743584296ef3`.
Actual capture receipt: `private-learning/codenet/import-description-v1/receipt.json`.

## Remaining limitations

LeetCode statements remain quarantined, and Premium status remains historical
filter evidence rather than row-verified metadata. AtCoder has no native algorithm
tags in the admitted catalog. Publisher AtCoder rows without explicit native IDs
can support a separately named dataset-identity retrieval cohort, but cannot be
joined to learner history, native difficulty, or old relevance judgments by name.
A full three-platform statement comparison and a matched cross-platform tag hybrid
are therefore conditional on additional legitimate data and reviewed identity/tag
annotations. Continue measurable available branches; report unavailable branches
as blocked rather than inventing records or labels.
