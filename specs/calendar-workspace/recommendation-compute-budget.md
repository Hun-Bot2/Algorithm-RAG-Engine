# Recommendation compute budget (2026-10-23)

This is a measurement and discussion receipt, not a new recommendation implementation or scheduled job.

## Verified local inputs
Read-only catalog-v4 query: Codeforces 11,457; AtCoder 9,793; LeetCode 3,053; total 24,303 registered problems (receipt eligible total 23,459). The existing title-vector array in private-learning/mastery/catalog-v4/run/vectors.npy is float32 [24303,384], 35.60 MiB. Both existing neighbor channels contain 729,090 entries (30 per registered problem). These are title neighbors, not validated solution-idea or statement-similarity representations.

## Measurement
Existing research.full_catalog.retrieve.dense_cosine and rank_row, local arm64, NumPy 2.1.3, VECLIB_MAXIMUM_THREADS/OPENBLAS_NUM_THREADS/OMP_NUM_THREADS requested 1. No model execution, new packages, network, learner writes or recommendation changes. Self matches excluded; platform/identity/learner constraints and DB persistence excluded.

Warmed three-trial medians, cosine + exact top-30 selection:
- 1 query vs entire catalog: 0.000579 s.
- 100 queries vs entire catalog: 0.030448 s.
- 1,000 queries vs entire catalog: 0.278561 s.

Full 24,303-query run, block size 512, single measurement:
- Vector normalization 0.016849 s.
- Dot products 0.441641 s.
- Top-30 selection 5.919211 s.
- Full block scan 6.416078 s (normalization separate).
- Every registered problem queried once; results were temporary in-memory arrays and were not persisted.

Raw measurement scripts/JSON: /tmp/estimate-recommendation-cost.py, /tmp/recommendation-cost-receipt.json, /tmp/benchmark-all-recommendation-cosine.py, /tmp/recommendation-full-scan-cost.json.

## Operation and storage estimates
For one 384D representation: N^2 = 590,635,809 directed comparisons including subsequently excluded self matches; roughly 453.6 GFLOPs for dense dot products (2N^2D). A full float32 score matrix would occupy 2.20 GiB; streaming blocks avoids retaining it. Top-30 int32 IDs + float32 scores occupy 5.56 MiB per channel before identifiers/indexes/database overhead.

## Implications and limits
Weekly full neighborhood refresh at the current size is practical locally for precomputed vectors. Two/three comparable independent 384D representations would have an approximate scan-only budget of 12.8/19.2 seconds if their implementation/candidate counts stayed comparable; these are extrapolations, not measurements. Weakness remediation is a learner-policy task and may reuse candidate relationships rather than require a third all-pairs vector scan.

Daily ranking can reuse weekly neighbor caches plus current attempts/review preferences. Actual latency for the planned system is unmeasured. Initial embedding of solution-idea descriptions or permitted problem content, model/context choice, text availability, data preparation, learner-policy computation and storage I/O are excluded and need their own benchmark. Current cosine timings do not establish semantic recommendation quality. Candidate scores from different spaces must retain their channel/model provenance rather than be averaged indiscriminately.
