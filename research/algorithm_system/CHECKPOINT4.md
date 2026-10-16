# Checkpoint 4 — compression, notebook, and recommendation architecture

[The compression receipt](compression.v1.json) verifies the completed original
384D corpus/vector/Top50 hashes before fitting any transform. All 9,424 rows
are evaluated with the same cross-platform masks and stable ranking. The observed
centered rank is 382. PCA and uncentered SVD fit this entire corpus explicitly
transductively; the result is not held-out inductive validation. Gaussian random
projection uses seed 17. Actual transform arrays, hashes, vector arrays,
Top10/20/50 outputs, warm latency, elapsed retrieval time, and disk/RAM measurements
are retained privately; aggregate receipts are committed.

| Representation | Dimensions | Mean Top10 overlap | Vector array MiB |
|---|---:|---:|---:|
| Original | 384 | 1.0000 | 13.805 |
| PCA | 256 | 0.6795 | 9.203 |
| PCA | 128 | 0.6525 | 4.602 |
| PCA | 64 | 0.5752 | 2.301 |
| Uncentered SVD | 256 | 0.9739 | 9.203 |
| Uncentered SVD | 128 | 0.8428 | 4.602 |
| Uncentered SVD | 64 | 0.6813 | 2.301 |
| Gaussian RP | 256 | 0.5984 | 9.203 |
| Gaussian RP | 128 | 0.4674 | 4.602 |
| Gaussian RP | 64 | 0.3241 | 2.301 |

Array storage excludes additional persisted transform files; the receipt includes
both. Full exact retrieval compute-only timings were about 0.46–0.49 seconds
per representation in the warm compression process, with no substantial latency
advantage demonstrated. These timings exclude output serialization and differ
from baseline end-to-end timings. RAM values are process high-water marks, not
isolated per-method allocation measurements.

**Keep original 384D as the reference.** SVD256 is a potentially useful storage
tradeoff for later validation, but 97.4% neighbor preservation is not relevance
improvement. No measured labels support a quality ranking or production change.
There is no reason to introduce FAISS for this measured bounded workload.

[The executed notebook](comparison.ipynb), [HTML](comparison.html), and aggregate
figures present token coverage, runtime, and compression tradeoffs. No UMAP or
t-SNE cluster appearance is used as quality evidence. The research-only
[recommendation architecture](RECOMMENDATION-ARCHITECTURE.md) covers equivalent
practice, progressive challenge, prerequisite repair, and spaced review while
retaining native difficulty uncertainty, inspected learner evidence, and the
no-AI mission/delivery path. It makes no deployment or learning-outcome claim.
