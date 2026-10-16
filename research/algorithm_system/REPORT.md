# Full-catalog algorithm similarity research

## Research scope and admission boundaries

This extends, rather than replaces, A1/A2/A1-all. The original 3,647 AI-reviewed
or abstained pair records, pinned MiniLM weights, source receipts, and previous
retrieval artifacts remain frozen. This study changes no production policy and
performs no deployment, notification activation, paid inference, or model training.

[Checkpoint 1](CHECKPOINT1.md) audits the 24,297 unique native problem keys and
Premium-status limitation. [Source coverage](SOURCE-COVERAGE.md) separates native
catalog coverage, publisher-description availability, and verified native joins.
[Checkpoint 2](CHECKPOINT2.md) records the revised scoped publisher-data admission
and the limits of cross-platform difficulty calibration. [Difficulty study](DIFFICULTY.md)
retains the native scales and uncertainty; semantic relevance labels are never
silently converted into difficulty-ordering labels.

## Methods and matched corpus

`statements.py` uses one sorted corpus for statement word TF-IDF (1–2 grams),
character TF-IDF (3–5 grams; 100,000-feature cap), pinned MiniLM title-only,
title+description prefix, and complete-content chunk pooling. TF-IDF fits the
entire retrieval corpus as a transductive baseline, not held-out language-model
training. The pinned MiniLM revision and model-file membership are verified by
the existing local backend. CPU-only inference uses local weights and offline
Hugging Face/Transformers settings. No OCI inference service is involved.

The complete-content variant partitions original tokenizer IDs into disjoint
254-token windows, adds special tokens (maximum 256), normalizes each chunk,
weights pooled chunks by content-token count, then L2 normalizes the result.
Tokenization completeness is checked; publisher section completeness is not.
Titles plus descriptions are a separate prefix-truncated comparator.

Exact blockwise NumPy cosine or sparse TF-IDF products retrieve every admitted
query against all other-platform candidates. Each method stores Top10/20/50 with
deterministic score/key ties. Identical normalized statement hashes are masked
as exact duplicates for retrieval; the original rows remain preserved. Ambiguous
semantic duplicates receive a review queue and are never silently merged.

AtCoder algorithm tags are unavailable and LeetCode statements remain quarantined.
Consequently, no matched cross-platform tag-only, hierarchical, or hybrid score
can currently be computed honestly. The taxonomy and scorer helpers are present,
but missing tags are not filled with model guesses or interpreted as empty truth.

## Evaluation interpretation

`evaluate.py` reports all-query coverage and diagnostics, plus query-bootstrap
intervals only for defined metrics. It distinguishes observed precision among
judged slots from lower/upper bounds including unknown slots; abstentions and
unlisted pairs remain unknown. Common-judged-set nDCG is conditional and is not
full-catalog nDCG. Historical labels are AI-reviewed evidence, not human gold.
AtCoder publisher rows without native IDs cannot match old native-ID judgments,
so a zero-overlap evaluation cannot establish method quality or improvement.

Hubness and platform diversity describe recommendation concentration. Tag
consistency requires known high-confidence tag mappings on both endpoints.
Native difficulty distance is computed only for comparable native units; no
arbitrary LeetCode/Codeforces/AtCoder equality is introduced. Cross-platform
missingness remains explicit. Review queues sample at most 100 queries/platform
and balance method/platform/difficulty/topic strata; their labels remain pending.
All-query aggregate metrics are independent of this review sampling.

## Compression and recommendation decisions

`compress.py` refuses to run before a complete original 384D chunk baseline and
matching corpus/vector/retrieval hashes exist. PCA, uncentered truncated SVD,
and seeded Gaussian projections are evaluated at 256/128/64 dimensions where
rank permits. All learned linear transforms fit the whole corpus, explicitly
transductively. Neighbor overlap, latency, array/transform storage, process RSS,
and exact retrieval time describe computational tradeoffs, not learning gains.
No UMAP/t-SNE appearance is treated as relevance evidence. The original vectors
remain the reference and no automatic model promotion is allowed.

[Adaptive recommendation architecture](RECOMMENDATION-ARCHITECTURE.md) specifies
equivalent practice, progressive challenge, prerequisite repair, and spaced
review using versioned learner evidence and reviewed prerequisite edges. It
retains the deterministic no-AI mission/delivery path and proposes no RL model.

## Reproduce

Run from repository root using the scientific `.venv`. New outputs must use fresh
versioned paths; private text/index artifacts remain excluded from Git.

```sh
.venv/bin/python -m unittest discover -s research/algorithm_system -v
.venv/bin/python research/algorithm_system/verify_capture.py
.venv/bin/python research/algorithm_system/statements.py --source private-learning/algorithm-system/codecontests-v1/statements.jsonl --output private-learning/algorithm-system/statement-run-NEW
.venv/bin/python research/algorithm_system/evaluate.py --corpus private-learning/algorithm-system/statement-run-v1/corpus.jsonl --run private-learning/algorithm-system/statement-run-v1 --output private-learning/algorithm-system/statement-evaluation-v1/evaluation.json
.venv/bin/python research/algorithm_system/compress.py --run private-learning/algorithm-system/statement-run-v1 --output private-learning/algorithm-system/compression-NEW
.venv/bin/python research/algorithm_system/build_notebook.py
```

Execute the generated notebook from repository root with the `.venv` Python
kernel, then export HTML. Aggregate figures and receipts accompany the executed
notebook. Source license notices and attribution remain in the acquisition
reports; imported text is not embedded in the public report or notebook.

## Executed results

[Checkpoint 3](CHECKPOINT3.md) records the five-method, 9,424-query run, token
coverage, runtime/RSS, null relevance metrics, and pending review datasets.
[Checkpoint 4](CHECKPOINT4.md) records all ten original/compressed representations.
SVD256 retains 97.39% of Top10 neighbors with 33.3% less vector storage; compute-only
latency changes are small, and no relevance improvement is established. The
original 384D embeddings remain active only as a research reference.

The evidence gates that remain open are: legitimate LeetCode statement admission
with row-level Premium evidence, native AtCoder ID resolution, independently
reviewed AtCoder concept annotations, human statement-based relevance judgments,
and separate difficulty-ordering labels. These prevent full three-platform
statement coverage, validated hybrid quality comparisons, and supervised
cross-platform calibration. The study implements and measures every currently
supported branch without fabricating those missing inputs.
