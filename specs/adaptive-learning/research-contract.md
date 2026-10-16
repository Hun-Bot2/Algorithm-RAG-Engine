# Offline relationship research contract

## Scope and evidence boundary

This contract defines reproducible local research tools. They do not change the
hosted policy, approve curriculum edges, train encoders, call a model service, or
claim learner benefit. The default catalog inputs are the 90-row curriculum
seed and same-key annotation records. The only text fields admitted to a scorer
are annotation `title`, `concept_rationale`, and curriculum `concepts`, plus
native platform/difficulty metadata for explicit metadata features. Problem
statements, solutions, private learner events, and legacy indexes are excluded
by default.

The 180-row judgment file is evaluation-only. A row counts as a reviewed label
only when it has a reviewed adjudication state and the document explicitly
permits evaluation. Draft/proposed rows are counted and reported but never
scored as ground truth. Training additionally requires a separate dataset with
`training` in `permitted_uses`. Historical evaluation pairs and LLM proposals
cannot satisfy that gate.

## Baselines and comparable candidate sets

The deterministic metadata baseline combines exact concept Jaccard overlap,
same-platform agreement, and same-native-difficulty agreement. It does not
convert difficulty labels across platforms. The TF-IDF baseline uses the same
allowlisted title, rationale, and concepts for every catalog item, a frozen
Unicode word tokenizer, smoothed IDF, cosine similarity, and a lexicographic
problem-key tie break. Feature weights, tokenizer version, relevance threshold,
cutoffs, and bootstrap seed are versioned in the report.

The metadata and TF-IDF methods share judged candidates but use different input
fields, so their comparison is explicitly an input ablation, not an equal-
information algorithm comparison. For each judged source and relationship type,
candidates are restricted to the same reviewed judgments and use the same
eligible curriculum snapshot for every method. Unjudged candidates are not
negative labels. The report includes judged
candidate coverage against the catalog and says that rankings are conditional
on the partial judged set. Recall@10 and graded nDCG@5 use relevance 1-3 as
positive gain; relevance 0 is non-relevant. Precision@5 for prerequisite and
transfer proposals uses the predeclared relevance threshold of 2. Direction
accuracy compares top-five predicted direction labels (the baseline default is
source-to-target) with the reviewed direction when relevance is at least 2, and
predicts none otherwise. A precomputed artifact may carry an explicit direction
prediction. This evaluates the supplied ordered pairs; reverse candidates must
be separately reviewed before this can compare both orientations.
Queries with fewer than two reviewed candidates are not ranking evidence:
Recall@10 and nDCG@5 remain unavailable for those queries, while their pair
labels may still contribute to directional precision/accuracy. Reports include
the singleton count and the number of rankable query groups.

Before scoring, every judgment whose unordered problem pair appears in the
curriculum relationship graph is removed, even when its judgment type differs
from the graph edge. The removed row and pair counts remain in the report.

Methods are compared on the same query groups with deterministic paired
bootstrap resampling clustered by source problem, so relation types from one
source are not treated as independent. Review-family cluster uncertainty is also
reported where there are enough distinct families. The report retains per-query ranks,
coverage, subgroup values by source/target platform pair and review family, and
paired intervals. Subgroup results include platform-pair coverage cells for every
ordered pair of supported platforms (AtCoder, Baekjoon, Codeforces, LeetCode);
absent cells are marked unavailable. Small or
empty groups remain explicitly unavailable. The
report also counts lexical concept overlap between admitted concept text and
judgment evidence so reviewers can identify a possible shared-information
path; this diagnostic is not a corrected score or a causal claim.

## Optional precomputed neighbors

An optional JSON neighbor artifact has schema version, namespace, revision,
license reference, dimension, normalization, metric, preprocessing version,
ordered keys, catalog-input hash, and per-query ranked neighbors. The loader
checks the complete manifest and key uniqueness before accepting it. Expected
namespace mismatch is a hard error. An absent artifact selects the metadata
baseline and makes zero network/model calls. Neighbor scores from different
namespaces are never merged, averaged, or compared as if calibrated.

## Split and scorer training gate

A frozen split manifest contains a version, `frozen: true`, training problem
keys, evaluation problem keys, training families, and evaluation families. Any
problem or family overlap blocks training. The initial 90 curriculum problems
and 180 relationship judgments are reserved for evaluation and cannot become
training data.

Training requires all of the following before fitting: explicit training use
permission; a separate approval receipt bound to the exact dataset SHA-256,
canonical snapshot hash (dataset, split, train/evaluation catalogs and
annotations, frozen scorer config, and reserved-evaluation hash), and positive
owned-compute time/RSS budget caps; at least 1,000 independently adjudicated rows; at least 300 distinct
problems; at least three platform-pair groups and three families; non-overlapping
training/evaluation problem and family sets; and a valid frozen split manifest.
The initial evaluation judgment artifact is reserved by its identity hash and
dataset version, even if its permission/version fields are edited. Current
independent review provenance is recorded as such, not as human empirical gold;
review labels alone do not authorize training. The receipt is an explicit
operator approval record, not cryptographic identity proof; the recorded caps
are approval metadata, not process-enforced resource limits. The default small scorer is deterministic ridge regression over the frozen
metadata/text features. Its loss, regularization, tokenizer, seed, and optimizer
are versioned. A failed gate emits a blocked report and produces no model file.
The tool contains no encoder fine-tuning or model download path.

Every successful run records exact input/config hashes, split identity, scorer
manifest and namespace, Python/platform/machine, elapsed time, peak RSS with
platform-aware units, artifact byte sizes, and cash cost as null unless a
receipt supports a value. Benchmark reports and model outputs are immutable:
the CLI refuses to overwrite an existing destination.

## Promotion boundary

A trained scorer or optional encoder remains a research artifact. A separate
promotion check requires at least +0.03 absolute held-out nDCG@5 against the
strongest eligible simpler baseline, uncertainty that supports the gain across
at least two independent clusters, complete predeclared subgroup coverage with
acceptable results, no eligibility regression, and the base
encoder comparison for any fine-tuned encoder. Missing test coverage, overlap,
or review evidence blocks promotion. Relationship quality does not establish
learning improvement; prospective transfer and delayed re-solves remain separate
research gates.

## Validation

Synthetic tests cover known rankings and metric values, deterministic ties and
bootstrap output, unjudged candidates, review/use gating, problem/family leakage,
namespace mismatch, absent-artifact zero-network fallback, immutable outputs,
and promotion rejection. Passing local tests validates code behavior only; it
does not validate the proposed judgment set, curriculum, learner outcomes,
hosted operation, or an encoder.


## Executed diagnostic and first scorer scope

The recorded baseline report in `research/reports/baseline-v1.json` is an
implementation diagnostic, not an unopened final test. It admits 180 reviewed
rows, excludes 66 graph-overlap rows (22 unordered pairs), and leaves 105
source/type groups, only six of which contain multiple judged candidates.
Both methods' conditional nDCG@5 and Recall@10 are 1.0 on that tiny rankable set;
this cannot establish quality, relevance coverage or model improvement. No
optional encoder was loaded and no method was promoted. The initial 90 keys
are frozen as evaluation-only in `research/splits/evaluation.v1.json`.

The first feature scorer is similarity-only; prerequisite and transfer are
separate reviewed directional tasks. Mixing all three labels into a symmetric
feature fit is rejected. The current blocked report records 180 judgments over
73 distinct endpoints, two review-family groups, no training use/right approval,
no separate training split, and no approved compute snapshot. No model file was
created. Synthetic ridge unit checks are algebra tests, not a trained artifact.
The optional local encoder adapter and its unexecuted-model boundary are in
`optional-encoder.md`.
