# Recommendation input comparison requirements

- The comparison SHALL support three distinct objectives: solving-idea practice, weakness remediation, and problem-content similarity.
- Methods SHALL use the same query/candidate cohort for any direct input comparison. Different platform coverage MUST be disclosed.
- Existing pinned runs and labels MUST remain immutable. Existing title-only labels MUST NOT become statement/strategy relevance labels.
- The first analysis SHALL reuse hash-verified stored title, title+prefix, full-content chunk, word TF-IDF and character TF-IDF outputs. It SHALL distinguish historical encoding timings from newly computed diagnostics.
- Ranking agreement and candidate concentration MUST NOT be reported as recommendation accuracy. Unjudged pairs remain unknown.
- New strategy/code representations SHALL retain source, authorship and language information. LLM-produced strategy descriptions are derived hypotheses, not verified algorithms.
- Weakness remediation SHALL be evaluated against learner needs separately from semantic similarity. A review flag alone is not weakness evidence.
- No collection, model installation/training, paid inference, new statement admission, ranking-policy change or scheduled job is included in this bounded comparison.

## Executed input experiments, second iteration
- New experiments SHALL use the already approved pinned offline MiniLM. Raw statement admission and model training remain outside scope.
- The authored curriculum rationale/concept experiment SHALL preserve its AI-draft/AI-review provenance and SHALL NOT call those explanations human gold or original publisher statements.
- Publisher input variants SHALL identify description/Input/Output/Constraints/Examples/Notes without inventing absent sections. Comparison SHALL use a shared parsed cohort and exclude identical source-content hashes and self matches.
- New vectors, results and reports SHALL live in fresh versioned outputs; old vectors/rankings remain immutable.
- Inspection SHALL expose the actual text fields used for each variant and field missingness, so the user can assess failure cases.
- Missing Cross-encoder weights SHALL be reported explicitly. No random classifier head, bi-encoder score, LLM draft or heuristic SHALL be presented as a Cross-encoder result.

## Approved reranker and requested notebook/blog
The user explicitly approved downloading the named pinned Cross-encoder and requested notebook-based execution plus a blog-style write-up. Download SHALL retain the reviewed revision and safetensors hash; no additional packages or paid API are required. The notebook SHALL validate source/model receipts, re-encode the small authored cohort, rerun actual offline Cross-encoder scoring, preserve outputs, and distinguish historical full-body encoding from fresh notebook inference. Plots SHALL show supported candidate agreement/rank/timing diagnostics, not invented accuracy. The Korean blog SHALL include failed cases, input/cohort boundaries and reproducible references. Publishing outside the repository is not authorized.
