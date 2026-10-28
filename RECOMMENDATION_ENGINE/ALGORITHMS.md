# Algorithms: product policy versus research

Source references below describe the audited checkout, not a claim of learning effectiveness.
Architecture and version boundaries are in [README](README.md); concrete learner replay is in
[DATA_FLOW](DATA_FLOW.md).

## 1. Actual local recommendation policy

### Independent outcomes and concept evidence

[ledger.mjs](../hosted/mastery/ledger.mjs) `solvedIndependently` L49 requires
`outcome === 'independent' && hints === false`. `related.mjs::trustedConcepts` L10–11 accepts
only `approved`/`ai_reviewed` catalog entries. Neither an accepted platform submission nor an
unmarked Git commit is automatically independent.

[progression.mjs](../hosted/mastery/progression.mjs) `collectEvidence` L34–78 replays attempts
chronologically per concept, resolving canonical keys. Easy/Medium evidence is a **Map keyed
by problem**, not the number of commits or repetitions. Variation signatures use an explicit
`variation_group`, else sorted trusted concepts + native tags (L15–20); this is a metadata
proxy, not a verified distinct solution technique.

The versioned [pattern](../hosted/mastery/progression-pattern.v1.json) requires:

1. Three distinct LeetCode Easy problems **and three distinct variation signatures**.
2. Two distinct Medium problems/variations, credited only once Easy requirements are already met
   at that point in replay; earlier Medium attempts are not automatically credited retroactively.
3. A previously unattempted CF/AtCoder problem, independent, `activity_type: transfer`, referencing
   a credited Medium source. Candidate native difficulty ceilings: CF 1,000 / AtCoder 800;
   experimental AtCoder difficulty models excluded.
4. An independent `review` at least seven days after transfer. Dedicated retention candidate
   selection returns null; the general review queue still operates.

Easy failure/assistance clears Easy and downstream evidence; Medium failure preserves Easy but
clears Medium/downstream; Hard failure is separately observed without erasing Easy/Medium.
These are explicit software rules, not empirically validated thresholds. Tests:
[mastery-progression.test.mjs](../hosted/test/mastery-progression.test.mjs).

`candidateFor` L85–103 filters active/free/eligible/trusted-concept candidates, excludes attempted
and previously recommended keys, and requires a new variation when configured. **Easy/Medium
choose the first remaining canonical key lexicographically**, not the closest embedding or
highest expected learning gain. Transfer sorts by conservative solve probability. Concept
selection L147–163 prioritizes remediation, configured targets, interview concept order, then
key order. `priorExposure` L80–83 considers all prior plans through the date: its exclusion is
not a finite “recent” window, despite the explanation wording.

### Per-platform difficulty calibration

[calibration.mjs](../hosted/mastery/calibration.mjs), `practiceObservations` L75–85, excludes
reviews and reported completions; it includes independent/assisted/failed practice outcomes.
Repeated attempts count here even though progression deduplicates problem identities.
`y=1` only for independent/no-hints; other included outcomes give `y=0`.

**LeetCode** (`fitLeetCode`, L53–70): per-level Beta priors are Easy `(3.2,.8)`, Medium `(2,2)`,
Hard `(.8,3.2)`. With s successes in n observations:

`alpha = alpha0+s; beta = beta0+n-s; mean = alpha/(alpha+beta)`

`variance = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))`.

Weighted pool-adjacent-violators enforces Easy mean ≥ Medium mean ≥ Hard mean, using posterior
pseudo-count weights. The implementation keeps each original cell's SD after pooling means;
this is a heuristic uncertainty summary, not a fully refitted joint posterior. Prediction is
`max(0, mean−SD)` / mean / `min(1,mean+SD)`. Fewer than **six** observations is labelled
`insufficient_evidence`, independently of the progression 3/2 gates.

**CF/AtCoder** (`fitAbility`, L36–50): Gaussian/Laplace sequential update for logistic likelihood,
`p = sigmoid(a*(mu−difficulty))`; CF `a=ln(10)/400`, AtCoder uses the native discrimination.
Before each observation, `v += 25^2 * elapsed_weeks`. Then

`v_new = 1 / (1/v + w*a^2*p*(1−p)); mu_new = mu + v_new*w*a*(y−p)`.

AtCoder experimental observations have weight .5. CF prior is 1000±350 absent a rating, otherwise
rating±250; AtCoder is 0±400 or reconstructed internal rating±300. Displayed AtCoder rating is
inverted using the participation correction (`atcoderInternalRating`, L12–16), not compared
numerically to CF or LeetCode. Unknown difficulty stays unknown; the catalog rejects the
−10000 AtCoder sentinel instead of clipping it to easy.

Prediction L99–110 uses `sigmoid(a*(mu−SD−d))` conservatively; marginal mean approximates logistic
uncertainty by dividing `a*(mu−d)` by `sqrt(1+pi*a^2*v/8)`. `calibrationReport` requires 30 pairs
before reporting Brier score `mean((p−y)^2)` and three-bin diagnostics. A valid formula/test does
not demonstrate calibration on this learner: today's n=3 is still sparse.

### Base planner, review and related-practice score

[planner.mjs](../hosted/mastery/planner.mjs) `planDay` L96–195 allocates due reviews first
(15 min each), repair, then 30-minute practice/exploration within budget. Cold-start targets
rotate .85/.65/.45; ordinary conservative success band is .60–.80, stretch .35–.55. Concept need
uses Beta(1+s,1+f) over 56 days plus .5 for an explicitly targeted concept. Exploration is a fixed
calendar rotation with a weekly budget cap, not a learned bandit. The journal keeps review slots
and substitutes progression practice (`journal.mjs::journalPlan`); these base slots are not all
shown in the journal.

[reviews.mjs](../hosted/mastery/reviews.mjs) ranks due reviews by urgency
`2*overdue_days + (priority ? 10+2*min(consecutive_failures,5) : 0)`, then retention risk, date/key.
The shared state machine [core/policy.mjs](../hosted/core/policy.mjs) L7,16–40 uses intervals
1/3/7/14/30/60 days. Failures reset to next-day priority review; unverified completion can enter
review without advancing mastery.

[related.mjs](../hosted/mastery/related.mjs) `relatedCandidates` L15–55 combines reviewed-edge
support E, concept Jaccard J, best title-channel rank r and conservative probability p:

`score = 4E + 3J + 120/(60+r) + 2*max(0,1−abs(p−.70)/.70) − D − 1.5*S`.

Absent ranks contribute 0; D=1 for a degraded/unknown endpoint; S=1 for semantic-only fallback.
Components are rounded to .001 before summation. Normal candidates require concept overlap or a
reviewed edge; an unclassified candidate may instead use reliable-title semantic neighbors,
clearly labelled and excluded from mastery evidence. These weights/penalties are policy choices,
not fitted or validated outcome effects.

## 2. Embeddings and similarity research

| Stage | What actually ran; status |
|---|---|
| Retired batch | OpenAI `text-embedding-3-small` / 1,536D index; squared-L2 and random top-pool sampling. Removed, preserved at `legacy-pipeline-final`. |
| Small pilot | [minilm_pilot](../research/minilm_pilot/REPORT.md): 90 authored English metadata annotations, title vs title+rationale; only three eligible ranking groups. Historical conditional scores do not establish catalog relevance. |
| A1/A2 full-catalog | [inputs.py](../research/full_catalog/inputs.py): A1 eligible titles without encoder flags; A1-all all eligible titles for lexical diagnostics; A2 LC/CF title+tags, matched within variant. Word/character TF-IDF and MiniLM compared on frozen inputs. |
| Statement research | [statements.py](../research/algorithm_system/statements.py) L62–116: one admitted 9,424-row publisher corpus, word/character TF-IDF, MiniLM title, prefix title+statement and complete-token chunk pooling. No quarantined LC text admitted. |
| Unified title map | [build.py](../research/unified_map/build.py): exact cosine title neighbors, taxonomy Jaccard and matched-cohort hybrid. [full_embed.py](../research/unified_map/full_embed.py) extends title vectors to 24,297 nodes. |
| Language correction | [language_bias](../research/language_bias/REPORT.md), commit `6580d21`: English-translated titles with the same MiniLM selected over multilingual alternatives; these cached title neighbors feed catalog channels, not a live model runtime. |

Pinned MiniLM is `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 384D, 256-token limit. The local backend verifies
exact model-directory membership and hashes; no download/server fallback. Normalized embeddings
use cosine `x·y/(||x||||y||)` with blockwise exact ranking, stable score/key ties and self/platform/
duplicate masks. Cosine is not a relevance probability. For unit vectors, squared-L2 = `2−2*cos`;
this equivalence does not justify comparing different embedding spaces or unmatched input fields.

Statement chunking uses original tokenizer IDs in disjoint 254-token windows, adds special tokens,
unit-normalizes chunk vectors, averages weighted by content-token count, then normalizes again.
It prevents prefix truncation, but does not establish original publisher-section completeness.
No model was trained by this pooling operation.

Unified research hybrid is **not the runtime score above**: equal-weight RRF over the union of
constituent Top50 lists, `RRF=sum(1/(60+rank))`. Only positive-Jaccard tags receive a rank/contribution
(`build.py::_rrf` L198–238). Missing tags remain unknown; evaluating tag rankers by tag agreement is
circular. Historical labels are directed, AI-reviewed metadata/recollection judgments; abstentions
and unjudged pairs are not negatives.

## 3. Dimensionality reduction: implementations and mathematics

These are research methods, **not the journal's online selection algorithm**.

| Method | Mathematics and actual implementation |
|---|---|
| PCA | Center X by column mean, SVD `(X−mean)=UΣVᵀ`, retain `Z=(X−mean)V_d`; normalize Z for cosine retrieval. Maximizes retained centered variance, not relevance. Title ablation uses randomized PCA; statement `compress.py::pca_project` uses full NumPy SVD; full 2D map uses sklearn full-SVD PCA. |
| Truncated SVD | Uncentered `X≈U_dΣ_dV_dᵀ`, project `XV_d`, normalize. Preserves a different geometry from centered PCA. Title uses sklearn randomized TruncatedSVD; statement `svd_project` uses NumPy SVD/rank checks. |
| Random projection | Statement `compress.py::gaussian_project`: seeded Gaussian matrix, entries `N(0,1/d)`, project/normalize. Johnson–Lindenstrauss motivates distance preservation probabilistically; no guarantee of semantic relevance for these data. |
| UMAP | Builds adaptive local fuzzy neighborhoods and optimizes a low-D fuzzy cross-entropy embedding. Executed cosine metric, 15 neighbors, min_dist=.1, 2D, one thread, fixed seed. Local structure emphasis; global distances/density are not calibrated. |
| t-SNE | Gaussian high-D neighborhoods (perplexity 30), symmetrized P; low-D Student-t Q; minimizes `KL(P||Q)`. Executed cosine input metric, Barnes–Hut, 1,000 iterations, automatic learning rate. Recorded sklearn PCA-init repeats were identical; random-init repeats are used for seed sensitivity. |

Sources: [reduction.py](../research/unified_map/reduction.py) L74–191,
[full_project.py](../research/unified_map/full_project.py) L32–63,
[compress.py](../research/algorithm_system/compress.py) `pca_project`, `svd_project`, `gaussian_project`.
2D neighbors are **Euclidean**, not coordinate cosine (the origin is arbitrary). Trustworthiness
penalizes false low-D neighbors by their high-D ranks; continuity penalizes lost high-D neighbors.
These and Top10 overlap diagnose projection fidelity, not algorithm clusters or learner benefit.

## 4. Metrics verified in this audit

Recomputed `mean_i |N10_original(i) ∩ N10_method(i)|/10` from saved ranked artifacts; hashes were
checked against their receipts. No encoder fitting or quarantined legacy-vector read was needed.

| Method | Title A1, n=21,297 | Admitted statements, n=9,424 |
|---|---:|---:|
| Original 384D | 1.000000 | 1.000000 |
| PCA 256 / 128 / 64 | .877203 / .770320 / .596600 | .679499 / .652472 / .575202 |
| SVD 256 / 128 / 64 | .957356 / .799221 / .611743 | .973875 / .842763 / .681335 |
| Gaussian RP 256 / 128 / 64, seed 17 | Not in this title ablation | .598376 / .467445 / .324066 |

Receipts: [title reduction](../research/unified_map/reduction-receipt.v2.json),
[statement compression](../research/algorithm_system/compression.v1.json).
Columns are different corpora/protocols: do not rank one corpus against the other.

For all-node 2D maps (24,297 titles), recomputed means of saved per-node neighbor-intersection
**counts divided by 10**, and checked coordinate hashes:

| Representation | PCA | UMAP | t-SNE |
|---|---:|---:|---:|
| raw full-v1 | .034519 | .289406 | .482689 |
| translated full-v2 | .034251 | .302107 | .486031 |

These are all-platform Top10 overlaps, not the separate 3,000-node stratified cross-platform
ablation in `reduction-receipt.v2.json`. Raw legacy 1,536D timing/geometry receipts were inspected
only; their underlying quarantined vectors were not reprocessed and their latency was not rerun.

Recounted [labels.v1.jsonl](../research/full_catalog/labels.v1.jsonl): **3,647** records = 2,886
judged + 761 abstentions; grades 2/3 give **44 positives**. The reported label sparsity therefore
is real. Evaluation uses positives `grade>=2`, observed precision `positive/judged`, and bounds
`[positive/all_slots, (positive+unknown)/all_slots]`. Conditional nDCG removes unknowns before
ranking a shared judged-target intersection, with gain `2^grade−1` and `log2(rank+1)` discount;
it is not full-catalog nDCG. See [evaluate.py](../research/algorithm_system/evaluate.py) L142–149,245–300.

Rehashed the language evaluation and reapplied `select_representation.decide`: winner **C2**,
consistency gain **.4899833055** over C0. Receipt values .505676 vs .015693 reproduce that gain;
MRR .664254 is a stored evaluation value, not newly rerun translation inference. The random
control demonstrates why lower language homophily alone is insufficient. No research metric here
establishes algorithm equivalence, calibrated human difficulty, or educational effectiveness.


## Contest classification: new optional practice-goal policy

`hosted/mastery/contest-discovery-fit.mjs::contestDiscoveryFit` (`contest-opening-goal-v1`)
does **not** change `liveProposals`, `virtualProposal`, problem progression or LLM prompts.
Those older proposals and the calendar's new classification are different interfaces.

1. Read active attempts through the canonical ledger, preserving correction filtering. Count distinct
   canonical problems on the contest's platform only when `outcome=independent`, `hints=false`,
   `activity_type=practice`. At least **6 distinct** independent solves with known native difficulty and the existing calibration's
   `status=estimated` (at least 6 usable observations) are required. Ratings/solves from another
   platform cannot satisfy this gate. Native difficulty units remain `cf_rating`, `atc_irt`, `lc_level`.
2. Reuse existing `fitAll/predict`; no second skill estimator. Compare at most the five most recent
   already-started catalog contests of the same platform/format. Format identity is a coarse series
   proxy from native AtCoder IDs or contest-name format markers, **not difficulty evidence itself**.
   Unknown formats and AHC are excluded. In each historical contest, retain the easiest eligible
   problem with known native difficulty; exclude AtCoder sentinel -10000. At least 3 such contests
   must supply comparable native difficulty evidence. Missing LC contest membership stays missing.
3. For historical contest j let p_j be the existing predictor's conservative probability on that
   problem (platform ability one SD lower; LC Beta mean minus SD). Define q = sum(p_j)/m.
   q >= .65 => **수준에 맞는 추천 / Recommended for your level**, specifically an opening-problem
   practice goal; .30 <= q < .65 => **도전 과제 / Stretch challenge**; otherwise =>
   **판단 근거 부족 / Insufficient evidence**. Stale schedules or missing evidence also yield insufficient.
4. Return policy/catalog version, input attempt IDs, independent native difficulties, calibration
   status/sample count, historical contest/problem keys, p_j and q. Reasons are deterministic Korean,
   explicitly disclosing that unpublished current problems, full completion and official rating are
   not predicted. The .30/.65 thresholds are new unvalidated heuristics, not measured accuracy.

Registration/rated eligibility is **unknown** for normalized schedules and must be checked at the
linked official announcement. AtCoder's published rated range is displayed literally; it is neither
problem difficulty nor a prohibition on unrated participation. A recommendation is conditional on
checking participation rules. The current user's three LC Easy solves cannot yield a confident
contest classification; the UI correctly reports insufficient evidence for all platforms.

For manual/older non-discovery entries, the preexisting UI can separately describe user selection or
same-contest participation as review/retry evidence. That is not the platform-native proxy above and
is not a skill estimate. Research projection metrics in earlier sections are never contest features.
