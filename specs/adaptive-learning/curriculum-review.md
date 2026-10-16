# Independent algorithm curriculum review v1

## Scope and decision

Luna drafted the candidate curriculum and relationship judgments. Sol reviewed
the 90 algorithm annotations, 64 proposed graph edges and 180 proposed labels
independently on 2026-10-16. The review approved 90 algorithm candidates,
retained and approved 40 graph edges (28 prerequisite, eight similarity, four
transfer), and accepted 180 directed relevance judgments across 60 pairs.
`sol-independent-review` is an AI reviewer identifier, not an independent human
expert or empirical learner validation. Each approved seed record has a
record-specific algorithm rationale. The 24 dropped edges are not part of the
current graph; their omission is not a measured effect on recommendations.

Approval means the short description and learning relation are reasonable for
curriculum design. It does not verify the full official statement on every
candidate, live submission, suitability for a particular learner, time
estimates, licensing, or improvement in learning outcomes. The access audit is
in `curriculum-sources.md`.

## Edge criteria and corrections

A prerequisite points from a useful instructional foundation to the later task;
it is not a claim that every solution literally requires that earlier problem.
Similarity requires a shared operative invariant or method. Transfer requires
a specific reusable step. Broad tags or vocabulary alone did not qualify.

The review removed the unsupported chain from Group Anagrams to the Premium
Encode and Decode Strings task, along with weak paths such as House Robber to
Coin Change, Unique Paths to Longest Common Subsequence, and Course Schedule to
undirected tree checking. It also removed comparisons based chiefly on words
like `area`, `interval`, or `string` when their solution invariants differed.
The Premium task replacements are reflected in the graph and evaluation set;
the graph remains a prerequisite DAG. Retained directed examples include Binary
Search to rotated search, Daily Temperatures to Largest Rectangle, and AtCoder
Coins to Otoshidama, each with a method-specific review rationale in the seed.

## Evaluation-label adjudication

All 180 judgments retain `annotator=luna-draft` and now record
`adjudicator=sol-independent-review`, `adjudication=accepted` and a 0-3
relevance score. `accepted` means the reviewed score is retained; a zero score
is an accepted negative example, not a rejected record. Twenty individual
scores and rationales were corrected after checking direction or the actual
invariant. For example, 3Sum is not prerequisite to Container With Most Water;
Network Delay Time is not prerequisite to the simpler Rotting Oranges BFS; and
Pacific Atlantic Water Flow is not prerequisite to Number of Islands.

`direction=source_to_target` records the proposition being judged. Reverse-order
countercases appear as zero prerequisite judgments, including RJ-019, RJ-030,
RJ-057 and RJ-060. Hard negatives include shared `area` language for Domino
Piling versus Container With Most Water (RJ-056), and shared `greedy` language
for Katana Thrower versus Jump Game (RJ-045). Labels are AI judgments, not
ground truth established by human study, and no model-derived completion or
recommendation performance is claimed.

Twenty of the 60 evaluation pairs also occur as source-target pairs in the
approved graph. A recommender that consumes the graph cannot treat those 20
as an independent held-out test. Exclude those pairs from held-out metrics or
label the full-set result as a diagnostic. The file remains `permitted_uses:
["evaluation"]`; do not train on it or use it as a mission source.

## Validation and open gates

Structural validation confirms the 90 approved nodes, 40 approved graph edges,
registered annotations, DAG, and 180 adjudicated labels. All 90 retained
statements were checked on official pages for public statement access: 87 on
direct task pages and three on official AtCoder printable contest pages.
`validate_curriculum(..., require_release=True)` therefore passes its narrowly
defined 90-approved-free structural gate. That check cannot establish algorithm
correctness beyond the recorded review, live account submission,
location-specific access, rights to republish content, or any measured learning
benefit. The three AtCoder direct-link inspector failures remain documented in
`curriculum-sources.md`.
