# Pilot annotation guide v1 — proposed, to validate with human reviewers

Keep four judgments separate. Similarity does not establish prerequisite direction,
transfer usefulness, equal difficulty, or learner mastery. Record uncertainty and
missing information instead of guessing. The publisher descriptions may omit
sections or have translation issues; note them in the rationale.

## Algorithmic similarity

- **0:** unrelated solution techniques/invariants, despite possible narrative overlap.
- **1:** only a broad concept in common; solving one gives little specific technique
  practice for the other (for example generic array handling).
- **2:** a shared core technique or invariant, with meaningful adaptation required.
- **3:** near-equivalent algorithmic practice: closely matching key invariants,
  solution structure and complexity demands. Difficulty can still differ.
- **Unable to judge (null):** insufficient understanding, missing statement content,
  alternative solutions not assessed, or another explicit limitation.

Use statement details, independently derived solution sketches, invariants and
constraints as evidence. Do not assume shared tags, similar titles or model rank
prove a specific algorithm. Record valid alternative solutions and disagreements.
No editorial or hidden tests are collected automatically by this tool.

## Prerequisite direction

`query_before_target` or the reverse requires a concrete foundational technique
in the earlier problem that the later problem assumes or combines. Explain that
connection and the later extension. `neither` means no supported directed
prerequisite; `unknown` means insufficient evidence. Similarity alone gives no edge.

## Transfer

`yes` means working through the query plausibly supports adapting a technique to
the target; state the adaptation. `no` requires an assessed absence of that useful
connection. `unknown` is appropriate when you cannot assess instructional value.
This is an expert judgment hypothesis, not measured learner transfer success.

## Relative difficulty

Choose query easier, similar, target easier, or unknown. Compare required insights,
implementation burden, constraint-driven complexity and unfamiliar combinations.
State the basis and topic. Do not translate platform ratings using arbitrary fixed
thresholds. These judgments compare problems for an explicitly described assumed
learner background; if that background changes the order, explain and use unknown
when necessary. They do not establish personalized difficulty for every learner.

## Native identity

A canonical URL plus a matching name is insufficient semantic verification. Compare
statement details, variable meanings, constraints and public samples where available.
Record which agree or differ and whether translation or version changes explain it.
Use a catalog-qualified key. Missing candidates remain unresolved; never force the
nearest title. If two reviewed identities disagree, reject/preserve unresolved and
collect new evidence; the acceptance gate requires agreement on a single identity.

## Independence and adjudication

Discuss the guide on practice examples outside the scored pilot. Then review the
pilot independently without seeing other labels or retrieval ranks. Distinct human
reviewers and a third adjudicator are required; AI drafts stay explicitly AI.
Adjudication may accept an all-unknown decision, which remains missing evidence in
analysis. Agreement is not objective truth. Log disputes and guide changes; freeze
a revised guide and new pilot before expanding when definitions need correction.
