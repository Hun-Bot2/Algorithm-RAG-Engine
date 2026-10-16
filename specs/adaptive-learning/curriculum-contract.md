# M4 curriculum validation contract

## Task and affected files

Add `src/learning/curriculum.py` and focused fixture-style tests in
`tests/learning/test_curriculum.py`. This is M4 validation tooling; it does not
approve a curriculum, create relationship judgments or implement mission policy.

## Requirements and design

`validate_curriculum(document, require_release=False)` SHALL validate a versioned
JSON-shaped dictionary and return it without mutation, or raise ValueError.
Strict field sets prevent misspelled evidence fields from being silently ignored.

Root fields: schema_version (integer 1), curriculum_version (nonempty string),
problems (list), relationships (list).

Each problem has exactly problem_key, canonical_url, native_difficulty, access_state,
concepts and review. Keys use existing platform-qualified validation. Difficulty
is a nonempty platform-native string, not a cross-platform numeric conversion.
Access is free, paid, unknown or unavailable. Concepts are unique nonempty strings.
URLs use HTTPS and the platform's host: leetcode.com, atcoder.jp, codeforces.com,
or www.acmicpc.net. Require a path; reject credentials, query strings and fragments.
URL validation is structural and does not establish current online availability.

Review has exactly status (proposed/approved), reviewer and evidence. Reviewer and
evidence are strings or null; approved requires both nonempty. These are recorded
review assertions, not proof of independent human approval or reviewer identity.

Each relationship has exactly source, target, type and review. Type is similarity,
prerequisite or transfer. Endpoints must exist and differ. Duplicate typed directed
edges are invalid. Approved edges require approved endpoints. Prerequisites point
from prerequisite to dependent problem; all prerequisite edges, including proposals,
must form a DAG. Similarity/transfer edges do not imply prerequisite order.

Draft validation accepts small proposed sets and explicitly unknown access.
Release validation additionally requires at least 90 problems, all problems and
relationships approved, and every problem explicitly free. This is a conservative
initial free-only gate, not rights clearance or publication permission. The
independent 180-judgment evaluation set is a separate release gate not verified
by this document or function. Nothing here admits historical LLM proposals as truth.

## Validation and rollout

Test malformed documents, duplicate identities/edges, missing endpoints,
self-relations, direction/cycles, approval requirements, cross-platform URLs,
unknown/paid access at release and preservation of input. No network calls or
model dependencies. Run focused tests then full existing offline suite; verify
historical asset hashes. Rollback removes this unused validator; legacy entry
points and stored learner events do not change. M4 remains open until actual
curriculum and evaluation material receive independent review.
