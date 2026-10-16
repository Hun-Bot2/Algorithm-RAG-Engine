# M1-M3 validation receipt

Executed locally on 2026-10-16. This receipt covers foundations, not a hosted release.

- `python3 -W error::ResourceWarning -m unittest discover -s tests -v`: 22 tests
  passed (18 new foundation tests plus 4 pre-existing artifact fixture tests).
- `python3 scripts/learning/assets.py verify`: 7 present assets verified; 1
  explicitly missing (slug manifest). Six original historical files and the new
  namespace manifest retain their recorded hashes. No vectors/data regenerated.
- `git diff --check`: passed.

Covered: event validation, UTC normalization, imported completion semantics,
append-only storage, corrections and branch rejection, owner partitioning,
conflicting/global IDs, atomic failed import, repeated import, export/restore,
open/unconfirmed/confirmed timers, overlapping-time union, timer replay after
restore, migration checksum and live-schema drift, concurrent duplicate writers,
restricted bundle exclusion, tampering/path escape/destination guards, and CLI
backup/restore without application credentials and with private file modes.

No network calls, API/model clients, model downloads, Slack sends, deployment,
schedule activation or training were performed. Passing fixture tests preserves
local fixture expectations; it does not prove live legacy Slack rendering.
No D1/Worker/Access authentication, hosted quotas, phone alert, curriculum review,
mission policy, model-quality result or longitudinal learning benefit is claimed.
The CLI is a trusted local tool, not an authenticated multi-user service.

Existing modified Ralph files and pre-existing tests were preserved. No commit
or staging was performed. Suggested commit (only intended foundation paths):
`feat(learning): add no-AI specifications and auditable local event storage`.


## M4 validation tooling receipt

Implemented the first small M4 step with Luna for the validator and repetitive
fixture cases, followed by parent review and fixes. This does not complete M4.

- `python3 -m unittest tests.learning.test_curriculum -v`: 14 tests passed.
- `python3 -W error::ResourceWarning -m unittest discover -s tests -q`: 36 tests
  passed, including existing foundation and legacy artifact fixture tests.
- `python3 scripts/learning/assets.py verify`: 7 present hashes match, 1 missing.
- `git diff --check`: passed.

Checks include platform-specific hosts, malformed envelopes, review evidence,
endpoint integrity, typed edge identity, proposal cycles, a 1,100-node chain and
free/approved release constraints. Fixture reviewer/evidence strings are synthetic
test inputs, not approved curriculum records. No real curriculum or evaluation
judgments were created. No external actions or optional model work were performed.

M1-M3 were committed as `b80d8e1`; unrelated Ralph changes and pre-existing tests
were excluded. The M4 validator is an unused offline helper and changes neither
legacy ranking nor stored learner events.
