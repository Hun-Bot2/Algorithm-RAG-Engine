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


## M4-M10 local implementation and open gates

Executed 2026-10-16 after integration:

- Python full discovery: 60 tests passed (56 learning tests and 4 pre-existing
  legacy artifact tests). `discover -s tests/learning` independently passes 56.
- Node full hosted suite: 53 tests passed, including actual loopback HTTP. The
  sandbox's socket restriction required an escalated local-only run; it then
  passed. No failed assertion is being waived as an environment issue.
- Original asset verification: 7 present hashes match; 1 known missing file.
  Curated release registry and generated-module hashes also match. Neither
  historical catalog/index nor evaluation files were regenerated.
- Strict 90-problem release validator passes. Reviews: Luna draft, independent
  Sol algorithm/access review. 40 approved edges; 180 adjudicated evaluation
  rows. These are AI-reviewed assertions, not human empirical ground truth.
- Real Chrome 154.0.8037.100 at 390x844: no horizontal overflow or unhandled JS
  exceptions; labeled controls, manual assisted attempt/history, timer recovery
  and explicit confirmation, offline reload with same-ID single insertion,
  learner-scoped pending writes, manual contest, participation with unavailable
  time, and full learning backup export all passed.
- Actual local workerd/D1 through pinned Wrangler/Miniflare: migrations, RSA/JWKS
  Access validation, authenticated assets, owner-bound event insert, identical
  retry, atomic conflicting restore, full backup, nonempty restore refusal and
  privacy-preserving operation counters passed. No inference or Slack call was
  allowed. This is local runtime evidence, not deployed Access/D1 proof.
- Wrangler 4.151.0 dry-run build passed: 188.41 KiB uncompressed / 35.27 KiB gzip,
  three static assets, DB and ASSETS bindings. No upload/deployment occurred.
- `git diff --check`: passed. Existing Ralph changes and unrelated tests remain
  outside the intended implementation commit.

Raw browser/runtime receipts and screenshots are in ignored private-learning/;
non-sensitive receipt copies accompany this spec. Benchmark outputs are under
research/reports/. After graph-overlap exclusion, only six groups have multiple
judged candidates: reported conditional perfect ranks are not quality evidence.
The blocked training receipt produced no model. Real compact-model inference,
new training labels and budget approval, shadow promotion and learning outcomes
remain open. Unit ridge algebra and injected-encoder checks are synthetic code
validation, not model runs.

The hosted account/D1/Access owner and Slack workspace/destination are not yet
selected. Delivery, scheduler, provider activation, real smoke messages and cron
remain disabled. Actual quotas, workstation-off delivery, phone alerts and the
30-day pilot are pending user selection and deployed verification.

Suggested implementation commit:
`feat(learning): implement private deterministic adaptive learning pilot`.
