# Progression-v2 execution and review (2026-10-29)

Baseline: 0c752ca, main. Runtime: Node26.8.1, darwin arm64; references generated with existing SciPy1.14.1 research venv. No dependency installed, model downloaded or learner DB opened. Catalog-v4 was read-only and hash checked (`27a4ca622de50bfa6b86eb7d789abcc856bf40dd8fd511001e91edba20cffde4`).

T1–T7 are separate commits. Full-suite results per part are in tasks.md and commit messages. Independent SciPy fixtures cover 49 beta points and 11 cell scenarios. The user resolved the cold-concept prior discrepancy: authored until own LeetCode evidence, then leave-concept-out updates. Config and evidence contain no title-derived fallback. V1 calculation files remain unchanged.

## T8 receipt and review

Command: `node research/recommendation_audit/adaptive_replay.mjs` (root). Raw output: research/recommendation_audit/receipts/adaptive_replay.out. Exit0. Four scenarios: all-solve, alternating, struggling each60 steps; CF-only25 steps then exhaustion. All36 combinations kappa2/4/8 × lambda.25/.5/1 × default/D-only weights × authored/flat priors completed. Fixed learner seed across variants. All41 runs (including separate unit-attribution comparison) report zero reversed-order mismatches. Each concept reports selection/coverage/waiver status; no claim every concept was selected within60 steps.

Unit-attribution comparison is an explicitly printed research-only source transformation (two own-count denominators 1/k ->1, source hash recorded); runtime stays1/k. It never changes production files. Source-transform membership count fails if that implementation changes. This receipt measures behavior, not relevance, calibrated probabilities or learning benefit.

Reviewer: independent 6.1 Terra AI source/receipt review (not authenticated human review). T8 behavioral sign-off recorded 2026-10-29 after completion. Reviewer-authored reference/hash/review helpers are not represented as independently self-reviewed; parent reviewed integrations and replay script. Findings fixed before rollout: separate review lane, action displayed as focus stage, complete v2 code fingerprint, historical rating cutoff. Earlier missing-best-action finding was retracted after re-reading final loop.

## Measured local cost (single run, not a platform benchmark)

Read-only catalog-v4, synthetic10 sequential recommendations and then ten evidence-prefix transitions, Node26.8.1 macOS arm64: ten plans618.259292ms, prefix replay619.67925ms, 11 transition entries. No Windows/Intel Mac measurement. Immutable catalog metadata indexed in WeakMaps; no learner evidence cached. Long-history scaling is still a release concern.

## Preservation and limitations

V1 serialized pre-integration fixture stays byte-identical when selected. V1 historical implementation fingerprint is deliberately frozen; v2 has a separate implementation fingerprint/config hash. No migrations, ledgers, Markdown parser, backups or v1 planner/calibration/progression/core-policy behavior changed. Existing snapshots remain history. Weekly retention/calibration metrics still use their existing v1 metric definitions; the label correction does not claim v2 metric validation. No live AI calls, browser smoke, Windows run or causal evaluation.


## T9 rollout receipt

V2 is now the default for journal/local runtime only after T8 behavioral sign-off. CLI `journal --progression v1` and local `PROGRESSION_VERSION=v1` select historical behavior. Legacy `plan` unchanged. T9 tests explicitly keep v1's old progression expectations on v1 while testing the default v2 separately. Actual learner ID is forwarded for tie-breaking. No real DB was opened to activate the new policy.

Final Node regression286/286 passed; Python74/74 passed. Chrome `SMOKE_OUTPUT_DIR=<fresh temporary directory> node hosted/workspace-browser-smoke.mjs` exited0: three mocked-provider scenarios (dual_success, partial_failure, both_failed), widths1440/834/390. Structured receipt is browser-smoke.v1.json; screenshots remain private/temporary at /tmp/progression-v2-browser.IpuWKR. This proves synthetic browser behavior, not live APIs. Slack completion-label parity is payload-only; nothing sent. Windows/Intel Mac and learning efficacy remain unverified.

Reviewer handoff: observations/config and beta-math -> leetCodeCells/platformAbilities -> readiness/follow-up/reviews -> planProgressionV2 -> journal-v2/journal -> CLI/local defaults. Review raw T8 output and its source hash; verify golden v1 snapshot, cutoff/seed/config tests and unchanged v1 policy files. No push performed. Unrelated distribution-rights, .claude and historical research-audit work preserved outside implementation commits.
