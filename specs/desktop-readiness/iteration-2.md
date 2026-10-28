# Iteration 2 — review and path correctness (2026-10-29)

Baseline: main, HEAD 5fe47c864dabb5a94e2e97aa6075ee371064863f. Existing changes to AGENTS.md, ReadME.md, RECOMMENDATION_ENGINE/README.md, git-sync and untracked .claude/distribution/desktop documents preserved. Reviewed status.md and review-log.md against source. No commit made.

Accepted F1/F2/F3: CLI main guard used an unencoded URL, root defaults used URL.pathname, and catalog ATTACH concatenated an unescaped URI. Changes use Node file URL conversion; catalog still verifies receipts and attaches immutable/read-only. New synthetic tests copy hosted/ into a special-character checkout and verify default DB location and actual CLI output; catalog tests cover spaces, Korean, #, %41 and apostrophe, reject writes and check no stray files.

F4 partially addressed: adapter refuses a schema_migrations version newer than known migrations before applying any migrations; closes the connection and gives an upgrade instruction. A synthetic version-999 DB retains its row and sole table. Automatic pre-migration backups remain open: backup location, retention, disk-full recovery and app-data upgrade contract need specification before implementation. Adapter stays at its current path to avoid an unrelated rename.

Documentation review correction: D1 is resolved by the user's Productization & Desktop Readiness instruction: target is an installable Windows/macOS product for users importing their own repositories. No new permission request is required for that direction. Framework, Git bundling, auth flow and catalog delivery decisions remain open. Existing Claude test counts are reported evidence, not tests re-executed here.

Validation: first path/CLI/git-sync selection passed 15/15. Final path/CLI/backup/core selection passed 30/30. git diff --check passed. Tests use temporary or in-memory stores; no real learner database opened. No Windows/Intel Mac/browser/package benchmark executed. Whole Node suite not repeated; Claude's 245/245 is independently reported, not claimed here. Python/assets unchanged.

Reviewer: inspect hosted/cli.mjs main guard and ROOT, hosted/local.mjs ROOT, hosted/mastery/catalog.mjs ATTACH URI and SQL quote escaping, hosted/test/helpers/d1.mjs newer-schema preflight, hosted/test/desktop-paths.test.mjs. Check no recommendation policy or ledger schema changed. Known remaining limitation: entry-point guard compares resolved argv URL to module URL; symlink invocation is not covered (temp path canonicalized in test). Actual Windows SQLite URI handling still needs a Windows run.

Next: app-data and safe migration/backup requirements, pre-migration backup implementation with failure tests, then runtime adapter relocation. Suggested commit: fix(desktop): handle encoded paths and reject newer database schemas.
