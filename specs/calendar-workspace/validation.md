# Calendar workspace validation — initial redesign (2026-10-20)

This is the initial redesign receipt. The current follow-up, including the later contest-discovery contract, is documented in [refinement-2.md](refinement-2.md). Initial source-availability statements below are historical.

## Implemented and verified locally

- Calendar is the default destination, with month/week navigation, Today, J/K/T shortcuts, arrow-key date navigation, actual/plan/contest layer toggles, daily dialog and responsive selected-day agenda.
- History searches available titles/keys/topics and filters platform, native difficulty, outcome and date. Problem dialog retains all prior/superseded attempts and available provenance. Manual corrections append via existing journal API; imported Git evidence stays read-only.
- Common native dialog supports Escape, contained focus, scroll lock, a sticky close control and return focus (including a date-button fallback when desktop event chips are hidden on mobile). Dialog transitions replace one dialog rather than nest.
- Analytics uses the loaded active attempts: annual count heatmap, daily/weekly attempts and completion records, outcome/topic/native platform difficulty counts, monthly difficulty and independence records, and descriptive seven-day recommendation follow-up matching. All marks open source-row dialogs. Year/platform filtering does not apply to the separately labeled all-history API usage table.
- Gemini/NVIDIA cached outputs, provider status/errors, deterministic fallback and original recommendation evidence are available in a two-column comparison dialog; mobile stacks them. Browsing does not request generation.
- Contest calendar uses existing CF/AtCoder cache and source status, precise start times, durations, safe external links, manual additions and participation forms. LeetCode can be entered explicitly as manual, with no invented provider. Weekend cells are shaded without moving any event.
- Fixed supported timezone is Asia/Seoul. New record/contest/participation forms explicitly convert KST; unchanged correction timestamps are preserved exactly.
- Warm paper/green design tokens, full desktop workspace, mobile navigation, reduced-motion support, live connection/error notices and empty states are implemented. Muted text and secondary dates were darkened after contrast calculation found the prior values below 4.5:1 on paper.

## Commands and results

- Before changes: `node --test hosted/test/*.test.mjs` — **169 passed**, with loopback socket permission. Initial sandbox-only run hit EPERM on local HTTP tests; that run is not reported as a pass.
- Final Node suite: `node --test hosted/test/*.test.mjs` — **177 passed**. New tests exercise leap months/year boundaries, Seoul midnight, explicit DST timezone conversion helper, actual versus plan mapping, superseded history, imported completion uncertainty, search/native difficulty filters, conversion observation window and manual LeetCode schedule provenance.
- `python3 -m unittest discover -s tests -v` — **74 passed**.
- `python3 scripts/learning/build_hosted.py` — passed, reviewed release hashes preserved.
- `python3 scripts/learning/assets.py verify` — **7 verified; 1 explicitly missing**, unchanged missingness.
- `npm run dry-run` in hosted/ (task-local Wrangler paths) — passed, **5 browser asset files**, no upload/deployment. This is bundle proof, not Worker runtime or hosted-browser proof.
- `.venv/bin/python -m research.unified_map.verify private-learning/unified-map/map-v2` — passed, **501 preserved input files**, existing quality flags retained.
- `node --check` on modified browser/server modules and `git diff --check` — passed. No repository lint script exists; no lint result is invented.
- `SMOKE_OUTPUT_DIR=<fresh private directory> node hosted/workspace-browser-smoke.mjs` (also `npm run test:browser` in hosted/) — **3 Chrome/CDP scenarios passed**: both explanations succeed, one fails, both fail. Mocked transports only; zero real provider calls.

Browser assertions cover 1440×900, 834×900 and 390×900 viewports across all four primary destinations; populated-history screenshots also use 1000px height. They check horizontal overflow, explanation side-by-side/stacked layout, modal Tab/Escape and focus restoration, actual record creation, append-only correction preserving timestamp, all-attempt detail, search, analytics drill-down, manual LeetCode contest submission/provenance, week/day views, reduced-motion and offline state. A real browser network-offline save fails, persists one pending event, then retries the same ID successfully once and clears the queue. Reloading does not add model API attempts.

`hosted/browser-smoke.mjs` and `hosted/explanation-browser-smoke.mjs` contain earlier journal-specific selectors and are historical checks; the npm browser command now runs the new workspace check, which covers the current interaction contract. Existing core timer/backup/participation Node tests remain in the full regression suite.

## Receipts and rendered inspection

Local synthetic screenshots and `receipt.json`: `/tmp/calendar-workspace-evidence-verified/` ; npm-command run: `/tmp/calendar-workspace-final/`; final scroll/provenance verification: `/tmp/calendar-workspace-final-checked/`. Browser fixture databases are disposable and never use the real learner store. Screenshots contain authored synthetic problem names and schedules only. Desktop calendar, desktop analytics, mobile agenda/history and comparison screenshots were visually inspected; month/weekday axes, count legend, event symbols and mobile count size were refined afterward.

Public reference Chrome screenshots and observed page summaries: `/tmp/calendar-reference-evidence/`. Notion Calendar redirected to login; no authenticated interface access is claimed. Linear's public page exposes an issue/detail product demo; GitHub's public torvalds profile rendered the actual annual contribution grid; Plot rendered interactive example charts; Muzli rendered its calendar gallery. Official Notion guide supplies the navigation patterns. See [design rationale](design.md) for adopted, modified and avoided patterns and source links. No private external accounts were accessed.

## Explicit unavailable or incomplete scope

- **LeetCode automatic synchronization is not implemented**: no admitted schedule adapter exists, and this task does not add a scraper or infer weekly recurrences. Manual entry and official calendar links are available.
- **Live CF/AtCoder freshness is not verified**: local runtime still reports providers disabled; existing cached/error adapters and fixture regression tests are retained. No successful external sync is claimed.
- **User-configurable timezone is not implemented**: the current backend contract has no timezone field; policy remains Asia/Seoul, visibly labeled. Alternate timezone day-boundary changes require separate backend requirements.
- **Separate attempt-start/completion timestamps and editable notes are unavailable** in the journal contract. The UI exposes actual recorded attempt timestamps, all previous attempts, recorded time and explicit missingness. It does not create unsupported fields or change imported Git provenance.
- **Individual API-request drill-down is not exposed by the journal API**: existing provider/model usage aggregates, coverage, latency, errors and token counts are inspectable; per-request private records remain in the existing CLI. No time-series usage data is fabricated.
- Recommendations have no explicit attempt attribution ID. Conversion is disclosed temporal matching, not causal conversion or an exclusive assignment; recent plans have incomplete observation windows.
- No live paid inference, production data writes, deployment, full screen-reader audit, human usability study or learning-effect validation was performed. There are no 3D features or new runtime dependencies.

## Architectural impact and commit suggestion

Learner SQLite, planner/rankings, journal event schema and dual-provider service remain intact. New browser-only modules perform date projection and interaction rendering. The only API validation extension accepts the existing `platform` field value `leetcode` for manual contests; no database migration is needed. local.mjs's static allowlist now includes the two new browser modules.

Suggested commit: `feat(ui): introduce calendar-first learning workspace`

Several files had explanation/daily-ingestion edits before this task; stage intended hunks only. This task does not commit or stage unrelated work.
