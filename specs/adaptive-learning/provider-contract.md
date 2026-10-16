# Platform provider contract

No LeetCode collector is added. Codeforces uses documented JSON APIs; AtCoder
uses a replaceable community source with explicit freshness and manual fallback.
Only canonical links and minimal metadata are persisted. Imported AC evidence is
reported completion with source submission ID; it cannot assert independent mastery.

`fetchContests(platform, {fetcher, now, previous, enabled})` returns a bounded
source/fetched_at/status/items/error envelope. Cache lifetime is 6 hours and stale
fallback is explicit. Disabled defaults never make network calls. Serialization
and at least 2 seconds between CF requests / 1 second for community requests are
implemented through a request gate. Requests have timeouts and response bounds.
Contest suitability is a user preference; default false until manually selected.

Submission adapters consume only supplied public handles. Sync functions expose
watermarks/missingness and do not claim all submissions are captured when bounded
pages truncate. AtCoder timestamps use overlap/deduplication to avoid dropping
same-second submissions. No inferred time or hints. API response errors retain
previous data and a timestamp rather than replace it with a recorded empty result.

Provider snapshots can be cached in D1; no automatic network sync is enabled by
this contract. Tests use fixed API responses and clocks, not live accounts.

`POST /api/providers/sync` is authenticated and same-origin, requires
PROVIDERS_ENABLED=true, and accepts platform and optional public handle. Calls
without a handle refresh a calendar; with a handle import reported completions.
Per-owner/source leases serialize sync. Cached calendars are always readable via
state; source metadata distinguishes never synced, stale and current empty data.
Submission sync exposes historical_gap separately from the current page limit so
a later incremental page cannot erase previously unavailable coverage. New events
are durable before watermarks advance; retry is safe through event IDs.

Codeforces problem identity is `codeforces:4A` (contest ID followed by native
index), matching the reviewed curriculum. The URL remains `/problemset/problem/4/A`.
This integration correction precedes provider activation; no recorded user keys
were rewritten.
