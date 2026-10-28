import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

function fakeElement() {
  return {
    classList: { toggle() {} },
    setAttribute() {},
    append() {},
    replaceChildren() {},
    addEventListener() {},
    click() {},
    remove() {},
  };
}

globalThis.window = { addEventListener() {} };
globalThis.document = {
  addEventListener() {},
  querySelector(selector) { return selector === "#app" || selector === "#notice" || selector === "#learner-label" ? fakeElement() : null; },
  querySelectorAll() { return []; },
  createElement() { return fakeElement(); },
  createTextNode(text) { return { textContent: text }; },
};
globalThis.localStorage = { getItem() { return null; }, setItem() {}, removeItem() {} };
globalThis.fetch = async () => { throw new Error("offline fixture"); };

const ui = await import("../public/app.mjs?ui-contract-test");

test("manual event envelope matches v1 and uses explicit UTC timestamps", () => {
  const event = ui.makeEvent({
    kind: "attempt",
    learnerId: "learner-1",
    payload: { problem_key: "leetcode:two-sum", outcome: "independent", hints: false },
    eventId: "stable-id",
    now: new Date("2026-10-16T01:02:03.000Z"),
  });
  assert.deepEqual(Object.keys(event), ["schema_version", "event_id", "learner_id", "occurred_at", "recorded_at", "kind", "source", "payload", "supersedes"]);
  assert.equal(event.schema_version, 1);
  assert.equal(event.event_id, "stable-id");
  assert.equal(event.occurred_at, "2026-10-16T01:02:03.000Z");
  assert.equal(event.recorded_at, event.occurred_at);
  assert.equal(event.source, "manual");
  assert.equal(event.supersedes, null);
});

test("problem links allow only canonical HTTPS platform URLs", () => {
  assert.equal(ui.safeProblemUrl({ canonical_url: "https://leetcode.com/problems/two-sum/" }), "https://leetcode.com/problems/two-sum/");
  for (const canonical_url of ["http://leetcode.com/problems/two-sum", "https://evil.example/problems/two-sum", "https://leetcode.com/problems/two-sum?next=x", "https://user@leetcode.com/problems/two-sum", "https://leetcode.com:8443/problems/two-sum", "https://leetcode.com/"]) {
    assert.equal(ui.safeProblemUrl({ canonical_url }), null);
  }
});

test("missing evidence and confirmed time are labeled without inventing progress", () => {
  assert.equal(ui.evidenceLabel("unknown"), "아직 확인된 기록이 없어요");
  assert.equal(ui.evidenceLabel(undefined), "기록 없음");
  assert.equal(ui.formatConfirmedMinutes(3599), "59분");
  assert.equal(ui.formatConfirmedMinutes(null), "0분");
});

test("participation records use the recovery contract and preserve stable retry IDs", () => {
  const entry = ui.makeParticipation({
    learnerId: "learner-1", contestId: "atcoder:abc300", mode: "upsolve",
    occurredAt: "2026-10-15T10:00:00.000Z", outcome: "partial", confirmedMinutes: null,
    entryId: "stable-entry", recordedAt: new Date("2026-10-16T01:02:03.000Z"),
  });
  assert.deepEqual(Object.keys(entry), ["schema_version", "entry_id", "learner_id", "contest_id", "mode", "occurred_at", "recorded_at", "outcome", "confirmed_minutes", "source", "supersedes"]);
  assert.equal(entry.entry_id, "stable-entry");
  assert.equal(entry.confirmed_minutes, null);
  assert.equal(entry.source, "manual");
  assert.equal(entry.supersedes, null);
});

test("progress helpers exclude corrected attempts and distinguish Slack from phone delivery", () => {
  const events = [
    { event_id: "old", kind: "attempt", supersedes: null },
    { event_id: "new", kind: "attempt", supersedes: "old" },
  ];
  assert.deepEqual(ui.activeEvents(events).map((event) => event.event_id), ["new"]);
  assert.equal(ui.deliveryLabel("scheduled"), "Slack 일정 등록");
  assert.equal(ui.deliveryLabel("unknown"), "Slack 결과 확인 필요");
});

test("journal filters preserve date groups and match requested facets", () => {
  const days = [
    { date: "2026-10-17", entries: [
      { type: "attempt", platform: "atcoder", pattern: "graphs", difficulty: "Medium", status: "learning", priority: "priority" },
      { type: "attempt", platform: "leetcode", pattern: "arrays", difficulty: "Easy", status: "demonstrated", priority: "normal" },
    ] },
    { date: "2026-10-16", entries: [{ type: "attempt", platform: "atcoder", pattern: "graphs", difficulty: "Easy", status: "learning", priority: "normal" }] },
  ];
  assert.deepEqual(ui.filterJournalDays(days, { from: "2026-10-17", platform: "atcoder", pattern: "graph", difficulty: "Medium", status: "learning", priority: "priority" }), [
    { date: "2026-10-17", entries: [days[0].entries[0]] },
  ]);
  assert.deepEqual(ui.filterJournalDays(days, { to: "2026-10-16", platform: "leetcode" }), []);
});

test("static shell exposes the four primary views and accessible live status", async () => {
  const html = await readFile(new URL("../public/index.html", import.meta.url), "utf8");
  for (const view of ["calendar", "history", "analytics", "contests"]) assert.ok(html.includes(`data-view="${view}"`));
  assert.match(html, /aria-live="polite"/);
  assert.match(html, /<main id="main"/);
});

test("UI source uses provider sync, full backup, and contest participation routes", async () => {
  const source = await readFile(new URL("../public/app.mjs", import.meta.url), "utf8");
  for (const route of ["/providers/sync", "/participation", "/backup/export", "/backup/import"]) assert.ok(source.includes(route), `missing ${route}`);
  assert.match(source, /delivery_reconciliation_required/);
  assert.match(source, /수정된 기록/);
});

test("calendar is the default view and retains dedicated journal read and attempt routes", async () => {
  const html = await readFile(new URL("../public/index.html", import.meta.url), "utf8");
  const source = await readFile(new URL("../public/app.mjs", import.meta.url), "utf8");
  assert.match(html, /data-view="calendar"/);
  assert.match(source, /currentView = "calendar"/);
  assert.match(source, /request\("\/journal"\)/);
  assert.match(source, /request\("\/journal\/attempt"/);
  assert.match(source, /adaptive-learning\.pending-journal-attempts\.v1/);
  assert.match(source, /같은 기록 ID로 다시 시도할 수 있어요/);
  assert.match(source, /textContent = String\(text\)/);
});


test("timer confirmation preserves seconds and milliseconds instead of rounding before the action", () => {
  const source = new Date("2026-10-16T01:02:39.123Z");
  const local = ui.toLocalInput(source.toISOString());
  assert.equal(new Date(local).getTime(), source.getTime());
});

test("calendar shows mock contests as a toggleable layer that links LeetCode and never records attempts by itself", async () => {
  const source = await readFile(new URL("../public/workspace.mjs", import.meta.url), "utf8");
  const styles = await readFile(new URL("../public/styles.css", import.meta.url), "utf8");
  assert.match(source, /layers=\{solved:true,attempted:true,planned:true,mock:true,contest:true\}/);
  assert.match(source, /\['mock','모의 대회'\]/);
  assert.match(source, /if\(e\.layer==='mock'\)return mockDetails\(e\);/);
  const detail = source.slice(source.indexOf("function mockDetails"), source.indexOf("function contestDetails"));
  assert.match(detail, /link\(p\.title \|\| p\.problem_key,p\.url\)/, "problems link through the HTTPS-only helper");
  assert.match(detail, /이 화면은 시도를 만들지 않고/);
  assert.doesNotMatch(detail, /start_at|placement_note|시작 \(KST\)/, "mock contests show a solving time, not a clock time");
  assert.match(detail, /if\(e\.archived\)/, "restored mock contests are labelled as archived history");
  assert.doesNotMatch(detail, /saveAttempt/, "opening a mock contest never saves an attempt");
  assert.match(detail, /edit\(null,localDate\(new Date\(\)\),p\.problem_key\)/, "recording goes through the normal attempt editor");
  for (const rule of [".calendar-event.mock", ".layer-toggle.mock", ".workspace-row.mock", ".mock-problem", ".activity-dot.mock"]) assert.ok(styles.includes(rule), rule);
});
