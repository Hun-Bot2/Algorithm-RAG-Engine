import {syncDiagnostic} from './workspace-data.mjs';
import {createWorkspace} from './workspace.mjs';
const PENDING_KEY = "adaptive-learning.pending-events.v1";
const PENDING_JOURNAL_KEY = "adaptive-learning.pending-journal-attempts.v1";
const PENDING_PARTICIPATION_KEY = "adaptive-learning.pending-participation.v1";
const API = "/api";
const $ = (selector, root = document) => root.querySelector(selector);
const app = $("#app");
const notice = $("#notice");
let state = null;
let journalState = null;
let currentView = "calendar";
if (["#similarity", "#library"].includes(window.location?.hash)) currentView = window.location.hash.slice(1);
let pendingWrites = [];
let pendingJournalWrites = [];
let pendingParticipation = null;
let providerSummary = "";

export function makeParticipation({ learnerId, contestId, mode, occurredAt, outcome, confirmedMinutes = null, entryId = crypto.randomUUID(), recordedAt = new Date() }) {
  return {
    schema_version: 1,
    entry_id: entryId,
    learner_id: learnerId,
    contest_id: contestId,
    mode,
    occurred_at: occurredAt,
    recorded_at: recordedAt.toISOString(),
    outcome,
    confirmed_minutes: confirmedMinutes,
    source: "manual",
    supersedes: null,
  };
}

export function activeEvents(events) {
  const superseded = new Set(events.map((event) => event.supersedes).filter(Boolean));
  return events.filter((event) => !superseded.has(event.event_id));
}

export function deliveryLabel(state) {
  return ({ scheduled: "Slack 일정 등록", posted: "Slack 게시 완료", unknown: "Slack 결과 확인 필요", missed: "Slack 게시 기한 지남", ready: "Slack 전송 대기", sending: "Slack 전송 중" })[state] || "Slack 기록 없음";
}

export function makeEvent({ kind, learnerId, payload, eventId = crypto.randomUUID(), now = new Date() }) {
  const timestamp = now.toISOString();
  return {
    schema_version: 1,
    event_id: eventId,
    learner_id: learnerId,
    occurred_at: timestamp,
    recorded_at: timestamp,
    kind,
    source: "manual",
    payload,
    supersedes: null,
  };
}

export function safeProblemUrl(problem) {
  if (!problem || typeof problem.canonical_url !== "string") return null;
  try {
    const url = new URL(problem.canonical_url);
    const hosts = new Set(["leetcode.com", "atcoder.jp", "codeforces.com", "www.acmicpc.net"]);
    return url.protocol === "https:" && hosts.has(url.hostname) && !url.port && !url.username && !url.password && !url.search && !url.hash && url.pathname !== "/" ? url.href : null;
  } catch { return null; }
}

export function evidenceLabel(value) {
  return ({ unknown: "아직 확인된 기록이 없어요", learning: "학습 중", demonstrated: "풀이 기록", "transfer-evidenced": "응용 풀이 기록" })[value] ?? "기록 없음";
}

export function formatConfirmedMinutes(seconds) {
  const safe = Number.isFinite(Number(seconds)) && Number(seconds) > 0 ? Math.floor(Number(seconds)) : 0;
  return `${Math.floor(safe / 60)}분`;
}

export function pendingEventIds(items) { return items.map((item) => item.event_id); }

function loadPending(learnerId) {
  try {
    const items = JSON.parse(localStorage.getItem(`${PENDING_KEY}:${encodeURIComponent(learnerId)}`) || "[]");
    return Array.isArray(items) ? items.filter((item) => item && typeof item.event_id === "string") : [];
  } catch { return []; }
}

function loadPendingJournal(learnerId) {
  try {
    const items = JSON.parse(localStorage.getItem(`${PENDING_JOURNAL_KEY}:${encodeURIComponent(learnerId)}`) || "[]");
    return Array.isArray(items) ? items.filter((item) => item?.learner_id === learnerId && typeof item.event_id === "string") : [];
  } catch { return []; }
}

function loadPendingParticipation(learnerId) {
  try {
    const item = JSON.parse(localStorage.getItem(`${PENDING_PARTICIPATION_KEY}:${encodeURIComponent(learnerId)}`) || "null");
    return item?.learner_id === learnerId && typeof item.entry_id === "string" ? item : null;
  } catch { return null; }
}

function savePending() {
  try { localStorage.setItem(`${PENDING_KEY}:${encodeURIComponent(state.learner_id)}`, JSON.stringify(pendingWrites)); }
  catch { showNotice("이 브라우저에 재시도 기록을 저장하지 못했어요. 연결이 복구될 때까지 이 화면을 닫지 말아 주세요.", true); }
}

function savePendingJournal() {
  try { localStorage.setItem(`${PENDING_JOURNAL_KEY}:${encodeURIComponent(state.learner_id)}`, JSON.stringify(pendingJournalWrites)); }
  catch { showNotice("풀이 기록의 재시도 정보를 저장하지 못했어요. 연결이 복구될 때까지 이 화면을 닫지 말아 주세요.", true); }
}

function savePendingParticipation() {
  try {
    const key = `${PENDING_PARTICIPATION_KEY}:${encodeURIComponent(state.learner_id)}`;
    if (pendingParticipation) localStorage.setItem(key, JSON.stringify(pendingParticipation));
    else localStorage.removeItem(key);
  } catch { showNotice("대회 기록의 재시도 정보를 저장하지 못했어요. 연결이 복구될 때까지 이 화면을 닫지 말아 주세요.", true); }
}

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, { credentials: "same-origin", ...options });
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) { const error = new Error(body?.error || `요청에 실패했어요 (${response.status}).`); error.status = response.status; throw error; }
  return body;
}

function showNotice(message, isError = false) {
  notice.textContent = message;
  notice.classList.toggle("error", isError);
  notice.hidden = false;
}

function clearNotice() { notice.hidden = true; notice.textContent = ""; }

async function refresh() {
  app.setAttribute("aria-busy", "true");
  try {
    const nextState = await request("/state");
    if (state?.learner_id !== nextState.learner_id) {
      pendingWrites = loadPending(nextState.learner_id).filter(item => item.learner_id === nextState.learner_id);
      pendingJournalWrites = loadPendingJournal(nextState.learner_id);
      pendingParticipation = loadPendingParticipation(nextState.learner_id);
    }
    state = nextState;
    try { journalState = await request("/journal"); }
    catch (error) { journalState = { available: false, reason: error.message }; }
    $("#learner-label").textContent = state.learner_id ? "내 학습 기록" : "학습 기록";
    clearNotice();
    render();
    return true;
  } catch (error) {
    if (error.status === 401 || error.status === 403) { state = null; pendingWrites = []; pendingJournalWrites = []; pendingParticipation = null; }
    app.replaceChildren();
    app.append(node("h1", "page-heading", "학습 정보를 불러오지 못했어요"), node("p", "intro", error.message));
    app.append(button("다시 불러오기", refresh));
    renderPending();
    showNotice("연결이 복구되면 다시 시도해 주세요.", true);
    return false;
  } finally { app.setAttribute("aria-busy", "false"); }
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}

function button(label, action, className = "button") {
  const element = node("button", className, label);
  element.type = "button";
  element.addEventListener("click", action);
  return element;
}

function field(labelText, control, full = false) {
  const wrap = node("div", `field${full ? " full" : ""}`);
  const label = node("label", "", labelText);
  label.htmlFor = control.id;
  wrap.append(label, control);
  return wrap;
}

function input(id, type = "text", value = "") {
  const element = document.createElement("input");
  element.id = id;
  element.name = id;
  element.type = type;
  if (value !== undefined && value !== null) element.value = value;
  return element;
}

function select(id, options, value) {
  const element = document.createElement("select");
  element.id = id;
  element.name = id;
  for (const [optionValue, label] of options) {
    const option = node("option", "", label);
    option.value = optionValue;
    element.append(option);
  }
  if (value !== undefined) element.value = value;
  return element;
}

function section(title, description) {
  const heading = node("h1", "page-heading", title);
  const intro = node("p", "intro", description);
  return [heading, intro];
}

function problemMap() {
  const items = state?.curriculum?.problems || [];
  return new Map(items.filter((item) => item?.problem_key).map((item) => [item.problem_key, item]));
}

function problemTitle(key) {
  const annotation = state?.annotations?.[key];
  return annotation?.title || annotation?.name || problemMap().get(key)?.title || key || "문제 정보 없음";
}

function accessLabel(value) {
  return ({ free: "무료 접근", paid: "유료 접근", unknown: "접근 정보 미확인", unavailable: "현재 이용 불가" })[value] || "접근 정보 미확인";
}

function problemLink(key) {
  const problem = problemMap().get(key);
  const href = safeProblemUrl(problem);
  if (!href) return node("span", "", problemTitle(key));
  const link = node("a", "", problemTitle(key));
  link.href = href;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.setAttribute("aria-label", `${problemTitle(key)} 문제 페이지, 새 탭에서 열기`);
  return link;
}

function render() {
  for (const tab of document.querySelectorAll("[data-view]")) {
    tab.setAttribute("aria-current", tab.dataset.view === currentView ? "page" : "false");
  }
  app.replaceChildren();
  if (["calendar", "history", "analytics", "contests", "similarity", "library"].includes(currentView)) workspace.render(currentView);
  else if (currentView === "journal") renderJournal();
  else if (currentView === "progress") renderProgress();
  else if (currentView === "contests") renderContests();
  else renderToday();
  renderPending();
}

export function filterJournalDays(days, filters = {}) {
  const from = filters.from || "";
  const to = filters.to || "";
  const platform = (filters.platform || "").toLowerCase();
  const pattern = (filters.pattern || "").toLowerCase();
  const difficulty = (filters.difficulty || "").toLowerCase();
  const status = (filters.status || "").toLowerCase();
  const priority = (filters.priority || "").toLowerCase();
  return (Array.isArray(days) ? days : []).filter((day) => {
    if (from && day.date < from) return false;
    if (to && day.date > to) return false;
    const entries = Array.isArray(day.entries) ? day.entries : [];
    return entries.some((entry) => {
      const values = [entry.platform, entry.pattern, entry.difficulty, entry.status, entry.priority, entry.review_priority]
        .map((value) => String(value ?? "").toLowerCase());
      return (!platform || values[0] === platform)
        && (!pattern || values[1].includes(pattern))
        && (!difficulty || values[2].includes(difficulty))
        && (!status || values[3].includes(status))
        && (!priority || values[4] === priority || values[5] === priority);
    });
  }).map((day) => ({ ...day, entries: (Array.isArray(day.entries) ? day.entries : []).filter((entry) => {
    const values = [entry.platform, entry.pattern, entry.difficulty, entry.status, entry.priority, entry.review_priority]
      .map((value) => String(value ?? "").toLowerCase());
    return (!platform || values[0] === platform)
      && (!pattern || values[1].includes(pattern))
      && (!difficulty || values[2].includes(difficulty))
      && (!status || values[3].includes(status))
      && (!priority || values[4] === priority || values[5] === priority);
  }) }));
}

function journalFilters() {
  const card = node("section", "card journal-filters");
  const details = document.createElement("details");
  details.className = "journal-filter-details";
  const summary = document.createElement("summary");
  summary.append(node("strong", "", "기록 필터"), node("span", "meta", "날짜 · 플랫폼 · 패턴 · 난이도 · 상태 · 복습 우선순위"));
  details.append(summary);
  const form = document.createElement("form");
  form.className = "form-grid";
  const values = journalState?.days?.flatMap((day) => day.entries || []) || [];
  const choices = (key) => [...new Set(values.map((item) => item[key]).filter((value) => value !== null && value !== undefined && value !== "").map(String))].sort();
  const from = input("journal-from", "date");
  const to = input("journal-to", "date");
  const platform = select("journal-platform", [["", "모든 플랫폼"], ...choices("platform").map((value) => [value, value])]);
  const pattern = select("journal-pattern", [["", "모든 패턴"], ...choices("pattern").map((value) => [value, value])]);
  const difficulty = select("journal-difficulty", [["", "모든 난이도"], ...choices("difficulty").map((value) => [value, value])]);
  const status = select("journal-status", [["", "모든 상태"], ...choices("status").map((value) => [value, value])]);
  const priority = select("journal-priority", [["", "모든 우선순위"], ["priority", "우선 복습"], ["normal", "일반 복습"]]);
  form.append(field("시작 날짜", from), field("종료 날짜", to), field("플랫폼", platform), field("패턴", pattern), field("난이도", difficulty), field("상태", status), field("복습 우선순위", priority));
  const results = node("div", "journal-results");
  const renderFiltered = () => {
    const days = filterJournalDays(journalState?.days, { from: from.value, to: to.value, platform: platform.value, pattern: pattern.value, difficulty: difficulty.value, status: status.value, priority: priority.value });
    results.replaceChildren();
    if (!days.length) results.append(node("p", "empty", "조건에 맞는 학습 기록이 없어요."));
    else for (const day of days) results.append(renderJournalDay(day));
  };
  form.addEventListener("change", renderFiltered);
  form.addEventListener("input", renderFiltered);
  details.append(form);
  card.append(details, results);
  renderFiltered();
  return card;
}

function renderJournal() {
  const date = journalState?.date || "";
  app.append(...section("학습 일지", "오늘의 계획과 풀이 기록이 날짜별로 쌓여요."));
  if (!journalState?.available) {
    const empty = node("section", "card");
    empty.append(node("h2", "", "일지를 불러올 수 없어요"), node("p", "empty", journalState?.reason || "학습 일지 정보를 사용할 수 없어요."));
    app.append(empty, todayShortcut());
    return;
  }
  app.append(todayPlanCard(), explanationUsageCard(), focusCard(), journalFilters(), reviewsCard(), patternHistoryCard(), weeklyCard(), upcomingCard());
  const limitations = Array.isArray(journalState.limitations) ? journalState.limitations.filter(Boolean) : [];
  if (limitations.length) {
    const note = node("section", "card journal-limitations");
    note.append(node("h2", "", "기록 범위 안내"));
    const list = node("ul", "event-list");
    for (const limitation of limitations) list.append(node("li", "event-row", typeof limitation === "string" ? limitation : limitation.message || limitation.reason || "확인할 수 없는 기록 범위가 있어요."));
    note.append(list);
    app.append(note);
  }
  app.append(todayShortcut());
  if (date) app.setAttribute("data-journal-date", date);
}

function explanationUsageCard() {
  const card = node("details", "card explanation-usage");
  card.append(node("summary", "", "AI 설명 사용량"));
  card.append(node("p", "meta", "실제 API 시도와 재시도만 집계해요. 저장된 결과 조회는 제외하며, 비용 추정은 청구 금액과 달라요. 확인 불가는 0이 아니에요."));
  const groups = journalState?.explanation_usage?.groups || [];
  if (!groups.length) { card.append(node("p", "empty", "기록된 API 호출이 없어요.")); return card; }
  const wrap = node("div", "usage-table-wrap"), table = node("table", "usage-table");
  const head = node("tr", "");
  for (const label of ["공급자 / 모델", "요청 / 재시도", "성공 / 실패 / 결과 미확인", "입력 / 출력 토큰", "평균 응답 시간", "추정 비용 / 실제 비용 (USD)"]) head.append(node("th", "", label));
  const thead = node("thead", ""); thead.append(head); table.append(thead);
  const tbody = node("tbody", "");
  const covered = (value, known, total) => value === null ? "확인 불가" : `${value} (${known}/${total}건 확인)`;
  for (const g of groups) {
    const row = node("tr", "");
    const cells = [`${g.provider} / ${g.model}`, `${g.requests} / ${g.retries}`, `${g.successes} / ${g.failures} / ${g.unresolved}`,
      `${covered(g.input_tokens, g.input_usage_known, g.requests)} / ${covered(g.output_tokens, g.output_usage_known, g.requests)}`,
      covered(g.latency_ms_mean === null ? null : `${Math.round(g.latency_ms_mean)}ms`, g.latency_known, g.requests),
      `${covered(g.estimated_cost_usd === null ? null : g.estimated_cost_usd.toFixed(6), g.estimated_cost_known, g.requests)} / ${covered(g.actual_cost_usd === null ? null : g.actual_cost_usd.toFixed(6), g.actual_cost_known, g.requests)}`];
    for (const cell of cells) row.append(node("td", "", cell));
    tbody.append(row);
  }
  table.append(tbody); wrap.append(table); card.append(wrap); return card;
}

function todayShortcut() {
  const card = node("section", "card");
  card.append(node("h2", "", "학습 기록 도구"), node("p", "meta", "목표 설정, 타이머, 백업과 복원은 오늘 화면에서 계속 사용할 수 있어요."));
  card.append(button("오늘 화면 열기", () => { currentView = "today"; render(); }, "button secondary"));
  return card;
}

function todayPlanCard() {
  const plan = journalState?.plan;
  const card = node("section", "card journal-plan");
  card.append(node("div", "card-head"));
  const head = card.firstChild;
  const title = node("h2", "", "오늘의 학습 계획");
  head.append(title, node("span", "tag", plan?.date || journalState?.date || "오늘"));
  if (!plan || !Array.isArray(plan.slots) || !plan.slots.length) {
    card.append(node("p", "empty", "오늘 제안된 활동이 없어요. 기록된 학습 일지는 아래에서 계속 확인할 수 있어요."));
    return card;
  }
  const list = node("ol", "journal-plan-list");
  for (const slot of plan.slots) {
    const item = node("li", "journal-plan-item");
    item.dataset.problemKey = slot.problem_key || "";
    item.dataset.slotKind = slot.kind || "";
    const heading = node("div", "split");
    const name = slot.title || slot.problem_key || "학습 활동";
    const link = safeProblemUrl({ canonical_url: slot.url });
    if (link) {
      const anchor = node("a", "", name); anchor.href = link; anchor.target = "_blank"; anchor.rel = "noopener noreferrer";
      heading.append(anchor);
    } else heading.append(node("strong", "", name));
    heading.append(node("span", "tag", `${slot.platform || "플랫폼 미상"}${slot.difficulty ? ` · ${slot.difficulty}` : ""}`));
    if (slot.semantic_only) heading.append(node("span", "tag warning semantic-only", "의미 유사 후보 · 알고리즘 동등성 미검증"));
    item.append(heading);
    if (slot.concepts?.length) item.append(node("p", "meta", `개념 · ${slot.concepts.join(", ")}`));
    if (slot.reason) {
      const rationale = document.createElement("details");
      rationale.className = "journal-plan-reason";
      const summary = node("summary", "meta", "추천 이유 보기");
      rationale.append(summary);
      if (slot.explanation?.results) {
        const comparison = node("div", "explanation-comparison");
        const statusLabels = {success:"생성 성공",failed:"생성 실패",not_configured:"설정 필요",not_generated:"아직 생성되지 않음",in_progress:"생성 중",unknown:"결과 확인 불가"};
        for (const result of slot.explanation.results) {
          const column = node("section", "explanation-provider");
          column.dataset.provider = result.provider;
          column.append(node("h4", "", result.provider === "gemini" ? "Gemini" : "NVIDIA NIM"));
          column.append(node("p", "meta", result.model || "모델 미설정"));
          column.append(node("span", "tag", statusLabels[result.status] || result.status));
          if (result.status === "success") {
            column.append(node("p", "reason", result.text));
            column.append(node("p", "meta", `AI 작성 · 사실성 검증 전${result.cache_hit ? " · 저장된 결과" : ""}`));
          } else column.append(node("p", "meta", result.error ? `오류: ${result.error.code}${result.error.http_status ? ` (${result.error.http_status})` : ""}` : "이 모델의 설명을 사용할 수 없어요."));
          if (result.error?.provider_category && result.error.provider_category !== "unspecified") column.append(node("p", "meta", `응답 분류: ${result.error.provider_category}`));
          if (result.error?.hint) column.append(node("p", "meta", result.error.hint));
          comparison.append(column);
        }
        rationale.append(comparison);
        if (slot.explanation.fallback) {
          rationale.append(node("p", "meta", "기본 설명 · 사용 가능한 AI 설명 없음"), node("p", "reason", slot.explanation.fallback.text));
        }
      } else rationale.append(node("p", "reason", slot.reason));
      const evidence = document.createElement("details");
      evidence.append(node("summary", "meta", "원래 추천 근거 보기"), node("p", "reason", slot.reason));
      if (slot.explanation?.evidence) evidence.append(node("pre", "explanation-evidence", JSON.stringify(slot.explanation.evidence, null, 2)));
      rationale.append(evidence);
      item.append(rationale);
    } else item.append(node("p", "reason", "제안 이유가 제공되지 않았어요."));
    if (slot.estimated_minutes !== null && slot.estimated_minutes !== undefined) item.append(node("p", "meta", `예상 ${slot.estimated_minutes}분${slot.review_priority ? ` · ${slot.review_priority === "priority" ? "우선 복습" : "일반 복습"}` : ""}`));
    if (slot.problem_key) item.append(journalAttemptForm(slot));
    list.append(item);
  }
  card.append(list);
  return card;
}

function journalAttemptForm(slot) {
  const details = document.createElement("details");
  details.className = "journal-attempt-details";
  details.dataset.problemKey = slot.problem_key;
  details.dataset.slotKind = slot.kind || "";
  const summary = node("summary", "button secondary", "풀이 결과 기록");
  details.append(summary);
  const form = document.createElement("form");
  form.className = "form-grid";
  const outcome = select(`journal-outcome-${slot.problem_key}`, [["independent", "O · 해결"], ["failed", "X · 아직 해결하지 못함"]]);
  const minutes = input(`journal-minutes-${slot.problem_key}`, "number"); minutes.min = "0"; minutes.step = "1"; minutes.placeholder = "선택";
  form.append(field("결과", outcome), field("직접 사용한 시간 (분)", minutes));
  const actions = node("div", "actions full"); const submit = node("button", "button", "풀이 기록 저장"); submit.type = "submit"; actions.append(submit); form.append(actions);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const selectedOutcome = outcome.value;
    const selfMinutes = minutes.value === "" ? null : Number(minutes.value);
    if (selfMinutes !== null && (!Number.isInteger(selfMinutes) || selfMinutes < 0 || selfMinutes > 1440)) return showNotice("직접 사용한 시간은 0분에서 1440분 사이로 입력해 주세요.", true);
    const activityType = slot.transfer_from ? "transfer" : slot.kind === "review" ? "review" : "practice";
    const payload = { event_id: crypto.randomUUID(), learner_id: state?.learner_id, problem_key: slot.problem_key, outcome: selectedOutcome, hints: selectedOutcome === "independent" ? false : null, activity_type: activityType, self_minutes: selfMinutes };
    if (slot.transfer_from) payload.transfer_from = slot.transfer_from;
    if (!payload.learner_id) return showNotice("학습자 정보를 확인할 수 없어 저장하지 못했어요.", true);
    pendingJournalWrites.push(payload);
    savePendingJournal();
    await retryJournalAttempt(payload.event_id);
  });
  details.append(form);
  return details;
}

function focusCard() {
  const focus = journalState?.focus;
  const card = node("section", "card");
  card.append(node("h2", "", "이번 주 학습 초점"));
  if (!focus) card.append(node("p", "empty", "아직 확인할 초점이 없어요."));
  else {
    card.append(node("div", "split"));
    card.lastChild.append(node("strong", "", focus.pattern || "학습 패턴 미확인"), node("span", "tag", progressionStatusLabel(focus.status)));
    card.append(node("p", "meta", progressionStageLabel(focus.stage)));
    if (focus.reason) {
      const rationale = document.createElement("details");
      rationale.className = "journal-plan-reason";
      rationale.append(node("summary", "meta", "초점 선정 이유 보기"), node("p", "reason", focus.reason));
      card.append(rationale);
    }
  }
  return card;
}

function reviewsCard() {
  const reviews = journalState?.reviews || {};
  const card = node("section", "card"); card.append(node("h2", "", "복습 목록"));
  for (const [key, label] of [["priority", "우선 복습"], ["normal", "일반 복습"]]) {
    const sectionNode = node("section", "review-group"); sectionNode.append(node("h3", "", label));
    const items = Array.isArray(reviews[key]) ? reviews[key] : [];
    if (!items.length) sectionNode.append(node("p", "empty", "예정된 문제가 없어요."));
    else {
      const list = node("ul", "problem-list");
      for (const review of items) {
        const row = node("li", "problem-row");
        row.append(node("strong", "", review.title || review.problem_key || "문제 정보 없음"));
        row.append(node("p", "meta", [review.platform, journalDifficulty(review.platform, review.difficulty), review.due_date, review.reason].filter(Boolean).join(" · ")));
        list.append(row);
      }
      sectionNode.append(list);
    }
    card.append(sectionNode);
  }
  return card;
}

function patternHistoryCard() {
  const card = node("section", "card journal-pattern-history");
  const details = document.createElement("details");
  details.className = "pattern-history-details";
  const summary = document.createElement("summary");
  const patterns = Array.isArray(journalState?.patterns) ? journalState.patterns : [];
  summary.append(node("strong", "", "개념 상태와 변경 기록"), node("span", "meta", `${patterns.length}개 개념`));
  details.append(summary);
  const content = node("div", "pattern-history-content");
  if (!patterns.length) content.append(node("p", "empty", "아직 개념 상태 기록이 없어요."));
  else {
    const list = node("ul", "event-list");
    for (const pattern of patterns) {
      const row = node("li", "event-row");
      row.append(node("div", "split")); row.firstChild.append(node("strong", "", pattern.pattern || "개념 미상"), node("span", "tag", progressionStatusLabel(pattern.status)));
      row.append(node("p", "meta", [progressionStageLabel(pattern.stage), pattern.changed_at, pattern.reason].filter(Boolean).join(" · ")));
      list.append(row);
    }
    content.append(list);
  }
  const changes = (journalState?.days || []).flatMap((day) => (day.entries || []).filter((entry) => entry.type === "mastery_change").map((entry) => ({ ...entry, date: day.date })));
  if (changes.length) {
    content.append(node("h3", "", "변경 내역"));
    const list = node("ul", "event-list");
    for (const entry of changes) list.append(node("li", "event-row", `${entry.date} · ${entry.pattern || entry.title || "개념 상태"} · ${progressionStatusLabel(entry.status)}${entry.reason ? ` · ${entry.reason}` : ""}`));
    content.append(list);
  }
  details.append(content);
  card.append(details);
  return card;
}

function weeklyCard() {
  const weekly = journalState?.weekly;
  const card = node("section", "card"); card.append(node("h2", "", "이번 주"));
  if (!weekly || !Object.keys(weekly).length) card.append(node("p", "empty", "주간 학습 기록이 아직 없어요."));
  else {
    const activity = weekly.activity || {};
    const review = weekly.review || {};
    const process = weekly.process || {};
    const onTime = review.on_time;
    const summary = [
      activity.total !== undefined ? `이번 주 풀이 ${activity.total}회` : null,
      process.done_slots !== undefined && process.planned_slots !== undefined ? `계획 ${process.done_slots}/${process.planned_slots}개 완료` : null,
      process.confirmed_timer_minutes !== undefined ? `확인 시간 ${process.confirmed_timer_minutes}분` : null,
      process.self_reported_seconds_total !== undefined ? `직접 입력 시간 ${process.self_reported_seconds_total}s` : process.self_reported_minutes !== undefined ? `직접 입력 시간 ${process.self_reported_minutes}분` : null,
      review.lapse_rate !== undefined && review.lapse_rate !== null ? `복습 실패율 ${Math.round(review.lapse_rate * 100)}% (${review.n ?? 0}회)` : null,
      onTime ? `정시 복습 ${onTime.successes ?? 0}/${onTime.n ?? 0} (${onTime.status || "기록 요약"})` : null,
    ].filter(Boolean);
    if (summary.length) card.append(node("p", "meta", summary.join(" · ")));
    const list = node("ul", "event-list");
    for (const group of ["by_platform", "by_outcome", "by_source", "by_difficulty"]) {
      for (const [key, value] of Object.entries(activity[group] || {}).sort(([a], [b]) => a.localeCompare(b))) list.append(node("li", "event-row", `${group.replace("by_", "")} · ${key} · ${value}`));
    }
    if (list.children.length) card.append(list);
    if (process.time_note) card.append(node("p", "meta", process.time_note));
    if (weekly.disclaimer) card.append(node("p", "meta", weekly.disclaimer));
    if (!summary.length && !list.children.length) card.append(node("p", "empty", "주간 기록을 확인할 수 없어요."));
  }
  return card;
}

function upcomingCard() {
  const card = node("section", "card"); card.append(node("h2", "", "다가오는 학습"));
  const upcoming = Array.isArray(journalState?.upcoming) ? journalState.upcoming : [];
  if (!upcoming.length) card.append(node("p", "empty", "가까운 날짜에 예정된 추천이 없어요."));
  else {
    const list = node("ul", "problem-list");
    for (const slot of upcoming) {
      const row = node("li", "problem-row");
      row.append(node("div", "split")); row.firstChild.append(node("strong", "", slot.title || slot.problem_key || "학습 활동"), node("span", "tag", slot.date || slot.platform || "예정"));
      row.append(node("p", "meta", [slot.platform, slot.difficulty, slot.reason].filter(Boolean).join(" · ")));
      if (slot.semantic_only) row.append(node("p", "meta semantic-only", "의미 유사 후보 · 알고리즘 동등성 미검증"));
      list.append(row);
    }
    card.append(list);
  }
  return card;
}

function renderJournalDay(day) {
  const card = node("section", "journal-day");
  const heading = node("div", "split"); heading.append(node("h3", "", formatJournalDate(day.date)), node("span", "tag", `${day.entries.length}개 기록`));
  card.append(heading);
  const list = node("ul", "event-list");
  for (const entry of day.entries) {
    const row = node("li", `event-row journal-entry ${entry.type || ""}`);
    const label = entry.status === "superseded" ? "수정" : entry.display_outcome === "O" || entry.display_outcome === "X" ? entry.display_outcome : entry.type === "recommendation" ? "계획" : entry.type === "mastery_change" ? "개념" : "기록";
    row.append(node("span", `journal-mark ${label === "O" ? "success" : label === "X" ? "failure" : entry.status === "superseded" ? "corrected" : ""}`, label));
    const content = node("div", "journal-entry-content");
    content.append(node("strong", "", entry.title || entry.problem_key || entry.pattern || "학습 기록"));
    const detail = [entry.platform, entry.difficulty, entry.pattern, entry.status ? progressionStatusLabel(entry.status) : null, entry.priority === "priority" || entry.review_priority === "priority" ? "우선 복습" : entry.priority === "normal" || entry.review_priority === "normal" ? "일반 복습" : null].filter(Boolean).join(" · ");
    if (detail) content.append(node("p", "meta", detail));
    if (entry.status === "superseded") content.append(node("p", "meta corrected-note", `수정된 기록 · 현재 학습 근거에서 제외${entry.superseded_by ? ` · 새 기록 ${entry.superseded_by}` : ""}`));
    if (entry.type === "attempt") {
      const timeBasis = ({ not_recorded: "시간 미기록", self_reported: "직접 입력한 시간", confirmed_intervals_same_day_not_attributed_to_attempt: "같은 날 확인된 타이머 (이 풀이에 직접 연결되지 않음)" })[entry.time_basis] || null;
      const attemptDetails = [entry.outcome ? outcomeLabel(entry.outcome) : null, ({practice:'연습',review:'알고 있는 문제 복습 · 새 문제 전이 증거 아님',transfer:'타 플랫폼 전이 검증'})[entry.activity_type] || null,
        entry.self_seconds != null ? `직접 기록 ${entry.self_seconds}s` : entry.self_minutes != null ? `직접 기록 ${entry.self_minutes}m` : null,
        Number.isFinite(Number(entry.confirmed_seconds)) && Number(entry.confirmed_seconds) > 0 ? `확인 시간 ${formatConfirmedMinutes(entry.confirmed_seconds)}` : null,
        timeBasis].filter(Boolean).join(" · ");
      if (attemptDetails) content.append(node("p", "meta", attemptDetails));
    }
    if (entry.reason) content.append(node("p", "reason", entry.reason));
    if (entry.semantic_only) content.append(node("p", "meta semantic-only", "의미 유사 후보 · 알고리즘 동등성 미검증"));
    if (entry.occurred_at) content.append(node("time", "meta", formatJournalTime(entry.occurred_at)));
    row.append(content); list.append(row);
  }
  card.append(list);
  return card;
}

function formatJournalDate(value) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(String(value || ""))) return String(value || "날짜 미상");
  return new Intl.DateTimeFormat("ko-KR", { dateStyle: "full", timeZone: "UTC" }).format(new Date(`${value}T12:00:00Z`));
}

function formatJournalTime(value) {
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? new Intl.DateTimeFormat("ko-KR", { dateStyle: "short", timeStyle: "short", timeZone: "Asia/Seoul" }).format(date) : "시각 미상";
}

function journalDifficulty(platform, value) {
  if (platform === "leetcode") return ({ 1: "Easy", 2: "Medium", 3: "Hard" })[value] || value;
  return value;
}

function progressionStageLabel(value) {
  return ({ step_down: "기초로 돌아가기", consolidate: "현재 수준 다지기", step_up: "다음 수준 도전", stretch: "상위 수준 탐색", transfer: "다른 플랫폼 전이", leetcode_easy: "Easy 기초", leetcode_medium: "Medium 적용", cross_platform_transfer: "다른 플랫폼 전이", retention_review: "간격 복습" })[value] || value || "단계 미확인";
}

function progressionStatusLabel(value) {
  return ({ in_progress: "진행 중", blocked_prerequisite: "선행 조건 대기", retention_evidenced: "복습 근거 확인", remediation: "보완 필요", current: "현재 단계", complete: "완료", locked: "대기", superseded: "수정된 기록" })[value] || value || "상태 미확인";
}

function renderToday() {
  app.append(...section("오늘의 학습", "제안된 활동과 직접 남긴 기록을 확인해요."));
  const mission = state?.mission;
  const activities = Array.isArray(mission?.activities) ? mission.activities : [];
  const missionCard = node("section", "card");
  const missionHeading = node("h2", "", "오늘의 제안");
  missionHeading.id = "mission-title";
  missionCard.setAttribute("aria-labelledby", "mission-title");
  const missionHead = node("div", "card-head");
  missionHead.append(missionHeading);
  if (mission?.snapshot_status === "preview") missionHead.append(node("span", "tag warning", "미리보기 · 변경될 수 있음"));
  else if (mission?.snapshot_status === "frozen") missionHead.append(node("span", "tag", "오늘 확정"));
  missionCard.append(missionHead);
  if (activities.length) {
    const list = node("ul", "problem-list");
    for (const activity of activities) {
      const row = node("li", "problem-row");
      const top = node("div", "card-head");
      top.append(node("h3", "", ""));
      if (activity.problem_key) top.firstChild.append(problemLink(activity.problem_key));
      else {
        const title=node("span", "", activity.title || "연습할 대회 선택");
        if (activity.url && safeExternalUrl(activity.url)) {
          const link=node("a", "", title.textContent); link.href=activity.url;link.target="_blank";link.rel="noopener noreferrer";top.firstChild.append(link);
        } else top.firstChild.append(title);
      }
      top.append(node("span", "tag", ({practice:"연습",review:"복습",transfer:"응용",contest:"대회",virtual:"버추얼",upsolve:"업솔빙"})[activity.type] || "학습"));
      row.append(top);
      const problem = problemMap().get(activity.problem_key);
      const detail = problem ? [problem?.platform || (activity.problem_key || "").split(":")[0], problem?.native_difficulty, accessLabel(problem?.access_state)].filter(Boolean).join(" · ") : "";
      if (detail) row.append(node("p", "meta", detail));
      if (problem?.review?.status) row.append(node("p", "meta", problem.review.status === "approved" ? "문제 정보 검토 완료" : "문제 정보 검토 중"));
      if (activity.estimated_minutes !== null && activity.estimated_minutes !== undefined && Number.isFinite(Number(activity.estimated_minutes))) row.append(node("p", "meta", `예상 ${Number(activity.estimated_minutes)}분`));
      row.append(node("p", "reason", activity.reason || "제안 이유가 제공되지 않았어요."));
      const timer = node("div", "actions");
      if(activity.problem_key) timer.append(button("이 문제 시간 기록 시작", () => startTimer(activity.problem_key), "button secondary"));
      else timer.append(button("대회 일정과 참여 기록 열기",()=>{currentView="contests";render();},"button secondary"));
      row.append(timer);
      list.append(row);
    }
    missionCard.append(list);
  } else {
    missionCard.append(node("p", "empty", mission ? "오늘 제안된 활동이 없어요." : "오늘의 학습 제안이 아직 준비되지 않았어요. 직접 문제를 선택해 기록할 수 있어요."));
  }
  app.append(missionCard);
  app.append(deliveryCard());
  app.append(attemptForm());
  app.append(timerCard());
  app.append(preferencesCard());
  app.append(dataCard());
}

function deliveryCard() {
  const card = node("section", "card");
  card.append(node("h2", "", "알림 전달 상태"), node("p", "meta", "Slack에 등록된 학습 알림 상태예요. 휴대전화 알림 수신 여부는 이 기록으로 확인할 수 없어요."));
  if (state?.delivery_reconciliation_required) card.append(node("p", "empty", "복원된 기록의 Slack 전달 상태를 확인하는 중이에요. 확인이 끝날 때까지 자동 전달이 멈춰 있어요."));
  const deliveries = Array.isArray(state?.delivery) ? state.delivery : [];
  if (!deliveries.length) card.append(node("p", "empty", state?.capabilities?.delivery_enabled ? "아직 확인할 Slack 전달 기록이 없어요." : "현재 Slack 학습 알림을 사용할 수 없어요."));
  else {
    const list = node("ul", "event-list");
    for (const delivery of deliveries.slice(0, 5)) {
      const item = node("li", "event-row");
      const detail = delivery.last_error_code ? " · 전달 상태를 추가로 확인해야 해요" : "";
      item.append(node("strong", "", deliveryLabel(delivery.state)), node("p", "meta", `${delivery.local_date || "날짜 미상"}${delivery.retry_at ? ` · 재시도 ${delivery.retry_at}` : ""}${detail}`));
      list.append(item);
    }
    card.append(list);
  }
  return card;
}

function attemptForm() {
  const card = node("section", "card");
  card.append(node("h2", "", "풀이 기록"), node("p", "meta", "결과를 직접 선택해 주세요. 문제 제출이나 방문만으로는 기록되지 않아요."));
  const form = document.createElement("form");
  form.id = "attempt-form";
  const grid = node("div", "form-grid");
  const problems = [...problemMap().values()];
  const keys = [...new Set([...problems.map((problem) => problem.problem_key), ...(state?.mission?.activities || []).map((item) => item.problem_key)].filter(Boolean))];
  const problem = select("attempt-problem", keys.map((key) => [key, problemTitle(key)]));
  if (!keys.length) problem.append(new Option("문제 목록을 불러오지 못했어요", ""));
  grid.append(field("문제", problem, true));
  grid.append(field("결과", select("attempt-outcome", [["independent", "해결"], ["failed", "아직 해결하지 못함"]])));
  const transferSources = (state?.mission?.activities || []).filter((item) => item.type === "transfer" && item.transfer_from).map((item) => item.transfer_from);
  const activityOptions = [["", "선택 안 함"], ["practice", "연습"], ["review", "복습"]];
  if (transferSources.length) activityOptions.push(["transfer", "응용"]);
  grid.append(field("활동 종류", select("attempt-type", activityOptions)));
  form.append(grid);
  const actions = node("div", "actions");
  const submit = node("button", "button", "풀이 결과 저장");
  submit.type = "submit";
  actions.append(submit);
  form.append(actions);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!problem.value) return showNotice("먼저 기록할 문제를 선택해 주세요.", true);
    const outcome = $("#attempt-outcome", form).value;
    const payload = { problem_key: problem.value, outcome, hints: outcome === "independent" ? false : null };
    const activityType = $("#attempt-type", form).value;
    if (activityType) payload.activity_type = activityType;
    if (activityType === "transfer") {
      const source = (state?.mission?.activities || []).find((item) => item.type === "transfer" && item.transfer_from && item.problem_key === problem.value);
      if (!source) return showNotice("응용 문제와 연결된 기존 문제 근거가 없어 기록할 수 없어요.", true);
      payload.transfer_from = source.transfer_from;
    }
    await queueEvent("attempt", payload);
  });
  card.append(form);
  return card;
}

function timerCard() {
  const card = node("section", "card");
  card.append(node("h2", "", "시간 기록"), node("p", "meta", "시작과 종료를 직접 기록해요. 확인한 구간만 학습 시간에 포함돼요."));
  const timers = state?.timers?.intervals || [];
  if (!timers.length) card.append(node("p", "empty", "열려 있거나 확인을 기다리는 시간 기록이 없어요."));
  for (const timer of timers) {
    const block = node("div", "timer-block");
    const status = ({ open: "진행 중 · 다시 열림", unconfirmed: "종료됨 · 시간 확인 필요", stopped: "종료됨 · 시간 확인 필요", confirmed: "확인 완료" })[timer.status] || "상태 확인 필요";
    block.append(node("p", "", problemTitle(timer.problem_key)), node("p", "meta", `${status} · ${timer.started_at || "시작 시간 정보 없음"}`));
    if (timer.status === "open") block.append(button("중단 기록", () => stopTimer(timer.interval_id), "button secondary"));
    if (["unconfirmed", "stopped"].includes(timer.status)) block.append(confirmTimerForm(timer));
    card.append(block);
  }
  return card;
}

function confirmTimerForm(timer) {
  const form = document.createElement("form");
  form.className = "timer-confirm form-grid";
  const start = input(`timer-start-${timer.interval_id}`, "datetime-local", toLocalInput(timer.started_at));
  const end = input(`timer-end-${timer.interval_id}`, "datetime-local", "");
  start.required = true;
  end.required = true;
  start.step = "0.001";
  end.step = "0.001";
  form.append(field("실제로 공부를 시작한 시각", start), field("실제로 공부를 마친 시각", end));
  const action = node("div", "actions full");
  const submit = node("button", "button secondary", "이 구간을 확인");
  submit.type = "submit";
  action.append(submit);
  form.append(action);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const started = new Date(`${start.value}+09:00`);
    const ended = new Date(end.value);
    const recordedStart = Date.parse(timer.started_at || "");
    const recordedStop = Date.parse(timer.stopped_at || "");
    if (!Number.isFinite(started.getTime()) || !Number.isFinite(ended.getTime()) || ended <= started || !Number.isFinite(recordedStart) || started.getTime() < recordedStart || (Number.isFinite(recordedStop) && ended.getTime() > recordedStop)) return showNotice("확인 구간은 기록된 시작과 종료 사이여야 하고, 종료 시각은 시작보다 뒤여야 해요.", true);
    await queueEvent("timer_confirmed", { interval_id: timer.interval_id, started_at: started.toISOString(), ended_at: ended.toISOString() });
  });
  return form;
}

export function toLocalInput(value) {
  if (!value || !Number.isFinite(Date.parse(value))) return "";
  const date = new Date(value);
  const offset = date.getTimezoneOffset() * 60000;
  return new Date(date.getTime() - offset).toISOString().slice(0, 23);
}

async function startTimer(problemKey) {
  if (!problemKey) return showNotice("문제를 선택한 뒤 시작해 주세요.", true);
  await queueEvent("timer_started", { interval_id: crypto.randomUUID(), problem_key: problemKey });
}

async function stopTimer(intervalId) { await queueEvent("timer_stopped", { interval_id: intervalId }); }

async function queueEvent(kind, payload) {
  if (!state?.learner_id) return showNotice("학습자 정보를 확인할 수 없어 저장하지 못했어요.", true);
  const event = makeEvent({ kind, learnerId: state.learner_id, payload });
  pendingWrites.push(event);
  savePending();
  await retryPending(event.event_id);
}

async function retryPending(eventId) {
  const event = pendingWrites.find((item) => item.event_id === eventId);
  if (!event) return;
  try {
    await request("/events", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(event) });
    pendingWrites = pendingWrites.filter((item) => item.event_id !== eventId);
    savePending();
    const refreshed = await refresh();
    showNotice(refreshed ? "기록을 저장했어요." : "기록은 저장했지만 최신 화면을 불러오지 못했어요. 연결을 확인해 주세요.", !refreshed);
  } catch (error) {
    showNotice(`기록을 저장하지 못했어요. 같은 기록 ID로 다시 시도할 수 있어요. (${error.message})`, true);
    render();
  }
}

function renderPending() {
  if (!pendingWrites.length && !pendingJournalWrites.length && !pendingParticipation) return;
  const region = node("section", "pending");
  region.setAttribute("aria-label", "전송 대기 중인 기록");
  const count = pendingWrites.length + pendingJournalWrites.length + Number(Boolean(pendingParticipation));
  region.append(node("p", "", `전송하지 못한 기록 ${count}개가 있어요. 다시 시도하면 같은 기록 ID를 사용해요.`));
  const list = node("ul", "");
  for (const event of pendingWrites) list.append(node("li", "", `${event.kind} · ${event.payload.problem_key || event.payload.interval_id || "시간 기록"}`));
  for (const event of pendingJournalWrites) list.append(node("li", "", `풀이 일지 · ${event.problem_key}`));
  if (pendingParticipation) list.append(node("li", "", `대회 참여 · ${pendingParticipation.contest_id}`));
  region.append(list, button("대기 중인 기록 다시 보내기", retryAll, "button secondary"));
  app.append(region);
}

async function retryAll() {
  for (const event of [...pendingWrites]) {
    await retryPending(event.event_id);
    if (pendingWrites.some((item) => item.event_id === event.event_id)) break;
  }
  for (const event of [...pendingJournalWrites]) {
    await retryJournalAttempt(event.event_id);
    if (pendingJournalWrites.some((item) => item.event_id === event.event_id)) break;
  }
  if (pendingParticipation) await retryParticipation(pendingParticipation.entry_id);
}

async function retryJournalAttempt(eventId) {
  const attempt = pendingJournalWrites.find((item) => item.event_id === eventId);
  if (!attempt) return;
  try {
    const { learner_id, ...body } = attempt;
    await request("/journal/attempt", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    pendingJournalWrites = pendingJournalWrites.filter((item) => item.event_id !== eventId);
    savePendingJournal();
    const refreshed = await refresh();
    showNotice(refreshed ? "풀이 기록을 저장했어요." : "기록은 저장했지만 최신 화면을 불러오지 못했어요. 연결을 확인해 주세요.", !refreshed);
    return true;
  } catch (error) {
    showNotice(`풀이 기록을 저장하지 못했어요. 같은 기록 ID로 다시 시도할 수 있어요. (${error.message})`, true);
    render();
    return false;
  }
}

async function retryParticipation(entryId) {
  const entry = pendingParticipation;
  if (!entry || entry.entry_id !== entryId) return;
  try {
    await request("/participation", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(entry) });
    pendingParticipation = null;
    savePendingParticipation();
    const refreshed = await refresh();
    if(refreshed) document.querySelector("dialog[open]")?.close();
    showNotice(refreshed ? "대회 참여 기록을 저장했어요." : "기록은 저장했지만 최신 화면을 불러오지 못했어요. 연결을 확인해 주세요.", !refreshed);
  } catch (error) {
    showNotice(`대회 참여 기록을 저장하지 못했어요. 같은 기록 ID로 다시 시도할 수 있어요. (${error.message})`, true);
    render();
  }
}

function preferencesCard() {
  const card = node("section", "card");
  card.append(node("h2", "", "학습 목표"), node("p", "meta", "평일 학습 시간과 집중할 문제를 설정해요."));
  const preferences = state?.preferences || {};
  const form = document.createElement("form");
  const grid = node("div", "form-grid");
  const minutes = input("weekday-minutes", "number", preferences.weekday_minutes ?? 60);
  minutes.min = "1"; minutes.max = "480"; minutes.required = true;
  grid.append(field("평일 목표 시간 (분)", minutes, true));
  const targets = select("target-keys", [...problemMap()].map(([key,p])=>[key,`${problemTitle(key)} · ${key.split(":")[0]} · ${p.native_difficulty}`]));
  targets.multiple=true;targets.size=5;
  for(const option of targets.options)option.selected=(preferences.target_keys || []).includes(option.value);
  grid.append(field("집중할 목표 문제 (여러 개 선택 가능)", targets, true));
  const rhythm=preferences.weekly_rhythm ?? ["practice","review","practice","review","practice","virtual","upsolve"];
  const rhythmOptions=[["practice","연습 중심"],["review","복습 우선"],["virtual","지난 대회 버추얼"],["upsolve","지난 대회 업솔빙"]];
  ["월요일","화요일","수요일","목요일","금요일","토요일","일요일"].forEach((day,index)=>grid.append(field(`${day} 학습`,select(`rhythm-${index}`,rhythmOptions,rhythm[index]))));
  form.append(grid);
  const action = node("div", "actions");
  const submit = node("button", "button", "목표 저장"); submit.type = "submit"; action.append(submit);
  form.append(action);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const value = Number(minutes.value);
    if (!Number.isInteger(value) || value < 1 || value > 480) return showNotice("평일 목표는 1분에서 480분 사이로 설정해 주세요.", true);
    const targetKeys = [...targets.selectedOptions].map(option=>option.value);
    const weeklyRhythm=Array.from({length:7},(_,index)=>$("#rhythm-"+index,form).value);
    try {
      await request("/preferences", { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ weekday_minutes: value, target_keys: targetKeys,weekly_rhythm:weeklyRhythm }) });
      if (await refresh()) showNotice("학습 목표를 저장했어요.");
    } catch (error) { showNotice(`목표를 저장하지 못했어요. ${error.message}`, true); }
  });
  card.append(form);
  return card;
}

function dataCard() {
  const card = node("section", "card");
  card.append(node("h2", "", "전체 학습 기록 백업과 복원"), node("p", "meta", "풀이 기록, 목표, 직접 추가한 대회와 참여 기록을 하나의 JSON 파일로 백업해요."));
  card.append(node("p", "meta", "복원은 기록이 비어 있는 계정에서만 가능해요. 기존 기록을 지우거나 덮어쓰지 않으며, 복원 뒤에는 Slack 전달 내역을 확인할 때까지 자동 전달이 멈춰요."));
  if (state?.delivery_reconciliation_required) card.append(node("p", "empty", "Slack 전달 내역 확인이 끝나지 않아 자동 전달이 멈춰 있어요."));
  const actions = node("div", "actions");
  actions.append(button("전체 백업 내려받기", exportData));
  const form = document.createElement("form");
  const file = input("import-file", "file"); file.accept = "application/json,.json"; file.required = true;
  form.append(field("복원할 JSON 백업 파일", file, true));
  const submit = node("button", "button secondary", "전체 백업 복원"); submit.type = "submit";
  form.append(node("div", "actions")); form.lastChild.append(submit);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const selected = file.files?.[0];
    if (!selected) return showNotice("먼저 백업 파일을 선택해 주세요.", true);
    try {
      const content = JSON.parse(await selected.text());
      await request("/backup/import", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(content) });
      if (await refresh()) showNotice("전체 학습 기록을 복원했어요. Slack 전달 내역을 확인할 때까지 자동 전달은 멈춰 있어요.");
    } catch (error) { showNotice(`전체 백업을 복원하지 못했어요. 기록이 비어 있는 계정인지 확인해 주세요. ${error.message}`, true); }
  });
  card.append(actions, form);
  return card;
}

async function exportData() {
  try {
    const response = await fetch(`${API}/backup/export`, { credentials: "same-origin" });
    if (!response.ok) {
      let detail = `요청에 실패했어요 (${response.status}).`;
      try { detail = (await response.json()).error || detail; } catch { /* keep HTTP detail */ }
      throw new Error(detail);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = node("a", "", "");
    anchor.href = url;
    anchor.download = `adaptive-learning-backup-${new Date().toISOString().slice(0, 10)}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
    showNotice("전체 학습 기록 백업 파일을 내려받았어요.");
  } catch (error) { showNotice(`백업을 내려받지 못했어요. ${error.message}`, true); }
}

function renderProgress() {
  app.append(...section("학습 기록", "기록된 근거와 아직 확인되지 않은 부분을 함께 보여드려요."));
  const progress = state?.progress || {};
  const confirmedSeconds = Number(state?.timers?.confirmed_seconds ?? 0);
  const timerIntervals = state?.timers?.intervals || [];
  const confirmedCount = timerIntervals.filter((item) => item.status === "confirmed").length;
  const events = Array.isArray(state?.events) ? state.events : [];
  const currentEvents = activeEvents(events);
  const attempts = currentEvents.filter((event) => event.kind === "attempt");
  const statCard = node("section", "card");
  statCard.append(node("h2", "", "기록된 활동"));
  const stats = node("div", "stat-grid");
  stats.append(stat("확인한 학습 시간", confirmedCount ? formatConfirmedMinutes(confirmedSeconds) : "기록 없음"), stat("풀이 결과 기록", `${attempts.length}개`));
  statCard.append(stats);
  const unconfirmed = timerIntervals.filter((item) => item.status !== "confirmed").length;
  if (unconfirmed) statCard.append(node("p", "meta", `확인 전 시간 기록 ${unconfirmed}개는 학습 시간 합계에 포함하지 않았어요.`));
  app.append(statCard);

  const timerCardProgress = node("section", "card");
  timerCardProgress.append(node("h2", "", "시간 기록 상태"));
  if (!timerIntervals.length) timerCardProgress.append(node("p", "empty", "시간 기록이 없어요. 실제 학습 시간을 기록하려면 시작과 종료를 직접 눌러 주세요."));
  else {
    const list = node("ul", "event-list");
    for (const interval of timerIntervals) {
      const item = node("li", "event-row");
      const label = interval.status === "confirmed" ? `확인 완료 · ${formatConfirmedMinutes(interval.confirmed_seconds)}` : interval.status === "open" ? "진행 중 · 아직 시간 미포함" : "종료됨 · 시간 확인 전, 합계 미포함";
      item.append(node("strong", "", problemTitle(interval.problem_key)), node("p", "meta", `${label} · 시작 ${interval.started_at || "시각 정보 없음"}`));
      list.append(item);
    }
    timerCardProgress.append(list);
  }
  app.append(timerCardProgress);

  const concepts = progress.concepts || {};
  const conceptCard = node("section", "card");
  conceptCard.append(node("h2", "", "개념 근거"));
  const conceptNames = Object.keys(concepts);
  if (!conceptNames.length) conceptCard.append(node("p", "empty", "아직 기록된 개념 근거가 없어요. 기록이 없다는 뜻이며, 실력이 없다는 의미는 아니에요."));
  else {
    const list = node("ul", "problem-list");
    for (const name of conceptNames.sort()) {
      const item = node("li", "problem-row");
      item.append(node("div", "split"));
      item.firstChild.append(node("strong", "", name), node("span", "tag", evidenceLabel(concepts[name]?.state)));
      const evidence = concepts[name]?.evidence;
      if (evidence) item.append(node("p", "meta", Array.isArray(evidence) ? `근거 ${evidence.length}개` : String(evidence)));
      list.append(item);
    }
    conceptCard.append(list);
  }
  app.append(conceptCard);

  const problemStates = progress.problems || {};
  const problemCard = node("section", "card");
  problemCard.append(node("h2", "", "문제별 기록"));
  const problemKeys = Object.keys(problemStates);
  if (!problemKeys.length) problemCard.append(node("p", "empty", "아직 기록된 문제별 결과가 없어요."));
  else {
    const list = node("ul", "problem-list");
    for (const key of problemKeys.sort()) {
      const item = node("li", "problem-row");
      const head = node("div", "split"); head.append(problemLink(key), node("span", "tag", evidenceLabel(problemStates[key]?.state)));
      item.append(head);
      list.append(item);
    }
    problemCard.append(list);
  }
  app.append(problemCard);

  const problemProgress = progress.problems || {};
  const dueItems = Object.entries(problemProgress).filter(([, detail]) => typeof detail?.due_date === "string" && detail.due_date).map(([problem_key, detail]) => ({ problem_key, due_date: detail.due_date }));
  const dueCard = node("section", "card");
  dueCard.append(node("h2", "", "복습 예정"));
  if (!dueItems.length) dueCard.append(node("p", "empty", "현재 확인 가능한 복습 예정일이 없어요."));
  else {
    const list = node("ul", "problem-list");
    for (const review of dueItems) {
      const item = node("li", "problem-row");
      item.append(problemLink(review.problem_key), node("p", "meta", `복습 예정일 · ${review.due_date}`));
      list.append(item);
    }
    dueCard.append(list);
  }
  app.append(dueCard);

  const eventCard = node("section", "card");
  eventCard.append(node("h2", "", "최근 기록"));
  const recent = [...events].sort((a, b) => String(b.occurred_at).localeCompare(String(a.occurred_at))).slice(0, 20);
  if (!recent.length) eventCard.append(node("p", "empty", "아직 저장된 학습 기록이 없어요."));
  else {
    const list = node("ul", "event-list");
    for (const event of recent) {
      const item = node("li", "event-row");
      const payload = event.payload || {};
      const subject = payload.problem_key ? problemTitle(payload.problem_key) : payload.interval_id || "시간 기록";
      const outcome = event.kind === "attempt" ? ` · ${payload.outcome ? outcomeLabel(payload.outcome) : "결과 미기록"}` : "";
      const source = event.source === "platform_import" ? "플랫폼에서 가져온 완료 표시" : "직접 기록";
      const correction = events.some((candidate) => candidate.supersedes === event.event_id);
      item.append(node("strong", "", subject), node("p", "meta", `${kindLabel(event.kind)}${outcome} · ${source}${correction ? " · 수정된 기록" : ""} · ${event.occurred_at || "시각 정보 없음"}`));
      list.append(item);
    }
    eventCard.append(list);
  }
  app.append(eventCard, dataCard());
}

function stat(label, value) {
  const item = node("div", "stat");
  item.append(node("strong", "", value), node("span", "", label));
  return item;
}

function outcomeLabel(value) { return ({ independent: "해결", assisted: "해결", failed: "미해결", reported_completion: "해결" })[value] || value; }
function kindLabel(value) { return ({ attempt: "풀이", timer_started: "시간 시작", timer_stopped: "시간 종료", timer_confirmed: "시간 확인" })[value] || value || "학습 기록"; }

function renderContests() {
  app.append(...section("대회 일정", "대회 정보의 출처와 최신 상태를 확인하고 직접 일정을 추가할 수 있어요."));
  const contests = Array.isArray(state?.contests) ? state.contests : [];
  const card = node("section", "card");
  card.append(node("h2", "", "확인된 일정"));
  const sources = Array.isArray(state?.contest_sources) ? state.contest_sources : [];
  if (sources.length) {
    const sourceList = node("ul", "event-list");
    for (const source of sources) {
      const statusLabel = source.status === "fresh" ? "최근 확인" : source.status === "stale" ? "이전 정보 사용 중" : "정보를 가져오지 못함";
      const sourceItem = node("li", "event-row");
      sourceItem.append(node("strong", "", source.platform || source.name || "대회 정보 제공처"), node("p", "meta", `${statusLabel}${source.fetched_at ? ` · 마지막 확인 ${source.fetched_at}` : ""}`));
      sourceList.append(sourceItem);
    }
    card.append(sourceList);
  }
  if (!contests.length) card.append(node("p", "empty", "표시할 대회 일정이 없어요. 제공처가 응답하지 않거나 일정이 아직 수집되지 않았을 수 있어요. 필요한 일정은 직접 추가해 주세요."));
  else {
    const list = node("ul", "contest-list");
    for (const contest of contests) {
      const item = node("li", "contest-row");
      const heading = node("div", "split"); heading.append(node("strong", "", contest.title || "이름 없는 대회"), node("span", `tag${contest.stale ? " warning" : ""}`, contest.stale ? "정보가 오래됨" : contest.source === "manual" ? "직접 입력" : "제공처 정보"));
      item.append(heading);
      item.append(node("p", "meta", `${contest.platform || "플랫폼 미상"} · ${contest.start_at || "시각 정보 없음"} · ${contest.duration_minutes ?? "시간 미상"}분`));
      if (contest.source && contest.source !== "manual") item.append(node("p", "meta", `출처 ${contest.source}${contest.fetched_at ? ` · 확인 ${contest.fetched_at}` : ""}`));
      if (contest.url && safeExternalUrl(contest.url)) {
        const link = node("a", "", "대회 페이지 열기"); link.href = contest.url; link.target = "_blank"; link.rel = "noopener noreferrer"; item.append(link);
      }
      if (contest.source === "manual" && contest.suitable !== undefined) item.append(node("p", "meta", contest.suitable ? "직접 학습 목표에 맞는 일정으로 표시했어요" : "직접 학습 목표에 맞지 않는 일정으로 표시했어요"));
      else if (contest.suitable === false) item.append(node("p", "meta", "학습 목표 적합성은 확인되지 않았어요."));
      list.append(item);
    }
    card.append(list);
  }
  app.append(card, providerSyncCard(), manualContestForm(), participationCard());
  const history = participationHistory();
  if (history) app.append(history);
}

function providerSyncCard() {
  const card = node("section", "card");
  card.append(node("h2", "", "대회 일정과 풀이 기록 가져오기"), node("p", "meta", "Codeforces와 AtCoder의 공개 일정을 새로 확인하거나, 공개 핸들로 제출 기록을 가져와요."));
  if (providerSummary) card.append(node("p", "empty", providerSummary));
  const enabled = state?.capabilities?.providers_enabled === true;
  if (!enabled) card.append(node("p", "empty", "현재 자동 가져오기를 사용할 수 없어요. 일정은 직접 추가하고 풀이 결과는 직접 기록할 수 있어요."));
  for (const [platform, label] of [["codeforces", "Codeforces"], ["atcoder", "AtCoder"]]) {
    const group = node("div", "timer-block");
    group.append(node("h3", "", label));
    const actions = node("div", "actions");
    const calendar = button(`${label} 일정 새로 확인`, () => syncProvider(platform), "button secondary");
    calendar.disabled = !enabled;
    actions.append(calendar);
    group.append(actions);
    const form = document.createElement("form");
    form.className = "form-grid";
    const handle = input(`provider-handle-${platform}`); handle.autocomplete = "off"; handle.maxLength = 64;
    form.append(field("공개 계정 이름", handle, true));
    const submit = node("button", "button secondary", "완료 기록 가져오기"); submit.type = "submit"; submit.disabled = !enabled;
    const submitWrap = node("div", "actions full"); submitWrap.append(submit); form.append(submitWrap);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const publicHandle = handle.value.trim();
      if (!publicHandle) return showNotice("공개 계정 이름을 입력해 주세요.", true);
      await syncProvider(platform, publicHandle);
    });
    group.append(form);
    card.append(group);
  }
  return card;
}

async function syncProvider(platform, handle) {
  if (state?.capabilities?.providers_enabled !== true) return showNotice("현재 자동 가져오기를 사용할 수 없어요. 직접 입력해 주세요.", true);
  try {
    const result = await request("/providers/sync", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ platform, ...(handle ? { handle } : {}) }) });
    const parts = [handle ? `완료 기록 ${Number(result.imported) || 0}개를 가져왔어요.` : "대회 일정을 확인했어요."];
    if (result.truncated) parts.push("이번 확인 범위 밖의 이전 기록이 있을 수 있어요.");
    if (result.historical_gap) parts.push("이전 기록 일부를 가져오지 못한 구간이 있어요.");
    if (result.status === "stale") parts.push("이전 확인 정보를 보여드려요.");
    if (result.status === "unavailable") parts.push("제공처에 연결되지 않아 직접 입력을 이용해 주세요.");
    providerSummary = parts.join(" ");
    const refreshed = await refresh();
    if (refreshed) showNotice(providerSummary, result.status === "unavailable");
  } catch (error) { showNotice(`가져오기에 실패했어요. 직접 기록할 수 있어요. ${error.message}`, true); }
}

function participationCard() {
  const card = node("section", "card");
  card.append(node("h2", "", "대회 참여 기록"), node("p", "meta", "실제 참여한 날짜와 결과를 직접 기록해요. 진행 시간은 본인이 확인한 경우에만 입력하고, 대회의 예정 시간으로 채우지 않아요."));
  const form = document.createElement("form");
  const grid = node("div", "form-grid");
  const contests = Array.isArray(state?.contests) ? state.contests : [];
  const contest = select("participation-contest", [["", "직접 입력 또는 선택"] , ...contests.map((item) => [item.id, item.title || item.id]).filter(([id]) => id)]);
  const manualId = input("participation-contest-id"); manualId.maxLength = 200; manualId.placeholder = "예: abc300";
  grid.append(field("대회 일정", contest, true), field("일정 목록에 없는 대회 ID", manualId, true));
  grid.append(field("참여 방식", select("participation-mode", [["live", "실시간 참여"], ["virtual", "버추얼 참여"], ["upsolve", "업솔빙"]])));
  grid.append(field("참여 결과", select("participation-outcome", [["completed", "완료"], ["partial", "일부 진행"], ["abandoned", "중단"], ["unknown", "확인되지 않음"]])));
  const occurred = input("participation-occurred", "datetime-local"); occurred.required = true;
  grid.append(field("참여 날짜와 시각 (Asia/Seoul)", occurred));
  const minutes = input("participation-minutes", "number"); minutes.min = "0"; minutes.step = "1"; minutes.placeholder = "모르면 비워 두기";
  grid.append(field("직접 확인한 진행 시간 (분, 선택)", minutes));
  form.append(grid);
  const actions = node("div", "actions"); const submit = node("button", "button", "참여 기록 저장"); submit.type = "submit"; actions.append(submit); form.append(actions);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (pendingParticipation) return showNotice("전송 대기 중인 대회 기록을 먼저 다시 보내 주세요.", true);
    const contestId = (contest.value || manualId.value).trim();
    const occurredAt = new Date(`${occurred.value}+09:00`);
    const confirmedMinutes = minutes.value === "" ? null : Number(minutes.value);
    if (!contestId) return showNotice("대회를 선택하거나 대회 ID를 입력해 주세요.", true);
    if (!Number.isFinite(occurredAt.getTime()) || occurredAt > new Date()) return showNotice("실제 참여 날짜와 시각을 확인해 주세요.", true);
    if (confirmedMinutes !== null && (!Number.isFinite(confirmedMinutes) || confirmedMinutes < 0)) return showNotice("확인한 진행 시간을 분 단위로 입력해 주세요.", true);
    pendingParticipation = makeParticipation({ learnerId: state.learner_id, contestId, mode: $("#participation-mode", form).value, occurredAt: occurredAt.toISOString(), outcome: $("#participation-outcome", form).value, confirmedMinutes });
    savePendingParticipation();
    await retryParticipation(pendingParticipation.entry_id);
  });
  card.append(form);
  return card;
}

function participationHistory() {
  const entries = Array.isArray(state?.participation) ? state.participation : [];
  const card = node("section", "card");
  card.append(node("h2", "", "참여 기록 내역"));
  if (!entries.length) { card.append(node("p", "empty", "아직 저장된 대회 참여 기록이 없어요.")); return card; }
  const superseded = new Set(entries.map((entry) => entry.supersedes).filter(Boolean));
  const list = node("ul", "event-list");
  for (const entry of [...entries].sort((a, b) => String(b.occurred_at).localeCompare(String(a.occurred_at)))) {
    const item = node("li", "event-row");
    const contest = (state?.contests || []).find((candidate) => candidate.id === entry.contest_id);
    const labels = { live: "실시간", virtual: "버추얼", upsolve: "업솔빙" };
    const outcomes = { completed: "완료", partial: "일부 진행", abandoned: "중단", unknown: "결과 미확인" };
    const detail = `${labels[entry.mode] || entry.mode} · ${outcomes[entry.outcome] || entry.outcome} · ${entry.confirmed_minutes === null ? "진행 시간 미기록" : `확인 ${entry.confirmed_minutes}분`} · ${entry.occurred_at || "날짜 미상"}`;
    item.append(node("strong", "", contest?.title || entry.contest_id), node("p", "meta", `${detail}${superseded.has(entry.entry_id) ? " · 수정된 기록" : ""}`));
    list.append(item);
  }
  card.append(list);
  return card;
}

function safeExternalUrl(value) {
  try { const url = new URL(value); return url.protocol === "https:" && !url.username && !url.password; } catch { return false; }
}

function manualContestForm() {
  const card = node("section", "card");
  card.append(node("h2", "", "대회 직접 추가"), node("p", "meta", "일정 제공처를 사용할 수 없을 때 직접 입력할 수 있어요."));
  const form = document.createElement("form");
  const grid = node("div", "form-grid");
  const title = input("contest-title"); title.required = true;
  const platform = select("contest-platform", [["atcoder", "AtCoder"], ["codeforces", "Codeforces"], ["leetcode", "LeetCode"], ["other", "기타"]]);
  const url = input("contest-url", "url"); url.required = true; url.placeholder = "https://...";
  const start = input("contest-start", "datetime-local"); start.required = true;
  const duration = input("contest-duration", "number"); duration.required = true; duration.min = "1";
  grid.append(field("대회 이름", title), field("플랫폼", platform), field("대회 페이지 (HTTPS)", url, true), field("시작 시각 (Asia/Seoul)", start), field("진행 시간 (분)", duration));
  const suitableWrap = node("div", "field");
  const suitableLabel = node("label", "check-line"); const suitable = input("contest-suitable", "checkbox");
  suitableLabel.append(suitable, document.createTextNode("내 학습 목표에 맞는 일정이에요"));
  suitableWrap.append(node("span", "", "학습 적합성 (직접 판단)"), suitableLabel);
  grid.append(suitableWrap);
  form.append(grid);
  const actions = node("div", "actions"); const submit = node("button", "button", "대회 저장"); submit.type = "submit"; actions.append(submit); form.append(actions);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const parsedUrl = safeExternalUrl(url.value);
    const minutes = Number(duration.value);
    if (!parsedUrl || !Number.isInteger(minutes) || minutes <= 0) return showNotice("HTTPS 대회 페이지와 올바른 진행 시간을 입력해 주세요.", true);
    const startDate = new Date(`${start.value}+09:00`);
    if (!Number.isFinite(startDate.getTime())) return showNotice("대회 시작 시각을 확인해 주세요.", true);
    try {
      await request("/contests/manual", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ id: crypto.randomUUID(), title: title.value.trim(), platform: platform.value, url: url.value.trim(), start_at: startDate.toISOString(), duration_minutes: minutes, suitable: suitable.checked }) });
      if (await refresh()) { document.querySelector("dialog[open]")?.close(); showNotice("대회 일정을 저장했어요."); }
    } catch (error) { showNotice(`대회 일정을 저장하지 못했어요. ${error.message}`, true); }
  });
  card.append(form);
  return card;
}

for (const tab of document.querySelectorAll("[data-view]")) {
  tab.addEventListener("click", () => { currentView = tab.dataset.view; if(["similarity","library"].includes(currentView))window.history.replaceState(null,"","#"+currentView);else if(["#similarity","#library"].includes(window.location.hash))window.history.replaceState(null,"",window.location.pathname+window.location.search);render(); window.scrollTo({ top: 0, behavior: "smooth" }); });
}

const workspace = createWorkspace({root:app,getState:()=>state,getJournal:()=>journalState,adapters:{
  problemTitle,explanationUsageCard,providerSyncCard,manualContestForm,participationCard,participationHistory,
  tools:()=>{currentView="today";render();},
  async syncContests(){
    try{await request('/contests/sync',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({})});await refresh();showNotice('대회 일정 확인을 마쳤어요. 플랫폼별 출처 상태와 마지막 성공 시각을 확인해 주세요.');}
    catch(error){showNotice(error.message,true);}
  },
  async saveAttempt(payload){
    if(!pendingJournalWrites.some(item=>item.event_id===payload.event_id))pendingJournalWrites.push({...payload,learner_id:state.learner_id});savePendingJournal();
    return await retryJournalAttempt(payload.event_id);
  },
}});
refresh();
window.addEventListener('offline',()=>showNotice("오프라인이에요. 저장되지 않은 기록은 재시도 목록에서 확인해 주세요.",true));
window.addEventListener('online',()=>showNotice("연결이 복구됐어요. 전송 대기 기록을 확인해 주세요."));

// Local-only evidence polling; no refresh while the learner is entering an outcome.
let localRevision=null, localPollBusy=false, localSyncWarning=null;
async function pollLocalEvidence(){
  if(localPollBusy || document.hidden || !["journal","calendar","history","analytics","contests"].includes(currentView) || app.getAttribute("aria-busy")==="true" || document.querySelector(".journal-attempt-details[open],dialog[open]") || document.activeElement?.matches("input,select,textarea"))return;
  localPollBusy=true;
  try{
    const result=await request("/local/sync");
    const warning=syncDiagnostic(result.status);
    if(warning!==localSyncWarning){const previous=localSyncWarning;localSyncWarning=warning;if(warning)showNotice(warning,true);else if(previous)showNotice('풀이 저장소의 확인 항목이 해소됐어요.');}
    if(result.revision===localRevision)return;
    const filters=[...document.querySelectorAll('.journal-filters input,.journal-filters select')].map(el=>[el.id,el.value]);
    const filterOpen=document.querySelector('.journal-filter-details')?.open;
    if(await refresh()){
      localRevision=result.revision;
      for(const [id,value] of filters){const el=document.getElementById(id);if(el)el.value=value;}
      const detail=document.querySelector('.journal-filter-details');if(detail)detail.open=Boolean(filterOpen);
      document.querySelector('.journal-filters form')?.dispatchEvent(new Event('change'));
      if(warning)showNotice(warning,true);
    }
  }catch{/* Hosted Worker and offline mode have no local sync endpoint. Keep current view. */}
  finally{localPollBusy=false;}
}
if(globalThis.location?.hostname==='127.0.0.1'){
  setInterval(pollLocalEvidence,5000);
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)pollLocalEvidence();});
}
