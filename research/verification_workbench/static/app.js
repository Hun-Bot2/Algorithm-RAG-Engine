(() => {
  "use strict";

  const state = { pairRequest: 0, pairReviewer: null, items: [], progress: null, writeToken: "", selectedId: null, selectedIndex: -1, pair: null, revealedPairs: new Set(), aiAssistedPairs: new Set(), reviewerIdentity: "", tab: "relationship" };
  const $ = (id) => document.getElementById(id);
  const api = async (url, options = {}) => {
    const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
    if (options.method === "POST" && state.writeToken) headers["X-Workbench-Token"] = state.writeToken;
    const response = await fetch(url, { headers, ...options });
    let body;
    try { body = await response.json(); } catch { body = {}; }
    if (!response.ok) throw new Error(body.error || body.message || `Request failed (${response.status})`);
    return body;
  };

  function setText(id, value, fallback = "—") { $(id).textContent = value == null || value === "" ? fallback : String(value); }
  function make(tag, className, value) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value != null) node.textContent = String(value);
    return node;
  }
  function evidenceLines(value) { return value.split("\n").map((line) => line.trim()).filter(Boolean); }
  function formValue(form, name) { return new FormData(form).get(name); }
  function showToast(message, error = false) {
    const node = $("toast");
    node.textContent = message;
    node.classList.toggle("error", error);
    node.classList.add("visible");
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => node.classList.remove("visible"), 4200);
  }
  function reviewerReady() { return Boolean($("reviewer").value.trim() && $("role").value); }
  function resetReviewForms() {
    ["relationship-form", "identity-form", "adjudication-form"].forEach((id) => $(id).reset());
    $("adjudication-canonical-wrap").hidden = true;
    $("decision").placeholder = '{"similarity":2,"prerequisite":"neither","transfer":"yes","difficulty":"similar"}';
    $("save-status").textContent = "";
  }
  function updateSessionUI() {
    const role = $("role").value;
    $("role-note").textContent = role === "ai" ? "AI reviewer · stored as an AI judgment, never as human gold." : role === "human" ? "Human reviewer · your role is recorded with each judgment." : "Choose a role before submitting.";
    $("role-note").classList.toggle("ai-note", role === "ai");
    $("submission-identity").textContent = reviewerReady() ? `${$("reviewer").value.trim()} · ${role === "ai" ? "AI reviewer" : "Human reviewer"} · self-reported` : "Reviewer and role required";
  }

  async function loadQueue() {
    $("progress").textContent = "Loading queue…";
    try {
      const data = await api("/api/pilot");
      state.items = Array.isArray(data.pilot?.items) ? data.pilot.items : [];
      state.progress = data.progress || null;
      state.writeToken = typeof data.write_token === "string" ? data.write_token : "";
      renderProgress();
      renderQueue();
      if (state.selectedId) {
        const index = state.items.findIndex((item) => item.pair_id === state.selectedId);
        if (index >= 0) { state.selectedIndex = index; await loadPair(state.selectedId); }
      }
      if (!state.selectedId && state.items.length) await selectPair(0);
    } catch (error) {
      $("progress").textContent = `Could not load pilot: ${error.message}`;
      $("queue-list").replaceChildren();
    }
  }
  function renderProgress() {
    const p = state.progress;
    if (!p) { $("progress").textContent = `${state.items.length} pairs`; return; }
    if (p.total_pairs != null) {
      $("progress").textContent = `${p.ai_draft_pairs ?? 0} AI drafts · ${p.pending_human_work ?? p.total_pairs} pending human work · ${p.accepted_adjudications ?? 0} accepted adjudications`;
      return;
    }
    const done = p.completed ?? p.reviewed ?? p.count ?? 0;
    const total = p.total ?? state.items.length;
    $("progress").textContent = `${done} of ${total} reviewed`;
  }
  function renderQueue() {
    const list = $("queue-list");
    list.replaceChildren();
    if (!state.items.length) { list.append(make("p", "muted queue-empty", "The pilot queue has no pairs.")); return; }
    state.items.forEach((item, index) => {
      const button = make("button", `queue-item${item.pair_id === state.selectedId ? " selected" : ""}`);
      button.type = "button";
      button.setAttribute("aria-current", item.pair_id === state.selectedId ? "true" : "false");
      const top = make("span", "queue-item-top");
      top.append(make("span", "queue-number", String(index + 1).padStart(2, "0")), make("span", "queue-pair", item.platform_pair || "cross-platform"));
      button.append(top, make("span", "queue-item-id", `${item.query || "Query"}  ↔  ${item.target || "Target"}`));
      button.append(make("span", "queue-item-status", item.status || "Pending review"));
      button.addEventListener("click", () => selectPair(index));
      list.append(button);
    });
  }
  async function selectPair(index) {
    if (index < 0 || index >= state.items.length) return;
    const nextId = state.items[index].pair_id;
    if (state.selectedId !== nextId) resetReviewForms();
    state.selectedIndex = index;
    state.selectedId = nextId;
    state.pair = null;
    renderQueue();
    $("empty-state").hidden = true;
    $("pair-view").hidden = false;
    $("pair-title").textContent = "Loading pair…";
    $("save-status").textContent = "";
    try { await loadPair(state.selectedId); }
    catch (error) { $("pair-title").textContent = "Pair unavailable"; showToast(`Could not load pair: ${error.message}`, true); }
  }
  async function loadPair(pairId) {
    const reviewer = $("reviewer").value.trim();
    const request = ++state.pairRequest;
    const key = `${reviewer}\n${pairId}`;
    const isRevealed = state.revealedPairs.has(key);
    const isAiAssisted = state.aiAssistedPairs.has(key);
    const mode = isRevealed ? "adjudicate" : isAiAssisted ? "ai_assisted" : "review";
    const data = await api(`/api/pair?id=${encodeURIComponent(pairId)}&reviewer=${encodeURIComponent(reviewer)}&mode=${mode}`);
    if (request !== state.pairRequest || pairId !== state.selectedId || reviewer !== $("reviewer").value.trim()) return;
    state.pair = data; state.pairReviewer = reviewer;
    renderPair(data);
  }
  function sideUrl(problem) {
    const link = problem?.original_url;
    return typeof link === "string" && /^https?:\/\//i.test(link) ? link : "";
  }
  function renderSide(prefix, problem, candidates) {
    problem = problem || {};
    setText(`${prefix}-title`, problem.title, "Title unavailable");
    setText(`${prefix}-meta`, [problem.platform, problem.native_key || problem.problem_key].filter(Boolean).join(" · "));
    setText(`${prefix}-source`, `Publisher description; section completeness unverified. Source: ${problem.source_dataset || "fixture/unknown"}. Language: ${problem.language || "unavailable"}. Native identity: ${problem.native_key ? "source-provided" : "unresolved"}.`);
    $(`${prefix}-statement`).textContent = problem.statement_text || "Statement text unavailable.";
    const link = $(`${prefix}-link`);
    const href = sideUrl(problem);
    if (href) { link.href = href; link.hidden = false; } else { link.removeAttribute("href"); link.hidden = true; }
    const tags = $(`${prefix}-tags`);
    tags.replaceChildren();
    (Array.isArray(problem.source_tags) ? problem.source_tags : []).forEach((tag) => tags.append(make("span", "tag", tag)));
    const candidateList = $(`${prefix}-candidates`);
    candidateList.replaceChildren();
    const rows = Array.isArray(candidates) ? candidates : [];
    $(`${prefix}-candidate-count`).textContent = rows.length ? `(${rows.length})` : "(none supplied)";
    if (!rows.length) candidateList.append(make("p", "muted", "No identity candidates supplied."));
    rows.forEach((candidate) => {
      const row = make("div", "candidate");
      const title = make("strong", "", candidate.title || candidate.problem_key || "Candidate");
      row.append(title, make("span", "candidate-key", candidate.problem_key || "ID unavailable"), make("span", "candidate-basis", candidate.basis || "Basis unavailable"));
      if (typeof candidate.url === "string" && /^https?:\/\//i.test(candidate.url)) {
        const a = make("a", "candidate-link", "Source ↗"); a.href = candidate.url; a.target = "_blank"; a.rel = "noreferrer"; row.append(a);
      }
      candidateList.append(row);
    });
  }
  function renderPair(data) {
    const pair = data.pair || {};
    const current = state.items[state.selectedIndex] || {};
    setText("pair-title", pair.pair_id || "Pair");
    setText("pair-meta", [current.stratum, current.platform_pair, current.status].filter(Boolean).join(" · "));
    renderSide("query", data.query, data.identity_candidates?.query);
    renderSide("target", data.target, data.identity_candidates?.target);
    const methods = $("methods");
    methods.replaceChildren();
    const ranks = data.ai_assisted ? {} : (pair.methods || current.methods || current.method_ranks || {});
    if (Array.isArray(ranks)) {
      ranks.forEach((method) => methods.append(make("span", "method-chip", typeof method === "string" ? method : `${method.name || method.method || "method"}${method.rank != null ? ` · rank ${method.rank}` : ""}`)));
    } else {
      Object.entries(ranks).forEach(([name, rank]) => methods.append(make("span", "method-chip", `${name} · rank ${rank}`)));
    }
    if (!methods.childElementCount) methods.append(make("span", "muted", data.ai_assisted ? "Retrieval ranks hidden during AI-assisted review." : data.blind ? "Retrieval ranks hidden during independent review." : "No method output supplied."));
    renderRecords(data.records);
    const revealKey = `${$("reviewer").value.trim()}\n${state.selectedId}`;
    const revealed = state.revealedPairs.has(revealKey);
    const aiAssisted = state.aiAssistedPairs.has(revealKey);
    $("view-ai-draft").disabled = revealed || aiAssisted || $("role").value !== "human" || !$("reviewer").value.trim();
    $("view-ai-draft").textContent = aiAssisted ? "AI draft viewed" : "View AI draft";
    $("reveal-reviews").disabled = revealed || $("role").value !== "human" || !$("reviewer").value.trim();
    $("reveal-reviews").textContent = revealed ? "Reviews revealed" : "Reveal reviews";
    $("reveal-warning").textContent = revealed
      ? "Other reviews are revealed. You cannot submit a new independent review for this pair; use this mode only for adjudication."
      : aiAssisted
        ? "AI drafts are visible. Record your judgment as AI-assisted; you cannot claim an independent judgment for this pair."
        : data.reviewer_exposed
          ? "This reviewer has previously viewed exposed records for this pair. Independent judgment is locked; choose an exposed view to inspect them again."
          : "Viewing AI drafts or revealing reviews locks independent judgment for this pair.";
    const independentLocked = revealed || aiAssisted || data.reviewer_exposed;
    document.querySelectorAll('input[name="independent"]').forEach((input) => { if (independentLocked) input.checked = false; input.disabled = independentLocked; });
    $("previous").disabled = state.selectedIndex <= 0;
    $("next").disabled = state.selectedIndex >= state.items.length - 1;
  }
  function renderRecords(records) {
    const box = $("records");
    box.replaceChildren();
    if (!Array.isArray(records) || !records.length) { box.append(make("p", "muted", "No other records are visible for this pair.")); return; }
    records.forEach((record) => {
      const card = make("article", "record-card");
      const role = record.role === "ai" ? "AI judgment" : record.role === "human" ? "Human judgment" : "Review record";
      card.append(make("div", "record-top", `${role} · ${record.kind || "record"}`), make("div", "record-reviewer", record.reviewer || "Reviewer unavailable"));
      const payload = record.decision || record;
      if (record.kind === "relationship") {
        const similarity = payload.similarity == null ? "unknown" : String(payload.similarity);
        const difficulty = { query_easier: "Query easier", target_easier: "Target easier", similar: "Similar", unknown: "Unknown" }[payload.difficulty] || payload.difficulty;
        card.append(make("p", "record-summary", `Similarity: ${similarity} · Difficulty: ${difficulty} · Transfer: ${payload.transfer} · Prerequisite: ${payload.prerequisite}`));
        card.append(make("p", "record-rationale", payload.rationale || "Rationale unavailable."));
        const evidence = make("details", "record-evidence");
        evidence.append(make("summary", "", "Evidence and source references"));
        const list = make("ul");
        (payload.evidence || []).forEach(item => list.append(make("li", "", item)));
        evidence.append(list); card.append(evidence);
      }
      const raw = make("details", "record-raw"); raw.append(make("summary", "", "Full review record"));
      raw.append(make("pre", "record-payload", JSON.stringify(payload, null, 2))); card.append(raw);
      if (record.id || record.record_id) card.append(make("code", "record-id", record.id || record.record_id));
      box.append(card);
    });
  }
  function getBaseRecord(kind) {
    if (!state.pair || state.pairReviewer !== $("reviewer").value.trim()) throw new Error("Wait for this reviewer’s blind view to load.");
    if (!reviewerReady()) throw new Error("Enter a reviewer name and choose human or AI reviewer.");
    return { pair_id: state.selectedId, reviewer: $("reviewer").value.trim(), role: $("role").value, kind };
  }
  async function submitForm(form, buildPayload) {
    const button = form.querySelector('button[type="submit"]');
    const status = $("save-status");
    button.disabled = true;
    status.className = "save-status";
    status.textContent = "Saving…";
    try {
      const payload = buildPayload(form);
      const response = await api("/api/records", { method: "POST", body: JSON.stringify(payload) });
      status.classList.add("success");
      status.textContent = response.message || "Record saved.";
      showToast("Review saved.");
      await loadPair(state.selectedId);
      await loadQueue();
    } catch (error) {
      status.classList.add("error");
      status.textContent = error.message || "Could not save the record.";
      showToast(`Save failed: ${error.message}`, true);
    } finally { button.disabled = false; }
  }
  function bindForms() {
    $("relationship-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const form = event.currentTarget;
      submitForm(form, () => {
        const payload = getBaseRecord("relationship");
        const similarity = formValue(form, "similarity");
        if (similarity === "") throw new Error("Choose a similarity rating or Unable to judge.");
        payload.similarity = similarity === "null" ? null : Number(similarity);
        payload.prerequisite = formValue(form, "prerequisite");
        payload.transfer = formValue(form, "transfer");
        payload.difficulty = formValue(form, "difficulty");
        payload.rationale = formValue(form, "rationale").trim();
        payload.evidence = evidenceLines(formValue(form, "evidence"));
        payload.independent = formValue(form, "independent") === "on";
        if (!Number.isInteger(payload.similarity) && payload.similarity !== null) throw new Error("Choose a similarity rating or Unable to judge.");
        for (const key of ["prerequisite", "transfer", "difficulty"]) if (!payload[key]) throw new Error("Complete all relationship dimensions.");
        if (payload.rationale.length < 10) throw new Error("Add a substantive rationale (at least 10 characters).");
        if (!payload.evidence.length) throw new Error("Add at least one evidence item.");
        return payload;
      });
    });
    $("identity-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const form = event.currentTarget;
      submitForm(form, () => {
        const payload = getBaseRecord("identity");
        const side = formValue(form, "subject_side");
        if (!side) throw new Error("Choose whether the identity proposal applies to the query or target.");
        const problem = state.pair?.[side] || {};
        payload.subject_key = problem.problem_key;
        payload.native_key = formValue(form, "native_key").trim();
        payload.status = "proposed";
        payload.rationale = formValue(form, "rationale").trim();
        payload.evidence = evidenceLines(formValue(form, "evidence"));
        if (!payload.subject_key) throw new Error("The selected problem has no key to attach this proposal to.");
        if (!payload.native_key || payload.rationale.length < 10) throw new Error("Provide a proposed native ID and substantive rationale (at least 10 characters).");
        const canonicalUrl = formValue(form, "canonical_url").trim();
        if (!/^https?:\/\//i.test(canonicalUrl)) throw new Error("Provide the canonical native problem URL as evidence.");
        payload.evidence = [...new Set([...payload.evidence, canonicalUrl])];
        payload.independent = formValue(form, "independent") === "on";
        return payload;
      });
    });
    $("adjudication-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const form = event.currentTarget;
      submitForm(form, () => {
        const payload = getBaseRecord("adjudication");
        payload.task = formValue(form, "task");
        payload.review_ids = formValue(form, "review_ids").split(",").map((id) => id.trim()).filter(Boolean);
        payload.outcome = formValue(form, "outcome");
        try { payload.decision = JSON.parse(formValue(form, "decision")); } catch { throw new Error("Decision details must be a valid JSON object."); }
        payload.rationale = formValue(form, "rationale").trim();
        payload.evidence = evidenceLines(formValue(form, "evidence"));
        if (payload.review_ids.length !== 2 || new Set(payload.review_ids).size !== 2 || !payload.outcome || payload.rationale.length < 10 || !payload.decision || Array.isArray(payload.decision) || typeof payload.decision !== "object") throw new Error("Provide two distinct review IDs, an outcome, a decision object, and substantive rationale.");
        if (payload.task === "relationship") {
          if (Object.keys(payload.decision).sort().join(",") !== "difficulty,prerequisite,similarity,transfer") throw new Error("Relationship decision must include exactly similarity, prerequisite, transfer, and difficulty.");
          if (!payload.evidence.length) throw new Error("Add at least one evidence item.");
          const similarity = payload.decision.similarity;
          if (similarity !== null && (!Number.isInteger(similarity) || similarity < 0 || similarity > 3)) throw new Error("Decision similarity must be 0–3 or null.");
          const choices = { prerequisite: ["query_before_target", "target_before_query", "neither", "unknown"], transfer: ["yes", "no", "unknown"], difficulty: ["query_easier", "similar", "target_easier", "unknown"] };
          for (const [key, values] of Object.entries(choices)) if (!values.includes(payload.decision[key])) throw new Error(`Invalid relationship decision: ${key}.`);
        } else {
          if (Object.keys(payload.decision).sort().join(",") !== "native_key,subject_key") throw new Error("Identity decision must include exactly subject_key and native_key.");
          const canonicalUrl = formValue(form, "canonical_url").trim();
          if (!/^https?:\/\//i.test(canonicalUrl)) throw new Error("Provide the canonical native problem URL as adjudication evidence.");
          payload.evidence = [...new Set([...payload.evidence, canonicalUrl])];
        }
        if (payload.role !== "human") throw new Error("Adjudication must be submitted by a human reviewer.");
        return payload;
      });
    });
  }
  function bindTabs() {
    document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
      state.tab = tab.dataset.tab;
      document.querySelectorAll(".tab").forEach((item) => { const active = item === tab; item.classList.toggle("active", active); item.setAttribute("aria-selected", String(active)); });
      ["relationship", "identity", "adjudication"].forEach((name) => { $(`${name}-form`).hidden = name !== state.tab; });
      $("save-status").textContent = "";
    }));
  }
  $("adjudication-form").elements.task.addEventListener("change", (event) => {
    const identity = event.currentTarget.value === "identity";
    $("decision").placeholder = identity ? '{"subject_key":"...","native_key":"..."}' : '{"similarity":2,"prerequisite":"neither","transfer":"yes","difficulty":"similar"}';
    $("adjudication-canonical-wrap").hidden = !identity;
  });
  $("refresh").addEventListener("click", loadQueue);
  $("previous").addEventListener("click", () => selectPair(state.selectedIndex - 1));
  $("next").addEventListener("click", () => selectPair(state.selectedIndex + 1));
  const reviewerChanged = () => {
    updateSessionUI();
    const reviewer = $("reviewer").value.trim();
    if (reviewer !== state.reviewerIdentity) {
      state.reviewerIdentity = reviewer;
      resetReviewForms();
    }
    if (state.selectedId) loadPair(state.selectedId).catch((error) => showToast(`Could not refresh visible records: ${error.message}`, true));
  };
  $("reveal-reviews").addEventListener("click", async () => {
    const reviewer = $("reviewer").value.trim();
    if (!reviewer || $("role").value !== "human") { showToast("Enter a reviewer name and choose Human reviewer before revealing records.", true); return; }
    const key = `${reviewer}\n${state.selectedId}`;
    if (state.revealedPairs.has(key)) return;
    state.revealedPairs.add(key);
    $("reveal-warning").textContent = "Other reviews are being revealed. Independent review is locked for this pair.";
    document.querySelectorAll('input[name="independent"]').forEach((input) => { input.checked = false; input.disabled = true; });
    $("reveal-reviews").disabled = true;
    $("reveal-reviews").textContent = "Loading reviews…";
    try {
      await loadPair(state.selectedId);
      showToast("Other reviews revealed. Independent review is now locked for this pair.");
    } catch (error) {
      $("reveal-reviews").textContent = "Reveal attempted";
      showToast(`Reveal outcome may be uncertain; independent review stays locked for this pair. ${error.message}`, true);
    }
  });
  $("view-ai-draft").addEventListener("click", async () => {
    const reviewer = $("reviewer").value.trim();
    if (!reviewer || $("role").value !== "human") { showToast("Enter a reviewer name and choose Human reviewer before viewing AI drafts.", true); return; }
    const key = `${reviewer}\n${state.selectedId}`;
    if (state.aiAssistedPairs.has(key)) return;
    state.aiAssistedPairs.add(key);
    document.querySelectorAll('input[name="independent"]').forEach((input) => { input.checked = false; input.disabled = true; });
    $("view-ai-draft").disabled = true;
    $("view-ai-draft").textContent = "Loading AI draft…";
    try {
      await loadPair(state.selectedId);
      showToast("AI draft shown. Independent judgment is locked for this pair.");
    } catch (error) {
      $("view-ai-draft").textContent = "AI draft view attempted";
      showToast(`AI draft view outcome may be uncertain; independent judgment stays locked. ${error.message}`, true);
    }
  });
  $("reviewer").addEventListener("input", () => {
    state.pairRequest += 1; state.pair = null; state.pairReviewer = null;
    renderRecords([]); $("methods").replaceChildren(make("span", "muted", "Retrieval ranks hidden during independent review."));
    resetReviewForms();
  });
  $("reviewer").addEventListener("change", reviewerChanged);
  $("reviewer").addEventListener("blur", updateSessionUI);
  $("role").addEventListener("change", updateSessionUI);
  bindForms(); bindTabs(); updateSessionUI(); loadQueue();
})();
