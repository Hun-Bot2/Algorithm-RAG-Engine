import {relationshipProviders as defaultProviders,fuseCandidates} from './relationships.mjs';
import {normalizePreferences} from './preferences.mjs';
import {activeEvents, timestamp} from './events.mjs';
import {getMission, learnerSnapshot, saveMission} from './store.mjs';

export const POLICY_VERSION = 'deterministic-v2';
export const REVIEW_DAYS = [1,3,7,14,30,60];
export const seoulDate = date => new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Seoul',year:'numeric',month:'2-digit',day:'2-digit'}).format(date);
const plusDays = (date,days) => new Date(Date.parse(date.slice(0,10)+'T00:00:00Z')+days*86400000).toISOString().slice(0,10);
const approved = item => item?.review?.status === 'approved';
const eligible = item => approved(item) && item.access_state === 'free';

export function deriveProgress(curriculum, events, cutoff) {
  const active = activeEvents(events,cutoff).filter(e=>e.kind==='attempt');
  const byKey = new Map();
  for (const e of active) {
    const key = e.payload.problem_key;
    if (!byKey.has(key)) byKey.set(key,[]);
    byKey.get(key).push(e);
  }
  const transferEdges = new Set((curriculum.relationships ?? []).filter(r=>r.type==='transfer' && approved(r)).map(r=>r.source+'\0'+r.target));
  const priorDemonstration = (key,at) => (byKey.get(key) ?? []).some(e=>e.occurred_at < at && e.payload.outcome==='independent' && e.payload.hints===false);
  const problems = {}, concepts = {};
  for (const p of curriculum.problems ?? []) {
    const attempts = byKey.get(p.problem_key) ?? [];
    let state = 'unknown', reviewStage = 0, dueDate = null, latestOutcome = null, evidence = [];
    for (const e of attempts) {
      const a = e.payload, date = seoulDate(new Date(e.occurred_at));
      latestOutcome = a.outcome;
      if (a.outcome === 'reported_completion') { evidence.push(e.event_id); continue; }
      if (a.outcome === 'independent' && a.hints === false) {
        const transfer = a.activity_type === 'transfer' && transferEdges.has(a.transfer_from+'\0'+p.problem_key) && priorDemonstration(a.transfer_from,e.occurred_at);
        if (transfer) state = 'transfer-evidenced';
        else if (state !== 'transfer-evidenced') state = 'demonstrated';
        reviewStage = a.activity_type === 'review' ? Math.min(reviewStage+1,REVIEW_DAYS.length-1) : 0;
        dueDate = plusDays(date,REVIEW_DAYS[reviewStage]);
      } else {
        if (state === 'unknown' || a.activity_type === 'review') state = 'learning';
        if (a.activity_type === 'review') { reviewStage = 0; dueDate = plusDays(date,1); }
      }
      evidence.push(e.event_id);
    }
    problems[p.problem_key] = {state,review_stage:reviewStage,due_date:dueDate,latest_outcome:latestOutcome,evidence};
    for (const concept of p.concepts ?? []) {
      const entry = concepts[concept] ?? {state:'unknown',evidence:[]};
      const rank = {'unknown':0,'learning':1,'demonstrated':2,'transfer-evidenced':3};
      if (rank[state] > rank[entry.state]) entry.state = state;
      entry.evidence.push(...evidence);
      concepts[concept] = entry;
    }
  }
  for (const value of Object.values(concepts)) value.evidence = [...new Set(value.evidence)].sort();
  return {problems,concepts};
}

function planKeys(curriculum, targetKeys, progress) {
  const lookup = new Map((curriculum.problems ?? []).map(p=>[p.problem_key,p]));
  const incoming = new Map();
  for (const r of curriculum.relationships ?? []) if (r.type==='prerequisite' && approved(r) && eligible(lookup.get(r.source)) && eligible(lookup.get(r.target))) {
    if (!incoming.has(r.target)) incoming.set(r.target,[]);
    incoming.get(r.target).push(r.source);
  }
  const targetPath = new Set(), ready = new Set();
  const visit = (key,seen=new Set()) => {
    if (seen.has(key) || !eligible(lookup.get(key))) return;
    seen.add(key); targetPath.add(key);
    const parents = incoming.get(key) ?? [];
    parents.forEach(parent=>visit(parent,seen));
    if (parents.every(parent=>['demonstrated','transfer-evidenced'].includes(progress.problems[parent]?.state))) ready.add(key);
  };
  targetKeys.forEach(key=>visit(key));
  if (!targetPath.size && targetKeys.length===0) for (const p of lookup.values()) if (eligible(p)) {
    const parents = incoming.get(p.problem_key) ?? [];
    if (parents.every(parent=>['demonstrated','transfer-evidenced'].includes(progress.problems[parent]?.state))) ready.add(p.problem_key);
  }
  return {lookup,targetPath,ready};
}

function difficultyBand(problem,annotations) {
  const annotation = annotations?.[problem.problem_key];
  const band = annotation?.difficulty_band;
  if (['beginner','intermediate','advanced'].includes(band)) return band;
  const native = problem.native_difficulty.toLowerCase();
  if (/\b(easy|beginner)\b|^abc[- ]?a$/.test(native)) return 'beginner';
  if (/\b(medium|intermediate)\b|^abc[- ]?b$/.test(native)) return 'intermediate';
  return 'advanced';
}
export function buildMission({curriculum,annotations={},events,preferences,now,contests=[],providers}) {
  const at = now instanceof Date ? now : new Date(now);
  if (Number.isNaN(at.getTime())) throw new Error('invalid mission time');
  preferences=normalizePreferences(preferences);
  const cutoff = timestamp(at.toISOString());
  const date = seoulDate(at), progress = deriveProgress(curriculum,events,cutoff);
  const rhythm=preferences.weekly_rhythm[(new Date(date+"T00:00:00Z").getUTCDay()+6)%7];
  const {lookup,targetPath,ready} = planKeys(curriculum,preferences.target_keys ?? [],progress);
  const active = activeEvents(events,cutoff).filter(e=>e.kind==='attempt');
  const hasIndependent = active.some(e=>e.payload.outcome==='independent' && e.payload.hints===false);
  const recent = active.at(-1)?.payload.problem_key?.split(':')[0];
  const relationships=fuseCandidates(providers ?? defaultProviders(curriculum,annotations),preferences.target_keys);
  const candidates = [];
  for (const p of lookup.values()) {
    if (!eligible(p)) continue;
    const difficulty = difficultyBand(p,annotations);
    if (!hasIndependent && difficulty !== 'beginner') continue;
    const state = progress.problems[p.problem_key];
    const due = !!state?.due_date && state.due_date <= date;
    let type = due ? 'review' : 'practice';
    const transfer = !due && (curriculum.relationships ?? []).find(r=>r.type==='transfer' && approved(r) && r.target===p.problem_key && ['demonstrated','transfer-evidenced'].includes(progress.problems[r.source]?.state));
    if(transfer)type='transfer';
    if (!due && (!ready.has(p.problem_key) || ['demonstrated','transfer-evidenced'].includes(state?.state))) continue;
    const components = {
      target_gap:targetPath.has(p.problem_key) ? 4 : 0,
      weakness:state?.state==='unknown' ? 3 : state?.state==='learning' ? 2 : due ? 2 : 0,
      suitability:ready.has(p.problem_key) ? difficulty==='beginner' ? 2 : difficulty==='intermediate' ? 1 : 0 : 0,
      variety:recent && recent !== p.problem_key.split(':')[0] ? 1 : 0,
      review_priority:rhythm==='review' && due ? 3 : 0,
      relationship_rank:relationships.candidates.get(p.problem_key)?.rank_fusion ?? 0,
    };
    const score = Object.values(components).reduce((a,b)=>a+b,0) + (due ? 5 : 0);
    const reason = type==='review' ? `기록된 복습 기한(${state.due_date})이 도래했습니다.` : targetPath.has(p.problem_key) ? '선택한 목표의 검토된 선행 경로에 있습니다.' : '검토된 무료 문제이며 현재 선행 조건을 충족합니다.';
    candidates.push({type,problem_key:p.problem_key,score,components,reason:transfer ? '독립 풀이 기록이 있는 문제에서 검토된 응용 관계로 이어집니다.' : reason,...(transfer?{transfer_from:transfer.source}:{}),estimated_minutes:type==='review'?15:30,evidence:state?.evidence ?? [],relationship_evidence:relationships.candidates.get(p.problem_key)?.evidence ?? []});
  }
  candidates.sort((a,b)=>b.score-a.score || a.problem_key.localeCompare(b.problem_key) || a.type.localeCompare(b.type));
  const exposure = candidates.map((c,index)=>({...c,rank:index+1}));
  const weekday = at.toLocaleString('en-US',{timeZone:'Asia/Seoul',weekday:'short'});
  const budget = ['Sat','Sun'].includes(weekday) ? 60 : preferences.weekday_minutes ?? 60;
  let used = 0; const activities = [];
  for (const c of candidates) if (used+c.estimated_minutes <= budget) {activities.push(c); used += c.estimated_minutes;}
  const suitableContests = contests.filter(c=>c?.suitable && c.start_at).sort((a,b)=>a.start_at.localeCompare(b.start_at) || a.id.localeCompare(b.id));
  const todayContest = suitableContests.find(c=>seoulDate(new Date(c.start_at))===date);
  const upcoming = suitableContests.find(c=>seoulDate(new Date(c.start_at))>date && seoulDate(new Date(c.start_at))<=plusDays(date,7));
  if (todayContest) {
    activities.length = 0;
    activities.push({type:'contest',contest_id:todayContest.id,title:todayContest.title,url:todayContest.url,reason:'적합하다고 표시된 오늘의 대회입니다.',duration_minutes:todayContest.duration_minutes,estimated_minutes:null});
  } else if (upcoming && activities.length) {
    const practice = activities.find(a=>a.type==='practice');
    if (practice) { practice.contest_id = upcoming.id; practice.reason += ' 예정 대회 준비에도 사용할 수 있습니다.'; }
  }
  const recentContest = suitableContests.filter(c=>seoulDate(new Date(c.start_at))<date && seoulDate(new Date(c.start_at))>=plusDays(date,-7)).at(-1);
  if (!todayContest && ['virtual','upsolve'].includes(rhythm)) {
    const past=suitableContests.filter(c=>seoulDate(new Date(c.start_at))<date).at(-1);
    activities.length=0;
    activities.push({type:rhythm,contest_id:past?.id ?? null,title:past?.title ?? '연습할 지난 대회 선택',url:past?.url ?? null,
      reason:rhythm==='virtual' ? '주간 계획에 따라 지난 대회를 시간 제한을 정해 연습합니다.' : '주간 계획에 따라 지난 대회의 미해결 문제를 선택합니다.',
      estimated_minutes:budget,planning_only:true});
  } else if (!todayContest && !activities.length && recentContest) activities.push({type:'upsolve',contest_id:recentContest.id,title:recentContest.title,url:recentContest.url,reason:'최근 대회의 미해결 문제를 수동으로 확인해 보세요.',estimated_minutes:budget,planning_only:true});
  return {date,policy_version:POLICY_VERSION,curriculum_version:curriculum.curriculum_version,event_cutoff:cutoff,weekday_minutes:preferences.weekday_minutes,weekly_rhythm:preferences.weekly_rhythm,day_focus:rhythm,relationship_providers:relationships.statuses,activities,exposure,candidate_count:exposure.length};
}

export async function prepareMission({db,learnerId,curriculum,annotations={},now=new Date(),preparedAt=now,contests=[]}) {
  const date = seoulDate(now instanceof Date ? now : new Date(now));
  const frozen = await getMission(db,learnerId,date);
  if (frozen) return frozen;
  const snap = await learnerSnapshot(db,learnerId);
  const mission = buildMission({curriculum,annotations,events:snap.events,preferences:snap.preferences,now,contests});
  mission.prepared_at=timestamp((preparedAt instanceof Date ? preparedAt : new Date(preparedAt)).toISOString());
  return saveMission(db,learnerId,date,mission,snap.revision);
}
