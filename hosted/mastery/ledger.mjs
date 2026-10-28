// Single reader for all learner evidence (design.md section 2.2). Policy modules use only this.
import {activeEvents,identifier} from '../core/events.mjs';
import {activeParticipation} from '../contest-history.mjs';
import {canonicalJson} from '../contest-history.mjs';
import {LEDGERS,activeEntries,listEntries} from './side-ledger.mjs';

export const LEDGER_TABLES = ['events','contest_participation',...Object.keys(LEDGERS)];
const RANK = {events:1,contest_participation:2,...Object.fromEntries(Object.entries(LEDGERS).map(([t,s])=>[t,s.rank]))};

async function rows(db,table,learnerId,limit) {
  const sql=`SELECT sequence,canonical FROM ${table} WHERE learner_id=? ORDER BY sequence`;
  const result=(await db.prepare(sql).bind(learnerId).all()).results;
  return result.filter(r=>limit===undefined || limit===null || r.sequence<=limit).map(r=>({...JSON.parse(r.canonical),sequence:r.sequence}));
}
export async function highWaterMarks(db,learnerId) {
  identifier(learnerId);
  const marks={};
  for (const table of LEDGER_TABLES) marks[table]=(await db.prepare(`SELECT COALESCE(MAX(sequence),0) AS m FROM ${table} WHERE learner_id=?`).bind(learnerId).first()).m;
  return marks;
}
// Reads every ledger. With highWater, only rows with sequence <= mark are visible (reproduction mode).
export async function readLedger(db,learnerId,{cutoff=null,highWater=null}={}) {
  identifier(learnerId);
  const marks=highWater ?? await highWaterMarks(db,learnerId);
  const raw={};
  for (const table of LEDGER_TABLES) raw[table]=await rows(db,table,learnerId,marks[table] ?? 0);
  const strip=list=>list.map(({sequence,...e})=>e);
  const ledger={
    high_water:marks, cutoff,
    events:activeEvents(strip(raw.events),cutoff),
    participation:activeParticipation(strip(raw.contest_participation),cutoff),
  };
  for (const table of Object.keys(LEDGERS)) ledger[table]=activeEntries(strip(raw[table]),cutoff);
  ledger.attempts=attemptStream(ledger);
  return ledger;
}
// Policy v3 (ADR 005): help is not tracked. Any completion counts as solved; stored evidence is never rewritten.
export const SOLVED_OUTCOMES = new Set(['independent','assisted','reported_completion']);
export const solvedView = (outcome,hints) => ({outcome:SOLVED_OUTCOMES.has(outcome)?'independent':outcome,hints:SOLVED_OUTCOMES.has(outcome)?false:null,recorded_outcome:outcome,recorded_hints:hints ?? null});
// Unified attempt evidence. Git rows are labeled evidence_source=git. Outcomes are projected once here (SO2).
export function attemptStream(ledger) {
  const out=[];
  for (const e of ledger.events) if (e.kind==='attempt') out.push({id:e.event_id,table:'events',occurred_at:e.occurred_at,recorded_at:e.recorded_at,
    problem_key:e.payload.problem_key,...solvedView(e.payload.outcome,e.payload.hints),activity_type:e.payload.activity_type ?? null,
    transfer_from:e.payload.transfer_from ?? null,evidence_source:e.source==='platform_import'?'platform_import':'manual',self_minutes:e.payload.self_minutes ?? null,...(e.payload.self_seconds!=null?{self_seconds:e.payload.self_seconds}:{})});
  for (const g of ledger.git_observations ?? []) out.push({id:g.entry_id,table:'git_observations',occurred_at:g.occurred_at,recorded_at:g.recorded_at,
    problem_key:g.payload.problem_key,...solvedView(g.payload.outcome,g.payload.hints),activity_type:g.payload.activity_type,
    transfer_from:g.payload.transfer_from,evidence_source:g.source==='manual_correction'?'git_correction':'git',self_minutes:g.payload.self_minutes,...(g.payload.self_seconds!=null?{self_seconds:g.payload.self_seconds}:{})});
  return out.sort((a,b)=>a.occurred_at.localeCompare(b.occurred_at) || RANK[a.table]-RANK[b.table] || a.id.localeCompare(b.id));
}
export const isKnownOutcome = a => ['independent','assisted','failed'].includes(a.outcome);
export const solvedIndependently = a => a.outcome==='independent' && a.hints===false;

// Immutable snapshot helpers (recommendation_snapshots / weekly_reports).
export async function getSnapshot(db,learnerId,kind,localDate) {
  const row=await db.prepare('SELECT payload FROM recommendation_snapshots WHERE learner_id=? AND kind=? AND local_date=?').bind(learnerId,kind,localDate).first();
  return row ? JSON.parse(row.payload) : null;
}
export async function saveSnapshot(db,learnerId,kind,localDate,payload) {
  identifier(learnerId); identifier(kind);
  await db.batch([db.prepare('INSERT OR IGNORE INTO learners(learner_id) VALUES (?)').bind(learnerId),
    db.prepare('INSERT OR IGNORE INTO recommendation_snapshots(learner_id,kind,local_date,payload) VALUES (?,?,?,?)').bind(learnerId,kind,localDate,canonicalJson(payload))]);
  return getSnapshot(db,learnerId,kind,localDate);
}
export async function getWeeklyReport(db,learnerId,weekStart) {
  const row=await db.prepare('SELECT payload FROM weekly_reports WHERE learner_id=? AND week_start=?').bind(learnerId,weekStart).first();
  return row ? JSON.parse(row.payload) : null;
}
export async function saveWeeklyReport(db,learnerId,weekStart,payload) {
  await db.batch([db.prepare('INSERT OR IGNORE INTO learners(learner_id) VALUES (?)').bind(learnerId),
    db.prepare('INSERT OR IGNORE INTO weekly_reports(learner_id,week_start,payload) VALUES (?,?,?)').bind(learnerId,weekStart,canonicalJson(payload))]);
  return getWeeklyReport(db,learnerId,weekStart);
}
