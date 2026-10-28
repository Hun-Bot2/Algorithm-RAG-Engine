// Append-only side ledgers (specs/mastery-system/design.md section 2.2).
// Same contract as contest_participation: idempotent IDs, same-table single-successor
// corrections, ownership, and occurred_at <= recorded_at. Worker-safe (no Node APIs).
import {identifier,problemKey,timestamp} from '../core/events.mjs';
import {canonicalJson} from '../contest-history.mjs';

const exact=(value,fields)=>value && typeof value==='object' && !Array.isArray(value) && Object.keys(value).sort().join('|')===[...fields].sort().join('|');
const oneOf=(value,options,message)=>{ if (!options.includes(value)) throw new Error(message); return value; };
const nullable=(value,check)=>value===null ? null : check(value);
const text=(value,max=500)=>{ if (typeof value!=='string' || !value.trim() || value.length>max) throw new Error('invalid ledger text'); return value; };
const count=value=>{ if (!Number.isInteger(value) || value<0) throw new Error('invalid ledger count'); return value; };

const OUTCOMES=['independent','assisted','failed','reported_completion'];
function gitObservation(p) {
  if (!exact(p,['repo_id','sha','path','problem_key','outcome','hints','self_minutes','activity_type','transfer_from','trailers',...('self_seconds' in p?['self_seconds']:[])])) throw new Error('invalid git observation payload');
  identifier(p.repo_id);
  if (typeof p.sha!=='string' || !/^[0-9a-f]{40}$/.test(p.sha)) throw new Error('invalid git observation sha');
  nullable(p.path,v=>text(v,1000));
  problemKey(p.problem_key);
  oneOf(p.outcome,OUTCOMES,'invalid git observation outcome');
  if (p.hints!==null && typeof p.hints!=='boolean') throw new Error('invalid git observation hints');
  if (p.outcome==='independent' && p.hints!==false) throw new Error('independent outcome requires explicit no hints');
  nullable(p.self_minutes,v=>{ if (!Number.isInteger(v) || v<0 || v>1440) throw new Error('invalid self-reported minutes'); return v; });
  if('self_seconds' in p)nullable(p.self_seconds,v=>{if(!Number.isInteger(v)||v<0||v>86400||p.self_minutes!==null)throw new Error('invalid self-reported seconds');return v;});
  nullable(p.activity_type,v=>oneOf(v,['practice','review','transfer'],'invalid activity type'));
  if (p.activity_type==='transfer') { problemKey(p.transfer_from); if (p.transfer_from===p.problem_key) throw new Error('invalid transfer source'); }
  else if (p.transfer_from!==null) throw new Error('transfer source requires transfer activity');
  if (!p.trailers || typeof p.trailers!=='object' || Array.isArray(p.trailers) || Object.entries(p.trailers).some(([k,v])=>typeof k!=='string' || typeof v!=='string' || v.length>500)) throw new Error('invalid git trailers');
  return p;
}
function gitWatermark(p) {
  if (!exact(p,['repo_id','branch','last_processed_sha','status','observed','detail'])) throw new Error('invalid git watermark payload');
  identifier(p.repo_id); text(p.branch,200);
  nullable(p.last_processed_sha,v=>{ if (!/^[0-9a-f]{40}$/.test(v)) throw new Error('invalid watermark sha'); return v; });
  oneOf(p.status,['ok','history_rewritten','error','anchored'],'invalid watermark status');
  count(p.observed); nullable(p.detail,v=>text(v,1000));
  return p;
}
function practiceFeedback(p) {
  if (!exact(p,['snapshot_ref','source_key','candidate_key','verdict','channel'])) throw new Error('invalid practice feedback payload');
  identifier(p.snapshot_ref); nullable(p.source_key,problemKey); problemKey(p.candidate_key);
  oneOf(p.verdict,['related','partly','unrelated','skip'],'invalid feedback verdict');
  nullable(p.channel,v=>identifier(v));
  return p;
}
function ratingObservation(p) {
  if (!exact(p,['platform','handle','rating','rated_participations','contest_id','old_rating','rank'])) throw new Error('invalid rating observation payload');
  oneOf(p.platform,['codeforces','atcoder'],'invalid rating platform');
  nullable(p.handle,v=>text(v,100));
  if (typeof p.rating!=='number' || !Number.isFinite(p.rating) || p.rating<-1000 || p.rating>5000) throw new Error('invalid rating value');
  nullable(p.rated_participations,count); nullable(p.contest_id,v=>identifier(String(v)));
  nullable(p.old_rating,v=>{ if (typeof v!=='number' || !Number.isFinite(v)) throw new Error('invalid rating value'); return v; });
  nullable(p.rank,count);
  return p;
}
function catalogSwitch(p) {
  if (!exact(p,['from_version','to_version','catalog_sha256'])) throw new Error('invalid catalog switch payload');
  nullable(p.from_version,identifier); identifier(p.to_version);
  if (typeof p.catalog_sha256!=='string' || !/^[0-9a-f]{64}$/.test(p.catalog_sha256)) throw new Error('invalid catalog hash');
  return p;
}

// Table rank fixes the deterministic cross-table merge order (R1.7).
export const LEDGERS = {
  git_observations:{rank:3,sources:['git','manual_correction'],payload:gitObservation},
  git_watermarks:{rank:6,sources:['git','manual'],payload:gitWatermark},
  practice_feedback:{rank:4,sources:['manual'],payload:practiceFeedback},
  rating_observations:{rank:7,sources:['manual','codeforces_api'],payload:ratingObservation},
  catalog_switches:{rank:5,sources:['manual'],payload:catalogSwitch},
};
const spec = table => { const s=LEDGERS[table]; if (!s) throw new Error('unknown ledger'); return s; };

export function normalizeEntry(table,input) {
  const s=spec(table);
  if (!input || typeof input!=='object' || Array.isArray(input)) throw new Error('invalid ledger envelope');
  const value={...structuredClone(input),supersedes:input.supersedes ?? null};
  if (!exact(value,['schema_version','entry_id','learner_id','occurred_at','recorded_at','source','payload','supersedes'])) throw new Error('invalid ledger envelope');
  if (value.schema_version!==1) throw new Error('unsupported ledger version');
  identifier(value.entry_id); identifier(value.learner_id);
  oneOf(value.source,s.sources,'invalid ledger source');
  value.occurred_at=timestamp(value.occurred_at); value.recorded_at=timestamp(value.recorded_at);
  if (value.occurred_at>value.recorded_at) throw new Error('ledger entry occurs after recording');
  if (value.supersedes!==null) { identifier(value.supersedes); if (value.supersedes===value.entry_id) throw new Error('ledger entry cannot supersede itself'); }
  if (value.source==='manual_correction' && value.supersedes===null) throw new Error('manual correction requires a superseded entry');
  value.payload=s.payload(value.payload);
  return value;
}
export function preflightEntries(table,existing,incoming,learnerId) {
  identifier(learnerId);
  const known=new Map(existing.map(e=>[e.entry_id,e]));
  const children=new Set(existing.map(e=>e.supersedes).filter(Boolean));
  const inserts=[];
  for (const input of incoming) {
    const e=normalizeEntry(table,input);
    if (e.learner_id!==learnerId) throw new Error('ledger ownership mismatch');
    const prior=known.get(e.entry_id);
    if (prior) { if (canonicalJson(prior)!==canonicalJson(e)) throw new Error('conflicting ledger entry ID'); continue; }
    if (e.supersedes) {
      const target=known.get(e.supersedes);
      if (!target) throw new Error('ledger correction target missing');
      if (target.learner_id!==learnerId || target.recorded_at>e.recorded_at) throw new Error('invalid ledger correction');
      if (children.has(e.supersedes)) throw new Error('ledger correction branches');
      children.add(e.supersedes);
    }
    known.set(e.entry_id,e); inserts.push(e);
  }
  return inserts;
}
export const insertStatement=(db,table,e)=>{ spec(table); return db.prepare(`INSERT INTO ${table}
  (entry_id,learner_id,schema_version,occurred_at,recorded_at,source,payload,supersedes,canonical) VALUES (?,?,?,?,?,?,?,?,?)`)
  .bind(e.entry_id,e.learner_id,e.schema_version,e.occurred_at,e.recorded_at,e.source,canonicalJson(e.payload),e.supersedes,canonicalJson(e)); };

export async function listEntries(db,table,learnerId,{highWater=null}={}) {
  spec(table); identifier(learnerId);
  const rows=(await db.prepare(`SELECT sequence,canonical FROM ${table} WHERE learner_id=? ORDER BY sequence`).bind(learnerId).all()).results;
  return rows.filter(r=>highWater===null || r.sequence<=highWater).map(r=>({...JSON.parse(r.canonical),sequence:r.sequence}));
}
const strip=({sequence,...entry})=>entry;
// Revision guard shared with events/participation so concurrent writers serialize.
export function revisionStatements(db,learnerId,revision) {
  return [db.prepare('INSERT OR IGNORE INTO learners(learner_id) VALUES (?)').bind(learnerId),
    db.prepare('UPDATE learners SET revision=revision+1 WHERE learner_id=? AND revision=?').bind(learnerId,revision),
    db.prepare('INSERT INTO mutation_guard(ok) VALUES ((SELECT changes()))'),db.prepare('DELETE FROM mutation_guard')];
}
// Appends entries to one or more ledgers in ONE transaction (R1.5): {table: [entries]}.
export async function appendLedger(db,learnerId,batches,extraStatements=[]) {
  for (let attempt=0;attempt<3;attempt++) {
    const row=await db.prepare('SELECT revision FROM learners WHERE learner_id=?').bind(learnerId).first();
    const statements=[...revisionStatements(db,learnerId,row?.revision ?? 0)];
    const added={};
    for (const [table,entries] of Object.entries(batches)) {
      const existing=(await listEntries(db,table,learnerId)).map(strip);
      const inserts=preflightEntries(table,existing,entries,learnerId);
      added[table]=inserts.length;
      inserts.forEach(e=>statements.push(insertStatement(db,table,e)));
    }
    if (statements.length===4 && !extraStatements.length) return added;
    statements.push(...extraStatements);
    try { await db.batch(statements); return added; }
    catch (error) { if (attempt===2 || !/SQLITE|D1_|constraint|database is locked/i.test(String(error))) throw error; }
  }
}
export function activeEntries(entries,cutoff=null) {
  const eligible=cutoff ? entries.filter(e=>e.occurred_at<=cutoff && e.recorded_at<=cutoff) : entries;
  const superseded=new Set(eligible.map(e=>e.supersedes).filter(Boolean));
  return eligible.filter(e=>!superseded.has(e.entry_id)).sort((a,b)=>a.occurred_at.localeCompare(b.occurred_at) || a.entry_id.localeCompare(b.entry_id));
}
