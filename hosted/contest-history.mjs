import {identifier,timestamp} from './core/events.mjs';

export const canonicalJson = value => JSON.stringify(value,(_,v)=>v && typeof v==='object' && !Array.isArray(v) ? Object.fromEntries(Object.entries(v).sort(([a],[b])=>a.localeCompare(b))) : v);
const exact = (value,fields) => value && typeof value==='object' && !Array.isArray(value) && Object.keys(value).sort().join('|')===[...fields].sort().join('|');

export function normalizeParticipation(input) {
  if (!input || typeof input!=='object' || Array.isArray(input)) throw new Error('invalid participation envelope');
  const value={...input,supersedes:input.supersedes ?? null};
  if (!exact(value,['schema_version','entry_id','learner_id','contest_id','mode','occurred_at','recorded_at','outcome','confirmed_minutes','source','supersedes'])) throw new Error('invalid participation envelope');
  if (value.schema_version!==1 || !Number.isInteger(value.schema_version)) throw new Error('unsupported participation version');
  identifier(value.entry_id);identifier(value.learner_id);identifier(value.contest_id);
  if (!['live','virtual','upsolve'].includes(value.mode) || !['completed','partial','abandoned','unknown'].includes(value.outcome) || value.source!=='manual') throw new Error('invalid participation evidence');
  value.occurred_at=timestamp(value.occurred_at);value.recorded_at=timestamp(value.recorded_at);
  if (value.occurred_at>value.recorded_at) throw new Error('participation occurs after recording');
  if (value.confirmed_minutes!==null && (typeof value.confirmed_minutes!=='number' || !Number.isFinite(value.confirmed_minutes) || value.confirmed_minutes<0)) throw new Error('invalid confirmed minutes');
  if (value.supersedes!==null) {identifier(value.supersedes);if(value.supersedes===value.entry_id) throw new Error('participation cannot supersede itself');}
  return value;
}
export function activeParticipation(entries,cutoff=null) {
  const eligible=cutoff?entries.filter(e=>e.occurred_at<=cutoff && e.recorded_at<=cutoff):entries;
  const superseded=new Set(eligible.map(e=>e.supersedes).filter(Boolean));
  return eligible.filter(e=>!superseded.has(e.entry_id)).sort((a,b)=>a.occurred_at.localeCompare(b.occurred_at) || a.entry_id.localeCompare(b.entry_id));
}
export function preflightParticipation(existing,incoming,learnerId) {
  identifier(learnerId);
  const known=new Map(existing.map(e=>[e.entry_id,e]));
  const children=new Set(existing.map(e=>e.supersedes).filter(Boolean));
  const inserts=[];
  for (const input of incoming) {
    const e=normalizeParticipation(input);
    if (e.learner_id!==learnerId) throw new Error('participation ownership mismatch');
    const prior=known.get(e.entry_id);
    if (prior) {if (canonicalJson(prior)!==canonicalJson(e)) throw new Error('conflicting participation ID');continue;}
    if (e.supersedes) {
      const target=known.get(e.supersedes);
      if (!target) throw new Error('participation correction target missing');
      if (target.learner_id!==learnerId || target.contest_id!==e.contest_id || target.recorded_at>e.recorded_at) throw new Error('invalid participation correction');
      if (children.has(e.supersedes)) throw new Error('participation correction branches');
      children.add(e.supersedes);
    }
    known.set(e.entry_id,e);inserts.push(e);
  }
  return inserts;
}
export const participationInsert = (db,e) => db.prepare(`INSERT INTO contest_participation
  (entry_id,learner_id,schema_version,contest_id,mode,occurred_at,recorded_at,outcome,confirmed_minutes,source,supersedes,canonical)
  VALUES (?,?,?,?,?,?,?,?,?,?,?,?)`).bind(e.entry_id,e.learner_id,e.schema_version,e.contest_id,e.mode,e.occurred_at,e.recorded_at,e.outcome,e.confirmed_minutes,e.source,e.supersedes,canonicalJson(e));

export async function listParticipation(db,learnerId,{active=false,cutoff=null}={}) {
  identifier(learnerId);
  const rows=(await db.prepare('SELECT canonical FROM contest_participation WHERE learner_id=? ORDER BY sequence').bind(learnerId).all()).results;
  const entries=rows.map(row=>JSON.parse(row.canonical));
  return active?activeParticipation(entries,cutoff):entries;
}
export async function appendParticipation(db,learnerId,input) {
  const incoming=normalizeParticipation(input);
  for (let attempt=0;attempt<3;attempt++) {
    const row=await db.prepare('SELECT revision FROM learners WHERE learner_id=?').bind(learnerId).first();
    const existing=await listParticipation(db,learnerId);
    const inserts=preflightParticipation(existing,[incoming],learnerId);
    if (!inserts.length) return 0;
    const statements=[db.prepare('INSERT OR IGNORE INTO learners(learner_id) VALUES (?)').bind(learnerId),db.prepare('UPDATE learners SET revision=revision+1 WHERE learner_id=? AND revision=?').bind(learnerId,row?.revision ?? 0),db.prepare('INSERT INTO mutation_guard(ok) VALUES ((SELECT changes()))'),db.prepare('DELETE FROM mutation_guard'),participationInsert(db,incoming)];
    try {await db.batch(statements);return 1;}
    catch(error) {if(attempt===2 || !/SQLITE|D1_|constraint|database is locked/i.test(String(error))) throw error;}
  }
}
