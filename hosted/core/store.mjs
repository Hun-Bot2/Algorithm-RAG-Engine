import {defaultPreferences,normalizePreferences} from './preferences.mjs';
export {defaultPreferences} from './preferences.mjs';
import {activeEvents, identifier, normalizeEvent, timerState} from './events.mjs';

const canonicalJson = value => JSON.stringify(value, (_, v) => v && typeof v === 'object' && !Array.isArray(v) ? Object.fromEntries(Object.entries(v).sort(([a],[b])=>a.localeCompare(b))) : v);
const all = async (db, sql, ...args) => (await db.prepare(sql).bind(...args).all()).results;
const first = async (db, sql, ...args) => db.prepare(sql).bind(...args).first();

export async function learnerSnapshot(db, learnerId) {
  identifier(learnerId);
  const row = await first(db, 'SELECT revision, preferences FROM learners WHERE learner_id=?', learnerId);
  const events = await listEvents(db, learnerId);
  return {revision:row?.revision ?? 0, preferences:row ? normalizePreferences(JSON.parse(row.preferences)) : defaultPreferences(), events};
}
export async function listEvents(db, learnerId) {
  identifier(learnerId);
  const rows = await all(db, 'SELECT canonical FROM events WHERE learner_id=? ORDER BY sequence', learnerId);
  return rows.map(row => JSON.parse(row.canonical));
}
export function validateMerged(existing, incoming, learnerId) {
  const known = new Map(existing.map(e => [e.event_id,e]));
  const child = new Set(existing.map(e => e.supersedes).filter(Boolean));
  const inserts = [];
  for (const e of incoming) {
    if (e.learner_id !== learnerId) throw new Error('learner ownership mismatch');
    const old = known.get(e.event_id);
    if (old) { if (canonicalJson(old) !== canonicalJson(e)) throw new Error('conflicting event ID'); else continue; }
    if (e.supersedes) {
      const target = known.get(e.supersedes);
      if (!target) throw new Error('correction target missing');
      if (target.learner_id !== learnerId || target.kind !== e.kind || target.recorded_at > e.recorded_at || target.payload.interval_id !== e.payload.interval_id) throw new Error('invalid correction ownership/type/time');
      if (child.has(e.supersedes)) throw new Error('correction target already superseded');
      child.add(e.supersedes);
    }
    known.set(e.event_id,e); inserts.push(e);
  }
  timerState([...existing,...inserts]);
  return inserts;
}
function revisionStatements(db, learnerId, revision) {
  return [
    db.prepare('INSERT OR IGNORE INTO learners(learner_id) VALUES (?)').bind(learnerId),
    db.prepare('UPDATE learners SET revision=revision+1 WHERE learner_id=? AND revision=?').bind(learnerId,revision),
    db.prepare('INSERT INTO mutation_guard(ok) VALUES ((SELECT changes()))'),
    db.prepare('DELETE FROM mutation_guard'),
  ];
}
export async function appendEvents(db, learnerId, inputs) {
  identifier(learnerId);
  if (!Array.isArray(inputs)) throw new Error('events must be a list');
  const incoming = inputs.map(normalizeEvent);
  for (let attempt=0; attempt<3; attempt++) {
    const {revision,events} = await learnerSnapshot(db,learnerId);
    const inserts = validateMerged(events,incoming,learnerId);
    if (!inserts.length) return 0;
    const statements = revisionStatements(db,learnerId,revision);
    for (const e of inserts) statements.push(db.prepare('INSERT INTO events(event_id,learner_id,schema_version,occurred_at,recorded_at,kind,source,payload,supersedes,canonical) VALUES (?,?,?,?,?,?,?,?,?,?)').bind(e.event_id,e.learner_id,e.schema_version,e.occurred_at,e.recorded_at,e.kind,e.source,canonicalJson(e.payload),e.supersedes,canonicalJson(e)));
    try { await db.batch(statements); return inserts.length; }
    catch (error) {
      if (attempt===2 || !/SQLITE|D1_|constraint|database is locked/i.test(String(error))) throw error;
    }
  }
}
export async function exportDocument(db, learnerId) {
  identifier(learnerId);
  return {format:'adaptive-learning-events', version:1, learner_id:learnerId, events:await listEvents(db,learnerId)};
}
export async function importDocument(db, learnerId, document) {
  if (!document || typeof document !== 'object' || Array.isArray(document) || Object.keys(document).sort().join('|') !== 'events|format|learner_id|version' || document.format !== 'adaptive-learning-events' || document.version !== 1 || document.learner_id !== learnerId || !Array.isArray(document.events)) throw new Error('invalid export envelope');
  return appendEvents(db,learnerId,document.events);
}
export async function getPreferences(db, learnerId) { return (await learnerSnapshot(db,learnerId)).preferences; }
export async function putPreferences(db, learnerId, input) {
  const snap = await learnerSnapshot(db,learnerId);
  const next = normalizePreferences({...snap.preferences,...input});
  // Validate the patch itself so non-object and unexpected fields fail closed.
  normalizePreferences(input);
  await db.batch([...revisionStatements(db,learnerId,snap.revision),db.prepare('UPDATE learners SET preferences=? WHERE learner_id=?').bind(canonicalJson(next),learnerId)]);
  return next;
}
export async function getMission(db, learnerId, date) {
  const row = await first(db,'SELECT payload FROM mission_snapshots WHERE learner_id=? AND local_date=?',learnerId,date);
  return row ? JSON.parse(row.payload) : null;
}
export async function saveMission(db, learnerId, date, mission, revision) {
  try { await db.batch([...revisionStatements(db,learnerId,revision),db.prepare('INSERT OR IGNORE INTO mission_snapshots(learner_id,local_date,payload) VALUES (?,?,?)').bind(learnerId,date,canonicalJson(mission))]); }
  catch (error) { const winner=await getMission(db,learnerId,date); if (winner) return winner; throw error; }
  return getMission(db,learnerId,date);
}
export async function listManualContests(db, learnerId) {
  return (await all(db,'SELECT payload FROM manual_contests WHERE learner_id=? ORDER BY contest_id',learnerId)).map(row=>JSON.parse(row.payload));
}
export async function putManualContest(db, learnerId, contest) {
  const {revision}=await learnerSnapshot(db,learnerId);
  await db.batch([...revisionStatements(db,learnerId,revision),db.prepare('INSERT INTO manual_contests(learner_id,contest_id,payload) VALUES (?,?,?) ON CONFLICT(learner_id,contest_id) DO UPDATE SET payload=excluded.payload').bind(learnerId,contest.id,canonicalJson(contest))]);
  return contest;
}
