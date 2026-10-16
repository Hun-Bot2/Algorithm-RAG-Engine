import {normalizePreferences} from './core/preferences.mjs';
import {identifier,normalizeEvent,problemKey,timerState,timestamp} from './core/events.mjs';
import {validateMerged} from './core/store.mjs';
import {canonicalJson,participationInsert,preflightParticipation} from './contest-history.mjs';

const exact=(value,fields)=>value && typeof value==='object' && !Array.isArray(value) && Object.keys(value).sort().join('|')===[...fields].sort().join('|');

function validateManualContest(value) {
  if (!exact(value,['id','title','platform','url','start_at','duration_minutes','suitable','source','fetched_at'])) throw new Error('invalid backup manual contest');
  identifier(value.id);
  if (typeof value.title!=='string' || !value.title.trim() || !['atcoder','codeforces','other'].includes(value.platform) || value.source!=='manual' || typeof value.suitable!=='boolean' || !Number.isInteger(value.duration_minutes) || value.duration_minutes<1) throw new Error('invalid backup manual contest');
  let url;
  try {url=new URL(value.url);} catch {throw new Error('invalid backup manual contest URL');}
  if (url.protocol!=='https:' || url.username || url.password) throw new Error('invalid backup manual contest URL');
  return {...structuredClone(value),start_at:timestamp(value.start_at),fetched_at:timestamp(value.fetched_at)};
}
function preflightBackup(document,learnerId) {
  if (!exact(document,['format','version','learner_id','event_document','preferences','manual_contests','participation','mission_archive','delivery_archive','delivery_reconciliation_required']) || document.format!=='adaptive-learning-backup' || document.version!==1 || document.learner_id!==learnerId || document.delivery_reconciliation_required!==true) throw new Error('invalid backup envelope');
  const eventDocument=document.event_document;
  if (!exact(eventDocument,['format','version','learner_id','events']) || eventDocument.format!=='adaptive-learning-events' || eventDocument.version!==1 || eventDocument.learner_id!==learnerId || !Array.isArray(eventDocument.events)) throw new Error('invalid backup event document');
  const events=validateMerged([],eventDocument.events.map(normalizeEvent),learnerId);
  timerState(events);
  const preferences=normalizePreferences(document.preferences);
  if (!Array.isArray(document.manual_contests) || !Array.isArray(document.participation) || !Array.isArray(document.mission_archive) || !Array.isArray(document.delivery_archive)) throw new Error('invalid backup lists');
  const manual=document.manual_contests.map(validateManualContest);
  if (new Set(manual.map(c=>c.id)).size!==manual.length) throw new Error('duplicate backup contest');
  const participation=preflightParticipation([],document.participation,learnerId);
  // Archives are inspectable evidence only; never execute or reinsert their contents.
  return {events,preferences,manual,participation};
}

export async function exportLearningBackup(db,learnerId) {
  identifier(learnerId);
  const [learnerResult,eventResult,manualResult,participationResult,missionResult,deliveryResult]=await db.batch([
    db.prepare('SELECT preferences FROM learners WHERE learner_id=?').bind(learnerId),
    db.prepare('SELECT canonical FROM events WHERE learner_id=? ORDER BY sequence').bind(learnerId),
    db.prepare('SELECT payload FROM manual_contests WHERE learner_id=? ORDER BY contest_id').bind(learnerId),
    db.prepare('SELECT canonical FROM contest_participation WHERE learner_id=? ORDER BY sequence').bind(learnerId),
    db.prepare('SELECT local_date,payload FROM mission_snapshots WHERE learner_id=? ORDER BY local_date').bind(learnerId),
    db.prepare(`SELECT local_date,channel_id,uniqueness_key,mission_hash,state,retry_at,scheduled_message_id,slack_ts,last_error_code,created_at,updated_at
      FROM delivery_outbox WHERE learner_id=? ORDER BY local_date,channel_id`).bind(learnerId),
  ]);
  const preferences=normalizePreferences(learnerResult.results[0] ? JSON.parse(learnerResult.results[0].preferences) : {});
  return {format:'adaptive-learning-backup',version:1,learner_id:learnerId,
    event_document:{format:'adaptive-learning-events',version:1,learner_id:learnerId,events:eventResult.results.map(row=>JSON.parse(row.canonical))},
    preferences,manual_contests:manualResult.results.map(row=>JSON.parse(row.payload)),participation:participationResult.results.map(row=>JSON.parse(row.canonical)),
    mission_archive:missionResult.results.map(row=>({local_date:row.local_date,payload:JSON.parse(row.payload)})),delivery_archive:deliveryResult.results,delivery_reconciliation_required:true};
}

export async function importLearningBackup(db,learnerId,document) {
  identifier(learnerId);
  const {events,preferences,manual,participation}=preflightBackup(document,learnerId);
  const existing=await db.prepare('SELECT revision FROM learners WHERE learner_id=?').bind(learnerId).first();
  if (existing?.revision!==undefined && existing.revision!==0) throw new Error('restore target is not empty');
  const statements=[
    db.prepare('INSERT OR IGNORE INTO learners(learner_id) VALUES (?)').bind(learnerId),
    db.prepare(`UPDATE learners SET revision=revision+1,preferences=?,delivery_reconciliation_required=1
      WHERE learner_id=? AND revision=0
      AND NOT EXISTS (SELECT 1 FROM events WHERE learner_id=?)
      AND NOT EXISTS (SELECT 1 FROM manual_contests WHERE learner_id=?)
      AND NOT EXISTS (SELECT 1 FROM contest_participation WHERE learner_id=?)
      AND NOT EXISTS (SELECT 1 FROM mission_snapshots WHERE learner_id=?)
      AND NOT EXISTS (SELECT 1 FROM delivery_outbox WHERE learner_id=?)
      AND NOT EXISTS (SELECT 1 FROM provider_cache WHERE learner_id=?)`).bind(canonicalJson(preferences),learnerId,learnerId,learnerId,learnerId,learnerId,learnerId,learnerId),
    db.prepare('INSERT INTO mutation_guard(ok) VALUES ((SELECT changes()))'),
    db.prepare('DELETE FROM mutation_guard'),
  ];
  for (const e of events) statements.push(db.prepare(`INSERT INTO events(event_id,learner_id,schema_version,occurred_at,recorded_at,kind,source,payload,supersedes,canonical)
    VALUES (?,?,?,?,?,?,?,?,?,?)`).bind(e.event_id,e.learner_id,e.schema_version,e.occurred_at,e.recorded_at,e.kind,e.source,canonicalJson(e.payload),e.supersedes,canonicalJson(e)));
  for (const c of manual) statements.push(db.prepare('INSERT INTO manual_contests(learner_id,contest_id,payload) VALUES (?,?,?)').bind(learnerId,c.id,canonicalJson(c)));
  for (const p of participation) statements.push(participationInsert(db,p));
  try {await db.batch(statements);} catch (error) {throw new Error('atomic backup restore failed',{cause:error});}
  return {events:events.length,manual_contests:manual.length,participation:participation.length,delivery_reconciliation_required:true};
}
