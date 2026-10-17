import {normalizePreferences} from './core/preferences.mjs';
import {identifier,normalizeEvent,problemKey,timerState,timestamp} from './core/events.mjs';
import {validateMerged} from './core/store.mjs';
import {canonicalJson,participationInsert,preflightParticipation} from './contest-history.mjs';
import {LEDGERS,insertStatement,preflightEntries} from './mastery/side-ledger.mjs';

// Backup v2 (mastery-system MS-206) adds side ledgers, Git registrations and an inspect-only
// snapshot archive. v1 documents still restore; their mastery sections are treated as empty.
const V1_FIELDS=['format','version','learner_id','event_document','preferences','manual_contests','participation','mission_archive','delivery_archive','delivery_reconciliation_required'];
const V2_FIELDS=[...V1_FIELDS,'mastery_ledgers','git_repositories','mastery_archive'];
function validateRepository(value,learnerId) {
  if (!exact(value,['learner_id','repo_id','branch','path_hint','convention_version','registered_at']) || value.learner_id!==learnerId) throw new Error('invalid backup git repository');
  identifier(value.repo_id);
  for (const field of ['branch','path_hint','convention_version']) if (typeof value[field]!=='string' || !value[field].trim()) throw new Error('invalid backup git repository');
  return {...value,registered_at:timestamp(value.registered_at)};
}

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
  const fields=document?.version===2 ? V2_FIELDS : V1_FIELDS;
  if (!exact(document,fields) || document.format!=='adaptive-learning-backup' || ![1,2].includes(document.version) || document.learner_id!==learnerId || document.delivery_reconciliation_required!==true) throw new Error('invalid backup envelope');
  const eventDocument=document.event_document;
  if (!exact(eventDocument,['format','version','learner_id','events']) || eventDocument.format!=='adaptive-learning-events' || eventDocument.version!==1 || eventDocument.learner_id!==learnerId || !Array.isArray(eventDocument.events)) throw new Error('invalid backup event document');
  const events=validateMerged([],eventDocument.events.map(normalizeEvent),learnerId);
  timerState(events);
  const preferences=normalizePreferences(document.preferences);
  if (!Array.isArray(document.manual_contests) || !Array.isArray(document.participation) || !Array.isArray(document.mission_archive) || !Array.isArray(document.delivery_archive)) throw new Error('invalid backup lists');
  const manual=document.manual_contests.map(validateManualContest);
  if (new Set(manual.map(c=>c.id)).size!==manual.length) throw new Error('duplicate backup contest');
  const participation=preflightParticipation([],document.participation,learnerId);
  const ledgers={}, repositories=[];
  if (document.version===2) {
    if (!document.mastery_ledgers || typeof document.mastery_ledgers!=='object' || Array.isArray(document.mastery_ledgers) || Object.keys(document.mastery_ledgers).some(t=>!LEDGERS[t])) throw new Error('invalid backup mastery ledgers');
    for (const [table,entries] of Object.entries(document.mastery_ledgers)) {
      if (!Array.isArray(entries)) throw new Error('invalid backup mastery ledgers');
      ledgers[table]=preflightEntries(table,[],entries,learnerId);
    }
    if (!Array.isArray(document.git_repositories) || !document.mastery_archive || typeof document.mastery_archive!=='object') throw new Error('invalid backup lists');
    repositories.push(...document.git_repositories.map(r=>validateRepository(r,learnerId)));
    if (new Set(repositories.map(r=>r.repo_id)).size!==repositories.length) throw new Error('duplicate backup git repository');
  }
  // Archives are inspectable evidence only; never execute or reinsert their contents.
  return {events,preferences,manual,participation,ledgers,repositories};
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
  const mastery_ledgers={};
  for (const table of Object.keys(LEDGERS)) mastery_ledgers[table]=(await db.prepare(`SELECT canonical FROM ${table} WHERE learner_id=? ORDER BY sequence`).bind(learnerId).all()).results.map(row=>JSON.parse(row.canonical));
  const git_repositories=(await db.prepare('SELECT learner_id,repo_id,branch,path_hint,convention_version,registered_at FROM git_repositories WHERE learner_id=? ORDER BY repo_id').bind(learnerId).all()).results;
  const mastery_archive={recommendation_snapshots:(await db.prepare('SELECT kind,local_date,payload FROM recommendation_snapshots WHERE learner_id=? ORDER BY kind,local_date').bind(learnerId).all()).results.map(r=>({...r,payload:JSON.parse(r.payload)})),
    weekly_reports:(await db.prepare('SELECT week_start,payload FROM weekly_reports WHERE learner_id=? ORDER BY week_start').bind(learnerId).all()).results.map(r=>({...r,payload:JSON.parse(r.payload)}))};
  return {format:'adaptive-learning-backup',version:2,learner_id:learnerId,mastery_ledgers,git_repositories,mastery_archive,
    event_document:{format:'adaptive-learning-events',version:1,learner_id:learnerId,events:eventResult.results.map(row=>JSON.parse(row.canonical))},
    preferences,manual_contests:manualResult.results.map(row=>JSON.parse(row.payload)),participation:participationResult.results.map(row=>JSON.parse(row.canonical)),
    mission_archive:missionResult.results.map(row=>({local_date:row.local_date,payload:JSON.parse(row.payload)})),delivery_archive:deliveryResult.results,delivery_reconciliation_required:true};
}

export async function importLearningBackup(db,learnerId,document) {
  identifier(learnerId);
  const {events,preferences,manual,participation,ledgers,repositories}=preflightBackup(document,learnerId);
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
      AND NOT EXISTS (SELECT 1 FROM provider_cache WHERE learner_id=?)
      ${Object.keys(LEDGERS).map(t=>`AND NOT EXISTS (SELECT 1 FROM ${t} WHERE learner_id=?)`).join('\n      ')}
      AND NOT EXISTS (SELECT 1 FROM git_repositories WHERE learner_id=?)
      AND NOT EXISTS (SELECT 1 FROM recommendation_snapshots WHERE learner_id=?)
      AND NOT EXISTS (SELECT 1 FROM weekly_reports WHERE learner_id=?)`).bind(canonicalJson(preferences),learnerId,learnerId,learnerId,learnerId,learnerId,learnerId,learnerId,...Object.keys(LEDGERS).map(()=>learnerId),learnerId,learnerId,learnerId),
    db.prepare('INSERT INTO mutation_guard(ok) VALUES ((SELECT changes()))'),
    db.prepare('DELETE FROM mutation_guard'),
  ];
  for (const e of events) statements.push(db.prepare(`INSERT INTO events(event_id,learner_id,schema_version,occurred_at,recorded_at,kind,source,payload,supersedes,canonical)
    VALUES (?,?,?,?,?,?,?,?,?,?)`).bind(e.event_id,e.learner_id,e.schema_version,e.occurred_at,e.recorded_at,e.kind,e.source,canonicalJson(e.payload),e.supersedes,canonicalJson(e)));
  for (const c of manual) statements.push(db.prepare('INSERT INTO manual_contests(learner_id,contest_id,payload) VALUES (?,?,?)').bind(learnerId,c.id,canonicalJson(c)));
  for (const p of participation) statements.push(participationInsert(db,p));
  for (const r of repositories) statements.push(db.prepare('INSERT INTO git_repositories(learner_id,repo_id,branch,path_hint,convention_version,registered_at) VALUES (?,?,?,?,?,?)').bind(r.learner_id,r.repo_id,r.branch,r.path_hint,r.convention_version,r.registered_at));
  for (const [table,entries] of Object.entries(ledgers)) for (const e of entries) statements.push(insertStatement(db,table,e));
  try {await db.batch(statements);} catch (error) {throw new Error('atomic backup restore failed',{cause:error});}
  const report={events:events.length,manual_contests:manual.length,participation:participation.length,delivery_reconciliation_required:true};
  const mastery=Object.fromEntries(Object.entries(ledgers).filter(([,e])=>e.length).map(([t,e])=>[t,e.length]));
  if (repositories.length) mastery.git_repositories=repositories.length;
  return Object.keys(mastery).length ? {...report,mastery} : report;
}
