import test from 'node:test';
import assert from 'node:assert/strict';
import {createDatabase} from './helpers/d1.mjs';
import {appendEvents} from '../core/store.mjs';
import {appendParticipation} from '../contest-history.mjs';
import {LEDGERS,appendLedger,listEntries} from '../mastery/side-ledger.mjs';
import {LEDGER_TABLES,highWaterMarks,readLedger,saveSnapshot,getSnapshot,saveWeeklyReport} from '../mastery/ledger.mjs';

const T = (d,h='00:00:00') => `2026-10-${d}T${h}.000000Z`;
const SHA = c => c.repeat(40);
export const payloads = {
  git_observations:{repo_id:'solutions',sha:SHA('a'),path:'leetcode/two-sum/a.py',problem_key:'leetcode:two-sum',outcome:'reported_completion',hints:null,self_minutes:null,activity_type:null,transfer_from:null,trailers:{}},
  git_watermarks:{repo_id:'solutions',branch:'main',last_processed_sha:SHA('a'),status:'ok',observed:1,detail:null},
  practice_feedback:{snapshot_ref:'plan-2026-10-12',source_key:'leetcode:two-sum',candidate_key:'codeforces:4A',verdict:'related',channel:'concept'},
  rating_observations:{platform:'codeforces',handle:null,rating:1234,rated_participations:5,contest_id:null,old_rating:null,rank:null},
  catalog_switches:{from_version:null,to_version:'catalog-v1',catalog_sha256:'b'.repeat(64)},
};
const entry = (table,id,extra={}) => ({schema_version:1,entry_id:id,learner_id:'owner',occurred_at:T('12'),recorded_at:T('12'),source:LEDGERS[table].sources[0],payload:structuredClone(payloads[table]),supersedes:null,...extra});

for (const table of Object.keys(LEDGERS)) {
  test(`ledger contract: ${table}`, async context => {
    const db=createDatabase(); context.after(()=>db.close());
    const base=entry(table,'e1');
    assert.equal((await appendLedger(db,'owner',{[table]:[base]}))[table],1);
    assert.equal((await appendLedger(db,'owner',{[table]:[base]}))[table],0,'identical re-insert is a no-op');
    const changed=structuredClone(base); changed.recorded_at=T('13');
    await assert.rejects(appendLedger(db,'owner',{[table]:[changed]}),/conflicting ledger entry ID/);
    await assert.rejects(appendLedger(db,'other',{[table]:[base]}),/ownership/);
    await assert.rejects(appendLedger(db,'owner',{[table]:[entry(table,'x',{supersedes:'missing'})]}),/target missing/);
    const correction=entry(table,'e2',{supersedes:'e1',recorded_at:T('14'),...(table==='git_observations'?{source:'manual_correction'}:{})});
    assert.equal((await appendLedger(db,'owner',{[table]:[correction]}))[table],1);
    await assert.rejects(appendLedger(db,'owner',{[table]:[entry(table,'e3',{supersedes:'e1',recorded_at:T('14'),...(table==='git_observations'?{source:'manual_correction'}:{})})]}),/branches/);
    await assert.rejects(appendLedger(db,'owner',{[table]:[entry(table,'e4',{occurred_at:T('15'),recorded_at:T('14')})]}),/after recording/);
    await assert.rejects(appendLedger(db,'owner',{[table]:[{...entry(table,'e5'),extra:1}]}),/envelope/);
    await assert.rejects(appendLedger(db,'owner',{[table]:[entry(table,'e6',{payload:{...payloads[table],unexpected:true}})]}),/payload/);
    assert.throws(()=>db.sqlite.prepare(`UPDATE ${table} SET payload='{}'`).run(),/append-only/);
    assert.throws(()=>db.sqlite.prepare(`DELETE FROM ${table}`).run(),/append-only/);
    const ledger=await readLedger(db,'owner');
    assert.deepEqual(ledger[table].map(e=>e.entry_id),['e2'],'correction replaces its target');
  });
}

test('git observation rules: independent needs explicit no hints; corrections need a target', async context => {
  const db=createDatabase(); context.after(()=>db.close());
  await assert.rejects(appendLedger(db,'owner',{git_observations:[entry('git_observations','g',{payload:{...payloads.git_observations,outcome:'independent',hints:null}})]}),/explicit no hints/);
  await assert.rejects(appendLedger(db,'owner',{git_observations:[entry('git_observations','g',{source:'manual_correction'})]}),/requires a superseded entry/);
  await assert.rejects(appendLedger(db,'owner',{git_observations:[entry('git_observations','g',{payload:{...payloads.git_observations,sha:'xyz'}})]}),/sha/);
});

test('multi-ledger append is one transaction', async context => {
  const db=createDatabase(); context.after(()=>db.close());
  await assert.rejects(appendLedger(db,'owner',{git_observations:[entry('git_observations','ok')],git_watermarks:[entry('git_watermarks','bad',{payload:{...payloads.git_watermarks,status:'nope'}})]}));
  assert.equal((await listEntries(db,'git_observations','owner')).length,0,'no partial write');
  const added=await appendLedger(db,'owner',{git_observations:[entry('git_observations','ok')],git_watermarks:[entry('git_watermarks','w')]});
  assert.deepEqual(added,{git_observations:1,git_watermarks:1});
});

test('ledger merge order, cutoff and high-water reproduction', async context => {
  const db=createDatabase(); context.after(()=>db.close());
  const attempt=(id,at,outcome='independent')=>({schema_version:1,event_id:id,learner_id:'owner',occurred_at:at,recorded_at:at,kind:'attempt',source:'manual',payload:{problem_key:'leetcode:two-sum',outcome,hints:outcome==='independent'?false:null},supersedes:null});
  await appendEvents(db,'owner',[attempt('ev-b',T('12'))]);
  await appendLedger(db,'owner',{git_observations:[entry('git_observations','ev-a')]});
  await appendParticipation(db,'owner',{schema_version:1,entry_id:'p1',learner_id:'owner',contest_id:'abc400',mode:'virtual',occurred_at:T('11'),recorded_at:T('11'),outcome:'completed',confirmed_minutes:null,source:'manual',supersedes:null});
  const marks=await highWaterMarks(db,'owner');
  assert.deepEqual(Object.keys(marks),LEDGER_TABLES);
  const first=await readLedger(db,'owner');
  // same occurred_at: events (rank 1) before git_observations (rank 3) regardless of id order
  assert.deepEqual(first.attempts.map(a=>[a.id,a.evidence_source]),[['ev-b','manual'],['ev-a','git']]);
  assert.equal(first.participation.length,1);
  await appendEvents(db,'owner',[attempt('late',T('12','01:00:00'),'failed')]);
  const reproduced=await readLedger(db,'owner',{highWater:marks});
  assert.deepEqual(reproduced.attempts.map(a=>a.id),first.attempts.map(a=>a.id),'high-water replay ignores later rows');
  const cut=await readLedger(db,'owner',{cutoff:T('11','12:00:00')});
  assert.equal(cut.attempts.length,0); assert.equal(cut.participation.length,1);
});

test('snapshots and weekly reports are immutable', async context => {
  const db=createDatabase(); context.after(()=>db.close());
  const saved=await saveSnapshot(db,'owner','plan','2026-10-12',{a:1});
  assert.deepEqual(saved,{a:1});
  assert.deepEqual(await saveSnapshot(db,'owner','plan','2026-10-12',{a:2}),{a:1},'first write wins');
  assert.deepEqual(await getSnapshot(db,'owner','plan','2026-10-12'),{a:1});
  assert.throws(()=>db.sqlite.prepare('UPDATE recommendation_snapshots SET payload=?').run('{}'),/immutable/);
  await saveWeeklyReport(db,'owner','2026-10-12',{w:1});
  assert.throws(()=>db.sqlite.prepare('DELETE FROM weekly_reports').run(),/immutable/);
});

import {exportLearningBackup,importLearningBackup} from '../backup.mjs';
test('backup v2 round-trips side ledgers and Git registrations; v1 documents still restore', async context => {
  const source=createDatabase(), target=createDatabase(), legacy=createDatabase(), busy=createDatabase();
  context.after(()=>[source,target,legacy,busy].forEach(d=>d.close()));
  await appendLedger(source,'owner',{git_observations:[entry('git_observations','g1')],rating_observations:[entry('rating_observations','r1')]});
  source.sqlite.prepare('INSERT INTO git_repositories VALUES (?,?,?,?,?,?)').run('owner','solutions','main','/tmp/x','git-convention-v1',T('12'));
  await saveSnapshot(source,'owner','plan','2026-10-12',{a:1});
  const backup=await exportLearningBackup(source,'owner');
  assert.equal(backup.version,2);
  assert.equal(backup.mastery_archive.recommendation_snapshots.length,1);
  const report=await importLearningBackup(target,'owner',backup);
  assert.deepEqual(report.mastery,{git_observations:1,rating_observations:1,git_repositories:1});
  const restored=await exportLearningBackup(target,'owner');
  assert.deepEqual(restored.mastery_ledgers,backup.mastery_ledgers);
  assert.deepEqual(restored.git_repositories,backup.git_repositories);
  assert.deepEqual(restored.mastery_archive.recommendation_snapshots,backup.mastery_archive.recommendation_snapshots,'restored snapshots remain visible through export in the original backup format');
  const archived=target.sqlite.prepare('SELECT kind,local_date,payload FROM recommendation_snapshots WHERE learner_id=?').all('owner');
  assert.equal(archived.length,1);
  assert.match(archived[0].kind,/^archive-/);
  assert.deepEqual(JSON.parse(archived[0].payload),{archive_type:'recommendation_snapshot',original_kind:'plan',original_payload:{a:1}});
  assert.deepEqual(restored.mastery_archive.weekly_reports,[], 'weekly report archive behavior remains unchanged');
  const {mastery_ledgers,git_repositories,mastery_archive,...v1}=await exportLearningBackup(createDatabase(),'owner');
  assert.deepEqual(await importLearningBackup(legacy,'owner',{...v1,version:1}),{events:0,manual_contests:0,participation:0,delivery_reconciliation_required:true});
  await assert.rejects(importLearningBackup(legacy,'owner',{...v1,version:1,mastery_ledgers:{}}),/envelope/);
  await appendLedger(busy,'owner',{practice_feedback:[entry('practice_feedback','f1')]});
  await assert.rejects(importLearningBackup(busy,'owner',backup),/not empty|atomic/);
});
