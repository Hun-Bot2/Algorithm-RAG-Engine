import test from 'node:test';
import assert from 'node:assert/strict';
import {createDatabase} from './helpers/d1.mjs';
import {appendEvents,putManualContest,putPreferences} from '../core/store.mjs';
import {appendParticipation,listParticipation,activeParticipation} from '../contest-history.mjs';
import {exportLearningBackup,importLearningBackup} from '../backup.mjs';
import {saveSnapshot} from '../mastery/ledger.mjs';
import {prepareMission} from '../core/policy.mjs';
import {ensureDelivery,listDeliveries} from '../notifications.mjs';
import {runDispatcher,deliveryTimes} from '../dispatcher.mjs';

const review={status:'approved',reviewer:'fixture',evidence:'fixture'};
const curriculum={schema_version:1,curriculum_version:'test',problems:[{problem_key:'leetcode:two-sum',canonical_url:'https://leetcode.com/problems/two-sum/',native_difficulty:'Easy',access_state:'free',concepts:['arrays'],review}],relationships:[]};
const stamp='2026-10-16T00:00:00Z';
const event=(id,owner='owner')=>({schema_version:1,event_id:id,learner_id:owner,occurred_at:stamp,recorded_at:stamp,kind:'attempt',source:'manual',payload:{problem_key:'leetcode:two-sum',outcome:'assisted',hints:true},supersedes:null});
const entry=(id,owner='owner',extras={})=>({schema_version:1,entry_id:id,learner_id:owner,contest_id:'atcoder:abc001',mode:'virtual',occurred_at:stamp,recorded_at:stamp,outcome:'partial',confirmed_minutes:null,source:'manual',supersedes:null,...extras});
const contest={id:'abc001',title:'AtCoder Beginner Contest',platform:'atcoder',url:'https://atcoder.jp/contests/abc001',start_at:'2026-10-16T12:00:00.000000Z',duration_minutes:100,suitable:true,source:'manual',fetched_at:'2026-10-15T20:00:00.000000Z'};
const enabled={DELIVERY_ENABLED:'true',DELIVERY_LEARNER_ID:'owner',SLACK_BOT_TOKEN:'test-token',SLACK_CHANNEL_ID:'D_TEST'};

test('participation is owner scoped, append-only, corrected without invented minutes',async context=>{
  const db=createDatabase();context.after(()=>db.close());
  assert.equal(await appendParticipation(db,'owner',entry('one')),1);
  assert.equal(await appendParticipation(db,'owner',entry('one')),0);
  await assert.rejects(appendParticipation(db,'other',entry('one')),/ownership/);
  const correction=entry('two','owner',{outcome:'completed',confirmed_minutes:45,supersedes:'one',recorded_at:'2026-10-16T01:00:00Z'});
  await appendParticipation(db,'owner',correction);
  assert.deepEqual((await listParticipation(db,'owner')).map(e=>e.entry_id),['one','two']);
  assert.equal((await listParticipation(db,'owner',{active:true}))[0].confirmed_minutes,45);
  assert.equal(activeParticipation(await listParticipation(db,'owner'),'2026-10-16T00:30:00.000000Z')[0].confirmed_minutes,null);
  await assert.rejects(appendParticipation(db,'owner',entry('branch','owner',{supersedes:'one',recorded_at:'2026-10-16T02:00:00Z'})),/branches/);
  await assert.rejects(appendParticipation(db,'owner',entry('bad','owner',{confirmed_minutes:-1})),/confirmed minutes/);
  await assert.rejects(db.prepare("DELETE FROM contest_participation").run(),/append-only/);
});

test('backup round trip restores evidence, settings, manual contests and participation atomically',async context=>{
  const source=createDatabase(),target=createDatabase();context.after(()=>{source.close();target.close();});
  await putPreferences(source,'owner',{weekday_minutes:45,target_keys:['leetcode:two-sum']});
  await appendEvents(source,'owner',[event('e1')]);
  await putManualContest(source,'owner',contest);
  await appendParticipation(source,'owner',entry('p1'));
  const mission=await prepareMission({db:source,learnerId:'owner',curriculum,now:deliveryTimes('2026-10-16').cutoff});
  await ensureDelivery(source,{learnerId:'owner',mission,curriculum,channelId:'D_TEST',now:new Date('2026-10-15T20:45:00Z')});
  const backup=await exportLearningBackup(source,'owner');
  assert.equal(backup.delivery_archive.length,1);
  assert.equal(backup.mission_archive.length,1);
  const report=await importLearningBackup(target,'owner',backup);
  assert.deepEqual(report,{events:1,manual_contests:1,participation:1,delivery_reconciliation_required:true});
  const restored=await exportLearningBackup(target,'owner');
  assert.deepEqual(restored.event_document,backup.event_document);
  assert.deepEqual(restored.preferences,backup.preferences);
  assert.deepEqual(restored.manual_contests,backup.manual_contests);
  assert.deepEqual(restored.participation,backup.participation);
  assert.deepEqual(restored.mission_archive,[]);
  assert.deepEqual(restored.delivery_archive,[]);
  assert.equal(target.sqlite.prepare('SELECT delivery_reconciliation_required FROM learners WHERE learner_id=?').get('owner').delivery_reconciliation_required,1);
  await assert.rejects(importLearningBackup(target,'owner',backup),/not empty/);
  let calls=0;
  const dispatch=await runDispatcher({db:target,curriculum,now:new Date('2026-10-15T20:45:00Z'),env:enabled,fetcher:async()=>{calls++;throw Error('must not call');},learnerIds:['owner']});
  assert.equal(calls,0);assert.equal(dispatch.processed[0].status,'restore_reconciliation_required');
  assert.deepEqual(await listDeliveries(target,'owner'),[]);
});

test('preflight rejects invalid backup and SQL conflict rolls back every category',async context=>{
  const db=createDatabase();context.after(()=>db.close());
  const template=await exportLearningBackup(db,'owner');
  const invalid={...template,event_document:{...template.event_document,events:[event('bad','other')]}};
  await assert.rejects(importLearningBackup(db,'owner',invalid),/ownership/);
  assert.equal(db.sqlite.prepare('SELECT count(*) AS n FROM learners').get().n,0);
  await appendEvents(db,'other',[event('collision','other')]);
  const conflict={...template,preferences:{weekday_minutes:42,target_keys:[]},manual_contests:[contest],participation:[entry('p1')],event_document:{...template.event_document,events:[event('collision','owner')]}};
  await assert.rejects(importLearningBackup(db,'owner',conflict),/atomic backup restore failed/);
  assert.equal(db.sqlite.prepare("SELECT count(*) AS n FROM learners WHERE learner_id='owner'").get().n,0);
  assert.equal(db.sqlite.prepare('SELECT count(*) AS n FROM manual_contests').get().n,0);
  assert.equal(db.sqlite.prepare('SELECT count(*) AS n FROM contest_participation').get().n,0);
  assert.equal(db.sqlite.prepare("SELECT count(*) AS n FROM events WHERE learner_id='other'").get().n,1);
});

test('manual contest participates in frozen mission input without duration inference',async context=>{
  const db=createDatabase();context.after(()=>db.close());
  await putManualContest(db,'owner',contest);
  const result=await runDispatcher({db,curriculum,now:new Date('2026-10-15T20:45:00Z'),env:enabled,learnerIds:['owner'],fetcher:async()=>new Response(JSON.stringify({ok:true,scheduled_message_id:'Q1'}))});
  assert.equal(result.processed[0].status,'scheduled');
  const mission=JSON.parse(db.sqlite.prepare('SELECT payload FROM mission_snapshots').get().payload);
  assert.equal(mission.activities[0].type,'contest');
  assert.equal(mission.activities[0].estimated_minutes,null);
  assert.equal(mission.activities[0].duration_minutes,100);
});

test('mock contest snapshots survive backup export/import and restore as inert archives',async context=>{
  const source=createDatabase(),target=createDatabase();context.after(()=>{source.close();target.close();});
  const mock={policy:'mock-contest-v2',local_date:'2026-10-31',duration_minutes:120,composition:{easy:4,medium:2,hard:0},
    problems:[{slot:1,problem_key:'leetcode:two-sum',title:'Two Sum',url:'https://leetcode.com/problems/two-sum/',level:1}]};
  await saveSnapshot(source,'owner','mock_contest','2026-10-31',mock);
  const backup=await exportLearningBackup(source,'owner');
  assert.deepEqual(backup.mastery_archive.recommendation_snapshots,[{kind:'mock_contest',local_date:'2026-10-31',payload:mock}]);
  await importLearningBackup(target,'owner',backup);
  const rows=target.sqlite.prepare('SELECT kind,local_date,payload FROM recommendation_snapshots WHERE learner_id=?').all('owner');
  assert.equal(rows.length,1);
  assert.match(rows[0].kind,/^archive-[0-9a-f]{32}$/,'restored mock contests are inert archive rows, not active mock_contest rows');
  assert.deepEqual(JSON.parse(rows[0].payload),{archive_type:'recommendation_snapshot',original_kind:'mock_contest',original_payload:mock});
  const restored=await exportLearningBackup(target,'owner');
  assert.deepEqual(restored.mastery_archive.recommendation_snapshots,backup.mastery_archive.recommendation_snapshots,'export unwraps the archive byte-for-byte');
});
