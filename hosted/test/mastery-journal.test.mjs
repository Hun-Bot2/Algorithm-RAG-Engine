import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,statSync} from 'node:fs';
import {resolve,join} from 'node:path';
import {createApp} from '../core/app.mjs';
import {createDatabase} from './helpers/d1.mjs';
import {exportLearningBackup,importLearningBackup} from '../backup.mjs';
import {appendLedger} from '../mastery/side-ledger.mjs';
import {activeLocalCatalog,createJournalService} from '../mastery/journal.mjs';
import {catalogFromData} from '../mastery/catalog.mjs';
import {saveSnapshot} from '../mastery/ledger.mjs';

const OWNER='journal-test-owner';
const CURRICULUM={schema_version:1,curriculum_version:'journal-fixture',problems:[],relationships:[]};
const at=(date,hour=12)=>`${date}T${String(hour).padStart(2,'0')}:00:00.000000Z`;

function fixtureCatalog(){
  const problems=[];
  for(let i=1;i<=3;i++) problems.push({key:`leetcode:easy-${i}`,platform:'leetcode',title:`Easy ${i}`,url:`https://leetcode.com/problems/easy-${i}/`,difficulty_kind:'lc_level',difficulty_value:1,variation_group:`easy-${i}`,concepts:[{id:'arrays-hashing',status:'approved'}]});
  for(let i=1;i<=2;i++) problems.push({key:`leetcode:medium-${i}`,platform:'leetcode',title:`Medium ${i}`,url:`https://leetcode.com/problems/medium-${i}/`,difficulty_kind:'lc_level',difficulty_value:2,variation_group:`medium-${i}`,concepts:[{id:'arrays-hashing',status:'approved'}]});
  problems.push({key:'codeforces:100A',platform:'codeforces',title:'CF transfer',url:'https://codeforces.com/problemset/problem/100/A',difficulty_kind:'cf_rating',difficulty_value:800,concepts:[{id:'arrays-hashing',status:'approved'}]});
  return catalogFromData({version:'catalog-journal-fixture',problems,concepts:[{id:'arrays-hashing',label:'Arrays and hashing'}]});
}

function setup(t,{catalogProvider=()=>fixtureCatalog(),catalogRoot='/no-such-catalog-root',clock=at('2026-10-19',12)}={}){
  const db=createDatabase(':memory:');t.after(()=>db.close());
  let currentClock=clock;
  const journal=createJournalService({db,catalogRoot,catalogProvider});
  const app=createApp({db,journal,curriculum:CURRICULUM,authenticate:async()=>({learner_id:OWNER}),now:()=>new Date(currentClock)});
  const request=async(path,{method='GET',body,clock:nextClock}={})=>{
    if(nextClock)currentClock=nextClock;
    const headers={};
    if(body!==undefined){headers.origin='http://journal.test';headers['content-type']='application/json';}
    const response=await app.fetch(new Request(`http://journal.test/api${path}`,{method,headers,...(body!==undefined?{body:JSON.stringify(body)}:{})}));
    return {status:response.status,body:await response.json()};
  };
  const snapshotRows=()=>db.sqlite.prepare('SELECT kind,local_date,payload FROM recommendation_snapshots WHERE learner_id=? ORDER BY local_date,kind').all(OWNER);
  return {db,journal,request,snapshotRows,setClock:value=>{currentClock=value;}};
}

function attempt(event_id,problem_key,outcome,{occurred_at=at('2026-10-18',10),hints=outcome==='independent'?false:outcome==='reported_completion'?null:true,self_minutes=24,activity_type='practice',supersedes=null}={}){
  return {event_id,problem_key,outcome,hints,occurred_at,self_minutes,activity_type,supersedes};
}

test('journal API groups entries by Seoul date and labels independent, failed, and unverified outcomes',async t=>{
  const {request}=setup(t);
  let result=await request('/journal/attempt',{method:'POST',body:attempt('j-old','leetcode:easy-1','failed',{occurred_at:at('2026-10-18',10)}),clock:at('2026-10-19',12)});
  assert.equal(result.status,201);
  result=await request('/journal/attempt',{method:'POST',body:attempt('j-new','leetcode:easy-2','independent',{occurred_at:at('2026-10-19',9),hints:false}),clock:at('2026-10-19',13)});
  assert.equal(result.status,201);
  const response=await request('/journal');
  assert.equal(response.status,200);
  assert.equal(response.body.available,true);
  assert.equal(response.body.plan.progression.next.problem_key,'leetcode:easy-3','journal planner consumes the progression next-candidate contract');
  assert.equal(response.body.focus.pattern,'arrays-hashing');
  assert.equal(response.body.focus.stage,'leetcode_easy');
  assert.deepEqual(response.body.days.map(day=>day.date),['2026-10-19','2026-10-18']);
  const oldDay=response.body.days.find(day=>day.date==='2026-10-18');
  assert.equal(oldDay.entries[0].display_outcome,'X');
  assert.equal(oldDay.entries[0].priority,'priority');
  assert.equal(oldDay.entries[0].self_minutes,24);
  assert.equal(oldDay.entries[0].time_basis,'self_reported');
  assert.equal(response.body.days[0].entries.find(entry=>entry.type==='attempt').display_outcome,'O');
  assert.ok(response.body.reviews.priority.some(row=>row.problem_key==='leetcode:easy-1'));
});

test('failed attempt adds an immutable snapshot, retry is idempotent across clocks, and correction keeps old snapshot',async t=>{
  const {db,journal,request,snapshotRows}=setup(t);
  const clock1=at('2026-10-19',12);
  const initial=await journal.view(OWNER,new Date(clock1));
  assert.equal(initial.available,true);
  const rowsBefore=snapshotRows();
  assert.equal(rowsBefore.length,1);

  const failed=attempt('same-event','leetcode:easy-1','failed',{occurred_at:at('2026-10-19',10)});
  const first=await request('/journal/attempt',{method:'POST',body:failed,clock:clock1});
  assert.equal(first.status,201);
  assert.equal(first.body.added,1);
  const failureEntry=first.body.journal.days[0].entries.find(row=>row.id==='same-event');
  assert.equal(failureEntry.display_outcome,'X');
  assert.equal(failureEntry.priority,'priority');
  assert.equal(snapshotRows().length,2,'new ledger high-water mark creates a new daily snapshot');
  const frozenPayloads=snapshotRows().map(row=>row.payload);

  const retry=await request('/journal/attempt',{method:'POST',body:failed,clock:at('2026-10-19',14)});
  assert.equal(retry.status,201);
  assert.equal(retry.body.added,0);
  assert.equal(db.sqlite.prepare('SELECT COUNT(*) AS n FROM events WHERE learner_id=?').get(OWNER).n,1);
  assert.equal(snapshotRows().length,2,'same Seoul day and evidence prefix reuse the frozen snapshot');

  const correction=await request('/journal/attempt',{method:'POST',body:attempt('corrected-event','leetcode:easy-1','independent',{occurred_at:at('2026-10-19',13),hints:false,supersedes:'same-event'}),clock:at('2026-10-19',15)});
  assert.equal(correction.status,201);
  assert.equal(correction.body.added,1);
  assert.equal(snapshotRows().length,3,'correction evidence creates a new snapshot');
  assert.deepEqual(snapshotRows().slice(0,2).map(row=>row.payload),frozenPayloads,'old snapshots remain byte-identical after correction');
  const current=await request('/journal');
  const attempts=current.body.days.flatMap(day=>day.entries).filter(row=>row.type==='attempt');
  assert.deepEqual(new Set(attempts.map(row=>row.id)),new Set(['same-event','corrected-event']));
  const active=attempts.filter(row=>row.status!=='superseded');
  assert.deepEqual(active.map(row=>row.id),['corrected-event'],'only the corrected event remains active evidence');
  assert.equal(active[0].display_outcome,'O');
  const historical=attempts.find(row=>row.id==='same-event');
  assert.equal(historical.status,'superseded');
  assert.equal(historical.superseded_by,'corrected-event');
  assert.equal(historical.display_outcome,'X','the corrected failure remains visible in the timeline');
  assert.throws(()=>db.sqlite.prepare('UPDATE recommendation_snapshots SET payload=? WHERE learner_id=? AND kind=?').run('{}',OWNER,snapshotRows()[0].kind),/immutable/);
  assert.equal((await journal.view(OWNER,new Date(at('2026-10-19',16)))).available,true);
});

test('timer-only events reuse the plan snapshot; date rollover snapshots age review naturally',async t=>{
  const {request,snapshotRows}=setup(t);
  const first=await request('/journal');
  const firstSnapshot=first.body.plan.snapshot_id;
  const firstNext=first.body.plan.progression.next?.problem_key ?? null;
  const timerAt=at('2026-10-19',10);
  const timer={schema_version:1,event_id:'timer-only',learner_id:OWNER,kind:'timer_started',source:'manual',occurred_at:timerAt,recorded_at:timerAt,
    payload:{interval_id:'timer-only-interval',problem_key:'leetcode:easy-1'},supersedes:null};
  assert.equal((await request('/events',{method:'POST',body:timer})).status,201);
  const afterTimer=await request('/journal');
  assert.equal(afterTimer.body.plan.snapshot_id,firstSnapshot,'timer evidence is not curriculum or review input');
  assert.equal(afterTimer.body.plan.progression.next?.problem_key ?? null,firstNext);
  assert.equal(snapshotRows().length,1);

  const failure=attempt('dated-failure','leetcode:easy-1','failed',{occurred_at:at('2026-10-19',11),self_minutes:null});
  const afterFailure=await request('/journal/attempt',{method:'POST',body:failure});
  assert.equal(afterFailure.status,201);
  const dueToday=afterFailure.body.journal.reviews.priority.find(row=>row.problem_key==='leetcode:easy-1');
  assert.equal(dueToday.due_date,'2026-10-20');
  assert.equal(dueToday.overdue_days,0);
  const dayLater=await request('/journal',{clock:at('2026-10-23',12)});
  assert.notEqual(dayLater.body.plan.snapshot_id,afterFailure.body.journal.plan.snapshot_id,'a new Seoul date receives a new immutable plan snapshot');
  assert.equal(dayLater.body.reviews.priority.find(row=>row.problem_key==='leetcode:easy-1').overdue_days,3,'review age advances with the requested journal date');
  assert.equal(snapshotRows().length,3);
});

test('journal merges Git failure observations into priority review with self-reported time',async t=>{
  const {db,journal}=setup(t);
  const git={schema_version:1,entry_id:'git-failure',learner_id:OWNER,occurred_at:at('2026-10-18',10),recorded_at:at('2026-10-19',12),source:'git',supersedes:null,
    payload:{repo_id:'solutions',sha:'a'.repeat(40),path:'leetcode/easy-1/solution.py',problem_key:'leetcode:easy-1',outcome:'failed',hints:true,self_minutes:31,activity_type:'practice',transfer_from:null,trailers:{}}};
  assert.deepEqual(await appendLedger(db,OWNER,{git_observations:[git]}),{git_observations:1});
  assert.deepEqual(await appendLedger(db,OWNER,{git_observations:[git]}),{git_observations:0},'same Git observation retry is idempotent');
  const result=await journal.view(OWNER,new Date(at('2026-10-19',12)));
  const review=result.reviews.priority.find(row=>row.problem_key==='leetcode:easy-1');
  assert.ok(review);
  assert.equal(review.due_date,'2026-10-19');
  const entry=result.days.flatMap(day=>day.entries).find(row=>row.id==='git-failure');
  assert.equal(entry.evidence_source,'git');
  assert.equal(entry.display_outcome,'X');
  assert.equal(entry.self_minutes,31);
  assert.equal(entry.time_basis,'self_reported');
});

test('backup v2 preserves journal snapshots as archive evidence and restores self-reported minutes',async t=>{
  const source=setup(t);
  const posted=await source.request('/journal/attempt',{method:'POST',body:attempt('backup-event','leetcode:easy-1','independent',{occurred_at:at('2026-10-18',10),hints:false,self_minutes:37}),clock:at('2026-10-19',12)});
  assert.equal(posted.status,201);
  const backup=await exportLearningBackup(source.db,OWNER);
  assert.equal(backup.version,2);
  assert.ok(backup.mastery_archive.recommendation_snapshots.some(row=>row.payload.policy_bundle==='journal-v1'));
  assert.equal(backup.event_document.events[0].payload.self_minutes,37);

  const target=createDatabase(':memory:');t.after(()=>target.close());
  assert.equal((await importLearningBackup(target,OWNER,backup)).events,1);
  const archived=target.sqlite.prepare('SELECT kind,local_date,payload FROM recommendation_snapshots WHERE learner_id=?').all(OWNER);
  assert.equal(archived.length,backup.mastery_archive.recommendation_snapshots.length,'restore retains historical recommendation snapshots');
  assert.ok(archived.every(row=>row.kind.startsWith('archive-')&&JSON.parse(row.payload).archive_type==='recommendation_snapshot'));
  const restored=createJournalService({db:target,catalogProvider:()=>fixtureCatalog()});
  const fresh=await restored.view(OWNER,new Date(at('2026-10-20',12)));
  assert.equal(fresh.available,true);
  assert.equal(target.sqlite.prepare('SELECT COUNT(*) AS n FROM recommendation_snapshots WHERE learner_id=?').get(OWNER).n,archived.length+1,'journal view derives a fresh active snapshot alongside retained archive');
  assert.ok(fresh.days.flatMap(day=>day.entries).some(row=>row.type==='recommendation'&&row.archived===true),'restored snapshots appear as historical journal entries');
  assert.equal(fresh.plan.snapshot_id.startsWith('journal-'),true,'archived snapshots are not returned as the current plan');
  const restoredAttempt=fresh.days.flatMap(day=>day.entries).find(row=>row.id==='backup-event');
  assert.equal(restoredAttempt.self_minutes,37);
  assert.equal(restoredAttempt.time_basis,'self_reported');
});

test('journal reports unavailable when no local catalog has been selected',async t=>{
  const db=createDatabase(':memory:');t.after(()=>db.close());
  const journal=createJournalService({db,catalogRoot:'/definitely-missing-catalog-root'});
  const result=await journal.view(OWNER,new Date(at('2026-10-19',12)));
  assert.equal(result.available,false);
  assert.match(result.reason,/no active catalog/);
  assert.equal(db.sqlite.prepare('SELECT COUNT(*) AS n FROM recommendation_snapshots WHERE learner_id=?').get(OWNER).n,0);
});

test('catalog-v4 is attached read-only from the active switch without learner DB writes',async t=>{
  const db=createDatabase(':memory:');t.after(()=>db.close());
  const root=resolve('private-learning/mastery');
  const version='catalog-v4';
  const receipt=JSON.parse(readFileSync(join(root,version,'receipt.json'),'utf8'));
  const path=join(root,version,`${version}.sqlite3`),before=statSync(path);
  const time=at('2026-10-19',12);
  await appendLedger(db,OWNER,{catalog_switches:[{schema_version:1,entry_id:'switch-v4',learner_id:OWNER,occurred_at:time,recorded_at:time,source:'manual',supersedes:null,
    payload:{from_version:null,to_version:version,catalog_sha256:receipt.output_sha256}}]});
  const ledger={catalog_switches:[{payload:{to_version:version,catalog_sha256:receipt.output_sha256}}]};
  const catalog=activeLocalCatalog(db,ledger,root);
  assert.equal(catalog.version,version);
  assert.equal(catalog.problems.size,24303);
  const journal=createJournalService({db,catalogRoot:root});
  const result=await journal.view(OWNER,new Date(time));
  assert.equal(result.available,true);
  assert.equal(result.plan.progression.catalog_version,version,'the real read-only catalog flows through journal and progression composition');
  assert.equal(result.plan.snapshot_id.startsWith('journal-'),true);
  assert.ok(Array.isArray(result.plan.slots));
  if(result.plan.progression.next) {
    assert.ok(catalog.problems.has(result.plan.progression.next.problem_key),'progression next candidate resolves in the attached catalog');
    assert.equal(result.plan.slots.some(slot=>slot.problem_key===result.plan.progression.next.problem_key),true,'the adapter carries the real progression candidate into the plan');
  }
  assert.throws(()=>db.sqlite.prepare('UPDATE catalog.problems SET title=? WHERE problem_key=?').run('mutated','leetcode:two-sum'),/readonly/);
  const after=statSync(path);
  assert.equal(after.size,before.size);
  assert.equal(after.mtimeMs,before.mtimeMs);
  assert.equal(db.sqlite.prepare('SELECT COUNT(*) AS n FROM catalog_switches WHERE learner_id=?').get(OWNER).n,1,'only the synthetic in-memory learner DB received the test switch');
});

test('journal does not emit unrelated initial mastery changes for every catalog concept',async t=>{
  const catalog=fixtureCatalog();catalog.concepts.push({id:'greedy',label:'Greedy'});
  const {request}=setup(t,{catalogProvider:()=>catalog});
  const result=await request('/journal/attempt',{method:'POST',body:attempt('failure-only','leetcode:easy-1','failed')});
  assert.equal(result.status,201);
  const changes=result.body.journal.days.flatMap(d=>d.entries).filter(e=>e.type==='mastery_change');
  assert.deepEqual(changes.map(c=>c.pattern),['arrays-hashing']);
});

test('second-based manual durations survive journal edits and backup',async t=>{
 const f=setup(t);
 const payload={...attempt('seconds-original','leetcode:easy-1','reported_completion',{self_minutes:null,occurred_at:at('2026-10-19',10)}),self_seconds:40};
 const posted=await f.request('/journal/attempt',{method:'POST',body:payload});assert.equal(posted.status,201);
 const row=posted.body.journal.days.flatMap(d=>d.entries).find(x=>x.id==='seconds-original');
 assert.equal(posted.body.journal.weekly.process.self_reported_seconds_total,40);assert.equal(row.self_seconds,40);assert.equal(row.self_minutes,null);assert.equal(row.time_basis,'self_reported');
 const changed=await f.request('/journal/attempt',{method:'POST',body:{...payload,event_id:'seconds-edit',self_seconds:0,supersedes:'seconds-original'}});assert.equal(changed.status,201);
 const rows=changed.body.journal.days.flatMap(d=>d.entries);
 assert.equal(rows.find(x=>x.id==='seconds-original').self_seconds,40);
 assert.equal(rows.find(x=>x.id==='seconds-edit').self_seconds,0);
 const backup=await exportLearningBackup(f.db,OWNER),target=createDatabase();t.after(()=>target.close());
 await importLearningBackup(target,OWNER,backup);
 const restored=await createJournalService({db:target,catalogProvider:()=>fixtureCatalog()}).view(OWNER,new Date(at('2026-10-20')));
 assert.equal(restored.days.flatMap(d=>d.entries).find(x=>x.id==='seconds-edit').self_seconds,0);
 for(const seconds of [-1,0.5,86401])assert.equal((await f.request('/journal/attempt',{method:'POST',body:{...payload,event_id:'invalid-'+String(seconds),self_seconds:seconds}})).status,400);
 assert.equal((await f.request('/journal/attempt',{method:'POST',body:{...payload,event_id:'both-units',self_minutes:1}})).status,400);
});

test('journal survives stored mock contests and treats their problems as exposures, not plans',async t=>{
  const baseline=setup(t);
  const first=(await baseline.request('/journal')).body.plan.progression.next.problem_key;
  assert.ok(first);
  const {db,request}=setup(t);
  await saveSnapshot(db,OWNER,'mock_contest','2026-10-17',{policy:'mock-contest-v1',local_date:'2026-10-17',problems:[{slot:1,problem_key:first,level:1}]});
  const response=await request('/journal');
  assert.equal(response.status,200);
  assert.equal(response.body.available,true);
  assert.notEqual(response.body.plan.progression.next?.problem_key,first,'a mock contest problem is not the next progression candidate');
  assert.deepEqual(response.body.mock_contests.map(m=>[m.local_date,m.problems[0].problem_key]),[['2026-10-17',first]],'the journal exposes stored mock contests for the calendar');
  assert.ok(response.body.days.flatMap(d=>d.entries).every(e=>e.type!=='recommendation' || e.problem_key!==first),'mock contest rows are not shown as recommendations');
});

test('restored (archived) mock contests are calendar history only and do not exclude problems',async t=>{
  const baseline=setup(t);
  const first=(await baseline.request('/journal')).body.plan.progression.next.problem_key;
  const {db,request}=setup(t);
  await saveSnapshot(db,OWNER,'archive-0123456789abcdef0123456789abcdef','2026-10-17',{archive_type:'recommendation_snapshot',original_kind:'mock_contest',
    original_payload:{policy:'mock-contest-v2',local_date:'2026-10-17',duration_minutes:120,problems:[{slot:1,problem_key:first,level:1}]}});
  const response=await request('/journal');
  assert.equal(response.status,200);
  assert.deepEqual(response.body.mock_contests.map(m=>[m.local_date,m.archived]),[['2026-10-17',true]]);
  assert.equal(response.body.plan.progression.next.problem_key,first,'archives never act as exposures');
});
