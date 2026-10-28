import test from 'node:test';
import assert from 'node:assert/strict';
import {createDatabase} from './helpers/d1.mjs';
import {appendEvents,exportDocument,importDocument,learnerSnapshot} from '../core/store.mjs';
import {activeEvents,confirmedSeconds,timerState} from '../core/events.mjs';
import {buildMission,deriveProgress,prepareMission} from '../core/policy.mjs';
import {createApp} from '../core/app.mjs';
import {authenticateAccess} from '../core/auth.mjs';

const t = (day,hour='00:00:00') => `2026-10-${day}T${hour}Z`;
const review = {status:'approved',reviewer:'fixture',evidence:'fixture'};
const problem = (key,concept='arrays',access='free') => ({problem_key:key,canonical_url:'https://leetcode.com/problems/'+key.split(':')[1]+'/',native_difficulty:'Easy',access_state:access,concepts:[concept],review});
const curriculum = {schema_version:1,curriculum_version:'fixture-1',problems:[problem('leetcode:array-base'),problem('leetcode:target'),problem('leetcode:paid','arrays','paid')],relationships:[{source:'leetcode:array-base',target:'leetcode:target',type:'prerequisite',review}]};
const event = (id,key='leetcode:array-base',outcome='independent',at=t('16'),extras={}) => ({schema_version:1,event_id:id,learner_id:'owner',occurred_at:at,recorded_at:at,kind:'attempt',source:'manual',payload:{problem_key:key,outcome,hints:outcome==='independent'?false:null,...extras},supersedes:null});
const clock = new Date('2026-10-16T01:00:00Z');

test('SQL-backed append, idempotency, ownership and atomic conflicting restore',async context=>{
  const db=createDatabase(); context.after(()=>db.close());
  const base=event('a');
  assert.equal(await appendEvents(db,'owner',[base]),1);
  assert.equal(await appendEvents(db,'owner',[base]),0);
  await assert.rejects(appendEvents(db,'other',[base]),/ownership/);
  await assert.rejects(importDocument(db,'owner',{format:'adaptive-learning-events',version:1,learner_id:'owner',events:[event('b'),{...base,payload:{...base.payload,outcome:'failed',hints:null}}]}),/conflicting/);
  assert.deepEqual((await exportDocument(db,'owner')).events.map(e=>e.event_id),['a']);
  assert.equal(await importDocument(db,'owner',await exportDocument(db,'owner')),0);
  assert.equal(db.sqlite.prepare('SELECT revision FROM learners WHERE learner_id=?').get('owner').revision,1);
});

test('concurrent identical append converges to one event',async context=>{
  const db=createDatabase(); context.after(()=>db.close());
  const pair=await Promise.all([appendEvents(db,'owner',[event('same')]),appendEvents(db,'owner',[event('same')])]);
  assert.deepEqual(pair.sort(),[0,1]);
  assert.equal((await exportDocument(db,'owner')).events.length,1);
});

test('revision guard rolls back stale insert, including its learner changes',async context=>{
  const db=createDatabase(); context.after(()=>db.close());
  const guard=[db.prepare('INSERT OR IGNORE INTO learners(learner_id) VALUES (?)').bind('owner'),db.prepare('UPDATE learners SET revision=revision+1 WHERE learner_id=? AND revision=?').bind('owner',42),db.prepare('INSERT INTO mutation_guard(ok) VALUES ((SELECT changes()))'),db.prepare('INSERT INTO events(event_id,learner_id,schema_version,occurred_at,recorded_at,kind,source,payload,supersedes,canonical) VALUES (?,?,?,?,?,?,?,?,?,?)').bind('bad','owner',1,t('16'),t('16'),'attempt','manual','{}',null,'{}')];
  await assert.rejects(db.batch(guard));
  assert.equal((await learnerSnapshot(db,'owner')).events.length,0);
  assert.equal(db.sqlite.prepare('SELECT count(*) AS n FROM learners').get().n,0);
});

test('future-recorded correction cannot rewrite past exposure and mission is frozen',async context=>{
  const db=createDatabase(); context.after(()=>db.close());
  const original=event('first','leetcode:array-base','independent',t('15'));
  const correction={...event('fix','leetcode:array-base','failed',t('15')),recorded_at:t('17'),supersedes:'first'};
  await appendEvents(db,'owner',[original,correction]);
  assert.equal(activeEvents((await learnerSnapshot(db,'owner')).events,'2026-10-16T01:00:00.000000Z')[0].event_id,'first');
  const first=await prepareMission({db,learnerId:'owner',curriculum,now:clock});
  assert.equal(first.event_cutoff,'2026-10-16T01:00:00.000000Z');
  assert.equal(first.activities[0].problem_key,'leetcode:array-base');
  assert.equal(first.activities[0].type,'review');
  const again=await prepareMission({db,learnerId:'owner',curriculum,now:new Date('2026-10-16T10:00:00Z')});
  assert.deepEqual(again,first);
});

test('review remediation, reported completion and transfer evidence are explicit',()=>{
  const graph={...curriculum,relationships:[...curriculum.relationships,{source:'leetcode:array-base',target:'leetcode:target',type:'transfer',review}]};
  const items=[event('imported','leetcode:target','reported_completion',t('14'))];
  assert.equal(deriveProgress(graph,items,'2026-10-16T01:00:00.000000Z').problems['leetcode:target'].state,'unknown');
  items.push(event('demo','leetcode:array-base','independent',t('14')));
  items.push(event('review','leetcode:array-base','failed',t('15'),{activity_type:'review'}));
  assert.equal(deriveProgress(graph,items,'2026-10-16T01:00:00.000000Z').problems['leetcode:array-base'].state,'learning');
  const mission=buildMission({curriculum:graph,events:items,preferences:{weekday_minutes:60,target_keys:['leetcode:target']},now:clock});
  assert.equal(mission.activities[0].type,'review');
  assert.equal(mission.activities[0].problem_key,'leetcode:array-base');
  items.push(event('transfer','leetcode:target','independent',t('16'),{activity_type:'transfer',transfer_from:'leetcode:array-base'}));
  assert.equal(deriveProgress(graph,items,'2026-10-16T01:00:00.000000Z').problems['leetcode:target'].state,'transfer-evidenced');
  assert.equal(deriveProgress(graph,[items[0],items.at(-1)],'2026-10-16T01:00:00.000000Z').problems['leetcode:target'].state,'demonstrated');
});

test('candidate order, free eligibility and no network/model dependency',()=>{
  const mission=buildMission({curriculum,events:[],preferences:{weekday_minutes:60,target_keys:[]},now:clock});
  assert.deepEqual(mission.activities.map(a=>a.problem_key),['leetcode:array-base']);
  assert.ok(mission.exposure.every(a=>a.problem_key!=='leetcode:paid'));
  assert.deepEqual(buildMission({curriculum,events:[],preferences:{weekday_minutes:60,target_keys:[]},now:clock}),mission);
});

test('budget never selects an oversized task and contest day overrides weekly practice',()=>{
  const preferences={weekday_minutes:10,target_keys:[]};
  const ordinary=buildMission({curriculum,events:[],preferences,now:clock});
  assert.deepEqual(ordinary.activities,[]);
  const contest={id:'abc-1',suitable:true,start_at:'2026-10-16T12:00:00Z',duration_minutes:100};
  const today=buildMission({curriculum,events:[],preferences,now:clock,contests:[contest]});
  assert.deepEqual(today.activities.map(a=>a.type),['contest']);
  assert.equal(today.activities[0].estimated_minutes,null);
});

test('cold start restricts unknown difficulty but accepts reviewed beginner annotation',()=>{
  const harder={...curriculum,problems:[{...curriculum.problems[0],native_difficulty:'Unknown'}],relationships:[]};
  assert.equal(buildMission({curriculum:harder,events:[],preferences:{weekday_minutes:60,target_keys:[]},now:clock}).candidate_count,0);
  assert.equal(buildMission({curriculum:harder,annotations:{'leetcode:array-base':{difficulty_band:'beginner'}},events:[],preferences:{weekday_minutes:60,target_keys:[]},now:clock}).candidate_count,1);
});

test('invalid target does not silently fall back to unrelated practice',()=>{
  const mission=buildMission({curriculum,events:[],preferences:{weekday_minutes:60,target_keys:['leetcode:not-reviewed']},now:clock});
  assert.equal(mission.candidate_count,0);
  assert.deepEqual(mission.activities,[]);
});

test('timer interruption stays unavailable; confirmed overlapping time is unioned',async context=>{
  const db=createDatabase(); context.after(()=>db.close());
  const make=(id,kind,at,payload)=>({schema_version:1,event_id:id,learner_id:'owner',occurred_at:at,recorded_at:at,kind,source:'manual',payload,supersedes:null});
  await appendEvents(db,'owner',[make('s1','timer_started',t('16','00:00:00'),{interval_id:'i1',problem_key:'leetcode:array-base'}),make('s2','timer_started',t('16','00:05:00'),{interval_id:'i2',problem_key:'leetcode:target'})]);
  assert.deepEqual(timerState((await learnerSnapshot(db,'owner')).events).map(x=>x.status),['open','open']);
  await appendEvents(db,'owner',[make('e1','timer_stopped',t('16','00:20:00'),{interval_id:'i1'}),make('e2','timer_stopped',t('16','00:25:00'),{interval_id:'i2'}),make('c1','timer_confirmed',t('16','00:21:00'),{interval_id:'i1',started_at:t('16','00:00:00'),ended_at:t('16','00:20:00')}),make('c2','timer_confirmed',t('16','00:26:00'),{interval_id:'i2',started_at:t('16','00:05:00'),ended_at:t('16','00:25:00')})]);
  const intervals=timerState((await learnerSnapshot(db,'owner')).events);
  assert.equal(confirmedSeconds(intervals),1500);
  const micro=[{status:'confirmed',confirmed_start:'2026-10-16T00:00:00.000001Z',confirmed_end:'2026-10-16T00:00:00.000003Z'}];
  assert.equal(confirmedSeconds(micro),0.000002);
  await assert.rejects(appendEvents(db,'owner',[make('s3','timer_started',t('16','00:30:00'),{interval_id:'i1',problem_key:'leetcode:array-base'})]),/duplicate timer start/);
});

test('API rejects caller owner and authenticates assets before fetch',async context=>{
  const db=createDatabase(); context.after(()=>db.close());
  let assetCalls=0;
  const app=createApp({db,curriculum,authenticate:async r=>{if (!r.headers.get('x-test-auth')) throw Error('missing'); return {learner_id:'owner'};},assets:{fetch:async()=>{assetCalls++;return new Response('ok');}},now:()=>clock});
  assert.equal((await app.fetch(new Request('https://local/'))).status,401);
  assert.equal(assetCalls,0);
  const unauthorized={...event('wrong'),learner_id:'other'};
  const response=await app.fetch(new Request('https://local/api/events',{method:'POST',headers:{'x-test-auth':'1','content-type':'application/json','origin':'https://local'},body:JSON.stringify(unauthorized)}));
  assert.equal(response.status,400);
  assert.equal((await exportDocument(db,'owner')).events.length,0);
  assert.equal((await app.fetch(new Request('https://local/api/events',{method:'POST',headers:{'x-test-auth':'1','content-type':'application/json','origin':'https://evil.example'},body:JSON.stringify(event('csrf'))}))).status,403);
  const state=await (await app.fetch(new Request('https://local/api/state',{headers:{'x-test-auth':'1'}}))).json();
  assert.equal(state.snapshot_status,'preview');
  assert.ok(state.mission.activities.length);
  assert.equal((await app.fetch(new Request('https://local/',{headers:{'x-test-auth':'1'}}))).status,200);
});

test('Access JWT verifies RS256, issuer/audience/time/email and rejects tampering',async()=>{
  const pair=await crypto.subtle.generateKey({name:'RSASSA-PKCS1-v1_5',modulusLength:2048,publicExponent:new Uint8Array([1,0,1]),hash:'SHA-256'},true,['sign','verify']);
  const publicJwk=await crypto.subtle.exportKey('jwk',pair.publicKey);
  const encode=value=>btoa(JSON.stringify(value)).replace(/=/g,'').replace(/\+/g,'-').replace(/\//g,'_');
  const env={ACCESS_ISSUER:'https://team.cloudflareaccess.com',ACCESS_AUDIENCE:'aud',ACCESS_EMAIL_ALLOWLIST:'user@example.com'};
  const claims={iss:env.ACCESS_ISSUER,aud:'aud',sub:'stable-sub',email:'user@example.com',nbf:100,exp:300};
  const sign=async payload=>{const head=encode({alg:'RS256',kid:'key'}),body=encode(payload);const sig=await crypto.subtle.sign('RSASSA-PKCS1-v1_5',pair.privateKey,new TextEncoder().encode(head+'.'+body));return head+'.'+body+'.'+btoa(String.fromCharCode(...new Uint8Array(sig))).replace(/=/g,'').replace(/\+/g,'-').replace(/\//g,'_');};
  const fetcher=async()=>Response.json({keys:[{...publicJwk,kid:'key',alg:'RS256'}]});
  const request=token=>new Request('https://local/',{headers:{'Cf-Access-Jwt-Assertion':token}});
  const token=await sign(claims);
  const identity=await authenticateAccess(request(token),env,{fetcher,now:()=>200_000});
  assert.match(identity.learner_id,/^access-[a-f0-9]{64}$/);
  await assert.rejects(authenticateAccess(request(token+'x'),env,{fetcher,now:()=>200_000}));
  await assert.rejects(authenticateAccess(request(await sign({...claims,email:'other@example.com'})),env,{fetcher,now:()=>200_000}));
  await assert.rejects(authenticateAccess(request(token),env,{fetcher,now:()=>300_000}));
});

test('manual LeetCode schedules preserve entered timing and provenance without inventing a provider',async context=>{
  const db=createDatabase();context.after(()=>db.close());
  const app=createApp({db,curriculum,authenticate:async()=>({learner_id:'owner'}),now:()=>clock});
  const schedule={id:'manual-lc',platform:'leetcode',title:'User verified contest',url:'https://leetcode.com/contest/',start_at:'2026-10-18T02:30:00.000Z',duration_minutes:90,suitable:false};
  const request=body=>app.fetch(new Request('https://learning.test/api/contests/manual',{method:'POST',headers:{origin:'https://learning.test','content-type':'application/json'},body:JSON.stringify(body)}));
  const response=await request(schedule);assert.equal(response.status,201);
  const state=await(await app.fetch(new Request('https://learning.test/api/state'))).json();
  const saved=state.contests.find(c=>c.id==='manual-lc');assert.equal(saved.source,'manual');assert.equal(saved.platform,'leetcode');assert.equal(saved.start_at,'2026-10-18T02:30:00.000000Z');assert.equal(saved.duration_minutes,90);
  assert.equal(state.capabilities.providers_enabled,false);
  assert.equal((await request({...schedule,id:'invalid',platform:'invented'})).status,400);
});
