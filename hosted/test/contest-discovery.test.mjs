import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeSchedule,fetchDiscovery,officialContestUrl,scheduleStatus,DISCOVERY_SOURCES} from '../contest-discovery.mjs';
import {cachedProviders,syncProvider} from '../provider-service.mjs';
import {createDatabase} from './helpers/d1.mjs';
import {catalogFromData} from '../mastery/catalog.mjs';
import {contestDiscoveryFit} from '../mastery/contest-discovery-fit.mjs';
import {workspaceEvents,syncDiagnostic,localDate} from '../public/workspace-data.mjs';
const now=new Date('2026-10-20T03:00:00Z');
const cf={status:'OK',result:[{id:123,name:'Round (Div. 3)',phase:'BEFORE',startTimeSeconds:Date.parse('2026-10-21T14:35:00Z')/1000,durationSeconds:7200}]};
const lc={schemaVersion:1,updatedAt:now.toISOString(),contests:[{platform:'leetcode',name:'Weekly fixture',url:'https://leetcode.com/contest/weekly-contest-999',startsAt:'2026-10-24T23:30:00Z',endsAt:'2026-10-25T01:00:00Z'}]};
const atc=`<div id="contest-table-upcoming"><table><tbody><tr><td><time class='fixtime'>2026-10-23 21:00:00+0900</time></td><td><a href="/contests/abc999">ABC &amp; fixture</a></td><td>01:40</td><td> - 1999</td></tr></tbody></table></div>`;
const bodies={codeforces:JSON.stringify(cf),atcoder:atc,leetcode:JSON.stringify(lc)};
test('all sources preserve official identity, real instant, duration and source provenance',()=>{
 const c=normalizeSchedule('codeforces',bodies.codeforces,{now}).items[0];assert.equal(c.start_at,'2026-10-21T14:35:00.000Z');assert.equal(localDate(c.start_at),'2026-10-21');assert.equal(c.duration_minutes,120);
 const a=normalizeSchedule('atcoder',atc,{now}).items[0];assert.equal(a.title,'ABC & fixture');assert.equal(a.duration_minutes,100);assert.equal(a.start_at,'2026-10-23T12:00:00.000Z');assert.equal(a.rated_range,'- 1999');assert.equal(a.eligibility.status,'unknown');
 const l=normalizeSchedule('leetcode',bodies.leetcode,{now}).items[0];assert.equal(l.duration_minutes,90);assert.equal(l.source_kind,'community_clist_mirror');assert.equal(l.source_updated_at,now.toISOString());assert.equal(localDate(l.start_at),'2026-10-25');
 assert.deepEqual(workspaceEvents({available:true},{contests:[c,a,l]}).map(x=>x.date),['2026-10-21','2026-10-23','2026-10-25']);
});
test('strict official links, schema and conflicting duplicates fail closed',()=>{
 for(const url of ['http://leetcode.com/contest/a','https://leetcode.com.evil/contest/a','https://x@leetcode.com/contest/a','https://leetcode.com/problems/a','https://leetcode.com:8000/contest/a'])assert.equal(officialContestUrl('leetcode',url),null);
 assert.throws(()=>normalizeSchedule('atcoder','<html>challenge</html>',{now}),/invalid_calendar/);
 assert.throws(()=>normalizeSchedule('leetcode',JSON.stringify({...lc,updatedAt:'2026-10-20T03:00:00'}),{now}),/invalid_calendar/);
 const copy={...lc,contests:[...lc.contests,...lc.contests]};assert.equal(normalizeSchedule('leetcode',JSON.stringify(copy),{now}).items.length,1);
 copy.contests=[lc.contests[0],{...lc.contests[0],name:'Conflict'}];assert.throws(()=>normalizeSchedule('leetcode',JSON.stringify(copy),{now}),/conflicting_duplicate/);
});
test('cache never fetches on fresh reads; failures retain stale items and throttle retries',async()=>{
 let calls=0;const fetcher=async()=>{calls++;return new Response(bodies.leetcode);};
 const first=await fetchDiscovery('leetcode',{now,fetcher});assert.equal(first.status,'fresh');
 const same=await fetchDiscovery('leetcode',{now,fetcher,previous:first});assert.deepEqual(same,first);assert.equal(calls,1);
 const later=new Date(now.getTime()+7*3600000),broken=async()=>{calls++;return new Response('',{status:503});};
 const stale=await fetchDiscovery('leetcode',{now:later,fetcher:broken,previous:first});assert.equal(stale.status,'stale');assert.equal(stale.error,'http_503');assert.equal(stale.fetched_at,first.fetched_at);assert.deepEqual(stale.items,first.items);
 await fetchDiscovery('leetcode',{now:later,fetcher:broken,previous:stale});assert.equal(calls,2);
 const old=await fetchDiscovery('leetcode',{now,fetcher:async()=>new Response(JSON.stringify({...lc,updatedAt:'2026-10-18T00:00:00Z'}))});assert.equal(old.status,'stale');assert.equal(scheduleStatus(old,now),'stale');
 const empty=await fetchDiscovery('codeforces',{now,fetcher:async()=>Response.json({status:'OK',result:[]})});assert.equal(empty.status,'fresh');assert.deepEqual(empty.items,[]);
});
test('durable owner-scoped discovery cache supersedes old calendars and survives reader recreation',async()=>{
 const db=createDatabase();try{
  let calls=0;const fetcher=async url=>{calls++;return new Response(bodies[Object.keys(DISCOVERY_SOURCES).find(k=>DISCOVERY_SOURCES[k]===url)]);};
  await Promise.all(['codeforces','atcoder','leetcode'].map(platform=>syncProvider({db,learnerId:'owner',platform,discovery:true,env:{PROVIDERS_ENABLED:'true'},now,fetcher})));
  const before=await cachedProviders(db,'owner',now);assert.equal(before.items.length,3);assert.equal(before.sources.every(s=>s.error===null),true);assert.equal(calls,3);assert.equal((await cachedProviders(db,'other',now)).items.length,0);
  await syncProvider({db,learnerId:'owner',platform:'leetcode',discovery:true,env:{PROVIDERS_ENABLED:'true'},now,fetcher});assert.equal(calls,3);
  assert.deepEqual(await cachedProviders(db,'owner',now),before);assert.equal((await cachedProviders(db,'owner',new Date('2026-10-21'))).sources.every(x=>x.status==='stale'),true);
  await assert.rejects(syncProvider({db,learnerId:'owner',platform:'leetcode',handle:'private',discovery:true,env:{PROVIDERS_ENABLED:'true'},now,fetcher}),/invalid provider/);
 }finally{db.close();}
});
function fixture({native=800,history=800,n=6}={}){
 const problems=Array.from({length:n},(_,i)=>({key:`codeforces:solve${i}`,platform:'codeforces',difficulty_kind:'cf_rating',difficulty_value:native}));
 const contests=Array.from({length:3},(_,i)=>({contest_id:String(i),platform:'codeforces',title:'Round (Div. 3)',start_at:`2026-10-0${i+1}T10:00:00Z`})),contestProblems={};
 for(const c of contests){const key=`codeforces:${c.contest_id}A`;problems.push({key,platform:'codeforces',difficulty_kind:'cf_rating',difficulty_value:history});contestProblems[c.contest_id]=[{problem_key:key}];}
 const catalog=catalogFromData({problems,contests,contestProblems}),ledger={attempts:Array.from({length:n},(_,i)=>({id:`a${i}`,problem_key:`codeforces:solve${i}`,outcome:'independent',hints:false,activity_type:'practice',occurred_at:`2026-10-${10+i}T00:00:00Z`})),rating_observations:[]};
 return {catalog,ledger,now};
}
const contest={id:'codeforces:123',platform:'codeforces',title:'Round (Div. 3)',start_at:'2026-10-21T10:00:00Z'};
test('classification requires independent distinct native evidence and discloses historical proxy',()=>{
 const f=fixture({native:2400,history:800});let r=contestDiscoveryFit(contest,f);assert.equal(r.status,'recommended');assert.match(r.reason,/이번 문제 난이도/);assert.equal(r.evidence.historical_contests.length,3);
 r=contestDiscoveryFit(contest,fixture({native:800,history:1200}));assert.equal(r.status,'stretch');
 assert.equal(contestDiscoveryFit(contest,fixture({n:3})).status,'insufficient');
 const repeated=fixture();repeated.ledger.attempts.forEach(a=>a.problem_key='codeforces:solve0');assert.equal(contestDiscoveryFit(contest,repeated).status,'insufficient');
 const assisted=fixture();assisted.ledger.attempts.forEach(a=>a.hints=true);assert.equal(contestDiscoveryFit(contest,assisted).status,'insufficient');
 assert.equal(contestDiscoveryFit({...contest,platform:'atcoder'},f).status,'insufficient');assert.equal(contestDiscoveryFit({...contest,stale:true},f).status,'insufficient');
 const unknown=fixture();for(let i=1;i<6;i++)unknown.catalog.problems.get(`codeforces:solve${i}`).difficulty_value=null;assert.equal(contestDiscoveryFit(contest,unknown).status,'insufficient');
 const missing=fixture();missing.catalog.contests=[];assert.equal(contestDiscoveryFit(contest,missing).status,'insufficient');
});
test('actual October 20 attempts and explicitly selected October 21 reviews remain separate; conflict diagnostics preserve meaning',()=>{
 const journal={available:true,review_selections:[{problem_key:'leetcode:a',date:'2026-10-21'}],days:[{date:'2026-10-21',entries:[{id:'attempt',type:'attempt',problem_key:'leetcode:a',occurred_at:'2026-10-19T15:00:00.000000Z',recorded_at:'2026-10-20T05:48:57Z',outcome:'independent',hints:false}]}],upcoming:[{problem_key:'leetcode:a',date:'2026-10-21',kind:'review'}]};
 const rows=workspaceEvents(journal,{});assert.equal(rows.find(x=>x.layer==='actual').date,'2026-10-20');assert.equal(rows.find(x=>x.layer==='plan').date,'2026-10-21');assert.equal(journal.days[0].entries[0].occurred_at,'2026-10-19T15:00:00.000000Z');
 const message=syncDiagnostic({lastWarnings:[{repo_id:'solutions',flags:['markdown_attempt_conflict:leetcode:a']}]});assert.match(message,/시간·결과·힌트/);assert.match(message,/이전 풀이와 시각은 보존/);assert.ok(!message.includes('Markdown 형식'));
 assert.equal(syncDiagnostic({lastWarnings:[]}),null);assert.match(syncDiagnostic({lastWarnings:[{repo_id:'r',flags:['f.md:daily_invalid_row:line_5']}]}),/line_5/);
});

test('historical AtCoder metadata beyond 5000 entries is accepted by compatible adapter',async()=>{
 const {fetchContests}=await import('../providers.mjs');
 const records=Array.from({length:6582},(_,i)=>({id:`abc${i}`,title:`ABC ${i}`,start_epoch_second:now.getTime()/1000-86400,duration_second:6000}));records[0].start_epoch_second+=172800;
 const response=await fetchContests('atcoder',{enabled:true,now,fetcher:async()=>Response.json(records)});assert.equal(response.status,'fresh');assert.equal(response.items.length,1);
});
test('local sync is explicit, same-origin and survives restart without new upstream requests',async()=>{
 const {startLocal}=await import('../local.mjs'),{mkdtemp,rm}=await import('node:fs/promises'),{tmpdir}=await import('node:os'),{join}=await import('node:path');
 const root=await mkdtemp(join(tmpdir(),'contest-local-test-'));let calls=0,service;
 const options={database:join(root,'learner.sqlite3'),port:0,autoGitSync:false,curriculum:{schema_version:1,curriculum_version:'fixture',problems:[],relationships:[]},catalogProvider:()=>catalogFromData({}),contestFetcher:async url=>{
  calls++;const clock=Date.now(),platform=Object.keys(DISCOVERY_SOURCES).find(k=>DISCOVERY_SOURCES[k]===url);
  if(platform==='codeforces')return Response.json({...cf,result:cf.result.map(r=>({...r,startTimeSeconds:Math.floor(clock/1000)+86400}))});
  if(platform==='atcoder')return new Response(atc.replace('2026-10-23',new Date(clock+2*86400000).toISOString().slice(0,10)));
  return Response.json({...lc,updatedAt:new Date(clock).toISOString(),contests:lc.contests.map(r=>({...r,startsAt:new Date(clock+3*86400000).toISOString(),endsAt:new Date(clock+3*86400000+5400000).toISOString()}))});
 },explanationEnv:{}};
 try{
  service=await startLocal(options);let response=await fetch(service.url+'/api/state');assert.equal(response.status,200);assert.equal(calls,0);
  response=await fetch(service.url+'/api/contests/sync',{method:'POST',headers:{'content-type':'application/json','origin':'https://evil.example'},body:'{}'});assert.equal(response.status,403);assert.equal(calls,0);
  response=await fetch(service.url+'/api/contests/sync',{method:'POST',headers:{'content-type':'application/json','origin':service.url},body:'{}'});assert.equal(response.status,200);assert.equal(calls,3);
  let state=await(await fetch(service.url+'/api/state')).json();assert.equal(state.contests.length,3);assert.equal(state.contests.every(c=>c.recommendation.status==='insufficient'),true);
  await service.close();service=await startLocal(options);state=await(await fetch(service.url+'/api/state')).json();assert.equal(state.contests.length,3);assert.equal(calls,3);
  response=await fetch(service.url+'/api/contests/sync',{method:'POST',headers:{'content-type':'application/json','origin':service.url},body:'{}'});assert.equal(response.status,200);assert.equal(calls,3);
 }finally{if(service)await service.close();await rm(root,{recursive:true,force:true});}
});
