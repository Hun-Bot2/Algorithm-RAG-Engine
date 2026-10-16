import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequestGate, fetchContests, fetchSubmissions, submissionEvent} from '../providers.mjs';
const now = new Date('2026-10-16T00:00:00Z');
const reply = object => new Response(JSON.stringify(object));

test('all providers disabled makes no network call and exposes missingness', async () => {
  const fetcher = () => { throw Error('network forbidden'); };
  const result = await fetchContests('codeforces', {fetcher,now});
  assert.equal(result.status,'unavailable'); assert.equal(result.fetched_at,null);
  assert.equal((await fetchSubmissions('atcoder',{fetcher,now})).error,'disabled');
});
test('fresh cache, stale outage, reported empty calendar are distinct', async () => {
  const previous = {source:'x', fetched_at:now.toISOString(),items:[{id:'cached'}]};
  const cached = await fetchContests('atcoder',{enabled:true,now,previous,fetcher:()=>{throw Error();}});
  assert.equal(cached.status,'fresh');
  const stale = await fetchContests('atcoder',{enabled:true,now:new Date('2026-10-18'),previous,fetcher:()=>{throw Error();}});
  assert.equal(stale.status,'stale'); assert.equal(stale.items[0].id,'cached');
  const empty = await fetchContests('atcoder',{enabled:true,now,fetcher:async()=>reply([])});
  assert.equal(empty.status,'fresh'); assert.deepEqual(empty.items,[]);
});
test('calendar normalized to canonical links; suitability not invented', async () => {
  const r = await fetchContests('codeforces',{enabled:true,now,fetcher:async()=>reply({status:'OK',result:[
    {id:123,name:'Round',phase:'BEFORE',startTimeSeconds:now.getTime()/1000+86400,durationSeconds:7200}
  ]})});
  assert.equal(r.items[0].url,'https://codeforces.com/contest/123');
  assert.equal(r.items[0].suitable,false); assert.equal(r.items[0].duration_minutes,120);
});
test('serialized gate spaces requests including failures', async () => {
  let clock=0, active=0, peak=0; const starts=[];
  const gate=createRequestGate({now:()=>clock,gapMs:2000,sleep:async ms=>{clock+=ms;},fetcher:async()=>{
    active++;peak=Math.max(peak,active);starts.push(clock);await Promise.resolve();active--;
    if(starts.length===1) throw Error('one'); return reply({});
  }});
  const result = await Promise.allSettled([gate('x'),gate('x'),gate('x')]);
  assert.equal(result[0].status,'rejected'); assert.equal(peak,1); assert.deepEqual(starts,[0,2000,4000]);
});
test('accepted submission is reported completion, never independent or timed',()=>{
  const event=submissionEvent('codeforces',{id:77,verdict:'OK',creationTimeSeconds:1,
    problem:{contestId:4,index:'A'}},'me',now);
  assert.equal(event.payload.outcome,'reported_completion');assert.equal(event.payload.hints,null);
  assert.equal(event.payload.problem_key,'codeforces:4A');assert.equal('duration' in event.payload,false);
});
test('AtCoder boundary overlap deduplicates same-second submissions',async()=>{
  const previous={cursor:100,seen_ids:['1']};
  const r=await fetchSubmissions('atcoder',{enabled:true,handle:'me',learnerId:'me',previous,now,
    fetcher:async()=>reply([{id:1,problem_id:'abc001_a',result:'AC',epoch_second:100},
      {id:2,problem_id:'abc001_b',result:'AC',epoch_second:100}])});
  assert.equal(r.events.length,1);assert.equal(r.cursor,100);assert.equal(r.truncated,false);
});
test('partial failure retains original watermark and no partial events',async()=>{
  let n=0;const previous={seen_ids:[],cursor:99,fetched_at:'2026-10-15T00:00:00Z'};
  const list=Array.from({length:100},(_,i)=>({id:i+1,verdict:'OK',creationTimeSeconds:i+1,problem:{contestId:4,index:'A'}}));
  const r=await fetchSubmissions('codeforces',{enabled:true,handle:'me',learnerId:'me',previous,now,
    fetcher:async()=>{if(n++)throw Error();return reply({status:'OK',result:list});}});
  assert.equal(r.status,'stale');assert.equal(r.cursor,99);assert.deepEqual(r.events,[]);
});
