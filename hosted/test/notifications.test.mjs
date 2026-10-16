import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createDatabase} from './helpers/d1.mjs';
import {runDispatcher,deliveryTimes,reconcileDelivery} from '../dispatcher.mjs';
import {claimDelivery,deliveryKey,ensureDelivery,formatLegacyRecommendationBlock,formatMissionSlack,listDeliveries} from '../notifications.mjs';
import {prepareMission} from '../core/policy.mjs';

const date='2026-10-16', utc=clock=>new Date(`2026-10-15T${clock}Z`);
const env={DELIVERY_ENABLED:'true',DELIVERY_LEARNER_ID:'owner',SLACK_BOT_TOKEN:'test-token',SLACK_CHANNEL_ID:'D_TEST'};
const review={status:'approved',reviewer:'fixture',evidence:'fixture'};
const curriculum={schema_version:1,curriculum_version:'test-v1',problems:[{problem_key:'leetcode:two-sum',canonical_url:'https://leetcode.com/problems/two-sum/',native_difficulty:'Easy',access_state:'free',concepts:['arrays'],review}],relationships:[]};
const annotations={'leetcode:two-sum':{title:'Two Sum',difficulty_band:'beginner'}};
const respond=(body,status=200,headers={})=>new Response(JSON.stringify(body),{status,headers});
const scheduled=(id='Q123')=>respond({ok:true,scheduled_message_id:id,channel:'D_TEST',post_at:String(Math.floor(deliveryTimes(date).target.getTime()/1000))});
const noSchedule=()=>respond({ok:true,scheduled_messages:[],response_metadata:{next_cursor:''}});
const noHistory=()=>respond({ok:true,messages:[],response_metadata:{next_cursor:''}});

test('disabled dispatcher makes no Slack calls or outbox row',async context=>{
  const db=createDatabase(); context.after(()=>db.close());
  let calls=0;
  const result=await runDispatcher({db,curriculum,now:utc('20:45:00'),env:{...env,DELIVERY_ENABLED:'false'},learnerIds:['owner'],fetcher:async()=>{calls++;return scheduled();}});
  assert.equal(result.disabled,true);assert.equal(calls,0);assert.deepEqual(await listDeliveries(db,'owner'),[]);
});

test('05:40 cutoff freezes immutable mission and schedules at 06:00 without metadata',async context=>{
  const db=createDatabase(); context.after(()=>db.close());
  const calls=[];
  const fetcher=async(url,options)=>{calls.push({url,body:JSON.parse(options.body)});return scheduled();};
  const result=await runDispatcher({db,curriculum,annotations,now:utc('20:45:00'),env,fetcher,learnerIds:['owner']});
  assert.equal(result.processed[0].status,'scheduled');
  assert.equal(calls.length,1);
  assert.match(calls[0].url,/chat\.scheduleMessage$/);
  assert.equal(calls[0].body.post_at,Math.floor(deliveryTimes(date).target.getTime()/1000));
  assert.equal('metadata' in calls[0].body,false);
  assert.match(calls[0].body.text,/ARE:v1:2026-10-16:/);
  assert.equal(JSON.parse(db.sqlite.prepare('SELECT mission_payload FROM delivery_outbox').get().mission_payload).event_cutoff,'2026-10-15T20:40:00.000000Z');
  assert.equal((await listDeliveries(db,'owner'))[0].state,'scheduled');
  await assert.rejects(db.prepare("UPDATE delivery_outbox SET mission_hash='tampered'").run(),/immutable/);
});

test('late recovery posts through 06:10; after deadline marks missed',async context=>{
  const db=createDatabase(); context.after(()=>db.close());
  const calls=[];
  await runDispatcher({db,curriculum,now:utc('21:05:00'),env,fetcher:async(url,options)=>{calls.push(url);return respond({ok:true,ts:'123.456',channel:'D_TEST'});},learnerIds:['owner']});
  assert.equal(calls.length,1);assert.match(calls[0],/chat\.postMessage$/);
  assert.equal((await listDeliveries(db,'owner'))[0].state,'posted');
  const other=createDatabase();context.after(()=>other.close());
  let lateCalls=0;
  await runDispatcher({db:other,curriculum,now:utc('21:11:00'),env,fetcher:async()=>{lateCalls++;return scheduled();},learnerIds:['owner']});
  assert.equal(lateCalls,0);assert.equal((await listDeliveries(other,'owner'))[0].state,'missed');
});

test('429 persists retry time and retries only on later dispatcher invocation',async context=>{
  const db=createDatabase();context.after(()=>db.close());
  let sends=0;
  const rate=async()=>{sends++;return respond({ok:false,error:'ratelimited'},429,{'Retry-After':'120'});};
  await runDispatcher({db,curriculum,now:utc('20:45:00'),env,fetcher:rate,learnerIds:['owner']});
  let row=(await listDeliveries(db,'owner'))[0];
  assert.equal(row.state,'ready');assert.equal(row.retry_at,'2026-10-15T20:47:00.000000Z');assert.equal(sends,1);
  await runDispatcher({db,curriculum,now:utc('20:46:00'),env,fetcher:rate,learnerIds:['owner']});
  assert.equal(sends,1);
  await runDispatcher({db,curriculum,now:utc('20:47:00'),env,fetcher:async()=>{sends++;return scheduled();},learnerIds:['owner']});
  assert.equal(sends,2);assert.equal((await listDeliveries(db,'owner'))[0].state,'scheduled');
});

test('schedule crossing target with definitive time_in_past retries as immediate post',async context=>{
  const db=createDatabase();context.after(()=>db.close());
  await runDispatcher({db,curriculum,now:utc('20:59:59'),env,fetcher:async()=>respond({ok:false,error:'time_in_past'}),learnerIds:['owner']});
  assert.equal((await listDeliveries(db,'owner'))[0].state,'ready');
  assert.equal((await listDeliveries(db,'owner'))[0].retry_at,'2026-10-15T21:00:00.000000Z');
  let endpoint='';
  await runDispatcher({db,curriculum,now:utc('21:00:00'),env,fetcher:async url=>{endpoint=url;return respond({ok:true,ts:'123.456'});},learnerIds:['owner']});
  assert.match(endpoint,/chat\.postMessage$/);
  assert.equal((await listDeliveries(db,'owner'))[0].state,'posted');
});

test('accepted timeout becomes unknown; absence is never permission to resend',async context=>{
  const db=createDatabase();context.after(()=>db.close());
  let sends=0,lookups=0;
  await runDispatcher({db,curriculum,now:utc('20:45:00'),env,fetcher:async()=>{sends++;throw Error('timeout after acceptance');},learnerIds:['owner']});
  assert.equal((await listDeliveries(db,'owner'))[0].state,'unknown');
  const fetcher=async url=>{lookups++;return url.includes('scheduledMessages')?noSchedule():noHistory();};
  await runDispatcher({db,curriculum,now:utc('20:50:00'),env,fetcher,learnerIds:['owner']});
  assert.equal(sends,1);assert.equal(lookups,2);assert.equal((await listDeliveries(db,'owner'))[0].state,'unknown');
});

test('reconciliation follows cursors, recognizes schedule and later history',async context=>{
  const db=createDatabase();context.after(()=>db.close());
  await runDispatcher({db,curriculum,now:utc('20:45:00'),env,fetcher:async()=>{throw Error('timeout');},learnerIds:['owner']});
  const row=db.sqlite.prepare('SELECT * FROM delivery_outbox').get();
  let pages=0;
  const scheduleFetch=async(url,options)=>{
    if (url.includes('scheduledMessages')) {
      const cursor=JSON.parse(options.body).cursor;
      pages++;
      return cursor ? respond({ok:true,scheduled_messages:[{id:'Q123',text:'message '+row.uniqueness_key}],response_metadata:{next_cursor:''}}) : respond({ok:true,scheduled_messages:[],response_metadata:{next_cursor:'next'}});
    }
    return noHistory();
  };
  assert.equal(await reconcileDelivery(db,row,{env,fetcher:scheduleFetch,now:utc('20:50:00')}),'scheduled');
  assert.equal(pages,2);
  const scheduledRow=db.sqlite.prepare('SELECT * FROM delivery_outbox').get();
  const historyFetch=async url=>url.includes('scheduledMessages')?noSchedule():respond({ok:true,messages:[{ts:'123.456',text:'message '+row.uniqueness_key}],response_metadata:{next_cursor:''}});
  assert.equal(await reconcileDelivery(db,scheduledRow,{env,fetcher:historyFetch,now:utc('21:05:00')}),'posted');
  assert.equal((await listDeliveries(db,'owner'))[0].slack_ts,'123.456');
});

test('bounded incomplete pagination keeps unknown',async context=>{
  const db=createDatabase();context.after(()=>db.close());
  await runDispatcher({db,curriculum,now:utc('20:45:00'),env,fetcher:async()=>{throw Error('timeout');},learnerIds:['owner']});
  const row=db.sqlite.prepare('SELECT * FROM delivery_outbox').get();
  let pages=0;
  const fetcher=async(url,options)=>{
    if (url.includes('scheduledMessages')) {pages++;return respond({ok:true,scheduled_messages:[],response_metadata:{next_cursor:'page'+pages}});}
    return noHistory();
  };
  assert.equal(await reconcileDelivery(db,row,{env,fetcher,now:utc('20:50:00')}),'unknown');
  assert.equal(pages,5);
});

test('atomic claim allows one concurrent sender; expired sending lease becomes unknown',async context=>{
  const db=createDatabase();context.after(()=>db.close());
  const mission=await prepareMission({db,learnerId:'owner',curriculum,now:deliveryTimes(date).cutoff});
  await ensureDelivery(db,{learnerId:'owner',mission,curriculum,channelId:'D_TEST',now:utc('20:45:00')});
  const args={learnerId:'owner',localDate:date,channelId:'D_TEST',now:utc('20:45:00'),deadline:deliveryTimes(date).deadline};
  const claims=await Promise.all([claimDelivery(db,args),claimDelivery(db,args)]);
  assert.equal(claims.filter(Boolean).length,1);
  let calls=0;
  await runDispatcher({db,curriculum,now:utc('20:47:00'),env,fetcher:async url=>{calls++;return url.includes('scheduledMessages')?noSchedule():noHistory();},learnerIds:['owner']});
  assert.equal((await listDeliveries(db,'owner'))[0].state,'unknown');
  assert.equal(calls,2);
});

test('stale frozen curriculum prevents delivery and retains original snapshot',async context=>{
  const db=createDatabase();context.after(()=>db.close());
  await prepareMission({db,learnerId:'owner',curriculum,now:utc('20:41:00')});
  let calls=0;
  await runDispatcher({db,curriculum,now:utc('20:45:00'),env,fetcher:async()=>{calls++;return scheduled();},learnerIds:['owner']});
  assert.equal(calls,0);
  assert.equal((await listDeliveries(db,'owner'))[0].state,'missed');
  assert.equal(JSON.parse(db.sqlite.prepare('SELECT mission_payload FROM delivery_outbox').get().mission_payload).event_cutoff,'2026-10-15T20:41:00.000000Z');
});

test('formatter uses canonical links and bounds; legacy fixture sections remain readable',async()=>{
  const marker=await deliveryKey('owner',date,'D_TEST');
  const mission={date,activities:[{type:'practice',problem_key:'leetcode:two-sum',reason:'검토된 무료 문제입니다.'}]};
  const formatted=formatMissionSlack(mission,curriculum,annotations,marker);
  assert.match(formatted.blocks[3].text.text,/https:\/\/leetcode.com\/problems\/two-sum/);
  assert.equal(formatted.blocks.length,6);assert.ok(formatted.blocks.every(b=>(b.text?.text?.length ?? 0)<=3000));
  const fixture=JSON.parse(readFileSync(new URL('./fixtures/legacy-recommendation.json',import.meta.url),'utf8'));
  const block=formatLegacyRecommendationBlock(fixture.input);
  assert.equal(block.text.text,fixture.expected_text);
});
