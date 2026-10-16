import test from 'node:test';
import assert from 'node:assert/strict';
import {createDatabase} from './helpers/d1.mjs';
import {putPreferences,getMission} from '../core/store.mjs';
import {runScheduler} from '../scheduler.mjs';
const review={status:'approved',reviewer:'fixture',evidence:'fixture'};
const curriculum={curriculum_version:'fixture',problems:[{problem_key:'leetcode:a',native_difficulty:'Easy',access_state:'free',concepts:[],review}],relationships:[]};
test('disabled scheduler needs no database, model or transport',async()=>{
  assert.deepEqual(await runScheduler(),{disabled:true,frozen:0});
});
test('mission recovery freezes with Slack and models off, records actual late preparation',async()=>{
  const db=createDatabase();try{
    await putPreferences(db,'me',{weekday_minutes:60,target_keys:[]});
    const now=new Date('2026-10-15T20:45:00Z');
    const result=await runScheduler({db,curriculum,now,env:{SCHEDULER_ENABLED:'true'},fetcher:()=>{throw Error('No Slack');}});
    assert.equal(result.delivery.disabled,true);assert.equal(result.frozen,1);
    const mission=await getMission(db,'me','2026-10-16');
    assert.equal(mission.event_cutoff,'2026-10-15T20:40:00.000000Z');
    assert.equal(mission.prepared_at,'2026-10-15T20:45:00.000000Z');
    assert.ok(mission.activities.length>0);
    await runScheduler({db,curriculum,now:new Date('2026-10-15T21:00:00Z'),env:{SCHEDULER_ENABLED:'true'}});
    assert.deepEqual(await getMission(db,'me','2026-10-16'),mission);
  }finally{db.close();}
});
