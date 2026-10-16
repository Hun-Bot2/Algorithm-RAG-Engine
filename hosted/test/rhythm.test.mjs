import test from 'node:test';
import assert from 'node:assert/strict';
import {buildMission} from '../core/policy.mjs';
import {putPreferences,learnerSnapshot} from '../core/store.mjs';
import {createDatabase} from './helpers/d1.mjs';
const review={status:'approved',reviewer:'fixture',evidence:'fixture'};
const p=key=>({problem_key:key,native_difficulty:'Easy',access_state:'free',concepts:['array'],review});
const curriculum={schema_version:1,curriculum_version:'fixture',problems:[p('leetcode:a'),p('leetcode:b')],relationships:[{source:'leetcode:a',target:'leetcode:b',type:'transfer',review}]};
const prefs={weekday_minutes:45,target_keys:[]};
test('weekly focus respects Seoul date and configurable virtual/upsolve planning',()=>{
  const saturday=buildMission({curriculum,events:[],preferences:prefs,now:new Date('2026-10-16T20:40:00Z')});
  assert.equal(saturday.day_focus,'virtual');assert.equal(saturday.activities[0].type,'virtual');
  assert.equal(saturday.activities[0].estimated_minutes,60);assert.equal(saturday.activities[0].planning_only,true);
  const custom=buildMission({curriculum,events:[],preferences:{...prefs,weekly_rhythm:Array(7).fill('upsolve')},now:new Date('2026-10-16T00:00:00Z')});
  assert.equal(custom.activities[0].type,'upsolve');assert.equal(custom.activities[0].estimated_minutes,45);
});
test('weekly preference round trips; invalid schedule does not mutate storage',async()=>{
  const db=createDatabase();try{
    await putPreferences(db,'me',{...prefs,weekly_rhythm:Array(7).fill('review')});
    assert.deepEqual((await learnerSnapshot(db,'me')).preferences.weekly_rhythm,Array(7).fill('review'));
    await assert.rejects(putPreferences(db,'me',{weekly_rhythm:['practice']}),/invalid weekly/);
    assert.deepEqual((await learnerSnapshot(db,'me')).preferences.weekly_rhythm,Array(7).fill('review'));
  }finally{db.close();}
});
test('reviewed transfer candidate requires prior independent source evidence',()=>{
  const event={schema_version:1,event_id:'e',learner_id:'me',kind:'attempt',source:'manual',occurred_at:'2026-10-15T00:00:00.000000Z',recorded_at:'2026-10-15T00:00:00.000000Z',supersedes:null,payload:{problem_key:'leetcode:a',outcome:'independent',hints:false}};
  const mission=buildMission({curriculum,events:[event],preferences:prefs,now:new Date('2026-10-15T01:00:00Z')});
  assert.equal(mission.activities.find(a=>a.problem_key==='leetcode:b').transfer_from,'leetcode:a');
  assert.equal(mission.activities.find(a=>a.problem_key==='leetcode:b').type,'transfer');
  assert.equal(buildMission({curriculum,events:[],preferences:prefs,now:new Date('2026-10-15T01:00:00Z')}).activities.some(a=>a.type==='transfer'),false);
});
test('all candidate providers disabled or failing still generates valid metadata missions',()=>{
  const input={curriculum,events:[],preferences:{...prefs,target_keys:['leetcode:b']},now:new Date('2026-10-16T00:00:00Z')};
  const disabled=buildMission({...input,providers:[]});
  assert.ok(disabled.activities.length>0);assert.deepEqual(disabled.relationship_providers,[]);
  const failing=buildMission({...input,providers:[{id:'missing-model',candidates(){throw Error('missing');}}]});
  assert.deepEqual(failing.activities.map(a=>a.problem_key),disabled.activities.map(a=>a.problem_key));
  assert.equal(failing.relationship_providers[0].status,'unavailable');
});
test('frozen mission cannot be altered by SQL updates or deletes',async()=>{
  const {prepareMission}=await import('../core/policy.mjs');
  const db=createDatabase();try{
    await prepareMission({db,learnerId:'me',curriculum,now:new Date('2026-10-16T00:00:00Z')});
    await assert.rejects(db.prepare("UPDATE mission_snapshots SET payload='{}'").run(),/immutable/);
    await assert.rejects(db.prepare('DELETE FROM mission_snapshots').run(),/immutable/);
  }finally{db.close();}
});
