import test from 'node:test';
import assert from 'node:assert/strict';
import baseline from './fixtures/journal-v1-preservation.json' with {type:'json'};
import {journalPlan} from '../mastery/journal.mjs';
import {journalInput} from './helpers/progression-journal-fixture.mjs';
import {v2Transitions} from '../mastery/journal-v2.mjs';
test('explicit v1 preserves the pre-integration journal output byte for byte',()=>{
  const input=journalInput();assert.equal(JSON.stringify(journalPlan({...input,progressionVersion:'v1'})),JSON.stringify(baseline));
  assert.equal(JSON.stringify(journalPlan({...input,progressionVersion:'v1'})),JSON.stringify(baseline));
});
test('v2 journal is opt-in with versioned slots and evidence-prefix transitions',()=>{
  const input=journalInput(),plan=journalPlan({...input,progressionVersion:'v2'});
  assert.equal(plan.policy_bundle,'journal-v2');assert.equal(plan.progression.policy,'progression-v2');
  assert.equal(plan.slots.find(s=>s.kind==='practice').config_sha256.length,64);
  const attempts=[{id:'a',problem_key:'leetcode:p1-0',outcome:'independent',occurred_at:'2026-10-01T00:00:00Z'},{id:'b',problem_key:'leetcode:p1-1',outcome:'independent',occurred_at:'2026-10-02T00:00:00Z'}];
  const entries=v2Transitions({...input,ledger:{...input.ledger,attempts}});
  assert.ok(entries.length);assert.equal(entries[0].cause_observation_id,'b');assert.ok(entries[0].before.cells);assert.ok(entries[0].after.cells);
});

import {createDatabase} from './helpers/d1.mjs';
import {createJournalService} from '../mastery/journal.mjs';
import {loadProgressionConfig,defaultConfig} from '../mastery/progression-v2-config.mjs';
test('journal service stores distinct snapshots per version/config without changing v1 snapshots',async t=>{
  const db=createDatabase();t.after(()=>db.close());const catalog=journalInput().catalog;
  const options={db,catalogRoot:'/unused',catalogProvider:()=>catalog};
  const clock=new Date('2026-10-29T00:00:00Z');
  const v1=await createJournalService({...options,progressionVersion:'v1'}).view('fixture-owner',clock);
  const v2=await createJournalService({...options,progressionVersion:'v2'}).view('fixture-owner',clock);
  assert.notEqual(v1.plan.snapshot_id,v2.plan.snapshot_id);
  assert.equal(v2.plan.progression.policy,'progression-v2');
  const changed=await createJournalService({...options,progressionVersion:'v2',progressionConfig:loadProgressionConfig({...defaultConfig,tau:.75})}).view('fixture-owner',clock);
  assert.notEqual(changed.plan.snapshot_id,v2.plan.snapshot_id);
  const again=await createJournalService({...options,progressionVersion:'v1'}).view('fixture-owner',clock);
  assert.deepEqual(again.plan,v1.plan);
});


test('due reviews never suppress the separate progression item',()=>{
  const input=journalInput();input.preferences={...input.preferences,weekday_minutes:30};
  input.ledger.attempts=[0,1].map(i=>({id:`a${i}`,problem_key:`leetcode:p1-${i}`,outcome:'failed',occurred_at:'2026-10-01T00:00:00Z'}));
  const plan=journalPlan({...input,progressionVersion:'v2'});
  assert.equal(plan.slots.filter(s=>s.kind==='review').length,2);
  assert.equal(plan.slots.filter(s=>s.kind==='practice').length,1);
  assert.equal(plan.review_lane_minutes,30);
});


test('v2 is now the journal default after T8 reviewed rollout',()=>{
  assert.equal(journalPlan(journalInput()).progression.policy,'progression-v2');
});

import {planProgressionV2} from '../mastery/progression-v2.mjs';
test('journal forwards the actual learner seed to candidate tie breaking',()=>{
  const input=journalInput();
  for(const learnerId of ['learner-one','learner-two']){
    const plan=journalPlan({...input,learnerId});
    const expected=planProgressionV2({...input,learnerId});
    assert.equal(plan.progression.next.tie_break.hash,expected.next.tie_break.hash);
    assert.equal(plan.progression.next.problem_key,expected.next.problem_key);
  }
});
