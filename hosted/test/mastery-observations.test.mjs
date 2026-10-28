import test from 'node:test';
import assert from 'node:assert/strict';
import {deriveObservations} from '../mastery/observations.mjs';
const catalog={aliases:new Map([['leetcode:alias','leetcode:p']]),problems:new Map([['leetcode:p',{platform:'leetcode',difficulty_kind:'lc_level',difficulty_value:1,concepts:['a','b','c'].map(id=>({id,status:'approved'}))}]])};
const attempt=(day,id=`a${day}`,extra={})=>({id,problem_key:'leetcode:p',occurred_at:`2026-10-${String(day).padStart(2,'0')}T00:00:00Z`,outcome:'independent',...extra});
test('first dated attempt is ability and repeated dates are retention with actual gaps',()=>{
  const result=deriveObservations([1,2,5,12].map(d=>attempt(d)),catalog);
  assert.equal(result.ability.length,1);assert.equal(result.retention.length,3);
  assert.deepEqual(result.retention.map(o=>o.gap_days),[1,3,7]);
  assert.equal(result.ability[0].weight,1/3);
});
test('latest timestamp then id wins within a Seoul date, independent of input order',()=>{
  const rows=[attempt(1,'a'),attempt(1,'b',{outcome:'failed'}),attempt(1,'c',{occurred_at:'2026-10-01T01:00:00Z'})];
  const result=deriveObservations(rows,catalog);assert.equal(result.ability[0].id,'c');
  assert.deepEqual(deriveObservations(rows.toReversed(),catalog),result);
  const midnight=[attempt(1,'a',{occurred_at:'2026-10-01T16:00:00Z'}),attempt(2,'b')];
  assert.equal(deriveObservations(midnight,catalog).observations.length,1);
});
test('aliases, diagnostics, time missingness and ignored flags are explicit',()=>{
  const row=attempt(1,'x',{problem_key:'leetcode:alias',self_seconds:0});
  const result=deriveObservations([row],catalog);
  assert.equal(result.ability[0].problem_key,'leetcode:p');assert.equal(result.ability[0].self_seconds,0);
  assert.equal(result.ability[0].self_minutes,null);
  assert.deepEqual(deriveObservations([{...row,memo:'failed',review:true,activity_type:'transfer'}],catalog),result);
  assert.equal(deriveObservations([attempt(1,'u',{problem_key:'missing'})],catalog).unknown_problems[0].reason,'unknown_problem');
  assert.equal(deriveObservations([attempt(1,'bad',{occurred_at:'bad'})],catalog).invalid_attempts.length,1);
});
