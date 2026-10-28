import test from 'node:test';
import assert from 'node:assert/strict';
import {cellReadiness,conceptReadiness,conceptFollowUp,validatePrerequisites} from '../mastery/progression-v2.mjs';
const problem=(key,concepts,level=1)=>({key,platform:'leetcode',eligible:true,status:'active',access:'free',difficulty_value:level,concepts:concepts.map(id=>({id,status:'approved'}))});
const catalog={problems:new Map([['p',problem('p',['a'])],['q',problem('q',['a','b'])],['m',problem('m',['a'],2)]])};
const obs=(id,day,y,level=1)=>({id,kind:'ability',problem_key:'q',platform:'leetcode',date:`2026-10-${String(day).padStart(2,'0')}`,occurred_at:`2026-10-${String(day).padStart(2,'0')}T00:00:00Z`,concepts:['a','b'],level,rating:null,y});
test('readiness needs own evidence, excludes dependent concepts, and reports waivers',()=>{
  assert.equal(cellReadiness({concept:'a',level:1,observations:[],catalog}).state,'not_ready');
  assert.equal(cellReadiness({concept:'a',level:1,observations:[obs('x',1,1)],catalog}).state,'ready');
  const r=conceptReadiness({concept:'b',observations:[obs('x',1,1)],catalog,prerequisites:{b:['a']}});
  assert.equal(r.prerequisites[0].state,'not_ready');
  assert.equal(conceptReadiness({concept:'b',observations:[],catalog,targets:['b'],prerequisites:{b:['a']}}).eligible,true);
  assert.equal(cellReadiness({concept:'missing',level:1,observations:[],catalog}).waiver,'no_catalog_coverage');
  assert.equal(cellReadiness({concept:'a',level:2,observations:[{...obs('m',1,0,2),problem_key:'m'}],catalog}).waiver,'exhausted');
  assert.throws(()=>validatePrerequisites({a:['b'],b:['a']}),/cyclic/);
});
test('follow-up closes on later same/lower solve and expires at fourteen days',()=>{
  const failure=obs('f',1,0,2);
  assert.equal(conceptFollowUp([failure],'a','2026-10-02').open,true);
  assert.equal(conceptFollowUp([failure,obs('hard',2,1,3)],'a','2026-10-02').open,true);
  assert.equal(conceptFollowUp([failure,obs('easy',2,1)],'a','2026-10-02').open,false);
  assert.equal(conceptFollowUp([failure],'a','2026-10-15').open,false);
});
