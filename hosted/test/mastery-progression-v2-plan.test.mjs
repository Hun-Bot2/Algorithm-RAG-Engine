import test from 'node:test';
import assert from 'node:assert/strict';
import {planProgressionV2} from '../mastery/progression-v2.mjs';
import {catalogFromData} from '../mastery/catalog.mjs';
const problems=[];
for(const c of ['A','B'])for(const l of [1,2,3])for(let i=0;i<12;i++)problems.push({key:`leetcode:${c}-${l}-${i}`,platform:'leetcode',difficulty_kind:'lc_level',difficulty_value:l,tags:[String(i)],concepts:[{id:c,status:'approved'}]});
for(const c of ['A','B'])problems.push({key:`codeforces:${c}`,platform:'codeforces',difficulty_kind:'cf_rating',difficulty_value:800,concepts:[{id:c,status:'approved'}]});
const catalog=catalogFromData({problems});
test('S1-S8 reproduce accepted focus/action scenarios',()=>{
for(const [name,E,M,transfer] of [['S1',[],[],false],['S2',[1,1],[],false],['S3',[1,1],[1,1],false],['S4',[1,1],[0],false],['S5',[1,1],[1,1,1,1],true],['S6',[1,0,0],[],false],['S7',[1,1,1,1,1],[1,1,1,1,1,1],true],['S8',[1,1],[0],false]]){
 let day=1;const attempts=[];
 for(const [l,ys] of [[1,E],[2,M]])for(const [i,y] of ys.entries()){attempts.push({id:`a${day}`,problem_key:`leetcode:A-${l}-${i}`,outcome:y?'independent':'failed',occurred_at:`2026-10-${String(day++).padStart(2,'0')}T00:00:00Z`});}
 if(transfer)attempts.push({id:'t',problem_key:'codeforces:A',outcome:'independent',occurred_at:'2026-10-15T00:00:00Z'});
 const p=planProgressionV2({date:name==='S8'?'2026-10-30':'2026-10-16',ledger:{attempts},catalog,prerequisites:{}});
 const expected={S1:['A','consolidate'],S2:['A','step_up'],S3:['A','transfer'],S4:['A','step_down'],S5:['B','consolidate'],S6:['A','consolidate'],S7:['B','consolidate'],S8:['A','step_down']};
 assert.deepEqual([p.next.concept,p.next.action],expected[name],name);
 assert.equal(p.next.config_sha256.length,64);
 assert.deepEqual(planProgressionV2({date:name==='S8'?'2026-10-30':'2026-10-16',ledger:{attempts},catalog,prerequisites:{}}),p);
 assert.ok(p.next.estimate.interval[0]<=p.next.estimate.p&&p.next.estimate.p<=p.next.estimate.interval[1]);
 assert.equal(p.next.components.F, name==='S4'||name==='S6'?1:0);

}

});


test('candidate insertion order, coverage and pure-concept ties are deterministic',()=>{
  const args={date:'2026-10-16',ledger:{attempts:[]},catalog,prerequisites:{}};
  const plan=planProgressionV2(args);
  const reversed=catalogFromData({problems:problems.toReversed()});
  assert.deepEqual(planProgressionV2({...args,catalog:reversed}),plan);
  assert.ok(!JSON.stringify(plan).includes('semantic_only'));
  const seen=planProgressionV2({...args,priorPlans:[{date:args.date,slots:[{problem_key:plan.next.problem_key}]}]});
  assert.notEqual(seen.next.problem_key,plan.next.problem_key);
});

test('finite generated ledgers cannot deadlock with any remaining eligible candidate',()=>{
  for(let i=0;i<30;i++){
    const attempts=problems.filter(p=>p.platform==='leetcode').slice(0,i).map((p,j)=>({id:`a${j}`,problem_key:p.key,occurred_at:`2026-10-${String(j%10+1).padStart(2,'0')}T00:00:00Z`,outcome:j%3?'independent':'failed'}));
    const plan=planProgressionV2({date:'2026-10-16',ledger:{attempts},catalog,prerequisites:{B:['A']}});
    assert.ok(plan.next);assert.ok(plan.next.problem_key);
  }
});


test('future ratings do not leak into historical plans',()=>{
  const c=catalogFromData({problems:problems.filter(p=>p.platform==='codeforces')});
  const args={date:'2026-01-01',catalog:c,ledger:{attempts:[]},prerequisites:{}};
  const baseline=planProgressionV2(args);
  const withFuture=planProgressionV2({...args,ledger:{attempts:[],rating_observations:[{entry_id:'future',occurred_at:'2026-02-01T00:00:00Z',recorded_at:'2026-02-01T00:00:00Z',payload:{platform:'codeforces',rating:3000}}]}});
  assert.deepEqual(withFuture,baseline);
});

test('equivalent cells prefer unseen variation and then concept purity',()=>{
  const base={platform:'leetcode',difficulty_kind:'lc_level',difficulty_value:1,tags:['unseen']};
  const c=catalogFromData({problems:[
    {...base,key:'leetcode:medium',difficulty_value:2,concepts:[{id:'A',status:'approved'}]},
    {...base,key:'leetcode:pure',concepts:[{id:'A',status:'approved'}]},
    {...base,key:'leetcode:mixed',concepts:[{id:'A',status:'approved'},{id:'B',status:'approved'}]},
    {...base,key:'leetcode:old',tags:['old'],concepts:[{id:'A',status:'approved'}]},
    {...base,key:'leetcode:seen',tags:['old'],concepts:[{id:'A',status:'approved'}]},
  ]});
  const attempts=[{id:'first',problem_key:'leetcode:old',outcome:'independent',occurred_at:'2026-10-01T00:00:00Z'}];
  const out=planProgressionV2({date:'2026-10-02',ledger:{attempts},catalog:c,prerequisites:{},preferences:{mastery:{target_concepts:['A']}}});
  assert.equal(out.next.problem_key,'leetcode:pure');assert.equal(out.next.tie_break.coverage_gain,1);assert.equal(out.next.tie_break.purity,1);
});
