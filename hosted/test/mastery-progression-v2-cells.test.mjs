import test from 'node:test';
import assert from 'node:assert/strict';
import reference from './fixtures/adaptive-cells.v1.json' with {type:'json'};
import {leetCodeCells,platformAbilities,platformPrediction} from '../mastery/progression-v2.mjs';
test('cells match independent leave-concept-out, fractional attribution and PAV reference',()=>{
  assert.ok(reference.rows.length>=8);
  for(const row of reference.rows){
    const actual=leetCodeCells(row.observations,row.concept);
    for(let i=0;i<3;i++){
      const a=actual[i],b=row.cells[i];
      for(const field of ['alpha','beta','p'])assert.ok(Math.abs(a[field]-b[field])<1e-9,`${row.name}: ${field}`);
      for(let j=0;j<2;j++)assert.ok(Math.abs(a.interval[j]-b.interval[j])<1e-9,`${row.name} interval`);
      assert.equal(a.n_own,b.n_own);assert.equal(a.n_level_outside,b.n_level_outside);assert.equal(a.prior,b.prior);
      assert.ok(a.interval[0]<=a.p&&a.p<=a.interval[1]);
    }
  }
});
test('retention, times and review metadata do not change ability cells',()=>{
  const rows=[{kind:'ability',level:1,concepts:['a'],y:1}];
  const expected=leetCodeCells(rows,'a');
  assert.deepEqual(leetCodeCells([...rows,...Array.from({length:20},()=>({...rows[0],kind:'retention'}))],'a'),expected);
  assert.deepEqual(leetCodeCells(rows.map(o=>({...o,self_minutes:500,review:true,memo:'fail'})),'a'),expected);
});


test('platform fitting ignores retention; prediction intervals bound the marginal mean',()=>{
  const problem={key:'codeforces:1A',platform:'codeforces',difficulty_value:800,difficulty_kind:'cf_rating'};
  const catalog={problems:new Map([[problem.key,problem]])};
  const o={kind:'ability',platform:'codeforces',problem_key:problem.key,occurred_at:'2026-10-01T00:00:00Z',y:1};
  const one=platformAbilities([o],catalog);
  assert.deepEqual(platformAbilities([o,...Array.from({length:19},()=>({...o,kind:'retention'}))],catalog),one);
  for(let i=0;i<100;i++){
    const abilities={codeforces:{...one.codeforces,mean:-1000+i*50,sd:50+i*7}};
    const p=platformPrediction(problem,abilities);
    assert.ok(p.interval[0]<=p.p&&p.p<=p.interval[1]);
    assert.ok(p.information>=0&&p.information<=1);
  }
});
