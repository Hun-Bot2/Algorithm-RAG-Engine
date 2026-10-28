import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultConfig,loadProgressionConfig} from '../mastery/progression-v2-config.mjs';

test('authored config is frozen, copied and has the specified constants',()=>{
  const input=structuredClone(defaultConfig), config=loadProgressionConfig(input);
  assert.equal(config.version,'progression-v2-config-1');
  assert.equal(config.tau,0.7);
  assert.deepEqual(config.level_priors,{'1':[3.2,0.8],'2':[2,2],'3':[0.8,3.2]});
  assert.deepEqual(config.review_days,[1,3,7,14,30,60]);
  assert.ok(Object.isFrozen(config) && Object.isFrozen(config.level_priors['1']) && Object.isFrozen(config.weights));
  assert.ok(Object.isFrozen(config.review_days) && Object.isFrozen(config.transfer_platforms));
  assert.equal(Object.isFrozen(input.level_priors['1']),false);
  input.level_priors['1'][0]=99;
  assert.equal(config.level_priors['1'][0],3.2);
  assert.throws(()=>{config.weights.coverage=1;},TypeError);
});

test('malformed configuration is rejected',()=>{
  const cases=[
    c=>{c.tau=0;}, c=>{c.tau=1;}, c=>{c.tau=NaN;}, c=>{c.credible_mass=1;},
    c=>{c.readiness_probability=-0.1;}, c=>{c.kappa=0;}, c=>{c.lambda=1.1;},
    c=>{c.level_priors['1']=[1,0];}, c=>{c.level_priors['2']=[1];},
    c=>{c.weights.coverage=-1;}, c=>{c.weights.information=Infinity;},
    c=>{c.review_days=[1,3,3];}, c=>{c.review_days=[];}, c=>{c.follow_up_days=1.5;},
    c=>{c.transfer_platforms=['codeforces','codeforces'];}, c=>{c.transfer_platforms=['leetcode'];},
    c=>{c.transfer_native_difficulty_floor=NaN;}, c=>{c.exclude_experimental_atcoder='true';},
    c=>{delete c.version;}, c=>{c.unexpected=1;},
  ];
  for (const change of cases) {
    const input=structuredClone(defaultConfig); change(input);
    assert.throws(()=>loadProgressionConfig(input),TypeError);
  }
});

test('sparse arrays cannot bypass finite-number validation',()=>{
  const priors=structuredClone(defaultConfig);priors.level_priors['1']=Array(2);
  assert.throws(()=>loadProgressionConfig(priors),/level_priors/);
  const reviews=structuredClone(defaultConfig);reviews.review_days=[,3];
  assert.throws(()=>loadProgressionConfig(reviews),/review_days/);
});
