import test from 'node:test';
import assert from 'node:assert/strict';
import reference from './fixtures/beta-reference.v1.json' with {type:'json'};
import {betaCdf,betaSurvival,betaQuantile,betaVariance,betaInformation} from '../mastery/beta-math.mjs';
test('beta math matches independent SciPy CDF/quantile references',()=>{
  assert.ok(reference.rows.length>=40);
  for(const r of reference.rows){
    assert.ok(Math.abs(betaCdf(r.x,r.alpha,r.beta)-r.cdf)<1e-9,JSON.stringify(r));
    assert.ok(Math.abs(betaQuantile(r.quantile_probability,r.alpha,r.beta)-r.quantile)<1e-7,JSON.stringify(r));
    assert.ok(Math.abs(betaSurvival(r.x,r.alpha,r.beta)+r.cdf-1)<1e-9);
    const cdf=betaCdf(r.x,r.alpha,r.beta);
    if(cdf>1e-10&&cdf<1-1e-10)assert.ok(Math.abs(betaQuantile(cdf,r.alpha,r.beta)-r.x)<1e-7);
  }
});
test('information value is exact expected relative posterior variance reduction',()=>{
  for(const [a,b] of [[.5,.5],[3.2,.8],[20,60]]){
    const p=a/(a+b),v=betaVariance(a,b);
    const expected=p*betaVariance(a+1,b)+(1-p)*betaVariance(a,b+1);
    assert.ok(Math.abs((v-expected)/v-betaInformation(a,b))<1e-12);
    assert.equal(betaQuantile(.5,a,b),betaQuantile(.5,a,b));
  }
  assert.throws(()=>betaCdf(.5,0,1),/positive/);
  assert.throws(()=>betaQuantile(-1,1,1),/probability/);
});
