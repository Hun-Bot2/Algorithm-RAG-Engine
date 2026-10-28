// Characterization of the current local policy, not an endorsement of score omission.
import test from 'node:test';
import assert from 'node:assert/strict';
import {catalogFromData} from '../mastery/catalog.mjs';
import {fitAll} from '../mastery/calibration.mjs';
import {relatedCandidates} from '../mastery/related.mjs';
import {planProgression} from '../mastery/progression.mjs';
import {explanationEvidence} from '../mastery/explanations.mjs';
const concept='arrays-hashing';
const problem=(key,platform='leetcode')=>({key,platform,title:key,difficulty_kind:platform==='leetcode'?'lc_level':'cf_rating',difficulty_value:platform==='leetcode'?1:800,tags:[key],concepts:[{id:concept,status:'ai_reviewed'}]});
test('audit: Easy progression selects by eligible key order without reading vector neighbors',()=>{
 const catalog=catalogFromData({problems:[problem('leetcode:z'),problem('leetcode:a')],concepts:[{id:concept,label:concept}]});
 catalog.neighbors=()=>{throw Error('vector channel must not be called by Easy stage');};
 const out=planProgression({date:'2026-10-23',ledger:{attempts:[],cutoff:'2026-10-23T12:00:00Z'},catalog});
 assert.equal(out.concepts[concept].next_candidate,'leetcode:a');
});
test('audit: related retrieval uses ranks while raw cosine values do not affect current scoring',()=>{
 const catalog=catalogFromData({problems:[problem('leetcode:source'),problem('codeforces:1A','codeforces'),problem('codeforces:2A','codeforces')]});
 let scores=[0.99,0.1];
 catalog.neighbors=()=>['codeforces:1A','codeforces:2A'].map((dst,i)=>({dst,rank:i+1,score:scores[i]}));
 const run=()=>relatedCandidates({sourceKey:'leetcode:source',catalog,abilities:fitAll([],catalog,[]),channel:'title_v2',platforms:['codeforces'],limit:2}).candidates;
 const before=run();assert.equal(before.length,2);
 scores=[-0.99,0.999];assert.deepEqual(run(),before);
 assert.ok(before[0].components.channel_rank>before[1].components.channel_rank);
 assert.equal('cosine_similarity' in before[0],false);
});
test('audit: current AI evidence includes policy reason but omits vector values and score components',()=>{
 const evidence=explanationEvidence({date:'2026-10-23',catalog_version:'fixture',policy_bundle:'fixture'},{problem_key:'leetcode:a',kind:'practice',reason:'policy reason',components:{channel_rank:1.9},channel_rank:1,cosine_similarity:0.99});
 assert.equal(evidence.policy_reason,'policy reason');
 for(const key of ['components','channel_rank','cosine_similarity'])assert.equal(key in evidence,false);
});
