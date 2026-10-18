import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {catalogFromData} from '../mastery/catalog.mjs';
import {planProgression,PROGRESSION_PATTERN_VERSION} from '../mastery/progression.mjs';

const concept='arrays-hashing';
const lc=(n,difficulty='easy',tag=`variation-${n}`)=>({key:`leetcode:${difficulty}-${n}`,platform:'leetcode',title:`${difficulty} ${n}`,url:`https://example.test/${difficulty}-${n}`,difficulty_kind:'lc_level',difficulty_value:difficulty==='easy'?1:2,access:'free_at_collection',tags:[tag],concepts:[{id:concept,status:'ai_reviewed'}]});
const cf=(n)=>({key:`codeforces:${n}A`,platform:'codeforces',title:`CF ${n}`,url:'https://example.test/cf',difficulty_kind:'cf_rating',difficulty_value:800,concepts:[{id:concept,status:'ai_reviewed'}]});
const atc=(n,concepts=[{id:concept,status:'ai_reviewed'}])=>({key:`atcoder:abc${n}_a`,platform:'atcoder',title:`AtCoder ${n}`,url:'https://example.test/atc',difficulty_kind:'atc_irt',difficulty_value:500,concepts,reliability:'reliable'});
const problemSet=()=>[...Array.from({length:8},(_,i)=>lc(i)),...Array.from({length:5},(_,i)=>lc(i,'medium',`medium-${i}`)),cf(100),atc(100),atc(101,[])];
const attempt=(id,key,day,{outcome='independent',activity_type='practice',transfer_from=null,hints=false}={})=>({id,problem_key:key,occurred_at:`2026-10-${String(day).padStart(2,'0')}T12:00:00.000000Z`,outcome,hints,activity_type,transfer_from});
const make=(attempts=[],opts={})=>{
  const catalog=catalogFromData({version:'catalog-test',problems:problemSet(),aliases:{'leetcode:legacy-easy-0':'leetcode:easy-0'},concepts:[{id:concept,label:'Arrays and hashing'}],neighbors:{title_v2:{'leetcode:medium-1':['atcoder:abc101_a']}}});
  return planProgression({date:'2026-10-30',ledger:{attempts,cutoff:opts.cutoff ?? '2026-10-30T23:59:59.000000Z'},catalog,curriculum:opts.curriculum ?? null,priorPlans:opts.priorPlans ?? []});
};
const easyPasses=()=>[0,1,2].map((i)=>attempt(`e${i}`,`leetcode:easy-${i}`,i+1));
const mediumPasses=()=>[0,1].map((i)=>attempt(`m${i}`,`leetcode:medium-${i}`,i+5));
const progressionConfig=()=>JSON.parse(readFileSync(new URL('../mastery/progression-pattern.v1.json',import.meta.url),'utf8'));

test('versioned pattern starts with an eligible LeetCode Easy variation',()=>{
  const out=make();
  assert.equal(out.pattern_version,PROGRESSION_PATTERN_VERSION);
  assert.equal(out.concepts[concept].next_stage,'leetcode_easy');
  assert.equal(out.concepts[concept].next_candidate,'leetcode:easy-0');
  assert.match(out.concepts[concept].next_candidate_reason,/variation metadata/);
});

test('repeated attempts and aliases do not count as distinct problems or variations',()=>{
  const out=make([attempt('a','leetcode:easy-0',1),attempt('b','leetcode:legacy-easy-0',2),attempt('c','leetcode:easy-0',3)]);
  assert.equal(out.concepts[concept].completed_counts.leetcode_easy,1);
  assert.equal(out.concepts[concept].stage_index,0);
});

test('three different Easy problem IDs with identical variation metadata do not pass the variation gate',()=>{
  const attempts=[0,1,2].map(i=>attempt(`same${i}`,`leetcode:easy-${i}`,i+1));
  const catalog=catalogFromData({version:'x',problems:problemSet().map(p=>p.platform==='leetcode'&&p.difficulty_value===1?{...p,tags:['same']} : p),concepts:[{id:concept,label:concept}]});
  const out=planProgression({date:'2026-10-30',ledger:{attempts,cutoff:'2026-10-30T23:59:59Z'},catalog});
  assert.equal(out.concepts[concept].completed_counts.leetcode_easy,3);
  assert.equal(out.concepts[concept].completed_counts.leetcode_easy_variations,1);
  assert.equal(out.concepts[concept].stage_index,0);
});

test('three distinct Easy variations unlock the Medium stage',()=>{
  const state=make(easyPasses()).concepts[concept];
  assert.equal(state.next_stage,'leetcode_medium');
  assert.equal(state.completed_counts.leetcode_easy_variations,3);
});

test('a fourth distinct Easy variation can recover after the first three were too similar',()=>{
  const attempts=[0,1,2,3,4,5].map(i=>attempt(`same${i}`,`leetcode:easy-${i}`,i+1));
  const catalog=catalogFromData({version:'x',problems:problemSet().map(p=>p.platform==='leetcode'&&p.difficulty_value===1?{...p,tags:p.key==='leetcode:easy-0'||p.key==='leetcode:easy-1'||p.key==='leetcode:easy-2'?['same']:[p.key]} : p),concepts:[{id:concept,label:concept}]});
  const out=planProgression({date:'2026-10-30',ledger:{attempts,cutoff:null},catalog});
  assert.equal(out.concepts[concept].next_stage,'leetcode_medium');
});

test('one Medium solve is insufficient; two distinct Medium variations unlock transfer',()=>{
  const one=make([...easyPasses(),attempt('m0','leetcode:medium-0',6)]).concepts[concept];
  assert.equal(one.next_stage,'leetcode_medium');
  const two=make([...easyPasses(),...mediumPasses()]).concepts[concept];
  assert.equal(two.next_stage,'cross_platform_transfer');
  assert.ok(['codeforces:100A','atcoder:abc100_a'].includes(two.next_candidate));
});

test('approved prerequisite blocks the target concept until the prerequisite reaches transfer evidence',()=>{
  const source='arrays-hashing', target='two-pointers';
  const catalog=catalogFromData({version:'x',problems:[...problemSet(),{...lc(20),key:'leetcode:target-easy',concepts:[{id:target,status:'approved'}]},{...lc(21),key:'leetcode:source-easy',concepts:[{id:source,status:'approved'}]}],concepts:[{id:source,label:source},{id:target,label:target}]});
  const curriculum={relationships:[{type:'prerequisite',source:'leetcode:source-easy',target:'leetcode:target-easy',review:{status:'approved'}}]};
  const out=planProgression({date:'2026-10-30',ledger:{attempts:[],cutoff:null},catalog,curriculum});
  assert.equal(out.concepts[target].status,'blocked_prerequisite');
  assert.equal(out.concepts[target].prerequisites_met,false);
});

test('pattern prerequisite maps are explicit and only the named prerequisite concept gates progress',()=>{
  const config=progressionConfig();
  config.prerequisites[concept]=['missing-prerequisite'];
  const out=planProgression({date:'2026-10-30',ledger:{attempts:[],cutoff:null},catalog:catalogFromData({version:'x',problems:problemSet(),concepts:[{id:concept,label:concept}]}),config});
  assert.equal(out.concepts[concept].status,'blocked_prerequisite');
});

test('concept selection follows interview order, then prioritizes configured target concepts',()=>{
  const extra=[
    {...lc(20),key:'leetcode:string-easy',concepts:[{id:'string-processing',status:'approved'}]},
    {...lc(21),key:'leetcode:sorting-easy',concepts:[{id:'sorting',status:'approved'}]},
  ];
  const catalog=catalogFromData({version:'x',problems:[...problemSet(),...extra],concepts:[{id:concept,label:concept},{id:'string-processing',label:'Strings'},{id:'sorting',label:'Sorting'}]});
  const config=progressionConfig(); config.concept_order=['string-processing','arrays-hashing','sorting']; config.target_concepts=['sorting'];
  const out=planProgression({date:'2026-10-30',ledger:{attempts:[],cutoff:null},catalog,config});
  assert.equal(out.next.concept,'sorting');
  assert.equal(out.stages[0].concepts[0].concept,'string-processing');
});

test('unresolved remediation outranks target and interview order',()=>{
  const extra=[{...lc(20),key:'leetcode:sorting-easy',concepts:[{id:'sorting',status:'approved'}]}];
  const catalog=catalogFromData({version:'x',problems:[...problemSet(),...extra],concepts:[{id:concept,label:concept},{id:'sorting',label:'Sorting'}]});
  const config=progressionConfig(); config.concept_order=['sorting',concept]; config.target_concepts=['sorting'];
  const out=planProgression({date:'2026-10-30',ledger:{attempts:[attempt('fail',`leetcode:easy-0`,1,{outcome:'failed'})],cutoff:null},catalog,config});
  assert.equal(out.concepts[concept].status,'remediation');
  assert.equal(out.next.concept,concept);
});

test('Medium threshold is configurable while invalid counts and variation ratios are rejected',()=>{
  const catalog=catalogFromData({version:'x',problems:problemSet(),concepts:[{id:concept,label:concept}]});
  const one=progressionConfig(); one.pattern[1].distinct_required=1; one.pattern[1].variation_signatures_required=1;
  const attempts=[...easyPasses(),attempt('m','leetcode:medium-0',6)];
  const out=planProgression({date:'2026-10-30',ledger:{attempts,cutoff:null},catalog,config:one});
  assert.equal(out.concepts[concept].next_stage,'cross_platform_transfer');
  const invalid=progressionConfig(); invalid.pattern[1].distinct_required=0;
  assert.throws(()=>planProgression({date:'2026-10-30',ledger:{attempts:[],cutoff:null},catalog,config:invalid}),/positive integers/);
  const invalidRatio=progressionConfig(); invalidRatio.pattern[1].distinct_required=1; invalidRatio.pattern[1].variation_signatures_required=2;
  assert.throws(()=>planProgression({date:'2026-10-30',ledger:{attempts:[],cutoff:null},catalog,config:invalidRatio}),/variation count no greater/);
});

test('four Easy and three Medium problems can be configured as a valid journey',()=>{
  const catalog=catalogFromData({version:'x',problems:problemSet(),concepts:[{id:concept,label:concept}]});
  const config=progressionConfig(); config.pattern[0].distinct_required=4; config.pattern[0].variation_signatures_required=4; config.pattern[1].distinct_required=3; config.pattern[1].variation_signatures_required=3;
  const attempts=[...Array.from({length:4},(_,i)=>attempt(`e${i}`,`leetcode:easy-${i}`,i+1)),...Array.from({length:3},(_,i)=>attempt(`m${i}`,`leetcode:medium-${i}`,i+6))];
  const out=planProgression({date:'2026-10-30',ledger:{attempts,cutoff:null},catalog,config});
  assert.equal(out.concepts[concept].next_stage,'cross_platform_transfer');
  assert.equal(out.concepts[concept].completed_counts.leetcode_easy,4);
  assert.equal(out.concepts[concept].completed_counts.leetcode_medium,3);
});

test('approved problem-key prerequisite relationships map to their trusted concepts',()=>{
  const source='source-concept', target='target-concept';
  const sourceProblem={...lc(30),key:'leetcode:source',concepts:[{id:source,status:'approved'}]};
  const targetProblem={...lc(31),key:'leetcode:target',concepts:[{id:target,status:'approved'}]};
  const catalog=catalogFromData({version:'x',problems:[...problemSet(),sourceProblem,targetProblem],concepts:[{id:source,label:source},{id:target,label:target}]});
  const curriculum={relationships:[{type:'prerequisite',source:'leetcode:source',target:'leetcode:target',review:{status:'approved'}}]};
  const out=planProgression({date:'2026-10-30',ledger:{attempts:[],cutoff:null},catalog,curriculum});
  assert.deepEqual(out.concepts[target].prerequisites,[source]);
  assert.equal(out.concepts[target].status,'blocked_prerequisite');
});

test('independent cross-platform solve must reference an already solved Medium problem',()=>{
  const attempts=[...easyPasses(),...mediumPasses(),attempt('bad','codeforces:100A',9,{activity_type:'transfer',transfer_from:'leetcode:medium-4'})];
  const state=make(attempts).concepts[concept];
  assert.equal(state.next_stage,'cross_platform_transfer');
  assert.equal(state.completed_counts.transfer,false);
});

test('successful CF transfer followed by a review at least seven days later reaches retention',()=>{
  const attempts=[...easyPasses(),...mediumPasses(),attempt('x','codeforces:100A',8,{activity_type:'transfer',transfer_from:'leetcode:medium-1'}),attempt('r','codeforces:100A',16,{activity_type:'review'})];
  const state=make(attempts).concepts[concept];
  assert.equal(state.status,'retention_evidenced');
  assert.equal(state.completed_counts.retention_review,true);
});

test('a later failure regresses progress and reports remediation; later evidence can recover',()=>{
  const progressed=[...easyPasses(),...mediumPasses(),attempt('x','codeforces:100A',8,{activity_type:'transfer',transfer_from:'leetcode:medium-1'}),attempt('r','codeforces:100A',16,{activity_type:'review'}),attempt('f','codeforces:100A',20,{outcome:'failed',activity_type:'review'})];
  const regressed=make(progressed).concepts[concept];
  assert.equal(regressed.status,'remediation');
  assert.equal(regressed.next_stage,'retention_review');
  const recovered=make([...progressed,attempt('r2','codeforces:100A',28,{activity_type:'review'})]).concepts[concept];
  assert.equal(recovered.status,'retention_evidenced');
});

test('a failed Medium attempt preserves completed Easy evidence and schedules Medium repair',()=>{
  const state=make([...easyPasses(),attempt('medium-fail','leetcode:medium-0',6,{outcome:'failed'})]).concepts[concept];
  assert.equal(state.status,'remediation');
  assert.equal(state.next_stage,'leetcode_medium');
  assert.equal(state.completed_counts.leetcode_easy,3);
  assert.equal(state.completed_counts.leetcode_easy_variations,3);
  assert.equal(state.completed_counts.leetcode_medium,0);
});

test('a failed LeetCode Hard attempt is reported without downgrading Easy or Medium evidence',()=>{
  const hard={key:'leetcode:hard-0',platform:'leetcode',title:'Hard problem',url:'https://example.test/hard',difficulty_kind:'lc_level',difficulty_value:3,access:'free_at_collection',tags:['hard-variation'],concepts:[{id:concept,status:'ai_reviewed'}]};
  const catalog=catalogFromData({version:'x',problems:[...problemSet(),hard],concepts:[{id:concept,label:concept}]});
  const attempts=[...easyPasses(),...mediumPasses(),attempt('hard-fail','leetcode:hard-0',8,{outcome:'failed'})];
  const state=planProgression({date:'2026-10-30',ledger:{attempts,cutoff:null},catalog}).concepts[concept];
  assert.equal(state.status,'in_progress');
  assert.equal(state.next_stage,'cross_platform_transfer');
  assert.equal(state.completed_counts.leetcode_easy,3);
  assert.equal(state.completed_counts.leetcode_medium,2);
  assert.deepEqual(state.advanced_evidence,{attempts:1,independent:0,failed:1,assisted:0,status:'observed_not_mastered'});
});

test('future and post-cutoff attempts are excluded from the derived progression',()=>{
  const attempts=[...easyPasses(),attempt('future','leetcode:medium-0',31)];
  const state=make(attempts,{cutoff:'2026-10-02T00:00:00.000000Z'}).concepts[concept];
  assert.equal(state.completed_counts.leetcode_easy,1);
  assert.equal(state.stage_index,0);
});

test('prior recommendations are exposures and semantic AtCoder fallback never credits transfer',()=>{
  const attempts=[...easyPasses(),...mediumPasses()];
  const state=make(attempts,{priorPlans:[{date:'2026-10-25',slots:[{problem_key:'codeforces:100A'}]}]}).concepts[concept];
  assert.equal(state.next_stage,'cross_platform_transfer');
  assert.equal(state.next_candidate,'atcoder:abc100_a');
  assert.equal(state.completed_counts.transfer,false);
});

test('already attempted transfer targets cannot be suggested as novel transfer evidence',()=>{
  const attempts=[...easyPasses(),...mediumPasses(),attempt('seen-cf','codeforces:100A',9),attempt('seen-atc','atcoder:abc100_a',10)];
  const state=make(attempts).concepts[concept];
  assert.equal(state.next_stage,'cross_platform_transfer');
  assert.equal(state.next_candidate,null);
  assert.equal(state.completed_counts.transfer,false);
});

test('a failed transfer target cannot later count; recovery needs a new target after re-earning Medium',()=>{
  const attempts=[...easyPasses(),...mediumPasses(),
    attempt('failed-x','codeforces:100A',8,{outcome:'failed',activity_type:'transfer',transfer_from:'leetcode:medium-1'}),
    attempt('repeat-x','codeforces:100A',9,{activity_type:'transfer',transfer_from:'leetcode:medium-1'}),
    attempt('m2','leetcode:medium-2',10),attempt('m3','leetcode:medium-3',11),
    attempt('new-x','codeforces:101A',12,{activity_type:'transfer',transfer_from:'leetcode:medium-3'})];
  const catalog=catalogFromData({version:'x',problems:[...problemSet(),cf(101)],concepts:[{id:concept,label:concept}]});
  const state=planProgression({date:'2026-10-30',ledger:{attempts,cutoff:null},catalog}).concepts[concept];
  assert.equal(state.completed_counts.transfer,true);
});

test('transfer selection stays within conservative native difficulty and carries its successful source',()=>{
  const high={...cf(99),key:'codeforces:99A',difficulty_value:10000};
  const catalog=catalogFromData({version:'x',problems:[...problemSet(),high],concepts:[{id:concept,label:concept}]});
  const out=planProgression({date:'2026-10-30',ledger:{attempts:[...easyPasses(),...mediumPasses()],cutoff:null},catalog});
  assert.notEqual(out.next.problem_key,high.key);
  assert.equal(out.next.stage,'cross_platform_transfer');
  assert.equal(out.next.kind,'transfer');
  assert.equal(out.next.transfer_from,'leetcode:medium-1');
  assert.ok(out.next.prediction.p_conservative<=1 && out.next.prediction.p_conservative>=0);
});

test('semantic AtCoder neighbor is surfaced only as unverified fallback and never as concept evidence',()=>{
  const problems=[...problemSet().filter(p=>p.platform==='leetcode'),atc(101,[])];
  const catalog=catalogFromData({version:'x',problems,concepts:[{id:concept,label:concept}],neighbors:{title_v2:{'leetcode:medium-1':['atcoder:abc101_a']}}});
  const out=planProgression({date:'2026-10-30',ledger:{attempts:[...easyPasses(),...mediumPasses()],cutoff:null},catalog});
  assert.equal(out.next.problem_key,'atcoder:abc101_a');
  assert.equal(out.next.semantic_only,true);
  assert.equal(out.next.status,'unverified');
  assert.equal(out.concepts[concept].completed_counts.transfer,false);
});

test('missing variation candidates are reported without inventing coverage',()=>{
  const attempts=[...easyPasses()];
  const catalog=catalogFromData({version:'x',problems:attempts.map(a=>{const p=problemSet().find(p=>p.key===a.problem_key);return {...p,tags:['same']};}),concepts:[{id:concept,label:concept}]});
  const state=planProgression({date:'2026-10-30',ledger:{attempts,cutoff:null},catalog}).concepts[concept];
  assert.equal(state.next_candidate,null);
  assert.equal(state.next_stage,'leetcode_easy');
  assert.match(state.next_candidate_reason,/No eligible unseen candidate/);
});
