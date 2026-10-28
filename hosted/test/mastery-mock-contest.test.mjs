import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {candidatePool,composeLevels,fnv1a,mockContest,MOCK_CONTEST_VERSION,mockExposurePlans,selectProblems,snapshotExposures} from '../mastery/mock-contest.mjs';
import {catalogFromData} from '../mastery/catalog.mjs';
import {fitAll} from '../mastery/calibration.mjs';

const EMPTY_CATALOG={problems:new Map(),aliases:new Map()};
const POOLS={1:792,2:1538,3:723};
// Synthetic LeetCode abilities: p_conservative = mean - sd.
const estimated=(e,m,h)=>({leetcode:{status:'estimated',n:12,levels:{1:{mean:e,sd:0},2:{mean:m,sd:0},3:{mean:h,sd:0}}}});

test('T1 prior-only composition leans easy and is labelled insufficient evidence', () => {
  const result=composeLevels({abilities:fitAll([],EMPTY_CATALOG,[]),pools:POOLS});
  assert.equal(result.policy,MOCK_CONTEST_VERSION);
  assert.deepEqual(result.probabilities,{1:{p_mean:0.8,p_conservative:0.621},2:{p_mean:0.5,p_conservative:0.276},3:{p_mean:0.2,p_conservative:0.021}});
  assert.deepEqual(result.composition,{easy:5,medium:1,hard:0});
  assert.equal(result.target_expected,3.6);
  assert.equal(result.expected_solved,3.381);
  assert.equal(result.ability_status,'insufficient_evidence');
});

test('T1 composition follows the estimated level probabilities', () => {
  const result=composeLevels({abilities:estimated(0.95,0.6,0.2),pools:POOLS});
  assert.deepEqual(result.composition,{easy:1,medium:4,hard:1});
  assert.equal(result.expected_solved,3.55);
  assert.equal(result.ability_status,'estimated');
  const seven=composeLevels({abilities:estimated(0.95,0.6,0.2),pools:POOLS,count:7});
  assert.equal(seven.target_expected,4.2);
  assert.equal(Object.values(seven.composition).reduce((a,b)=>a+b,0),7);
});

test('T1 ties prefer more Hard, then more Medium', () => {
  // Equal probabilities make every composition tie on |E - T|.
  assert.deepEqual(composeLevels({abilities:estimated(0.5,0.5,0.5),pools:POOLS}).composition,{easy:1,medium:1,hard:4});
  assert.deepEqual(composeLevels({abilities:estimated(0.5,0.5,0.5),pools:{1:10,2:10,3:2}}).composition,{easy:1,medium:3,hard:2});
});

test('T1 respects pool sizes and fails instead of returning a partial contest', () => {
  const limited=composeLevels({abilities:fitAll([],EMPTY_CATALOG,[]),pools:{1:2,2:100,3:100}});
  assert.deepEqual(limited.composition,{easy:2,medium:4,hard:0});
  assert.equal(limited.expected_solved,2.346);
  assert.throws(()=>composeLevels({abilities:estimated(0.9,0.5,0.1),pools:{1:0,2:100,3:100}}),/not enough unattempted/);
  assert.throws(()=>composeLevels({abilities:estimated(0.9,0.5,0.1),pools:{1:1,2:1,3:2}}),/not enough unattempted/);
});

test('T1 rejects invalid counts and pool sizes', () => {
  const abilities=estimated(0.9,0.5,0.1);
  for (const count of [4,8,5.5,'6',null]) assert.throws(()=>composeLevels({abilities,pools:POOLS,count}),/count must be an integer/);
  assert.throws(()=>composeLevels({abilities,pools:{1:10,2:10}}),/invalid mock contest pool sizes/);
  assert.throws(()=>composeLevels({abilities,pools:{1:10,2:-1,3:10}}),/invalid mock contest pool sizes/);
});

test('T1 module stays pure: no Node-only imports', () => {
  const source=readFileSync(new URL('../mastery/mock-contest.mjs',import.meta.url),'utf8');
  const imports=[...source.matchAll(/^import .* from '([^']+)';$/gm)].map(m=>m[1]);
  assert.deepEqual(imports,['./calibration.mjs','./related.mjs']);
});

const lc=(slug,level,extra={})=>({key:`leetcode:${slug}`,platform:'leetcode',title:slug,url:`https://leetcode.com/problems/${slug}`,difficulty_kind:'lc_level',difficulty_value:level,access:'free_at_collection',...extra});
const reviewed=(...ids)=>ids.map(id=>({id,status:'ai_reviewed'}));
const DATE='2026-10-31';
function fixtureCatalog(rename=null) {
  const problems=[
    lc('e1',1,{tags:['Array','Hash Table']}),lc('e2',1,{tags:['Array','Hash Table']}),lc('e3',1,{tags:['String']}),lc('e4',1,{tags:['Math']}),
    lc('m1',2,{tags:['Graph']}),lc('m2',2,{tags:['Tree']}),lc('m3',2,{tags:['Dynamic Programming'],concepts:reviewed('dp')}),
    lc('m4',2,{tags:['Greedy'],concepts:[{id:'dp',status:'proposed'}]}),lc('h1',3,{tags:['Heap']}),lc('h2',3,{tags:['Trie']}),
    lc('premium',2,{access:'premium'}),lc('pandas-only',1),lc('ineligible',2,{eligible:false}),lc('retired',2,{status:'retired'}),lc('unrated',2,{difficulty_kind:'none',difficulty_value:null}),
    {key:'codeforces:4A',platform:'codeforces',title:'Watermelon',difficulty_kind:'cf_rating',difficulty_value:800},
  ];
  return catalogFromData({problems:rename?problems.map(p=>({...p,title:rename(p.title)})):problems,aliases:{'leetcode:e4-old':'leetcode:e4'}});
}
const keys=list=>list.map(p=>p.key ?? p.problem_key);

test('T2 pool excludes attempted, aliased, exposed, premium, ineligible, unrated and other platforms', () => {
  const catalog=fixtureCatalog();
  const pool=candidatePool({catalog,date:DATE,cooldownDays:30,
    attempts:[{problem_key:'leetcode:e3',outcome:'failed'},{problem_key:'leetcode:e4-old',outcome:'reported_completion'}],
    exposures:[{date:'2026-10-28',problem_keys:['leetcode:m1']},{date:'2026-09-01',problem_keys:['leetcode:m2']},{date:DATE,problem_keys:['leetcode:h1']}]});
  assert.deepEqual(keys(pool.byLevel[1]),['leetcode:e1','leetcode:e2']);
  assert.deepEqual(keys(pool.byLevel[2]),['leetcode:m2','leetcode:m3','leetcode:m4'],'exposure older than cooldown is allowed again');
  assert.deepEqual(keys(pool.byLevel[3]),['leetcode:h1','leetcode:h2'],'same-date exposure is not prior');
  assert.deepEqual(pool.sizes,{1:2,2:3,3:2});
  assert.deepEqual(pool.excluded,{attempted:2,exposed:1,untagged:1},'untagged non-algorithm exercises are excluded');
  assert.throws(()=>candidatePool({catalog,date:'2026/10/31',cooldownDays:30}),/invalid mock contest date/);
  assert.throws(()=>candidatePool({catalog,date:DATE,cooldownDays:-1}),/invalid cooldown days/);
});

test('T2 trusted weak concepts rank first; proposed concepts and unclassified problems are unknown', () => {
  const pool=candidatePool({catalog:fixtureCatalog(),date:DATE,cooldownDays:30});
  const {problems}=selectProblems({pool,composition:{easy:1,medium:2,hard:1},weakConcepts:['dp'],learnerId:'learner-a',date:DATE});
  const m3=problems.find(p=>p.problem_key==='leetcode:m3');
  assert.equal(problems.filter(p=>p.level===2)[0].problem_key,'leetcode:m3');
  assert.deepEqual(m3.components,{weak_concepts_covered:['dp'],weakness_coverage:1});
  assert.match(m3.reason,/Covers weak concepts: dp/);
  for (const p of problems.filter(p=>p.problem_key!=='leetcode:m3')) assert.equal(p.components.weakness_coverage,'unknown');
  assert.deepEqual(problems.map(p=>p.slot),[1,2,3,4]);
  assert.deepEqual(problems.map(p=>p.level),[1,2,2,3]);
});

test('T2 identical tag sets are avoided, then relaxed with a note when a level would run short', () => {
  const pool=candidatePool({catalog:fixtureCatalog(),date:DATE,cooldownDays:30});
  const easy=selectProblems({pool,composition:{easy:3,medium:1,hard:1},learnerId:'learner-a',date:DATE}).problems.filter(p=>p.level===1);
  const tagSets=easy.map(p=>p.tags.join('|'));
  assert.equal(new Set(tagSets).size,3,'e1/e2 share a tag set, so one is replaced by e3 or e4');
  const tight=candidatePool({catalog:fixtureCatalog(),date:DATE,cooldownDays:30,attempts:[{problem_key:'leetcode:e3'},{problem_key:'leetcode:e4'}]});
  const relaxed=selectProblems({pool:tight,composition:{easy:2,medium:1,hard:1},learnerId:'learner-a',date:DATE});
  assert.deepEqual(relaxed.problems.filter(p=>p.level===1).map(p=>p.problem_key).sort(),['leetcode:e1','leetcode:e2']);
  assert.deepEqual(relaxed.notes,['Easy: tag-set diversity relaxed to fill 2 problems.']);
  assert.throws(()=>selectProblems({pool:tight,composition:{easy:3,medium:1,hard:1},learnerId:'learner-a',date:DATE}),/not enough unattempted/);
});

test('T2 selection is deterministic, seeded, and ignores titles', () => {
  assert.equal(fnv1a(''),0x811c9dc5);
  assert.equal(fnv1a('a'),0xe40c292c);
  const run=(catalog,date=DATE,composition={easy:1,medium:2,hard:1})=>selectProblems({pool:candidatePool({catalog,date,cooldownDays:30}),composition,learnerId:'learner-a',date});
  const first=run(fixtureCatalog());
  assert.deepEqual(run(fixtureCatalog()),first);
  assert.deepEqual(run(fixtureCatalog(title=>`Renamed ${title.toUpperCase()}`)),first);
  const many=catalogFromData({problems:[...Array.from({length:20},(_,i)=>lc(`e${String(i).padStart(2,'0')}`,1,{tags:[`T${i}`]})),lc('m',2,{tags:['M']}),lc('h',3,{tags:['H']})]});
  const picks=new Set(['2026-10-31','2026-11-07','2026-11-14','2026-11-21','2026-11-28'].map(date=>run(many,date,{easy:1,medium:1,hard:1}).problems[0].problem_key));
  assert.ok(picks.size>1,'the seed varies the choice by date instead of always taking the first key');
});

const SAT='2026-10-31';

const MASTERY={cooldown_days:30};
function recordCatalog() {
  const problems=[];
  for (const [level,prefix,n] of [[1,'easy',12],[2,'medium',10],[3,'hard',6]]) for (let i=0;i<n;i++) problems.push(lc(`${prefix}-${i}`,level,{tags:[`${prefix}-tag-${i}`]}));
  problems.find(p=>p.key==='leetcode:medium-4').concepts=reviewed('graph');
  problems.push({key:'codeforces:4A',platform:'codeforces',title:'Watermelon',difficulty_kind:'cf_rating',difficulty_value:800},
    {key:'atcoder:abc100_a',platform:'atcoder',title:'Happy Birthday!',difficulty_kind:'atc_irt',difficulty_value:-500});
  return catalogFromData({version:'catalog-fixture',problems});
}
const attempt=(id,key,outcome='independent',extra={})=>({id,problem_key:key,outcome,hints:outcome==='independent'?false:true,activity_type:'practice',occurred_at:'2026-10-20T12:00:00.000000Z',...extra});
const build=(extra={})=>mockContest({catalog:recordCatalog(),learnerId:'synthetic-mock',date:SAT,mastery:MASTERY,...extra});

test('T4 prior-only record carries policy, insufficient-evidence label and display fields', () => {
  const record=build({
    exposures:[{date:'2026-10-25',problem_keys:['leetcode:easy-0']},{date:'2026-08-01',problem_keys:['leetcode:easy-1']}]});
  assert.equal(record.policy,MOCK_CONTEST_VERSION);
  assert.equal(record.catalog_version,'catalog-fixture');
  assert.equal(record.local_date,SAT);
  assert.equal(record.ability.status,'insufficient_evidence');
  assert.match(record.evidence_label,/insufficient LeetCode evidence \(0 practice observations\)/);
  assert.deepEqual(record.composition,{easy:5,medium:1,hard:0});
  assert.equal(record.duration_minutes,120);
  for (const field of ['start_at','placement_note']) assert.ok(!(field in record),`v2 records have no ${field}`);
  assert.ok(!('busy_contests' in record.inputs));
  assert.deepEqual(record.inputs.exposure_dates,['2026-10-25']);
  assert.deepEqual(record.inputs.excluded,{attempted:0,exposed:1,untagged:0});
  assert.ok(!record.problems.some(p=>p.problem_key==='leetcode:easy-0'));
  for (const p of record.problems) {
    assert.equal(p.url,`https://leetcode.com/problems/${p.problem_key.slice('leetcode:'.length)}`);
    assert.deepEqual([p.p_mean,p.p_conservative],p.level===1?[0.8,0.621]:[0.5,0.276]);
  }
  assert.equal(build({count:7}).duration_minutes,120,'solving time does not scale with the count');
  assert.equal(build({minutes:90}).duration_minutes,90);
  for (const minutes of [29,301,90.5,'120']) assert.throws(()=>build({minutes}),/minutes must be an integer/);
  assert.throws(()=>build({date:'2026-10-31T00:00'}),/invalid mock contest date/);
});

test('T4 uses only LeetCode evidence and is byte-stable', () => {
  const base=build();
  const withOtherPlatforms=build({attempts:[attempt('cf-1','codeforces:4A'),attempt('atc-1','atcoder:abc100_a','failed')]});
  assert.equal(JSON.stringify(withOtherPlatforms),JSON.stringify(base));
  assert.equal(JSON.stringify(build()),JSON.stringify(base));
});

test('T4 estimated LeetCode evidence changes the mix and excludes attempted problems', () => {
  const attempts=[...Array.from({length:6},(_,i)=>attempt(`lc-e${i}`,`leetcode:easy-${i}`)),
    attempt('lc-m0','leetcode:medium-0'),attempt('lc-m1','leetcode:medium-1'),attempt('lc-h0','leetcode:hard-0','failed'),
    attempt('lc-review','leetcode:medium-2','independent',{activity_type:'review'}),attempt('cf-1','codeforces:4A')];
  const record=build({attempts,weakConcepts:['graph','graph','dp']});
  assert.equal(record.ability.status,'estimated');
  assert.equal(record.ability.n,9,'review attempts are not calibration observations');
  assert.equal(record.evidence_label,null);
  assert.deepEqual(record.inputs.attempt_ids,['lc-e0','lc-e1','lc-e2','lc-e3','lc-e4','lc-e5','lc-h0','lc-m0','lc-m1','lc-review']);
  assert.deepEqual(record.inputs.excluded,{attempted:10,exposed:0,untagged:0});
  assert.deepEqual(record.weak_concepts,['dp','graph']);
  const attempted=new Set(attempts.map(a=>a.problem_key));
  assert.ok(record.problems.every(p=>!attempted.has(p.problem_key)));
  assert.notDeepEqual(record.composition,build().composition);
  assert.equal(record.problems.reduce((n,p)=>n+(p.level===1?1:0),0),record.composition.easy);
  const medium=record.problems.filter(p=>p.level===2);
  assert.deepEqual(record.composition,{easy:2,medium:4,hard:0});
  assert.equal(medium[0].problem_key,'leetcode:medium-4','the reviewed graph problem ranks first');
});

test('T4 rejects a missing learner id', () => {
  assert.throws(()=>mockContest({catalog:recordCatalog(),date:SAT,mastery:MASTERY}),/invalid learner id/);
});

test('T6 snapshot exposures read slots and mock problems, never archive wrappers', () => {
  const rows=[
    {kind:'plan',local_date:'2026-10-19',payload:{date:'2026-10-19',slots:[{problem_key:'leetcode:a'},{kind:'contest',problem_key:null}]}},
    {kind:'journal-abc',local_date:'2026-10-20',payload:{date:'2026-10-20',slots:[{problem_key:'leetcode:b'}]}},
    {kind:'mock_contest',local_date:'2026-10-24',payload:{local_date:'2026-10-24',problems:[{problem_key:'leetcode:c'}]}},
    {kind:'archive-1',local_date:'2026-10-01',payload:{archive_type:'recommendation_snapshot',original_kind:'plan',original_payload:{date:'2026-10-01',slots:[{problem_key:'leetcode:d'}]}}},
  ];
  assert.deepEqual(snapshotExposures(rows),[
    {date:'2026-10-19',problem_keys:['leetcode:a']},{date:'2026-10-20',problem_keys:['leetcode:b']},{date:'2026-10-24',problem_keys:['leetcode:c']}]);
  assert.deepEqual(mockExposurePlans([{local_date:'2026-10-24',problems:[{problem_key:'leetcode:c',title:'ignored'}]}]),
    [{date:'2026-10-24',source_kind:'mock_contest',slots:[{problem_key:'leetcode:c'}]}]);
});
