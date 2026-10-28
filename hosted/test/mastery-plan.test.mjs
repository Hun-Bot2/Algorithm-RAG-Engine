import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {atcoderInternalRating,atcoderSolveProbability,calibrationReport,codeforcesSolveProbability,fitAbility,fitLeetCode,fitAll,predict,priorFor} from '../mastery/calibration.mjs';
import {EXPLORE_KINDS,exploreIndex,planDay,relatedChannelFor,weekInfo} from '../mastery/planner.mjs';
import {relatedCandidates,SEMANTIC_ONLY_LABEL,trustedConcepts} from '../mastery/related.mjs';
import {catalogFromData} from '../mastery/catalog.mjs';
import {mockExposurePlans} from '../mastery/mock-contest.mjs';
import {normalizePreferences} from '../core/preferences.mjs';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/atcoder-calibration.json',import.meta.url)));

test('AtCoder Problems formulas match the independent Python reference within 1e-12 (MS-401)', () => {
  assert.equal(fixture.solve.length,15);
  for (const [key,difficulty,discrimination,theta,expected] of fixture.solve) assert.ok(Math.abs(atcoderSolveProbability(theta,difficulty,discrimination)-expected)<1e-12,key);
  for (const [rating,n,expected] of fixture.internal) assert.ok(Math.abs(atcoderInternalRating(rating,n)-expected)<1e-9,`${rating}/${n}`);
  assert.equal(atcoderInternalRating(1200,0),null,'no rated participation means no internal rating');
  assert.equal(codeforcesSolveProbability(1500,1500),0.5);
});

const at = (d,h=12) => `2026-10-${String(d).padStart(2,'0')}T${String(h).padStart(2,'0')}:00:00.000000Z`;
const cf = (key,rating,concepts=['greedy']) => ({key,platform:'codeforces',title:key,url:`https://codeforces.com/problemset/problem/${key.slice(11,-1)}/${key.slice(-1)}`,difficulty_kind:'cf_rating',difficulty_value:rating,concepts:concepts.map(id=>({id,status:'ai_reviewed'}))});
const lc = (slug,level,concepts=['arrays-hashing']) => ({key:`leetcode:${slug}`,platform:'leetcode',title:slug,url:`https://leetcode.com/problems/${slug}/`,difficulty_kind:'lc_level',difficulty_value:level,access:'free_at_collection',concepts:concepts.map(id=>({id,status:'ai_reviewed'}))});
const atc = (task,d,experimental=false) => ({key:`atcoder:${task}`,platform:'atcoder',title:task,url:`https://atcoder.jp/contests/${task.split('_')[0]}/tasks/${task}`,difficulty_kind:'atc_irt',difficulty_value:d,atc_discrimination:0.004479398673070138,atc_experimental:experimental,concepts:[{id:'greedy',status:'ai_reviewed'}]});

test('sparse history stays conservative and uncertain; evidence moves ability in the right direction', () => {
  const prior=priorFor('codeforces',[]);
  assert.deepEqual([prior.mean,prior.sd,prior.source],[1000,350,'default']);
  const empty=fitAbility('codeforces',[],prior);
  assert.equal(empty.status,'insufficient_evidence');
  const p=predict(cf('codeforces:1A',1000),{codeforces:empty});
  assert.ok(p.p_conservative<p.p_mean && p.p_mean<p.p_optimistic,'conservative estimate is below the marginal mean');
  assert.ok(p.p_mean>0.45 && p.p_mean<0.55);
  const wins=fitAbility('codeforces',[1,2,3].map(i=>({occurred_at:at(i),problem:cf('codeforces:1A',1200),y:1})),prior);
  const losses=fitAbility('codeforces',[1,2,3].map(i=>({occurred_at:at(i),problem:cf('codeforces:1A',800),y:0})),prior);
  assert.ok(wins.mean>1000 && losses.mean<1000);
  assert.ok(wins.sd<350,'evidence shrinks uncertainty');
  const drift=fitAbility('codeforces',[{occurred_at:at(1),problem:cf('codeforces:1A',1000),y:1},{occurred_at:'2026-12-31T00:00:00.000000Z',problem:cf('codeforces:1A',1000),y:1}],prior);
  assert.ok(drift.sd>fitAbility('codeforces',[{occurred_at:at(1),problem:cf('codeforces:1A',1000),y:1},{occurred_at:at(2),problem:cf('codeforces:1A',1000),y:1}],prior).sd,'elapsed time widens uncertainty');
  const atPrior=priorFor('atcoder',[{entry_id:'r',occurred_at:at(1),payload:{platform:'atcoder',rating:1200,rated_participations:10}}]);
  assert.ok(Math.abs(atPrior.mean-atcoderInternalRating(1200,10))<1e-9);
});

test('LeetCode calibration is ordered by level and reports insufficient evidence', () => {
  const fit=fitLeetCode([{problem:lc('a',2),y:1},{problem:lc('b',2),y:1},{problem:lc('c',2),y:1},{problem:lc('d',1),y:0},{problem:lc('e',1),y:0}]);
  assert.equal(fit.status,'insufficient_evidence');
  assert.ok(fit.levels[1].mean>=fit.levels[2].mean && fit.levels[2].mean>=fit.levels[3].mean,'PAV enforces Easy >= Medium >= Hard');
  const report=calibrationReport(Array.from({length:40},(_,i)=>({p:i%2?0.8:0.2,y:i%2})));
  assert.equal(report.status,'estimated'); assert.ok(report.beats_baseline);
  assert.equal(calibrationReport([{p:0.5,y:1}]).status,'insufficient_evidence');
});

function catalog() {
  const problems=[];
  for (let i=0;i<12;i++) problems.push(lc(`easy-${i}`,1,[i%2?'arrays-hashing':'greedy']),lc(`med-${i}`,2,[i%2?'arrays-hashing':'dynamic-programming']),lc(`hard-${i}`,3,['dynamic-programming']));
  for (let r=800;r<=2000;r+=100) problems.push(cf(`codeforces:${r}A`,r,['greedy']),cf(`codeforces:${r}B`,r,['arrays-hashing','dynamic-programming']));
  for (let d=-400;d<=1600;d+=200) problems.push(atc(`abc${500+d/100}_c`,d));
  problems.push({...lc('premium',1),access:'unknown'},{...lc('excluded',1),eligible:false});
  return catalogFromData({version:'catalog-v1',problems,aliases:{'leetcode:old-easy-0':'leetcode:easy-0'},
    concepts:['arrays-hashing','dynamic-programming','greedy'].map(id=>({id,label:id})),
    neighbors:{title_v2_cross:{'leetcode:med-1':['codeforces:1000B','codeforces:1600B']}}});
}
const prefs = (extra={}) => normalizePreferences({weekday_minutes:60,target_keys:[],...extra});
const attempt = (id,key,outcome,day,extra={}) => ({id,table:'events',occurred_at:at(day),recorded_at:at(day),problem_key:key,outcome,hints:outcome==='independent'?false:outcome==='reported_completion'?null:true,activity_type:null,transfer_from:null,evidence_source:'manual',self_minutes:null,...extra});
const ledgerOf = attempts => ({attempts,rating_observations:[],cutoff:at(31),high_water:{events:attempts.length}});
const MONDAY='2026-10-19', TUESDAY='2026-10-20';

test('cold start uses the placement pattern with conservative predictions; plans are deterministic', () => {
  const c=catalog();
  const plan=planDay({date:MONDAY,now:at(19),ledger:ledgerOf([]),catalog:c,preferences:prefs()});
  assert.equal(JSON.stringify(plan),JSON.stringify(planDay({date:MONDAY,now:at(19),ledger:ledgerOf([]),catalog:c,preferences:prefs()})),'deterministic');
  const exploit=plan.slots.filter(s=>s.kind==='exploit');
  assert.equal(exploit[0].sub_kind,'placement'); assert.equal(exploit[0].platform,'leetcode');
  assert.equal(exploit[0].components.placement_target,0.85);
  assert.equal(exploit[0].prediction.status,'insufficient_evidence');
  assert.ok(plan.slots.every(s=>!['leetcode:premium','leetcode:excluded'].includes(s.problem_key)),'unknown access and excluded rows never recommended');
  assert.equal(plan.abilities.codeforces.prior,'default');
  assert.match(plan.rotation_note,/not causal/);
  assert.equal(plan.policy_bundle,'mastery-policy-v2');
});

test('explore rotation is a fixed calendar schedule independent of outcomes', () => {
  const c=catalog();
  const indices=['2026-10-19','2026-10-21','2026-10-23','2026-10-26'].map(exploreIndex);
  assert.deepEqual(indices,indices.map((_,i)=>exploreIndex(['2026-10-19','2026-10-21','2026-10-23','2026-10-26'][i])));
  assert.equal(new Set(['2026-10-19','2026-10-20','2026-10-21','2026-10-22'].map(exploreIndex)).size,4,'consecutive weekdays cycle all sub-kinds');
  const good=ledgerOf(Array.from({length:8},(_,i)=>attempt('g'+i,`leetcode:easy-${i}`,'independent',10+i)));
  const bad=ledgerOf(Array.from({length:8},(_,i)=>attempt('b'+i,`leetcode:easy-${i}`,'failed',10+i)));
  const a=planDay({date:MONDAY,ledger:good,catalog:c,preferences:prefs()}), b=planDay({date:MONDAY,ledger:bad,catalog:c,preferences:prefs()});
  assert.equal(a.rotation.scheduled,b.rotation.scheduled,'outcomes do not change the scheduled sub-kind');
  assert.equal(a.rotation.scheduled,EXPLORE_KINDS[exploreIndex(MONDAY)]);
  assert.ok(['concept','title_v2','title_v2_cross'].includes(relatedChannelFor(MONDAY)));
  assert.deepEqual(weekInfo('2026-10-19'),{weekday:0,isoWeek:43,isoYear:2026,weekStart:'2026-10-19'});
});

test('weekly explore cap, remediation override, reviews first and duplicate prevention', () => {
  const c=catalog();
  const history=[
    ...Array.from({length:6},(_,i)=>attempt('p'+i,`leetcode:easy-${i}`,'independent',1+i)),
    attempt('rc','leetcode:med-0','reported_completion',8),
    attempt('alias','leetcode:old-easy-0','independent',9),
  ];
  const plan=planDay({date:MONDAY,ledger:ledgerOf(history),catalog:c,preferences:prefs()});
  const keys=plan.slots.map(s=>s.problem_key);
  for (let i=0;i<6;i++) assert.ok(!keys.includes(`leetcode:easy-${i}`) || plan.slots.find(s=>s.problem_key===`leetcode:easy-${i}`).kind==='review','solved problems are only reviews');
  assert.ok(!keys.includes('leetcode:med-0'),'reported completion counts as solved for duplicate prevention');
  // reviews: independent solves on day 1..6 have 1-day reviews due by Monday
  assert.equal(plan.slots[0].kind,'review');
  assert.ok(plan.slots.filter(s=>s.kind==='review').reduce((m,s)=>m+s.estimated_minutes,0)<=30,'practice day caps reviews at half the budget');
  assert.ok(plan.notes.some(n=>n.startsWith('review deferred')));
  assert.ok(plan.slots.some(s=>s.kind==='exploit'),'exploit is planned before explore when reviews use half the budget');
  assert.ok(plan.slots.reduce((m,s)=>m+s.estimated_minutes,0)<=60,'plan fits the weekday budget');
  // cap: two 30-minute explores already this week exhaust 25% of 5x60
  const capped=planDay({date:'2026-10-23',ledger:ledgerOf(history),catalog:c,preferences:prefs(),priorPlans:[{date:'2026-10-19',slots:[{kind:'explore',estimated_minutes:30,problem_key:'x:1'}]},{date:'2026-10-21',slots:[{kind:'explore',estimated_minutes:30,problem_key:'x:2'}]}]});
  assert.equal(capped.rotation.allowed,false); assert.ok(!capped.slots.some(s=>s.kind==='explore'));
  const remediation=planDay({date:MONDAY,ledger:ledgerOf([...history,attempt('rv','leetcode:easy-1','failed',18,{activity_type:'review'})]),catalog:c,preferences:prefs({weekday_minutes:120})});
  const slot=remediation.slots.find(s=>s.kind==='remediation');
  assert.ok(slot,'failed review replaces exploration'); assert.equal(remediation.rotation.used,'remediation');
  assert.ok(slot.concepts.includes('arrays-hashing'));
  // exposure in an earlier plan within the cooldown blocks re-recommendation
  const first=plan.slots.find(s=>s.kind==='exploit');
  const next=planDay({date:TUESDAY,ledger:ledgerOf(history),catalog:c,preferences:prefs(),priorPlans:[plan]});
  assert.ok(!next.slots.some(s=>s.kind!=='review' && s.problem_key===first.problem_key));
});

test('weekend days produce contest and upsolve slots; mastery settings validate', () => {
  const c=catalog();
  const saturday=planDay({date:'2026-10-24',ledger:ledgerOf([]),catalog:c,preferences:prefs(),contestProposals:{live:[],virtual:null}});
  assert.equal(saturday.day_type,'virtual'); assert.equal(saturday.slots[0].kind,'contest');
  assert.equal(planDay({date:'2026-10-25',ledger:ledgerOf([]),catalog:c,preferences:prefs()}).slots[0].kind,'upsolve');
  assert.throws(()=>normalizePreferences({mastery:{explore_cap:0.9}}),/explore cap/);
  assert.throws(()=>normalizePreferences({mastery:{contest_platforms:['leetcode']}}),/platforms/);
  assert.throws(()=>normalizePreferences({mastery:{bogus:1}}),/mastery settings/);
  assert.equal(normalizePreferences({mastery:{cooldown_days:14}}).mastery.cooldown_days,14);
  assert.equal('mastery' in normalizePreferences({}),false,'absent stays absent (backward compatible)');
});

test('related practice requires concept overlap and labels evidence; embedding rank alone is not enough', () => {
  const c=catalog();
  const abilities=fitAll([],c,[]);
  const result=relatedCandidates({sourceKey:'leetcode:med-1',catalog:c,abilities,channel:'title_v2_cross',limit:5});
  assert.ok(result.candidates.length>0);
  for (const cand of result.candidates) { assert.ok(cand.shared_concepts.length>0); assert.match(cand.evidence_label,/AI-reviewed concept match/); }
  const top=result.candidates.find(x=>x.problem_key==='codeforces:1000B');
  assert.ok(top.components.channel_rank>0,'channel rank contributes');
  const isolated=catalogFromData({problems:[lc('src',2,['tries']),cf('codeforces:900A',900,['greedy'])],neighbors:{title_v2:{'leetcode:src':['codeforces:900A']}}});
  assert.equal(relatedCandidates({sourceKey:'leetcode:src',catalog:isolated,abilities:fitAll([],isolated,[]),channel:'title_v2'}).candidates.length,0,'zero-overlap neighbor rejected (R5.2)');
  assert.match(result.note,/not claimed to be algorithmically equivalent/);
});

const unlabeled = (task,d,extra={}) => ({...atc(task,d),concepts:[],reliability:'reliable',...extra});

test('semantic fallback: unclassified candidates appear only from a title channel, labelled, and never as concept evidence', () => {
  const src=cf('codeforces:900A',900,['greedy']);
  const near=unlabeled('abc300_c',1000), far=unlabeled('abc300_d',1000), proposed=unlabeled('abc300_e',1000,{concepts:[{id:'sorting',status:'proposed'}]});
  const mismatch={...atc('abc300_f',1000),concepts:[{id:'trees',status:'ai_reviewed'}]};
  const c=catalogFromData({problems:[src,near,far,proposed,mismatch],neighbors:{title_v2_cross:{'codeforces:900A':['atcoder:abc300_c','atcoder:abc300_e','atcoder:abc300_f']}}});
  const abilities=fitAll([],c,[]);
  const result=relatedCandidates({sourceKey:'codeforces:900A',catalog:c,abilities,channel:'title_v2_cross',limit:10});
  const keys=result.candidates.map(x=>x.problem_key);
  assert.deepEqual(keys.sort(),['atcoder:abc300_c','atcoder:abc300_e'],'unlabeled or proposed-only neighbors qualify; a labeled non-matching one and a non-neighbor do not');
  for (const cand of result.candidates) {
    assert.equal(cand.semantic_only,true);
    assert.equal(cand.evidence_label,SEMANTIC_ONLY_LABEL);
    assert.deepEqual(cand.shared_concepts,[]);
    assert.equal(cand.components.semantic_only,-1.5);
    assert.equal(cand.components.concept_jaccard,0);
    assert.ok(cand.components.channel_rank>0);
  }
  assert.equal(trustedConcepts(c.problems.get('atcoder:abc300_e')).size,0,'proposed is never trusted');
  const conceptOnly=relatedCandidates({sourceKey:'codeforces:900A',catalog:c,abilities,channel:'concept',limit:10});
  assert.equal(conceptOnly.candidates.length,0,'the concept channel never yields unclassified problems');
});

test('semantic fallback requires reliable titles on both sides and ranks below a concept match', () => {
  const src=cf('codeforces:900A',900,['greedy']);
  const degraded=unlabeled('abc301_c',1000,{reliability:'degraded'});
  const good=unlabeled('abc301_d',1000);
  const matched={...atc('abc301_e',1000)};
  const c=catalogFromData({problems:[src,degraded,good,matched],neighbors:{title_v2_cross:{'codeforces:900A':['atcoder:abc301_c','atcoder:abc301_d','atcoder:abc301_e']}}});
  const result=relatedCandidates({sourceKey:'codeforces:900A',catalog:c,abilities:fitAll([],c,[]),channel:'title_v2_cross',limit:10});
  assert.deepEqual(result.candidates.map(x=>x.problem_key),['atcoder:abc301_e','atcoder:abc301_d']);
  assert.equal(result.candidates[0].semantic_only,false);
  const degradedSource=catalogFromData({problems:[{...src,reliability:'degraded'},good],neighbors:{title_v2_cross:{'codeforces:900A':['atcoder:abc301_d']}}});
  assert.equal(relatedCandidates({sourceKey:'codeforces:900A',catalog:degradedSource,abilities:fitAll([],degradedSource,[]),channel:'title_v2_cross'}).candidates.length,0);
});

test('mock contest problems are prior exposures for new daily plans within the cooldown (mock-contest R6.4)', () => {
  const c=catalog();
  const base=planDay({date:MONDAY,now:at(19),ledger:ledgerOf([]),catalog:c,preferences:prefs()});
  const picked=base.slots.map(s=>s.problem_key).filter(Boolean);
  assert.ok(picked.length>0);
  const mock=date=>mockExposurePlans([{local_date:date,problems:picked.map(problem_key=>({problem_key}))}]);
  const after=planDay({date:MONDAY,now:at(19),ledger:ledgerOf([]),catalog:c,preferences:prefs(),priorPlans:mock('2026-10-17')});
  assert.ok(after.slots.every(s=>!picked.includes(s.problem_key)),'Saturday mock contest problems are not re-recommended on Monday');
  const old=planDay({date:MONDAY,now:at(19),ledger:ledgerOf([]),catalog:c,preferences:prefs(),priorPlans:mock('2026-08-01')});
  assert.deepEqual(old.slots.map(s=>s.problem_key),base.slots.map(s=>s.problem_key),'outside the cooldown the exposure expires');
});
