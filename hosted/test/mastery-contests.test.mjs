import test from 'node:test';
import assert from 'node:assert/strict';
import {contestKind,fittingKinds,inWeekendWindow,liveProposals,virtualProposal} from '../mastery/contests.mjs';
import {catalogFromData} from '../mastery/catalog.mjs';
import {fitAll} from '../mastery/calibration.mjs';
import {DEFAULT_MASTERY} from '../core/preferences.mjs';
import {fetchCodeforcesRatings,ratingEntries} from '../mastery/ratings.mjs';
import {createDatabase} from './helpers/d1.mjs';
import {appendLedger} from '../mastery/side-ledger.mjs';
import {readLedger} from '../mastery/ledger.mjs';

const mastery=structuredClone(DEFAULT_MASTERY);
test('contest kinds, conservative fit and weekend windows across KST midnight', () => {
  assert.equal(contestKind('atcoder','abc430'),'abc');
  assert.equal(contestKind('codeforces','2100','Codeforces Round 1050 (Div. 2)'),'div2');
  assert.equal(contestKind('codeforces','2101','Codeforces Round (Div. 1 + Div. 2)'),'div1+2');
  assert.equal(contestKind('codeforces','2102','Educational Codeforces Round 180 (Rated for Div. 2)'),'div2','explicit division wins');
  const sparse=fittingKinds({codeforces:{mean:1500,sd:350},atcoder:{mean:1300,sd:400}});
  assert.ok(!sparse.has('div2') && !sparse.has('arc'),'sparse history: mean - sd keeps proposals easy');
  assert.ok(fittingKinds({codeforces:{mean:1600,sd:100},atcoder:{mean:1300,sd:50}}).has('div2'));
  // ABC: Saturday 21:00 KST = 12:00Z
  assert.equal(inWeekendWindow('2026-10-24T12:00:00Z',mastery.weekend_window),true);
  // Saturday 23:35 KST is inside; Sunday 00:30 KST (Sat 15:30Z) is outside the Sunday window
  assert.equal(inWeekendWindow('2026-10-24T14:35:00Z',mastery.weekend_window),true);
  assert.equal(inWeekendWindow('2026-10-24T15:30:00Z',mastery.weekend_window),false);
  assert.equal(inWeekendWindow('2026-10-23T12:00:00Z',mastery.weekend_window),false,'Friday');
  const abilities={codeforces:{mean:1000,sd:350},atcoder:{mean:0,sd:400}};
  const live=liveProposals({calendar:[
    {id:'atcoder:abc430',platform:'atcoder',title:'ABC 430',url:'https://atcoder.jp/contests/abc430',start_at:'2026-10-24T12:00:00Z',duration_minutes:100},
    {id:'codeforces:2100',platform:'codeforces',title:'Codeforces Round (Div. 2)',url:'https://codeforces.com/contest/2100',start_at:'2026-10-24T14:35:00Z',duration_minutes:120},
    {id:'codeforces:2103',platform:'codeforces',title:'Codeforces Round (Div. 3)',url:'https://codeforces.com/contest/2103',start_at:'2026-10-25T14:35:00Z',duration_minutes:135}],abilities,mastery,from:'2026-10-24',to:'2026-10-25'});
  assert.deepEqual(live.map(l=>[l.contest_id,l.proposed]),[['atcoder:abc430',true],['codeforces:2103',true],['codeforces:2100',false]]);
  assert.match(live[0].acceptance,/Proposal only/);
});

const problem=(key,platform,d,concepts=[])=>({key,platform,title:key,url:'https://example.invalid',difficulty_kind:platform==='atcoder'?'atc_irt':'cf_rating',difficulty_value:d,atc_discrimination:0.004479398673070138,concepts:concepts.map(id=>({id,status:'ai_reviewed'}))});
test('virtual contests exclude any contest with an attempted (alias-resolved) problem and report unknown coverage', () => {
  const problems=[], contestProblems={};
  for (const [id,base] of [['abc100',-800],['abc101',-600],['abc102',-400]]) {
    contestProblems[id]=['a','b','c','d'].map((l,i)=>({problem_key:`atcoder:${id}_${l}`,label:l.toUpperCase()}));
    contestProblems[id].forEach((p,i)=>problems.push(problem(p.problem_key,'atcoder',base+i*400)));
  }
  contestProblems['1900']=['A','B','C'].map(l=>({problem_key:`codeforces:1900${l}`,label:l}));
  problems.push(problem('codeforces:1900A','codeforces',800,['greedy']),problem('codeforces:1900B','codeforces',1000,['dynamic-programming']),problem('codeforces:1900C','codeforces',1300,['greedy']));
  const catalog=catalogFromData({problems,contestProblems,aliases:{'atcoder:legacy_c':'atcoder:abc101_c'},
    contests:[{contest_id:'abc100',platform:'atcoder',title:'ABC 100',kind:'abc',start_at:'2018-06-16T12:00:00Z',url:'https://atcoder.jp/contests/abc100'},
      {contest_id:'abc102',platform:'atcoder',title:'ABC 102',kind:'abc',start_at:'2026-10-20T12:00:00Z',url:'https://atcoder.jp/contests/abc102'},
      {contest_id:'1900',platform:'codeforces',title:'Codeforces Round (Div. 3)',kind:'div3',start_at:'2025-12-01T14:35:00Z',url:'https://codeforces.com/contest/1900'}]});
  const attempts=[{problem_key:'atcoder:legacy_c',outcome:'reported_completion',occurred_at:'2026-01-01T00:00:00.000000Z'}];
  const result=virtualProposal({catalog,attempts,abilities:fitAll([],catalog,[]),mastery,date:'2026-10-24',weakConcepts:['greedy']});
  const ids=result.candidates.map(c=>c.contest_id);
  assert.ok(!ids.includes('abc101'),'alias-resolved solved problem excludes the contest');
  assert.ok(!ids.includes('abc102'),'contests younger than 14 days are excluded');
  const cf=result.candidates.find(c=>c.contest_id==='1900'), atc=result.candidates.find(c=>c.contest_id==='abc100');
  assert.equal(cf.weakness_coverage,1); assert.equal(atc.weakness_coverage,'unknown');
  assert.equal(atc.components.weakness_coverage,0);
  assert.deepEqual(ids,['1900','abc100'],'weakness coverage and recency rank Codeforces first');
  assert.match(result.note,/only selects and records/);
});

test('Codeforces rating sync maps history to rating observations and stops on refusal', async context => {
  const body={status:'OK',result:[{contestId:1,contestName:'r1',rank:500,ratingUpdateTimeSeconds:1700000000,oldRating:0,newRating:400},{contestId:2,contestName:'r2',rank:300,ratingUpdateTimeSeconds:1700600000,oldRating:400,newRating:700}]};
  const changes=await fetchCodeforcesRatings('tourist_x',{fetcher:async url=>{ assert.equal(url,'https://codeforces.com/api/user.rating?handle=tourist_x'); return new Response(JSON.stringify(body)); }});
  const entries=ratingEntries('owner','tourist_x',changes,'2026-10-17T00:00:00.000000Z');
  assert.deepEqual(entries.map(e=>[e.entry_id,e.payload.rating,e.payload.rated_participations]),[['cf-rating-tourist_x-1',400,1],['cf-rating-tourist_x-2',700,2]]);
  const db=createDatabase(); context.after(()=>db.close());
  assert.equal((await appendLedger(db,'owner',{rating_observations:entries})).rating_observations,2);
  assert.equal((await appendLedger(db,'owner',{rating_observations:entries})).rating_observations,0,'re-sync is idempotent');
  assert.equal((await readLedger(db,'owner')).rating_observations.at(-1).payload.rating,700);
  await assert.rejects(fetchCodeforcesRatings('x',{fetcher:async()=>new Response('',{status:429})}),/refused/);
  await assert.rejects(fetchCodeforcesRatings('bad handle!',{fetcher:async()=>{throw new Error('must not call');}}),/invalid/);
});
