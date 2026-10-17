// outcomes-v1 (design.md section 7, R8). Pure. Observational, single-learner metrics with
// explicit samples and Wilson intervals; never presented as causal effects of the system.
import {REVIEW_DAYS,seoulDate} from '../core/policy.mjs';
import {confirmedSeconds,timerState} from '../core/events.mjs';
import {calibrationReport,fitAll} from './calibration.mjs';
import {canonicalKey,trustedConcepts} from './related.mjs';

export const OUTCOMES_VERSION = 'outcomes-v1';
export const MINIMUMS = {isr:10,excess:10,transfer:8,transfer_channel:8,retention_stage:8,weakness:4};
const DAY=86400000;
const addDays=(date,n)=>new Date(Date.parse(date+'T00:00:00Z')+n*DAY).toISOString().slice(0,10);
const dateOf=iso=>seoulDate(new Date(iso));
const solved=a=>a.outcome==='independent' && a.hints===false;
const known=a=>['independent','assisted','failed'].includes(a.outcome);
const r3=x=>x===null || x===undefined ? null : Math.round(x*1000)/1000;

export function wilson(successes,n,z=1.96) {
  if (!n) return null;
  const p=successes/n, d=1+z*z/n, centre=(p+z*z/(2*n))/d, half=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;
  return [r3(Math.max(0,centre-half)),r3(Math.min(1,centre+half))];
}
function rate(successes,n,minimum) {
  return {successes,n,rate:n?r3(successes/n):null,interval:wilson(successes,n),minimum,status:n>=minimum?'estimated':'insufficient_evidence'};
}
// The prediction stored when the problem was recommended (latest plan on/before the attempt, <=14 days).
export function storedPrediction(plans,key,attemptDate) {
  let best=null;
  for (const plan of plans) if (plan.date<=attemptDate && plan.date>=addDays(attemptDate,-14)) for (const slot of plan.slots ?? [])
    if (slot.problem_key===key && slot.prediction?.p_mean!==null && slot.prediction?.p_mean!==undefined && (!best || plan.date>best.date)) best={date:plan.date,slot};
  return best;
}
// First known-outcome, non-review attempt on a problem never attempted before (new practice).
export function firstAttempts(attempts,catalog) {
  const touched=new Set(), out=[];
  for (const a of attempts) {
    const key=canonicalKey(catalog,a.problem_key);
    if (touched.has(key)) continue;
    touched.add(key);
    if (known(a) && a.activity_type!=='review') out.push({...a,key,problem:catalog.problems.get(key) ?? null,date:dateOf(a.occurred_at)});
  }
  return out;
}
// Review outcomes per interval stage, replaying the same state machine as policy.mjs.
export function reviewOutcomes(attempts,catalog) {
  const byKey=new Map(), out=[];
  for (const a of attempts) { const key=canonicalKey(catalog,a.problem_key); if (!byKey.has(key)) byKey.set(key,[]); byKey.get(key).push(a); }
  for (const [key,list] of byKey) {
    let stage=0, due=null;
    for (const a of list) {
      const date=dateOf(a.occurred_at);
      if (a.outcome==='reported_completion') continue;
      if (a.activity_type==='review' && due) out.push({key,date,interval:REVIEW_DAYS[stage],on_time:Math.abs((Date.parse(date)-Date.parse(due))/DAY)<=1,success:solved(a)});
      if (solved(a)) { stage=a.activity_type==='review' ? Math.min(stage+1,REVIEW_DAYS.length-1) : 0; due=addDays(date,REVIEW_DAYS[stage]); }
      else if (a.activity_type==='review') { stage=0; due=addDays(date,1); }
    }
  }
  return out;
}

export function computeOutcomes({ledger,catalog,plans=[],weekStart,windowWeeks=4}) {
  const weekEnd=addDays(weekStart,6), windowStart=addDays(weekStart,-7*(windowWeeks-1));
  const attempts=ledger.attempts;
  const inRange=(d,from,to)=>d>=from && d<=to;
  const firsts=firstAttempts(attempts,catalog);
  const scopes={week:[weekStart,weekEnd],rolling:[windowStart,weekEnd]};
  const isr={}, excess={};
  for (const [scope,[from,to]] of Object.entries(scopes)) {
    const rows=firsts.filter(a=>inRange(a.date,from,to));
    const byPlatform={};
    for (const a of rows) { const p=a.problem?.platform ?? a.key.split(':')[0]; (byPlatform[p] ??= {s:0,n:0}); byPlatform[p].n++; byPlatform[p].s+=solved(a)?1:0; }
    isr[scope]={overall:rate(rows.filter(solved).length,rows.length,MINIMUMS.isr),by_platform:Object.fromEntries(Object.entries(byPlatform).sort().map(([p,v])=>[p,rate(v.s,v.n,MINIMUMS.isr)]))};
    const withPrediction=rows.map(a=>({a,stored:storedPrediction(plans,a.key,a.date)})).filter(x=>x.stored);
    const residual=withPrediction.reduce((s,x)=>s+((solved(x.a)?1:0)-x.stored.slot.prediction.p_mean),0);
    excess[scope]={n:withPrediction.length,without_stored_prediction:rows.length-withPrediction.length,
      excess_solve_rate:withPrediction.length?r3(residual/withPrediction.length):null,minimum:MINIMUMS.excess,status:withPrediction.length>=MINIMUMS.excess?'estimated':'insufficient_evidence'};
  }
  // Concepts (rolling window) for weaknesses.
  const concepts={};
  for (const a of attempts) {
    const d=dateOf(a.occurred_at);
    if (!known(a) || !inRange(d,addDays(weekEnd,-27),weekEnd)) continue;
    for (const c of trustedConcepts(catalog.problems.get(canonicalKey(catalog,a.problem_key)))) {
      const v=concepts[c] ??= {s:0,n:0,review_failures:0};
      v.n++; if (solved(a)) v.s++; else if (a.activity_type==='review') v.review_failures++;
    }
  }
  const conceptRows=Object.entries(concepts).sort(([a],[b])=>a.localeCompare(b)).map(([concept,v])=>({concept,...rate(v.s,v.n,MINIMUMS.weakness),review_failures:v.review_failures,
    weak:(v.n>=MINIMUMS.weakness && v.s/v.n<0.5) || v.review_failures>=2}));
  // Transfer: B shares a trusted concept with an independently solved A in the prior 30 days and
  // is on another platform, came from a related-practice slot, or is labeled transfer.
  const solvedHistory=attempts.filter(solved).map(a=>({key:canonicalKey(catalog,a.problem_key),date:dateOf(a.occurred_at)}));
  const transferRows=[];
  for (const b of firsts.filter(a=>inRange(a.date,windowStart,weekEnd))) {
    const bConcepts=trustedConcepts(b.problem);
    if (!bConcepts.size) continue;
    const stored=storedPrediction(plans,b.key,b.date);
    const fromRelated=stored?.slot.sub_kind==='related_channel';
    const source=solvedHistory.find(s=>s.key!==b.key && s.date<b.date && s.date>=addDays(b.date,-30) && [...trustedConcepts(catalog.problems.get(s.key))].some(c=>bConcepts.has(c)) &&
      (catalog.problems.get(s.key)?.platform!==b.problem?.platform || fromRelated || b.activity_type==='transfer'));
    if (source) transferRows.push({key:b.key,source:source.key,y:solved(b)?1:0,p:stored?.slot.prediction.p_mean ?? null,channel:fromRelated?stored.slot.components.channel:'other'});
  }
  const lift=rows=>{ const withP=rows.filter(r=>r.p!==null); return withP.length ? r3(withP.reduce((s,r)=>s+r.y-r.p,0)/withP.length) : null; };
  const channels={};
  for (const r of transferRows) (channels[r.channel] ??= []).push(r);
  const transfer={...rate(transferRows.reduce((s,r)=>s+r.y,0),transferRows.length,MINIMUMS.transfer),lift:lift(transferRows),
    by_channel:Object.fromEntries(Object.entries(channels).sort().map(([c,rows])=>[c,{...rate(rows.reduce((s,r)=>s+r.y,0),rows.length,MINIMUMS.transfer_channel),lift:lift(rows)}])),
    note:'Channel splits follow the fixed explore rotation; differences are descriptive, not causal.'};
  // Review retention per stage (rolling window).
  const reviews=reviewOutcomes(attempts,catalog).filter(r=>inRange(r.date,windowStart,weekEnd));
  const retention={by_interval:Object.fromEntries(REVIEW_DAYS.map(days=>{ const rows=reviews.filter(r=>r.interval===days && r.on_time); return [days,rate(rows.filter(r=>r.success).length,rows.length,MINIMUMS.retention_stage)]; })),
    on_time:rate(reviews.filter(r=>r.on_time).length,reviews.length,MINIMUMS.retention_stage),lapse_rate:reviews.length?r3(reviews.filter(r=>!r.success).length/reviews.length):null,n:reviews.length};
  // Contest performance (descriptive).
  const participation=(ledger.participation ?? []).filter(p=>inRange(dateOf(p.occurred_at),weekStart,weekEnd));
  const contestRows=participation.map(p=>{
    const problems=catalog.contestProblems.get(p.contest_id) ?? [];
    const keys=new Set(problems.map(x=>x.problem_key));
    const day=dateOf(p.occurred_at);
    const during=attempts.filter(a=>keys.has(canonicalKey(catalog,a.problem_key)) && dateOf(a.occurred_at)===day);
    const solvedDuring=new Set(during.filter(a=>solved(a) || a.outcome==='reported_completion').map(a=>canonicalKey(catalog,a.problem_key)));
    const after=attempts.filter(a=>keys.has(canonicalKey(catalog,a.problem_key)) && dateOf(a.occurred_at)>day && dateOf(a.occurred_at)<=addDays(day,7) && (solved(a) || a.outcome==='reported_completion'));
    const proposal=plans.find(pl=>pl.date===day && (pl.slots ?? []).some(s=>s.kind==='contest'));
    const expected=proposal?.slots.find(s=>s.kind==='contest')?.proposals?.virtual?.contest_id===p.contest_id ? proposal.slots.find(s=>s.kind==='contest').proposals.virtual.expected_solved : null;
    return {contest_id:p.contest_id,mode:p.mode,outcome:p.outcome,problems_known:problems.length,solved:problems.length?solvedDuring.size:null,expected_solved:expected,
      upsolved_within_7_days:new Set(after.map(a=>canonicalKey(catalog,a.problem_key)).filter(k=>!solvedDuring.has(k))).size,confirmed_minutes:p.confirmed_minutes};
  });
  const ratings=(ledger.rating_observations ?? []).filter(r=>inRange(dateOf(r.occurred_at),windowStart,weekEnd)).map(r=>({platform:r.payload.platform,rating:r.payload.rating,
    delta:r.payload.old_rating===null?null:r.payload.rating-r.payload.old_rating,contest_id:r.payload.contest_id,source:r.source,date:dateOf(r.occurred_at)}));
  // Ability trajectory and calibration on stored predictions.
  const abilities=fitAll(attempts,catalog,ledger.rating_observations ?? []);
  const trajectory={};
  for (const platform of ['codeforces','atcoder']) {
    const a=abilities[platform];
    const weekly={};
    for (const h of a.history) weekly[weekInfoStart(dateOf(h.occurred_at))]=h;
    trajectory[platform]={current:{mean:Math.round(a.mean),sd:Math.round(a.sd),n:a.n,status:a.status,prior:a.prior.source},weekly_end:Object.entries(weekly).sort().slice(-windowWeeks).map(([week,h])=>({week,mean:h.mean,sd:h.sd}))};
  }
  trajectory.leetcode={status:abilities.leetcode.status,n:abilities.leetcode.n,levels:Object.fromEntries(Object.entries(abilities.leetcode.levels).map(([l,v])=>[l,{mean:r3(v.mean),sd:r3(v.sd),n:v.n}]))};
  const calibration={};
  for (const platform of ['leetcode','codeforces','atcoder']) {
    const pairs=firsts.filter(a=>(a.problem?.platform ?? '')===platform).map(a=>({p:storedPrediction(plans,a.key,a.date)?.slot.prediction.p_mean,y:solved(a)?1:0})).filter(x=>Number.isFinite(x.p));
    calibration[platform]=calibrationReport(pairs);
  }
  // Process: planned vs done, time evidence kept separate (R2.4).
  const weekPlans=plans.filter(p=>inRange(p.date,weekStart,weekEnd));
  const planned=weekPlans.flatMap(p=>(p.slots ?? []).filter(s=>s.problem_key).map(s=>({...s,date:p.date})));
  const done=planned.filter(s=>attempts.some(a=>canonicalKey(catalog,a.problem_key)===s.problem_key && [s.date,addDays(s.date,1)].includes(dateOf(a.occurred_at))));
  // Replay all timers (intervals may cross week boundaries), then keep intervals that started this week.
  const intervals=timerState(ledger.events ?? [],ledger.cutoff ?? null).filter(i=>inRange(dateOf(i.started_at),weekStart,weekEnd));
  const exploreKinds={};
  for (const s of planned.filter(x=>x.kind==='explore')) {
    const v=exploreKinds[s.sub_kind] ??= {planned:0,attempted:0,independent:0};
    v.planned++;
    const a=attempts.find(x=>canonicalKey(catalog,x.problem_key)===s.problem_key && dateOf(x.occurred_at)>=s.date);
    if (a) { v.attempted++; if (solved(a)) v.independent++; }
  }
  return {version:OUTCOMES_VERSION,week_start:weekStart,week_end:weekEnd,window_start:windowStart,isr,excess,trajectory,transfer,retention,
    concepts:conceptRows,contests:{participation:contestRows,ratings},calibration,
    process:{planned_slots:planned.length,done_slots:done.length,confirmed_timer_minutes:Math.round(confirmedSeconds(intervals)/60),
      self_reported_minutes:attempts.filter(a=>inRange(dateOf(a.occurred_at),weekStart,weekEnd)).reduce((s,a)=>s+(a.self_minutes ?? 0),0),
      time_note:'Confirmed timer minutes and self-reported minutes are separate evidence and are never added together.'},
    explore_kinds:exploreKinds,
    disclaimer:'Observational single-learner metrics; they describe what happened and do not prove that any policy caused it.'};
}
function weekInfoStart(date) { const d=new Date(date+'T00:00:00Z'); return addDays(date,-((d.getUTCDay()+6)%7)); }
