// planner-v1 (design.md section 5, R4). Pure and deterministic: the same ledger, catalog,
// preferences and date always yield the same plan. The explore rotation is a fixed calendar
// schedule for reproducibility; it is NOT evidence that any sub-kind or channel works better.
import {reviewQueue} from './reviews.mjs';
import {REVIEW_DAYS,replayProblemAttempts,seoulDate} from '../core/policy.mjs';
import {normalizeMastery} from '../core/preferences.mjs';
export {DEFAULT_MASTERY,normalizeMastery} from '../core/preferences.mjs';
import {MIN_EVIDENCE,fitAll,predict,CALIBRATION_VERSION} from './calibration.mjs';
import {CHANNELS,RELATED_VERSION,canonicalKey,recommendable,relatedCandidates,trustedConcepts} from './related.mjs';

export const PLANNER_VERSION = 'planner-v2';
export const POLICY_BUNDLE = 'mastery-policy-v3';
export const EXPLORE_KINDS = ['uncertain_concept','stretch_probe','related_channel','new_problem'];
const MINUTES = {review:15,practice:30};
const ZONE = {low:0.60,high:0.80,center:0.70};
const STRETCH = {low:0.35,high:0.55};
const PLACEMENT = [0.85,0.65,0.45];
const DAY = 86400000;
const addDays = (date,n) => new Date(Date.parse(date+'T00:00:00Z')+n*DAY).toISOString().slice(0,10);
const dateOf = iso => seoulDate(new Date(iso));

export function weekInfo(date) {
  const d=new Date(date+'T00:00:00Z'), weekday=(d.getUTCDay()+6)%7;
  const thursday=new Date(d.getTime()+(3-weekday)*DAY);
  const yearStart=new Date(Date.UTC(thursday.getUTCFullYear(),0,1));
  const isoWeek=1+Math.floor((thursday-yearStart)/(7*DAY));
  return {weekday,isoWeek,isoYear:thursday.getUTCFullYear(),weekStart:addDays(date,-weekday)};
}
export const exploreIndex = date => { const {isoWeek,weekday}=weekInfo(date); return (isoWeek*5+weekday)%4; };
export const relatedChannelFor = date => CHANNELS[weekInfo(date).isoWeek%CHANNELS.length];

// Per-problem review state over every attempt (aliases resolved), reusing the core state machine.
export function problemStates(attempts,catalog) {
  const byKey=new Map();
  for (const a of attempts) { const key=canonicalKey(catalog,a.problem_key); if (!byKey.has(key)) byKey.set(key,[]); byKey.get(key).push(a); }
  const states=new Map();
  for (const [key,list] of byKey) states.set(key,{...replayProblemAttempts(key,list),attempts:list.length,
    ever_solved:list.some(a=>a.outcome==='reported_completion' || (a.outcome==='independent' && a.hints===false)),last_at:list.at(-1).occurred_at});
  return states;
}
// Beta(1+s,1+f) posterior over known outcomes per trusted concept in the window.
export function conceptStats(attempts,catalog,date,windowDays=56) {
  const from=addDays(date,-windowDays), stats=new Map();
  for (const c of catalog.concepts) stats.set(c.id,{concept:c.id,s:0,f:0,n:0,review_failures:0});
  for (const a of attempts) {
    if (!['independent','assisted','failed'].includes(a.outcome) || dateOf(a.occurred_at)<from || dateOf(a.occurred_at)>date) continue;
    const p=catalog.problems.get(canonicalKey(catalog,a.problem_key));
    for (const id of trustedConcepts(p)) {
      const s=stats.get(id) ?? {concept:id,s:0,f:0,n:0,review_failures:0};
      const ok=a.outcome==='independent' && a.hints===false;
      s.n++; ok ? s.s++ : s.f++;
      if (!ok && a.activity_type==='review') s.review_failures++;
      stats.set(id,s);
    }
  }
  for (const s of stats.values()) {
    const alpha=1+s.s, beta=1+s.f;
    s.mean=alpha/(alpha+beta); s.sd=Math.sqrt(alpha*beta/((alpha+beta)**2*(alpha+beta+1)));
    s.status=s.n<4?'insufficient_evidence':'estimated';
  }
  return stats;
}
// Concepts whose latest review on some problem failed/was assisted in the last 7 days without a later independent solve.
export function remediationConcepts(attempts,catalog,date) {
  const latest=new Map();
  for (const a of attempts) latest.set(canonicalKey(catalog,a.problem_key),a);
  const out=new Set();
  for (const [key,a] of latest) if (['failed','assisted'].includes(a.outcome) && dateOf(a.occurred_at)<=date)
    for (const c of trustedConcepts(catalog.problems.get(key))) out.add(c);
  return out;
}
function exposures(priorPlans,date,cooldown) {
  const from=addDays(date,-cooldown), keys=new Set();
  for (const plan of priorPlans) if (plan.date>=from && plan.date<date) for (const slot of plan.slots ?? []) if (slot.problem_key) keys.add(slot.problem_key);
  return keys;
}
const knownPracticeCount = (attempts,catalog,platform) => attempts.filter(a=>['independent','assisted','failed'].includes(a.outcome) && a.activity_type!=='review' && catalog.problems.get(canonicalKey(catalog,a.problem_key))?.platform===platform).length;

// Picks the problem whose predicted probability is closest to a target, within an optional band.
function pick(pool,abilities,{target,low=0,high=1,field='p_conservative',prefer=new Set()}) {
  let best=null;
  for (const p of pool) {
    const prediction=predict(p,abilities), value=prediction[field];
    if (value===null) continue;
    const inBand=value>=low && value<=high;
    const distance=Math.abs(value-target);
    const candidate={problem:p,prediction,in_band:inBand,distance,preferred:prefer.has(p.key)};
    const better=!best || (candidate.in_band!==best.in_band ? candidate.in_band : candidate.preferred!==best.preferred ? candidate.preferred : candidate.distance!==best.distance ? candidate.distance<best.distance : p.key<best.problem.key);
    if (better) best=candidate;
  }
  return best;
}
const slotFrom=(kind,sub,choice,components,reason,extra={})=>({kind,sub_kind:sub,problem_key:choice.problem.key,platform:choice.problem.platform,title:choice.problem.title,url:choice.problem.url,
  concepts:[...trustedConcepts(choice.problem)].sort(),prediction:choice.prediction,outside_target_zone:!choice.in_band,components,reason,estimated_minutes:MINUTES.practice,...extra});

export function planDay({date,now,ledger,catalog,preferences,priorPlans=[],curriculum=null,contestProposals=null}) {
  const mastery=normalizeMastery(preferences.mastery ?? {});
  const {weekday,weekStart}=weekInfo(date);
  const dayType=preferences.weekly_rhythm[weekday];
  const attempts=ledger.attempts;
  const abilities=fitAll(attempts,catalog,ledger.rating_observations ?? []);
  const states=problemStates(attempts,catalog);
  const exposed=exposures(priorPlans,date,mastery.cooldown_days);
  const cooldownFrom=addDays(date,-mastery.cooldown_days);
  const seen=new Set([...exposed]);
  for (const [key,s] of states) if (s.ever_solved || dateOf(s.last_at)>=cooldownFrom) seen.add(key);
  const curriculumKeys=new Set((curriculum?.problems ?? []).map(p=>p.problem_key));
  const reviewedEdges=new Set((curriculum?.relationships ?? []).filter(r=>r.review?.status==='approved' && ['similarity','transfer'].includes(r.type)).map(r=>r.source+'\0'+r.target));
  const stats=conceptStats(attempts,catalog,date);
  const remediation=remediationConcepts(attempts,catalog,date);
  const budget=['virtual','upsolve'].includes(dayType) ? 0 : preferences.weekday_minutes;
  const slots=[], notes=[];
  let used=0;
  const used_keys=new Set();
  const take=slot=>{ slots.push(slot); used+=slot.estimated_minutes; used_keys.add(slot.problem_key); seen.add(slot.problem_key); };

  // 1. Reviews (existing intervals). Practice days cap reviews at half the budget.
  const due=reviewQueue(states,catalog,date).filter(r=>r.due).map(r=>[r.problem_key,{...states.get(r.problem_key),...r}]);
  const reviewCap=dayType==='review' ? budget : Math.floor(budget/2);
  for (const [key,s] of due) {
    if (used+MINUTES.review>(s.review_priority==='priority'?budget:reviewCap)) { notes.push(`review deferred: ${key}`); continue; }
    const p=catalog.problems.get(key);
    slots.push({kind:'review',sub_kind:null,problem_key:key,platform:p?.platform ?? key.split(':')[0],title:p?.title ?? key,url:p?.url ?? null,concepts:[...trustedConcepts(p)].sort(),
      review_priority:s.review_priority,prediction:predict(p,abilities),components:{failure_count:s.failure_count,urgency:s.urgency,review_stage:s.review_stage,overdue_days:Math.round((Date.parse(date)-Date.parse(s.due_date))/DAY)},
      reason:`${s.review_priority==='priority'?'Priority review after an unsolved attempt. ':''}Review due ${s.due_date} (stage ${s.review_stage}, interval ${REVIEW_DAYS[s.review_stage]} days).`,estimated_minutes:MINUTES.review});
    used+=MINUTES.review; used_keys.add(key);
  }
  if (['virtual','upsolve'].includes(dayType)) {
    return finalize({date,now,dayType,slots:[{kind:dayType==='virtual'?'contest':'upsolve',sub_kind:null,problem_key:null,proposals:contestProposals,
      reason:dayType==='virtual'?'Weekend contest day: accept a live contest or run the proposed virtual contest.':'Upsolve the weekend contest, then read the weekly report.',estimated_minutes:null}],
      abilities,stats,mastery,catalog,ledger,notes,rotation:null});
  }
  const poolFor=(platform,filter=()=>true)=>[...catalog.problems.values()].filter(p=>p.platform===platform && recommendable(p) && !seen.has(p.key) && filter(p));
  const targetConcepts=new Set(mastery.target_concepts);
  const priority=c=>(1-c.mean)+(targetConcepts.has(c.concept)?0.5:0);
  const conceptOrder=[...stats.values()].filter(c=>!targetConcepts.size || targetConcepts.has(c.concept)).sort((a,b)=>priority(b)-priority(a) || a.concept.localeCompare(b.concept));

  // Targeted repair precedes ordinary practice and is not constrained by exploration quotas.
  let earlyRemediation=false;
  if(remediation.size && used+MINUTES.practice<=budget){
    for(const concept of [...remediation].sort()){
      const choice=pick(poolFor(mastery.practice_platform,p=>trustedConcepts(p).has(concept)),abilities,{target:0.80,low:0.70,high:0.95,prefer:curriculumKeys});
      if(choice){take(slotFrom('remediation',null,choice,{concept},`Targeted remediation in ${concept} after an unsolved attempt.`));earlyRemediation=true;break;}
    }
  }

  // 2. Exploit (cold start: placement pattern) on the practice platform, then contest platforms.
  const exploitPlatforms=[mastery.practice_platform,mastery.contest_platforms[weekday%mastery.contest_platforms.length]];
  const exploreAllowed=dayType==='practice' && (()=>{
    const weekPractice=preferences.weekly_rhythm.filter(d=>['practice','review'].includes(d)).length*preferences.weekday_minutes;
    const spent=priorPlans.filter(p=>p.date>=weekStart && p.date<date).flatMap(p=>p.slots ?? []).filter(s=>s.kind==='explore').reduce((s,x)=>s+(x.estimated_minutes ?? 0),0);
    return spent+MINUTES.practice<=mastery.explore_cap*weekPractice;
  })();
  // Exploit always gets the first practice slot; explore only uses minutes left after it.
  const exploitBudget=Math.max(used+MINUTES.practice,budget-(exploreAllowed?MINUTES.practice:0));
  for (const platform of exploitPlatforms) {
    if (used+MINUTES.practice>exploitBudget) break;
    const placementDone=knownPracticeCount(attempts,catalog,platform);
    if (placementDone<MIN_EVIDENCE) {
      const target=PLACEMENT[placementDone%PLACEMENT.length];
      const choice=pick(poolFor(platform,p=>trustedConcepts(p).size>0),abilities,{target,low:target-0.1,high:target+0.1,prefer:curriculumKeys});
      if (choice) take(slotFrom('exploit','placement',choice,{placement_target:target,placement_attempts:placementDone},
        `Placement ${placementDone+1}/${MIN_EVIDENCE} on ${platform}: sparse history, so difficulty is chosen from a conservative estimate (target P≈${target}${choice.in_band?'':`; closest available P ${choice.prediction.p_conservative}`}).`));
      continue;
    }
    let chosen=null;
    for (const c of conceptOrder) {
      const choice=pick(poolFor(platform,p=>trustedConcepts(p).has(c.concept)),abilities,{target:ZONE.center,low:ZONE.low,high:ZONE.high,prefer:curriculumKeys});
      if (choice) { chosen={choice,concept:c}; break; }
    }
    if (chosen) take(slotFrom('exploit',null,chosen.choice,{concept:chosen.concept.concept,need:Math.round((1-chosen.concept.mean)*1000)/1000,target_gap:targetConcepts.has(chosen.concept.concept)?0.5:0,concept_n:chosen.concept.n},
      `Highest learning need: ${chosen.concept.concept} (independent rate posterior ${chosen.concept.mean.toFixed(2)}, n=${chosen.concept.n}); target predicted success 0.60-0.80.`));
  }

  // 3. Explore (practice days only, capped). Rotation is a reproducible schedule.
  let rotation=null;
  if (dayType==='practice') {
    const index=exploreIndex(date);
    rotation={index,scheduled:EXPLORE_KINDS[index],used:null,fallbacks:[],channel:null,allowed:exploreAllowed};
    if (earlyRemediation) rotation.used='remediation';
    else if (!exploreAllowed) notes.push('explore slot skipped: weekly explore cap reached');
    else if (used+MINUTES.practice>budget) { rotation.allowed=false; notes.push('explore slot skipped: no budget left after reviews and exploit'); }
    else if (remediation.size) {
      const concept=[...remediation].sort()[0];
      const choice=pick(poolFor(mastery.practice_platform,p=>trustedConcepts(p).has(concept)),abilities,{target:0.80,low:0.70,high:0.95,prefer:curriculumKeys});
      if (choice) { take(slotFrom('remediation',null,choice,{concept},`Remediation replaces exploration: recent review in ${concept} failed or needed help.`)); rotation.used='remediation'; }
    } else {
      for (let step=0; step<EXPLORE_KINDS.length && !rotation.used; step++) {
        const kind=EXPLORE_KINDS[(index+step)%EXPLORE_KINDS.length];
        const slot=exploreSlot(kind,{date,catalog,abilities,stats,poolFor,mastery,attempts,seen,reviewedEdges,rotation,curriculumKeys});
        if (slot) { take(slot); rotation.used=kind; } else rotation.fallbacks.push(kind);
      }
    }
  }
  return finalize({date,now,dayType,slots,abilities,stats,mastery,catalog,ledger,notes,rotation});
}

function exploreSlot(kind,{date,catalog,abilities,stats,poolFor,mastery,attempts,seen,reviewedEdges,rotation,curriculumKeys}) {
  const platform=mastery.practice_platform;
  if (kind==='uncertain_concept') {
    const uncertain=[...stats.values()].filter(c=>c.n<4).sort((a,b)=>b.sd-a.sd || a.n-b.n || a.concept.localeCompare(b.concept));
    for (const c of uncertain) {
      const choice=pick(poolFor(platform,p=>trustedConcepts(p).has(c.concept)),abilities,{target:ZONE.center,low:ZONE.low,high:ZONE.high,prefer:curriculumKeys});
      if (choice) return slotFrom('explore',kind,choice,{concept:c.concept,posterior_sd:Math.round(c.sd*1000)/1000,concept_n:c.n},`Explore an uncertain concept (${c.concept}, n=${c.n}).`);
    }
    return null;
  }
  if (kind==='stretch_probe') {
    const demonstrated=[...stats.values()].filter(c=>c.s>0).sort((a,b)=>b.mean-a.mean || a.concept.localeCompare(b.concept));
    for (const c of demonstrated) {
      const choice=pick(poolFor(platform,p=>trustedConcepts(p).has(c.concept)),abilities,{target:0.45,low:STRETCH.low,high:STRETCH.high,field:'p_mean'});
      if (choice?.in_band) return slotFrom('explore',kind,choice,{concept:c.concept,concept_mean:Math.round(c.mean*1000)/1000},`Stretch probe in a demonstrated concept (${c.concept}).`);
    }
    return null;
  }
  if (kind==='related_channel') {
    const channel=relatedChannelFor(date); rotation.channel=channel;
    const source=[...attempts].reverse().find(a=>catalog.problems.get(canonicalKey(catalog,a.problem_key))?.platform===platform);
    if (!source) return null;
    const result=relatedCandidates({sourceKey:source.problem_key,catalog,abilities,seen,channel,platforms:mastery.contest_platforms,reviewedEdges,limit:1});
    const c=result.candidates[0];
    if (!c) return null;
    return {kind:'explore',sub_kind:kind,problem_key:c.problem_key,platform:c.platform,title:c.title,url:c.url,concepts:c.shared_concepts,prediction:c.prediction,
      outside_target_zone:c.prediction.p_conservative<ZONE.low || c.prediction.p_conservative>ZONE.high,components:{...c.components,source_key:result.source_key,channel},semantic_only:c.semantic_only,
      evidence_label:c.evidence_label,reason:`Related practice from ${result.source_key} via ${channel} (${c.evidence_label}).`,estimated_minutes:MINUTES.practice,related_policy:RELATED_VERSION};
  }
  if (kind==='new_problem') {
    const base=catalog.base_version ?? catalog.version;
    const choice=pick(poolFor(platform,p=>p.first_seen_version!==base && trustedConcepts(p).size>0).concat(mastery.contest_platforms.flatMap(pl=>poolFor(pl,p=>p.first_seen_version!==base && trustedConcepts(p).size>0))),abilities,{target:ZONE.center,low:ZONE.low,high:ZONE.high});
    return choice ? slotFrom('explore',kind,choice,{first_seen_version:choice.problem.first_seen_version},'Newly added catalog problem.') : null;
  }
  return null;
}

function finalize({date,now,dayType,slots,abilities,stats,mastery,catalog,ledger,notes,rotation}) {
  const summary=platform=>{ const a=abilities[platform]; return platform==='leetcode' ? {status:a.status,n:a.n,levels:Object.fromEntries(Object.entries(a.levels).map(([l,v])=>[l,{mean:Math.round(v.mean*1000)/1000,sd:Math.round(v.sd*1000)/1000,n:v.n}]))}
    : {status:a.status,n:a.n,mean:Math.round(a.mean),sd:Math.round(a.sd),prior:a.prior.source}; };
  return {date,policy_bundle:POLICY_BUNDLE,components:{planner:PLANNER_VERSION,calibration:CALIBRATION_VERSION,related:RELATED_VERSION},
    catalog_version:catalog.version,event_cutoff:ledger.cutoff,high_water:ledger.high_water,prepared_at:now ?? null,day_type:dayType,
    slots,rotation,abilities:{leetcode:summary('leetcode'),codeforces:summary('codeforces'),atcoder:summary('atcoder')},
    rotation_note:'Explore sub-kinds follow a fixed calendar rotation for reproducibility; comparisons between them are descriptive, not causal.',
    mastery,notes};
}
