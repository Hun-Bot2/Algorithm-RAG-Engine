// calibration-v1 (design.md section 4). Pure, deterministic, Worker-safe.
// Per-platform practice ability with explicit uncertainty. No cross-platform conversion.
export const CALIBRATION_VERSION = 'calibration-v1';
export const MIN_EVIDENCE = 6;
const CF_SLOPE = Math.LN10/400;          // Codeforces rating scale (assumed Elo-like logistic, see design.md)
const DRIFT_PER_WEEK = 25;               // rating points SD added per week of elapsed time
const LC_PRIORS = {1:[3.2,0.8],2:[2,2],3:[0.8,3.2]};
const sigmoid = x => 1/(1+Math.exp(-x));

// Inverse of AtCoder's displayed-rating corrections, per AtCoder Problems RatingInfo.ts
// (research/mastery_catalog/atcoder-model-provenance.v1.json). Null without rated participations.
export function atcoderInternalRating(rating,participations) {
  if (!Number.isFinite(rating) || !Number.isInteger(participations) || participations<1 || rating<=0) return null;
  const lower = rating<=400 ? 400*(1-Math.log(400/rating)) : rating;
  const n=participations;
  return lower + ((Math.sqrt(1-0.9**(2*n))/(1-0.9**n)-1)/(Math.sqrt(19)-1))*1200;
}
// AtCoder Problems predictSolveProbability: contest-condition model on the internal scale.
export const atcoderSolveProbability = (internal,difficulty,discrimination) => sigmoid(discrimination*(internal-difficulty));
export const codeforcesSolveProbability = (rating,difficulty) => sigmoid(CF_SLOPE*(rating-difficulty));

// Priors: contest ratings seed practice ability (R3.3) but are never overwritten by it.
export function priorFor(platform,ratings=[]) {
  const latest=ratings.filter(r=>r.payload.platform===platform).sort((a,b)=>a.occurred_at.localeCompare(b.occurred_at) || a.entry_id.localeCompare(b.entry_id)).at(-1);
  if (platform==='codeforces') return latest ? {mean:latest.payload.rating,sd:250,source:'codeforces_rating',at:latest.occurred_at} : {mean:1000,sd:350,source:'default',at:null};
  if (platform==='atcoder') {
    const internal=latest ? atcoderInternalRating(latest.payload.rating,latest.payload.rated_participations) : null;
    return internal!==null ? {mean:internal,sd:300,source:'atcoder_rating',at:latest.occurred_at} : {mean:0,sd:400,source:'default',at:null};
  }
  return null;
}
const slopeFor = (platform,problem) => platform==='codeforces' ? CF_SLOPE : (problem.atc_discrimination ?? 0.004479398673070138);

// Gaussian (Laplace) sequential update of a logistic ability: one Newton step per observation,
// with drift between observations. Deterministic; weight halves experimental AtCoder models.
export function fitAbility(platform,observations,prior) {
  let mean=prior.mean, variance=prior.sd**2, last=prior.at, n=0, weightSum=0;
  const history=[];
  for (const o of observations) {
    if (last) variance += DRIFT_PER_WEEK**2 * Math.max(0,(Date.parse(o.occurred_at)-Date.parse(last))/(7*86400000));
    last=o.occurred_at;
    const a=slopeFor(platform,o.problem), p=sigmoid(a*(mean-o.problem.difficulty_value)), w=o.weight ?? 1;
    const precision=1/variance + w*a*a*p*(1-p);
    variance=1/precision;
    mean += variance*w*a*(o.y-p);
    n++; weightSum+=w;
    history.push({occurred_at:o.occurred_at,mean:Math.round(mean*10)/10,sd:Math.round(Math.sqrt(variance)*10)/10});
  }
  return {platform,mean,sd:Math.sqrt(variance),n,effective_n:weightSum,prior,history,
    status:n<MIN_EVIDENCE?'insufficient_evidence':'estimated'};
}
// LeetCode: Beta-binomial per native level, ordered Easy >= Medium >= Hard by pooling.
export function fitLeetCode(observations) {
  const cells={1:{s:0,n:0},2:{s:0,n:0},3:{s:0,n:0}};
  for (const o of observations) { const c=cells[o.problem.difficulty_value]; if (c) { c.n++; c.s+=o.y; } }
  let levels=[1,2,3].map(l=>{ const [a,b]=LC_PRIORS[l]; const alpha=a+cells[l].s, beta=b+cells[l].n-cells[l].s;
    return {level:l,alpha,beta,mean:alpha/(alpha+beta),weight:alpha+beta,n:cells[l].n}; });
  // Pool-adjacent-violators on posterior means (weights = posterior pseudo-counts).
  const blocks=levels.map(l=>({mean:l.mean,weight:l.weight,members:[l.level]}));
  for (let i=0;i<blocks.length-1;) {
    if (blocks[i].mean < blocks[i+1].mean) {
      const w=blocks[i].weight+blocks[i+1].weight;
      blocks.splice(i,2,{mean:(blocks[i].mean*blocks[i].weight+blocks[i+1].mean*blocks[i+1].weight)/w,weight:w,members:[...blocks[i].members,...blocks[i+1].members]});
      i=Math.max(0,i-1);
    } else i++;
  }
  const pooled=Object.fromEntries(blocks.flatMap(b=>b.members.map(m=>[m,b.mean])));
  levels=levels.map(l=>({...l,mean:pooled[l.level],sd:Math.sqrt(l.alpha*l.beta/((l.alpha+l.beta)**2*(l.alpha+l.beta+1)))}));
  const n=observations.length;
  return {platform:'leetcode',levels:Object.fromEntries(levels.map(l=>[l.level,l])),n,status:n<MIN_EVIDENCE?'insufficient_evidence':'estimated'};
}

// Practice observations: known-outcome attempts excluding reviews and reported completions
// (R2.3). Repeated attempts on one problem all count. y=1 only for independent, no hints.
export function practiceObservations(attempts,catalog,platform) {
  const out=[];
  for (const a of attempts) {
    if (!['independent','assisted','failed'].includes(a.outcome) || a.activity_type==='review') continue;
    const key=catalog.aliases.get(a.problem_key) ?? a.problem_key;
    const problem=catalog.problems.get(key);
    if (!problem || problem.platform!==platform || problem.difficulty_value===null || problem.difficulty_value===undefined) continue;
    if (platform==='atcoder' && problem.difficulty_kind!=='atc_irt') continue;
    out.push({occurred_at:a.occurred_at,problem,y:a.outcome==='independent' && a.hints===false ? 1 : 0,
      weight:platform==='atcoder' && problem.atc_experimental ? 0.5 : 1,attempt_id:a.id});
  }
  return out;
}
export function fitAll(attempts,catalog,ratings=[]) {
  return {
    version:CALIBRATION_VERSION,
    leetcode:fitLeetCode(practiceObservations(attempts,catalog,'leetcode')),
    codeforces:fitAbility('codeforces',practiceObservations(attempts,catalog,'codeforces'),priorFor('codeforces',ratings)),
    atcoder:fitAbility('atcoder',practiceObservations(attempts,catalog,'atcoder'),priorFor('atcoder',ratings)),
  };
}
const round = x => Math.round(x*1000)/1000;
// Prediction with explicit uncertainty. p_conservative assumes ability one SD lower
// (sparse history => easier selections). Unknown difficulty => p_mean null.
export function predict(problem,abilities) {
  if (!problem || problem.difficulty_value===null || problem.difficulty_value===undefined || problem.difficulty_kind==='none') return {p_mean:null,p_conservative:null,p_optimistic:null,basis:'unknown_difficulty',status:'unknown'};
  if (problem.platform==='leetcode') {
    const l=abilities.leetcode.levels[problem.difficulty_value];
    return {p_mean:round(l.mean),p_conservative:round(Math.max(0,l.mean-l.sd)),p_optimistic:round(Math.min(1,l.mean+l.sd)),basis:'lc_level_beta',status:abilities.leetcode.status};
  }
  const ability=abilities[problem.platform];
  if (!ability) return {p_mean:null,p_conservative:null,p_optimistic:null,basis:'unsupported_platform',status:'unknown'};
  const a=slopeFor(problem.platform,problem), d=problem.difficulty_value;
  const marginal=sigmoid(a*(ability.mean-d)/Math.sqrt(1+Math.PI*a*a*ability.sd*ability.sd/8));
  return {p_mean:round(marginal),p_conservative:round(sigmoid(a*(ability.mean-ability.sd-d))),p_optimistic:round(sigmoid(a*(ability.mean+ability.sd-d))),
    basis:problem.platform==='codeforces'?'cf_logistic':'atcoder_problems_irt',status:ability.status};
}
// Calibration quality on stored predictions (R3.5): Brier vs constant-rate baseline, 3 bins.
export function calibrationReport(pairs,minimum=30) {
  const valid=pairs.filter(p=>Number.isFinite(p.p) && (p.y===0 || p.y===1));
  if (valid.length<minimum) return {n:valid.length,status:'insufficient_evidence',minimum};
  const rate=valid.reduce((s,p)=>s+p.y,0)/valid.length;
  const brier=valid.reduce((s,p)=>s+(p.p-p.y)**2,0)/valid.length;
  const baseline=valid.reduce((s,p)=>s+(rate-p.y)**2,0)/valid.length;
  const bins=[[0,1/3],[1/3,2/3],[2/3,1.0001]].map(([lo,hi])=>{ const b=valid.filter(p=>p.p>=lo && p.p<hi);
    return {range:[round(lo),round(Math.min(hi,1))],n:b.length,mean_predicted:b.length?round(b.reduce((s,p)=>s+p.p,0)/b.length):null,observed:b.length?round(b.reduce((s,p)=>s+p.y,0)/b.length):null}; });
  return {n:valid.length,status:'estimated',brier:round(brier),baseline_brier:round(baseline),beats_baseline:brier<baseline,bins};
}
