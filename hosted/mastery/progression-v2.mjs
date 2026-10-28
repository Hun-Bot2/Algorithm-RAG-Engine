// progression-v2: pure derived estimates. v1 modules remain unchanged.
import {fitAbility,priorFor,predict} from './calibration.mjs';
import {seoulDate} from '../core/policy.mjs';
import {deriveObservations} from './observations.mjs';
import {canonicalConfigHash} from './progression-hash.mjs';
import {defaultConfig} from './progression-v2-config.mjs';
import {canonicalKey,recommendable,trustedConcepts} from './related.mjs';
import curriculum from './interview-curriculum.v1.json' with {type:'json'};
import {betaInformation,betaSurvival,betaQuantile} from './beta-math.mjs';
const logit=p=>Math.log(p/(1-p)),sigmoid=x=>1/(1+Math.exp(-x));
export function leetCodeCells(observations,concept,config=defaultConfig){
  const ability=observations.filter(o=>o.kind==='ability'&&(!o.platform||o.platform==='leetcode')&&[1,2,3].includes(o.level));
  const fresh=!ability.some(o=>o.concepts.includes(concept));
  const cells=[];let delta=0;
  for(const level of [1,2,3]){
    const outside=ability.filter(o=>o.level===level&&!o.concepts.includes(concept));
    const own=ability.filter(o=>o.level===level&&o.concepts.includes(concept));
    const [a,b]=config.level_priors[level];
    const mean=fresh?a/(a+b):(a+outside.reduce((s,o)=>s+o.y,0))/(a+b+outside.length);
    const centre=sigmoid(logit(mean)+config.lambda*delta);
    const alpha=config.kappa*centre+own.reduce((s,o)=>s+o.y/o.concepts.length,0);
    const beta=config.kappa*(1-centre)+own.reduce((s,o)=>s+(1-o.y)/o.concepts.length,0);
    delta=own.length?logit(alpha/(alpha+beta))-logit(centre):0;
    cells.push({level,alpha,beta,n_own:own.length,n_level_outside:outside.length,prior:!fresh&&outside.length?'learner':'authored'});
  }
  const blocks=cells.map(c=>({alpha:c.alpha,beta:c.beta,members:[c.level]}));
  for(let i=0;i<blocks.length-1;){
    const a=blocks[i],b=blocks[i+1];
    if(a.alpha/(a.alpha+a.beta)<b.alpha/(b.alpha+b.beta)){
      blocks.splice(i,2,{alpha:a.alpha+b.alpha,beta:a.beta+b.beta,members:[...a.members,...b.members]});i=Math.max(0,i-1);
    }else i++;
  }
  const tail=(1-config.credible_mass)/2;
  return cells.map(cell=>{
    const block=blocks.find(b=>b.members.includes(cell.level));
    return {...cell,alpha:block.alpha,beta:block.beta,p:block.alpha/(block.alpha+block.beta),
      interval:[betaQuantile(tail,block.alpha,block.beta),betaQuantile(1-tail,block.alpha,block.beta)]};
  });
}


export function platformAbilities(observations,catalog,ratings=[]){
  return Object.fromEntries(['codeforces','atcoder'].map(platform=>{
    const rows=observations.filter(o=>o.kind==='ability'&&o.platform===platform).map(o=>({
      occurred_at:o.occurred_at,y:o.y,problem:catalog.problems.get(o.problem_key),
      weight:platform==='atcoder'&&catalog.problems.get(o.problem_key)?.atc_experimental?0.5:1,
    })).filter(o=>Number.isFinite(o.problem?.difficulty_value)&&o.problem.difficulty_kind===(platform==='atcoder'?'atc_irt':'cf_rating'));
    return [platform,fitAbility(platform,rows,priorFor(platform,ratings))];
  }));
}
export function platformPrediction(problem,abilities,config=defaultConfig){
  const p=predict(problem,abilities).p_mean,ability=abilities[problem.platform];
  if(p===null||!ability)return null;
  const a=problem.platform==='codeforces'?Math.LN10/400:(problem.atc_discrimination??0.004479398673070138);
  const z=1.2816,d=problem.difficulty_value;
  const lo=sigmoid(a*(ability.mean-z*ability.sd-d)),hi=sigmoid(a*(ability.mean+z*ability.sd-d));
  const fisher=ability.sd**2*a*a*p*(1-p);
  return {p,interval:[Math.min(lo,p),Math.max(hi,p)],n_own:ability.n,n_level_outside:0,
    prior:ability.n?'learner':'authored',information:fisher/(1+fisher)};
}


export function validatePrerequisites(graph){
  const active=new Set(),done=new Set();
  function visit(c){if(active.has(c))throw new Error('cyclic concept prerequisites');if(done.has(c))return;
    active.add(c);for(const p of graph[c]??[])visit(p);active.delete(c);done.add(c);}
  for(const c of Object.keys(graph))visit(c);
}
export function cellReadiness({concept,level,observations,catalog,config=defaultConfig,excludeConcept=null}){
  const eligible=[...catalog.problems.values()].filter(p=>recommendable(p)&&p.platform==='leetcode'&&Number(p.difficulty_value)===level&&trustedConcepts(p).has(concept)&&(!excludeConcept||!trustedConcepts(p).has(excludeConcept)));
  const attempted=new Set(observations.map(o=>o.problem_key));
  const cell=leetCodeCells(observations.filter(o=>!excludeConcept||!o.concepts.includes(excludeConcept)),concept,config)[level-1];
  const probability=betaSurvival(config.tau,cell.alpha,cell.beta);
  if(cell.n_own>0&&probability>=config.readiness_probability)return {state:'ready',probability,cell};
  if(!eligible.length||eligible.every(p=>attempted.has(p.key)))return {state:'waived',probability,waiver:eligible.length?'exhausted':'no_catalog_coverage',cell};
  return {state:'not_ready',probability,cell};
}
export function conceptReadiness({concept,observations,catalog,config=defaultConfig,targets=[],prerequisites=curriculum.prerequisites}){
  validatePrerequisites(prerequisites);
  const medium=cellReadiness({concept,level:2,observations,catalog,config});
  const dependencies=(prerequisites[concept]??[]).map(c=>({concept:c,...cellReadiness({concept:c,level:1,observations,catalog,config,excludeConcept:concept})}));
  if(targets.includes(concept))return {medium,prerequisites:dependencies.map(p=>({...p,state:'bypassed',note:'user target concept'})),eligible:true};
  return {medium,prerequisites:dependencies,eligible:dependencies.every(p=>p.state!=='not_ready')};
}
export function conceptFollowUp(observations,concept,date,config=defaultConfig){
  let failure=null;
  for(const o of observations.filter(o=>o.concepts.includes(concept)&&o.date<=date).toSorted((a,b)=>a.occurred_at.localeCompare(b.occurred_at)||a.id.localeCompare(b.id))){
    if(o.y===0)failure=o;
    else if(failure&&((o.level!==null&&failure.level!==null&&o.level<=failure.level)||
      (o.platform===failure.platform&&o.rating!==null&&failure.rating!==null&&o.rating<=failure.rating)))failure=null;
  }
  const expires=failure?new Date(Date.parse(failure.date)+config.follow_up_days*86400000).toISOString().slice(0,10):null;
  return {open:Boolean(failure&&date<expires),failed_observation_id:failure?.id??null,expires};
}

const variation=p=>JSON.stringify([...(p.tags??[])].map(String).sort());
const fnv=text=>{let h=2166136261;for(let i=0;i<text.length;i++){h^=text.charCodeAt(i);h=Math.imul(h,16777619);}return h>>>0;};
export function planProgressionV2({date,ledger,catalog,preferences={},priorPlans=[],config=defaultConfig,prerequisites=curriculum.prerequisites,learnerId='local-owner',configHash=null}){
  configHash??=canonicalConfigHash(config);
  const attempts=(ledger.attempts??[]).filter(a=>a.occurred_at&&seoulDate(new Date(a.occurred_at))<=date&&(!ledger.cutoff||a.occurred_at<=ledger.cutoff));
  const derived=deriveObservations(attempts,catalog),obs=derived.observations;
  const abilities=platformAbilities(obs,catalog,ledger.rating_observations??[]);
  const attempted=new Set(obs.map(o=>o.problem_key)),exposed=new Set();
  const cooldown=preferences.mastery?.cooldown_days??30;
  for(const plan of priorPlans){
    const age=(Date.parse(date)-Date.parse(plan.date??plan.local_date))/86400000;
    if(age>=0&&age<cooldown)for(const slot of plan.slots??[])if(!attempted.has(slot.problem_key))exposed.add(canonicalKey(catalog,slot.problem_key));
  }
  const pool=[...catalog.problems.values()].filter(p=>recommendable(p)&&!attempted.has(p.key)&&!exposed.has(p.key)&&trustedConcepts(p).size);
  const concepts=[...new Set([...catalog.problems.values()].flatMap(p=>[...trustedConcepts(p)]))].sort();
  const latest=derived.ability.at(-1),states={},all=[];
  validatePrerequisites(prerequisites);
  for(const concept of concepts){
    const own=derived.ability.filter(o=>o.concepts.includes(concept));
    const highest=Math.max(1,...own.filter(o=>o.platform==='leetcode').map(o=>o.level));
    const cells=leetCodeCells(obs,concept,config),readiness=conceptReadiness({concept,observations:obs,catalog,config,targets:preferences.mastery?.target_concepts??[],prerequisites});
    const follow_up=conceptFollowUp(obs,concept,date,config);
    const transferDone=own.some(o=>config.transfer_platforms.includes(o.platform)&&o.y===1);
    const seenVariations=new Map([1,2,3].map(l=>[l,new Set(obs.filter(o=>o.concepts.includes(concept)&&o.level===l).map(o=>variation(catalog.problems.get(o.problem_key))))]));
    const coverage={},actions=[];
    for(const level of [1,2,3]){
      const action=level<highest?'step_down':level===highest?'consolidate':level===highest+1?'step_up':'stretch';
      if(action==='stretch'&&!own.length)continue;
      const candidates=pool.filter(p=>p.platform==='leetcode'&&Number(p.difficulty_value)===level&&trustedConcepts(p).has(concept));
      const ties=p=>({coverage_gain:seenVariations.get(level).has(variation(p))?0:1,purity:trustedConcepts(p).size,hash:fnv(`${learnerId}|${date}|${concept}|${p.key}`)});
      candidates.sort((a,b)=>{const x=ties(a),y=ties(b);return y.coverage_gain-x.coverage_gain||x.purity-y.purity||x.hash-y.hash;});
      coverage[action]=candidates.length?'ok':'insufficient_catalog_coverage';
      if(candidates.length){const problem=candidates[0],cell=cells[level-1];actions.push({action,problem,estimate:cell,information:betaInformation(cell.alpha,cell.beta),tie_break:ties(problem),level});}
    }
    if(readiness.medium.state!=='not_ready'&&!transferDone){
      const candidates=pool.filter(p=>config.transfer_platforms.includes(p.platform)&&trustedConcepts(p).has(concept)&&!(config.exclude_experimental_atcoder&&p.platform==='atcoder'&&p.atc_experimental)&&
        (config.transfer_native_difficulty_floor===null||p.difficulty_value>=config.transfer_native_difficulty_floor)).map(problem=>({problem,estimate:platformPrediction(problem,abilities,config)})).filter(p=>p.estimate);
      const tie=p=>({coverage_gain:obs.some(o=>o.concepts.includes(concept)&&o.platform===p.platform&&variation(catalog.problems.get(o.problem_key))===variation(p))?0:1,purity:trustedConcepts(p).size,hash:fnv(`${learnerId}|${date}|${concept}|${p.key}`)});
      candidates.sort((a,b)=>{const d=Math.abs(a.estimate.p-config.tau)-Math.abs(b.estimate.p-config.tau);const x=tie(a.problem),y=tie(b.problem);return d||y.coverage_gain-x.coverage_gain||x.purity-y.purity||x.hash-y.hash;});
      coverage.transfer=candidates.length?'ok':'insufficient_catalog_coverage';
      if(candidates.length){const {problem,estimate}=candidates[0];actions.push({action:'transfer',problem,estimate,information:estimate.information,tie_break:tie(problem),rating:problem.difficulty_value});}
    }
    for(const a of actions){
      const components={D:1-Math.abs(a.estimate.p-config.tau)/Math.max(config.tau,1-config.tau),I:a.information,C:a.tie_break.coverage_gain,F:follow_up.open?1:0,K:latest?.concepts.includes(concept)&&readiness.medium.state==='not_ready'?1:0};
      const w=config.weights,B=w.challenge_fit*components.D+w.information*components.I+w.coverage*components.C+w.follow_up*components.F+w.continuity*components.K;
      const p=a.problem;
      all.push({policy:'progression-v2',config_version:config.version,config_sha256:configHash??canonicalConfigHash(config),catalog_version:catalog.version,concept,action:a.action,kind:a.action==='transfer'?'transfer':'practice',
        problem_key:p.key,platform:p.platform,...(a.level?{level:a.level}:{rating:a.rating}),estimate:{p:a.estimate.p,interval:a.estimate.interval,n_own:a.estimate.n_own,n_level_outside:a.estimate.n_level_outside,prior:a.estimate.prior},components,weights:w,benefit_proxy:B,
        readiness,follow_up,coverage,tie_break:a.tie_break,eligible:readiness.eligible,reason:`${concept}: ${a.action}; estimated solve probability ${a.estimate.p.toFixed(3)}, heuristic benefit ${B.toFixed(3)}.`,limitations:['estimates are not calibrated','benefit is a heuristic']});
    }
    states[concept]={concept,cells,readiness,follow_up,coverage};
  }
  const order=(a,b)=>b.benefit_proxy-a.benefit_proxy||b.tie_break.coverage_gain-a.tie_break.coverage_gain||a.tie_break.purity-b.tie_break.purity||a.tie_break.hash-b.tie_break.hash;
  let available=all.filter(a=>a.eligible).sort(order),deadlockWaiver=false;
  if(!available.length&&all.length){available=all.sort(order);deadlockWaiver=true;}
  const winner=available[0]??null;
  if(winner){
    winner.alternatives=available.slice(1,4).map(a=>({concept:a.concept,action:a.action,benefit_proxy:a.benefit_proxy}));
    if(deadlockWaiver){winner.readiness={...winner.readiness,waiver:'candidate_coverage_deadlock'};winner.reason+=' Prerequisite coverage waiver prevents a deadlock.';}
  }
  return {policy:'progression-v2',config_version:config.version,config_sha256:configHash??canonicalConfigHash(config),status:winner?'available':'insufficient_catalog_coverage',next:winner,concepts:states,diagnostics:{unknown_problems:derived.unknown_problems,invalid_attempts:derived.invalid_attempts},alternatives:winner?.alternatives??[]};
}
