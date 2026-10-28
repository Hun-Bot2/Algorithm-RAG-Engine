// progression-v2: pure derived estimates. v1 modules remain unchanged.
import {fitAbility,priorFor,predict} from './calibration.mjs';
import {defaultConfig} from './progression-v2-config.mjs';
import {recommendable,trustedConcepts} from './related.mjs';
import curriculum from './interview-curriculum.v1.json' with {type:'json'};
import {betaSurvival,betaQuantile} from './beta-math.mjs';
const logit=p=>Math.log(p/(1-p)),sigmoid=x=>1/(1+Math.exp(-x));
export function leetCodeCells(observations,concept,config=defaultConfig){
  const ability=observations.filter(o=>o.kind==='ability'&&(!o.platform||o.platform==='leetcode')&&[1,2,3].includes(o.level));
  const cells=[];let delta=0;
  for(const level of [1,2,3]){
    const outside=ability.filter(o=>o.level===level&&!o.concepts.includes(concept));
    const own=ability.filter(o=>o.level===level&&o.concepts.includes(concept));
    const [a,b]=config.level_priors[level];
    const mean=(a+outside.reduce((s,o)=>s+o.y,0))/(a+b+outside.length);
    const centre=sigmoid(logit(mean)+config.lambda*delta);
    const alpha=config.kappa*centre+own.reduce((s,o)=>s+o.y/o.concepts.length,0);
    const beta=config.kappa*(1-centre)+own.reduce((s,o)=>s+(1-o.y)/o.concepts.length,0);
    delta=own.length?logit(alpha/(alpha+beta))-logit(centre):0;
    cells.push({level,alpha,beta,n_own:own.length,n_level_outside:outside.length,prior:outside.length?'learner':'authored'});
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

