// T8: synthetic learners against the immutable, receipt-verified catalog. No learner DB.
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {DatabaseSync} from 'node:sqlite';
import {attachCatalog,loadCatalog} from '../../hosted/mastery/catalog.mjs';
import {planProgressionV2} from '../../hosted/mastery/progression-v2.mjs';
import {defaultConfig,loadProgressionConfig} from '../../hosted/mastery/progression-v2-config.mjs';

const db=new DatabaseSync(':memory:');
const info=attachCatalog(db,'private-learning/mastery/catalog-v4/catalog-v4.sqlite3',{verify:true});
if(info.version!=='catalog-v4')throw new Error('Expected catalog-v4');
const catalog=loadCatalog(db);
const day=i=>new Date(Date.UTC(2026,0,1+i)).toISOString().slice(0,10);
const cfCatalog={...catalog,problems:new Map([...catalog.problems].filter(([,p])=>p.platform==='codeforces'))};
const reversed=c=>({...c,problems:new Map([...c.problems].reverse())});
const fmt=n=>Number.isFinite(n)?n.toFixed(4):'-';

function run(name,c,config,behavior,verbose,planner=planProgressionV2){
  const attempts=[],visited=new Set(),path=[];
  let last=null,orderFailures=0;
  for(let i=0;i<60;i++){
    const date=day(i),args={date,ledger:{attempts},catalog:c,config,learnerId:'synthetic-replay'};
    const plan=planner(args);
    const shuffled=planner({...args,catalog:reversed(c)});
    if(plan.next?.problem_key!==shuffled.next?.problem_key)orderFailures++;
    last=plan;
    if(!plan.next){if(verbose)console.log(`step=${i+1} status=${plan.status}`);break;}
    const slot=plan.next;
    visited.add(slot.concept);
    path.push(`${slot.concept}/${slot.action}/${slot.level??slot.rating}`);
    if(verbose)console.log(`step=${i+1} concept=${slot.concept} action=${slot.action} level=${slot.level??slot.rating} P=${fmt(slot.estimate.p)} B=${fmt(slot.benefit_proxy)}`);
    const y=behavior(i);
    attempts.push({id:`synthetic-${i}`,problem_key:slot.problem_key,occurred_at:`${date}T03:00:00Z`,outcome:y?'independent':'failed',hints:y?false:null,activity_type:'practice'});
  }
  console.log(`run=${name} steps=${path.length} catalog_order_mismatches=${orderFailures}`);
  if(verbose&&last)for(const [concept,state] of Object.entries(last.concepts)){
    const coverage=Object.entries(state.coverage).map(([action,status])=>`${action}:${status}`).join(',')||'no_actions';
    const waivers=[state.readiness.medium,...state.readiness.prerequisites].filter(x=>x.state==='waived').map(x=>x.waiver).join(',')||'-';
    console.log(`concept=${concept} ${visited.has(concept)?'recommended':'not_recommended'} coverage=${coverage} waivers=${waivers}`);
  }
  return path;
}

console.log('Fixed learner seed synthetic-replay is shared across sensitivity runs.');
console.log(`catalog=${info.version} sha256=${info.sha256}`);
console.log('Synthetic behavior only; this does not measure recommendation relevance or learning effects.');
for(const [name,c,behavior] of [
  ['all-solve',catalog,()=>1],['alternating',catalog,i=>i%2===0?1:0],
  ['struggling',catalog,i=>i%4===3?1:0],['cf-only',cfCatalog,()=>1],
]) run(name,c,defaultConfig,behavior,true);

console.log('sensitivity: 60 all-solve steps; counts and action paths are behavior observations only');
for(const kappa of [2,4,8])for(const lambda of [0.25,0.5,1])for(const weightMode of ['default','D-only'])for(const priorMode of ['authored','flat']){
  const config=loadProgressionConfig({...defaultConfig,kappa,lambda,
    weights:weightMode==='default'?defaultConfig.weights:{challenge_fit:1,information:0,coverage:0,follow_up:0,continuity:0},
    level_priors:priorMode==='authored'?defaultConfig.level_priors:{1:[2,2],2:[2,2],3:[2,2]}});
  const name=`k=${kappa},lambda=${lambda},weights=${weightMode},priors=${priorMode}`;
  const path=run(name,catalog,config,()=>1,false);
  console.log(`sensitivity ${name} path=${path.join(' ')}`);
}
// Research-only counterfactual: exact source transformation, never shipped or written back.
const sourceUrl=new URL('../../hosted/mastery/progression-v2.mjs',import.meta.url);
const original=readFileSync(sourceUrl,'utf8');
const occurrences=original.split('/o.concepts.length').length-1;
if(occurrences!==2)throw new Error('Unit-attribution counterfactual requires review after source changes');
const counterfactual=original.replaceAll('/o.concepts.length','/1').replace(/from '([^']+)'/g,(_,path)=>`from '${new URL(path,sourceUrl).href}'`);
const unit=await import('data:text/javascript;charset=utf-8,'+encodeURIComponent(counterfactual));
console.log(`attribution counterfactual source_sha256=${createHash('sha256').update(original).digest('hex')} replacements=${occurrences}; research-only source transform 1/k -> 1`);
const unitPath=run('unit-attribution',catalog,defaultConfig,()=>1,true,unit.planProgressionV2);
console.log(`attribution unit path=${unitPath.join(' ')}`);
db.close();
