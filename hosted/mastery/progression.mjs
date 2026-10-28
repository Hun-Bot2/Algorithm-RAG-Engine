// progression-v1 is a deterministic view over existing ledger and catalog data.
// It derives no durable mastery state and performs no learner writes.
import pattern from './progression-pattern.v1.json' with {type:'json'};
import {canonicalKey,trustedConcepts,recommendable} from './related.mjs';
import {seoulDate} from '../core/policy.mjs';
import {fitAll,predict} from './calibration.mjs';

export const PROGRESSION_VERSION='progression-v1';
export const PROGRESSION_PATTERN_VERSION=pattern.version;
const REQUIRED_STAGES=['leetcode_easy','leetcode_medium','cross_platform_transfer','retention_review'];
const success=(a,config)=>a.outcome===config.evidence.success_outcome && a.hints===config.evidence.required_hints;
const failure=(a,config)=>config.evidence.failure_outcomes.includes(a.outcome);
const eligible=p=>recommendable(p) && ['free','free_at_collection'].includes(p.access);
const level=p=>p?.difficulty_kind==='lc_level' ? ({1:'easy',2:'medium',3:'hard',easy:'easy',medium:'medium',hard:'hard'}[p.difficulty_value] ?? null) : null;
const signatures=(p,concept)=>{
  if (p?.variation_group) return [`group:${p.variation_group}`];
  const concepts=[...trustedConcepts(p)].sort();
  const tags=[...(p?.tags ?? [])].map(x=>String(x).toLowerCase()).sort();
  if (!concepts.includes(concept) || (!tags.length && concepts.length<2)) return [];
  return [`metadata:${concepts.join(',')}|${tags.join(',')}`];
};
function inScope(a,date,cutoff) {
  const time=Date.parse(a.occurred_at);
  return Number.isFinite(time) && seoulDate(new Date(time))<=date && (!cutoff || time<=Date.parse(cutoff));
}
function prerequisites(concept,curriculum,catalog) {
  const needed=new Set();
  for (const edge of curriculum?.relationships ?? []) if (edge.type==='prerequisite' && edge.review?.status==='approved') {
    const target=catalog.problems.get(canonicalKey(catalog,edge.target));
    if (trustedConcepts(target).has(concept)) for (const id of trustedConcepts(catalog.problems.get(canonicalKey(catalog,edge.source)))) needed.add(id);
  }
  return [...needed].sort();
}
function collectEvidence(concept,attempts,catalog,config) {
  const easyStage=config.pattern.find(s=>s.id==='leetcode_easy'), mediumStage=config.pattern.find(s=>s.id==='leetcode_medium'), transferStage=config.pattern.find(s=>s.id==='cross_platform_transfer'), retentionStage=config.pattern.find(s=>s.id==='retention_review');
  const easy=new Map(), medium=new Map(), mediumSources=new Set(); let transfer=null, retained=null, regression=null, regressionTarget=null;
  const earlierKeys=new Map(); const exposed=new Set();
  for (const a of attempts) { earlierKeys.set(a.id,new Set(exposed)); exposed.add(canonicalKey(catalog,a.problem_key)); }
  const evidence=attempts.flatMap(a=>{
    const key=canonicalKey(catalog,a.problem_key), problem=catalog.problems.get(key);
    return trustedConcepts(problem).has(concept)?[{...a,key,problem,earlierAttempted:earlierKeys.get(a.id) ?? new Set()}]:[];
  });
  for (const a of evidence) {
    if (failure(a,config)) {
      // Failure reopens the most recently completed stage and clears evidence
      // that depended on it. A later success must earn the stage again.
      const easyVariationsBefore=new Set([...easy.values()].flat()).size, mediumVariationsBefore=new Set([...medium.values()].flat()).size;
      const stage=easy.size<easyStage.distinct_required || easyVariationsBefore<easyStage.variation_signatures_required?0:medium.size<mediumStage.distinct_required || mediumVariationsBefore<mediumStage.variation_signatures_required?1:!transfer?2:!retained?3:4;
      if (a.problem.platform==='leetcode' && level(a.problem)==='hard') continue;
      if (a.problem.platform==='leetcode' && level(a.problem)==='easy') {
        easy.clear(); medium.clear(); mediumSources.clear(); transfer=null; retained=null; regressionTarget=0;
      } else if (a.problem.platform==='leetcode' && level(a.problem)==='medium') {
        medium.clear(); mediumSources.clear(); transfer=null; retained=null; regressionTarget=easy.size>=easyStage.distinct_required && easyVariationsBefore>=easyStage.variation_signatures_required?1:0;
      } else if (stage>=4) { retained=null; regressionTarget=4; }
      else if (stage>=3) { transfer=null; retained=null; regressionTarget=3; }
      else if (stage>=1) { medium.clear(); mediumSources.clear(); transfer=null; retained=null; regressionTarget=1; }
      else { easy.clear(); medium.clear(); mediumSources.clear(); transfer=null; retained=null; regressionTarget=0; }
      regression={problem_key:a.key,occurred_at:a.occurred_at};
      continue;
    }
    if (!success(a,config)) continue;
    const p=a.problem, key=a.key;
    if (p.platform===easyStage.platform && level(p)===easyStage.difficulty) easy.set(key,signatures(p,concept));
    const easyVariations=new Set([...easy.values()].flat()).size;
    if (p.platform===mediumStage.platform && level(p)===mediumStage.difficulty && easy.size>=easyStage.distinct_required && easyVariations>=easyStage.variation_signatures_required) { medium.set(key,signatures(p,concept)); mediumSources.add(key); }
    const mediumVariations=new Set([...medium.values()].flat()).size;
    if (transferStage.platforms.includes(p.platform) && easy.size>=easyStage.distinct_required && easyVariations>=easyStage.variation_signatures_required && medium.size>=mediumStage.distinct_required && mediumVariations>=mediumStage.variation_signatures_required && a.activity_type==='transfer' && mediumSources.has(canonicalKey(catalog,a.transfer_from)) && !a.earlierAttempted.has(key)) transfer=a;
    if (a.activity_type===retentionStage.activity_type && transfer && Date.parse(a.occurred_at)-Date.parse(transfer.occurred_at)>=retentionStage.minimum_days_after_transfer*86400000) retained=a;
    const afterStage=easy.size<easyStage.distinct_required || easyVariations<easyStage.variation_signatures_required?0:medium.size<mediumStage.distinct_required || mediumVariations<mediumStage.variation_signatures_required?1:!transfer?2:!retained?3:4;
    if (regression && (afterStage>regressionTarget || (regressionTarget===4 && afterStage===4 && a.activity_type==='review'))) regression=null;
  }
  const uniqueSignatures=map=>new Set([...map.values()].flat()).size;
  const easyReady=easy.size>=easyStage.distinct_required && uniqueSignatures(easy)>=easyStage.variation_signatures_required;
  const mediumReady=medium.size>=mediumStage.distinct_required && uniqueSignatures(medium)>=mediumStage.variation_signatures_required;
  const stageIndex=!easyReady?0:!mediumReady?1:!transfer?2:!retained?3:4;
  const advanced=evidence.filter(a=>a.problem.platform==='leetcode' && level(a.problem)==='hard');
  const advancedEvidence={attempts:advanced.length,independent:advanced.filter(a=>success(a,config)).length,failed:advanced.filter(a=>failure(a,config)).length,assisted:advanced.filter(a=>a.outcome==='assisted').length,status:'observed_not_mastered'};
  return {stageIndex,easy,medium,mediumSources,transfer,retained,regression,regressionTarget,easyReady,mediumReady,advancedEvidence};
}
function priorExposure(priorPlans,date,catalog) {
  const seen=new Set();
  for (const plan of priorPlans ?? []) if (plan.date<=date) for (const slot of plan.slots ?? []) if (slot.problem_key) seen.add(canonicalKey(catalog,slot.problem_key));
  return seen;
}
function candidateFor(concept,index,{attempts,catalog,curriculum,seen,transferSource,config,abilities}) {
  const stage=config.pattern[index];
  if (stage.id==='retention_review') return null;
  const attempted=new Set(attempts.map(a=>canonicalKey(catalog,a.problem_key)));
  const candidates=[...catalog.problems.values()].filter(p=>eligible(p) && trustedConcepts(p).has(concept) && !attempted.has(p.key) && !seen.has(p.key)).sort((a,b)=>a.key.localeCompare(b.key));
  let pool=candidates;
  if (stage.id==='leetcode_easy') pool=pool.filter(p=>p.platform===stage.platform && level(p)===stage.difficulty);
  if (stage.id==='leetcode_medium') pool=pool.filter(p=>p.platform===stage.platform && level(p)===stage.difficulty);
  if (stage.id==='cross_platform_transfer') pool=pool.filter(p=>stage.platforms.includes(p.platform) && (trustedConcepts(p).has(concept) || (curriculum?.relationships ?? []).some(r=>r.type==='transfer' && r.review?.status==='approved' && canonicalKey(catalog,r.source)===transferSource && canonicalKey(catalog,r.target)===p.key)) && (!stage.exclude_experimental_atcoder || p.platform!=='atcoder' || !p.atc_experimental) && Number.isFinite(p.difficulty_value) && p.difficulty_value<=(stage.native_difficulty_max?.[p.platform] ?? Infinity));
  const required=stage.variation_signatures_required ?? 0;
  if (required) {
    const used=new Set([...attempts].flatMap(a=>{
      const p=catalog.problems.get(canonicalKey(catalog,a.problem_key));
      return p?.platform==='leetcode' && level(p)===stage.difficulty ? signatures(p,concept) : [];
    }));
    pool=pool.filter(p=>signatures(p,concept).some(sig=>!used.has(sig)));
  }
  if (stage.id==='cross_platform_transfer') return pool.map(problem=>({problem,prediction:predict(problem,abilities)})).filter(x=>x.prediction.p_conservative!==null).sort((a,b)=>b.prediction.p_conservative-a.prediction.p_conservative || a.problem.platform.localeCompare(b.problem.platform) || a.problem.key.localeCompare(b.problem.key))[0] ?? null;
  return pool[0] ? {problem:pool[0],prediction:predict(pool[0],abilities)} : null;
}

function validateConfig(config) {
  if (config.version!==PROGRESSION_PATTERN_VERSION || !Array.isArray(config.pattern) || config.pattern.map(s=>s.id).join('\0')!==REQUIRED_STAGES.join('\0')) throw new Error('unsupported progression pattern version or stage order');
  const [easy,medium,transfer,retention]=config.pattern;
  const validCount=stage=>Number.isInteger(stage.distinct_required) && stage.distinct_required>=1 && stage.distinct_required<=20 && Number.isInteger(stage.variation_signatures_required) && stage.variation_signatures_required>=1 && stage.variation_signatures_required<=stage.distinct_required;
  if (easy.platform!=='leetcode' || easy.difficulty!=='easy' || !validCount(easy)) throw new Error('Easy stage counts must be positive integers up to 20, with variation count no greater than problem count');
  if (medium.platform!=='leetcode' || medium.difficulty!=='medium' || !validCount(medium)) throw new Error('Medium stage counts must be positive integers up to 20, with variation count no greater than problem count');
  if (!Array.isArray(transfer.platforms) || !transfer.platforms.includes('codeforces') || !transfer.platforms.includes('atcoder') || !Number.isFinite(retention.minimum_days_after_transfer) || retention.minimum_days_after_transfer<7) throw new Error('progression transfer or retention constraints are invalid');
  return config;
}

export function planProgression({date,ledger,catalog,curriculum=null,priorPlans=[],prerequisitesByConcept={},config=pattern,evidenceOnly=false}={}) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date ?? '')) throw new Error('progression date must be YYYY-MM-DD');
  if (!Array.isArray(ledger?.attempts) || !catalog?.problems) throw new Error('progression requires ledger attempts and catalog problems');
  validateConfig(config);
  const cutoff=ledger.cutoff ?? null;
  const attempts=ledger.attempts.filter(a=>inScope(a,date,cutoff)).slice().sort((a,b)=>a.occurred_at.localeCompare(b.occurred_at)||String(a.id).localeCompare(String(b.id)));
  const abilities=evidenceOnly?null:fitAll(attempts,catalog,ledger.rating_observations ?? []);
  const concepts=[...new Set([...(catalog.concepts ?? []).map(c=>c.id),...attempts.flatMap(a=>[...trustedConcepts(catalog.problems.get(canonicalKey(catalog,a.problem_key)))])])].sort();
  const raw={};
  for (const concept of concepts) raw[concept]=collectEvidence(concept,attempts,catalog,config);
  const seen=priorExposure(priorPlans,date,catalog), states={};
  for (const concept of concepts) {
    const evidence=raw[concept], index=evidence.stageIndex, stage=config.pattern[index] ?? null;
    const prereq=[...new Set([...(config.prerequisites?.[concept] ?? []),...(prerequisitesByConcept[concept] ?? []),...prerequisites(concept,curriculum,catalog)])].sort(), prerequisitesMet=prereq.every(id=>raw[id]?.stageIndex>=3);
    const transferSource=[...attempts].reverse().find(a=>raw[concept].mediumSources.has(canonicalKey(catalog,a.problem_key)));
    const candidateChoice=!evidenceOnly && stage && prerequisitesMet ? candidateFor(concept,index,{attempts,catalog,curriculum,seen,transferSource:transferSource?canonicalKey(catalog,transferSource.problem_key):null,config,abilities}) : null;
    const candidate=candidateChoice?.problem ?? null;
    const stageStates=config.pattern.map((item,i)=>({stage:item.id,status:i<index?'complete':i===index?(prerequisitesMet?'current':'blocked_prerequisite'):'locked'}));
    const state={concept,stage_index:index,status:evidence.regression?'remediation':stage?(prerequisitesMet?'in_progress':'blocked_prerequisite'):'retention_evidenced',stages:stageStates,
      next_stage:stage?.id ?? null,next_candidate:candidate?.key ?? null,next_candidate_reason:candidate?`Concept ${concept}: ${evidence.easy.size}/${config.pattern[0].distinct_required} Easy and ${evidence.medium.size}/${config.pattern[1].distinct_required} Medium solves; next ${stage.id} candidate has not been attempted or recently recommended. ${evidence.regression?'Repair after '+evidence.regression.problem_key+'; earlier failure remains recorded. ':''}Native ${candidate.platform} difficulty ${candidate.difficulty_value ?? 'unknown'}; ${candidateChoice?.prediction?.status ?? 'uncertain'} calibration. Catalog variation metadata is a coverage proxy, not verified algorithm equivalence.`:stage?`No eligible unseen candidate with sufficient variation evidence for ${stage.id}.`:null,
      next_candidate_prediction:candidateChoice?.prediction ?? null,
      completed_counts:{leetcode_easy:evidence.easy.size,leetcode_easy_variations:new Set([...evidence.easy.values()].flat()).size,leetcode_medium:evidence.medium.size,leetcode_medium_variations:new Set([...evidence.medium.values()].flat()).size,transfer:Boolean(evidence.transfer),retention_review:Boolean(evidence.retained)},
      advanced_evidence:evidence.advancedEvidence,
      prerequisites:prereq,prerequisites_met:prerequisitesMet,regression:evidence.regression};
    if (!evidenceOnly && config.atcoder_fallback==='semantic_unverified_only' && stage?.id==='cross_platform_transfer' && !candidate) {
      const neighbors=transferSource ? catalog.neighbors(canonicalKey(catalog,transferSource.problem_key),'title_v2').map(n=>n.dst) : [];
      const fallback=neighbors.map(key=>catalog.problems.get(key)).filter(p=>p?.platform==='atcoder' && eligible(p) && !trustedConcepts(p).size && p.reliability==='reliable' && !seen.has(p.key) && !attempts.some(a=>canonicalKey(catalog,a.problem_key)===p.key)).sort((a,b)=>a.key.localeCompare(b.key))[0];
      if (fallback) state.unverified_semantic_fallback={problem_key:fallback.key,evidence:'semantic title similarity only; does not satisfy concept or transfer mastery'};
    }
    states[concept]=state;
  }
  const conceptOrder=[...new Set([...(config.concept_order ?? []).filter(id=>concepts.includes(id)),...concepts])];
  const targetConcepts=new Set(config.target_concepts ?? []), rank=new Map(conceptOrder.map((id,index)=>[id,index]));
  const selectionOrder=concepts.slice().sort((a,b)=>{
    const remediation=Number(states[b].status==='remediation')-Number(states[a].status==='remediation');
    const target=Number(targetConcepts.has(b))-Number(targetConcepts.has(a));
    return remediation || target || rank.get(a)-rank.get(b) || a.localeCompare(b);
  });
  const stages=config.pattern.map(stage=>({id:stage.id,concepts:conceptOrder.map(concept=>{
    const state=states[concept], row=state.stages.find(s=>s.stage===stage.id);
    const fallback=state.unverified_semantic_fallback;
    return {concept,status:row.status,candidate:row.status==='current'?(state.next_candidate ?? fallback?.problem_key ?? null):null,reason:row.status==='current'?(state.next_candidate_reason ?? (fallback?'Semantic title neighbor only; it does not establish concept mastery.':null)):null};
  })}));
  const next=selectionOrder.map(concept=>({concept,state:states[concept]})).find(x=>x.state.next_candidate || x.state.unverified_semantic_fallback);
  const semanticOnly=Boolean(next && !next.state.next_candidate && next.state.unverified_semantic_fallback);
  const nextKey=semanticOnly?next.state.unverified_semantic_fallback.problem_key:next?.state.next_candidate;
  const transferFrom=next?.state.next_stage==='cross_platform_transfer' && !semanticOnly ? [...attempts].reverse().find(a=>raw[next.concept].mediumSources.has(canonicalKey(catalog,a.problem_key))) : null;
  const nextAction=next?{concept:next.concept,stage:next.state.next_stage,status:semanticOnly?'unverified':next.state.status,kind:semanticOnly?'transfer':next.state.next_stage==='cross_platform_transfer'?'transfer':next.state.next_stage==='retention_review'?'review':'practice',problem_key:nextKey,reason:semanticOnly?'Semantic title neighbor only; this candidate cannot satisfy concept or transfer mastery.':next.state.next_candidate_reason,semantic_only:semanticOnly,evidence_label:semanticOnly?'Semantic title neighbor: concept not classified':undefined,transfer_from:transferFrom?canonicalKey(catalog,transferFrom.problem_key):undefined,prediction:semanticOnly?null:next.state.next_candidate_prediction}:null;
  return {version:PROGRESSION_VERSION,pattern_version:config.version,date,event_cutoff:cutoff,catalog_version:catalog.version,stages,next:nextAction,focus:nextAction?{pattern:nextAction.concept,stage:nextAction.stage,status:nextAction.status,reason:nextAction.reason}:null,concepts:states};
}
