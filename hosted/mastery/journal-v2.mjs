// Node journal composition for the opt-in v2 view; no durable evidence changes.
import {planProgressionV2} from './progression-v2.mjs';
import {deriveObservations} from './observations.mjs';
import {replayReviewsV2} from './reviews-v2.mjs';
import {reviewQueue} from './reviews.mjs';
import {seoulDate} from '../core/policy.mjs';
import {defaultConfig} from './progression-v2-config.mjs';
import {canonicalConfigHash} from './progression-hash.mjs';

export function v2ReviewQueue(ledger,catalog,date,config=defaultConfig){
  const states=replayReviewsV2(deriveObservations(ledger.attempts,catalog).observations,config);
  // Preserve v1 ordering contract while using v2's configured intervals for risk.
  const rows=reviewQueue(states,catalog,date);
  for(const row of rows)row.retention_risk=Math.min(1,(row.overdue_days+1)/config.review_days[row.review_stage]);
  return rows.sort((a,b)=>Number(b.due)-Number(a.due)||b.urgency-a.urgency||b.retention_risk-a.retention_risk||a.due_date.localeCompare(b.due_date)||a.problem_key.localeCompare(b.problem_key));
}
export function journalPlanV2({date,now,ledger,catalog,preferences,priorPlans=[],config=defaultConfig,implementationHash,learnerId='local-owner'}){
  const progression={catalog_version:catalog.version,...planProgressionV2({date,ledger,catalog,preferences,priorPlans,config,learnerId})};
  const budget=preferences.weekday_minutes,slots=[];let used=0;
  for(const row of v2ReviewQueue(ledger,catalog,date,config).filter(r=>r.due)){
    if(used+row.estimated_minutes>budget)continue;
    slots.push({...row,kind:'review',reason:`${row.review_priority} review due ${row.due_date}`});used+=row.estimated_minutes;
  }
  const next=progression.next;
  if(next){const p=catalog.problems.get(next.problem_key);slots.push({...next,title:p.title,url:p.url,concepts:[next.concept],estimated_minutes:30,difficulty:p.difficulty_value});}
  return {date,prepared_at:now,event_cutoff:ledger.cutoff??now,policy_bundle:'journal-v2',catalog_version:catalog.version,
    slots,progression,high_water:ledger.high_water,budget_minutes:budget,review_lane_minutes:used,notes:[...(next&&used+30>budget?['Review lane is separate from the 30-minute progression item; total suggested time exceeds the daily budget.']:[]),...(!next?['No concept-backed candidate; see coverage statuses.']:[])],
    progression_config_sha256:canonicalConfigHash(config),implementation_sha256:implementationHash};
}
const signature=state=>JSON.stringify({best_action:state.best_action??null,best_level:state.best_level??null,medium_ready:state.readiness.medium.state,prerequisites_state:state.readiness.prerequisites.map(p=>[p.concept,p.state]),follow_up_open:state.follow_up.open});
export function v2Transitions({ledger,catalog,preferences,config=defaultConfig,learnerId='local-owner'}){
  const observations=deriveObservations(ledger.attempts,catalog).observations,entries=[];
  let before=planProgressionV2({date:observations[0]?.date??'2000-01-01',ledger:{...ledger,attempts:[],cutoff:observations[0]?.occurred_at??ledger.cutoff},catalog,preferences,config,learnerId});
  const rows=new Map(ledger.attempts.map(a=>[a.id,a])),attempts=[];
  for(const o of observations){
    attempts.push(rows.get(o.id));
    const after=planProgressionV2({date:o.date,ledger:{...ledger,attempts,cutoff:o.occurred_at},catalog,preferences,config,learnerId});
    for(const [concept,state] of Object.entries(after.concepts)){
      const previous=before.concepts[concept];
      if(previous&&signature(previous)!==signature(state))entries.push({id:`mastery-${o.id}-${concept}`,type:'mastery_change',occurred_at:o.occurred_at,pattern:concept,stage:state.best_action??null,status:state.readiness.medium.state,cause_observation_id:o.id,
        before:{signature:JSON.parse(signature(previous)),cells:previous.cells},after:{signature:JSON.parse(signature(state)),cells:state.cells},reason:'Evidence-derived progression-v2 action/readiness changed.'});
    }
    before=after;
  }
  return entries;
}
