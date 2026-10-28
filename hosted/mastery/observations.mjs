// progression-v2 evidence projection. Input is the existing policy-v3 attempt stream.
import {seoulDate} from '../core/policy.mjs';
import {canonicalKey,trustedConcepts} from './related.mjs';

const compare=(a,b)=>a.occurred_at.localeCompare(b.occurred_at)||a.id.localeCompare(b.id);
export function deriveObservations(attempts,catalog) {
  const dated=new Map(),unknown=[],invalid=[];
  for(const attempt of attempts){
    const key=canonicalKey(catalog,attempt.problem_key),problem=catalog.problems.get(key);
    if(!problem){unknown.push({id:attempt.id,problem_key:attempt.problem_key,reason:'unknown_problem'});continue;}
    const time=Date.parse(attempt.occurred_at);
    if(!Number.isFinite(time)||typeof attempt.id!=='string'||!['independent','assisted','reported_completion','failed'].includes(attempt.outcome)){
      invalid.push({id:attempt.id,problem_key:key,reason:'invalid_attempt'});continue;
    }
    const date=seoulDate(new Date(time)),token=`${key}\0${date}`;
    const row={...attempt,problem_key:key,occurred_at:new Date(time).toISOString(),date};
    const prior=dated.get(token);
    if(!prior||compare(prior,row)<0)dated.set(token,row);
  }
  const previous=new Map(),observations=[];
  for(const row of [...dated.values()].sort(compare)){
    const problem=catalog.problems.get(row.problem_key),concepts=[...trustedConcepts(problem)].sort();
    const prior=previous.get(row.problem_key),kind=prior?'retention':'ability';
    const level=problem.difficulty_kind==='lc_level'?Number(problem.difficulty_value):null;
    observations.push({id:row.id,problem_key:row.problem_key,platform:problem.platform,kind,
      date:row.date,occurred_at:row.occurred_at,level,rating:level===null?problem.difficulty_value:null,
      concepts,weight:concepts.length?1/concepts.length:0,y:row.outcome==='failed'?0:1,
      gap_days:prior?Math.round((Date.parse(row.date)-Date.parse(prior.date))/86400000):null,
      self_minutes:row.self_minutes??null,...(row.self_seconds!=null?{self_seconds:row.self_seconds}:{})});
    previous.set(row.problem_key,row);
  }
  const diagnosticOrder=(a,b)=>String(a.id).localeCompare(String(b.id))||String(a.problem_key).localeCompare(String(b.problem_key));
  return {observations,ability:observations.filter(o=>o.kind==='ability'),retention:observations.filter(o=>o.kind==='retention'),
    unknown_problems:unknown.sort(diagnosticOrder),invalid_attempts:invalid.sort(diagnosticOrder)};
}
