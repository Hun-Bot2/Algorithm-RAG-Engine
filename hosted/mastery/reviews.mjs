// Shared review ordering: failure urgency plus unbounded overdue aging prevents starvation.
export function reviewQueue(states,catalog,date) {
  const days=(a,b)=>Math.max(0,Math.floor((Date.parse(a)-Date.parse(b))/86400000));
  return [...states].filter(([,s])=>s.due_date).map(([key,s])=>{
    const p=catalog.problems.get(key), overdue=days(date,s.due_date);
    const priority=s.review_priority ?? 'normal';
    return {problem_key:key,title:p?.title ?? key,platform:p?.platform ?? key.split(':')[0],url:p?.url ?? null,
      difficulty:p?.difficulty_value ?? null,due_date:s.due_date,due:s.due_date<=date,review_priority:priority,
      review_stage:s.review_stage,failure_count:s.failure_count ?? 0,consecutive_failures:s.consecutive_failures ?? 0,
      overdue_days:overdue,urgency:overdue*2+(priority==='priority'?10+Math.min(s.consecutive_failures ?? 0,5)*2:0),
      retention_risk:Math.min(1,(overdue+1)/Math.max(1,[1,3,7,14,30,60][s.review_stage] ?? 1)),
      estimated_minutes:15};
  }).sort((a,b)=>Number(b.due)-Number(a.due)||b.urgency-a.urgency||b.retention_risk-a.retention_risk||a.due_date.localeCompare(b.due_date)||a.problem_key.localeCompare(b.problem_key));
}
