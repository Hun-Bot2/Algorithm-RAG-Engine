// weekly-v1 (design.md section 11, R10). Pure: report object + Markdown text. Writing files and
// committing are the CLI's and the learner's business; this module never touches Git.
import {seoulDate} from '../core/policy.mjs';
import {computeOutcomes} from './metrics.mjs';
import {canonicalKey,trustedConcepts} from './related.mjs';
import {POLICY_BUNDLE} from './planner.mjs';

export const WEEKLY_VERSION = 'weekly-v1';
const dateOf=iso=>seoulDate(new Date(iso));
const addDays=(date,n)=>new Date(Date.parse(date+'T00:00:00Z')+n*86400000).toISOString().slice(0,10);

export function buildWeeklyReport({ledger,catalog,plans=[],weekStart,preferences={}}) {
  const weekEnd=addDays(weekStart,6);
  const outcomes=computeOutcomes({ledger,catalog,plans,weekStart});
  const inWeek=a=>dateOf(a.occurred_at)>=weekStart && dateOf(a.occurred_at)<=weekEnd;
  const weekAttempts=ledger.attempts.filter(inWeek);
  const activity={total:weekAttempts.length,by_source:{},by_outcome:{},by_platform:{},by_difficulty:{}};
  for (const a of weekAttempts) {
    const p=catalog.problems.get(canonicalKey(catalog,a.problem_key));
    const bump=(group,key)=>{ activity[group][key]=(activity[group][key] ?? 0)+1; };
    bump('by_source',a.evidence_source); bump('by_outcome',a.outcome); bump('by_platform',p?.platform ?? a.problem_key.split(':')[0]);
    bump('by_difficulty',p ? `${p.platform}:${p.difficulty_kind==='lc_level'?['','Easy','Medium','Hard'][p.difficulty_value]:p.difficulty_value===null?'unknown':Math.round(p.difficulty_value/100)*100}` : 'not_in_catalog');
  }
  const weak=outcomes.concepts.filter(c=>c.weak).sort((a,b)=>(a.rate ?? 1)-(b.rate ?? 1) || b.review_failures-a.review_failures || a.concept.localeCompare(b.concept));
  const targets=(preferences.mastery?.target_concepts ?? []).filter(c=>!weak.some(w=>w.concept===c));
  const evidence=Object.fromEntries(outcomes.concepts.map(c=>[c.concept,c.n]));
  const focus=weak.slice(0,2).map(c=>({concept:c.concept,reason:`weak: solve rate ${c.rate} over n=${c.n}, review failures ${c.review_failures}`}));
  for (const c of targets.sort((a,b)=>(evidence[a] ?? 0)-(evidence[b] ?? 0) || a.localeCompare(b))) if (focus.length<2) focus.push({concept:c,reason:`target concept with least evidence (n=${evidence[c] ?? 0})`});
  const feedback={total:0,by_verdict:{},by_channel:{}};
  for (const f of (ledger.practice_feedback ?? []).filter(inWeek)) {
    feedback.total++; feedback.by_verdict[f.payload.verdict]=(feedback.by_verdict[f.payload.verdict] ?? 0)+1;
    const ch=f.payload.channel ?? 'unknown'; feedback.by_channel[ch]=(feedback.by_channel[ch] ?? 0)+1;
  }
  const repos={};
  for (const w of ledger.git_watermarks ?? []) repos[w.payload.repo_id]=w;
  const unclassifiedCommits=(ledger.git_watermarks ?? []).filter(inWeek).reduce((s,w)=>s+Number(/unclassified commits: (\d+)/.exec(w.payload.detail ?? '')?.[1] ?? 0),0);
  const notInCatalog=new Set(weekAttempts.map(a=>canonicalKey(catalog,a.problem_key)).filter(k=>!catalog.problems.has(k)));
  const atcoderNoConcepts=new Set(weekAttempts.map(a=>canonicalKey(catalog,a.problem_key)).filter(k=>catalog.problems.get(k)?.platform==='atcoder' && !trustedConcepts(catalog.problems.get(k)).size));
  const gitObservations=weekAttempts.filter(a=>a.evidence_source==='git');
  return {
    version:WEEKLY_VERSION,policy_bundle:POLICY_BUNDLE,week_start:weekStart,week_end:weekEnd,catalog_version:catalog.version,
    event_cutoff:ledger.cutoff,high_water:ledger.high_water,activity,review:outcomes.retention,outcomes:{isr:outcomes.isr,excess:outcomes.excess,trajectory:outcomes.trajectory,transfer:outcomes.transfer},
    calibration:outcomes.calibration,weaknesses:weak,next_week_focus:focus,contests:outcomes.contests,process:outcomes.process,
    evaluation:{feedback,explore_kinds:outcomes.explore_kinds,ai_relatedness:'No AI relatedness judgments are compiled into this catalog version; AI-learner agreement is unavailable.'},
    data_quality:{
      git_untrailered_share:gitObservations.length?Math.round(gitObservations.filter(a=>a.outcome==='reported_completion').length/gitObservations.length*1000)/1000:null,
      unclassified_commits:unclassifiedCommits,
      repositories:Object.fromEntries(Object.entries(repos).sort().map(([id,w])=>[id,{status:w.payload.status,last_processed_sha:w.payload.last_processed_sha,at:w.recorded_at}])),
      attempts_not_in_catalog:[...notInCatalog].sort(),
      atcoder_attempts_without_concepts:atcoderNoConcepts.size,
      catalog_built_at:catalog.meta?.built_at ?? null,
    },
    disclaimer:outcomes.disclaimer,
  };
}

const fmtRate=r=>r.n===0?'no data':r.status==='insufficient_evidence'?`${r.successes}/${r.n} (insufficient evidence; need ${r.minimum})`:`${(r.rate*100).toFixed(0)}% (${r.successes}/${r.n}, 95% CI ${r.interval.map(x=>(x*100).toFixed(0)).join('-')}%)`;
export function renderMarkdown(report) {
  const L=[];
  L.push(`# Weekly report ${report.week_start} to ${report.week_end}`,'');
  L.push(`Policy ${report.policy_bundle}, catalog ${report.catalog_version}, event cutoff ${report.event_cutoff}.`,'');
  L.push(`> ${report.disclaimer}`,'');
  L.push('## Activity','',`Attempts: ${report.activity.total}.`);
  for (const group of ['by_platform','by_outcome','by_source','by_difficulty']) {
    const entries=Object.entries(report.activity[group]).sort();
    if (entries.length) L.push(`- ${group.replace('by_','By ')}: ${entries.map(([k,v])=>`${k} ${v}`).join(', ')}`);
  }
  L.push('','## Review','',`- On-time reviews: ${fmtRate(report.review.on_time)}`,`- Lapse rate: ${report.review.lapse_rate ?? 'no data'}`);
  for (const [days,r] of Object.entries(report.review.by_interval)) if (r.n) L.push(`- ${days}-day retention: ${fmtRate(r)}`);
  L.push('','## Outcomes','');
  for (const scope of ['week','rolling']) {
    L.push(`- Completion rate (policy v3) (${scope==='week'?'this week':'4 weeks'}): ${fmtRate(report.outcomes.isr[scope].overall)}`);
    for (const [p,r] of Object.entries(report.outcomes.isr[scope].by_platform)) L.push(`  - ${p}: ${fmtRate(r)}`);
    const e=report.outcomes.excess[scope];
    L.push(`- Excess solve rate vs stored predictions (${scope}): ${e.excess_solve_rate===null?'no data':e.excess_solve_rate} (n=${e.n}, ${e.status}; ${e.without_stored_prediction} attempts had no stored prediction)`);
  }
  for (const p of ['codeforces','atcoder']) { const t=report.outcomes.trajectory[p].current; L.push(`- ${p} practice ability: ${t.mean} ± ${t.sd} (n=${t.n}, ${t.status}, prior ${t.prior})`); }
  const lcLevels=report.outcomes.trajectory.leetcode.levels;
  L.push(`- LeetCode success by level: ${['1','2','3'].map(l=>`${['','Easy','Medium','Hard'][l]} ${lcLevels[l].mean}±${lcLevels[l].sd} (n=${lcLevels[l].n})`).join(', ')} (${report.outcomes.trajectory.leetcode.status})`);
  L.push(`- Transfer success: ${fmtRate(report.outcomes.transfer)}; lift vs prediction ${report.outcomes.transfer.lift ?? 'no data'}`);
  for (const [ch,r] of Object.entries(report.outcomes.transfer.by_channel)) L.push(`  - ${ch}: ${fmtRate(r)}, lift ${r.lift ?? 'no data'}`);
  L.push('','## Calibration','');
  for (const [p,c] of Object.entries(report.calibration)) L.push(`- ${p}: ${c.status==='estimated'?`Brier ${c.brier} vs baseline ${c.baseline_brier} (${c.beats_baseline?'beats':'does not beat'} baseline), n=${c.n}`:`insufficient evidence (${c.n}/${c.minimum})`}`);
  L.push('','## Weaknesses and next-week focus','');
  if (!report.weaknesses.length) L.push('- No concept meets the weakness threshold (4 known outcomes in 28 days).');
  for (const w of report.weaknesses) L.push(`- ${w.concept}: ${fmtRate(w)}, review failures ${w.review_failures}`);
  for (const f of report.next_week_focus) L.push(`- Focus: ${f.concept} (${f.reason})`);
  L.push('','## Contests','');
  if (!report.contests.participation.length && !report.contests.ratings.length) L.push('- No contest participation or rating changes recorded.');
  for (const c of report.contests.participation) L.push(`- ${c.contest_id} (${c.mode}, ${c.outcome}): solved ${c.solved ?? 'unknown'} / expected ${c.expected_solved ?? 'unknown'}, upsolved within 7 days ${c.upsolved_within_7_days}`);
  for (const r of report.contests.ratings) L.push(`- ${r.platform} rating ${r.rating}${r.delta===null?'':` (${r.delta>=0?'+':''}${r.delta})`} on ${r.date} [${r.source}]`);
  L.push('','## Process','',`- Planned slots done: ${report.process.done_slots}/${report.process.planned_slots}`,
    `- Confirmed timer minutes: ${report.process.confirmed_timer_minutes}; self-reported minutes: ${report.process.self_reported_minutes} (minute-unit records)${report.process.self_reported_seconds_total!=null?'; self-reported total: '+report.process.self_reported_seconds_total+'s':''} (${report.process.time_note})`);
  L.push('','## Evaluation','',`- Relatedness verdicts: ${report.evaluation.feedback.total} ${JSON.stringify(report.evaluation.feedback.by_verdict)}`);
  for (const [k,v] of Object.entries(report.evaluation.explore_kinds).sort()) L.push(`- Explore ${k}: planned ${v.planned}, attempted ${v.attempted}, solved ${v.independent}`);
  L.push(`- ${report.evaluation.ai_relatedness}`,'- Explore kinds follow a fixed calendar rotation; their comparison is descriptive, not causal.');
  const dq=report.data_quality;
  L.push('','## Data quality','',`- Git observations without outcome metadata (reported completion only): ${dq.git_untrailered_share ?? 'no Git evidence'}`,
    `- Unclassified commits this week: ${dq.unclassified_commits}`,`- AtCoder attempts without concepts: ${dq.atcoder_attempts_without_concepts}`,
    `- Attempts not in catalog: ${dq.attempts_not_in_catalog.length ? dq.attempts_not_in_catalog.join(', ') : 'none'}`,`- Catalog built at: ${dq.catalog_built_at ?? 'unknown'}`);
  for (const [id,r] of Object.entries(dq.repositories)) L.push(`- Repository ${id}: ${r.status} at ${r.last_processed_sha?.slice(0,12) ?? 'none'}`);
  return L.join('\n')+'\n';
}
