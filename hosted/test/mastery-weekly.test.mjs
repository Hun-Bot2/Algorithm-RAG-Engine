import test from 'node:test';
import assert from 'node:assert/strict';
import {catalogFromData} from '../mastery/catalog.mjs';
import {computeOutcomes,wilson,reviewOutcomes} from '../mastery/metrics.mjs';
import {buildWeeklyReport,renderMarkdown} from '../mastery/weekly.mjs';

const lc=(slug,concept='greedy')=>({key:`leetcode:${slug}`,platform:'leetcode',title:slug,url:`https://leetcode.com/problems/${slug}/`,difficulty_kind:'lc_level',difficulty_value:2,concepts:[{id:concept,status:'ai_reviewed'}]});
const cf=(key,concept='greedy')=>({key,platform:'codeforces',title:key,url:'https://codeforces.com/problemset/problem/1/A',difficulty_kind:'cf_rating',difficulty_value:1200,concepts:[{id:concept,status:'ai_reviewed'}]});
const catalog=catalogFromData({version:'catalog-v1',problems:[...Array.from({length:12},(_,i)=>lc(`p${i}`,i<6?'greedy':'dynamic-programming')),lc('rev'),cf('codeforces:1A'),lc('weak1','tries'),lc('weak2','tries'),lc('weak3','tries'),lc('weak4','tries')],
  concepts:['greedy','dynamic-programming','tries'].map(id=>({id,label:id})),contestProblems:{'abc400':[{problem_key:'codeforces:1A',label:'A'}]}});
const iso=(date,h=12)=>`${date}T${String(h).padStart(2,'0')}:00:00.000000Z`;
let n=0;
const attempt=(key,outcome,date,extra={})=>({id:`a${String(++n).padStart(3,'0')}`,table:'events',occurred_at:iso(date),recorded_at:iso(date),problem_key:key,outcome,
  hints:outcome==='independent'?false:outcome==='reported_completion'?null:true,activity_type:null,transfer_from:null,evidence_source:'manual',self_minutes:null,...extra});
const WEEK='2026-10-19';
// 12 first attempts in the 4-week window: p0..p8 independent, p9..p11 failed; p8..p11 fall in the report week.
const dates=['2026-09-29','2026-10-01','2026-10-03','2026-10-06','2026-10-08','2026-10-10','2026-10-13','2026-10-15','2026-10-19','2026-10-20','2026-10-21','2026-10-22'];
const attempts=[
  attempt('leetcode:old','independent','2026-08-01'),
  ...dates.map((d,i)=>attempt(`leetcode:p${i}`,i<9?'independent':'failed',d)),
  attempt('leetcode:p0','failed','2026-10-02'),                                   // repeat attempt: not a first attempt
  attempt('leetcode:rev','reported_completion','2026-10-05'),                     // excluded from rates
  attempt('leetcode:rev','independent','2026-10-12'),                             // not first (rev touched before)
  attempt('leetcode:rev','independent','2026-10-13',{activity_type:'review'}),    // 1-day review, on time, success
  attempt('leetcode:rev','failed','2026-10-17',{activity_type:'review'}),         // 3-day review (due 10-16), on time, failure
  attempt('codeforces:1A','independent','2026-10-23',{evidence_source:'git',table:'git_observations',self_minutes:20}), // transfer from greedy LC solves
  ...['weak1','weak2','weak3','weak4'].map((s,i)=>attempt(`leetcode:${s}`,i===0?'independent':'failed','2026-10-24')),
].sort((a,b)=>a.occurred_at.localeCompare(b.occurred_at) || a.id.localeCompare(b.id));
const plans=[{date:'2026-09-29',slots:dates.slice(0,10).map((d,i)=>({kind:'exploit',problem_key:`leetcode:p${i}`,prediction:{p_mean:0.5}}))},
  {date:'2026-10-23',slots:[{kind:'explore',sub_kind:'related_channel',problem_key:'codeforces:1A',prediction:{p_mean:0.4},components:{channel:'concept'},estimated_minutes:30}]},
  {date:'2026-10-24',slots:[{kind:'contest',problem_key:null,proposals:{virtual:{contest_id:'abc400',expected_solved:1.2}}}]}];
// Note: stored predictions are only matched within 14 days of the plan date (p0..p6 here).
const timer=[
  {event_id:'t1',kind:'timer_started',occurred_at:iso('2026-10-20',10),recorded_at:iso('2026-10-20',10),payload:{interval_id:'i1',problem_key:'leetcode:p9'},supersedes:null},
  {event_id:'t2',kind:'timer_stopped',occurred_at:'2026-10-20T10:25:00.000000Z',recorded_at:'2026-10-20T10:25:00.000000Z',payload:{interval_id:'i1'},supersedes:null},
  {event_id:'t3',kind:'timer_confirmed',occurred_at:'2026-10-20T10:26:00.000000Z',recorded_at:'2026-10-20T10:26:00.000000Z',payload:{interval_id:'i1',started_at:iso('2026-10-20',10),ended_at:'2026-10-20T10:25:00.000000Z'},supersedes:null},
];
const ledger={attempts,events:timer,participation:[{entry_id:'c1',contest_id:'abc400',mode:'virtual',occurred_at:iso('2026-10-24'),recorded_at:iso('2026-10-24'),outcome:'completed',confirmed_minutes:100}],
  rating_observations:[{entry_id:'r1',occurred_at:iso('2026-10-24'),source:'codeforces_api',payload:{platform:'codeforces',rating:1250,old_rating:1200,contest_id:'1999',rated_participations:5,handle:null,rank:100}}],
  practice_feedback:[{entry_id:'f1',occurred_at:iso('2026-10-23'),payload:{verdict:'related',channel:'concept'}}],git_watermarks:[],cutoff:iso('2026-10-26'),high_water:{events:3}};

test('Wilson interval matches the closed form', () => {
  assert.deepEqual(wilson(9,12),[0.468,0.911]);
  assert.equal(wilson(0,0),null);
});

test('outcome metrics on a synthetic six-week ledger match hand-computed answers', () => {
  const o=computeOutcomes({ledger,catalog,plans,weekStart:WEEK});
  assert.equal(o.window_start,'2026-09-28');
  // rolling: p0..p11 (12) + codeforces:1A (indep) + weak1..4 (1 indep, 3 failed) = 17 first attempts, 11 solved
  assert.deepEqual([o.isr.rolling.overall.successes,o.isr.rolling.overall.n,o.isr.rolling.overall.status],[11,17,'estimated']);
  // week (10-19..10-25): p8 indep, p9..p11 failed, 1A indep, weak1 indep, weak2..4 failed => 3/9
  assert.deepEqual([o.isr.week.overall.successes,o.isr.week.overall.n,o.isr.week.overall.status],[3,9,'insufficient_evidence']);
  assert.equal(o.isr.week.by_platform.codeforces.n,1);
  // stored predictions (p_mean 0.5 for p0..p9 planned 09-29, matched within 14 days => p0..p6 only, all independent),
  // plus 1A (0.4, independent): residual = 7*0.5 + 0.6 = 4.1 over 8
  assert.equal(o.excess.rolling.n,8);
  assert.equal(o.excess.rolling.excess_solve_rate,0.513);
  // review retention: 1-day success on time, 3-day failure on time
  assert.deepEqual([o.retention.by_interval[1].successes,o.retention.by_interval[1].n],[1,1]);
  assert.deepEqual([o.retention.by_interval[3].successes,o.retention.by_interval[3].n],[0,1]);
  assert.equal(o.retention.lapse_rate,0.5);
  // transfer: 1A (codeforces, greedy) after LC greedy solves in prior 30 days, from a related slot
  assert.deepEqual([o.transfer.successes,o.transfer.n],[1,1]);
  assert.equal(o.transfer.lift,0.6);
  assert.deepEqual(Object.keys(o.transfer.by_channel),['concept']);
  // weaknesses: tries 1/4 => weak
  assert.deepEqual(o.concepts.find(c=>c.concept==='tries').weak,true);
  assert.equal(o.concepts.find(c=>c.concept==='greedy').weak,false);
  // contest: virtual abc400, 1A solved after the contest day => upsolve, expected from the stored proposal
  assert.deepEqual(o.contests.participation[0],{contest_id:'abc400',mode:'virtual',outcome:'completed',problems_known:1,solved:0,expected_solved:1.2,upsolved_within_7_days:0,confirmed_minutes:100});
  assert.equal(o.contests.ratings[0].delta,50);
  // time evidence stays separate
  assert.equal(o.process.confirmed_timer_minutes,25); assert.equal(o.process.self_reported_minutes,20);
  assert.match(o.disclaimer,/do not prove/);
});

test('review outcomes replay the core review stages', () => {
  const rows=reviewOutcomes(attempts,catalog).filter(r=>r.key==='leetcode:rev');
  assert.deepEqual(rows.map(r=>[r.interval,r.on_time,r.success]),[[1,true,true],[3,true,false]]);
});

test('weekly report is deterministic, lists focus and data quality, and renders Markdown', () => {
  const report=buildWeeklyReport({ledger,catalog,plans,weekStart:WEEK,preferences:{mastery:{target_concepts:['dynamic-programming']}}});
  assert.equal(JSON.stringify(report),JSON.stringify(buildWeeklyReport({ledger,catalog,plans,weekStart:WEEK,preferences:{mastery:{target_concepts:['dynamic-programming']}}})));
  assert.deepEqual(report.next_week_focus.map(f=>f.concept),['tries','dynamic-programming']);
  assert.equal(report.activity.by_source.git,1);
  assert.equal(report.data_quality.git_untrailered_share,0);
  assert.equal(report.evaluation.feedback.by_verdict.related,1);
  const md=renderMarkdown(report);
  assert.equal(md,renderMarkdown(report));
  for (const line of ['# Weekly report 2026-10-19 to 2026-10-25','## Calibration','- Focus: tries','Confirmed timer minutes: 25; self-reported minutes: 20','do not prove','descriptive, not causal'])
    assert.ok(md.includes(line),line);
  assert.ok(md.includes('Independent solve rate (this week): 3/9 (insufficient evidence; need 10)'));
});

import {weeklySlackPayload} from '../mastery/slack-weekly.mjs';
test('Slack weekly payload is valid Block Kit and escapes user text (dry run, nothing sent)', () => {
  const report=buildWeeklyReport({ledger,catalog,plans,weekStart:WEEK,preferences:{mastery:{target_concepts:['dynamic-programming']}}});
  report.next_week_focus.push({concept:'<script>',reason:'a & b'});
  const payload=weeklySlackPayload(report);
  assert.equal(payload.blocks[0].type,'header');
  assert.ok(payload.blocks.length<=50);
  for (const b of payload.blocks.filter(b=>b.type==='section')) assert.ok(b.text.text.length<=3000);
  const text=JSON.stringify(payload);
  assert.ok(text.includes('&lt;script&gt;') && !text.includes('<script>'));
  assert.match(payload.text,/Weekly report 2026-10-19: \d+ attempts/);
});
