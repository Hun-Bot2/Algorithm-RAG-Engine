import test from 'node:test';
import assert from 'node:assert/strict';
import {replayProblemAttempts} from '../core/policy.mjs';
import {problemStates} from '../mastery/planner.mjs';
import {reviewQueue} from '../mastery/reviews.mjs';
import {catalogFromData} from '../mastery/catalog.mjs';

const at=(day,hour=12)=>`2026-10-${String(day).padStart(2,'0')}T${String(hour).padStart(2,'0')}:00:00.000000Z`;
const attempt=(id,outcome,day,{activity_type=null,hints=outcome==='independent'?false:outcome==='reported_completion'?null:true}={})=>
  ({id,outcome,occurred_at:at(day),hints,activity_type});
const catalog=()=>catalogFromData({problems:[
  {key:'leetcode:alpha',platform:'leetcode',title:'alpha',difficulty_value:1},
  {key:'leetcode:beta',platform:'leetcode',title:'beta',difficulty_value:2},
  {key:'leetcode:gamma',platform:'leetcode',title:'gamma',difficulty_value:3},
  {key:'leetcode:canonical',platform:'leetcode',title:'canonical',difficulty_value:1},
],aliases:{'leetcode:legacy':'leetcode:canonical'}});

test('universal review replay assigns due dates for every attempt outcome',()=>{
  const outcomes=[
    ['independent',false,'normal',0],
    ['failed',true,'priority',1],
    ['assisted',true,'priority',1],
    ['reported_completion',false,'normal',0],
  ];
  for(const [outcome,priority,failureCount,expectedFailures] of outcomes){
    const state=replayProblemAttempts('leetcode:alpha',[attempt(outcome,outcome,1)]);
    assert.equal(state.due_date,'2026-10-02',`${outcome} gets a next-day review date`);
    assert.equal(state.review_priority,priority?'priority':'normal',`${outcome} priority`);
    assert.equal(state.failure_count,expectedFailures,`${outcome} failure count`);
  }
});

test('non-independent practice and transfer evidence receive review priority',()=>{
  for(const activity_type of ['practice','transfer']){
    const state=replayProblemAttempts('leetcode:alpha',[attempt(activity_type,'failed',3,{activity_type})]);
    assert.equal(state.due_date,'2026-10-04');
    assert.equal(state.review_priority,'priority');
    assert.equal(state.failure_count,1);
    assert.equal(state.consecutive_failures,1);
  }
});

test('repeated failures increase severity and independent recovery retains lifetime failure count',()=>{
  const attempts=[
    attempt('first','failed',1,{activity_type:'review'}),
    attempt('second','assisted',2,{activity_type:'review'}),
  ];
  const repeated=replayProblemAttempts('leetcode:alpha',attempts);
  assert.equal(repeated.failure_count,2);
  assert.equal(repeated.consecutive_failures,2);
  assert.equal(repeated.review_priority,'priority');
  assert.equal(repeated.due_date,'2026-10-03');

  const recovered=replayProblemAttempts('leetcode:alpha',[...attempts,attempt('recovery','independent',3,{activity_type:'review'})]);
  assert.equal(recovered.failure_count,2,'historical failures remain inspectable after recovery');
  assert.equal(recovered.consecutive_failures,0);
  assert.equal(recovered.review_priority,'normal');
  assert.equal(recovered.review_stage,1);
  assert.equal(recovered.due_date,'2026-10-06');
});

test('review queue prioritizes failed practice and transfer before normal due reviews',()=>{
  const c=catalog();
  const states=problemStates([
    {...attempt('normal','independent',8),problem_key:'leetcode:alpha'},
    {...attempt('practice-failure','failed',9,{activity_type:'practice'}),problem_key:'leetcode:beta'},
    {...attempt('transfer-failure','assisted',9,{activity_type:'transfer'}),problem_key:'leetcode:gamma'},
  ],c);
  const queued=reviewQueue(states,c,'2026-10-10').filter(r=>r.due);
  assert.deepEqual(queued.slice(0,2).map(r=>r.problem_key),['leetcode:beta','leetcode:gamma']);
  assert.ok(queued.slice(0,2).every(r=>r.review_priority==='priority'));
  assert.equal(queued.at(-1).problem_key,'leetcode:alpha');
});

test('aged normal review can outrank a recent failure and queue separates future items',()=>{
  const c=catalog();
  const states=new Map([
    ['leetcode:alpha',{due_date:'2026-10-01',review_stage:2,review_priority:'normal'}],
    ['leetcode:beta',{due_date:'2026-10-10',review_stage:0,review_priority:'priority',consecutive_failures:1}],
    ['leetcode:gamma',{due_date:'2026-10-12',review_stage:0,review_priority:'priority',consecutive_failures:2}],
  ]);
  const queued=reviewQueue(states,c,'2026-10-10');
  assert.equal(queued[0].problem_key,'leetcode:alpha','long-overdue ordinary review beats today\'s failure');
  assert.equal(queued[0].urgency,18);
  assert.equal(queued[1].problem_key,'leetcode:beta');
  assert.equal(queued[2].due,false);
});

test('planner problem states merge aliases before replaying review failures',()=>{
  const states=problemStates([
    {...attempt('legacy-solve','independent',1),problem_key:'leetcode:legacy'},
    {...attempt('canonical-failure','failed',2,{activity_type:'review'}),problem_key:'leetcode:canonical'},
  ],catalog());
  assert.equal(states.size,1);
  assert.equal(states.has('leetcode:legacy'),false);
  assert.equal(states.get('leetcode:canonical').failure_count,1);
  assert.equal(states.get('leetcode:canonical').consecutive_failures,1);
  assert.equal(states.get('leetcode:canonical').due_date,'2026-10-03');
});
