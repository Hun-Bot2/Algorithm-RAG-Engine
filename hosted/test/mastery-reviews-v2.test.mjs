import test from 'node:test';
import assert from 'node:assert/strict';
import {replayReviewsV2} from '../mastery/reviews-v2.mjs';

const row=(day,kind,gap,y=1,key='leetcode:a')=>({id:`${key}-${day}`,problem_key:key,date:`2026-01-${String(day).padStart(2,'0')}`,
  occurred_at:`2026-01-${String(day).padStart(2,'0')}T00:00:00Z`,kind,gap_days:gap,y});

test('spacing advances only when the actual gap reaches the current interval',()=>{
  const rows=[row(1,'ability',null),row(2,'retention',1),row(5,'retention',3),row(12,'retention',7)];
  const expected=[[0,'2026-01-02'],[1,'2026-01-05'],[2,'2026-01-12'],[3,'2026-01-26']];
  for (let i=0;i<rows.length;i++) {
    const s=replayReviewsV2(rows.slice(0,i+1)).get('leetcode:a');
    assert.deepEqual([s.review_stage,s.due_date],expected[i]);
  }
  const early=replayReviewsV2([...rows.slice(0,2),row(3,'retention',1)]).get('leetcode:a');
  assert.equal(early.review_stage,1); assert.equal(early.due_date,'2026-01-06');
});

test('failure resets spacing and priority; later success clears consecutive failures',()=>{
  const rows=[row(1,'ability',null),row(2,'retention',1),row(5,'retention',3,0)];
  const failed=replayReviewsV2(rows).get('leetcode:a');
  assert.deepEqual([failed.review_stage,failed.due_date,failed.review_priority,failed.failure_count,failed.consecutive_failures],
    [0,'2026-01-06','priority',1,1]);
  const solved=replayReviewsV2([...rows,row(6,'retention',1)]).get('leetcode:a');
  assert.deepEqual([solved.review_stage,solved.due_date,solved.review_priority,solved.failure_count,solved.consecutive_failures],
    [1,'2026-01-09','normal',1,0]);
  assert.equal(solved.last_observation_id,'leetcode:a-6');
});

test('replay is deterministic for shuffled input and honours configured intervals',()=>{
  const rows=[row(1,'ability',null),row(2,'retention',1),row(1,'ability',null,1,'leetcode:b')];
  assert.deepEqual(replayReviewsV2(rows),replayReviewsV2([...rows].reverse()));
  assert.equal(replayReviewsV2(rows,{review_days:[2,4]}).get('leetcode:a').due_date,'2026-01-04');
});
