// Replay spacing from dated v2 observations without modifying the learner ledger.
import {defaultConfig} from './progression-v2-config.mjs';

const plusDays=(date,days)=>new Date(Date.parse(`${date}T00:00:00Z`)+days*86400000).toISOString().slice(0,10);

export function replayReviewsV2(observations,config=defaultConfig) {
  const states=new Map();
  const ordered=[...observations].sort((a,b)=>a.date.localeCompare(b.date) ||
    String(a.occurred_at ?? '').localeCompare(String(b.occurred_at ?? '')) ||
    a.id.localeCompare(b.id) || a.problem_key.localeCompare(b.problem_key));
  for (const observation of ordered) {
    let state=states.get(observation.problem_key);
    if (!state) {
      state={state:'unknown',review_stage:0,due_date:null,latest_outcome:null,evidence:[],
        review_priority:'normal',failure_count:0,consecutive_failures:0,last_failure:null,
        last_observation_id:null,last_date:null};
      states.set(observation.problem_key,state);
    }
    if (observation.y===0) {
      state.state='learning'; state.review_stage=0; state.review_priority='priority';
      state.failure_count++; state.consecutive_failures++;
      state.last_failure=observation.occurred_at ?? observation.date;
    } else {
      if (observation.kind==='retention' && observation.gap_days>=config.review_days[state.review_stage]) {
        state.review_stage=Math.min(state.review_stage+1,config.review_days.length-1);
      }
      state.state='demonstrated'; state.review_priority='normal'; state.consecutive_failures=0;
    }
    state.due_date=plusDays(observation.date,observation.y===0?1:config.review_days[state.review_stage]);
    state.latest_outcome=observation.y===0?'failed':'solved';
    state.evidence.push(observation.id);
    state.last_observation_id=observation.id; state.last_date=observation.date;
  }
  return new Map([...states].sort(([a],[b])=>a.localeCompare(b)));
}
