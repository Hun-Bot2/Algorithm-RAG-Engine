// M3-compatible event envelopes and evidence replay. No learner inference.
export const keyPattern = /^(leetcode|atcoder|codeforces|baekjoon):[A-Za-z0-9][A-Za-z0-9_.-]*$/;
export function identifier(value) {
  if (typeof value !== 'string' || !value.trim() || value.length > 200) throw new Error('invalid identifier');
  return value;
}
export function problemKey(value) {
  if (typeof value !== 'string' || !keyPattern.test(value)) throw new Error('platform-qualified problem key required');
  return value;
}
export function timestamp(value) {
  if (typeof value !== 'string') throw new Error('timestamp must be text');
  const match = /^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(?:\.(\d{1,6}))?(Z|\+00:00|-00:00)$/.exec(value);
  if (!match) throw new Error('UTC timestamp required');
  const [, y, mo, d, h, mi, s, fraction = ''] = match;
  const milliseconds = Number(fraction.padEnd(6, '0').slice(0, 3));
  const date = new Date(Date.UTC(+y, +mo - 1, +d, +h, +mi, +s, milliseconds));
  if (date.getUTCFullYear() !== +y || date.getUTCMonth() + 1 !== +mo || date.getUTCDate() !== +d || date.getUTCHours() !== +h || date.getUTCMinutes() !== +mi || date.getUTCSeconds() !== +s) throw new Error('invalid timestamp');
  return date.toISOString().slice(0, 19) + '.' + fraction.padEnd(6, '0') + 'Z';
}
const microseconds = value => Date.parse(value.slice(0,19)+'Z')*1000 + Number(value.slice(20,26));
const exact = (obj, fields) => Object.keys(obj).sort().join('|') === [...fields].sort().join('|');
export function normalizeEvent(input) {
  if (!input || typeof input !== 'object' || Array.isArray(input) || !exact(input, ['schema_version','event_id','learner_id','occurred_at','recorded_at','kind','source','payload','supersedes'])) throw new Error('invalid event envelope');
  const e = structuredClone(input);
  identifier(e.event_id); identifier(e.learner_id);
  if (e.schema_version !== 1 || !Number.isInteger(e.schema_version)) throw new Error('unsupported event version');
  if (!['attempt','timer_started','timer_stopped','timer_confirmed'].includes(e.kind) || !['manual','platform_import'].includes(e.source)) throw new Error('unsupported kind/source');
  e.occurred_at = timestamp(e.occurred_at); e.recorded_at = timestamp(e.recorded_at);
  if (e.occurred_at > e.recorded_at) throw new Error('event occurs after recording');
  if (e.supersedes !== null) { identifier(e.supersedes); if (e.supersedes === e.event_id) throw new Error('event cannot supersede itself'); }
  const p = e.payload;
  if (!p || typeof p !== 'object' || Array.isArray(p)) throw new Error('payload must be an object');
  if (e.kind === 'attempt') {
    problemKey(p.problem_key);
    if (!['independent','assisted','failed','reported_completion'].includes(p.outcome)) throw new Error('invalid outcome');
    if (p.hints !== null && typeof p.hints !== 'boolean') throw new Error('hints must be boolean or null');
    if (p.outcome === 'independent' && p.hints !== false) throw new Error('independent outcome requires explicit no hints');
    const fields = ['problem_key','outcome','hints'];
    if (e.source === 'platform_import') {
      fields.push('platform','external_id');
      if (p.outcome !== 'reported_completion') throw new Error('imports are reported completion only');
      identifier(p.external_id);
      if (p.platform !== p.problem_key.split(':')[0]) throw new Error('import platform mismatch');
    }
    if ('activity_type' in p) {
      if (e.source !== 'manual' || !['practice','review','transfer'].includes(p.activity_type)) throw new Error('invalid activity type');
      fields.push('activity_type');
      if (p.activity_type === 'transfer') { problemKey(p.transfer_from); if(p.transfer_from===p.problem_key)throw new Error('invalid transfer source'); fields.push('transfer_from'); }
    }
    if ('self_minutes' in p) {
      if (e.source!=='manual' || (p.self_minutes!==null && (!Number.isInteger(p.self_minutes) || p.self_minutes<0 || p.self_minutes>1440))) throw new Error('invalid self-reported minutes');
      fields.push('self_minutes');
    }
    if('self_seconds' in p){
      if(e.source!=='manual'||p.self_seconds!==null&&(!Number.isInteger(p.self_seconds)||p.self_seconds<0||p.self_seconds>86400||p.self_minutes!=null))throw new Error('invalid self-reported seconds');
      fields.push('self_seconds');
    }
    if (!exact(p, fields)) throw new Error('unexpected or missing payload fields');
  } else {
    if (e.source !== 'manual') throw new Error('timer actions require manual evidence');
    identifier(p.interval_id);
    const fields = ['interval_id'];
    if (e.kind === 'timer_started') { problemKey(p.problem_key); fields.push('problem_key'); }
    if (e.kind === 'timer_confirmed') {
      fields.push('started_at','ended_at');
      p.started_at = timestamp(p.started_at); p.ended_at = timestamp(p.ended_at);
      if (!(p.started_at <= p.ended_at && p.ended_at <= e.occurred_at)) throw new Error('invalid confirmed interval bounds');
    }
    if (!exact(p, fields)) throw new Error('unexpected or missing payload fields');
  }
  return e;
}
export function activeEvents(events, cutoff = null) {
  const eligible = cutoff ? events.filter(e => e.occurred_at <= cutoff && e.recorded_at <= cutoff) : events;
  const superseded = new Set(eligible.map(e => e.supersedes).filter(Boolean));
  return eligible.filter(e => !superseded.has(e.event_id)).sort((a,b) => a.occurred_at.localeCompare(b.occurred_at) || a.event_id.localeCompare(b.event_id));
}
export function timerState(events, cutoff = null) {
  const intervals = new Map();
  const order = {timer_started:0,timer_stopped:1,timer_confirmed:2,attempt:3};
  for (const e of activeEvents(events, cutoff).sort((a,b) => a.occurred_at.localeCompare(b.occurred_at) || order[a.kind]-order[b.kind] || a.event_id.localeCompare(b.event_id))) {
    if (e.kind === 'attempt') continue;
    const id = e.payload.interval_id;
    if (e.kind === 'timer_started') {
      if (intervals.has(id)) throw new Error('duplicate timer start');
      intervals.set(id, {interval_id:id, problem_key:e.payload.problem_key, started_at:e.occurred_at, status:'open', confirmed_seconds:null});
    } else {
      const state = intervals.get(id);
      if (!state) throw new Error('timer action without start');
      if (e.kind === 'timer_stopped') {
        if (state.status !== 'open') throw new Error('duplicate or misplaced timer stop');
        state.status = 'unconfirmed'; state.stopped_at = e.occurred_at;
      } else {
        if (state.status !== 'unconfirmed') throw new Error('confirmation requires one stopped interval');
        const {started_at:start,ended_at:end} = e.payload;
        if (!(state.started_at <= start && start <= end && end <= state.stopped_at)) throw new Error('confirmation outside recorded actions');
        state.status = 'confirmed'; state.confirmed_start = start; state.confirmed_end = end;
        state.confirmed_seconds = (microseconds(end)-microseconds(start))/1_000_000;
      }
    }
  }
  return [...intervals.values()].sort((a,b) => a.interval_id.localeCompare(b.interval_id));
}
export function confirmedSeconds(intervals) {
  const spans = intervals.filter(s => s.status === 'confirmed').map(s => [s.confirmed_start,s.confirmed_end]).sort((a,b)=>a[0].localeCompare(b[0]));
  let total = 0, current = null;
  for (const [start,end] of spans) {
    if (current && start <= current[1]) current[1] = current[1] > end ? current[1] : end;
    else { if (current) total += (microseconds(current[1])-microseconds(current[0]))/1_000_000; current = [start,end]; }
  }
  if (current) total += (microseconds(current[1])-microseconds(current[0]))/1_000_000;
  return total;
}
