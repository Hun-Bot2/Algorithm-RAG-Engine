import {problemKey} from './events.mjs';
export const DEFAULT_RHYTHM = ['practice','review','practice','review','practice','virtual','upsolve'];
export const defaultPreferences = () => ({weekday_minutes:60,target_keys:[],weekly_rhythm:[...DEFAULT_RHYTHM]});
// Optional mastery-system settings (specs/mastery-system/design.md section 5). Absent means defaults;
// existing preference documents stay valid unchanged.
export const DEFAULT_MASTERY = {practice_platform:'leetcode',contest_platforms:['codeforces','atcoder'],target_concepts:[],cooldown_days:30,
  explore_cap:0.25,weekend_window:{sat:['20:00','24:00'],sun:['14:00','24:00']}};
export function normalizeMastery(input={}) {
  if (!input || typeof input!=='object' || Array.isArray(input) || Object.keys(input).some(k=>!(k in DEFAULT_MASTERY))) throw new Error('invalid mastery settings');
  const m={...structuredClone(DEFAULT_MASTERY),...structuredClone(input)};
  if (!['leetcode','codeforces','atcoder'].includes(m.practice_platform) || !Array.isArray(m.contest_platforms) || !m.contest_platforms.length || m.contest_platforms.some(p=>!['codeforces','atcoder'].includes(p)) || new Set(m.contest_platforms).size!==m.contest_platforms.length) throw new Error('invalid mastery platforms');
  if (!Array.isArray(m.target_concepts) || m.target_concepts.some(c=>typeof c!=='string' || !/^[a-z0-9-]{1,60}$/.test(c))) throw new Error('invalid mastery target concepts');
  if (!Number.isInteger(m.cooldown_days) || m.cooldown_days<0 || m.cooldown_days>365) throw new Error('invalid mastery cooldown');
  if (typeof m.explore_cap!=='number' || !(m.explore_cap>=0 && m.explore_cap<=0.5)) throw new Error('invalid mastery explore cap');
  const hhmm=/^([01]\d|2[0-3]):[0-5]\d$|^24:00$/;
  if (!m.weekend_window || typeof m.weekend_window!=='object' || Object.keys(m.weekend_window).sort().join('|')!=='sat|sun') throw new Error('invalid mastery weekend window');
  for (const day of ['sat','sun']) { const w=m.weekend_window[day]; if (!Array.isArray(w) || w.length!==2 || w.some(t=>typeof t!=='string' || !hhmm.test(t)) || w[0]>=w[1]) throw new Error('invalid mastery weekend window'); }
  return m;
}
export function normalizePreferences(input) {
  if (!input || typeof input!=='object' || Array.isArray(input) || Object.keys(input).some(k=>!['weekday_minutes','target_keys','weekly_rhythm','mastery'].includes(k))) throw new Error('invalid preferences');
  const next={...defaultPreferences(),...input};
  if(!Number.isInteger(next.weekday_minutes)||next.weekday_minutes<1||next.weekday_minutes>480||!Array.isArray(next.target_keys)||new Set(next.target_keys).size!==next.target_keys.length)throw new Error('invalid preferences');
  next.target_keys.forEach(problemKey);
  if(!Array.isArray(next.weekly_rhythm)||next.weekly_rhythm.length!==7||next.weekly_rhythm.some(day=>!['practice','review','virtual','upsolve'].includes(day)))throw new Error('invalid weekly rhythm');
  if ('mastery' in next) next.mastery=normalizeMastery(next.mastery);
  return structuredClone(next);
}
