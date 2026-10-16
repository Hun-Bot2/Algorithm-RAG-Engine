import {problemKey} from './events.mjs';
export const DEFAULT_RHYTHM = ['practice','review','practice','review','practice','virtual','upsolve'];
export const defaultPreferences = () => ({weekday_minutes:60,target_keys:[],weekly_rhythm:[...DEFAULT_RHYTHM]});
export function normalizePreferences(input) {
  if (!input || typeof input!=='object' || Array.isArray(input) || Object.keys(input).some(k=>!['weekday_minutes','target_keys','weekly_rhythm'].includes(k))) throw new Error('invalid preferences');
  const next={...defaultPreferences(),...input};
  if(!Number.isInteger(next.weekday_minutes)||next.weekday_minutes<1||next.weekday_minutes>480||!Array.isArray(next.target_keys)||new Set(next.target_keys).size!==next.target_keys.length)throw new Error('invalid preferences');
  next.target_keys.forEach(problemKey);
  if(!Array.isArray(next.weekly_rhythm)||next.weekly_rhythm.length!==7||next.weekly_rhythm.some(day=>!['practice','review','virtual','upsolve'].includes(day)))throw new Error('invalid weekly rhythm');
  return structuredClone(next);
}
