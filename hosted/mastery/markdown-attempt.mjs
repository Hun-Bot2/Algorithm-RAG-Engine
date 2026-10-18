// Deliberately small parser for the opt-in learning metadata block. This is not YAML.
import {createHash} from 'node:crypto';
import {keyPattern} from '../core/events.mjs';

const FIELDS=new Set(['learning_schema','problem','attempt','outcome','hints','activity','transfer_from','minutes']);
const cleanScalar=(raw)=>{
  const s=raw.trim();
  if (!s) return '';
  if (s[0]==='"' || s[0]==="'") {
    const q=s[0];
    if (s.length<2 || s.at(-1)!==q) throw new Error('unclosed quoted scalar');
    const inside=s.slice(1,-1);
    if (inside.includes(q) || /[\\\r\n]/.test(inside)) throw new Error('unsupported quoted scalar');
    return inside;
  }
  if (/["'`]|[#]|\s{2,}/.test(s)) throw new Error('unsupported scalar syntax');
  return s;
};

export function parseMarkdownAttempt(source,{inferredProblem=null}={}) {
  const text=String(source ?? '');
  const m=/^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(text);
  if (!m) return {marked:false,metadata:null,error:null};
  const lines=m[1].split(/\r?\n/), raw={};
  const marked=lines.some(line=>/^learning_schema\s*:/.test(line));
  if (!marked) return {marked:false,metadata:null,error:null};
  let discoveredProblem=inferredProblem;
  for (const line of lines) {
    const pair=/^problem:(?:\s*(.*))?$/.exec(line);
    if (pair) { try { discoveredProblem=cleanScalar(pair[1] ?? '') || discoveredProblem; } catch {} }
  }
  const invalid=error=>({marked:true,metadata:null,error,problem_key:keyPattern.test(discoveredProblem ?? '')?discoveredProblem:null});
  for (const line of lines) {
    if (!line.trim() || /^\s*#/.test(line)) continue;
    const pair=/^([A-Za-z_][A-Za-z0-9_]*):(?:\s*(.*))?$/.exec(line);
    if (!pair) return invalid('unsupported_learning_frontmatter');
    const key=pair[1];
    if (!FIELDS.has(key)) return invalid(`unknown_learning_field:${key}`);
    if (Object.hasOwn(raw,key)) return invalid(`duplicate_learning_field:${key}`);
    try { raw[key]=cleanScalar(pair[2] ?? ''); }
    catch { return invalid(`invalid_learning_field:${key}`); }
  }
  if (!marked) return {marked:false,metadata:null,error:null};
  const simple=raw.learning_schema==='2';
  if (!simple && raw.learning_schema!=='1') return invalid('unsupported_learning_schema');
  if (simple && raw.hints!==undefined) return invalid('unknown_learning_field:hints');
  const problem=raw.problem || inferredProblem;
  if (!problem || !keyPattern.test(problem) || /(?:TODO|TBD|PLACEHOLDER|YOUR[-_]|(?:^|:)problem(?:[-_](?:id|key|slug))?$|(?:^|:)example$)/i.test(problem)) return invalid('invalid_learning_problem');
  if (!raw.attempt || raw.attempt.length>100 || !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(raw.attempt)) return invalid('invalid_learning_attempt');
  const outcomeMap={O:'independent',X:'failed',independent:'independent',assisted:'assisted',failed:'failed',accepted:'reported_completion'};
  const outcome=(!simple || ['O','X'].includes(raw.outcome)) && Object.hasOwn(outcomeMap,raw.outcome) ? outcomeMap[raw.outcome] : null;
  if (!outcome) return invalid('invalid_learning_outcome');
  // Schema 2 O is the learner's explicit independent-success declaration.
  let hints=simple && raw.outcome==='O' ? false : null;
  if (raw.hints!==undefined) {
    if (raw.hints==='none') hints=false;
    else if (raw.hints==='used') hints=true;
    else return invalid('invalid_learning_hints');
  }
  if (outcome==='independent' && hints!==false) return invalid('independent_requires_no_hints');
  const activity=raw.activity ?? (simple ? 'practice' : null);
  if (activity!==null && !['practice','review','transfer'].includes(activity)) return invalid('invalid_learning_activity');
  const transfer=raw.transfer_from ?? null;
  if ((activity==='transfer') !== (transfer!==null) || (transfer!==null && (!keyPattern.test(transfer) || transfer===problem))) return invalid('invalid_learning_transfer');
  let minutes=null;
  if (raw.minutes!==undefined) {
    if (!/^\d{1,4}$/.test(raw.minutes) || Number(raw.minutes)>1440) return invalid('invalid_learning_minutes');
    minutes=Number(raw.minutes);
  }
  return {marked:true,metadata:{problem_key:problem,attempt:raw.attempt,outcome,hints,self_minutes:minutes,activity_type:activity,transfer_from:transfer},error:null};
}

export const markdownBlobHash=text=>createHash('sha256').update(String(text)).digest('hex');
export const markdownAttemptSignature=m=>JSON.stringify([m.problem_key,m.outcome,m.hints,m.self_minutes,m.activity_type,m.transfer_from]);
