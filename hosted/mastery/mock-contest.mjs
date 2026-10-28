// mock-contest-v1 (specs/mock-contest/design.md section 3). Pure and Worker-safe: no Node-only imports.
// Selects and records only; solving and judging happen on leetcode.com.
import {fitLeetCode,practiceObservations,predict} from './calibration.mjs';
import {canonicalKey,recommendable,trustedConcepts} from './related.mjs';

export const MOCK_CONTEST_VERSION = 'mock-contest-v2';
export const MOCK_COUNT = {min:5,max:7,default:6};
// Unvalidated practice heuristic (R3.4): aim for 60% of the problems solved under conservative estimates.
export const TARGET_RATIO = 0.6;
const LEVELS = [1,2,3];
const round = x => Math.round(x*1000)/1000;

function levelPredictions(abilities) {
  return Object.fromEntries(LEVELS.map(level=>{
    const p=predict({platform:'leetcode',difficulty_kind:'lc_level',difficulty_value:level},abilities);
    return [level,{p_mean:p.p_mean,p_conservative:p.p_conservative}];
  }));
}

// R3: choose (easy, medium, hard) with e>=1, m>=1 and per-level pool limits, minimizing |E - T|.
// Ties prefer more Hard, then more Medium.
export function composeLevels({abilities,pools,count=MOCK_COUNT.default}) {
  if (!Number.isInteger(count) || count<MOCK_COUNT.min || count>MOCK_COUNT.max) throw new Error(`mock contest count must be an integer in [${MOCK_COUNT.min}, ${MOCK_COUNT.max}]`);
  for (const level of LEVELS) if (!Number.isInteger(pools?.[level]) || pools[level]<0) throw new Error('invalid mock contest pool sizes');
  const probabilities=levelPredictions(abilities);
  const target=round(TARGET_RATIO*count);
  let best=null;
  for (let easy=1; easy<=count-1; easy++) for (let medium=1; easy+medium<=count; medium++) {
    const hard=count-easy-medium;
    if (easy>pools[1] || medium>pools[2] || hard>pools[3]) continue;
    const expected=easy*probabilities[1].p_conservative+medium*probabilities[2].p_conservative+hard*probabilities[3].p_conservative;
    const gap=Math.round(Math.abs(expected-target)*1e6);
    if (!best || gap<best.gap || (gap===best.gap && (hard>best.hard || (hard===best.hard && medium>best.medium)))) best={easy,medium,hard,expected,gap};
  }
  if (!best) throw new Error('not enough unattempted LeetCode problems for a mock contest');
  return {policy:MOCK_CONTEST_VERSION,count,composition:{easy:best.easy,medium:best.medium,hard:best.hard},
    target_expected:target,expected_solved:round(best.expected),probabilities,ability_status:abilities.leetcode.status};
}

const DAY = 86400000;
const addDays = (date,n) => new Date(Date.parse(date+'T00:00:00Z')+n*DAY).toISOString().slice(0,10);
const LEVEL_NAMES = {1:'Easy',2:'Medium',3:'Hard'};

// FNV-1a 32-bit over UTF-8 bytes: a stable seeded order that is not alphabetical (R4.3).
export function fnv1a(text) {
  let hash=0x811c9dc5;
  for (const byte of new TextEncoder().encode(text)) hash=Math.imul(hash^byte,0x01000193)>>>0;
  return hash;
}

// R2: unattempted, recommendable LeetCode problems with a native level and at least one native tag, outside the
// exposure cooldown. Untagged rows in catalog-v4 are JavaScript/Pandas/SQL exercises, not algorithm problems (R2.1).
// `exposures` is [{date, problem_keys}] from daily plan and earlier mock contest snapshots.
export function candidatePool({catalog,attempts=[],exposures=[],date,cooldownDays}) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date ?? '')) throw new Error('invalid mock contest date');
  if (!Number.isInteger(cooldownDays) || cooldownDays<0) throw new Error('invalid cooldown days');
  const attempted=new Set(attempts.map(a=>canonicalKey(catalog,a.problem_key)));
  const from=addDays(date,-cooldownDays), exposed=new Set();
  for (const e of exposures) if (e.date>=from && e.date<date) for (const key of e.problem_keys ?? []) exposed.add(canonicalKey(catalog,key));
  const byLevel={1:[],2:[],3:[]}, excluded={attempted:0,exposed:0,untagged:0};
  for (const p of catalog.problems.values()) {
    if (p.platform!=='leetcode' || !recommendable(p) || p.difficulty_kind!=='lc_level' || !LEVELS.includes(p.difficulty_value) || canonicalKey(catalog,p.key)!==p.key) continue;
    if (!(p.tags ?? []).length) { excluded.untagged++; continue; }
    if (attempted.has(p.key)) { excluded.attempted++; continue; }
    if (exposed.has(p.key)) { excluded.exposed++; continue; }
    byLevel[p.difficulty_value].push(p);
  }
  return {byLevel,sizes:Object.fromEntries(LEVELS.map(l=>[l,byLevel[l].length])),excluded};
}

// R4: inside each level rank by trusted weak-concept coverage, then the seeded hash, then key.
// Two selections never share an identical non-empty tag set unless a level would otherwise run short.
// Titles are never read here (R2.4).
export function selectProblems({pool,composition,weakConcepts=[],learnerId,date}) {
  const weak=new Set(weakConcepts), notes=[], usedTagSets=new Set(), selected=[];
  const counts={1:composition.easy,2:composition.medium,3:composition.hard};
  for (const level of LEVELS) {
    const ranked=pool.byLevel[level].map(p=>{
      const concepts=[...trustedConcepts(p)].sort(), covered=concepts.filter(c=>weak.has(c));
      return {problem:p,concepts,covered,unknown:!concepts.length,seed:fnv1a(`${learnerId}|${date}|${p.key}`),tagSet:[...new Set(p.tags ?? [])].sort().join('\0')};
    }).sort((a,b)=>b.covered.length-a.covered.length || a.seed-b.seed || a.problem.key.localeCompare(b.problem.key));
    const chosen=[];
    for (const c of ranked) {
      if (chosen.length===counts[level]) break;
      if (c.tagSet && usedTagSets.has(c.tagSet)) continue;
      chosen.push(c); if (c.tagSet) usedTagSets.add(c.tagSet);
    }
    if (chosen.length<counts[level]) {
      notes.push(`${LEVEL_NAMES[level]}: tag-set diversity relaxed to fill ${counts[level]} problems.`);
      for (const c of ranked) { if (chosen.length===counts[level]) break; if (!chosen.includes(c)) chosen.push(c); }
    }
    if (chosen.length<counts[level]) throw new Error('not enough unattempted LeetCode problems for a mock contest');
    selected.push(...chosen.map(c=>({problem_key:c.problem.key,level,tags:[...new Set(c.problem.tags ?? [])].sort(),concepts:c.concepts,
      components:{weak_concepts_covered:c.covered,weakness_coverage:c.unknown?'unknown':weak.size?Math.round(c.covered.length/weak.size*1000)/1000:0},
      reason:`${LEVEL_NAMES[level]} slot. `+(c.unknown?'No reviewed concepts, so weakness coverage is unknown.':c.covered.length?`Covers weak concepts: ${c.covered.join(', ')}.`:'Covers none of the current weak concepts.')})));
  }
  return {problems:selected.map((p,i)=>({slot:i+1,...p})),notes};
}

// R1.2/R5.2 (v2): a mock contest has a date and a solving time only; official contest schedules are never read.
export const MOCK_MINUTES = {min:30,max:300,default:120};

// T4: one immutable mock contest record (design.md section 3.5). Only LeetCode evidence is read (R3.3):
// the ability comes from LeetCode practice observations, and only LeetCode attempts are listed as inputs.
export function mockContest({catalog,attempts=[],exposures=[],weakConcepts=[],learnerId,date,mastery,count=MOCK_COUNT.default,minutes=null}) {
  if (typeof learnerId!=='string' || !learnerId) throw new Error('invalid learner id');
  const durationMinutes=minutes ?? MOCK_MINUTES.default;
  if (!Number.isInteger(durationMinutes) || durationMinutes<MOCK_MINUTES.min || durationMinutes>MOCK_MINUTES.max) throw new Error(`mock contest minutes must be an integer in [${MOCK_MINUTES.min}, ${MOCK_MINUTES.max}]`);
  const lcAttempts=attempts.filter(a=>catalog.problems.get(canonicalKey(catalog,a.problem_key))?.platform==='leetcode');
  const leetcode=fitLeetCode(practiceObservations(lcAttempts,catalog,'leetcode'));
  const pool=candidatePool({catalog,attempts:lcAttempts,exposures,date,cooldownDays:mastery.cooldown_days});
  const composed=composeLevels({abilities:{leetcode},pools:pool.sizes,count});
  const weak=[...new Set(weakConcepts)].sort();
  const selection=selectProblems({pool,composition:composed.composition,weakConcepts:weak,learnerId,date});
  const from=addDays(date,-mastery.cooldown_days);
  return {policy:MOCK_CONTEST_VERSION,catalog_version:catalog.version,learner_id:learnerId,local_date:date,
    duration_minutes:durationMinutes,
    ability:{status:leetcode.status,n:leetcode.n,levels:Object.fromEntries(LEVELS.map(l=>[l,{mean:round(leetcode.levels[l].mean),sd:round(leetcode.levels[l].sd),n:leetcode.levels[l].n}]))},
    composition:composed.composition,target_expected:composed.target_expected,expected_solved:composed.expected_solved,
    evidence_label:leetcode.status==='estimated'?null:`Based on insufficient LeetCode evidence (${leetcode.n} practice observations); calibration priors set the difficulty mix.`,
    weak_concepts:weak,
    problems:selection.problems.map(p=>{
      const problem=catalog.problems.get(p.problem_key), probability=composed.probabilities[p.level];
      return {slot:p.slot,problem_key:p.problem_key,title:problem.title ?? p.problem_key,url:problem.url ?? null,level:p.level,tags:p.tags,concepts:p.concepts,
        p_mean:probability.p_mean,p_conservative:probability.p_conservative,components:p.components,reason:p.reason};
    }),
    notes:selection.notes,
    inputs:{count,attempt_ids:lcAttempts.map(a=>a.id).filter(Boolean).sort(),
      exposure_dates:[...new Set(exposures.filter(e=>e.date>=from && e.date<date).map(e=>e.date))].sort(),
      pool_sizes:pool.sizes,excluded:pool.excluded}};
}

// R2.3/R6.4 exposure plumbing. Snapshot rows are {kind, local_date, payload}; archive wrappers are inert history
// and never count. Mock contests store `problems`, every other snapshot kind stores `slots`.
export function snapshotExposures(rows) {
  const out=[];
  for (const r of rows) {
    const p=r.payload;
    if (!p || p.archive_type==='recommendation_snapshot') continue;
    if (r.kind==='mock_contest') out.push({date:r.local_date,problem_keys:(p.problems ?? []).map(x=>x.problem_key)});
    else out.push({date:p.date ?? r.local_date,problem_keys:(p.slots ?? []).map(s=>s.problem_key).filter(Boolean)});
  }
  return out;
}
// Plan-shaped exposures so planDay/planProgression skip mock contest problems without changing their logic.
export const mockExposurePlans = mocks => mocks.map(m=>({date:m.local_date,source_kind:'mock_contest',slots:m.problems.map(p=>({problem_key:p.problem_key}))}));
