// contests-v1 (design.md section 6, R7). Pure. Live contests are an annotated calendar and
// virtual contests a ranked proposal; acceptance stays with the learner (existing `suitable`).
import {seoulDate} from '../core/policy.mjs';
import {predict} from './calibration.mjs';
import {canonicalKey,trustedConcepts} from './related.mjs';

export const CONTESTS_VERSION = 'contests-v1';
const DAY=86400000;
const addDays=(date,n)=>new Date(Date.parse(date+'T00:00:00Z')+n*DAY).toISOString().slice(0,10);
const seoulClock=iso=>new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Seoul',hour:'2-digit',minute:'2-digit',hourCycle:'h23'}).format(new Date(iso));
const seoulWeekday=iso=>new Intl.DateTimeFormat('en-US',{timeZone:'Asia/Seoul',weekday:'short'}).format(new Date(iso));

export function contestKind(platform,id,title='') {
  if (platform==='atcoder') { const m=/^(abc|arc|agc|ahc)\d+/i.exec(id); return m ? m[1].toLowerCase() : 'other'; }
  if (/div\.?\s*1\s*\+\s*(div\.?\s*)?2/i.test(title)) return 'div1+2';
  const div=/div\.?\s*([1-4])/i.exec(title);
  if (div) return 'div'+div[1];
  if (/educational/i.test(title)) return 'educational';
  if (/global/i.test(title)) return 'global';
  return 'other';
}
// Conservative fit: uses ability mean minus one SD so sparse history proposes easier contests.
export function fittingKinds(abilities) {
  const cf=abilities.codeforces ? abilities.codeforces.mean-abilities.codeforces.sd : -Infinity;
  const atc=abilities.atcoder ? abilities.atcoder.mean-abilities.atcoder.sd : -Infinity;
  const kinds=new Set(['abc','div3','div4']);
  if (cf>=1400) ['div2','educational','div1+2'].forEach(k=>kinds.add(k));
  if (cf>=2100) ['div1','global'].forEach(k=>kinds.add(k));
  if (atc>=1200) kinds.add('arc');
  if (atc>=2000) kinds.add('agc');
  return kinds;
}
export function inWeekendWindow(startAt,window) {
  const day=seoulWeekday(startAt), clock=seoulClock(startAt);
  const w=day==='Sat' ? window.sat : day==='Sun' ? window.sun : null;
  return !!w && clock>=w[0] && clock<w[1];
}

export function liveProposals({calendar=[],abilities,mastery,from,to}) {
  const kinds=fittingKinds(abilities);
  return calendar.filter(c=>c.start_at && mastery.contest_platforms.includes(c.platform) && seoulDate(new Date(c.start_at))>=from && seoulDate(new Date(c.start_at))<=to)
    .map(c=>{ const kind=contestKind(c.platform,c.id.replace(/^(codeforces|atcoder):/,''),c.title);
      const window=inWeekendWindow(c.start_at,mastery.weekend_window), fit=kinds.has(kind);
      return {contest_id:c.id,platform:c.platform,title:c.title,url:c.url,start_at:c.start_at,duration_minutes:c.duration_minutes,kind,in_weekend_window:window,fits_level:fit,
        proposed:window && fit,reason:`${kind} ${fit?'fits':'does not fit'} the conservative level estimate; ${window?'inside':'outside'} the weekend window (Asia/Seoul).`,
        acceptance:'Proposal only: mark it suitable to schedule it.'}; })
    .sort((a,b)=>Number(b.proposed)-Number(a.proposed) || a.start_at.localeCompare(b.start_at) || a.contest_id.localeCompare(b.contest_id));
}

export function virtualProposal({catalog,attempts,abilities,mastery,date,weakConcepts=[],limit=3}) {
  const touched=new Set(attempts.map(a=>canonicalKey(catalog,a.problem_key)));
  const kinds=fittingKinds(abilities);
  const meta=new Map(catalog.contests.map(c=>[c.contest_id,c]));
  const weak=new Set(weakConcepts);
  const rows=[];
  for (const [contestId,problems] of catalog.contestProblems) {
    const m=meta.get(contestId);
    const platform=m?.platform ?? catalog.problems.get(problems[0]?.problem_key)?.platform;
    if (!platform || !mastery.contest_platforms.includes(platform) || problems.length<3) continue;
    const kind=m?.kind ?? contestKind(platform,contestId,m?.title ?? '');
    if (!kinds.has(kind)) continue;
    const startDate=m?.start_at ? seoulDate(new Date(m.start_at)) : null;
    if (startDate && startDate>addDays(date,-14)) continue;
    const keys=problems.map(p=>canonicalKey(catalog,p.problem_key));
    if (keys.some(k=>touched.has(k))) continue;
    const predictions=keys.map(k=>predict(catalog.problems.get(k),abilities));
    const known=predictions.filter(p=>p.p_mean!==null);
    const expected=known.reduce((s,p)=>s+p.p_mean,0);
    const conceptSet=new Set(keys.flatMap(k=>[...trustedConcepts(catalog.problems.get(k))]));
    const coverage=!weak.size ? 0 : conceptSet.size ? [...weak].filter(c=>conceptSet.has(c)).length/weak.size : null;
    const fit=known.length ? Math.max(0,1-Math.max(0,1.5-expected,expected-4)/2) : 0;
    const recency=startDate ? Math.max(0,1-(Date.parse(date)-Date.parse(startDate))/(1095*DAY)) : 0;
    const components={weakness_coverage:coverage===null?0:Math.round(2*coverage*1000)/1000,expected_solved_fit:Math.round(fit*1000)/1000,recency:Math.round(0.5*recency*1000)/1000};
    rows.push({contest_id:contestId,platform,kind,title:m?.title ?? contestId,url:m?.url ?? null,start_at:m?.start_at ?? null,problems:keys,
      expected_solved:Math.round(expected*100)/100,problems_without_prediction:keys.length-known.length,
      weakness_coverage:coverage===null?'unknown':Math.round(coverage*1000)/1000,recency_known:!!startDate,
      score:Math.round(Object.values(components).reduce((a,b)=>a+b,0)*1000)/1000,components});
  }
  rows.sort((a,b)=>b.score-a.score || a.contest_id.localeCompare(b.contest_id));
  return {policy:CONTESTS_VERSION,date,candidates:rows.slice(0,limit),eligible_count:rows.length,
    note:'Run the chosen virtual contest on AtCoder Problems or Codeforces; this system only selects and records. Unknown weakness coverage contributes 0 and is reported as unknown.'};
}
