// Git evidence convention git-convention-v1 (design.md section 3). Pure functions.
// Backward compatible: every trailer is optional. Without trailers a commit can only
// produce reported_completion, never independent evidence.
import {keyPattern} from '../core/events.mjs';

export const CONVENTION_VERSION = 'git-convention-v1';
const PLATFORM_DIRS = {leetcode:'leetcode',codeforces:'codeforces',atcoder:'atcoder',baekjoon:'baekjoon',boj:'baekjoon'};
const KNOWN_TRAILERS = ['problem','outcome','hints','minutes','activity','transfer-from'];

// "Key: value" lines from `%(trailers:unfold,only)`. Repeated Problem trailers are kept.
export function parseTrailers(block) {
  const trailers={}, problems=[], unknown=[];
  for (const line of String(block ?? '').split('\n')) {
    const match=/^([A-Za-z][A-Za-z-]*)\s*:\s*(.+?)\s*$/.exec(line);
    if (!match) continue;
    const key=match[1].toLowerCase(), value=match[2];
    if (!KNOWN_TRAILERS.includes(key)) { unknown.push(match[1]); continue; }
    if (key==='problem') problems.push(value); else trailers[key]=value;
  }
  return {trailers,problems,unknown};
}

const stem = name => name.replace(/\.[A-Za-z0-9]+$/,'');
// Same skip words as the legacy note parser (src/main/generate_index_map.py), plus housekeeping files.
const NON_PROBLEM = /^(readme|index|notes?|template|license|overview|intro|algorithm|\.gitkeep)$/i;
// Maps one repository path to a platform-qualified key, or null.
export function keyFromPath(path) {
  if (/^(?:daily|leetcode|atcoder|codeforces)\/\d{2}-\d{2}-\d{2}\.md$/.test(String(path))) return null;
  const parts=String(path).split('/').filter(Boolean);
  for (let i=0;i<parts.length-1;i++) {
    const platform=PLATFORM_DIRS[parts[i].toLowerCase()];
    if (!platform) continue;
    const raw=stem(parts[i+1]);
    if (NON_PROBLEM.test(raw)) return null;
    let id=null;
    if (platform==='leetcode') id=raw.toLowerCase().replace(/^\d+[-_.]/,'').replace(/_/g,'-');
    if (platform==='codeforces') { const m=/^(?:cf[-_]?)?(\d+)[-_]?([A-Za-z]\d?)$/i.exec(raw); if (m) id=m[1]+m[2].toUpperCase(); }
    if (platform==='atcoder') { const m=/^[a-z0-9_-]+_[a-z0-9]+$/i.exec(raw); if (m) id=raw.toLowerCase(); }
    if (platform==='baekjoon') { const m=/^(?:bj[-_]?|boj[-_]?)?(\d+)\b/i.exec(raw); if (m) id=m[1]; }
    if (!id) return null;
    const key=`${platform}:${id}`;
    return keyPattern.test(key) ? key : null;
  }
  return null;
}
// Legacy study notes: front-matter `id: bj-10986` under a baekjoon directory.
export function frontMatterKey(path,text) {
  if (!/\.(md|mdx)$/i.test(path) || !path.split('/').some(p=>['baekjoon','boj'].includes(p.toLowerCase()))) return null;
  const block=/^---\r?\n([\s\S]*?)\r?\n---/.exec(String(text ?? ''));
  if (!block) return null;
  const line=block[1].split(/\r?\n/).find(l=>/^id\s*:/.test(l));
  const m=line && /^id\s*:\s*['"]?(?:bj[-_]?|boj[-_]?)?(\d+)['"]?\s*$/i.exec(line);
  return m ? `baekjoon:${m[1]}` : null;
}

// Classifies trailers into ledger outcome fields plus data-quality flags.
export function classify(trailers) {
  const flags=[];
  const hintsRaw=trailers.hints?.toLowerCase();
  const hints=hintsRaw===undefined ? null : ['none','no','0','false'].includes(hintsRaw) ? false : ['used','yes','true'].includes(hintsRaw) || /^\d+$/.test(hintsRaw) ? true : (flags.push('unknown_hints_value'),null);
  const raw=trailers.outcome?.toLowerCase();
  let outcome='reported_completion';
  if (raw===undefined) flags.push('no_outcome_trailer');
  else if (['independent','solved-independently','self'].includes(raw)) {
    if (hints===false) outcome='independent';
    else if (hints===true) outcome='assisted';
    else flags.push('independent_without_hints_trailer');
  } else if (['assisted','hinted','editorial'].includes(raw)) outcome='assisted';
  else if (['failed','fail','wa','tle','gave-up'].includes(raw)) outcome='failed';
  else if (['accepted','ac','solved','done'].includes(raw)) outcome='reported_completion';
  else flags.push('unknown_outcome_value');
  let selfMinutes=null;
  if (trailers.minutes!==undefined) { const m=/^(\d{1,4})\s*(m|min|mins|minutes)?$/i.exec(trailers.minutes); if (m && Number(m[1])<=1440) selfMinutes=Number(m[1]); else flags.push('invalid_minutes'); }
  const activity=trailers.activity?.toLowerCase();
  let activityType=null, transferFrom=null;
  if (activity!==undefined) { if (['practice','review','transfer'].includes(activity)) activityType=activity; else flags.push('unknown_activity'); }
  if (activityType==='transfer') {
    transferFrom=trailers['transfer-from'] ?? null;
    if (!transferFrom || !keyPattern.test(transferFrom)) { activityType=null; transferFrom=null; flags.push('transfer_without_source'); }
  }
  return {outcome,hints:outcome==='independent'?false:hints,self_minutes:selfMinutes,activity_type:activityType,transfer_from:transferFrom,flags};
}
