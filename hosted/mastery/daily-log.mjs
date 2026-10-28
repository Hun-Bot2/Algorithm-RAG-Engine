// Pure daily-log parser. Resolve only exact catalog identities; never slugify titles.
export const dailyPath = path => /^(daily|leetcode|atcoder|codeforces)\/(\d{2}-\d{2}-\d{2})\.md$/.exec(path);
const normalized = title => title.normalize('NFC').trim().replace(/\s+/g,' ').toLowerCase();
const urlId = input => {
  try {
    const u=new URL(input), host=u.hostname.toLowerCase(), p=u.pathname;
    if (!['http:','https:'].includes(u.protocol) || u.username || u.password) return null;
    let m;
    if (['leetcode.com','www.leetcode.com'].includes(host) && (m=/^\/problems\/([a-z0-9-]+)(?:\/|$)/i.exec(p))) return `leetcode:${m[1].toLowerCase()}`;
    if (['codeforces.com','www.codeforces.com'].includes(host) && (m=/^\/(?:problemset\/problem\/(\d+)\/|contest\/(\d+)\/problem\/)([a-z]\d?)(?:\/|$)/i.exec(p))) return `codeforces:${m[1] ?? m[2]}${m[3].toUpperCase()}`;
    if (['atcoder.jp','www.atcoder.jp'].includes(host) && (m=/^\/contests\/[^/]+\/tasks\/([a-z0-9_]+)(?:\/|$)/i.exec(p))) return `atcoder:${m[1].toLowerCase()}`;
  } catch {}
  return null;
};
export function parseDailyLog(path,source,catalog) {
  const match=dailyPath(path), rows=[], flags=[], review_targets=[];
  if (!match) return {rows,review_targets,flags};
  const date=`20${match[2]}`, time=Date.parse(`${date}T00:00:00Z`);
  if (!Number.isFinite(time) || new Date(time).toISOString().slice(0,10)!==date) return {rows,review_targets,flags:['daily_invalid_date']};
  const lines=String(source).split(/\r?\n/);
  if (lines[0]!==`# ${date}`) return {rows,review_targets,flags:['daily_title_mismatch']};
  // Only the summary table before the first detailed section; fenced examples are inert.
  let fence=null, header=-1;
  for (let i=1;i<lines.length;i++) {
    const f=/^\s*(`{3,}|~{3,})/.exec(lines[i]);
    if(f){if(!fence)fence=f[1][0];else if(fence===f[1][0])fence=null;continue;}
    if(fence)continue;
    if(/^#{1,6}\s/.test(lines[i]))break;
    if(/^\|\s*문제\s*\|\s*결과\s*\|(?:\s*시간\s*\|)?(?:\s*복습\s*\|)?\s*메모\s*\|\s*$/.test(lines[i])){header=i;break;}
  }
  const columns=header<0?[]:lines[header].split('|').slice(1,-1).map(x=>x.trim());
  const timeIndex=columns.indexOf('시간'),reviewIndex=columns.indexOf('복습');
  const separators=(lines[header+1] ?? '').trim().split('|').slice(1,-1);
  if(header<0 || separators.length!==columns.length || !separators.every(x=>/^\s*:?-{3,}:?\s*$/.test(x)))return {rows,review_targets,flags:['daily_invalid_table']};
  const platform=match[1]==='daily'?null:match[1];
  const canonical=key=>{const visited=new Set();while(catalog?.aliases?.has(key)){if(visited.has(key))return null;visited.add(key);key=catalog.aliases.get(key);}return catalog?.problems.has(key)?key:null;};
  const titles=new Map();
  for(const p of catalog?.problems.values() ?? []) {
    if(platform && p.platform!==platform)continue;
    const title=normalized(p.title),key=canonical(p.key);
    if(!titles.has(title))titles.set(title,new Set());
    if(key)titles.get(title).add(key);
  }
  const seen=new Map();
  for(let i=header+2;i<lines.length && /^(?:\||\[\|)/.test(lines[i].trimStart());i++) {
    const at=`line_${i+1}`;
    const parts=lines[i].trim().replace(/^\[\|/,'|[').split('|').slice(1,-1).map(x=>x.trim());
    if(parts.length!==columns.length || !parts[0] || !/^(1|0|O|X|-|)$/.test(parts[1])){flags.push(`daily_invalid_row:${at}`);continue;}
    const raw=parts[0],outcome=parts[1],duration=timeIndex<0?'':parts[timeIndex];
    if(reviewIndex>=0 && !['','1','0','O','-'].includes(parts[reviewIndex])){flags.push(`daily_invalid_review:${at}`);continue;}
    if(duration!=='' && duration!=='-' && (!/^\d{1,5}[ms]$/.test(duration) || parseInt(duration,10)>(duration.endsWith('s')?86400:1440))){flags.push(`daily_invalid_minutes:${at}`);continue;}
    const seconds=duration.endsWith('s')?parseInt(duration,10):null;
    const minutes=duration.endsWith('m')?parseInt(duration,10):null;
    const link=/^\[[^\]]+\]\(([^\s)]+)\)$/.exec(raw);
    const target=link?.[1] ?? raw.replace(/^<(https?:\/\/[^>]+)>$/,'$1');
    let key=null;
    if(link || /^https?:\/\//i.test(target))key=canonical(urlId(target));
    else {
      const candidates=titles.get(normalized(raw));
      if(candidates?.size===1)key=[...candidates][0];
      else if(candidates?.size>1){flags.push(`daily_ambiguous_problem:${at}`);continue;}
    }
    if(!key || (platform && catalog.problems.get(key).platform!==platform)){flags.push(`daily_unresolved_problem:${at}`);continue;}
    if(seen.has(key)) {
      if(seen.get(key)!==outcome){flags.push(`daily_conflicting_rows:${key}`);for(let j=rows.length-1;j>=0;j--)if(rows[j].metadata.problem_key===key)rows.splice(j,1);seen.set(key,'conflict');for(let j=review_targets.length-1;j>=0;j--)if(review_targets[j].problem_key===key)review_targets.splice(j,1);}
      continue;
    }
    seen.set(key,outcome);
    const memo=parts.at(-1);
    if(reviewIndex>=0?['1','O'].includes(parts[reviewIndex]):/(?:^|\s)#복습(?=$|\s|[,.;!])/.test(memo))review_targets.push({problem_key:key,date});
    if(outcome==='-' || outcome===''){flags.push(`daily_outcome_unrecorded:${at}`);continue;}
    rows.push({date,line:i+1,raw:lines[i],metadata:{problem_key:key,attempt:`daily-${date}`,outcome:outcome==='1'?'reported_completion':outcome==='O'?'independent':'failed',hints:outcome==='O'?false:null,self_minutes:minutes,...(seconds!==null?{self_seconds:seconds}:{}),activity_type:'practice',transfer_from:null}});
  }
  return {rows,review_targets,flags};
}
