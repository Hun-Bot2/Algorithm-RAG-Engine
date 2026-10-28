/** Public schedule metadata only. No learner history, credentials or problem statements. */
import {createRequestGate} from './providers.mjs';
export const DISCOVERY_SOURCES=Object.freeze({
 codeforces:'https://codeforces.com/api/contest.list',
 atcoder:'https://atcoder.jp/contests/?lang=en',
 leetcode:'https://raw.githubusercontent.com/MaskedCarrot/contestdeck-data/main/data/contests.json',
});
export const CALENDAR_TTL=6*3600000;
const request=createRequestGate();
const explicitTime=v=>typeof v==='string'&&/(Z|[+-]\d{2}:?\d{2})$/.test(v)&&Number.isFinite(Date.parse(v));
const decode=s=>s.replace(/<[^>]*>/g,'').replace(/&(?:amp|lt|gt|quot|#39);/g,e=>({'&amp;':'&','&lt;':'<','&gt;':'>','&quot;':'"','&#39;':"'"}[e])).trim();
export function officialContestUrl(platform,value){
 try{const u=new URL(value),hosts={codeforces:['codeforces.com'],atcoder:['atcoder.jp'],leetcode:['leetcode.com','www.leetcode.com']};
  const path=platform==='codeforces'?/^\/contest\/(\d+)\/?$/:platform==='atcoder'?/^\/contests\/([a-zA-Z0-9_-]+)\/?$/:/^\/contest\/([a-zA-Z0-9_-]+)\/?$/;
  const m=path.exec(u.pathname);if(u.protocol!=='https:'||u.username||u.password||u.port||!hosts[platform]?.includes(u.hostname)||!m||u.search||u.hash)return null;
  return {id:`${platform}:${m[1]}`,url:u.href};
 }catch{return null;}
}
export function normalizeSchedule(platform,text,{now=new Date()}={}){
 let rows=[],sourceUpdated=null;
 if(platform==='codeforces'){
  const raw=JSON.parse(text);if(raw.status!=='OK'||!Array.isArray(raw.result)||raw.result.length>20000)throw Error('invalid_calendar');
  rows=raw.result.filter(r=>['BEFORE','CODING'].includes(r.phase)).map(r=>({title:r.name,url:`https://codeforces.com/contest/${r.id}`,start_at:Number.isFinite(r.startTimeSeconds)?new Date(r.startTimeSeconds*1000).toISOString():null,duration_minutes:r.durationSeconds/60}));
 }else if(platform==='atcoder'){
  const section=/<div\b[^>]*id=["']contest-table-upcoming["'][^>]*>([\s\S]*?)<\/tbody>/.exec(text);
  if(!section)throw Error('invalid_calendar');
  const tags=[...section[1].matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/g)].map(m=>m[1]).filter(r=>/<td\b/.test(r));
  rows=tags.map(row=>{
   const cells=[...row.matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/g)].map(m=>m[1]);
   const time=/<time\b[^>]*>([^<]+)<\/time>/.exec(cells[0]??'')?.[1];
   const link=/<a\b[^>]*href=["'](\/contests\/[a-zA-Z0-9_-]+)["'][^>]*>([\s\S]*?)<\/a>/.exec(cells[1]??'');
   const duration=/^(\d{1,4}):(\d{2})$/.exec(decode(cells[2]??''));
   if(!time||!link||!duration||Number(duration[2])>=60)throw Error('invalid_contest');
   return {title:decode(link[2]),url:'https://atcoder.jp'+link[1],start_at:time.replace(' ','T').replace(/([+-]\d{2})(\d{2})$/,'$1:$2'),duration_minutes:Number(duration[1])*60+Number(duration[2]),rated_range:decode(cells[3]??'')||null};
  });
 }else if(platform==='leetcode'){
  const raw=JSON.parse(text);if(raw.schemaVersion!==1||!Array.isArray(raw.contests)||raw.contests.length>20000||!explicitTime(raw.updatedAt))throw Error('invalid_calendar');
  sourceUpdated=new Date(raw.updatedAt).toISOString();
  if(Date.parse(sourceUpdated)>now.getTime()+300000)throw Error('future_source_timestamp');
  rows=raw.contests.filter(r=>r.platform==='leetcode').map(r=>({title:r.name,url:r.url,start_at:r.startsAt,duration_minutes:explicitTime(r.startsAt)&&explicitTime(r.endsAt)?(Date.parse(r.endsAt)-Date.parse(r.startsAt))/60000:NaN}));
 }else throw Error('unsupported_platform');
 const unique=new Map();
 for(const r of rows){
  const official=officialContestUrl(platform,r.url);
  if(!official||typeof r.title!=='string'||!r.title.trim()||!explicitTime(r.start_at)||!Number.isFinite(r.duration_minutes)||r.duration_minutes<=0||r.duration_minutes>44640)throw Error('invalid_contest');
  const item={...r,...official,platform,title:r.title.slice(0,240),start_at:new Date(r.start_at).toISOString(),source:DISCOVERY_SOURCES[platform],source_kind:platform==='leetcode'?'community_clist_mirror':'official',source_updated_at:sourceUpdated,fetched_at:now.toISOString(),suitable:false,eligibility:{status:'unknown',reason:'참가·rated 조건은 공식 공지에서 확인해 주세요. Rated range는 참가 자격이나 문제 난이도가 아니에요.'}};
  if(Date.parse(item.start_at)+item.duration_minutes*60000<now.getTime())continue;
  const prior=unique.get(item.id);if(prior&&JSON.stringify(prior)!==JSON.stringify(item))throw Error('conflicting_duplicate');
  unique.set(item.id,item);
 }
 return {items:[...unique.values()].sort((a,b)=>a.start_at.localeCompare(b.start_at)||a.id.localeCompare(b.id)).slice(0,200),source_updated_at:sourceUpdated};
}
export function scheduleStatus(envelope,now=new Date()){
 if(!envelope?.fetched_at)return 'unavailable';
 const times=[envelope.fetched_at,...(envelope.source_updated_at?[envelope.source_updated_at]:[])];
 return envelope.status==='fresh'&&times.every(t=>Number.isFinite(Date.parse(t))&&now-Date.parse(t)>=0&&now-Date.parse(t)<CALENDAR_TTL)?'fresh':'stale';
}
export async function fetchDiscovery(platform,{fetcher=request,previous=null,now=new Date()}={}){
 const source=DISCOVERY_SOURCES[platform];if(!source)throw Error('unsupported_platform');
 const age=now-Date.parse(previous?.last_attempt_at);
 if(Number.isFinite(age)&&age>=0&&age<300000)return {...previous,status:scheduleStatus(previous,now)};
 if(scheduleStatus(previous,now)==='fresh')return previous;
 try{
  const response=await fetcher(source,{signal:AbortSignal.timeout(10000)});if(!response.ok)throw Error(`http_${response.status}`);
  const text=await response.text();if(text.length>4000000)throw Error('response_too_large');
  const normalized=normalizeSchedule(platform,text,{now});
  const result={...normalized,source,fetched_at:now.toISOString(),last_attempt_at:now.toISOString(),status:'fresh',error:null};
  result.status=scheduleStatus(result,now);if(result.status==='stale')result.error='upstream_snapshot_stale';return result;
 }catch(error){return {source,items:previous?.items??[],source_updated_at:previous?.source_updated_at??null,fetched_at:previous?.fetched_at??null,last_attempt_at:now.toISOString(),status:previous?.fetched_at?'stale':'unavailable',error:/^(http_\d+|invalid_calendar|invalid_contest|response_too_large|future_source_timestamp|conflicting_duplicate)$/.test(error.message)?error.message:'provider_failed'};}
}
