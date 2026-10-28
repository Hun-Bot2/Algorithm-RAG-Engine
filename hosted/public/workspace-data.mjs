/** Presentation-only projections. Never infer an attempt from a plan or a timer. */
export const TIME_ZONE = 'Asia/Seoul';
export function localDate(value, timeZone = TIME_ZONE) {
  if(value===null||value===undefined||value==='')return null;
  if(typeof value==='string' && /^\d{4}-\d{2}-\d{2}$/.test(value)){const civil=new Date(value+'T12:00:00Z');return Number.isFinite(civil.getTime())&&civil.toISOString().slice(0,10)===value?value:null;}
  if(typeof value==='string' && !/(Z|[+-]\d{2}:\d{2})$/i.test(value))return null;
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return null;
  const parts = new Intl.DateTimeFormat('en-US', {timeZone,year:'numeric',month:'2-digit',day:'2-digit'}).formatToParts(date);
  return ['year','month','day'].map(k=>parts.find(p=>p.type===k).value).join('-');
}
export function shiftDate(date, days) {
  const d = new Date(`${date}T12:00:00Z`); d.setUTCDate(d.getUTCDate()+days); return d.toISOString().slice(0,10);
}
export function calendarDates(date, mode='month') {
  const first = mode==='week' ? date : `${date.slice(0,7)}-01`;
  const offset = (new Date(`${first}T12:00:00Z`).getUTCDay()+6)%7;
  const start=shiftDate(first,-offset);
  const last = new Date(`${first.slice(0,7)}-01T12:00:00Z`); last.setUTCMonth(last.getUTCMonth()+1); last.setUTCDate(0);
  const count=mode==='week'?7:Math.ceil((offset+last.getUTCDate())/7)*7;
  return Array.from({length:count},(_,i)=>shiftDate(start,i));
}
export function navigateDate(date, amount, mode='month') {
  if(mode==='week')return shiftDate(date,amount*7);
  const d=new Date(`${date.slice(0,7)}-01T12:00:00Z`); d.setUTCMonth(d.getUTCMonth()+amount);return d.toISOString().slice(0,10);
}
export function completion(entry) {
  return ['independent','assisted','reported_completion'].includes(entry.outcome);
}
export function outcomeText(entry) {
  if(entry.status==='superseded')return '수정 전';
  // Policy v3 (ADR 005): help is not tracked; every completion is shown as solved.
  return completion(entry)?'해결':entry.outcome==='failed'?'미해결':'결과 미상';
}
export function outcomeSymbol(entry) {
  return completion(entry)?'✓':'·';
}
export function workspaceEvents(journal, state, {today=localDate(new Date())}={}) {
  const events=[],seen=new Set();
  const add=e=>{if(e.date && !seen.has(e.id)){seen.add(e.id);events.push(e);}};
  for(const day of journal?.days || [])for(const entry of day.entries || []){
    if(!['attempt','recommendation'].includes(entry.type))continue;
    if(entry.type==='recommendation'&&entry.kind==='review')continue;
    const date=entry.type==='attempt'?localDate(entry.occurred_at):entry.local_date || day.date;
    add({...entry,date,layer:entry.type==='attempt'?'actual':'plan',id:entry.id || `${entry.type}:${date}:${entry.problem_key}`});
  }
  // Worker-only runtime has core events but no catalog-backed journal.
  if(!journal?.available) {
    const replaced=new Set((state?.events || []).map(e=>e.supersedes).filter(Boolean));
    for(const e of state?.events || [])if(e.kind==='attempt'){
      const p=(state.curriculum?.problems || []).find(p=>p.problem_key===e.payload.problem_key);
      add({...e.payload,id:e.event_id,type:'attempt',layer:'actual',date:localDate(e.occurred_at),occurred_at:e.occurred_at,recorded_at:e.recorded_at,status:replaced.has(e.event_id)?'superseded':undefined,title:p?.title || e.payload.problem_key,platform:p?.platform || e.payload.problem_key.split(':')[0],difficulty:p?.native_difficulty});
    }
  }
  for(const [i,slot] of (journal?.plan?.slots || []).entries())if(slot.kind!=='review')add({...slot,id:`current:${i}:${slot.problem_key}`,type:'recommendation',layer:'plan',snapshot_id:journal.plan.snapshot_id,date:journal.plan.date || journal.date,current:true});
  // The current snapshot is preferred when its revisions are grouped below.
  const filtered=events;
  if(!journal?.available)for(const [i,s] of (state?.mission?.activities || []).entries())if(s.type!=='review')filtered.push({...s,id:`mission:${i}`,title:s.title || s.problem_key,type:'recommendation',layer:'plan',date:state.mission.date || localDate(new Date()),current:true});
  for(const target of journal?.review_selections ?? state?.preferences?.review_targets ?? []){
    const problem=(state?.curriculum?.problems || []).find(p=>p.problem_key===target.problem_key);
    filtered.push({...target,id:`manual-review:${target.date}:${target.problem_key}`,title:target.title || problem?.title || target.problem_key,url:target.url || problem?.canonical_url || null,
      platform:target.platform || target.problem_key.split(':')[0],type:'recommendation',layer:'plan',kind:'review',explicit_review:true,selection_source:'manual',reason:'사용자가 직접 복습 대상으로 지정한 문제예요.'});
  }
  // Automatic review queues are not explicit learner labels. Do not display them as scheduled work.
  // Mock contests (specs/mock-contest R5.2, R6.2): date and solving time only, never a clock time (v1 `start_at` is
  // dropped). Results are matched for display only and never stored as evidence.
  const solved=events.filter(e=>e.layer==='actual'&&e.status!=='superseded');
  for(const m of journal?.mock_contests ?? []){
    const problems=(m.problems ?? []).map(p=>{
      const hit=solved.filter(a=>a.problem_key===p.problem_key&&a.date>=m.local_date).sort((a,b)=>String(a.occurred_at).localeCompare(String(b.occurred_at))||a.id.localeCompare(b.id))[0];
      return {...p,result:hit?{attempt_id:hit.id,date:hit.date,outcome:hit.outcome,symbol:outcomeSymbol(hit),text:outcomeText(hit)}:null};
    });
    filtered.push({id:`${m.archived?'mock-archive':'mock'}:${m.local_date}`,type:'mock_contest',layer:'mock',date:m.local_date,archived:Boolean(m.archived),platform:'leetcode',title:`LeetCode 모의 대회 · ${problems.length}문제`,
      duration_minutes:m.duration_minutes,policy:m.policy,composition:m.composition,expected_solved:m.expected_solved,target_expected:m.target_expected,
      evidence_label:m.evidence_label ?? null,notes:m.notes ?? [],problems,attempted:problems.filter(p=>p.result).length});
  }
  for(const c of state?.contests || []){const date=localDate(c.start_at);if(date)filtered.push({...c,id:`contest:${c.id}`,contest_id:c.id,title:c.title,type:'contest',layer:'contest',date});}
  // Snapshot revisions are provenance, not separate tasks on the same date.
  const plans=new Map(),projected=[];
  for(const event of filtered){
    if(event.layer!=='plan'||!event.problem_key){projected.push(event);continue;}
    const key=`${event.date}:${event.problem_key}`;
    if(!plans.has(key))plans.set(key,[]);plans.get(key).push(event);
  }
  for(const versions of plans.values()){
    versions.sort((a,b)=>Number(Boolean(b.explicit_review))-Number(Boolean(a.explicit_review))||Number(Boolean(b.current))-Number(Boolean(a.current))||String(b.occurred_at || '').localeCompare(String(a.occurred_at || ''))||b.id.localeCompare(a.id));
    const revisions=[...new Map(versions.map(v=>[v.snapshot_id || v.id,v]).reverse()).values()].reverse();
    projected.push({...versions[0],...(revisions.length>1?{versions:revisions}:{} )});
  }
  // Recommendation provenance is independent of outcome, and manual review labels are not recommendations.
  const recommendations=filtered.filter(e=>e.layer==='plan'&&!e.explicit_review&&e.kind!=='review'&&e.problem_key);
  for(const event of projected){
    const prior=recommendations.filter(r=>r.problem_key===event.problem_key&&r.date<=event.date&&
      (event.layer!=='actual'||!r.occurred_at||!event.occurred_at||Date.parse(r.occurred_at)<=Date.parse(event.occurred_at)));
    event.recommended=event.layer==='plan'?!event.explicit_review&&event.kind!=='review':event.layer==='actual'&&prior.length>0;
    if(event.recommended)event.recommended_dates=[...new Set(prior.map(r=>r.date))].sort();
  }
  carryOver(projected,today);
  const order={actual:0,plan:1,mock:2,contest:3};
  return projected.sort((a,b)=>a.date.localeCompare(b.date)||order[a.layer]-order[b.layer]||String(a.occurred_at || a.start_at || '').localeCompare(String(b.occurred_at || b.start_at || '')));
}
/** Unsolved system plans roll forward to today for at most CARRY_DAYS days. */
export const CARRY_DAYS=7;
const carriable=e=>e.layer==='plan'&&e.problem_key&&!e.explicit_review&&e.kind!=='review'&&!e.archived;
// Presentation only: snapshots stay immutable and the planner never sees carried items.
// Any attempt on or after the plan date settles the plan; failures go to the review policy instead.
function carryOver(events,today){
  const attempts=activeAttempts(events);
  for(const e of events)if(e.layer==='plan'&&e.problem_key){
    const after=attempts.filter(a=>a.problem_key===e.problem_key&&a.date>=e.date);
    e.progress=after.some(completion)?'done':after.length?'tried':'pending';
  }
  if(!today)return;
  const first=new Map();
  for(const e of events)if(carriable(e)&&e.progress==='pending'&&e.date<today&&e.date>=shiftDate(today,-CARRY_DAYS)){
    e.carried_to=today;
    if(!first.has(e.problem_key)||e.date<first.get(e.problem_key).date)first.set(e.problem_key,e);
  }
  for(const [key,origin] of first){
    const existing=events.find(e=>e.layer==='plan'&&e.date===today&&e.problem_key===key);
    if(existing){if(carriable(existing))existing.carried_from=origin.date;continue;}
    const {versions,current,carried_to,...rest}=origin;
    events.push({...rest,id:`carry:${today}:${key}`,date:today,carry:true,carried_from:origin.date,origin_id:origin.id});
  }
}
/** Plans still to work on for a day: not settled and not moved to a later day. */
export function pendingPlans(events) {return events.filter(e=>e.layer==='plan'&&e.progress==='pending'&&!e.carried_to);}
export function activeAttempts(events) {return events.filter(e=>e.layer==='actual' && e.status!=='superseded');}
export function filterAttempts(events, filters={}) {
  const q=(filters.query || '').toLowerCase();
  return activeAttempts(events).filter(e=>(!filters.from||e.date>=filters.from)&&(!filters.to||e.date<=filters.to)&&(!filters.platform||e.platform===filters.platform)&&(!filters.outcome||e.outcome===filters.outcome)&&(!filters.difficulty||String(e.difficulty)===filters.difficulty)&&(!q||[e.title,e.problem_key,e.pattern,e.platform].filter(Boolean).join(' ').toLowerCase().includes(q)));
}
export function groupCounts(rows, key) {
  const groups=new Map();for(const row of rows){const value=key(row);groups.set(value,(groups.get(value)||[]).concat(row));}return [...groups].sort((a,b)=>b[1].length-a[1].length);
}
export function conversion(events) {
  // Only identifiable dated problem plans; repeated snapshots count once per problem/day.
  const plans=new Map(events.filter(e=>e.layer==='plan'&&e.problem_key&&!e.carry).map(e=>[`${e.date}:${e.problem_key}`,e]));
  const attempts=activeAttempts(events);
  const rows=[...plans.values()].map(p=>({...p,matched:attempts.some(a=>a.problem_key===p.problem_key&&a.date>=p.date&&a.date<=shiftDate(p.date,7))}));
  return {rows,matched:rows.filter(r=>r.matched).length,total:rows.length};
}

/** Disjoint categories: attempted means an attempt without recorded completion. */
export function eventCategory(event) {
  if(event.layer==='actual')return completion(event)?'solved':'attempted';
  return event.layer==='plan'?'planned':event.layer==='contest'?'contest':event.layer==='mock'?'mock':null;
}
export function filterCalendarEvents(events, enabled={solved:true,attempted:true,planned:true,contest:true}) {
  return events.filter(e=>e.status!=='superseded' && enabled[eventCategory(e)]===true);
}
export function daySummary(events) {
  const rows=activeAttempts(events);
  return {attempted:rows.length,solved:rows.filter(completion).length,unsolved:rows.filter(e=>e.outcome==='failed').length,
    unknown:rows.filter(e=>!completion(e)&&e.outcome!=='failed').length};
}
export function contestSuitability(contest,events=[],participation=[]) {
  if(contest.recommendation)return contest.recommendation;
  const id=contest.contest_id || contest.id?.replace(/^contest:/,'');
  const superseded=new Set(participation.map(p=>p.supersedes).filter(Boolean));
  const records=participation.filter(p=>!superseded.has(p.entry_id)&&p.contest_id===id).sort((a,b)=>String(b.occurred_at).localeCompare(String(a.occurred_at)));
  const attempts=activeAttempts(events).filter(e=>e.platform===contest.platform);
  const evidence={platform_attempts:attempts.length,independent:attempts.filter(e=>e.outcome==='independent'&&e.hints===false).length,participation_ids:records.map(p=>p.entry_id)};
  const prior=records[0];
  if(prior?.outcome==='completed')return {status:'recommended',label:'추천 · 재참여/복습',reason:'같은 대회의 완료 참여 기록이 있어 복습 대상으로 볼 수 있어요. 새 대회의 난이도 적합성이나 실력 인증은 아니에요.',evidence};
  if(['partial','abandoned'].includes(prior?.outcome))return {status:'stretch',label:'도전 · 다시 시도',reason:'같은 대회에서 일부 진행 또는 중단한 기록이 있어요. 재도전 관점의 표시이며 대회 제목으로 실력을 추정하지 않아요.',evidence};
  if(contest.source==='manual'&&contest.suitable===true)return {status:'recommended',label:'추천 · 직접 선택',reason:'학습 목표에 맞는 일정으로 직접 선택했어요. 개인 실력과 요구 난이도의 비교 근거는 아직 없어요.',evidence};
  return {status:'insufficient',label:'판단 근거 부족',reason:'대회별 요구 난이도와 대응하는 학습 근거가 부족해요. 플랫폼 풀이 수나 대회 제목만으로 추천·도전을 정하지 않아요.',evidence};
}
export function contestList(events,{platform='',scope='upcoming',now=new Date()}={}) {
  return events.filter(e=>e.layer==='contest'&&(!platform||e.platform===platform)&&
    (scope==='all'||Date.parse(e.start_at)+(Number.isFinite(e.duration_minutes)?e.duration_minutes*60000:0)>=new Date(now).getTime()))
    .sort((a,b)=>Date.parse(a.start_at)-Date.parse(b.start_at)||a.id.localeCompare(b.id));
}

/** Explain importer safeguards rather than implying every warning is malformed Markdown. */
export function syncDiagnostic(status){
 const blocked=status?.lastResults?.filter(r=>['history_rewritten','error'].includes(r.status))??[];
 if(blocked.length)return `풀이 저장소 읽기 확인 필요: ${blocked.map(r=>`${r.repo_id} (${r.status})`).join(', ')}. 경로와 Git 이력을 확인해 주세요. 기존 기록은 유지돼요.`;
 const warnings=(status?.lastWarnings??[]).flatMap(w=>(w.flags??[]).map(flag=>({repo:w.repo_id,flag})));
 if(!warnings.length)return null;
 const conflicts=warnings.filter(w=>w.flag.startsWith('markdown_attempt_conflict:'));
 const other=warnings.filter(w=>!w.flag.startsWith('markdown_attempt_conflict:'));
 const parts=[];
 if(conflicts.length)parts.push(`기존 시도와 원본 수정 내용이 달라요: ${conflicts.map(w=>`${w.repo} / ${w.flag.slice('markdown_attempt_conflict:'.length)}`).join(', ')}. 시간·결과 변경도 같은 시도 ID의 충돌로 남아요. 이전 풀이와 시각은 보존했으며, 수정된 값은 자동 반영하지 않았어요. 원본 수정이 새 시도인지 정정인지 확인해 주세요.`);
 if(other.length)parts.push(`가져오기 확인 항목: ${other.map(w=>`${w.repo} / ${w.flag}`).join(', ')}. 해당 파일·행의 형식이나 결과를 확인해 주세요.`);
 return parts.join(' ');
}

/** Describe only the selection mechanism recorded on this slot; never invent a cosine score. */
export function recommendationBasis(event){
  if(event.explicit_review)return '직접 지정한 복습 대상 · 시스템 추천과 구분해요.';
  if(event.semantic_only)return '제목 벡터 이웃을 이용한 후보예요. 코사인 유사도 값은 이 추천 근거에 기록되지 않았어요. 알고리즘 동등성을 뜻하지 않아요.';
  if(Number.isFinite(event.components?.channel_rank)&&event.components.channel_rank>0)return '유사 문제 후보 순위를 점수에 반영했어요. 순위 가중 점수와 코사인 유사도 값은 서로 달라요. 원래 코사인 값은 이 추천에 기록되지 않았어요.';
  if(event.stage||event.kind==='review')return '학습 단계·복습 정책에 따른 선정이에요. 벡터 유사도 점수를 기준으로 선택한 추천은 아니에요.';
  return '기록된 정책 근거를 표시해요. 이 추천에는 벡터 유사도 수치가 제공되지 않았어요.';
}
