import {TIME_ZONE,localDate,shiftDate,calendarDates,navigateDate,workspaceEvents,activeAttempts,filterAttempts,completion,outcomeText,outcomeSymbol,groupCounts,conversion,eventCategory,filterCalendarEvents,daySummary,contestSuitability,contestList,recommendationBasis} from './workspace-data.mjs';
const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined&&text!==null)n.textContent=String(text);return n;};
const btn=(text,action,cls='button secondary')=>{const n=el('button',cls,text);n.type='button';if(text)n.dataset.focus=text;n.addEventListener('click',event=>{n.focus({preventScroll:true});action(event);});return n;};
const fmt=(v,options={dateStyle:'medium',timeStyle:'short'})=>Number.isFinite(new Date(v).getTime())?new Intl.DateTimeFormat('ko-KR',{timeZone:TIME_ZONE,hourCycle:'h23',...options}).format(new Date(v)):'시각 미기록';
const labelDate=d=>new Intl.DateTimeFormat('ko-KR',{dateStyle:'full',timeZone:'UTC'}).format(new Date(`${d}T12:00:00Z`));
const link=(title,url)=>{try{const u=new URL(url);if(u.protocol!=='https:'||u.username||u.password)return null;const a=el('a','external-link',title+' ↗');a.href=u.href;a.target='_blank';a.rel='noopener noreferrer';return a;}catch{return null;}};
const recordedDuration=e=>e.self_seconds!=null?`${e.self_seconds}s`:e.self_minutes!=null?`${e.self_minutes}m`:'미기록';
let dialogSerial=0;
export function openDialog(title,content,{wide=false,sourceUrl=null,titleIcon=null}={}) {
  const old=document.querySelector('.workspace-dialog');
  const origin=old?old.returnFocus:document.activeElement;
  const scroll=old?.returnScroll || {left:window.scrollX,top:window.scrollY};
  if(old){old.close();old.remove();}
  const d=el('dialog',`workspace-dialog${wide?' wide':''}`);d.returnFocus=origin;d.returnScroll=scroll;
  const id=`dialog-title-${++dialogSerial}`,head=el('header','dialog-head'),heading=el('h2','',title);heading.id=id;
  const close=btn('닫기 ×',()=>d.close(),'dialog-close');close.setAttribute('aria-label','대화상자 닫기');const titleGroup=el('div','dialog-title');if(titleIcon)titleGroup.append(titleIcon);titleGroup.append(heading);
  const source=link('문제 원문',sourceUrl);if(source){source.textContent='↗';source.classList.add('problem-source-link');source.setAttribute('aria-label',`${title} 문제 원문 열기 (새 탭)`);source.title='문제 원문 열기';titleGroup.append(source);}
  head.append(titleGroup,close);
  d.setAttribute('aria-labelledby',id);const body=el('div','dialog-body');body.append(content);d.append(head,body);document.body.append(d);
  d.addEventListener('close',()=>{d.remove();if(!document.querySelector('.workspace-dialog[open]')){document.body.classList.remove('dialog-open');const target=origin?.isConnected?(origin.getClientRects().length?origin:origin.closest('.calendar-cell')?.querySelector('.calendar-date')):document.querySelector('#main');target?.focus({preventScroll:true});window.scrollTo({...scroll,behavior:'instant'});}},{once:true});
  d.addEventListener('click',event=>{if(event.target===d){const r=d.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)d.close();}});
  document.body.classList.add('dialog-open');d.showModal();(content.querySelector('input:not([readonly]):not([disabled])') || close).focus();return d;
}
export function createWorkspace({root,getState,getJournal,adapters}) {
  let selected=localDate(new Date()),cursor=selected,mode='month',view='calendar';
  let layers={solved:true,attempted:true,planned:true,mock:true,contest:true},filters={query:'',platform:'',outcome:'',difficulty:'',from:'',to:''};
  let contestPlatform='',contestScope='upcoming';
  let analyticsYear=selected.slice(0,4),analyticsPlatform='',analyticsGrain='day';
  const data=()=>workspaceEvents(getJournal(),getState());
  const title=e=>e.title || adapters.problemTitle(e.problem_key) || e.problem_key || '학습 활동';
  const heading=(name,subtitle)=>{const h=el('div','workspace-heading');h.append(el('div','eyebrow','MY LEARNING WORKSPACE'),el('h1','page-heading',name),el('p','intro',subtitle));return h;};
  const empty=text=>el('p','empty',text);
  const card=(name,description)=>{const c=el('section','card analytics-card');c.append(el('h2','',name));if(description)c.append(el('p','meta',description));return c;};
  function status(e){return e.layer==='actual'?outcomeText(e):e.layer==='contest'?'대회':e.layer==='mock'?'모의 대회':e.kind==='review'?'복습 예정':'학습 예정';}
  const symbol=e=>e.layer==='actual'?outcomeSymbol(e):e.layer==='contest'?'⚑':e.layer==='mock'?'◆':'◇';
  const LEVELS=['','Easy','Medium','Hard'];
  const solvingTime=m=>m%60?(m>=60?`${Math.floor(m/60)}시간 ${m%60}분`:`${m}분`):`${m/60}시간`;
  const mockMeta=e=>[`풀이 시간 ${solvingTime(e.duration_minutes)}`,`E${e.composition?.easy ?? 0} M${e.composition?.medium ?? 0} H${e.composition?.hard ?? 0}`,`기록 ${e.attempted}/${e.problems.length}`].join(' · ');
  const recommendationBadge=e=>{
    if(!e.recommended)return null;
    const icon=el('span','recommendation-badge');icon.setAttribute('role','img');icon.setAttribute('aria-label','시스템 추천 문제');icon.title='시스템 추천 문제';
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 20 20');svg.setAttribute('aria-hidden','true');
    const path=document.createElementNS('http://www.w3.org/2000/svg','path');path.setAttribute('d','M10 2l2.2 5.8L18 10l-5.8 2.2L10 18l-2.2-5.8L2 10l5.8-2.2Z');svg.append(path);icon.append(svg);return icon;
  };
  function row(e,action=()=>details(e)) {
    const b=btn('',action,`workspace-row ${e.layer}${e.status==='superseded'?' superseded':''}`);
    const mark=el('span','event-symbol',symbol(e));
    const text=el('span','row-text');text.append(el('strong','',title(e)),el('span','meta',e.layer==='mock'?mockMeta(e):[e.platform,e.difficulty,e.pattern].filter(Boolean).join(' · ')||'추가 정보 미기록'));
    if(e.recommended)text.append(recommendationBadge(e));
    b.append(mark,text,el('span','row-status',status(e)));return b;
  }
  function summaryStats(items) {
    const summary=daySummary(items),stats=el('div','day-stats');
    for(const [key,name,rows] of [['solved','해결',activeAttempts(items).filter(completion)],['attempted','시도',activeAttempts(items)],['unsolved','미해결',activeAttempts(items).filter(e=>e.outcome==='failed')]]) {
      const b=btn('',()=>drill(`${labelDate(selected)} · ${name}`,rows),'day-stat');b.dataset.stat=key;
      b.append(el('strong','',summary[key]),el('span','',name));stats.append(b);
    }
    if(summary.unknown)stats.append(el('p','meta day-unknown',`결과 미상 ${summary.unknown}회 · 시도 횟수 기준`));
    return stats;
  }
  function daily(date){
    selected=date;const content=el('div','daily-content');
    const items=filterCalendarEvents(data(),layers).filter(e=>e.date===date);
    content.append(el('p','meta',`${TIME_ZONE} · 현재 캘린더 필터 기준 · 실제 풀이 시각과 계획 날짜를 구분해요.`),summaryStats(items));
    if(!items.length)content.append(empty('이 날짜에 표시할 기록이 없어요. 풀이를 기록하거나 필터를 바꿔 보세요.'));
    for(const [layer,name] of [['actual','실제 풀이'],['plan','학습 계획'],['mock','모의 대회'],['contest','프로그래밍 대회']]){
      const rows=items.filter(e=>e.layer===layer);if(!rows.length)continue;
      content.append(el('h3','group-title',`${name} · ${rows.length}`));rows.forEach(e=>content.append(row(e)));
    }
    content.append(btn('＋ 풀이 기록 추가',()=>edit(null,date),'button'));openDialog(labelDate(date),content);
  }
  function details(e){
    if(e.layer==='contest')return contestDetails(e);
    if(e.layer==='plan')return recommendation(e);
    if(e.layer==='mock')return mockDetails(e);
    const c=el('div','problem-details');c.append(el('p','eyebrow',`${e.platform || '플랫폼 미상'} · ${e.problem_key}`));
    const info=el('dl','detail-grid');
    for(const [k,v] of [['결과',outcomeText(e)],['풀이 날짜·시각 (KST)',fmt(e.occurred_at)],['기록 저장 시각',e.recorded_at?fmt(e.recorded_at):'미기록'],['난이도',e.difficulty??'미기록'],['주제',e.pattern||'미기록'],['힌트',e.hints===true?'사용':e.hints===false?'사용하지 않음':'미기록'],['직접 기록한 시간',recordedDuration(e)],['활동',e.activity_type||'미기록'],['출처',e.evidence_source||'수동 기록']]){info.append(el('dt','',k),el('dd','',v));}
    c.append(info,el('p','meta','날짜만 있는 일일 Markdown은 해당 날짜의 KST 00:00 경계로 저장돼요. 실제 풀이 시계 시각을 확인한 값은 아니에요. 별도 시작·완료 시각과 메모는 현재 기록 계약에 없어요. 확인된 타이머는 직접 입력 시간에 합산하지 않아요.'));
    const url=e.url||getState()?.curriculum?.problems?.find(p=>p.problem_key===e.problem_key)?.canonical_url;

    const editable=(getState()?.events || []).some(x=>x.event_id===e.id&&x.source==='manual')&&e.status!=='superseded';
    if(editable)c.append(btn('기록 수정',()=>edit(e),'button'));
    else c.append(el('p','meta',e.status==='superseded'?'수정 전 기록은 읽기 전용이에요.':'가져온 기록은 원본 풀이 저장소에서 수정해 주세요.'));
    c.append(btn('새 시도 기록',()=>edit(null,localDate(new Date()),e.problem_key)));
    c.append(el('h3','group-title','이 문제의 모든 시도'));
    data().filter(x=>x.layer==='actual'&&x.problem_key===e.problem_key).sort((a,b)=>b.date.localeCompare(a.date)).forEach(x=>{const r=el('div','attempt-audit');r.append(el('strong','',`${x.date} · ${outcomeText(x)}`),el('p','meta',`${fmt(x.occurred_at)} · ${recordedDuration(x)} · ${x.id}`));if(x.reason)r.append(el('p','meta',x.reason));c.append(r);});
    openDialog(title(e),c,{sourceUrl:url,titleIcon:recommendationBadge(e)});
  }
  function recommendation(e){
    const c=el('div','recommendation-detail');c.append(el('p','eyebrow',`예정 · ${e.date} · ${e.platform || ''}`),el('p','meta',[e.difficulty,e.kind,e.estimated_minutes==null?null:`예상 ${e.estimated_minutes}분`].filter(Boolean).join(' · ')));
    c.append(el('p','meta recommendation-basis',recommendationBasis(e)));
    if(e.semantic_only)c.append(el('p','notice','의미 유사 후보 · 알고리즘 동등성 미검증'));
    c.append(el('p','reason',e.reason || '추천 근거가 제공되지 않았어요.'));
    if(e.versions?.length){const versions=el('details','recommendation-versions');versions.append(el('summary','',`이 날짜의 추천 근거 변경 기록 · ${e.versions.length}개`));for(const version of e.versions){const item=el('section','attempt-audit');item.append(el('strong','',version.current?'현재 추천':version.archived?'보관된 추천':'이전 추천'),el('p','meta',[version.occurred_at?fmt(version.occurred_at):'시각 미기록',version.snapshot_id || version.id].join(' · ')),el('p','reason',version.reason || '근거 미기록'));versions.append(item);}c.append(versions);}
    c.append(explanationContent(e));
    openDialog(title(e),c,{wide:true,sourceUrl:e.url,titleIcon:recommendationBadge(e)});
  }
  function explanationContent(e){
    const c=el('div','comparison-content journal-plan');c.append(el('p','meta','저장된 AI 설명 · 사실성 검증 전. 공급자 호출은 기존 CLI에서 각각 실행할 수 있어요. 화면 조회는 API 생성을 실행하지 않아요.'));
    const columns=el('div','explanation-comparison');const labels={success:'생성 성공',failed:'생성 실패',not_configured:'설정 필요',not_generated:'아직 생성되지 않음',in_progress:'생성 중',unknown:'결과 확인 불가'};
    for(const provider of ['gemini','nvidia']){
      const r=(e.explanation?.results || []).find(x=>x.provider===provider),s=el('section','explanation-provider');s.dataset.provider=provider;
      s.append(el('h3','',provider==='gemini'?'Gemini':'NVIDIA NIM'),el('p','meta',r?.model || '모델 미설정'),el('span','tag',labels[r?.status]||'아직 생성되지 않음'));
      if(r?.status==='success')s.append(el('p','reason',r.text),el('p','meta',r.cache_hit?'저장된 결과':'기록된 모델 출력'));
      if(r?.error)s.append(el('p','notice error',`${r.error.code || '오류'} ${r.error.http_status || ''}`),el('p','meta',r.error.hint || '이 공급자의 설명을 사용할 수 없어요.'));
      columns.append(s);
    }
    c.append(columns);
    if(e.explanation?.fallback)c.append(el('h3','','기본 설명 · 사용 가능한 AI 설명 없음'),el('p','reason',e.explanation.fallback.text));
    const evidence=el('details','original-evidence');evidence.append(el('summary','','원래 추천 근거 보기'),el('p','reason',e.reason||'근거 미제공'));
    if(e.explanation?.evidence)evidence.append(el('pre','explanation-evidence',JSON.stringify(e.explanation.evidence,null,2)));c.append(evidence);
    return c;
  }
  function edit(previous,date=localDate(new Date()),key=''){
    const c=el('div','record-editor');c.append(el('p','meta',previous?'수정 내용은 새 기록으로 추가되며 기존 시도는 보존돼요.':'추천 날짜와 별개로 실제 풀이한 날짜·시각을 기록해 주세요.'));
    if(!getJournal()?.available){c.append(empty('카탈로그 일지를 사용할 수 없어요. 학습 도구의 기존 풀이 기록 양식을 이용해 주세요.'),btn('학습 도구',()=>{document.querySelector('.workspace-dialog')?.close();adapters.tools();}));return openDialog('풀이 기록',c);}
    const form=el('form','form-grid');
    const control=(name,label,type,value)=>{const wrap=el('label','field',label),n=el('input');n.name=name;n.id=`record-${name}`;n.type=type;n.value=value??'';wrap.htmlFor=n.id;wrap.append(n);form.append(wrap);return n;};
    const problem=control('problem','문제 키','text',previous?.problem_key||key);problem.required=true;problem.readOnly=!!previous;problem.placeholder='예: leetcode:two-sum';
    const options=el('datalist');options.id='record-problems';
    const choices=new Map();for(const x of [...data(),...(getState()?.curriculum?.problems || [])])if(x.problem_key)choices.set(x.problem_key,title(x));
    for(const [k,v]of choices){const o=el('option','',v);o.value=k;options.append(o);}problem.setAttribute('list',options.id);form.append(options);
    const dateInput=control('date','실제 풀이 날짜','date',previous?.date||date);dateInput.required=true;dateInput.max=localDate(new Date());
    const time=previous?.occurred_at?new Intl.DateTimeFormat('en-GB',{timeZone:TIME_ZONE,hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'}).format(new Date(previous.occurred_at)):'12:00:00';
    const timeInput=control('time',`시각 (${TIME_ZONE})`,'time',time);timeInput.required=true;timeInput.step='0.001';
    const wrap=el('label','field','결과'),outcome=el('select');outcome.id='record-outcome';wrap.htmlFor=outcome.id;
    for(const [v,t]of [['independent','독립 해결 · 힌트 없음'],['assisted','도움을 받아 해결'],['failed','아직 해결하지 못함']]){const o=el('option','',t);o.value=v;outcome.append(o);}outcome.value=previous?.outcome==='reported_completion'?'assisted':previous?.outcome||'independent';wrap.append(outcome);form.append(wrap);
    const hintWrap=el('label','field','힌트와 도움'),hints=el('select');hints.id='record-hints';hintWrap.htmlFor=hints.id;for(const [v,t] of [['false','사용하지 않음'],['true','사용'],['','미기록']]){const o=el('option','',t);o.value=v;hints.append(o);}hints.value=previous?.hints==null?(previous?'':'false'):String(previous.hints);hintWrap.append(hints);form.append(hintWrap);
    const minutes=control('minutes','직접 사용한 시간 (예: 10m, 40s · 선택)','text',previous&&(previous.self_seconds!=null||previous.self_minutes!=null)?recordedDuration(previous):'');minutes.placeholder='10m 또는 40s';minutes.pattern='[0-9]+[ms]';
    const error=el('p','form-error full');error.setAttribute('role','alert');const submit=el('button','button','저장');submit.type='submit';form.append(error,submit);c.append(form);
    const d=openDialog(previous?'풀이 기록 수정':'풀이 기록 추가',c);
    let submittedPayload=null;
    form.addEventListener('submit',async event=>{event.preventDefault();error.textContent='';
      const unchanged=previous&&dateInput.value===previous.date&&timeInput.value===time;
      const occurred=unchanged?previous.occurred_at:new Date(`${dateInput.value}T${timeInput.value}+09:00`).toISOString();
      if(new Date(occurred)>new Date()){error.textContent='미래 시각에는 풀이 기록을 저장할 수 없어요.';return;}
      if(outcome.value==='independent'&&hints.value!=='false'){error.textContent='독립 해결은 힌트를 사용하지 않은 경우만 기록해요.';return;}
      if(outcome.value==='assisted'&&hints.value==='false'){error.textContent='도움을 받아 해결한 경우 힌트와 도움을 사용 또는 미기록으로 선택해 주세요.';return;}
      const duration=minutes.value.trim();
      if(duration&&!/^\d{1,5}[ms]$/.test(duration)||duration&&parseInt(duration,10)>(duration.endsWith('s')?86400:1440)){error.textContent='시간은 10m 또는 40s처럼 입력해 주세요 (최대 24시간).';return;}
      const payload={event_id:crypto.randomUUID(),problem_key:problem.value.trim(),outcome:outcome.value,hints:hints.value===''?null:hints.value==='true',self_minutes:duration.endsWith('m')?parseInt(duration,10):null,...(duration.endsWith('s')?{self_seconds:parseInt(duration,10)}:{}),occurred_at:occurred,activity_type:previous?.activity_type||'practice',...(previous?{supersedes:previous.id}:{}),...(previous?.transfer_from?{transfer_from:previous.transfer_from}:{})};
      submittedPayload??=payload;for(const control of form.querySelectorAll('input,select'))control.disabled=true;submit.disabled=true;try{const ok=await adapters.saveAttempt(submittedPayload);if(ok)d.close();else error.textContent='저장하지 못했어요. 화면 상단의 오류와 재시도 기록을 확인해 주세요.';}catch(e){error.textContent=e.message;}finally{submit.disabled=false;}
    });
  }
  function mockDetails(e){
    // Selection record only: problems are solved and judged on LeetCode; results come from recorded attempts.
    const c=el('div','mock-detail'),info=el('dl','detail-grid');
    for(const[k,v]of [['날짜',labelDate(e.date)],['풀이 시간',solvingTime(e.duration_minutes)],['난이도 구성',`Easy ${e.composition?.easy ?? 0} · Medium ${e.composition?.medium ?? 0} · Hard ${e.composition?.hard ?? 0}`],
      ['예상 해결 수 (보수적)',`${e.expected_solved} · 목표 ${e.target_expected}`],['기록된 결과',`${e.attempted} / ${e.problems.length}문제`],['선정 정책',e.policy || '미기록']])info.append(el('dt','',k),el('dd','',v));
    c.append(info);
    if(e.archived)c.append(el('p','notice','백업에서 복원된 보관 기록이에요. 일일 추천에서 문제를 제외하는 데에는 쓰이지 않아요.'));
    if(e.evidence_label)c.append(el('p','notice','LeetCode 풀이 근거가 부족해 기본 난이도 구성을 사용했어요.'),el('p','meta',e.evidence_label));
    c.append(el('h3','group-title',`문제 · ${e.problems.length}`));
    const list=el('ol','mock-problems');
    for(const p of e.problems){
      const item=el('li',`mock-problem${p.result?' recorded':''}`),head=el('div','mock-problem-head');
      head.append(el('span','tag',LEVELS[p.level] || '레벨 미상'));const a=link(p.title || p.problem_key,p.url);head.append(a || el('strong','',p.title || p.problem_key));
      item.append(head,el('p','meta',`${p.problem_key} · 해결 확률 ${p.p_conservative}–${p.p_mean} (보수적–평균) · ${(p.tags || []).join(', ') || '태그 없음'}`),el('p','reason',p.reason));
      item.append(el('p',`mock-result${p.result?'':' empty'}`,p.result?`${p.result.symbol} ${p.result.text} · ${p.result.date}`:'아직 기록된 시도가 없어요.'));
      if(!p.result)item.append(btn('풀이 기록',()=>edit(null,localDate(new Date()),p.problem_key)));
      list.append(item);
    }
    c.append(list);
    if(e.notes?.length)for(const n of e.notes)c.append(el('p','meta',n));
    c.append(el('p','meta','LeetCode에서 풀고 채점해 주세요. 결과는 풀이 기록·daily log·Git으로 남기면 여기에 표시돼요. 이 화면은 시도를 만들지 않고, 결과 연결은 표시용이에요.'));
    openDialog(title(e),c,{wide:true});
  }
  function contestDetails(e){const c=el('div','contest-detail');const info=el('dl','detail-grid');
    for(const[k,v]of [['플랫폼',e.platform],['시작 (KST)',fmt(e.start_at)],['진행 시간',e.duration_minutes==null?'미확인':`${e.duration_minutes}분`],['제공처',e.source_kind==='community_clist_mirror'?'CLIST → ContestDeck (제3자 피드)':e.source_kind==='official'?'플랫폼 공식 일정':'수동 / 기존 캐시'],['출처 갱신 시각',e.source_updated_at?fmt(e.source_updated_at):'제공처 미제공'],['Rated range',e.rated_range||'미제공 · 난이도/참가 자격과 다름'],['출처',e.source==='manual'?'직접 입력 · 자동 검증 아님':e.source||'미확인'],['출처 유형',e.source_kind==='community_clist_mirror'?'커뮤니티 일정 미러 · 공식 공지에서 재확인':e.source_kind==='official'?'공식 일정':e.source==='manual'?'사용자 입력':'제공처 정보'],[e.source==='manual'?'수동 입력 저장 시각':'마지막 성공 동기화',e.fetched_at?fmt(e.fetched_at):'없음 / 미확인'],['상태',e.stale?'오래된 캐시 · 공식 페이지 확인 필요':e.source==='manual'?'수동 일정':'제공처 캐시']])info.append(el('dt','',k),el('dd','',v||'미확인'));
    const suitability=contestSuitability(e,data(),getState()?.participation);
    const fit=el('section',`contest-fit ${suitability.status}`);fit.append(el('span','fit-badge',suitability.label),el('p','',suitability.reason),el('p','meta',`같은 플랫폼의 기록: ${suitability.evidence.platform_attempts}회 시도 · 명시적 독립 해결 ${suitability.evidence.independent}회. 이 수치는 대회 실력 등급이 아니에요.`));
    const evidence=el('details','suitability-evidence');evidence.append(el('summary','','판단에 사용한 기록 보기'));if(suitability.evidence.participation_ids?.length)for(const id of suitability.evidence.participation_ids){const entry=getState().participation.find(x=>x.entry_id===id);if(!entry)continue;evidence.append(el('p','meta',`${entry.mode} · ${entry.outcome} · ${fmt(entry.occurred_at)}`));}else evidence.append(el('p','meta',e.source==='manual'&&e.suitable?'사용자가 직접 선택한 일정이에요.':'같은 대회의 참여 기록이 없어요.'));if(suitability.evidence.native_difficulties?.length)evidence.append(el('pre','explanation-evidence',JSON.stringify({catalog:suitability.evidence.catalog_version,calibration:suitability.evidence.calibration,distinct_independent:suitability.evidence.distinct_independent,attempt_ids:suitability.evidence.attempt_ids,calibration_attempt_ids:suitability.evidence.calibration_attempt_ids,rating_ids:suitability.evidence.rating_ids,independent_difficulties:suitability.evidence.independent_difficulties,historical_contests:suitability.evidence.historical_contests,native_difficulties:suitability.evidence.native_difficulties},null,2)));if(suitability.evidence.calibration)evidence.append(el('p','meta',`플랫폼별 보정 상태: ${suitability.evidence.calibration.status} · ${suitability.evidence.calibration.n}회 관측`));if(suitability.evidence.historical_contests?.length)evidence.append(el('p','meta',`과거 대회 비교: ${suitability.evidence.historical_contests.join(', ')} · 이번 대회의 실제 난이도가 아닌 첫 문제 연습용 대리 지표`));if(suitability.evidence.attempt_ids?.length){const ids=new Set(suitability.evidence.attempt_ids),rows=activeAttempts(data()).filter(x=>ids.has(x.id));if(rows.length)evidence.append(btn('근거 풀이 기록 확인',()=>drill('대회 판단의 학습 근거',rows)));}fit.append(evidence);c.append(info,fit,el('p','meta','등록 가능 여부와 참여 조건은 공식 페이지에서 확인해 주세요.'));const a=link(e.source==='manual'?'직접 입력한 대회 페이지':'제공처의 공식 대회 링크',e.url);if(a)c.append(a);c.append(btn('캘린더에서 날짜 보기',()=>{selected=cursor=e.date;document.querySelector('dialog[open]')?.close();document.querySelector('[data-view=calendar]')?.click();}));openDialog(title(e),c);
  }
  function selectDay(date,{open=false}={}) {
    selected=date;render();root.querySelector(`[data-date="${date}"] .calendar-date`)?.focus({preventScroll:true});
    if(open)daily(date);
  }
  function selectedPanel(events,today) {
    const rail=el('aside','agenda-rail');rail.setAttribute('aria-label','선택한 날짜의 학습');rail.dataset.date=selected;
    const dateHead=el('div','selected-day-heading');dateHead.append(el('p','eyebrow',selected===today?'TODAY · SELECTED':'SELECTED DAY'),el('h2','',labelDate(selected)));rail.append(dateHead);
    const items=events.filter(e=>e.date===selected);rail.append(summaryStats(items));
    const actions=el('div','day-actions');actions.append(btn('＋ 풀이 기록',()=>edit(null,selected),'button'),btn('하루 자세히',()=>daily(selected)));rail.append(actions);
    if(!items.length)rail.append(el('p','agenda-empty','표시할 활동이 없어요. 풀이를 기록하거나 다른 날짜를 선택해 보세요.'));
    for(const [k,t]of [['actual','실제 풀이'],['plan','예정된 학습'],['mock','모의 대회'],['contest','대회']]){
      const rows=items.filter(e=>e.layer===k);if(!rows.length)continue;
      const section=el('section',`day-section ${k}`);section.append(el('h3','group-title',`${t} · ${rows.length}`));
      rows.slice(0,k==='actual'?4:3).forEach(e=>section.append(row(e)));const remaining=rows.length-(k==='actual'?4:3);if(remaining>0)section.append(btn(`+${remaining}개 전체 보기`,()=>daily(selected),'more-events'));rail.append(section);
    }
    return rail;
  }
  function renderCalendar(){
    const events=filterCalendarEvents(data(),layers),today=localDate(new Date());
    root.append(heading('학습 캘린더','실제 풀이를 먼저, 계획과 대회를 함께. 날짜를 선택해 하루를 살펴보세요.'));
    const tools=el('div','calendar-toolbar'),left=el('div','calendar-navigation');
    const prev=btn('‹',()=>{cursor=navigateDate(cursor,-1,mode);render();},'icon-button');prev.setAttribute('aria-label',mode==='week'?'이전 주':'이전 달');
    const next=btn('›',()=>{cursor=navigateDate(cursor,1,mode);render();},'icon-button');next.setAttribute('aria-label',mode==='week'?'다음 주':'다음 달');
    left.append(el('h2','month-title',`${cursor.slice(0,4)}년 ${Number(cursor.slice(5,7))}월`),prev,next,btn('오늘',()=>{selected=today;cursor=today;render();}));
    const right=el('div','actions'),switcher=el('div','segmented');
    for(const[v,t]of [['month','월'],['week','주']]){const b=btn(t,()=>{mode=v;cursor=selected;render();},'segment');b.setAttribute('aria-pressed',String(mode===v));switcher.append(b);}
    right.append(el('span','timezone',TIME_ZONE),switcher,btn('＋ 풀이 기록',()=>edit(null,selected),'button'));tools.append(left,right);root.append(tools);
    const legend=el('div','calendar-legend');
    for(const[k,t]of [['solved','해결 기록'],['attempted','시도 · 미해결'],['planned','추천 · 복습 예정'],['mock','모의 대회'],['contest','대회']]){
      const l=el('label',`layer-toggle ${k}`),i=el('input');i.type='checkbox';i.dataset.focus=`layer:${k}`;i.setAttribute('aria-label',`${t} 표시`);i.checked=layers[k];i.addEventListener('change',()=>{layers[k]=i.checked;render();});l.append(i,document.createTextNode(t));legend.append(l);
    }
    legend.append(el('span','calendar-key','● 오늘  □ 선택한 날짜'));root.append(legend);
    const layout=el('div','calendar-workspace'),calendar=el('section',`calendar-board ${mode}`);calendar.setAttribute('aria-label',mode==='month'?'월간 학습 캘린더':'주간 학습 캘린더');
    const headers=el('div','calendar-weekdays');for(const day of ['월','화','수','목','금','토','일'])headers.append(el('span','',day));calendar.append(headers);
    const grid=el('div','calendar-grid');
    for(const date of calendarDates(cursor,mode)){
      const weekDay=new Date(`${date}T12:00:00Z`).getUTCDay();
      const cell=el('div',`calendar-cell${[0,6].includes(weekDay)?' weekend':''}${date.slice(0,7)!==cursor.slice(0,7)?' outside':''}${date===selected?' selected':''}${date===today?' is-today':''}`);cell.dataset.date=date;
      const list=events.filter(e=>e.date===date),stats=daySummary(list);
      const dateButton=btn(String(Number(date.slice(-2))),()=>selectDay(date),'calendar-date');
      dateButton.setAttribute('aria-label',`${labelDate(date)} · ${stats.attempted}회 시도 · ${list.filter(e=>e.layer==='plan').length}개 계획 · ${list.filter(e=>e.layer==='mock').length}개 모의 대회 · ${list.filter(e=>e.layer==='contest').length}개 대회`);
      dateButton.setAttribute('aria-pressed',String(date===selected));if(date===today)dateButton.setAttribute('aria-current','date');
      dateButton.addEventListener('keydown',event=>{const delta={ArrowLeft:-1,ArrowRight:1,ArrowUp:-7,ArrowDown:7}[event.key];if(delta){event.preventDefault();const target=shiftDate(date,delta);if(!calendarDates(cursor,mode).includes(target))cursor=target;selectDay(target);}});
      cell.append(dateButton);
      for(const e of list.slice(0,3)){
        const b=btn('',()=>details(e),`calendar-event ${e.layer} ${eventCategory(e)}`);b.title=`${status(e)} · ${title(e)}`;b.setAttribute('aria-label',b.title);
        const concise=e.layer==='mock'?`모의 대회 · ${solvingTime(e.duration_minutes)}`:title(e).length<=24?title(e):e.layer==='contest'?`${e.platform} · ${fmt(e.start_at,{hour:'2-digit',minute:'2-digit',hourCycle:'h23'})}`:`${e.platform || '문제'} · ${status(e)}`;
        b.append(el('span','event-symbol',symbol(e)),el('span','event-title',concise));if(e.recommended)b.append(recommendationBadge(e));cell.append(b);
      }
      if(list.length>3)cell.append(btn(`+${list.length-3}개 더 보기`,()=>selectDay(date,{open:true}),'more-events'));
      if(list.length){const dots=el('div','mobile-activity');for(const[k,name]of [['actual','풀이'],['plan','계획'],['mock','모의'],['contest','대회']]){const n=list.filter(e=>e.layer===k).length;if(n)dots.append(el('span',`activity-dot ${k}`,`${n}`));}dots.setAttribute('aria-hidden','true');cell.append(dots);}
      grid.append(cell);
    }
    calendar.append(grid,el('p','calendar-footnote','✓ 독립 해결  ↗ 도움 해결  ? 완료 보고  · 미해결  ◆ 모의 대회   /   더보기에서 하루 전체 확인'));
    layout.append(calendar,selectedPanel(events,today));root.append(layout,monthlyInsights(events));
  }
  function monthlyInsights(events) {
    const month=cursor.slice(0,7),attempts=activeAttempts(events).filter(e=>e.date.startsWith(month)),section=el('section','monthly-insights');
    const head=el('div','insights-heading');head.append(el('h2','',`${Number(month.slice(5))}월의 학습 기록`),el('p','meta','현재 필터의 활성 시도 · 차트를 눌러 기록 확인'));section.append(head);
    if(!attempts.length){section.append(el('p','meta','아직 이달에 표시할 풀이 기록이 없어요. 기록이 쌓이면 활동과 풀이 분포를 보여드려요.'));return section;}
    const grid=el('div','insights-grid'),heat=card('월간 활동','하루 시도 수 · 미기록은 실패가 아니에요.'),cells=el('div','month-heatmap');
    for(const day of ['월','화','수','목','금','토','일'])cells.append(el('span','month-heat-weekday',day));
    for(const date of calendarDates(cursor)){
      const rows=attempts.filter(e=>e.date===date),b=btn(date.slice(-2).replace(/^0/,''),()=>{cursor=date;selectDay(date,{open:true});},'month-heat-day');b.dataset.level=rows.length===0?'0':rows.length===1?'1':rows.length<=3?'2':rows.length<=6?'3':'4';
      if(!date.startsWith(month))b.disabled=true;b.title=`${date} · ${rows.length}회 시도`;b.setAttribute('aria-label',b.title);cells.append(b);
    }
    heat.append(cells,el('p','monthly-scale','미기록 · 1회 · 2–3회 · 4–6회 · 7회+'));grid.append(heat);
    const trend=card('시도 / 해결','주 시작일 기준 · 이달 기록만 집계'),groups=groupCounts(attempts,e=>shiftDate(e.date,-((new Date(`${e.date}T12:00:00Z`).getUTCDay()+6)%7))).sort((a,b)=>a[0].localeCompare(b[0]));
    const max=Math.max(1,...groups.map(x=>x[1].length));for(const[d,rows]of groups){const b=btn('',()=>drill(`${d} 시작 주 · ${month} 기록`,rows),'trend-row');const tracks=el('span','trend-tracks');for(const[cls,n]of [['attempted',rows.length],['completed',rows.filter(completion).length]]){const fill=el('span',cls);fill.style.width=`${n/max*100}%`;tracks.append(fill);}b.append(el('time','',d.slice(5)),tracks,el('span','',`${rows.length} / ${rows.filter(completion).length}`));trend.append(b);}if(!groups.length)trend.append(el('p','meta','기록된 시도가 없어요.'));grid.append(trend);
    const independent=bars('독립 / 도움 해결',groupCounts(attempts.filter(completion),outcomeText),'해결 기록의 도움 여부 · 완료 보고는 별도 구분');grid.append(independent);
    const topicGroups=new Map();for(const e of attempts){const topics=[...new Set((e.pattern || '').split(',').map(x=>x.trim()).filter(Boolean))];for(const topic of topics.length?topics:['주제 미기록'])topicGroups.set(topic,[...(topicGroups.get(topic)||[]),e]);}
    const topicRows=[...topicGroups].sort((a,b)=>b[1].length-a[1].length),topics=bars('알고리즘 주제',topicRows.slice(0,4),'태그별 시도 수 · 한 시도에 여러 태그가 있을 수 있어요.');if(topicRows.length>4)topics.append(btn(`${topicRows.length-4}개 주제 더 보기`,()=>{const c=bars('전체 주제',topicRows,'숙련도 점수가 아닌 풀이 기록이에요.');openDialog('월간 주제 기록',c);}));grid.append(topics);
    section.append(grid);return section;
  }
  function renderContestList() {
    root.append(heading('프로그래밍 대회','조회된 일정을 시간순으로 모았어요. 같은 일정이 학습 캘린더에도 표시돼요.'));
    const controls=el('div','contest-list-toolbar'),platform=el('select');platform.setAttribute('aria-label','대회 플랫폼');platform.dataset.focus='contest-platform-filter';
    for(const[v,t]of [['','모든 플랫폼'],['codeforces','Codeforces'],['atcoder','AtCoder'],['leetcode','LeetCode'],['other','기타']]){const o=el('option','',t);o.value=v;platform.append(o);}platform.value=contestPlatform;platform.addEventListener('change',()=>{contestPlatform=platform.value;render();});
    const scope=el('div','segmented');for(const[v,t]of [['upcoming','예정 · 진행 중'],['all','조회된 전체']]){const b=btn(t,()=>{contestScope=v;render();},'segment');b.setAttribute('aria-pressed',String(contestScope===v));scope.append(b);}
    controls.append(platform,scope,el('span','meta',TIME_ZONE));const sync=btn('공개 일정 동기화',async()=>{sync.disabled=true;await adapters.syncContests?.();sync.disabled=false;},'button');sync.disabled=!getState()?.capabilities?.contest_discovery;controls.append(sync);root.append(controls);
    const items=contestList(data(),{platform:contestPlatform,scope:contestScope}),list=el('section','contest-agenda');list.setAttribute('aria-label','조회된 대회 목록');
    if(!items.length)list.append(empty('이 조건에 맞는 일정이 없어요. 아래에서 출처 상태를 확인하거나 일정을 직접 추가할 수 있어요.'));
    let lastDate='';for(const e of items){if(e.date!==lastDate){lastDate=e.date;const h=el('h2','contest-date-group',labelDate(e.date));if([0,6].includes(new Date(`${e.date}T12:00:00Z`).getUTCDay()))h.append(el('span','weekend-badge','주말'));list.append(h);}
      const fit=contestSuitability(e,data(),getState()?.participation),b=btn('',()=>contestDetails(e),'contest-list-item');b.dataset.contestId=e.contest_id;
      const time=el('span','contest-time');time.append(el('strong','',fmt(e.start_at,{hour:'2-digit',minute:'2-digit',hourCycle:'h23'})),el('small','',e.duration_minutes==null?'시간 미확인':`${e.duration_minutes}분`));
      const name=el('span','contest-list-name');name.append(el('strong','',title(e)),el('span','meta',`${e.platform} · ${e.source==='manual'?'직접 입력':e.source_kind==='community_clist_mirror'?'커뮤니티 일정 미러':e.stale?'이전 조회 캐시':'조회된 일정'}`));
      name.append(el('span','meta',`마지막 성공: ${e.fetched_at?fmt(e.fetched_at):'미확인'}`));const badges=el('span','contest-list-badges');badges.append(el('span',`fit-badge ${fit.status}`,fit.label));if(e.stale)badges.append(el('span','stale-badge','오래된 캐시'));b.append(time,name,badges,el('span','list-chevron','›'));list.append(b);
    }
    root.append(list);contestSources();
  }
  function contestSources(){
    const panel=card('일정 출처와 동기화','실패한 조회는 마지막 성공 캐시를 유지해요. 커뮤니티 일정은 공식 공지에서 확인해 주세요.');
    for(const platform of ['codeforces','atcoder','leetcode']){
      const source=(getState()?.contest_sources || []).filter(s=>String(s.source).includes(platform));
      const row=el('div','source-row');row.append(el('strong','',platform==='leetcode'?'LeetCode':platform==='atcoder'?'AtCoder':'Codeforces'));
      row.append(el('span','meta',source.length?source.map(s=>`${({fresh:'최근 조회 캐시',stale:'오래된 캐시',unavailable:'일정 확인 불가'})[s.status] || s.status} · ${s.fetched_at?'마지막 성공 '+fmt(s.fetched_at):'성공한 조회 없음'}${s.status!=='fresh'&&s.error&&!['not_synced','stale'].includes(s.error)?' · 오류 기록 있음':''}`).join(' / '):'성공한 동기화 없음 · 공개 일정 동기화로 확인'));
      const a=link('공식 일정',platform==='leetcode'?'https://leetcode.com/contest/':platform==='atcoder'?'https://atcoder.jp/contests/':'https://codeforces.com/contests');if(a)row.append(a);if(source[0]?.source_updated_at)row.append(el('span','meta',`원본 갱신 ${fmt(source[0].source_updated_at)}`));panel.append(row);
    }
    const actions=el('div','actions');actions.append(btn('일정 동기화 / 제출 가져오기',()=>openDialog('대회 데이터 가져오기',adapters.providerSyncCard())),btn('＋ 대회 직접 추가',()=>openDialog('대회 직접 추가',adapters.manualContestForm())),btn('참여 기록',()=>{const c=el('div');c.append(adapters.participationCard(),adapters.participationHistory());openDialog('대회 참여 기록',c);}));panel.append(actions);root.append(panel);
  }
  function filterControl(label,name,choices,rerender){const wrap=el('label','filter-control',label),s=el('select');s.setAttribute('aria-label',label);for(const[v,t]of choices){const o=el('option','',t);o.value=v;s.append(o);}s.value=filters[name]||'';s.addEventListener('change',()=>{filters[name]=s.value;rerender();});wrap.append(s);return wrap;}
  function history(){root.append(heading('풀이 이력','실제 시도를 검색하고, 이전 풀이와 수정 기록을 함께 확인하세요.'));
    const controls=el('div','history-filters'),search=el('input');search.type='search';search.placeholder='문제 이름, 키 또는 알고리즘 주제 검색';search.setAttribute('aria-label','풀이 이력 검색');search.value=filters.query;
    const results=el('section','history-results');const update=()=>{results.replaceChildren();const rows=filterAttempts(data(),filters).sort((a,b)=>b.date.localeCompare(a.date));results.append(el('p','meta',`${rows.length}회 시도 · 현재 기록 기준 · 수정 전 기록은 문제 상세에서 확인`));if(!rows.length)results.append(empty('조건에 맞는 풀이 기록이 없어요.'));else{const table=el('div','history-table');const header=el('div','history-table-head');for(const t of ['문제 / 주제','플랫폼 · 난이도','결과','풀이 날짜','시간'])header.append(el('span','',t));table.append(header);for(const e of rows){const b=btn('',()=>details(e),'history-table-row');const name=el('span','history-problem');name.append(el('strong','',title(e)),el('span','meta',e.pattern||'주제 미기록'));if(e.recommended)name.append(recommendationBadge(e));b.append(name,el('span','',`${e.platform || '미상'} · ${e.difficulty??'미기록'}`),el('span','',outcomeText(e)),el('time','',e.date),el('span','',recordedDuration(e)));table.append(b);}results.append(table);}};
    search.addEventListener('input',()=>{filters.query=search.value;update();});controls.append(search);
    const all=activeAttempts(data()),choices=key=>[...new Set(all.map(e=>e[key]).filter(v=>v!=null).map(String))].sort().map(v=>[v,v]);
    controls.append(filterControl('플랫폼','platform',[['','모든 플랫폼'],...choices('platform')],update),filterControl('결과','outcome',[['','모든 결과'],['independent','독립 해결'],['assisted','도움 해결'],['failed','미해결'],['reported_completion','완료 보고']],update),filterControl('난이도 (플랫폼별)','difficulty',[['','모든 난이도'],...choices('difficulty')],update));
    for(const[k,t]of [['from','시작일'],['to','종료일']]){const l=el('label','filter-control',t),i=el('input');i.type='date';i.value=filters[k];i.setAttribute('aria-label',t);i.addEventListener('change',()=>{filters[k]=i.value;update();});l.append(i);controls.append(l);}
    controls.append(btn('초기화',()=>{filters={query:'',platform:'',outcome:'',difficulty:'',from:'',to:''};render();}),btn('＋ 기록',()=>edit(),'button'));root.append(controls,results);update();
  }
  function drill(name,rows){const c=el('div','drilldown');c.append(el('p','meta',`${rows.length}개 원본 항목 · 현재 선택 범위`));if(!rows.length)c.append(empty('기록된 항목이 없어요.'));rows.forEach(e=>{const wrap=el('div');wrap.append(el('span','meta',e.date),row(e));c.append(wrap);});openDialog(name,c,{wide:true});}
  function bars(name,groups,description){const c=card(name,description),max=Math.max(1,...groups.map(x=>x[1].length));if(!groups.length)c.append(empty('기록된 데이터가 없어요.'));for(const[k,rows]of groups){const b=btn('',()=>drill(`${name} · ${k}`,rows),'analytics-bar');b.append(el('span','bar-label',k));const track=el('span','bar-track'),fill=el('span','bar-fill');fill.style.width=`${rows.length/max*100}%`;track.append(fill);b.append(track,el('strong','bar-value',`${rows.length}회`));c.append(b);}return c;}
  function analytics(){const events=data();root.append(heading('학습 분석','기록된 풀이를 살펴보세요. 활동량은 숙련도나 학습 효과를 뜻하지 않아요.'));
    const controls=el('div','analytics-controls');const year=el('select');year.setAttribute('aria-label','분석 연도');const years=[...new Set([localDate(new Date()).slice(0,4),...activeAttempts(events).map(e=>e.date.slice(0,4))])].sort().reverse();for(const y of years){const o=el('option','',`${y}년`);o.value=y;year.append(o);}year.value=analyticsYear;year.addEventListener('change',()=>{analyticsYear=year.value;render();});
    const platform=el('select');platform.setAttribute('aria-label','분석 플랫폼');for(const p of ['',...new Set(activeAttempts(events).map(e=>e.platform).filter(Boolean))]){const o=el('option','',p||'모든 플랫폼');o.value=p;platform.append(o);}platform.value=analyticsPlatform;platform.addEventListener('change',()=>{analyticsPlatform=platform.value;render();});controls.append(year,platform,el('span','meta',`${TIME_ZONE} · 불러온 기록 범위`));root.append(controls);
    const attempts=activeAttempts(events).filter(e=>e.date.startsWith(analyticsYear)&&(!analyticsPlatform||e.platform===analyticsPlatform));
    const totals=el('div','metric-strip');for(const [n,rows]of [['전체 시도',attempts],['해결 기록',attempts.filter(completion)],['독립 해결',attempts.filter(e=>e.outcome==='independent'&&e.hints===false)],['학습한 날짜',[...new Set(attempts.map(e=>e.date))].map(date=>({date}))]]){const b=btn('',()=>drill(n,n==='학습한 날짜'?attempts:rows),'metric');b.append(el('span','meta',n),el('strong','',rows.length));totals.append(b);}root.append(totals);
    const heat=card('연간 학습 활동','색 농도 = 기록된 시도 수 · 빈 날짜 = 기록 없음 · 클릭하면 해당 날짜를 확인해요.');const heatScroll=el('div','heatmap-scroll'),heatInner=el('div','heatmap-inner'),months=el('div','heatmap-months'),weekdays=el('div','heatmap-days'),heatmap=el('div','activity-heatmap');const first=`${analyticsYear}-01-01`,last=`${analyticsYear}-12-31`,offset=(new Date(`${first}T12:00:00Z`).getUTCDay()+6)%7;
    for(let month=1;month<=12;month++){const d=`${analyticsYear}-${String(month).padStart(2,'0')}-01`,index=Math.floor((Math.round((Date.parse(d)-Date.parse(first))/86400000)+offset)/7)+1;const label=el('span','',`${month}월`);label.style.gridColumn=String(index);months.append(label);}
    for(const name of ['월','화','수','목','금','토','일'])weekdays.append(el('span','',name));
    for(let i=0;i<offset;i++){const p=el('span','heat-spacer');heatmap.append(p);}for(let d=first;d<=last;d=shiftDate(d,1)){const rows=attempts.filter(e=>e.date===d),count=rows.length,b=btn('',()=>drill(`${labelDate(d)} · 실제 시도`,rows),'heat-cell');b.dataset.level=String(count===0?0:count===1?1:count<=3?2:count<=6?3:4);b.title=`${d} · ${count}회 시도`;b.dataset.date=d;b.tabIndex=d===first?0:-1;b.setAttribute('aria-label',b.title);b.addEventListener('keydown',event=>{const delta={ArrowRight:7,ArrowLeft:-7,ArrowDown:1,ArrowUp:-1}[event.key];if(delta){event.preventDefault();const next=heatmap.querySelector(`[data-date="${shiftDate(b.dataset.date,delta)}"]`);if(next){b.tabIndex=-1;next.tabIndex=0;next.focus();}}});heatmap.append(b);}heatInner.append(months,weekdays,heatmap);heatScroll.append(heatInner);heat.append(heatScroll);const legend=el('div','heat-legend');for(const[level,text]of [['0','미기록'],['1','1회'],['2','2–3회'],['3','4–6회'],['4','7회 이상']]){const item=el('span','heat-key'),swatch=el('span','heat-cell');swatch.dataset.level=level;item.append(swatch,document.createTextNode(text));legend.append(item);}heat.append(legend);root.append(heat);
    const grid=el('div','analytics-grid');const activity=card('시도와 해결 추이','시도 = 활성 풀이 기록 수 · 해결 = 독립/도움/완료 보고. 플랫폼 간 난이도를 합산하지 않아요.');const switcher=el('div','segmented');for(const[v,t]of [['day','일간'],['week','주간']]){const b=btn(t,()=>{analyticsGrain=v;render();},'segment');b.setAttribute('aria-pressed',String(analyticsGrain===v));switcher.append(b);}activity.append(switcher);
    const chronological=groupCounts(attempts,e=>analyticsGrain==='week'?shiftDate(e.date,-((new Date(`${e.date}T12:00:00Z`).getUTCDay()+6)%7)):e.date).sort((a,b)=>a[0].localeCompare(b[0]));const activityChart=el('div','activity-trend');const max=Math.max(1,...chronological.map(x=>x[1].length));for(const [d,rows]of chronological){const b=btn('',()=>drill(`${d} ${analyticsGrain==='week'?'시작 주':'일'}`,rows),'trend-row');b.append(el('time','',d));const tracks=el('span','trend-tracks');for(const [cls,count]of [['attempted',rows.length],['completed',rows.filter(completion).length]]){const f=el('span',cls);f.style.width=`${count/max*100}%`;tracks.append(f);}b.append(tracks,el('span','',`${rows.length} / ${rows.filter(completion).length}`));b.title='시도 / 해결';activityChart.append(b);}activity.append(activityChart);if(!chronological.length)activity.append(empty('기록된 시도가 없어요.'));grid.append(activity);
    grid.append(bars('풀이 결과',groupCounts(attempts,outcomeText),'독립 / 도움 / 미해결 / 도움 미상 구분 · 클릭하여 원본 시도 확인'));
    const topics=groupCounts(attempts,e=>e.pattern || '주제 미기록');grid.append(bars('알고리즘 주제 기록',topics,'기록에 포함된 주제 조합별 시도 수 · 태그는 숙련도 점수가 아니에요.'));
    grid.append(bars('플랫폼별 난이도 분포',groupCounts(attempts,e=>`${e.platform || '미상'} · ${e.difficulty??'난이도 미기록'}`),'원래 플랫폼 척도를 유지해요. 같은 숫자도 플랫폼 간 비교할 수 없어요.'));
    grid.append(bars('플랫폼별 난이도의 월별 기록',groupCounts(attempts,e=>`${e.date.slice(0,7)} · ${e.platform || '미상'} · ${e.difficulty??'미기록'}`).sort((a,b)=>a[0].localeCompare(b[0])),'월별 원래 난이도의 시도 수 · 학습 효과나 플랫폼 간 난이도 상승을 뜻하지 않아요.'));
    grid.append(bars('독립·도움 해결의 월별 기록',groupCounts(attempts.filter(e=>['independent','assisted'].includes(e.outcome)),e=>`${e.date.slice(0,7)} · ${outcomeText(e)}`),'완료 보고와 미해결은 이 비교에서 제외해요. 원본 데이터로 드릴다운할 수 있어요.'));
    const scoped=events.filter(e=>e.date.startsWith(analyticsYear)&&(!analyticsPlatform||e.platform===analyticsPlatform)),cv=conversion(scoped),conversionCard=card('추천 이후 시도','계획일로부터 7일 이내 같은 문제의 시도 여부 · 계획일+문제별 중복 제거 · 인과 관계나 완료율이 아니에요.');conversionCard.append(btn(`${cv.matched} / ${cv.total}개 계획에 후속 시도`,()=>drill('추천과 후속 시도',cv.rows)),el('p','meta','최근 7일 계획은 관찰 기간이 끝나지 않았어요. 반복 계획에 같은 시도가 연결될 수 있어요.'));grid.append(conversionCard);root.append(grid);
    const usage=adapters.explanationUsageCard();usage.open=true;root.append(usage);root.append(el('p','meta','API 사용량은 전체 저장 기록 범위이며 위 연도·플랫폼 필터와 별개예요. 미기록 토큰·지연·비용은 0으로 채우지 않아요.'));
    const usageRows=getJournal()?.explanation_usage?.groups || [];if(usageRows.length)root.append(btn('API 집계 원본 보기',()=>{const c=el('pre','explanation-evidence',JSON.stringify(usageRows,null,2));openDialog('API 사용 집계',c,{wide:true});}));
  }
  let comparisonObserver=null;
  function similarity(){
    root.append(heading('문제 유사도 비교','기준 문제를 골라 풀이 설명과 문제 구조의 입력·후보를 직접 비교해 보세요.'));
    const panel=el('section','similarity-workspace');const status=el('p','meta','비교 화면을 불러오는 중이에요.');status.setAttribute('role','status');panel.append(status);root.append(panel);
    fetch('/api/local/comparison',{cache:'no-store'}).then(async response=>{if(!response.ok)throw Error('unavailable');return response.json();}).then(meta=>{
      if(!panel.isConnected)return;if(!meta.available)throw Error('unavailable');
      status.remove();const frame=el('iframe','similarity-frame');frame.title='문제별 임베딩·태그 유사도 비교';frame.src=meta.scopes.strategy_leetcode?'/comparison/?scope=strategy_leetcode':'/comparison/?scope=leetcode';frame.referrerPolicy='no-referrer';
      frame.addEventListener('load',()=>{if(!frame.isConnected)return;const body=frame.contentDocument?.body;if(!body)return;comparisonObserver?.disconnect();comparisonObserver=new ResizeObserver(()=>{if(frame.isConnected)frame.style.height=`${Math.max(760,body.scrollHeight+8)}px`;});comparisonObserver.observe(body);});panel.append(frame);
    }).catch(()=>{if(!panel.isConnected)return;status.textContent='지금은 비교 자료를 불러올 수 없어요. 비교 서비스가 준비되면 다시 시도해 주세요.';panel.append(btn('다시 불러오기',()=>render('similarity'),'button secondary'));});
  }
  function render(nextView=view){const focusKey=document.activeElement?.dataset?.focus;comparisonObserver?.disconnect();comparisonObserver=null;view=nextView;root.replaceChildren();if(view==='history')history();else if(view==='analytics')analytics();else if(view==='contests')renderContestList();else if(view==='similarity')similarity();else renderCalendar();if(focusKey)[...root.querySelectorAll('[data-focus]')].find(n=>n.dataset.focus===focusKey)?.focus({preventScroll:true});}
  document.addEventListener('keydown',event=>{if(document.querySelector('dialog[open]')||event.target.matches('input,textarea,select,[contenteditable=true]')||event.ctrlKey||event.altKey||event.metaKey)return;if(view!=='calendar')return;if(['j','k','t'].includes(event.key.toLowerCase())){event.preventDefault();if(event.key.toLowerCase()==='t')selected=cursor=localDate(new Date());else cursor=navigateDate(cursor,event.key.toLowerCase()==='j'?1:-1,mode);render();}});
  return {render,openDay:daily,openRecord:edit};
}
