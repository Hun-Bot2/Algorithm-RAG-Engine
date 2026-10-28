/** Calendar workspace Chrome/CDP regression using disposable synthetic data. No live provider requests. */
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp,mkdir,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {startLocal} from './local.mjs';
import {createJournalService} from './mastery/journal.mjs';
import {createExplanationService} from './mastery/explanations.mjs';
import {catalogFromData} from './mastery/catalog.mjs';

const temp=await mkdtemp(join(tmpdir(),'workspace-browser-'));
const output=resolve(process.env.SMOKE_OUTPUT_DIR ?? join(temp,'receipts'));await mkdir(output,{recursive:true,mode:0o700});
const chrome=spawn(process.env.CHROME_BINARY ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',[
  '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking',
  '--disable-sync','--disable-extensions','--remote-debugging-port=0',`--user-data-dir=${join(temp,'chrome')}`,'about:blank'],{stdio:'ignore'});
let chromeError=null;chrome.on('error',error=>{chromeError=error;});
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const receipts=[];
// Fixture dates follow the real clock so upcoming-only lists never expire (civil Asia/Seoul dates).
const kstDate=ms=>new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Seoul'}).format(new Date(ms));
const addDays=(d,n)=>{const x=new Date(`${d}T12:00:00Z`);x.setUTCDate(x.getUTCDate()+n);return x.toISOString().slice(0,10);};
const todayKst=kstDate(Date.now()),monday=addDays(todayKst,7-(new Date(`${todayKst}T12:00:00Z`).getUTCDay()+6)%7);
const WED=addDays(monday,2),FRI=addDays(monday,4),SUN=addDays(monday,6),YESTERDAY=addDays(todayKst,-1);
try{
 let debugPort;
 for(let i=0;i<100;i++){
  if(chromeError)throw chromeError;
  try{debugPort=(await readFile(join(temp,'chrome','DevToolsActivePort'),'utf8')).split('\n')[0];break;}catch{await delay(100);}
 }
 assert.ok(debugPort,'Chrome debug endpoint did not start');
 const debug=`http://127.0.0.1:${debugPort}`;
 for(const scenario of ['dual_success','partial_failure','both_failed']){
  const directory=join(temp,scenario);await mkdir(directory);
  const learner='browser-smoke',env={GEMINI_API_KEY:'fixture-key',NVIDIA_API_KEY:'fixture-key',NVIDIA_EXPLANATION_MODEL:'fixture/model'};
  const catalog=catalogFromData({version:'fixture-browser',problems:Array.from({length:3},(_,i)=>({key:`leetcode:easy-${i}`,platform:'leetcode',title:`Easy ${i}`,url:`https://leetcode.com/problems/easy-${i}/`,difficulty_value:1,difficulty_kind:'lc_level',variation_group:`v${i}`,concepts:[{id:'arrays-hashing',status:'approved'}]})),concepts:[{id:'arrays-hashing',label:'Arrays'}]});
  let contestCalls=0;
  const contestFetcher=async url=>{
    contestCalls++;
    if(url.includes('codeforces'))return Response.json({status:'OK',result:[{id:99001,name:'Synthetic Wednesday Round',phase:'BEFORE',startTimeSeconds:Date.parse(`${WED}T14:35:00Z`)/1000,durationSeconds:7200}]});
    if(url.includes('atcoder.jp'))return new Response(`<div id="contest-table-upcoming"><tbody><tr><td><time>${FRI} 21:00:00+0900</time></td><td><a href="/contests/abc999">Synthetic Friday ABC</a></td><td>01:40</td><td>- 1999</td></tr></tbody></div>`);
    return Response.json({schemaVersion:1,updatedAt:new Date().toISOString(),contests:[{platform:'leetcode',name:'Synthetic Sunday Weekly',url:'https://leetcode.com/contest/weekly-contest-999',startsAt:`${SUN}T02:30:00Z`,endsAt:`${SUN}T04:00:00Z`}]});
  };
  const service=await startLocal({port:0,database:join(directory,'learner.sqlite3'),learnerId:learner,curriculum:{schema_version:1,curriculum_version:'fixture-browser',problems:[],relationships:[]},catalogProvider:()=>catalog,autoGitSync:false,explanationEnv:env,contestFetcher});
  let socket;
  try{
   let calls=0;
   const explanations=createExplanationService({cacheRoot:join(directory,'explanations'),env,fetcher:async(url,options)=>{
    calls++;const body=JSON.parse(options.body),gemini=Boolean(body.contents);
    if(scenario==='both_failed'||scenario==='partial_failure'&&!gemini)return new Response('',{status:503});
    const packet=JSON.parse(gemini?body.contents[0].parts[0].text:body.messages[1].content);
    const result=JSON.stringify({text:`${gemini?'Gemini':'NVIDIA'}에서 작성한 합성 테스트 설명이에요. 현재 학습 단계에 따라 다음 문제를 골랐어요.`,evidence_id:packet.evidence_id});
    return Response.json(gemini?{candidates:[{content:{parts:[{text:result}]}}],usageMetadata:{promptTokenCount:10,candidatesTokenCount:20,thoughtsTokenCount:0,totalTokenCount:30}}:{choices:[{message:{content:result}}],usage:{prompt_tokens:10,completion_tokens:20,total_tokens:30}});
   }});
   const journal=createJournalService({db:service.db,catalogProvider:()=>catalog}),raw=await journal.view(learner);
   await explanations.decorate(learner,raw,{generate:true});const before=explanations.usage(learner);
   const target=await (await fetch(debug+'/json/new?'+encodeURIComponent(service.url),{method:'PUT'})).json();
   socket=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
   let seq=0;const pending=new Map(),errors=[];
   socket.addEventListener('message',event=>{const message=JSON.parse(event.data);if(message.id){const task=pending.get(message.id);pending.delete(message.id);message.error?task.reject(Error(message.error.message)):task.resolve(message.result);}if(message.method==='Runtime.exceptionThrown')errors.push(message.params.exceptionDetails.text);});
   const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});
   const evaluate=async expression=>{const result=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value;};
   await call('Runtime.enable');await call('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false});
   const wait=async expression=>{for(let i=0;i<100;i++){if(await evaluate(expression))return;await delay(80);}throw Error('Timed out: '+expression);};
   // The month view starts at today; page to the month that holds a relative fixture date.
   const showDate=async date=>{for(let i=0;i<3;i++){if(await evaluate(`Boolean(document.querySelector('[data-date="${date}"]'))`))return;await evaluate(`document.querySelector('.calendar-navigation [aria-label="${date>todayKst?'다음 달':'이전 달'}"]').click()`);}throw Error('Calendar never showed '+date);};
   await wait("document.querySelector('.calendar-grid') && document.querySelector('#app').getAttribute('aria-busy')==='false'");
   assert.equal(await evaluate("document.querySelector('[data-view=calendar]').getAttribute('aria-current')"),'page');
   assert.ok(await evaluate("document.querySelectorAll('.calendar-cell').length>=28"));
   await call('Emulation.setTimezoneOverride',{timezoneId:'America/Los_Angeles'});
   const todayKst=new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Seoul'}).format(new Date());
   assert.equal(await evaluate("document.querySelector('.calendar-cell.selected').dataset.date"),todayKst);
   assert.equal(await evaluate("document.querySelector('.calendar-cell.is-today').dataset.date"),todayKst);
   const previous=new Date(Date.parse(todayKst+'T12:00:00Z')-86400000).toISOString().slice(0,10);
   await evaluate(`document.querySelector('[data-date="${previous}"] .calendar-date').click()`);
   assert.equal(await evaluate("document.querySelectorAll('dialog[open]').length"),0);
   assert.equal(await evaluate("document.querySelector('.day-panel').dataset.date"),previous);
   assert.equal(await evaluate("document.querySelector('.calendar-cell.is-today').dataset.date"),todayKst);
   assert.equal(await evaluate("document.querySelector('.calendar-cell.is-today .calendar-date').getAttribute('aria-pressed')"),'false');
   await evaluate("[...document.querySelectorAll('.calendar-navigation button')].find(b=>b.textContent==='오늘').click()");

   assert.equal(await evaluate("document.querySelectorAll('.explanation-provider').length"),0,'explanations must stay off the main calendar');
   await writeFile(join(output,scenario+'-calendar-desktop.png'),Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
   await evaluate("document.querySelector('.calendar-event.plan').click()");
   await wait("document.querySelectorAll('.explanation-provider').length===2");
   assert.equal(await evaluate("document.querySelectorAll('.explanation-provider').length"),2);
   await evaluate("document.querySelectorAll('details').forEach(d=>d.open=true)");
   const layout=await evaluate("(()=>{const a=[...document.querySelectorAll('.explanation-provider')].map(x=>x.getBoundingClientRect());return a.map(x=>({x:x.x,y:x.y,width:x.width}));})()");
   assert.ok(layout[1].x>layout[0].x);assert.ok(Math.abs(layout[0].y-layout[1].y)<2);
   const content=await evaluate("document.querySelector('dialog .journal-plan').innerText");
   assert.ok(content.includes('Gemini'));assert.ok(content.includes('NVIDIA NIM'));assert.ok(content.includes('원래 추천 근거 보기'));
   if(scenario==='dual_success'){assert.ok(content.includes('Gemini에서 작성'));assert.ok(content.includes('NVIDIA에서 작성'));assert.ok(!content.includes('기본 설명 ·'));}
   if(scenario==='partial_failure'){assert.ok(content.includes('Gemini에서 작성'));assert.ok(content.includes('생성 실패'));assert.ok(!content.includes('기본 설명 ·'));}
   if(scenario==='both_failed'){assert.ok(content.includes('기본 설명 ·'));assert.ok(!content.includes('에서 작성한 합성'));}
   await writeFile(join(output,scenario+'-desktop.png'),Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
   await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
   const mobile=await evaluate("(()=>{const a=[...document.querySelectorAll('.explanation-provider')].map(x=>x.getBoundingClientRect());return {stacked:a[1].y>a[0].y,overflow:document.documentElement.scrollWidth>innerWidth};})()");
   assert.equal(mobile.stacked,true);assert.equal(mobile.overflow,false);
   await writeFile(join(output,scenario+'-comparison-mobile.png'),Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
   // Native modal: focus stays inside on Tab and Escape closes the single active dialog.
   await evaluate("document.querySelector('dialog .dialog-close').focus()");
   await call('Input.dispatchKeyEvent',{type:'keyDown',key:'Tab',code:'Tab',windowsVirtualKeyCode:9});
   assert.ok(await evaluate("!!document.activeElement.closest('dialog')"));
   await call('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
   await wait("!document.querySelector('dialog[open]')");
   await wait("document.activeElement.classList.contains('calendar-event')||document.activeElement.classList.contains('calendar-date')");
   for(const width of [1440,834,390]){
    await call('Emulation.setDeviceMetricsOverride',{width,height:900,deviceScaleFactor:1,mobile:width<600});
    for(const view of ['calendar','history','analytics','contests']){
     await evaluate(`document.querySelector('[data-view=${view}]').click()`);
     assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true,`${view} overflow at ${width}`);
     await writeFile(join(output,scenario+'-'+view+'-'+width+'.png'),Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
    }
   }
   if(scenario==='dual_success'){
    // Exercise an actual append and correction through UI forms and existing SQLite journal API.
    await evaluate("document.querySelector('[data-view=calendar]').click();[...document.querySelectorAll('.calendar-toolbar button')].find(b=>b.textContent.includes('풀이 기록')).click()");
    await evaluate("document.querySelector('#record-problem').value='leetcode:easy-0';document.querySelector('#record-date').value=new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Seoul'}).format(new Date(Date.now()-86400000));document.querySelector('#record-time').value='11:12:13.123';document.querySelector('#record-minutes').value='17m';document.querySelector('dialog form').requestSubmit()");
    await wait("!document.querySelector('dialog[open]')");
    let state=await evaluate("(async()=>await(await fetch('/api/state')).json())()");
    const original=state.events.find(e=>e.kind==='attempt');assert.ok(original);assert.equal(original.payload.self_minutes,17);assert.ok(original.occurred_at.includes('02:12:13.123'));
    await evaluate("document.querySelector('[data-view=history]').click();document.querySelector('.history-table-row').click();[...document.querySelectorAll('dialog button')].find(b=>b.textContent==='기록 수정').click()");
    await evaluate("if(document.querySelector('#record-hints'))throw Error('policy v3 forms must not ask about hints');document.querySelector('#record-outcome').value='failed';document.querySelector('#record-minutes').value='40s';document.querySelector('dialog form').requestSubmit()");
    await wait("!document.querySelector('dialog[open]')");
    state=await evaluate("(async()=>await(await fetch('/api/state')).json())()");
    assert.equal(state.events.filter(e=>e.kind==='attempt').length,2);const correction=state.events.find(e=>e.supersedes===original.event_id);assert.ok(correction);assert.equal(correction.occurred_at,original.occurred_at);assert.equal(correction.payload.self_minutes,null);assert.equal(correction.payload.self_seconds,40);assert.equal(correction.payload.outcome,'failed');assert.equal(correction.payload.hints,null);
    assert.equal(await evaluate("document.querySelectorAll('.history-table-row').length"),1);
    await evaluate("document.querySelector('.history-table-row').click()");assert.match(await evaluate("document.querySelector('dialog').innerText"),/수정 전/);
    await evaluate("document.querySelector('dialog').close();document.querySelector('.history-filters input[type=search]').value='absent fixture';document.querySelector('.history-filters input[type=search]').dispatchEvent(new Event('input'))");
    assert.equal(await evaluate("document.querySelectorAll('.history-table-row').length"),0);
    await evaluate("document.querySelector('[data-view=analytics]').click();document.querySelector('.heat-cell[data-level=\"1\"]').click()");assert.match(await evaluate("document.querySelector('dialog').innerText"),/1개 원본/);
    await evaluate("document.querySelector('dialog').close();document.querySelector('[data-view=contests]').click();[...document.querySelectorAll('#app button')].find(b=>b.textContent.includes('대회 직접 추가')).click();document.querySelector('#contest-title').value='Synthetic LeetCode fixture';document.querySelector('#contest-platform').value='leetcode';document.querySelector('#contest-url').value='https://leetcode.com/contest/';document.querySelector('#contest-start').value=new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Seoul'}).format(new Date(Date.now()+86400000))+'T20:00';document.querySelector('#contest-duration').value='90';document.querySelector('dialog form').requestSubmit()");
    await wait("document.querySelector('#notice').innerText.includes('대회 일정을 저장했어요')");
    await evaluate("document.querySelector('dialog')?.close()");
    const contests=await evaluate("(async()=>await(await fetch('/api/state')).json())()");assert.equal(contests.contests[0].platform,'leetcode');assert.equal(contests.contests[0].source,'manual');
    await evaluate("document.querySelector('.contest-list-item').click()");assert.match(await evaluate("document.querySelector('dialog .contest-source').textContent"),/직접 입력/);
    // Contests show participation info, not a suitability verdict.
    assert.equal(await evaluate("Boolean(document.querySelector('dialog .contest-fit'))"),false);assert.match(await evaluate("document.querySelector('dialog').innerText"),/참여 기록 남기기/);
    await evaluate("[...document.querySelectorAll('dialog button')].find(b=>b.textContent==='캘린더에서 날짜 보기').click()");
    assert.equal(await evaluate("document.querySelectorAll('.calendar-event.contest').length"),1,'manual contest retained in primary calendar');
    assert.equal(await evaluate("document.querySelector('[data-view=calendar]').getAttribute('aria-current')"),'page');
    await evaluate("document.querySelector('.calendar-event.contest').click()");

    await evaluate("document.querySelector('dialog').close()");

    for(const [i,platform]of ['codeforces','atcoder'].entries()){
      const fetched=new Date(Date.now()-(i?86400000:0)).toISOString();
      const item={id:platform+':fixture',platform,title:'Synthetic '+platform+' contest',start_at:new Date(Date.now()+(i+1)*86400000).toISOString(),duration_minutes:120,url:platform==='codeforces'?'https://codeforces.com/contests':'https://atcoder.jp/contests/',source:'synthetic-cache-fixture',fetched_at:fetched};
      const envelope={status:i?'stale':'fresh',fetched_at:fetched,error:i?'synthetic_outage':null,items:[item]};
      await service.db.prepare('INSERT INTO provider_cache(learner_id,provider_key,payload) VALUES (?,?,?)').bind(learner,platform+':calendar',JSON.stringify(envelope)).run();
    }
    await call('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false});
    // Real UI sync path; all upstream responses are synthetic fixtures.
    await evaluate("document.querySelector('[data-view=contests]').click();[...document.querySelectorAll('button')].find(b=>b.textContent==='공개 일정 동기화').click()");
    await wait("document.querySelectorAll('.contest-list-item').length===4");assert.equal(contestCalls,3);
    await evaluate("[...document.querySelectorAll('.contest-list-item')].find(b=>b.textContent.includes('Synthetic Wednesday')).click()");
    const contestText=await evaluate("document.querySelector('dialog').textContent");assert.match(contestText,/23:35/);assert.match(contestText,/120분/);assert.doesNotMatch(contestText,/판단 근거 부족/);assert.match(contestText,/마지막 동기화/);
    await writeFile(join(output,'contest-details-desktop.png'),Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
    await evaluate("document.querySelector('dialog').close();document.querySelector('[data-view=calendar]').click()");await showDate(WED);
    assert.ok(await evaluate(`[...document.querySelectorAll('[data-date="${WED}"] .calendar-event.contest')].some(b=>b.title.includes('Synthetic Wednesday'))`));
    await showDate(FRI);assert.ok(await evaluate(`[...document.querySelectorAll('[data-date="${FRI}"] .calendar-event.contest')].some(b=>b.title.includes('Synthetic Friday'))`));
    assert.equal(await evaluate(`document.querySelector('[data-date="${WED}"]').classList.contains('weekend')`),false);
    await showDate(SUN);assert.equal(await evaluate(`document.querySelector('[data-date="${SUN}"]').classList.contains('weekend')`),true);
    await writeFile(join(output,'contest-weekday-weekend-desktop.png'),Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
    // An attempt at 00:00 KST yesterday is 15:00Z the day before; it must land on yesterday's KST date.
    await journal.attempt(learner,{event_id:'kst-midnight-proof',problem_key:'leetcode:easy-1',outcome:'independent',hints:false,occurred_at:`${addDays(YESTERDAY,-1)}T15:00:00.000Z`,activity_type:'practice'});
    await call('Page.reload');await wait("document.querySelector('.calendar-grid')");await showDate(YESTERDAY);await wait(`document.querySelector('[data-date="${YESTERDAY}"] .calendar-event.actual')`);
    await evaluate(`document.querySelector('[data-date="${YESTERDAY}"] .calendar-event.actual').click()`);
    // Two or more attempts collapse into one summary chip that opens the day; open the midnight attempt from there.
    await evaluate("[...document.querySelectorAll('dialog .done-chip')].find(b=>b.textContent.includes('Easy 1'))?.click()");
    assert.match(await evaluate("document.querySelector('dialog').innerText"),new RegExp(`${Number(YESTERDAY.slice(0,4))}\\. ${Number(YESTERDAY.slice(5,7))}\\. ${Number(YESTERDAY.slice(8))}\\. 00:00`));
    await evaluate("document.querySelector('dialog').close()");
    const staleRow=await service.db.prepare('SELECT payload FROM provider_cache WHERE learner_id=? AND provider_key=?').bind(learner,'atcoder:discovery').first();
    const staleEnvelope=JSON.parse(staleRow.payload);staleEnvelope.status='stale';staleEnvelope.fetched_at=new Date(Date.now()-86400000).toISOString();staleEnvelope.error='synthetic_outage';
    await service.db.prepare('UPDATE provider_cache SET payload=? WHERE learner_id=? AND provider_key=?').bind(JSON.stringify(staleEnvelope),learner,'atcoder:discovery').run();
    // Populate only this disposable fixture for visual review of history density and analytical marks.
    for(let i=1;i<=14;i++)await journal.attempt(learner,{event_id:`visual-fixture-${i}`,problem_key:`leetcode:easy-${i%3}`,outcome:i%4===0?'failed':i%3===0?'assisted':'independent',hints:i%4===0?null:i%3===0,occurred_at:new Date(Date.now()-i*86400000).toISOString(),self_minutes:i%5===0?null:10+i,activity_type:'practice'});
    for(let i=0;i<4;i++)await journal.attempt(learner,{event_id:`dense-fixture-${i}`,problem_key:`leetcode:easy-${i%3}`,outcome:i===3?'failed':'independent',hints:i===3?null:false,occurred_at:new Date(Date.now()-(i+1)*60000).toISOString(),self_minutes:null,activity_type:'practice'});
    await call('Page.reload');await wait("!!document.querySelector('.calendar-grid')");
    // Main grid and selected panel follow all four disjoint filters.
    await evaluate("document.querySelector('.layer-toggle.solved input').click()");
    assert.equal(await evaluate("document.querySelectorAll('.calendar-event.solved').length"),0);
    assert.ok(await evaluate("document.querySelectorAll('.calendar-event.attempted').length>0"));
    await evaluate("document.querySelector('.layer-toggle.solved input').click()");
    assert.ok(await evaluate("document.querySelectorAll('.calendar-event.solved').length>0"));
    await evaluate("document.querySelector('.layer-toggle.planned input').click()");
    assert.equal(await evaluate("document.querySelectorAll('.calendar-event.plan').length"),0);
    await evaluate("document.querySelector('.layer-toggle.planned input').click()");
    // Cells show at most three unfinished plans/schedules plus one attempt chip; several attempts collapse into a summary.
    assert.ok(await evaluate("[...document.querySelectorAll('.calendar-cell')].every(c=>c.querySelectorAll('.calendar-event:not(.actual)').length<=3&&c.querySelectorAll('.calendar-event.actual').length<=1)"));
    const opener="document.querySelector('.more-events') || document.querySelector('.calendar-event.summary')";
    const moreDate=await evaluate(`(${opener}).closest('.calendar-cell').dataset.date`);
    await evaluate(`(${opener}).click()`);
    assert.equal(await evaluate("document.querySelectorAll('dialog[open]').length"),1);
    assert.equal(await evaluate("document.querySelector('.day-panel').dataset.date"),moreDate);
    await evaluate("document.querySelector('dialog').close()");
    for(const width of [1440,834,390]){
      await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<600});
      for(const view of ['calendar','history','analytics','contests']){
        await evaluate(`document.querySelector('[data-view=${view}]').click()`);
        assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true,`populated ${view} overflow at ${width}`);
        await writeFile(join(output,`populated-${view}-${width}.png`),Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
        if(view==='calendar'){
          const previousDate=new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Seoul'}).format(new Date(Date.now()-86400000));
          await evaluate(`document.querySelector('.calendar-cell[data-date="${previousDate}"] .calendar-date').click()`);
          await writeFile(join(output,`selected-day-${width}.png`),Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
          await evaluate("document.querySelector('.monthly-insights').scrollIntoView({block:'start'})");
          await writeFile(join(output,`monthly-insights-${width}.png`),Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
          assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true);
          await evaluate("window.scrollTo({top:0,behavior:'instant'})");
        }

      }
    }

    await evaluate("document.querySelector('[data-view=contests]').click()");
    assert.equal(await evaluate("document.querySelectorAll('.calendar-board').length"),0,'contest destination is a list');
    assert.equal(await evaluate("document.querySelectorAll('.contest-list-item').length"),4);
    assert.doesNotMatch(await evaluate("document.querySelector('.contest-agenda').innerText"),/오래된 캐시|판단 근거/,'no cache or verdict badges on contests');
    await evaluate("document.querySelector('[aria-label=\"대회 플랫폼\"]').value='atcoder';document.querySelector('[aria-label=\"대회 플랫폼\"]').dispatchEvent(new Event('change'))");
    assert.equal(await evaluate("document.querySelectorAll('.contest-list-item').length"),1);
    await evaluate("document.querySelector('.contest-list-item').click()");
    assert.match(await evaluate("document.querySelector('dialog .contest-source').textContent"),/이전에 받아 둔 일정/,'stale schedules are explained under 일정 출처');
    await evaluate("[...document.querySelectorAll('dialog button')].find(b=>b.textContent==='캘린더에서 날짜 보기').click()");
    const cachedContests=await evaluate("(async()=>(await(await fetch('/api/state')).json()).contests)()");
    for(const contest of cachedContests){
      const date=new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Seoul'}).format(new Date(contest.start_at));
      await evaluate(`document.querySelector('.calendar-cell[data-date="${date}"] .calendar-date').click()`);
      assert.ok((await evaluate("document.querySelector('.day-panel .day-section.schedule').innerText")).includes(contest.title),'cached/manual contest remains on its real calendar date even when chips overflow');
    }
    // Month heatmap drills into the day while preserving a single dialog and date context.
    // Relative contest fixtures can sit in next month; return to the month that holds the fixture attempts.
    await evaluate("[...document.querySelectorAll('.calendar-navigation button')].find(b=>b.textContent==='오늘').click()");
    await evaluate("document.querySelector('.month-heat-day:not([disabled])').click()");
    assert.equal(await evaluate("document.querySelectorAll('dialog[open]').length"),1);
    await evaluate("document.querySelector('dialog').close()");
    await evaluate("document.querySelector('[data-view=calendar]').click();[...document.querySelectorAll('.segment')].find(b=>b.textContent==='주').click()");
    assert.equal(await evaluate("document.querySelectorAll('.calendar-cell').length"),7);
    await evaluate("document.querySelector('.calendar-date').click()");
    assert.equal(await evaluate("document.querySelectorAll('dialog[open]').length"),0,'date click selects without opening a dialog');
    await evaluate("document.querySelector('.calendar-cell.selected .more-events, .calendar-cell.selected .calendar-event.summary')?.click()");if(await evaluate("Boolean(document.querySelector('dialog'))"))assert.match(await evaluate("document.querySelector('dialog').innerText"),/풀이 기록|풀어야 할 문제|남은 계획|일정/);
    await evaluate("document.querySelector('dialog')?.close();[...document.querySelectorAll('.segment')].find(b=>b.textContent==='월').click()");
    await call('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});
    assert.equal(await evaluate("getComputedStyle(document.querySelector('button')).transitionDuration"),'0s');
    await evaluate("window.dispatchEvent(new Event('offline'))");assert.match(await evaluate("document.querySelector('#notice').innerText"),/오프라인/);
    await evaluate("[...document.querySelectorAll('.calendar-toolbar button')].find(b=>b.textContent.includes('풀이 기록')).click();document.querySelector('#record-problem').value='leetcode:easy-0';document.querySelector('#record-date').value=new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Seoul'}).format(new Date(Date.now()-86400000));");
    await call('Network.enable');await call('Network.emulateNetworkConditions',{offline:true,latency:0,downloadThroughput:-1,uploadThroughput:-1});
    await evaluate("document.querySelector('dialog form').requestSubmit()");await wait("document.querySelector('dialog .form-error')?.textContent.length>0");
    const pendingBefore=await evaluate("JSON.parse(localStorage.getItem('adaptive-learning.pending-journal-attempts.v1:browser-smoke'))");assert.equal(pendingBefore.length,1);
    await call('Network.emulateNetworkConditions',{offline:false,latency:0,downloadThroughput:-1,uploadThroughput:-1});
    await evaluate("document.querySelector('dialog form').requestSubmit()");await wait("!document.querySelector('dialog[open]')");
    const retried=await evaluate("(async()=>await(await fetch('/api/state')).json())()");assert.equal(retried.events.filter(e=>e.event_id===pendingBefore[0].event_id).length,1);
    assert.deepEqual(await evaluate("JSON.parse(localStorage.getItem('adaptive-learning.pending-journal-attempts.v1:browser-smoke'))"),[]);




   }
   await call('Page.reload');await delay(300);
   assert.deepEqual(explanations.usage(learner),before);assert.equal(calls,2);assert.deepEqual(errors,[]);
   receipts.push({scenario,desktop_side_by_side:true,mobile_stacked:true,no_overflow:true,cache_view_no_new_api_attempts:true,provider_calls:calls,real_providers:false,responsive_widths:[1440,834,390],record_edit_and_offline_retry:scenario==='dual_success',contest_sync_and_kst_dates:scenario==='dual_success',contest_fixture_calls:contestCalls});
   await fetch(debug+'/json/close/'+target.id);
  }finally{socket?.close();await service.close();}
 }
 await writeFile(join(output,'receipt.json'),JSON.stringify({synthetic:true,receipts},null,2),{mode:0o600});
 console.log(JSON.stringify({output,scenarios:receipts.length,passed:true}));
}finally{
 chrome.kill('SIGTERM');await Promise.race([new Promise(resolve=>chrome.once('exit',resolve)),delay(2000)]);
 // Preserve screenshot receipts, remove only our disposable process profile/databases.
 if(!output.startsWith(temp+'/'))await rm(temp,{recursive:true,force:true});
}
