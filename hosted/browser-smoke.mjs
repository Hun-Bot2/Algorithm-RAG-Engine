/** Actual Chrome CDP smoke check. Requires isolated local service and headless Chrome. */
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
const base=process.env.SMOKE_URL ?? 'http://127.0.0.1:8787';
const debug=process.env.CHROME_DEBUG_URL ?? 'http://127.0.0.1:9227';
const outputDirectory=process.env.SMOKE_OUTPUT_DIR ?? 'private-learning/browser-evidence';
const target=await (await fetch(debug+'/json/new?'+encodeURIComponent(base),{method:'PUT'})).json();
const socket=new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
let sequence=0;const pending=new Map(),errors=[];
socket.addEventListener('message',event=>{
  const message=JSON.parse(event.data);
  if(message.id){const task=pending.get(message.id);pending.delete(message.id);message.error?task.reject(Error(message.error.message)):task.resolve(message.result);}
  if(message.method==='Runtime.exceptionThrown')errors.push(message.params.exceptionDetails.text);
});
const call=(method,params={})=>new Promise((resolve,reject)=>{
  const id=++sequence;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));
});
const evaluate=async(expression)=>{
  const value=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});
  if(value.exceptionDetails)throw Error(JSON.stringify(value.exceptionDetails));
  return value.result.value;
};
const wait=async expression=>{
  for(let i=0;i<100;i++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,50));}
  throw Error('Timed out: '+expression);
};
try{
  await call('Runtime.enable');await call('Page.enable');
  await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await call('Page.navigate',{url:base});
  await wait(`document.querySelector('#app')?.getAttribute('aria-busy')==='false' && document.querySelector('[data-view="journal"]')`);
  assert.equal(await evaluate(`document.querySelector('[data-view="journal"]')?.getAttribute('aria-current')`),'page','journal should open by default');
  assert.match(await evaluate('document.body.innerText'),/학습 일지/);
  const identity=await evaluate(`(async()=>{return (await (await fetch('/api/state')).json()).learner_id})()`);
  assert.equal(identity,'browser-smoke','Smoke service must use the isolated browser-smoke learner');
  assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true,'journal mobile horizontal overflow');
  const journalAvailable=await evaluate(`!!document.querySelector('.journal-filters')`);
  assert.equal(journalAvailable,true,'smoke service must provide a catalog-backed journal');
  let journalAttemptSaved=false,journalOXPassed=false,journalFilterPassed=false,journalRecommendationChanged=false,journalTimelineVisible=false;
  let journalScreenshot=null;
  if(journalAvailable){
    assert.ok(await evaluate(`!!document.querySelector('.journal-plan')`),'available journal should show today plan');
    assert.ok(await evaluate(`!!document.querySelector('#journal-from') && !!document.querySelector('#journal-priority')`),'journal facets should be available');
    assert.ok(await evaluate(`document.querySelector('.journal-plan').compareDocumentPosition(document.querySelector('.journal-filters'))&Node.DOCUMENT_POSITION_FOLLOWING`),'plan and focus should precede the main journal list');
    assert.ok(await evaluate(`document.querySelector('.journal-filters').compareDocumentPosition(document.querySelector('.journal-pattern-history'))&Node.DOCUMENT_POSITION_FOLLOWING`),'journal list should precede the expanded history sections');
    assert.equal(await evaluate(`document.querySelector('.journal-filter-details')?.open`),false,'journal filter controls should start collapsed');
    const initialJournal=await evaluate(`(async()=>await(await fetch('/api/journal')).json())()`);
    assert.equal(initialJournal.available,true);
    const initialPractice=initialJournal.plan.slots.find(slot=>slot.kind!=='review'&&slot.problem_key);
    assert.ok(initialPractice,'fresh catalog-backed learner should receive a practice candidate');
    assert.ok(await evaluate(`!!Array.from(document.querySelectorAll('.journal-attempt-details')).find(el=>el.dataset.slotKind!=='review')`),'journal candidate should expose an attempt form');
    {
      await evaluate(`document.querySelector('#notice').hidden=true`);
      await evaluate(`(()=>{const details=Array.from(document.querySelectorAll('.journal-attempt-details')).find(el=>el.dataset.slotKind!=='review');const form=details.querySelector('form');const outcome=form.querySelector('select[id^="journal-outcome-"]');const hints=form.querySelector('select[id^="journal-hints-"]');const minutes=form.querySelector('input[id^="journal-minutes-"]');if(!Array.from(outcome.options).some(x=>x.textContent.startsWith('O'))||!Array.from(outcome.options).some(x=>x.textContent.startsWith('X')))throw Error('journal attempt form must offer O and X');outcome.value='independent';hints.value='false';minutes.value='17';form.requestSubmit();})()`);
      await wait(`document.querySelector('#notice') && !document.querySelector('#notice').hidden && document.querySelector('#notice').textContent.includes('풀이 기록을 저장했어요')`);
      journalAttemptSaved=true;
      const afterO=await evaluate(`(async()=>await(await fetch('/api/journal')).json())()`);
      assert.ok(afterO.days.flatMap(day=>day.entries).some(entry=>entry.type==='attempt'&&entry.display_outcome==='O'),'independent journal attempt should appear as O');
      const nextAfterO=afterO.plan.slots.find(slot=>slot.kind!=='review'&&slot.problem_key);
      journalRecommendationChanged=Boolean(nextAfterO&&nextAfterO.problem_key!==initialPractice.problem_key);
      assert.ok(journalRecommendationChanged,'recommendation should advance after the recorded O');
      const hasPracticeForm=await evaluate(`!!Array.from(document.querySelectorAll('.journal-attempt-details')).find(el=>el.dataset.slotKind!=='review')`);
      assert.ok(hasPracticeForm,'updated recommendation should expose an attempt form');
      {
        await evaluate(`document.querySelector('#notice').hidden=true`);
        await evaluate(`(()=>{const details=Array.from(document.querySelectorAll('.journal-attempt-details')).find(el=>el.dataset.slotKind!=='review');const form=details.querySelector('form');form.querySelector('select[id^="journal-outcome-"]').value='failed';form.querySelector('select[id^="journal-hints-"]').value='true';form.querySelector('input[id^="journal-minutes-"]').value='12';form.requestSubmit();})()`);
        await wait(`document.querySelector('#notice') && !document.querySelector('#notice').hidden && document.querySelector('#notice').textContent.includes('풀이 기록을 저장했어요')`);
        const afterX=await evaluate(`(async()=>await(await fetch('/api/journal')).json())()`);
        const marks=afterX.days.flatMap(day=>day.entries).filter(entry=>entry.type==='attempt').map(entry=>entry.display_outcome);
        assert.ok(marks.includes('O')&&marks.includes('X'),'journal history should preserve both O and X');
        journalOXPassed=true;
        await evaluate(`(()=>{const date=${JSON.stringify(afterX.date)};for(const id of ['journal-from','journal-to']){const el=document.getElementById(id);el.value=date;el.dispatchEvent(new Event('input',{bubbles:true}));}})()`);
        journalFilterPassed=await evaluate(`document.querySelectorAll('.journal-day').length===1 && document.querySelector('.journal-day h3')?.textContent.includes(${JSON.stringify(afterX.date.slice(-2))})`);
        assert.ok(journalFilterPassed,'date filter should show one matching date group');
        await evaluate(`document.querySelector('#notice').hidden=true`);
        journalTimelineVisible=await evaluate(`document.querySelector('.journal-day')?.getBoundingClientRect().top < innerHeight`);
        assert.ok(journalTimelineVisible,'date-grouped journal should be visible in the mobile first viewport');
        assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true,'journal history mobile horizontal overflow');
      }
      journalScreenshot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
    }
  }
  await evaluate(`document.querySelector('[data-view=today]').click()`);
  await wait(`document.querySelector('#app')?.getAttribute('aria-busy')==='false' && document.querySelector('#attempt-problem')`);
  const labels=await evaluate(`Array.from(document.querySelectorAll('input:not([type=checkbox]),select')).map(el=>({id:el.id,label:!!document.querySelector('label[for="'+el.id+'"]')}))`);
  assert.ok(labels.every(x=>x.id&&x.label),'form controls must have labels');
  await evaluate(`document.querySelector('#notice').hidden=true`);
  await evaluate(`(()=>{document.querySelector('#attempt-problem').selectedIndex=1;document.querySelector('#attempt-outcome').value='assisted';document.querySelector('#attempt-hints').checked=true;document.querySelector('#attempt-problem').closest('form').requestSubmit();})()`);
  await wait(`document.querySelector('#notice') && !document.querySelector('#notice').hidden && document.querySelector('#notice').textContent.includes('기록을 저장했어요')`);
  await evaluate(`document.querySelector('[data-view=progress]').click()`);
  assert.match(await evaluate('document.body.innerText'),/도움 받아 해결/);
  // Insert start/stop actions through the real same-origin HTTP API, then test browser recovery.
  const intervalId=await evaluate(`(async()=>{const state=await(await fetch('/api/state')).json();const id=crypto.randomUUID();const start=new Date(Date.now()-120000).toISOString(),stop=new Date().toISOString();for(const [kind,occurred,payload] of [['timer_started',start,{interval_id:id,problem_key:state.curriculum.problems[0].problem_key}],['timer_stopped',stop,{interval_id:id}]]){const r=await fetch('/api/events',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({schema_version:1,event_id:crypto.randomUUID(),learner_id:state.learner_id,occurred_at:occurred,recorded_at:new Date().toISOString(),kind,source:'manual',payload,supersedes:null})});if(!r.ok)throw Error(await r.text());}return id;})()`);
  await call('Page.reload');await wait(`document.querySelector('#app')?.getAttribute('aria-busy')==='false' && document.querySelector('[data-view="journal"]')`);
  await evaluate(`document.querySelector('[data-view=today]').click()`);
  await wait(`!!document.querySelector('.timer-confirm')`);
  assert.match(await evaluate('document.body.innerText'),/時間確認|時間|시간 확인 필요/);
  await evaluate(`(async()=>{const state=await(await fetch('/api/state')).json();const timer=state.timers.intervals.find(t=>t.interval_id===${JSON.stringify(intervalId)});const date=new Date(timer.stopped_at);const value=new Date(date.getTime()-date.getTimezoneOffset()*60000).toISOString().slice(0,23);const end=document.getElementById('timer-end-'+timer.interval_id);end.value=value;end.closest('form').requestSubmit();})()`);
  await wait(`document.querySelector('#notice')?.textContent.includes('저장했어요')`);
  assert.equal(await evaluate(`(async()=>{const state=await(await fetch('/api/state')).json();return state.timers.intervals.find(t=>t.interval_id===${JSON.stringify(intervalId)}).status;})()`),'confirmed');
  // Intentionally fail one write, reload, and confirm the original ID is retried once.
  await call('Network.enable');
  await call('Network.setBlockedURLs',{urls:[base+'/api/events']});
  await evaluate(`(()=>{document.querySelector('#attempt-problem').selectedIndex=1;document.querySelector('#attempt-outcome').value='assisted';document.querySelector('#attempt-hints').checked=true;document.querySelector('#attempt-problem').closest('form').requestSubmit();})()`);
  await wait(`!!document.querySelector('[aria-label="전송 대기 중인 기록"]')`);
  const queuedId=await evaluate(`JSON.parse(localStorage.getItem('adaptive-learning.pending-events.v1:browser-smoke'))[0].event_id`);
  await call('Network.setBlockedURLs',{urls:[]});
  await evaluate(`localStorage.setItem('adaptive-learning.pending-events.v1:other-owner',JSON.stringify([{event_id:'other-private-id',learner_id:'other-owner',kind:'attempt',payload:{problem_key:'private-other-history'}}]))`);
  await call('Page.reload');await wait(`!!document.querySelector('[aria-label="전송 대기 중인 기록"]')`);
  assert.equal((await evaluate('document.body.innerText')).includes('private-other-history'),false);
  assert.equal(await evaluate(`JSON.parse(localStorage.getItem('adaptive-learning.pending-events.v1:browser-smoke'))[0].event_id`),queuedId);
  await evaluate(`Array.from(document.querySelectorAll('button')).find(b=>b.textContent==='대기 중인 기록 다시 보내기').click()`);
  await wait(`!document.querySelector('[aria-label="전송 대기 중인 기록"]')`);
  assert.equal(await evaluate(`(async()=>{return (await(await fetch('/api/state')).json()).events.filter(e=>e.event_id===${JSON.stringify(queuedId)}).length;})()`),1);

  await evaluate(`document.querySelector('[data-view=contests]').click()`);
  await wait(`!!document.querySelector('#contest-title')`);
  await evaluate(`(()=>{document.querySelector('#contest-title').value='Browser smoke contest';document.querySelector('#contest-url').value='https://atcoder.jp/contests/abc001';document.querySelector('#contest-start').value='2026-12-01T12:00';document.querySelector('#contest-duration').value='100';document.querySelector('#contest-title').closest('form').requestSubmit();})()`);
  await wait(`document.querySelector('#notice')?.textContent.includes('추가') || document.body.innerText.includes('Browser smoke contest')`);
  await evaluate(`(()=>{document.querySelector('#participation-contest-id').value='browser-smoke-contest';document.querySelector('#participation-mode').value='virtual';document.querySelector('#participation-outcome').value='partial';document.querySelector('#participation-occurred').value='2026-10-01T12:00';document.querySelector('#participation-minutes').value='';document.querySelector('#participation-contest-id').closest('form').requestSubmit();})()`);
  await wait(`document.querySelector('#notice')?.textContent.includes('참여 기록을 저장')`);
  assert.match(await evaluate('document.body.innerText'),/진행 시간 미기록/);
  const backup=await evaluate(`(async()=>{return await(await fetch('/api/backup/export')).json();})()`);
  assert.equal(backup.format,'adaptive-learning-backup');assert.ok(backup.participation.length>0);
  assert.deepEqual(errors,[],'browser runtime exceptions');
  await mkdir(outputDirectory,{recursive:true,mode:0o700});
  if(journalScreenshot) await writeFile(join(outputDirectory,'journal.png'),Buffer.from(journalScreenshot.data,'base64'),{mode:0o600,flag:'wx'});
  await writeFile(join(outputDirectory,'receipt.json'),JSON.stringify({at:new Date().toISOString(),browser:await call('Browser.getVersion'),viewport:[390,844],journalAvailable,journalAttemptSaved,journalOXPassed,journalFilterPassed,journalRecommendationChanged,journalTimelineVisible,checks:['journal default and current plan or explicit unavailable state','O/X journal attempt controls and durable history when a supported candidate exists','recommendation progression after journal attempt','date-group filtering','first-viewport date-group visibility','journal mobile overflow','accessible form labels','manual attempt storage','history rendering','timer recovery and explicit confirmation','offline reload and stable-ID retry','learner-scoped pending writes','manual contest storage','participation missing-time preservation','full learning backup export'],errors},null,2),{mode:0o600,flag:'wx'});
  console.log(`Actual Chrome smoke passed; evidence saved under ${outputDirectory}.`);
}finally{socket.close();await fetch(debug+'/json/close/'+target.id);}
