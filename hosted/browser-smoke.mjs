/** Actual Chrome CDP smoke check. Requires isolated local service and headless Chrome. */
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
const base=process.env.SMOKE_URL ?? 'http://127.0.0.1:8787';
const debug=process.env.CHROME_DEBUG_URL ?? 'http://127.0.0.1:9227';
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
  await wait(`document.querySelector('#app')?.getAttribute('aria-busy')==='false' && document.querySelector('#attempt-problem')`);
  const identity=await evaluate(`(async()=>{return (await (await fetch('/api/state')).json()).learner_id})()`);
  assert.equal(identity,'browser-smoke','Smoke service must use the isolated browser-smoke learner');
  assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true,'mobile horizontal overflow');
  const labels=await evaluate(`Array.from(document.querySelectorAll('input:not([type=checkbox]),select')).map(el=>({id:el.id,label:!!document.querySelector('label[for="'+el.id+'"]')}))`);
  assert.ok(labels.every(x=>x.id&&x.label),'form controls must have labels');
  await evaluate(`(()=>{document.querySelector('#attempt-problem').selectedIndex=1;document.querySelector('#attempt-outcome').value='assisted';document.querySelector('#attempt-hints').checked=true;document.querySelector('#attempt-problem').closest('form').requestSubmit();})()`);
  await wait(`document.querySelector('#notice')?.textContent.includes('저장했어요')`);
  await evaluate(`document.querySelector('[data-view=progress]').click()`);
  assert.match(await evaluate('document.body.innerText'),/도움 받아 해결/);
  // Insert start/stop actions through the real same-origin HTTP API, then test browser recovery.
  const intervalId=await evaluate(`(async()=>{const state=await(await fetch('/api/state')).json();const id=crypto.randomUUID();const start=new Date(Date.now()-120000).toISOString(),stop=new Date().toISOString();for(const [kind,occurred,payload] of [['timer_started',start,{interval_id:id,problem_key:state.curriculum.problems[0].problem_key}],['timer_stopped',stop,{interval_id:id}]]){const r=await fetch('/api/events',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({schema_version:1,event_id:crypto.randomUUID(),learner_id:state.learner_id,occurred_at:occurred,recorded_at:new Date().toISOString(),kind,source:'manual',payload,supersedes:null})});if(!r.ok)throw Error(await r.text());}return id;})()`);
  await call('Page.reload');await wait(`document.querySelector('#app')?.getAttribute('aria-busy')==='false' && document.querySelector('.timer-confirm')`);
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
  await evaluate(`document.querySelector('[data-view=today]').click()`);
  assert.deepEqual(errors,[],'browser runtime exceptions');
  const image=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  await mkdir('private-learning/browser-evidence',{recursive:true,mode:0o700});
  await writeFile('private-learning/browser-evidence/mobile.png',Buffer.from(image.data,'base64'),{mode:0o600});
  await writeFile('private-learning/browser-evidence/receipt.json',JSON.stringify({at:new Date().toISOString(),browser:await call('Browser.getVersion'),viewport:[390,844],checks:['mobile layout','accessible form labels','manual attempt storage','history rendering','timer recovery and explicit confirmation','offline reload and stable-ID retry','learner-scoped pending writes','manual contest storage','participation missing-time preservation','full learning backup export'],errors},null,2),{mode:0o600});
  console.log('Actual Chrome smoke passed; private-learning/browser-evidence/mobile.png and receipt.json saved.');
}finally{socket.close();await fetch(debug+'/json/close/'+target.id);}
