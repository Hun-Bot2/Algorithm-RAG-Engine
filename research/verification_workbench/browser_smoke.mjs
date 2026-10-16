// Requires local Chrome CDP :9223, real read-only :8765 and synthetic fixture :8766.
import fs from 'node:fs/promises';
const targets=await (await fetch('http://127.0.0.1:9223/json')).json();
const ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
let seq=0;const pending=new Map();const errors=[];
ws.onmessage=event=>{const m=JSON.parse(event.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text)};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(r.exceptionDetails.text);return r.result.value};
const wait=async expression=>{for(let i=0;i<100;i++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,100))}throw Error('DOM condition timed out: '+expression)};
await call('Runtime.enable');await call('Page.enable');
const out='private-learning/verification-workbench/browser-smoke';await fs.mkdir(out,{recursive:true});
async function screenshot(name){const r=await call('Page.captureScreenshot',{format:'png'});await fs.writeFile(`${out}/${name}.png`,Buffer.from(r.data,'base64'))}
await call('Emulation.setDeviceMetricsOverride',{width:1360,height:1100,deviceScaleFactor:1,mobile:false});
await call('Page.navigate',{url:'http://127.0.0.1:8765/'});
await wait("document.querySelectorAll('.queue-item').length===40 && document.getElementById('query-statement').textContent.length>100 && document.getElementById('target-statement').textContent.length>100 && document.getElementById('query-source').textContent.includes('completeness unverified')");
await screenshot('real-pilot-desktop');
await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
await screenshot('real-pilot-mobile');
const mobileOverflow=await evaluate('document.documentElement.scrollWidth>window.innerWidth');
if(mobileOverflow)throw Error('Mobile viewport overflows horizontally');
await call('Emulation.setDeviceMetricsOverride',{width:1360,height:1100,deviceScaleFactor:1,mobile:false});
await call('Page.navigate',{url:'http://127.0.0.1:8766/'});
await wait("document.querySelectorAll('.queue-item').length===2");
async function reviewer(name){await evaluate(`document.getElementById('role').value='human';document.getElementById('role').dispatchEvent(new Event('change'));document.getElementById('reviewer').value=${JSON.stringify(name)};document.getElementById('reviewer').dispatchEvent(new Event('change'));`);await new Promise(r=>setTimeout(r,250))}
async function saveRelationship(name){
 await reviewer(name);
 const clear=await evaluate("document.querySelector('#relationship-form [name=similarity]').value===''");if(!clear)throw Error('Reviewer form leaked previous selection');
 await evaluate(`(()=>{const f=document.getElementById('relationship-form');for(const [k,v] of Object.entries({similarity:'2',prerequisite:'unknown',transfer:'unknown',difficulty:'unknown',rationale:'Synthetic browser fixture rationale only.',evidence:'Synthetic fixture evidence; not research labels.'}))f.elements[k].value=v;f.elements.independent.checked=true;f.requestSubmit()})()`);
 await wait("document.getElementById('save-status').textContent==='Record saved.'");
}
await saveRelationship('fixture-browser-human-1');
await reviewer('fixture-browser-human-2');
if(!await evaluate("document.querySelectorAll('#records .record-card').length===0"))throw Error('Other reviewer labels exposed');
await saveRelationship('fixture-browser-human-2');
await reviewer('fixture-browser-adjudicator');
await evaluate("document.getElementById('reveal-reviews').click()");
await wait("document.getElementById('reveal-reviews').textContent==='Reviews revealed'");
const read=await evaluate(`(async()=>{const p=await(await fetch('/api/pilot')).json();return await(await fetch('/api/pair?id='+p.pilot.items[0].pair_id+'&reviewer=fixture-browser-adjudicator&mode=adjudicate')).json()})()`);
const ids=read.records.filter(r=>r.kind==='relationship').map(r=>r.id);
if(ids.length!==2)throw Error('Two fixture reviews missing');
await evaluate(`(()=>{document.querySelector('[data-tab=adjudication]').click();const f=document.getElementById('adjudication-form');f.elements.review_ids.value=${JSON.stringify(ids.join(','))};f.elements.outcome.value='accept';f.elements.decision.value=JSON.stringify({similarity:2,prerequisite:'unknown',transfer:'unknown',difficulty:'unknown'});f.elements.rationale.value='Synthetic adjudication browser fixture only.';f.elements.evidence.value='Synthetic fixture evidence.';f.requestSubmit()})()`);
await wait("document.getElementById('save-status').textContent==='Record saved.'");
await wait("document.getElementById('progress').textContent.includes('1')");
await screenshot('synthetic-adjudication');
if(errors.length)throw Error('Browser exceptions: '+errors.join(', '));
await fs.writeFile(`${out}/receipt.json`,JSON.stringify({real_pilot_read_only:true,real_pilot_pairs:40,synthetic_only_writes:true,two_reviewers_and_third_adjudicator:true,reviewer_form_reset_checked:true,mobile_horizontal_overflow:mobileOverflow,browser_exceptions:errors,screenshots_private:true},null,2));
console.log('Browser smoke passed; screenshots and receipt kept private.');ws.close();
