/** Actual Chrome/CDP test using disposable synthetic learners and mocked provider responses. */
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp,mkdir,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {startLocal} from './local.mjs';
import {createJournalService} from './mastery/journal.mjs';
import {createExplanationService} from './mastery/explanations.mjs';
import {catalogFromData} from './mastery/catalog.mjs';

const temp=await mkdtemp(join(tmpdir(),'explanation-browser-'));
const output=resolve(process.env.SMOKE_OUTPUT_DIR ?? join(temp,'receipts'));await mkdir(output,{recursive:true,mode:0o700});
const chrome=spawn(process.env.CHROME_BINARY ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',[
  '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking',
  '--disable-sync','--disable-extensions','--remote-debugging-port=0',`--user-data-dir=${join(temp,'chrome')}`,'about:blank'],{stdio:'ignore'});
let chromeError=null;chrome.on('error',error=>{chromeError=error;});
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const receipts=[];
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
  const service=await startLocal({port:0,database:join(directory,'learner.sqlite3'),learnerId:learner,curriculum:{schema_version:1,curriculum_version:'fixture-browser',problems:[],relationships:[]},catalogProvider:()=>catalog,autoGitSync:false,explanationEnv:env});
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
   for(let i=0;i<100;i++){if(await evaluate("document.querySelectorAll('.explanation-provider').length===2"))break;await delay(100);}
   assert.equal(await evaluate("document.querySelectorAll('.explanation-provider').length"),2);
   await evaluate("document.querySelectorAll('details').forEach(d=>d.open=true)");
   const layout=await evaluate("(()=>{const a=[...document.querySelectorAll('.explanation-provider')].map(x=>x.getBoundingClientRect());return a.map(x=>({x:x.x,y:x.y,width:x.width}));})()");
   assert.ok(layout[1].x>layout[0].x);assert.ok(Math.abs(layout[0].y-layout[1].y)<2);
   const content=await evaluate("document.querySelector('.journal-plan').innerText");
   assert.ok(content.includes('Gemini'));assert.ok(content.includes('NVIDIA NIM'));assert.ok(content.includes('원래 추천 근거 보기'));
   if(scenario==='dual_success'){assert.ok(content.includes('Gemini에서 작성'));assert.ok(content.includes('NVIDIA에서 작성'));assert.ok(!content.includes('기본 설명 ·'));}
   if(scenario==='partial_failure'){assert.ok(content.includes('Gemini에서 작성'));assert.ok(content.includes('생성 실패'));assert.ok(!content.includes('기본 설명 ·'));}
   if(scenario==='both_failed'){assert.ok(content.includes('기본 설명 ·'));assert.ok(!content.includes('에서 작성한 합성'));}
   await writeFile(join(output,scenario+'-desktop.png'),Buffer.from((await call('Page.captureScreenshot',{format:'png'})).data,'base64'));
   await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
   const mobile=await evaluate("(()=>{const a=[...document.querySelectorAll('.explanation-provider')].map(x=>x.getBoundingClientRect());return {stacked:a[1].y>a[0].y,overflow:document.documentElement.scrollWidth>innerWidth};})()");
   assert.equal(mobile.stacked,true);assert.equal(mobile.overflow,false);
   await call('Page.reload');await delay(300);
   assert.deepEqual(explanations.usage(learner),before);assert.equal(calls,2);assert.deepEqual(errors,[]);
   receipts.push({scenario,desktop_side_by_side:true,mobile_stacked:true,no_overflow:true,refresh_no_new_attempts:true,provider_calls:calls,real_providers:false});
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
