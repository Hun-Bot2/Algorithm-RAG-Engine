import test from 'node:test';
import assert from 'node:assert/strict';
import {fork} from 'node:child_process';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';

function message(child,type){
 return new Promise((resolve,reject)=>{
  const timeout=setTimeout(()=>finish(new Error(`restart child did not send ${type}`)),10000);
  const onMessage=data=>{if(data.type===type)finish(null,data);};
  const onExit=code=>finish(new Error(`restart child exited ${code} before ${type}: ${child.restartStderr || "no stderr"}`));
  const onError=error=>finish(error);
  function finish(error,value){clearTimeout(timeout);child.off('message',onMessage);child.off('exit',onExit);child.off('error',onError);error?reject(error):resolve(value);}
  child.on('message',onMessage);child.once('exit',onExit);child.once('error',onError);
 });
}
async function status(child){const response=message(child,'status');child.send({type:'status'});return response;}
async function stop(child){
 if(child.exitCode!==null||child.signalCode!==null)return;
 await new Promise((resolve,reject)=>{
  const timer=setTimeout(()=>{child.kill();reject(new Error('restart child did not stop'));},10000);
  child.once('exit',code=>{clearTimeout(timer);code===0?resolve():reject(new Error(`restart child exited ${code}`));});child.send({type:'close'});
 });
}
for(const scenario of ['success','partial'])test(`server process restart reuses persisted ${scenario} responses without API attempts or token changes`,async t=>{
 const root=await mkdtemp(join(tmpdir(),'explanation-restart-')),children=[];
 t.after(async()=>{for(const child of children)if(child.exitCode===null&&child.signalCode===null)child.kill();await rm(root,{recursive:true,force:true});});
 const start=async mode=>{
  const child=fork(fileURLToPath(new URL('./helpers/explanation-restart-worker.mjs',import.meta.url)),[join(root,'learner.sqlite3'),mode],{
   stdio:['ignore','ignore','pipe','ipc'],env:{PATH:process.env.PATH},
  });child.restartStderr='';child.stderr.on('data',chunk=>{child.restartStderr=(child.restartStderr+chunk).slice(-4000);});children.push(child);const ready=await message(child,'ready');return {child,...ready};
 };
 const original=await start('seed-'+scenario);
 const before=await (await fetch(original.url+'/api/journal')).json();
 const originalState=await status(original.child);assert.equal(originalState.externalCalls,0);assert.equal(originalState.mockCalls,2);
 assert.deepEqual(before.plan.slots[0].explanation.results.map(r=>r.status),scenario==='success'?['success','success']:['success','failed']);
 await stop(original.child);
 const restarted=await start('read-only');assert.notEqual(restarted.pid,original.pid);
 for(let i=0;i<3;i++){
  const response=await (await fetch(restarted.url+'/api/journal')).json();
  assert.equal(response.plan.snapshot_id,before.plan.snapshot_id);
  assert.deepEqual(response.plan.slots[0].explanation,before.plan.slots[0].explanation);
 }
 const after=await status(restarted.child);
 assert.equal(after.externalCalls,0);assert.equal(after.mockCalls,0);
 assert.deepEqual(after.records,originalState.records);assert.deepEqual(after.usage,originalState.usage);
 await stop(restarted.child);
});
