import test from 'node:test';
import assert from 'node:assert/strict';
import {get} from 'node:http';
import {startLocal} from '../local.mjs';
const curriculum={schema_version:1,curriculum_version:'test',problems:[],relationships:[]};
test('loopback serves real assets and APIs, rejects foreign mutation and host',async()=>{
  const service=await startLocal({port:0,database:':memory:',curriculum});
  try {
    const html=await fetch(service.url);assert.equal(html.status,200);assert.match(await html.text(),/html/i);
    const state=await (await fetch(service.url+'/api/state')).json();assert.equal(state.learner_id,'local-owner');
    assert.equal((await fetch(service.url+'/api/preferences',{method:'PUT',headers:{'content-type':'application/json'},body:'{}'})).status,403);
    assert.equal((await fetch(service.url+'/api/preferences',{method:'PUT',headers:{'origin':'https://evil.test','content-type':'application/json'},body:'{}'})).status,403);
    const saved=await fetch(service.url+'/api/preferences',{method:'PUT',headers:{'origin':service.url,'content-type':'application/json'},body:JSON.stringify({weekday_minutes:45,target_keys:[]})});
    assert.equal(saved.status,200);
    const hostStatus=await new Promise((resolve,reject)=>get(service.url,{headers:{host:'evil.test'}},response=>{response.resume();resolve(response.statusCode);}).once('error',reject));
    assert.equal(hostStatus,403);
  } finally {await service.close();}
});
