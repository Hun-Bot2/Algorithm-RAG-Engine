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

test('local server automatically ingests registered Git commits and exposes changed evidence revision',async t=>{
  const {mkdtempSync,mkdirSync,writeFileSync,rmSync}=await import('node:fs');
  const {tmpdir}=await import('node:os');const {join}=await import('node:path');
  const {execFileSync}=await import('node:child_process');
  const {registerRepository}=await import('../mastery/git-ingest.mjs');
  const dir=mkdtempSync(join(tmpdir(),'journal-auto-git-'));t.after(()=>rmSync(dir,{recursive:true,force:true}));
  const git=(...args)=>execFileSync('git',['-C',dir,...args],{stdio:'pipe',env:{...process.env,GIT_AUTHOR_NAME:'Test',GIT_AUTHOR_EMAIL:'test@example.test',GIT_COMMITTER_NAME:'Test',GIT_COMMITTER_EMAIL:'test@example.test'}});
  git('init','-q','-b','main');writeFileSync(join(dir,'README.md'),'fixture');git('add','.');git('commit','-qm','initial');
  const service=await startLocal({port:0,database:':memory:',curriculum,gitSyncIntervalMs:20});
  try{
    await registerRepository(service.db,'local-owner',{repoId:'solutions',path:dir});
    const read=async()=>await(await fetch(service.url+'/api/local/sync')).json();
    const before=await read();mkdirSync(join(dir,'leetcode','two-sum'),{recursive:true});writeFileSync(join(dir,'leetcode','two-sum','solution.py'),'# fixture');
    git('add','.');git('commit','-qm','solve\n\nOutcome: failed\nHints: used');
    let after;
    for(let i=0;i<100;i++){
      after=await read();
      if(service.db.sqlite.prepare('SELECT COUNT(*) n FROM git_observations').get().n===1)break;
      await new Promise(r=>setTimeout(r,10));
    }
    assert.equal(service.db.sqlite.prepare('SELECT COUNT(*) n FROM git_observations').get().n,1);
    assert.notEqual(after.revision,before.revision);assert.equal(after.enabled,true);
    await service.gitSync.tick();assert.equal(service.db.sqlite.prepare('SELECT COUNT(*) n FROM git_observations').get().n,1);
  }finally{await service.close();}
});
