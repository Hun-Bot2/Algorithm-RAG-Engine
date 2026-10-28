import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdtempSync,mkdirSync,writeFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,dirname} from 'node:path';
import {createDatabase} from './helpers/d1.mjs';
import {registerRepository} from '../mastery/git-ingest.mjs';
import {createGitSync} from '../mastery/git-sync.mjs';
import {readLedger} from '../mastery/ledger.mjs';

function repo() {
  const dir=mkdtempSync(join(tmpdir(),'mastery-git-sync-'));
  const env={...process.env,GIT_AUTHOR_NAME:'t',GIT_AUTHOR_EMAIL:'t@x',GIT_COMMITTER_NAME:'t',GIT_COMMITTER_EMAIL:'t@x'};
  const run=(...args)=>execFileSync('git',['-C',dir,...args],{encoding:'utf8',env}).trim();
  run('init','-q','-b','main');
  function commit(path,body,message,date='2026-10-17T10:00:00+09:00') {
    mkdirSync(dirname(join(dir,path)),{recursive:true}); writeFileSync(join(dir,path),body);
    run('add','-A');
    execFileSync('git',['-C',dir,'commit','-q','-m',message],{env:{...env,GIT_AUTHOR_DATE:date,GIT_COMMITTER_DATE:date}});
    return run('rev-parse','HEAD');
  }
  return {dir,run,commit,cleanup:()=>rmSync(dir,{recursive:true,force:true})};
}
const NOW=new Date('2026-10-18T00:00:00Z');

test('manual ticks ingest only new commits and retain failed independent-claim provenance',async t=>{
  const r=repo(),db=createDatabase(); t.after(()=>{db.close();r.cleanup();});
  r.commit('leetcode/two-sum/main.py','one','first\n\nOutcome: independent');
  await registerRepository(db,'owner',{repoId:'solutions',path:r.dir});
  const sync=createGitSync({db,learnerId:'owner',intervalMs:60000}); t.after(()=>sync.close());
  const first=await sync.tick();
  assert.equal(first.lastResults[0].status,'ok');
  assert.ok(Object.values(first.lastResults[0].flags).flat().includes('independent_without_hints_trailer'));
  let ledger=await readLedger(db,'owner');
  assert.equal(ledger.attempts.length,1);
  assert.equal(ledger.attempts[0].recorded_outcome,'reported_completion');
  assert.equal(ledger.attempts[0].outcome,'independent','policy v3 counts any completion as solved');
  assert.equal(ledger.git_observations[0].payload.outcome,'reported_completion');
  assert.ok(first.lastWarnings[0].flags.includes('independent_without_hints_trailer'));
  const unchanged=await sync.tick();
  assert.equal(unchanged.lastResults[0].status,'up_to_date');
  assert.deepEqual(unchanged.lastWarnings,first.lastWarnings,'warnings survive idle polling');
  assert.equal((await readLedger(db,'owner')).attempts.length,1,'repeat tick adds no duplicate');
  r.commit('atcoder/abc300_d/main.py','two','second\n\nOutcome: failed','2026-10-17T11:00:00+09:00');
  const second=await sync.tick();
  assert.equal(second.lastResults[0].observed,1);
  assert.deepEqual(second.lastWarnings,[],'clean subsequent ingestion clears warnings');
  ledger=await readLedger(db,'owner');
  assert.deepEqual(ledger.attempts.map(row=>row.recorded_outcome),['reported_completion','failed']);
});

test('history rewrite remains halted on later sync ticks',async t=>{
  const r=repo(),db=createDatabase(); t.after(()=>{db.close();r.cleanup();});
  r.commit('README.md','base','base');
  r.commit('baekjoon/1000.py','old','old');
  await registerRepository(db,'owner',{repoId:'notes',path:r.dir});
  const sync=createGitSync({db,learnerId:'owner'}); t.after(()=>sync.close());
  await sync.tick();
  r.run('reset','-q','--hard','HEAD~1');
  r.commit('baekjoon/1001.py','new','rewritten','2026-10-17T11:00:00+09:00');
  assert.equal((await sync.tick()).lastResults[0].status,'history_rewritten');
  assert.equal((await sync.tick()).lastResults[0].status,'history_rewritten');
  assert.deepEqual((await readLedger(db,'owner')).attempts.map(row=>row.problem_key),['baekjoon:1000']);
});

test('per-repository failures are reported and do not reject ticks',async t=>{
  const r=repo(),db=createDatabase(); t.after(()=>{db.close();r.cleanup();});
  r.commit('README.md','ok','initial');
  await registerRepository(db,'owner',{repoId:'gone',path:r.dir});
  rmSync(r.dir,{recursive:true,force:true});
  const sync=createGitSync({db,learnerId:'owner'}); t.after(()=>sync.close());
  const result=await sync.tick();
  assert.equal(result.lastResults[0].status,'error');
  assert.match(result.lastError,/gone:/);
});

test('close stops polling and awaits an active tick',async()=>{
  const db=createDatabase();
  let release;
  const gate=new Promise(resolve=>{release=resolve;});
  const original=db.prepare.bind(db);
  db.prepare=sql=>sql.startsWith('SELECT repo_id FROM git_repositories') ? {
    bind:()=>({all:async()=>{await gate;return {results:[]};}}),
  } : original(sql);
  const sync=createGitSync({db,learnerId:'owner',intervalMs:60000});
  assert.equal(sync.start().started,true);
  const tick=sync.tick();
  const closing=sync.close();
  let closed=false; void closing.then(()=>{closed=true;});
  await Promise.resolve();
  assert.equal(closed,false);
  release();
  await Promise.all([tick,closing]);
  assert.equal(sync.status().closed,true);
  assert.equal(sync.status().started,false);
  assert.equal(sync.status().running,false);
  assert.throws(()=>sync.start(),/closed/);
  db.close();
});
