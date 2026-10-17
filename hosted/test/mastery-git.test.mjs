import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdtempSync,mkdirSync,writeFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,dirname} from 'node:path';
import {createDatabase} from './helpers/d1.mjs';
import {classify,frontMatterKey,keyFromPath,parseTrailers} from '../mastery/git-convention.mjs';
import {ingestRepository,registerRepository} from '../mastery/git-ingest.mjs';
import {readLedger} from '../mastery/ledger.mjs';

test('path convention maps common layouts and ignores everything else', () => {
  const cases = [
    ['leetcode/two-sum/solution.py','leetcode:two-sum'],
    ['LeetCode/0001-two-sum/0001-two-sum.py','leetcode:two-sum'],
    ['solutions/leetcode/coin_change.cpp','leetcode:coin-change'],
    ['codeforces/1000A.cpp','codeforces:1000A'],
    ['codeforces/cf-1352c/main.py','codeforces:1352C'],
    ['codeforces/1335F1/a.cpp','codeforces:1335F1'],
    ['atcoder/abc300_d/main.py','atcoder:abc300_d'],
    ['baekjoon/10986.py','baekjoon:10986'],
    ['BOJ/bj-1000/a.cpp','baekjoon:1000'],
    ['README.md',null], ['leetcode/README.md',null], ['codeforces/notes.md',null], ['atcoder/readme.txt',null],
  ];
  for (const [path,key] of cases) assert.equal(keyFromPath(path),key,path);
  assert.equal(frontMatterKey('study/docs/Algorithm/Baekjoon/prefix.mdx','---\nid: bj-10986\ntitle: x\n---\nbody'),'baekjoon:10986');
  assert.equal(frontMatterKey('study/docs/Algorithm/Baekjoon/intro.mdx','---\nid: algorithm-intro\n---'),null);
  assert.equal(frontMatterKey('study/notes/x.md','---\nid: bj-1\n---'),null,'only under a baekjoon directory');
  assert.equal(frontMatterKey('baekjoon/x.md','no front matter'),null);
});

test('trailer classification never upgrades evidence without explicit metadata', () => {
  const c = block => classify(parseTrailers(block).trailers);
  assert.equal(c('').outcome,'reported_completion');
  assert.deepEqual(c('').flags,['no_outcome_trailer']);
  assert.deepEqual([c('Outcome: independent\nHints: none').outcome,c('Outcome: independent\nHints: none').hints],['independent',false]);
  assert.equal(c('Outcome: independent').outcome,'reported_completion','independent claim needs Hints: none');
  assert.ok(c('Outcome: independent').flags.includes('independent_without_hints_trailer'));
  assert.equal(c('Outcome: independent\nHints: used').outcome,'assisted');
  assert.equal(c('Outcome: failed').outcome,'failed');
  assert.equal(c('Outcome: accepted').outcome,'reported_completion');
  assert.equal(c('Outcome: maybe').outcome,'reported_completion');
  assert.equal(c('Minutes: 25').self_minutes,25);
  assert.ok(c('Minutes: soon').flags.includes('invalid_minutes'));
  const transfer=c('Outcome: failed\nActivity: transfer\nTransfer-From: leetcode:two-sum');
  assert.deepEqual([transfer.activity_type,transfer.transfer_from],['transfer','leetcode:two-sum']);
  assert.equal(c('Activity: transfer').activity_type,null);
  const parsed=parseTrailers('Problem: leetcode:a\nProblem: codeforces:4A\nReviewed-by: x');
  assert.deepEqual(parsed.problems,['leetcode:a','codeforces:4A']); assert.deepEqual(parsed.unknown,['Reviewed-by']);
});

function repo() {
  const dir=mkdtempSync(join(tmpdir(),'mastery-git-'));
  const run=(...args)=>execFileSync('git',['-C',dir,...args],{encoding:'utf8',env:{...process.env,GIT_AUTHOR_NAME:'t',GIT_AUTHOR_EMAIL:'t@x',GIT_COMMITTER_NAME:'t',GIT_COMMITTER_EMAIL:'t@x'}}).trim();
  run('init','-q','-b','main');
  let n=0;
  const commit=(files,message,date)=>{
    for (const [path,body] of Object.entries(files)) { mkdirSync(dirname(join(dir,path)),{recursive:true}); writeFileSync(join(dir,path),body ?? String(++n)); }
    run('add','-A');
    const d=date ?? `2026-10-${String(10+(++n%9)).padStart(2,'0')}T10:00:00+09:00`;
    execFileSync('git',['-C',dir,'commit','-q','-m',message],{env:{...process.env,GIT_AUTHOR_NAME:'t',GIT_AUTHOR_EMAIL:'t@x',GIT_COMMITTER_NAME:'t',GIT_COMMITTER_EMAIL:'t@x',GIT_AUTHOR_DATE:d,GIT_COMMITTER_DATE:d}});
    return run('rev-parse','HEAD');
  };
  return {dir,run,commit,cleanup:()=>rmSync(dir,{recursive:true,force:true})};
}
const NOW=new Date('2026-10-17T00:00:00Z');

test('linear history: trailers, untrailered first touch only, idempotent rerun, incremental watermark', async context => {
  const r=repo(), db=createDatabase(); context.after(()=>{db.close();r.cleanup();});
  r.commit({'leetcode/two-sum/a.py':'x'},'two sum\n\nOutcome: independent\nHints: none\nMinutes: 20');
  r.commit({'leetcode/two-sum/a.py':'y'},'refactor two sum');
  r.commit({'codeforces/4A.cpp':'z'},'watermelon');
  r.commit({'codeforces/4A.cpp':'zz'},'tidy');
  r.commit({'README.md':'r'},'docs');
  await registerRepository(db,'owner',{repoId:'sol',path:r.dir});
  const first=await ingestRepository(db,'owner','sol',{now:NOW});
  assert.equal(first.status,'ok'); assert.equal(first.commits,5);
  const ledger=await readLedger(db,'owner');
  assert.deepEqual(ledger.attempts.map(a=>[a.problem_key,a.outcome,a.evidence_source]),[
    ['leetcode:two-sum','independent','git'],['leetcode:two-sum','reported_completion','git'],['codeforces:4A','reported_completion','git']]);
  assert.equal(ledger.attempts[0].self_minutes,20);
  assert.equal((await ingestRepository(db,'owner','sol',{now:NOW})).status,'up_to_date');
  r.commit({'atcoder/abc300_d/main.py':'q'},'abc300 d\n\nOutcome: failed');
  const second=await ingestRepository(db,'owner','sol',{now:new Date('2026-10-17T01:00:00Z')});
  assert.equal(second.commits,1); assert.equal(second.observed,1);
  assert.equal((await readLedger(db,'owner')).attempts.at(-1).outcome,'failed');
});

test('merge commits: merged branch commits are observed once; merge diff is not double counted', async context => {
  const r=repo(), db=createDatabase(); context.after(()=>{db.close();r.cleanup();});
  r.commit({'README.md':'0'},'init');
  r.run('checkout','-q','-b','side');
  r.commit({'leetcode/coin-change/a.py':'1'},'coin change\n\nOutcome: assisted');
  r.run('checkout','-q','main');
  r.commit({'leetcode/jump-game/a.py':'2'},'jump game');
  execFileSync('git',['-C',r.dir,'merge','-q','--no-ff','-m','merge side','side'],{env:{...process.env,GIT_AUTHOR_NAME:'t',GIT_AUTHOR_EMAIL:'t@x',GIT_COMMITTER_NAME:'t',GIT_COMMITTER_EMAIL:'t@x'}});
  await registerRepository(db,'owner',{repoId:'sol',path:r.dir});
  const report=await ingestRepository(db,'owner','sol',{now:NOW});
  assert.equal(report.commits,4);
  const keys=(await readLedger(db,'owner')).attempts.map(a=>a.problem_key).sort();
  assert.deepEqual(keys,['leetcode:coin-change','leetcode:jump-game']);
});

test('force-push stops ingestion until an explicit anchor; legacy Baekjoon notes are read', async context => {
  const r=repo(), db=createDatabase(); context.after(()=>{db.close();r.cleanup();});
  r.commit({'study/docs/Algorithm/Baekjoon/prefix-sum.mdx':'---\nid: bj-10986\ntitle: 나머지 합\ntags: [prefix-sum]\n---\n노트'},'notes');
  const keep=r.commit({'study/docs/Algorithm/Baekjoon/intro.mdx':'---\nid: algorithm-intro\n---'},'intro');
  await registerRepository(db,'owner',{repoId:'notes',path:r.dir});
  await ingestRepository(db,'owner','notes',{now:NOW});
  assert.deepEqual((await readLedger(db,'owner')).attempts.map(a=>[a.problem_key,a.outcome]),[['baekjoon:10986','reported_completion']]);
  r.run('reset','-q','--hard','HEAD~1');
  r.commit({'baekjoon/1000.py':'a'},'rewritten');
  const stopped=await ingestRepository(db,'owner','notes',{now:new Date('2026-10-17T02:00:00Z')});
  assert.equal(stopped.status,'history_rewritten');
  assert.equal((await ingestRepository(db,'owner','notes',{now:new Date('2026-10-17T03:00:00Z')})).status,'history_rewritten','stays stopped');
  assert.equal((await readLedger(db,'owner')).attempts.length,1,'nothing ingested from rewritten history');
  const head=r.run('rev-parse','HEAD~1');
  assert.notEqual(head,keep);
  const resumed=await ingestRepository(db,'owner','notes',{now:new Date('2026-10-17T04:00:00Z'),anchorSha:head});
  assert.equal(resumed.status,'ok'); assert.equal(resumed.observed,1);
  assert.equal((await readLedger(db,'owner')).attempts.at(-1).problem_key,'baekjoon:1000');
});

test('registration is immutable and verifies the branch', async context => {
  const r=repo(), db=createDatabase(); context.after(()=>{db.close();r.cleanup();});
  r.commit({'a.txt':'1'},'init');
  await assert.rejects(registerRepository(db,'owner',{repoId:'x',path:r.dir,branch:'nope'}),/git rev-parse failed/);
  await registerRepository(db,'owner',{repoId:'x',path:r.dir});
  await assert.rejects(registerRepository(db,'owner',{repoId:'x',path:r.dir+'/other'}),/different path/);
  assert.throws(()=>db.sqlite.prepare("UPDATE git_repositories SET branch='dev'").run(),/append-only/);
});
