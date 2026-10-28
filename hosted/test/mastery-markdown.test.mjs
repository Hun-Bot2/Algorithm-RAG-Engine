import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdtempSync,mkdirSync,writeFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,dirname} from 'node:path';
import {createDatabase} from './helpers/d1.mjs';
import {ingestRepository,registerRepository} from '../mastery/git-ingest.mjs';
import {parseMarkdownAttempt} from '../mastery/markdown-attempt.mjs';
import {readLedger} from '../mastery/ledger.mjs';

const NOW=new Date('2026-10-18T00:00:00Z');
const fm=(fields,body='note')=>`---\nlearning_schema: 1\n${fields}\n---\n${body}\n`;
function repo() {
  const dir=mkdtempSync(join(tmpdir(),'mastery-markdown-'));
  const env={...process.env,GIT_AUTHOR_NAME:'test',GIT_AUTHOR_EMAIL:'test@x',GIT_COMMITTER_NAME:'test',GIT_COMMITTER_EMAIL:'test@x'};
  const run=(...args)=>execFileSync('git',['-C',dir,...args],{encoding:'utf8',env}).trim();
  run('init','-q','-b','main');
  execFileSync('git',['-C',dir,'commit','--allow-empty','-q','-m','baseline'],{env:{...env,GIT_AUTHOR_DATE:'2026-10-01T00:00:00Z',GIT_COMMITTER_DATE:'2026-10-01T00:00:00Z'}});
  let n=0;
  const commit=(files,message='attempt')=>{
    for(const [path,body] of Object.entries(files)){mkdirSync(dirname(join(dir,path)),{recursive:true});writeFileSync(join(dir,path),body);}
    run('add','-A'); const date=`2026-10-${String(10+(++n%8)).padStart(2,'0')}T10:00:00+09:00`;
    execFileSync('git',['-C',dir,'commit','-q','-m',message],{env:{...env,GIT_AUTHOR_DATE:date,GIT_COMMITTER_DATE:date}});
    return run('rev-parse','HEAD');
  };
  return {dir,commit,cleanup:()=>rmSync(dir,{recursive:true,force:true})};
}
async function setup(context,r) {
  const db=createDatabase(); context.after(()=>{db.close();r.cleanup();});
  await registerRepository(db,'owner',{repoId:'solutions',path:r.dir});
  return db;
}
const ingest=async(db)=>ingestRepository(db,'owner','solutions',{now:NOW});
const attempts=async(db)=>(await readLedger(db,'owner')).attempts;

test('flat marked parser validates values and rejects duplicate, unknown, multiline and quoted syntax',()=>{
  const parsed=parseMarkdownAttempt(fm('problem: leetcode:two-sum\nattempt: 2026-10-18-01\noutcome: O\nhints: none'));
  assert.equal(parsed.metadata.outcome,'independent');
  assert.equal(parseMarkdownAttempt(fm('problem: leetcode:two-sum\nattempt: x\noutcome: __proto__')).error,'invalid_learning_outcome');
  assert.equal(parseMarkdownAttempt(fm('problem: leetcode:two-sum\nattempt: x\nattempt: y\noutcome: X')).error,'duplicate_learning_field:attempt');
  assert.equal(parseMarkdownAttempt(fm('problem: leetcode:two-sum\nattempt: x\noutcome: X\ncustom: nope')).error,'unknown_learning_field:custom');
  assert.equal(parseMarkdownAttempt(fm('problem: leetcode:two-sum\nattempt: x\noutcome: X\nminutes: 2\n  nested: yes')).error,'unsupported_learning_frontmatter');
  assert.equal(parseMarkdownAttempt(fm('problem: "leetcode:two-sum"\nattempt: x\noutcome: X')).metadata.problem_key,'leetcode:two-sum');
  assert.equal(parseMarkdownAttempt(fm('problem: leetcode:TODO\nattempt: x\noutcome: X')).error,'invalid_learning_problem');
});

test('records independent O with explicit no hints, X, assisted, missing outcome fails closed, and transfer metadata',async context=>{
  const r=repo(),db=await setup(context,r);
  r.commit({'leetcode/two-sum.md':fm('problem: leetcode:two-sum\nattempt: a1\noutcome: O\nhints: none')});
  r.commit({'leetcode/valid-parentheses.md':fm('problem: leetcode:valid-parentheses\nattempt: a2\noutcome: X')});
  r.commit({'leetcode/merge-intervals.md':fm('problem: leetcode:merge-intervals\nattempt: a3\noutcome: assisted\nhints: used')});
  r.commit({'atcoder/abc300_d.md':fm('problem: atcoder:abc300_d\nattempt: a4\noutcome: O\nhints: none\nactivity: transfer\ntransfer_from: leetcode:two-sum\nminutes: 24','independent solve')});
  r.commit({'leetcode/group-anagrams.md':'---\nlearning_schema: 1\nproblem: leetcode:group-anagrams\nattempt: a5\n---\nno outcome'});
  const report=await ingest(db), rows=await attempts(db);
  assert.equal(report.observed,4);
  // Ingestion fidelity: the stored outcome is preserved as recorded_* (policy v3 projects it to solved/unsolved).
  assert.deepEqual(rows.map(x=>[x.problem_key,x.recorded_outcome,x.recorded_hints,x.activity_type,x.transfer_from,x.self_minutes]),[
    ['leetcode:two-sum','independent',false,null,null,null],
    ['leetcode:valid-parentheses','failed',null,null,null,null],
    ['leetcode:merge-intervals','assisted',true,null,null,null],
    ['atcoder:abc300_d','independent',false,'transfer','leetcode:two-sum',24],
  ]);
  assert.ok(report.flags && Object.values(report.flags).flat().some(x=>x==='invalid_markdown_metadata:leetcode:group-anagrams'));
});

test('same attempt note edits deduplicate; a new token records; changed metadata under old token is flagged and skipped',async context=>{
  const r=repo(),db=await setup(context,r),path='leetcode/two-sum.md';
  r.commit({[path]:fm('problem: leetcode:two-sum\nattempt: a1\noutcome: O\nhints: none')}); await ingest(db);
  r.commit({[path]:fm('problem: leetcode:two-sum\nattempt: a1\noutcome: O\nhints: none','added explanation')});
  assert.equal((await ingest(db)).observed,0); assert.equal((await attempts(db)).length,1);
  r.commit({[path]:fm('problem: leetcode:two-sum\nattempt: a2\noutcome: X')});
  const next=await ingest(db); assert.equal(next.observed,1,JSON.stringify(next)); assert.equal((await attempts(db)).length,2);
  r.commit({[path]:fm('problem: leetcode:two-sum\nattempt: a2\noutcome: assisted')});
  const report=await ingest(db);
  assert.equal(report.observed,0); assert.ok(Object.values(report.flags).flat().includes('markdown_attempt_conflict:leetcode:two-sum'));
  assert.equal((await attempts(db)).length,2);
});

test('multi-problem commits bind each file to its own key; explicit outcome/hints trailers win per key',async context=>{
  const r=repo(),db=await setup(context,r);
  r.commit({
    'leetcode/two-sum.md':fm('problem: leetcode:two-sum\nattempt: a1\noutcome: O\nhints: none'),
    'codeforces/4A.md':fm('problem: codeforces:4A\nattempt: a2\noutcome: O\nhints: none'),
  },'pair\n\nOutcome: failed\nHints: used');
  await ingest(db);
  const rows=await attempts(db);
  assert.deepEqual(rows.map(x=>[x.problem_key,x.recorded_outcome,x.recorded_hints]).sort((a,b)=>a[0].localeCompare(b[0])),[['codeforces:4A','failed',true],['leetcode:two-sum','failed',true]]);
  const raw=(await readLedger(db,'owner')).git_observations;
  assert.deepEqual(raw.map(x=>x.payload.path).sort(),['codeforces/4A.md','leetcode/two-sum.md']);
});

test('Problem-only trailers do not override structured metadata; legacy no-marker markdown remains unverified',async context=>{
  const r=repo(),db=await setup(context,r);
  r.commit({'leetcode/two-sum.md':fm('attempt: a1\noutcome: O\nhints: none')},'metadata\n\nProblem: leetcode:two-sum');
  r.commit({'leetcode/valid-parentheses.md':'---\nid: ignored\ntitle: old note\n---\ntext'});
  await ingest(db); const rows=await attempts(db);
  assert.deepEqual(rows.map(x=>[x.problem_key,x.recorded_outcome]),[['leetcode:two-sum','independent'],['leetcode:valid-parentheses','reported_completion']]);
});

test('templates and path/frontmatter mismatches are ignored with a visible flag',async context=>{
  const r=repo(),db=await setup(context,r);
  r.commit({'templates/solution.md':fm('problem: leetcode:two-sum\nattempt: fake\noutcome: O\nhints: none')});
  r.commit({'leetcode/two-sum.md':fm('problem: leetcode:valid-parentheses\nattempt: a1\noutcome: X')});
  const report=await ingest(db);
  assert.equal((await attempts(db)).length,0);
  assert.ok(Object.values(report.flags).flat().some(x=>x.startsWith('invalid_markdown_metadata:')));
});

test('solution README in a recognized problem folder keeps the folder key',async context=>{
  const r=repo(),db=await setup(context,r);
  r.commit({'leetcode/two-sum/README.md':fm('problem: leetcode:two-sum\nattempt: readme-1\noutcome: X')});
  await ingest(db);
  assert.deepEqual((await readLedger(db,'owner')).git_observations.map(x=>[x.payload.problem_key,x.payload.path]),[['leetcode:two-sum','leetcode/two-sum/README.md']]);
});

test('same-commit attempts for one problem retain per-file provenance; duplicate token conflicts fail closed',async context=>{
  const r=repo(),db=await setup(context,r);
  r.commit({
    'leetcode/two-sum.md':fm('problem: leetcode:two-sum\nattempt: a1\noutcome: X'),
    'leetcode/two-sum/attempt-2.md':fm('problem: leetcode:two-sum\nattempt: a2\noutcome: assisted'),
  });
  assert.equal((await ingest(db)).observed,2);
  const paths=(await readLedger(db,'owner')).git_observations.map(x=>x.payload.path).sort();
  assert.deepEqual(paths,['leetcode/two-sum.md','leetcode/two-sum/attempt-2.md']);
  r.commit({
    'leetcode/two-sum.md':fm('problem: leetcode:two-sum\nattempt: a3\noutcome: X'),
    'leetcode/two-sum/attempt-2.md':fm('problem: leetcode:two-sum\nattempt: a3\noutcome: assisted'),
  });
  const report=await ingest(db);
  assert.equal(report.observed,0);
  assert.ok(Object.values(report.flags).flat().includes('conflicting_markdown_metadata:leetcode:two-sum'));
  assert.equal((await attempts(db)).length,2);
});


test('simplified schema uses short-year attempt IDs, O/X and implicit practice',()=>{
  const parse=extra=>parseMarkdownAttempt(`---\nlearning_schema: 2\nattempt: 26-10-18-01\n${extra}\n---\nSolution`,{inferredProblem:'leetcode:two-sum'});
  const success=parse('outcome: O');
  assert.equal(success.metadata.attempt,'26-10-18-01');
  assert.equal(success.metadata.outcome,'independent');
  assert.equal(success.metadata.hints,false);
  assert.equal(success.metadata.activity_type,'practice');
  assert.equal(parse('outcome: X').metadata.outcome,'failed');
  assert.equal(parse('outcome: O\nactivity: review').metadata.activity_type,'review');
  assert.equal(parse('outcome: assisted').error,'invalid_learning_outcome');
  assert.equal(parse('outcome: O\nhints: none').error,'unknown_learning_field:hints');
});

test('schema 2 committed success and failed review enter the existing ledger',async context=>{
  const r=repo(),db=await setup(context,r);
  const body=(token,outcome,extra='')=>`---\nlearning_schema: 2\nattempt: ${token}\noutcome: ${outcome}\n${extra}---\n# 풀이`;
  r.commit({'leetcode/two-sum.md':body('26-10-18-01','O')});
  r.commit({'leetcode/two-sum.md':body('26-10-19-01','X','activity: review\n')});
  assert.equal((await ingest(db)).observed,2);
  assert.deepEqual((await attempts(db)).map(x=>[x.outcome,x.hints,x.activity_type]),[
    ['independent',false,'practice'],['failed',null,'review'],
  ]);
});
