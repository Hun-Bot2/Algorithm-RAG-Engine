import test from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync,execFileSync} from 'node:child_process';
import {mkdtempSync,writeFileSync,readFileSync,rmSync,existsSync,mkdirSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {run} from '../cli.mjs';

const ROOT=resolve(new URL('../..',import.meta.url).pathname);
function fixtureCatalog(dir) {
  const rows=[];
  for (let i=0;i<10;i++) rows.push({problem_key:`leetcode:p${i}`,platform:'leetcode',title:`P${i}`,url:`https://leetcode.com/problems/p${i}/`,native_difficulty:['Easy','Medium','Hard'][i%3],tags:['Hash Table'],contest:null,eligibility:'eligible',exclusion_reasons:[]});
  for (let r=800;r<=1600;r+=100) rows.push({problem_key:`codeforces:${r}A`,platform:'codeforces',title:`CF ${r}`,url:`https://codeforces.com/problemset/problem/${r}/A`,native_difficulty:r,tags:['implementation'],contest:r,eligibility:'eligible',exclusion_reasons:[]});
  writeFileSync(join(dir,'catalog.jsonl'),rows.map(r=>JSON.stringify(r)).join('\n')+'\n');
  writeFileSync(join(dir,'cmap.json'),JSON.stringify({schema:'mastery-concept-map.v1',entries:[
    {platform:'leetcode',native_tag:'Hash Table',judgment_ref:'t1',concepts:[{concept_id:'arrays-hashing',evidence_status:'ai_reviewed'}]},
    {platform:'codeforces',native_tag:'implementation',judgment_ref:'t2',concepts:[{concept_id:'implementation',evidence_status:'ai_reviewed'},{concept_id:'arrays-hashing',evidence_status:'ai_reviewed'}]}]}));
  const script=`import sys; sys.path.insert(0, ${JSON.stringify(ROOT)})
from pathlib import Path
from research.mastery_catalog.build import build
d=Path(${JSON.stringify(dir)})
import json
from research.mastery_catalog.archive import compile_rows
doc=json.loads(Path(${JSON.stringify(join(ROOT,'assets/mastery/archive.example.v1.json'))}).read_text().replace('codeforces:4A','codeforces:900A').replace('leetcode:two-sum','leetcode:p1'))
build(d/'cat'/'catalog-t1','catalog-t1',{'catalog':d/'catalog.jsonl','aliases':Path(${JSON.stringify(join(ROOT,'assets/problem_identity_aliases.v1.json'))}),'concepts':Path(${JSON.stringify(join(ROOT,'assets/mastery/concepts.v1.json'))})},concept_map=d/'cmap.json',archive=compile_rows(doc),built_at='2026-10-17T00:00:00+00:00')`;
  const result=spawnSync('python3',['-I','-c',script],{encoding:'utf8'});
  assert.equal(result.status,0,result.stderr);
}

test('CLI end to end: catalog switch, Git ingest, plan freeze, attempts, find, weekly report', async context => {
  const dir=mkdtempSync(join(tmpdir(),'mastery-cli-')); context.after(()=>rmSync(dir,{recursive:true,force:true}));
  fixtureCatalog(dir);
  const repo=join(dir,'repo'); mkdirSync(join(repo,'leetcode','p1'),{recursive:true});
  const env={...process.env,GIT_AUTHOR_NAME:'t',GIT_AUTHOR_EMAIL:'t@x',GIT_COMMITTER_NAME:'t',GIT_COMMITTER_EMAIL:'t@x',GIT_AUTHOR_DATE:'2026-10-19T09:00:00+09:00',GIT_COMMITTER_DATE:'2026-10-19T09:00:00+09:00'};
  execFileSync('git',['init','-q','-b','main',repo]); writeFileSync(join(repo,'leetcode','p1','a.py'),'x');
  execFileSync('git',['-C',repo,'add','-A'],{env}); execFileSync('git',['-C',repo,'commit','-q','-m','p1\n\nOutcome: independent\nHints: none\nMinutes: 18'],{env});
  const lines=[];
  const cli=(...args)=>run([...args,'--db',join(dir,'learner.sqlite3'),'--catalog-root',join(dir,'cat'),'--json'],{print:x=>lines.push(x)});
  const last=()=>JSON.parse(lines.at(-1));
  await assert.rejects(cli('plan','--date','2026-10-19','--now','2026-10-18T22:00:00Z'),/no active catalog/);
  assert.equal(await cli('catalog','use','--version','catalog-t1','--now','2026-10-18T21:00:00Z'),0);
  assert.equal(await cli('repo','add','--id','sol','--path',repo,'--now','2026-10-18T21:00:00Z'),0);
  assert.equal(await cli('repo','ingest','--now','2026-10-19T01:00:00Z'),0);
  assert.equal(last()[0].observed,1);
  assert.equal(await cli('plan','--date','2026-10-20','--now','2026-10-19T22:00:00Z'),0);
  const plan=last();
  assert.equal(plan.catalog_version,'catalog-t1'); assert.ok(plan.high_water.git_observations>=1);
  assert.ok(plan.slots.some(s=>s.kind==='review' && s.problem_key==='leetcode:p1'),'Git independent solve schedules a review');
  assert.equal(await cli('plan','--date','2026-10-20','--now','2026-10-20T03:00:00Z'),0);
  assert.deepEqual(last(),plan,'a saved plan is frozen');
  assert.equal(await cli('attempt','--problem','codeforces:900A','--outcome','independent','--hints','none','--now','2026-10-20T10:00:00Z'),0);
  await assert.rejects(cli('attempt','--problem','codeforces:900A','--outcome','independent','--now','2026-10-20T10:00:00Z'),/explicit no hints/);
  assert.equal(await cli('find','--platform','codeforces','--status','solved_independent','--now','2026-10-20T11:00:00Z'),0);
  assert.deepEqual(last().map(r=>r.problem_key),['codeforces:900A']);
  assert.equal(await cli('find','--platform','codeforces','--difficulty','1400..1500','--now','2026-10-20T11:00:00Z'),0);
  assert.deepEqual(last().map(r=>r.problem_key),['codeforces:1400A','codeforces:1500A']);
  await assert.rejects(cli('find','--date','2026-10-01..','--now','2026-10-20T11:00:00Z'),/name its basis/);
  assert.equal(await cli('find','--date','attempt:2026-10-19..2026-10-19','--now','2026-10-20T11:00:00Z'),0);
  assert.deepEqual(last().map(r=>r.problem_key),['leetcode:p1']);
  assert.equal(await cli('find','--archive-category','grad_entrance','--now','2026-10-20T11:00:00Z'),0);
  assert.deepEqual(last().map(r=>r.problem_key),['codeforces:900A'],'only the current (non-superseded) archive version links');
  assert.equal(await cli('find','--provenance','personal_recollection','--now','2026-10-20T11:00:00Z'),0);
  assert.deepEqual(last().map(r=>r.problem_key),['leetcode:p1']);
  assert.equal(await cli('feedback','--plan','2026-10-20','--candidate','codeforces:900A','--source','leetcode:p1','--verdict','related','--channel','concept','--now','2026-10-20T12:00:00Z'),0);
  assert.equal(last().practice_feedback,1);
  const outDir=join(dir,'reports');
  assert.equal(await run(['weekly','--week','2026-10-19','--out',outDir,'--db',join(dir,'learner.sqlite3'),'--catalog-root',join(dir,'cat'),'--now','2026-10-26T01:00:00Z'],{print:x=>lines.push(x)}),0);
  const md=readFileSync(join(outDir,'2026-10-19.md'),'utf8');
  assert.match(md,/# Weekly report 2026-10-19 to 2026-10-25/);
  assert.match(md,/self-reported minutes: 18/);
  await assert.rejects(run(['weekly','--week','2026-10-19','--out',outDir,'--db',join(dir,'learner.sqlite3'),'--catalog-root',join(dir,'cat'),'--now','2026-10-26T01:00:00Z'],{print:()=>{}}),/refusing to overwrite/);
  assert.equal(await cli('status','--now','2026-10-26T01:00:00Z'),0);
  assert.equal(last().active_catalog,'catalog-t1');
});
