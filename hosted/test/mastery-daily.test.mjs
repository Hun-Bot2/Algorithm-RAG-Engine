import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdtempSync,mkdirSync,writeFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,dirname} from 'node:path';
import {parseDailyLog} from '../mastery/daily-log.mjs';
import {keyFromPath} from '../mastery/git-convention.mjs';
import {catalogFromData} from '../mastery/catalog.mjs';
import {createDatabase} from './helpers/d1.mjs';
import {registerRepository,ingestRepository} from '../mastery/git-ingest.mjs';
import {readLedger} from '../mastery/ledger.mjs';
import {createGitSync} from '../mastery/git-sync.mjs';
const catalog=catalogFromData({problems:[
  {key:'leetcode:two-sum',platform:'leetcode',title:'Two Sum'},
  {key:'leetcode:other',platform:'leetcode',title:'Other'},
  {key:'codeforces:4A',platform:'codeforces',title:'Watermelon'},
  {key:'atcoder:abc001_a',platform:'atcoder',title:'Same'},
  {key:'atcoder:abc002_a',platform:'atcoder',title:'Same'},
],aliases:{'codeforces:5A':'codeforces:4A'}});
const log=(rows,date='2026-10-20')=>`# ${date}\n\n| 문제 | 결과 | 메모 |\n| --- | --- | --- |\n${rows}\n\n## 자유로운 메모\n`;
const parse=(rows,path='daily/26-10-20.md')=>parseDailyLog(path,log(rows),catalog);
test('daily resolves exact names, URLs, aliases and platform scopes without slug guessing',()=>{
 const r=parse('| Two Sum | O | memo |\n| https://codeforces.com/problemset/problem/5/A | X | |\n| [Other](https://leetcode.com/problems/other/description/?foo=1) | - | |');
 assert.deepEqual(r.rows.map(x=>[x.metadata.problem_key,x.metadata.outcome,x.metadata.hints]),[['leetcode:two-sum','independent',false],['codeforces:4A','failed',null]]);
 assert.ok(r.flags.includes('daily_outcome_unrecorded:line_7'));
 assert.equal(parse('|  two   sum | O | |','leetcode/26-10-20.md').rows.length,1);
 for(const input of ['Same','Unknown','https://evil.com/problems/two-sum/','https://leetcode.com/problems/missing/','[Two Sum](javascript:alert)'])assert.equal(parse(`| ${input} | O | |`).rows.length,0,input);
 assert.ok(parse('| Same | O | |').flags.some(x=>x.startsWith('daily_ambiguous_problem')));
 assert.equal(parse('| Watermelon | O | |','leetcode/26-10-20.md').rows.length,0);
 assert.equal(parse('| Two Sum | O | |','daily/26-02-30.md').rows.length,0);
 assert.equal(keyFromPath('leetcode/26-10-20.md'),null);
});
test('summary validation excludes code examples and rejects malformed or conflicting rows',()=>{
 assert.equal(parse('| Two Sum | assisted | |').rows.length,0);
 assert.equal(parse('| Two Sum | O | |\n| Two Sum | X | |').rows.length,0);
 const example='# 2026-10-20\n```md\n'+log('| Two Sum | O | |').split('\n').slice(2).join('\n')+'\n```';
 assert.equal(parseDailyLog('daily/26-10-20.md',example,catalog).rows.length,0);
 assert.equal(parseDailyLog('daily/26-10-20.md','# 2026-10-20\n## Detail\n'+log('| Two Sum | O | |'),catalog).rows.length,0);
});
function fixture(t){
 const dir=mkdtempSync(join(tmpdir(),'daily-ingest-')),db=createDatabase();
 const env={...process.env,GIT_AUTHOR_NAME:'test',GIT_AUTHOR_EMAIL:'test@x',GIT_COMMITTER_NAME:'test',GIT_COMMITTER_EMAIL:'test@x'};
 const git=(...args)=>execFileSync('git',['-C',dir,...args],{encoding:'utf8',env}).trim();
 git('init','-q','-b','main');git('commit','--allow-empty','-q','-m','baseline');
 const commit=(path,body)=>{mkdirSync(dirname(join(dir,path)),{recursive:true});writeFileSync(join(dir,path),body);git('add','-A');git('commit','-q','-m','daily\n\nOutcome: failed');};
 t.after(()=>{db.close();rmSync(dir,{recursive:true,force:true});});
 return {dir,db,commit};
}
test('Git daily import preserves outcomes, dates, provenance, conflicts and edit idempotency',async t=>{
 const {dir,db,commit}=fixture(t);await registerRepository(db,'owner',{repoId:'solutions',path:dir});
 let tick=0;
 const ingest=()=>ingestRepository(db,'owner','solutions',{catalog,now:new Date(Date.parse('2026-10-22T00:00:00Z')+1000*tick++)});
 commit('leetcode/26-10-20.md',log(`| Two Sum | O | ${'long note '.repeat(70)} |\n| Other | X | |`));
 assert.equal((await ingest()).observed,2);
 let ledger=await readLedger(db,'owner');
 assert.deepEqual(ledger.attempts.map(x=>[x.problem_key,x.outcome]),[['leetcode:other','failed'],['leetcode:two-sum','independent']]);
 assert.ok(ledger.attempts.every(x=>x.occurred_at==='2026-10-19T15:00:00.000000Z'));
 assert.equal(ledger.git_observations[0].payload.trailers.daily_date,'2026-10-20');
 assert.ok(ledger.git_observations.some(x=>x.payload.trailers.daily_row.startsWith('row_sha256:')));
 commit('leetcode/26-10-20.md',log('| Other | X | extra note |\n| Two Sum | O | edited |'));
 assert.equal((await ingest()).observed,0);
 commit('leetcode/26-10-20.md',log('| Other | X | |\n| Two Sum | X | corrected |'));
 const conflict=await ingest();assert.equal(conflict.observed,0);assert.ok(Object.values(conflict.flags).flat().includes('markdown_attempt_conflict:leetcode:two-sum'));
 commit('leetcode/26-10-21.md',log('| Two Sum | X | |','2026-10-21'));
 assert.equal((await ingest()).observed,1);
 assert.equal((await ingest()).observed,0);
 ledger=await readLedger(db,'owner');assert.equal(ledger.attempts.length,3);
 assert.ok(!ledger.attempts.some(x=>x.problem_key==='leetcode:10-20'));
});
test('missing catalog leaves the watermark retryable; sync passes its catalog provider',async t=>{
 const {dir,db,commit}=fixture(t);await registerRepository(db,'owner',{repoId:'solutions',path:dir});
 commit('leetcode/26-10-20.md',log('| Two Sum | O | |'));
 await assert.rejects(ingestRepository(db,'owner','solutions'),/active catalog/);
 assert.equal((await readLedger(db,'owner')).git_watermarks.length,0);
 const sync=createGitSync({db,learnerId:'owner',catalogProvider:()=>catalog});
 const state=await sync.tick();await sync.close();
 assert.equal(state.lastError,null);assert.equal(state.lastResults[0].observed,1);
});


test('pasted links wrapping the leading pipe remain importable with original provenance',()=>{
 const body=log('[|Two Sum](https://leetcode.com/problems/two-sum/)|O|memo|\n| Other | X | |');
 const parsed=parseDailyLog('leetcode/26-10-20.md',body,catalog);
 assert.equal(parsed.rows.length,2);
 assert.equal(parsed.rows[0].metadata.problem_key,'leetcode:two-sum');
 assert.ok(parsed.rows[0].raw.startsWith('[|Two Sum]'));
});
