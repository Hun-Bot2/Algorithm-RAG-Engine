import test from 'node:test';
import assert from 'node:assert/strict';
import {realpathSync,mkdtempSync,cpSync,existsSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawnSync} from 'node:child_process';

test('CLI executes and uses decoded default root in a copied checkout',t=>{
  const root=realpathSync(mkdtempSync(join(tmpdir(),'desktop 한글 # %41 ')));
  t.after(()=>rmSync(root,{recursive:true,force:true}));
  cpSync(new URL('../',import.meta.url),join(root,'hosted'),{recursive:true,filter:src=>!String(src).includes('node_modules')&&!String(src).includes('/.wrangler')});
  const result=spawnSync(process.execPath,[join(root,'hosted/cli.mjs'),'status'],{encoding:'utf8',cwd:tmpdir()});
  assert.equal(result.status,0,result.stderr);
  assert.ok(result.stdout.trim(),'main guard must execute');
  assert.ok(existsSync(join(root,'private-learning/hosted.sqlite3')),'default DB belongs to decoded checkout root');
});

