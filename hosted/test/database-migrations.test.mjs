import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,rmSync,readFileSync,readdirSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {DatabaseSync} from 'node:sqlite';
import {createDatabase} from './helpers/d1.mjs';

test('newer learner schema is refused without modifying the database',t=>{
  const root=mkdtempSync(join(tmpdir(),'newer-schema-'));
  t.after(()=>rmSync(root,{recursive:true,force:true}));
  const path=join(root,'learner.sqlite3'),db=new DatabaseSync(path);
  db.exec("CREATE TABLE schema_migrations(version INTEGER,name TEXT); INSERT INTO schema_migrations VALUES(999,'future')");db.close();
  assert.throws(()=>createDatabase(path),/newer than this application/);
  const check=new DatabaseSync(path);t.after(()=>check.close());
  assert.deepEqual(check.prepare('SELECT version,name FROM schema_migrations').all().map(r=>({...r})),[{version:999,name:'future'}]);
  assert.equal(check.prepare("SELECT count(*) AS n FROM sqlite_master WHERE type='table'").get().n,1);
});


test('pending migration snapshots committed WAL rows; current and fresh stores do not back up',t=>{
  const root=mkdtempSync(join(tmpdir(),'upgrade-wal-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
  const path=join(root,'learner.sqlite3'),old=new DatabaseSync(path);t.after(()=>old.close());
  old.exec('PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0');
  old.exec(readFileSync(new URL('../migrations/001_core.sql',import.meta.url),'utf8'));
  old.exec("INSERT INTO learners(learner_id) VALUES('preserved')");
  const upgraded=createDatabase(path);assert.ok(upgraded.migrationBackup);
  const saved=new DatabaseSync(upgraded.migrationBackup,{readOnly:true});
  assert.equal(saved.prepare('SELECT learner_id FROM learners').get().learner_id,'preserved');
  assert.equal(saved.prepare('SELECT MAX(version) AS v FROM schema_migrations').get().v,1);
  assert.equal(saved.prepare('PRAGMA integrity_check').get().integrity_check,'ok');saved.close();
  assert.equal(upgraded.sqlite.prepare('SELECT MAX(version) AS v FROM schema_migrations').get().v,8);upgraded.close();
  const current=createDatabase(path);assert.equal(current.migrationBackup,null);current.close();
  const fresh=createDatabase(join(root,'fresh.sqlite3'));assert.equal(fresh.migrationBackup,null);fresh.close();
  assert.equal(readdirSync(root).filter(n=>n.startsWith('.migration-backup-')).length,1);
});

test('backup failure prevents any pending migration and closes the handle',t=>{
  const root=mkdtempSync(join(tmpdir(),'backup-failure-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
  const path=join(root,'learner.sqlite3'),old=new DatabaseSync(path);
  old.exec(readFileSync(new URL('../migrations/001_core.sql',import.meta.url),'utf8'));old.close();
  const prepare=DatabaseSync.prototype.prepare;
  const mocked=t.mock.method(DatabaseSync.prototype,'prepare',function(sql){
    if(sql==='VACUUM INTO ?')throw new Error('synthetic disk full');
    return prepare.call(this,sql);
  });
  assert.throws(()=>createDatabase(path),/pre-migration backup failed/);mocked.mock.restore();
  const check=new DatabaseSync(path);t.after(()=>check.close());
  assert.equal(check.prepare('SELECT MAX(version) AS v FROM schema_migrations').get().v,1);
});
