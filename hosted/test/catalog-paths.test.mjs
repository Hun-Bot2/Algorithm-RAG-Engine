import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,rmSync,readdirSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {DatabaseSync} from 'node:sqlite';
import {attachCatalog} from '../mastery/catalog.mjs';

for(const name of ['space name','한글', 'hash#name', 'percent%41', "quote'name"]){
  test(`catalog path ${name} stays read-only without stray files`,t=>{
    const root=mkdtempSync(join(tmpdir(),'catalog-path-'));
    t.after(()=>rmSync(root,{recursive:true,force:true}));
    const path=join(root,`${name}.sqlite3`),source=new DatabaseSync(path);
    source.exec("CREATE TABLE catalog_meta(key TEXT,value TEXT); INSERT INTO catalog_meta VALUES('version','fixture')");source.close();
    const db=new DatabaseSync(':memory:');t.after(()=>db.close());
    assert.equal(attachCatalog(db,path,{verify:false}).version,'fixture');
    assert.throws(()=>db.exec("INSERT INTO catalog.catalog_meta VALUES('x','y')"),/readonly/i);
    assert.deepEqual(readdirSync(root),[`${name}.sqlite3`]);
  });
}


