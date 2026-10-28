import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync,mkdtempSync,chmodSync,statSync} from 'node:fs';
import {dirname,join,resolve} from 'node:path';

// Local D1 surface backed by the actual hosted SQL; batch is one SQLite transaction.
export function createDatabase(path=':memory:') {
  const existing=path!==':memory:' && (()=>{try{return statSync(path).size>0;}catch(error){if(error.code==='ENOENT')return false;throw error;}})();
  const sqlite = new DatabaseSync(path,{timeout:5000});
  try {
    sqlite.exec('PRAGMA foreign_keys=ON');
    const dir=new URL('../migrations/',import.meta.url);
    const migrations=readdirSync(dir).filter(name=>/^\d{3}_[a-z0-9_]+\.sql$/.test(name)).sort();
    const hasReceipts=sqlite.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'").get();
    const newest=migrations.length ? Number(migrations.at(-1).slice(0,3)) : 0;
    if (hasReceipts && sqlite.prepare('SELECT MAX(version) AS version FROM schema_migrations').get().version>newest) {
      throw new Error('database schema is newer than this application; upgrade the application before opening it');
    }
    const applied=new Map(hasReceipts ? sqlite.prepare('SELECT version,name FROM schema_migrations').all().map(r=>[r.version,r.name]) : []);
    for (const name of migrations) {
      const recorded=applied.get(Number(name.slice(0,3)));
      if (recorded!==undefined && recorded!==name.slice(0,-4)) throw new Error('migration receipt drift');
    }
    const pending=migrations.filter(name=>!applied.has(Number(name.slice(0,3))));
    let migrationBackup=null;
    if (existing && pending.length) {
      // VACUUM INTO snapshots through SQLite, including committed WAL pages.
      // A unique adjacent private directory avoids overwrites and makes recovery discoverable.
      const directory=mkdtempSync(join(dirname(resolve(path)),'.migration-backup-'));
      chmodSync(directory,0o700);
      migrationBackup=join(directory,'before.sqlite3');
      try {
        sqlite.prepare('VACUUM INTO ?').run(migrationBackup);
        chmodSync(migrationBackup,0o600);
      } catch (error) {
        throw new Error(`pre-migration backup failed; database not migrated: ${error.message}`);
      }
    }
    for (const name of migrations) {
      const version=Number(name.slice(0,3));
      const exists=sqlite.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'").get();
      const row=exists ? sqlite.prepare('SELECT name FROM schema_migrations WHERE version=?').get(version) : null;
      if (row) { if (row.name!==name.slice(0,-4)) throw new Error('migration receipt drift'); continue; }
      sqlite.exec('BEGIN IMMEDIATE');
      try {
        sqlite.exec(readFileSync(new URL(name,dir),'utf8'));
        const receipt=sqlite.prepare('SELECT name FROM schema_migrations WHERE version=?').get(version);
        if (receipt?.name!==name.slice(0,-4)) throw new Error('migration receipt missing');
        sqlite.exec('COMMIT');
      } catch (error) { sqlite.exec('ROLLBACK'); throw error; }
    }
    const statement = (sql,args=[]) => ({
      bind(...values) { return statement(sql,values); },
      async all() { return {results:sqlite.prepare(sql).all(...args)}; },
      async first() { return sqlite.prepare(sql).get(...args) ?? null; },
      async run() { return sqlite.prepare(sql).run(...args); },
      execute() { return /^\s*SELECT\b/i.test(sql) ? {results:sqlite.prepare(sql).all(...args)} : sqlite.prepare(sql).run(...args); },
    });
    return {
      prepare(sql) { return statement(sql); },
      async batch(statements) {
        sqlite.exec('BEGIN IMMEDIATE');
        try {
          const results = statements.map(s=>s.execute());
          sqlite.exec('COMMIT');
          return results;
        } catch (error) { sqlite.exec('ROLLBACK'); throw error; }
      },
      close() { sqlite.close(); },
      sqlite,
      migrationBackup,
    };
  } catch (error) { sqlite.close(); throw error; }
}
