import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync} from 'node:fs';

// Local D1 surface backed by the actual hosted SQL; batch is one SQLite transaction.
export function createDatabase(path=':memory:') {
  const sqlite = new DatabaseSync(path,{timeout:5000});
  sqlite.exec('PRAGMA foreign_keys=ON');
  const dir=new URL('../../migrations/',import.meta.url);
  const migrations=readdirSync(dir).filter(name=>/^\d{3}_[a-z0-9_]+\.sql$/.test(name)).sort();
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
  };
}
