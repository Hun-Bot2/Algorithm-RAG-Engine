// Node-only local research ledger; independent of learner-state schemas and backups.
import {DatabaseSync} from 'node:sqlite';
import {existsSync,mkdirSync,chmodSync} from 'node:fs';
import {dirname} from 'node:path';
import {canonicalJson} from '../contest-history.mjs';

const SCHEMA=`
CREATE TABLE IF NOT EXISTS explanation_inputs (
  id TEXT PRIMARY KEY, learner_hash TEXT NOT NULL, evidence_hash TEXT NOT NULL,
  prompt_version TEXT NOT NULL, payload TEXT NOT NULL CHECK(json_valid(payload))
) STRICT;
CREATE TABLE IF NOT EXISTS explanation_attempts (
  id TEXT PRIMARY KEY, input_id TEXT NOT NULL REFERENCES explanation_inputs(id),
  cache_key TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
  started_ms INTEGER NOT NULL, lease_until_ms INTEGER NOT NULL,
  payload TEXT NOT NULL CHECK(json_valid(payload))
) STRICT;
CREATE INDEX IF NOT EXISTS explanation_attempt_cache ON explanation_attempts(cache_key,started_ms);
CREATE TABLE IF NOT EXISTS explanation_results (
  attempt_id TEXT PRIMARY KEY REFERENCES explanation_attempts(id),
  status TEXT NOT NULL CHECK(status IN ('success','failed')),
  payload TEXT NOT NULL CHECK(json_valid(payload))
) STRICT;
CREATE TABLE IF NOT EXISTS explanation_cost_receipts (
  receipt_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL REFERENCES explanation_attempts(id),
  payload TEXT NOT NULL CHECK(json_valid(payload))
) STRICT;
`;
export function createExplanationStore(path){
  function open(write=false){
    if(!write&&!existsSync(path))return null;
    if(write)mkdirSync(dirname(path),{recursive:true,mode:0o700});
    const db=new DatabaseSync(path,{readOnly:!write});
    db.exec('PRAGMA busy_timeout=5000; PRAGMA foreign_keys=ON;');
    if(write){
      chmodSync(path,0o600);
      db.exec('PRAGMA journal_mode=WAL;');
      db.exec('BEGIN IMMEDIATE');
      try{
        const version=db.prepare('PRAGMA user_version').get().user_version;
        if(version>1)throw new Error('unsupported explanation database version');
        if(version===0){
          db.exec(SCHEMA);
          for(const table of ['explanation_inputs','explanation_attempts','explanation_results','explanation_cost_receipts'])for(const action of ['UPDATE','DELETE'])
            db.exec(`CREATE TRIGGER ${table}_${action.toLowerCase()} BEFORE ${action} ON ${table} BEGIN SELECT RAISE(ABORT,'append-only explanation ledger'); END;`);
          db.exec('PRAGMA user_version=1');
        }
        db.exec('COMMIT');
      }catch(error){db.exec('ROLLBACK');db.close();throw error;}
    }
    return db;
  }
  function latestOn(db,cacheKey){
    // Any validated success remains usable; an interrupted later attempt cannot hide it.
    const success=db.prepare(`SELECT a.payload AS attempt,r.payload AS result FROM explanation_attempts a
      JOIN explanation_results r ON r.attempt_id=a.id WHERE a.cache_key=? AND r.status='success' ORDER BY a.rowid DESC LIMIT 1`).get(cacheKey);
    if(success)return {attempt:JSON.parse(success.attempt),result:JSON.parse(success.result)};
    const last=db.prepare(`SELECT a.payload AS attempt,r.payload AS result FROM explanation_attempts a
      LEFT JOIN explanation_results r ON r.attempt_id=a.id WHERE a.cache_key=? ORDER BY a.rowid DESC LIMIT 1`).get(cacheKey);
    return last?{attempt:JSON.parse(last.attempt),result:last.result?JSON.parse(last.result):null}:null;
  }
  function latest(cacheKey){const db=open();if(!db)return null;try{return latestOn(db,cacheKey);}finally{db.close();}}
  function claim(input,attempt,{retryFailed=false,nowMs}){
    const db=open(true);db.exec('BEGIN IMMEDIATE');
    try{
      const old=latestOn(db,attempt.cache_key);
      if(old && (old.result?.status==='success'||(!old.result&&old.attempt.lease_until_ms>nowMs)||!retryFailed)){
        db.exec('COMMIT');return {claimed:false,record:old};
      }
      attempt={...attempt,retry_of:old?.attempt.attempt_id ?? null};
      db.prepare('INSERT OR IGNORE INTO explanation_inputs VALUES (?,?,?,?,?)').run(input.input_id,input.learner_hash,input.evidence_hash,input.prompt_version,JSON.stringify(input));
      db.prepare('INSERT INTO explanation_attempts VALUES (?,?,?,?,?,?,?,?)').run(attempt.attempt_id,attempt.input_id,attempt.cache_key,attempt.provider,attempt.model,attempt.started_ms,attempt.lease_until_ms,JSON.stringify(attempt));
      db.exec('COMMIT');return {claimed:true,attempt};
    }catch(error){db.exec('ROLLBACK');throw error;}finally{db.close();}
  }
  function finish(attemptId,result){const db=open(true);try{db.prepare('INSERT INTO explanation_results VALUES (?,?,?)').run(attemptId,result.status,JSON.stringify(result));}finally{db.close();}}
  function records(learnerHash){
    const db=open();if(!db)return [];
    try{return db.prepare(`SELECT i.payload AS input,a.payload AS attempt,r.payload AS result,
      (SELECT c.payload FROM explanation_cost_receipts c WHERE c.attempt_id=a.id ORDER BY c.rowid DESC LIMIT 1) AS cost_receipt FROM explanation_inputs i
      JOIN explanation_attempts a ON a.input_id=i.id LEFT JOIN explanation_results r ON r.attempt_id=a.id
      WHERE i.learner_hash=? ORDER BY a.rowid`).all(learnerHash).map(r=>({input:JSON.parse(r.input),attempt:JSON.parse(r.attempt),result:r.result?JSON.parse(r.result):null,cost_receipt:r.cost_receipt?JSON.parse(r.cost_receipt):null}));}
    finally{db.close();}
  }
  function addCostReceipts(learnerHash,receipts){
    if(!Array.isArray(receipts))throw new Error('cost receipts must be an array');
    const db=open(true);db.exec('BEGIN IMMEDIATE');
    try{
      let added=0;
      for(const receipt of receipts){
        if(!receipt||Object.keys(receipt).sort().join(',')!=='actual_usd,attempt_id,receipt_id,source' ||
          typeof receipt.receipt_id!=='string'||!receipt.receipt_id||typeof receipt.attempt_id!=='string'||typeof receipt.source!=='string'||!receipt.source.trim()||
          typeof receipt.actual_usd!=='number'||!Number.isFinite(receipt.actual_usd)||receipt.actual_usd<0)throw new Error('invalid cost receipt');
        const own=db.prepare('SELECT a.id FROM explanation_attempts a JOIN explanation_inputs i ON a.input_id=i.id WHERE a.id=? AND i.learner_hash=?').get(receipt.attempt_id,learnerHash);
        if(!own)throw new Error('unknown cost receipt attempt');
        const prior=db.prepare('SELECT payload FROM explanation_cost_receipts WHERE receipt_id=?').get(receipt.receipt_id);
        if(prior){if(prior.payload!==canonicalJson(receipt))throw new Error('conflicting cost receipt');continue;}
        db.prepare('INSERT INTO explanation_cost_receipts VALUES (?,?,?)').run(receipt.receipt_id,receipt.attempt_id,canonicalJson(receipt));added++;
      }
      db.exec('COMMIT');return {added};
    }catch(error){db.exec('ROLLBACK');throw error;}finally{db.close();}
  }
  return {latest,claim,finish,records,addCostReceipts};
}
