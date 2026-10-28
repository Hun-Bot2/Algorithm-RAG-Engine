// Node-only access to an immutable catalog-vN.sqlite3 (design.md section 2.3, MS-302).
// The catalog is ATTACHed read-only (SQLite enforces it) and loaded into plain data for
// the pure policy modules. Learner data never flows into the catalog file.
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {readFileSync,existsSync} from 'node:fs';
import {resolve,dirname,join} from 'node:path';

const fileSha256 = path => createHash('sha256').update(readFileSync(path)).digest('hex');

export function catalogPathFor(root,version) { return resolve(root,version,`${version}.sqlite3`); }

// Verifies the file against its receipt, attaches it read-only as `catalog`.
export function attachCatalog(sqlite,path,{verify=true}={}) {
  const absolute=resolve(path);
  if (!existsSync(absolute)) throw new Error(`catalog not found: ${absolute}`);
  let sha256=null;
  if (verify) {
    const receiptPath=join(dirname(absolute),'receipt.json');
    if (!existsSync(receiptPath)) throw new Error('catalog receipt missing');
    const receipt=JSON.parse(readFileSync(receiptPath,'utf8'));
    sha256=fileSha256(absolute);
    if (receipt.output_sha256!==sha256) throw new Error('catalog hash does not match its receipt');
  }
  const attached=sqlite.prepare('PRAGMA database_list').all().some(row=>row.name==='catalog');
  if (attached) sqlite.exec('DETACH DATABASE catalog');
  sqlite.exec(`ATTACH DATABASE '${pathToFileURL(absolute).href.replace(/'/g,"''")}?mode=ro&immutable=1' AS catalog`);
  const meta=Object.fromEntries(sqlite.prepare('SELECT key,value FROM catalog.catalog_meta').all().map(r=>[r.key,r.value]));
  return {version:meta.version,meta,sha256};
}

// Loads problems, concepts, aliases and contests. Neighbors stay in SQL (looked up on demand).
export function loadCatalog(sqlite) {
  const meta=Object.fromEntries(sqlite.prepare('SELECT key,value FROM catalog.catalog_meta').all().map(r=>[r.key,r.value]));
  const problems=new Map();
  for (const r of sqlite.prepare('SELECT * FROM catalog.problems').all()) problems.set(r.problem_key,{
    key:r.problem_key,platform:r.platform,title:r.title,url:r.url,contest_id:r.contest_id,contest_start:r.contest_start,
    difficulty_kind:r.difficulty_kind,difficulty_value:r.difficulty_value,atc_discrimination:r.atc_discrimination,
    atc_experimental:r.atc_experimental===1,access:r.access,eligible:r.eligible===1,status:r.status,
    first_seen_version:r.first_seen_version,reliability:'unknown',tags:[],concepts:[]});
  for (const r of sqlite.prepare('SELECT problem_key,native_tag FROM catalog.problem_tags ORDER BY problem_key,native_tag').all()) problems.get(r.problem_key)?.tags.push(r.native_tag);
  for (const r of sqlite.prepare('SELECT problem_key,concept_id,evidence_status,via FROM catalog.problem_concepts ORDER BY problem_key,concept_id,via').all()) {
    const p=problems.get(r.problem_key);
    if (p && !p.concepts.some(c=>c.id===r.concept_id && c.status===r.evidence_status)) p.concepts.push({id:r.concept_id,status:r.evidence_status,via:r.via});
  }
  for (const r of sqlite.prepare('SELECT problem_key,tier FROM catalog.reliability').all()) { const p=problems.get(r.problem_key); if (p) p.reliability=r.tier; }
  const aliases=new Map(sqlite.prepare('SELECT alias_key,canonical_key FROM catalog.aliases').all().map(r=>[r.alias_key,r.canonical_key]));
  const concepts=sqlite.prepare('SELECT concept_id,label FROM catalog.concepts ORDER BY concept_id').all().map(r=>({id:r.concept_id,label:r.label}));
  const contests=sqlite.prepare('SELECT * FROM catalog.contests ORDER BY contest_id').all().map(r=>({...r}));
  const contestProblems=new Map();
  for (const r of sqlite.prepare('SELECT contest_id,problem_key,label FROM catalog.contest_problems ORDER BY contest_id,label,problem_key').all()) {
    if (!contestProblems.has(r.contest_id)) contestProblems.set(r.contest_id,[]);
    contestProblems.get(r.contest_id).push({problem_key:r.problem_key,label:r.label});
  }
  const neighborStatement=sqlite.prepare('SELECT dst,rank,score FROM catalog.neighbors WHERE src=? AND channel=? ORDER BY rank');
  return {version:meta.version,meta,problems,aliases,concepts,contests,contestProblems,
    neighbors:(key,channel)=>neighborStatement.all(key,channel).map(r=>({dst:r.dst,rank:r.rank,score:r.score}))};
}

// Pure-data catalog for tests and Worker-safe callers.
export function catalogFromData({version='catalog-test',problems=[],aliases={},concepts=[],contests=[],contestProblems={},neighbors={}}) {
  return {version,meta:{version},problems:new Map(problems.map(p=>[p.key,{tags:[],concepts:[],status:'active',eligible:true,access:'free',first_seen_version:version,reliability:'reliable',...p}])),
    aliases:new Map(Object.entries(aliases)),concepts,contests,contestProblems:new Map(Object.entries(contestProblems)),
    neighbors:(key,channel)=>(neighbors[channel]?.[key] ?? []).map((dst,i)=>({dst,rank:i+1,score:1-i/100}))};
}
