// Node-only Git ingester (design.md section 3, R2.2-R2.5). Runs local `git` only; no network.
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {identifier,timestamp} from '../core/events.mjs';
import {CONVENTION_VERSION,classify,frontMatterKey,keyFromPath,parseTrailers} from './git-convention.mjs';
import {appendLedger} from './side-ledger.mjs';
import {readLedger} from './ledger.mjs';

const git = (cwd,args,{allowFail=false}={}) => {
  try { return execFileSync('git',['-C',cwd,...args],{encoding:'utf8',maxBuffer:256*1024*1024,stdio:['ignore','pipe','pipe']}); }
  catch (error) { if (allowFail) return null; throw new Error(`git ${args[0]} failed: ${String(error.stderr ?? error.message).trim().split('\n')[0]}`); }
};
const utc = iso => timestamp(new Date(iso).toISOString().replace(/\.\d{3}Z$/,'Z'));
const hash = (...parts) => createHash('sha256').update(parts.join('\0')).digest('hex');

export async function registerRepository(db,learnerId,{repoId,path,branch='main',now=new Date()}) {
  identifier(learnerId); identifier(repoId);
  const existing=await db.prepare('SELECT * FROM git_repositories WHERE learner_id=? AND repo_id=?').bind(learnerId,repoId).first();
  if (existing) { if (existing.path_hint!==path || existing.branch!==branch) throw new Error('repository already registered with different path or branch'); return existing; }
  git(path,['rev-parse','--verify',`${branch}^{commit}`]);
  await db.batch([db.prepare('INSERT OR IGNORE INTO learners(learner_id) VALUES (?)').bind(learnerId),
    db.prepare('INSERT INTO git_repositories(learner_id,repo_id,branch,path_hint,convention_version,registered_at) VALUES (?,?,?,?,?,?)')
      .bind(learnerId,repoId,branch,path,CONVENTION_VERSION,utc(now.toISOString()))]);
  return db.prepare('SELECT * FROM git_repositories WHERE learner_id=? AND repo_id=?').bind(learnerId,repoId).first();
}

// Latest watermark row for a repo (the log is append-only; the last row wins).
function currentWatermark(ledger,repoId) {
  return ledger.git_watermarks.filter(w=>w.payload.repo_id===repoId).sort((a,b)=>a.recorded_at.localeCompare(b.recorded_at) || a.entry_id.localeCompare(b.entry_id)).at(-1) ?? null;
}

// Reads commits after the watermark, maps them to observations, and appends observations
// plus the new watermark in one transaction. Returns a report; never rescans silently.
export async function ingestRepository(db,learnerId,repoId,{now=new Date(),anchorSha=null}={}) {
  const repo=await db.prepare('SELECT * FROM git_repositories WHERE learner_id=? AND repo_id=?').bind(learnerId,repoId).first();
  if (!repo) throw new Error('unknown repository');
  const recordedAt=utc(now.toISOString());
  const ledger=await readLedger(db,learnerId);
  const mark=currentWatermark(ledger,repoId);
  const watermarkEntry=(sha,status,observed,detail)=>({schema_version:1,entry_id:'wm-'+hash(repoId,recordedAt,sha ?? '',status).slice(0,32),learner_id:learnerId,
    occurred_at:recordedAt,recorded_at:recordedAt,source:'git',supersedes:null,payload:{repo_id:repoId,branch:repo.branch,last_processed_sha:sha,status,observed,detail}});
  const head=git(repo.path_hint,['rev-parse',`${repo.branch}^{commit}`]).trim();
  if (mark?.payload.status==='history_rewritten' && !anchorSha) return {repo_id:repoId,status:'history_rewritten',observed:0,message:'re-anchor explicitly with --anchor <sha>'};
  let from=anchorSha ?? mark?.payload.last_processed_sha ?? null;
  if (from) {
    if (!/^[0-9a-f]{40}$/.test(from)) from=git(repo.path_hint,['rev-parse',`${from}^{commit}`]).trim();
    const ancestor=git(repo.path_hint,['merge-base','--is-ancestor',from,head],{allowFail:true});
    if (ancestor===null) {
      await appendLedger(db,learnerId,{git_watermarks:[watermarkEntry(mark?.payload.last_processed_sha ?? null,'history_rewritten',0,`watermark ${from} is not an ancestor of ${repo.branch}`)]});
      return {repo_id:repoId,status:'history_rewritten',observed:0};
    }
    if (anchorSha) {
      await appendLedger(db,learnerId,{git_watermarks:[watermarkEntry(from,'anchored',0,'explicit anchor')]});
      if (from===head) return {repo_id:repoId,status:'anchored',observed:0};
    }
  }
  if (from===head) return {repo_id:repoId,status:'up_to_date',observed:0};
  const range=from ? `${from}..${head}` : head;
  const commits=git(repo.path_hint,['log','--reverse','--topo-order',`--format=%H%x1f%P%x1f%aI%x1f%(trailers:unfold,only)%x1e`,range])
    .split('\x1e').map(s=>s.replace(/^\n/,'')).filter(s=>s.trim()).map(s=>{ const [sha,parents,authored,trailerBlock]=s.split('\x1f'); return {sha,parents:parents.trim().split(' ').filter(Boolean),authored,trailerBlock}; });
  // Problems already observed without trailers in this repo: later trailer-less edits are not new attempts.
  const seenUntrailered=new Set(ledger.git_observations.filter(o=>o.payload.repo_id===repoId).map(o=>o.payload.problem_key));
  const observations=[], unclassified=[], flagsByCommit={};
  for (const c of commits) {
    const {trailers,problems,unknown}=parseTrailers(c.trailerBlock);
    let keys=[], paths={};
    if (problems.length) keys=problems.filter(k=>/^(leetcode|atcoder|codeforces|baekjoon):[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(k));
    else if (c.parents.length<=1) {
      const changed=git(repo.path_hint,['diff-tree','-r','--root','--no-commit-id','--name-status','--no-renames',c.sha]).split('\n').filter(Boolean)
        .map(line=>line.split('\t')).filter(([status])=>status!=='D').map(([,path])=>path);
      for (const path of changed) {
        let key=keyFromPath(path);
        if (!key && /\.(md|mdx)$/i.test(path)) key=frontMatterKey(path,git(repo.path_hint,['show',`${c.sha}:${path}`],{allowFail:true}));
        if (key && !paths[key]) { paths[key]=path; keys.push(key); }
      }
    }
    if (!keys.length) { if (problems.length || Object.keys(trailers).length) unclassified.push(c.sha); continue; }
    const result=classify(trailers);
    const flags=[...result.flags,...unknown.map(u=>'unknown_trailer:'+u)];
    if (problems.length && keys.length!==problems.length) flags.push('invalid_problem_trailer');
    const hasMetadata=Object.keys(trailers).length>0 || problems.length>0;
    const occurredAt=utc(new Date(Math.min(Date.parse(c.authored),now.getTime())).toISOString());
    for (const key of keys) {
      if (!hasMetadata) { if (seenUntrailered.has(key)) continue; seenUntrailered.add(key); }
      const transfer=result.transfer_from===key ? null : result.transfer_from;
      observations.push({schema_version:1,entry_id:'git-'+hash(repoId,c.sha,key).slice(0,40),learner_id:learnerId,occurred_at:occurredAt,recorded_at:recordedAt,source:'git',supersedes:null,
        payload:{repo_id:repoId,sha:c.sha,path:paths[key] ?? null,problem_key:key,outcome:result.outcome,hints:result.hints,self_minutes:result.self_minutes,
          activity_type:transfer ? result.activity_type : result.activity_type==='transfer' ? null : result.activity_type,transfer_from:transfer,
          trailers:Object.fromEntries(Object.entries({...trailers,...(problems.length?{problem:problems.join(', ')}:{})}).map(([k,v])=>[k,String(v)]))}});
    }
    if (flags.length) flagsByCommit[c.sha]=flags;
  }
  const added=await appendLedger(db,learnerId,{git_observations:observations,git_watermarks:[watermarkEntry(head,'ok',observations.length,unclassified.length?`unclassified commits: ${unclassified.length}`:null)]});
  return {repo_id:repoId,status:'ok',from,to:head,commits:commits.length,observed:added.git_observations,unclassified,flags:flagsByCommit};
}
