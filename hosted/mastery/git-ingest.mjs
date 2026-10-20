// Node-only Git ingester (design.md section 3, R2.2-R2.5). Runs local `git` only; no network.
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {identifier,timestamp} from '../core/events.mjs';
import {CONVENTION_VERSION,classify,frontMatterKey,keyFromPath,parseTrailers} from './git-convention.mjs';
import {markdownAttemptSignature,markdownBlobHash,parseMarkdownAttempt} from './markdown-attempt.mjs';
import {appendLedger,listEntries} from './side-ledger.mjs';
import {readLedger} from './ledger.mjs';
import {dailyPath,parseDailyLog} from './daily-log.mjs';
import {attachCatalog,loadCatalog} from './catalog.mjs';
import {fileURLToPath} from 'node:url';
import {join} from 'node:path';

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
export async function ingestRepository(db,learnerId,repoId,{now=new Date(),anchorSha=null,catalog=null,catalogRoot=fileURLToPath(new URL('../../private-learning/mastery/',import.meta.url))}={}) {
  const repo=await db.prepare('SELECT * FROM git_repositories WHERE learner_id=? AND repo_id=?').bind(learnerId,repoId).first();
  if (!repo) throw new Error('unknown repository');
  const recordedAt=utc(now.toISOString());
  const ledger=await readLedger(db,learnerId);
  const mark=currentWatermark(ledger,repoId);
  const getDailyCatalog=()=>{
    if(catalog)return catalog;
    const current=ledger.catalog_switches.at(-1);
    if(!current)throw new Error('daily logs require an active catalog; run catalog use, then retry repo ingest');
    const version=current.payload.to_version;
    const info=attachCatalog(db.sqlite,join(catalogRoot,version,`${version}.sqlite3`));
    if(info.sha256!==current.payload.catalog_sha256)throw new Error('active catalog file differs from the logged switch');
    catalog=loadCatalog(db.sqlite);return catalog;
  };
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
  const priorMarkdown=new Map();
  const allGitRows=await listEntries(db,'git_observations',learnerId);
  for (const row of allGitRows.filter(o=>o.payload.repo_id===repoId)) {
    const token=row.payload.trailers?.markdown_attempt, signature=row.payload.trailers?.markdown_semantic;
    if (token && signature) priorMarkdown.set(`${row.payload.problem_key}\0${token}`,signature);
  }
  const observations=[], unclassified=[], flagsByCommit={};
  for (const c of commits) {
    const {trailers,problems,unknown}=parseTrailers(c.trailerBlock);
    let keys=[], paths={}, markdownByKey=new Map(), invalidMarkdown=new Set();
    if (c.parents.length<=1) {
      const changed=git(repo.path_hint,['diff-tree','-r','--root','--no-commit-id','--name-status','--no-renames',c.sha]).split('\n').filter(Boolean)
        .map(line=>line.split('\t')).filter(([status])=>status!=='D').map(([,path])=>path);
      for (const path of changed) {
        const isMarkdown=/\.(md|mdx)$/i.test(path);
        if (!isMarkdown) { const key=keyFromPath(path); if (key && !paths[key]) paths[key]=path; continue; }
        const parts=path.toLowerCase().split('/');
        const pathKey=keyFromPath(path);
        if (parts.some(x=>['template','templates','example','examples'].includes(x)) || /(?:^|[-_.])template(?:[-_.]|$)/i.test(parts.at(-1)) || (!pathKey && /^(?:readme|index|notes?)(?:\.|$)/i.test(parts.at(-1)))) continue;
        const blob=git(repo.path_hint,['show',`${c.sha}:${path}`],{allowFail:true});
        if (blob===null) continue;
        if(dailyPath(path)) {
          const parsed=parseDailyLog(path,blob,getDailyCatalog());
          if(parsed.flags.length)(flagsByCommit[c.sha]??=[]).push(...parsed.flags.map(flag=>`${path}:${flag}`));
          for(const row of parsed.rows) {
            const md=row.metadata,key=md.problem_key,semantic=markdownAttemptSignature(md),token=`${key}\0${md.attempt}`;
            const prior=priorMarkdown.get(token);
            if(prior!==undefined){if(prior!==semantic)(flagsByCommit[c.sha]??=[]).push('markdown_attempt_conflict:'+key);continue;}
            priorMarkdown.set(token,semantic);
            const observedAt=utc(new Date(Math.min(Date.parse(`${row.date}T00:00:00+09:00`),now.getTime())).toISOString());
            const provenance={markdown_source:path.length<=500?path:`path_sha256:${hash(path)}`,markdown_sha256:markdownBlobHash(blob),markdown_attempt:md.attempt,markdown_semantic:semantic,
              daily_date:row.date,daily_line:String(row.line),daily_row:row.raw.length<=500?row.raw:`row_sha256:${hash(row.raw)}`,daily_catalog_version:catalog.version};
            observations.push({schema_version:1,entry_id:'git-'+hash(repoId,key,md.attempt).slice(0,40),learner_id:learnerId,occurred_at:observedAt,recorded_at:recordedAt,source:'git',supersedes:null,
              payload:{repo_id:repoId,sha:c.sha,path,problem_key:key,outcome:md.outcome,hints:md.hints,self_minutes:null,activity_type:'practice',transfer_from:null,trailers:provenance}});
          }
          continue;
        }
        const parsed=parseMarkdownAttempt(blob,{inferredProblem:pathKey});
        if (parsed.marked) {
          const inferred=pathKey ?? frontMatterKey(path,blob);
          if (parsed.error) { const invalidKey=parsed.problem_key ?? inferred; if (invalidKey) invalidMarkdown.add(invalidKey); continue; }
          if (pathKey && parsed.metadata.problem_key!==pathKey) { invalidMarkdown.add(pathKey); invalidMarkdown.add(parsed.metadata.problem_key); continue; }
          const list=markdownByKey.get(parsed.metadata.problem_key) ?? [];
          list.push({path,blob,metadata:parsed.metadata}); markdownByKey.set(parsed.metadata.problem_key,list);
          continue;
        }
        const key=pathKey ?? frontMatterKey(path,blob);
        if (key && !paths[key]) paths[key]=path;
      }
    }
    if (problems.length) keys=problems.filter(k=>/^(leetcode|atcoder|codeforces|baekjoon):[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(k));
    else keys=[...new Set([...Object.keys(paths),...markdownByKey.keys(),...invalidMarkdown])];
    for (const key of keys) if (!paths[key]) paths[key]=markdownByKey.get(key)?.[0]?.path ?? null;
    if (!keys.length) { if (problems.length || Object.keys(trailers).length) unclassified.push(c.sha); continue; }
    const result=classify(trailers);
    const flags=[...result.flags.filter(f=>f!=='no_outcome_trailer'),...unknown.map(u=>'unknown_trailer:'+u)];
    if (problems.length && keys.length!==problems.length) flags.push('invalid_problem_trailer');
    const hasAttemptTrailer=trailers.outcome!==undefined || trailers.hints!==undefined;
    const hasOtherTrailer=trailers.minutes!==undefined || trailers.activity!==undefined || trailers['transfer-from']!==undefined;
    const hasMetadata=hasAttemptTrailer || hasOtherTrailer || problems.length>0;
    const occurredAt=utc(new Date(Math.min(Date.parse(c.authored),now.getTime())).toISOString());
    for (const key of keys) {
      if (invalidMarkdown.has(key)) { (flagsByCommit[c.sha]??=[]).push('invalid_markdown_metadata:'+key); continue; }
      const files=markdownByKey.get(key) ?? [];
      if (problems.length && !problems.includes(key)) continue;
      const byToken=new Map();
      for (const file of files) {
        const list=byToken.get(file.metadata.attempt) ?? [];
        list.push(file); byToken.set(file.metadata.attempt,list);
      }
      const candidates=[];
      for (const [token,group] of byToken) {
        const signatures=new Set(group.map(f=>markdownAttemptSignature(f.metadata)));
        if (signatures.size>1) { (flagsByCommit[c.sha]??=[]).push('conflicting_markdown_metadata:'+key); candidates.length=0; break; }
        candidates.push(group[0]);
      }
      if (files.length && !candidates.length) continue;
      if (!files.length) candidates.push({metadata:null,blob:null});
      if (!files.length && !hasMetadata) { if (seenUntrailered.has(key)) continue; seenUntrailered.add(key); }
      for (const file of candidates) {
        const md=file.metadata;
        let outcome=result.outcome, hints=result.hints;
        if (md && !hasAttemptTrailer) { outcome=md.outcome; hints=md.hints; }
        const minutes=trailers.minutes!==undefined ? result.self_minutes : md?.self_minutes ?? result.self_minutes;
        let activity=trailers.activity!==undefined ? result.activity_type : md?.activity_type ?? result.activity_type;
        let transfer=trailers.activity!==undefined || trailers['transfer-from']!==undefined ? result.transfer_from : md?.transfer_from ?? result.transfer_from;
        if (transfer===key) { activity=null; transfer=null; }
        const semantic=md?JSON.stringify([key,outcome,hints,minutes,activity,transfer]):null;
        const source=file.path ?? paths[key] ?? '';
        const provenance=md?{markdown_source:source.length<=500?source:`path_sha256:${hash(source)}`,markdown_sha256:markdownBlobHash(file.blob),markdown_attempt:md.attempt,
          markdown_semantic:semantic}:{};
        const entryId=md ? 'git-'+hash(repoId,key,md.attempt).slice(0,40) : 'git-'+hash(repoId,c.sha,key).slice(0,40);
        if (md) {
          const attemptKey=`${key}\0${md.attempt}`, prior=priorMarkdown.get(attemptKey);
          if (prior!==undefined) {
            if (prior!==semantic) (flagsByCommit[c.sha]??=[]).push('markdown_attempt_conflict:'+key);
            continue;
          }
          priorMarkdown.set(attemptKey,semantic);
        }
        observations.push({schema_version:1,entry_id:entryId,learner_id:learnerId,occurred_at:occurredAt,recorded_at:recordedAt,source:'git',supersedes:null,
          payload:{repo_id:repoId,sha:c.sha,path:file.path ?? paths[key] ?? null,problem_key:key,outcome,hints,self_minutes:minutes,
            activity_type:activity,transfer_from:transfer,
            trailers:Object.fromEntries(Object.entries({...trailers,...(problems.length?{problem:problems.join(', ')}:{}),...provenance}).map(([k,v])=>[k,String(v)]))}});
      }
    }
    if (flags.length) flagsByCommit[c.sha]=[...(flagsByCommit[c.sha]??[]),...flags];
  }
  const added=await appendLedger(db,learnerId,{git_observations:observations,git_watermarks:[watermarkEntry(head,'ok',observations.length,unclassified.length?`unclassified commits: ${unclassified.length}`:null)]});
  return {repo_id:repoId,status:'ok',from,to:head,commits:commits.length,observed:added.git_observations,unclassified,flags:flagsByCommit};
}
