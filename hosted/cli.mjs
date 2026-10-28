#!/usr/bin/env node
// Local mastery-system CLI (ADR 003: the local SQLite database is canonical). No network unless a
// command explicitly says so (`rating sync-cf`). Shares the database with `npm run dev`.
import {fileURLToPath,pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {existsSync,mkdirSync,writeFileSync,readFileSync} from 'node:fs';
import {resolve,dirname,join} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {createDatabase} from './runtime/database.mjs';
import {appendEvents,getPreferences,putPreferences} from './core/store.mjs';
import {seoulDate} from './core/policy.mjs';
import {timestamp} from './core/events.mjs';
import {createJournalService} from './mastery/journal.mjs';
import {createExplanationService,withExplanations,snapshotPlanReader} from './mastery/explanations.mjs';
import {appendLedger} from './mastery/side-ledger.mjs';
import {readLedger,getSnapshot,saveSnapshot,saveWeeklyReport,getWeeklyReport} from './mastery/ledger.mjs';
import {attachCatalog,loadCatalog} from './mastery/catalog.mjs';
import {planDay,problemStates,weekInfo} from './mastery/planner.mjs';
import {fitAll} from './mastery/calibration.mjs';
import {relatedCandidates} from './mastery/related.mjs';
import {liveProposals,virtualProposal} from './mastery/contests.mjs';
import {mockContest,mockExposurePlans,snapshotExposures} from './mastery/mock-contest.mjs';
import {buildWeeklyReport,renderMarkdown} from './mastery/weekly.mjs';
import {registerRepository,ingestRepository} from './mastery/git-ingest.mjs';
import {fetchCodeforcesRatings,ratingEntries} from './mastery/ratings.mjs';
import {createViews,findProblems} from './mastery/views.mjs';
import {normalizeMastery} from './core/preferences.mjs';
import {cachedProviders} from './provider-service.mjs';

const USAGE=`usage: node hosted/cli.mjs <command> [options]
  catalog use --version catalog-vN            verify and switch the active catalog (logged)
  repo add --id ID --path DIR [--branch main]  register a solutions/notes Git repository
  repo ingest [--id ID] [--anchor SHA]        ingest new commits (all repositories by default)
  review add|remove --problem KEY [--date YYYY-MM-DD]   explicitly select a review (no attempt)
  attempt --problem KEY --outcome independent|assisted|failed|reported_completion [--hints none|used] [--minutes N] [--activity practice|review|transfer] [--from KEY] [--at ISO]
  feedback --plan DATE --candidate KEY --verdict related|partly|unrelated|skip [--source KEY] [--channel C]
  rating set --platform codeforces|atcoder --rating N [--participations N]
  rating sync-cf --handle H                   NETWORK: one Codeforces API call (opt-in)
  prefs [--mastery JSON] [--weekday-minutes N] show or update preferences
  explain                                    NETWORK: dual Gemini + NVIDIA explanations [--retry-failed]
  explanation-usage                          aggregate actual dispatch usage by provider/model
  explanation-records                        private local research records as JSON
  explanation-costs --path FILE               append user-supplied actual USD cost receipts
  explanation-otel [--out FILE]               redacted offline OTLP export (no network)
  journal                                    current curriculum journal as JSON (immutable evidence revisions)
  plan [--date YYYY-MM-DD]                    build (or show the frozen) plan for a day
  mock-contest create|show [--date YYYY-MM-DD] [--count 5-7] [--minutes N]   weekly LeetCode mock contest: date + solving time (default 120m), immutable, no attempts
  related --problem KEY [--channel concept|title_v2|title_v2_cross]
  weekly [--week YYYY-MM-DD] [--out DIR]      build the weekly report; Markdown never overwrites
  find [--platform] [--concept] [--concept-status] [--tag] [--difficulty a..b] [--predicted a..b] [--status] [--near KEY --channel C --max-rank N]
       [--source] [--archive-category] [--provenance] [--date attempt:a..b|contest:a..b|first_seen:V] [--limit N]
  status
common: --db PATH (default private-learning/hosted.sqlite3) --learner ID (default local-owner) --catalog-root DIR (default private-learning/mastery)`;

const ROOT=fileURLToPath(new URL('..',import.meta.url));
const options={db:{type:'string'},learner:{type:'string'},'catalog-root':{type:'string'},version:{type:'string'},id:{type:'string'},path:{type:'string'},branch:{type:'string'},
  anchor:{type:'string'},problem:{type:'string'},outcome:{type:'string'},minutes:{type:'string'},hints:{type:'string'},activity:{type:'string'},from:{type:'string'},at:{type:'string'},
  plan:{type:'string'},candidate:{type:'string'},verdict:{type:'string'},source:{type:'string'},channel:{type:'string'},platform:{type:'string'},rating:{type:'string'},
  participations:{type:'string'},handle:{type:'string'},mastery:{type:'string'},'weekday-minutes':{type:'string'},date:{type:'string'},week:{type:'string'},out:{type:'string'},
  concept:{type:'string'},'concept-status':{type:'string'},tag:{type:'string'},difficulty:{type:'string'},predicted:{type:'string'},status:{type:'string'},near:{type:'string'},
  'max-rank':{type:'string'},'archive-category':{type:'string'},provenance:{type:'string'},limit:{type:'string'},json:{type:'boolean'},now:{type:'string'},'retry-failed':{type:'boolean'},count:{type:'string'}};

export async function run(argv,{print=console.log,fetcher=fetch,env=process.env}={}) {
  const {values:o,positionals}=parseArgs({args:argv,options,allowPositionals:true});
  const [command,sub]=positionals;
  if (!command) { print(USAGE); return 1; }
  const dbPath=resolve(o.db ?? join(ROOT,'private-learning/hosted.sqlite3'));
  mkdirSync(dirname(dbPath),{recursive:true,mode:0o700});
  const db=createDatabase(dbPath);
  const learnerId=o.learner ?? 'local-owner';
  const now=o.now ? new Date(o.now) : new Date();
  const nowIso=timestamp(now.toISOString());
  const catalogRoot=resolve(o['catalog-root'] ?? join(ROOT,'private-learning/mastery'));
  const out=value=>print(o.json ? JSON.stringify(value,null,1) : typeof value==='string' ? value : JSON.stringify(value,null,1));
  const activeCatalog=async ledger=>{
    const switches=(ledger ?? await readLedger(db,learnerId)).catalog_switches;
    const current=switches.at(-1);
    if (!current) throw new Error('no active catalog: run `catalog use --version catalog-v1`');
    const info=attachCatalog(db.sqlite,join(catalogRoot,current.payload.to_version,`${current.payload.to_version}.sqlite3`));
    if (info.sha256!==current.payload.catalog_sha256) throw new Error('active catalog file differs from the logged switch');
    const catalog=loadCatalog(db.sqlite);
    catalog.base_version=switches[0].payload.to_version;
    return catalog;
  };
  const plansOf=async ()=>(await db.prepare("SELECT payload FROM recommendation_snapshots WHERE learner_id=? AND kind='plan' ORDER BY local_date").bind(learnerId).all()).results.map(r=>JSON.parse(r.payload));
  const mockContestsOf=async ()=>(await db.prepare("SELECT payload FROM recommendation_snapshots WHERE learner_id=? AND kind='mock_contest' ORDER BY local_date").bind(learnerId).all()).results.map(r=>JSON.parse(r.payload));
  try {
    if (command==='catalog' && sub==='use') {
      if (!o.version) throw new Error('--version required');
      const info=attachCatalog(db.sqlite,join(catalogRoot,o.version,`${o.version}.sqlite3`));
      const ledger=await readLedger(db,learnerId);
      const previous=ledger.catalog_switches.at(-1)?.payload.to_version ?? null;
      if (previous===o.version) { out({active:o.version,changed:false}); return 0; }
      await appendLedger(db,learnerId,{catalog_switches:[{schema_version:1,entry_id:`catalog-${o.version}-${createHash('sha256').update(nowIso).digest('hex').slice(0,12)}`,learner_id:learnerId,
        occurred_at:nowIso,recorded_at:nowIso,source:'manual',supersedes:null,payload:{from_version:previous,to_version:o.version,catalog_sha256:info.sha256}}]});
      out({active:o.version,previous,sha256:info.sha256}); return 0;
    }
    if (command==='repo' && sub==='add') { out(await registerRepository(db,learnerId,{repoId:o.id,path:resolve(o.path),branch:o.branch ?? 'main',now})); return 0; }
    if (command==='repo' && sub==='ingest') {
      const repos=(await db.prepare('SELECT repo_id FROM git_repositories WHERE learner_id=? ORDER BY repo_id').bind(learnerId).all()).results.map(r=>r.repo_id).filter(id=>!o.id || id===o.id);
      const reports=[];
      for (const id of repos) reports.push(await ingestRepository(db,learnerId,id,{now,anchorSha:o.anchor ?? null,catalogRoot}));
      out(reports); return reports.some(r=>r.status==='history_rewritten') ? 3 : 0;
    }
    if(command==='review'&&['add','remove'].includes(sub)){
      if(!o.problem)throw new Error('--problem required');
      const catalog=await activeCatalog(),problem=catalog.problems.get(o.problem);
      if(!problem)throw new Error('invalid review problem: exact active catalog key required');
      const target={problem_key:o.problem,date:o.date ?? seoulDate(now)},preferences=await getPreferences(db,learnerId);
      const before=preferences.review_targets ?? [],exists=before.some(t=>t.problem_key===target.problem_key&&t.date===target.date);
      // Validate even a no-op removal; unrelated preferences are preserved by putPreferences.
      const {normalizePreferences}=await import('./core/preferences.mjs');normalizePreferences({review_targets:[target]});
      const changed=sub==='add'?!exists:exists;
      const targets=sub==='add'?(exists?before:[...before,target]):before.filter(t=>t.problem_key!==target.problem_key||t.date!==target.date);
      if(changed)await putPreferences(db,learnerId,{review_targets:targets});
      out({changed,selection:target,title:problem.title,review_targets:targets,attempts_added:0});return 0;
    }
    if (command==='attempt') {
      if (o.hints!==undefined && !['none','used'].includes(o.hints)) throw new Error('--hints must be none or used');
      // Independent requires an explicit --hints none (the core validator enforces it).
      const payload={problem_key:o.problem,outcome:({O:'independent',X:'failed'}[o.outcome] ?? o.outcome),hints:o.hints===undefined ? null : o.hints==='none' ? false : true};
      if(o.minutes!==undefined)payload.self_minutes=Number(o.minutes);
      if (o.activity) payload.activity_type=o.activity;
      if (o.activity==='transfer') payload.transfer_from=o.from;
      const at=o.at ? timestamp(o.at) : nowIso;
      out({added:await appendEvents(db,learnerId,[{schema_version:1,event_id:'cli-'+randomUUID(),learner_id:learnerId,occurred_at:at,recorded_at:nowIso,kind:'attempt',source:'manual',payload,supersedes:null}])}); return 0;
    }
    if (command==='feedback') {
      out(await appendLedger(db,learnerId,{practice_feedback:[{schema_version:1,entry_id:'fb-'+randomUUID(),learner_id:learnerId,occurred_at:nowIso,recorded_at:nowIso,source:'manual',supersedes:null,
        payload:{snapshot_ref:`plan-${o.plan}`,source_key:o.source ?? null,candidate_key:o.candidate,verdict:o.verdict,channel:o.channel ?? null}}]})); return 0;
    }
    if (command==='rating' && sub==='set') {
      out(await appendLedger(db,learnerId,{rating_observations:[{schema_version:1,entry_id:'rating-'+randomUUID(),learner_id:learnerId,occurred_at:nowIso,recorded_at:nowIso,source:'manual',supersedes:null,
        payload:{platform:o.platform,handle:null,rating:Number(o.rating),rated_participations:o.participations===undefined?null:Number(o.participations),contest_id:null,old_rating:null,rank:null}}]})); return 0;
    }
    if (command==='rating' && sub==='sync-cf') {
      const changes=await fetchCodeforcesRatings(o.handle,{fetcher});
      out(await appendLedger(db,learnerId,{rating_observations:ratingEntries(learnerId,o.handle,changes,nowIso)})); return 0;
    }
    if (command==='prefs') {
      const patch={};
      if (o.mastery) patch.mastery=normalizeMastery(JSON.parse(o.mastery));
      if (o['weekday-minutes']) patch.weekday_minutes=Number(o['weekday-minutes']);
      out(Object.keys(patch).length ? await putPreferences(db,learnerId,patch) : await getPreferences(db,learnerId)); return 0;
    }
    if(command.startsWith('explanation-')){
      const explanations=createExplanationService({cacheRoot:join(dirname(dbPath),'explanations'),env,fetcher});
      if(command==='explanation-usage'){out(explanations.usage(learnerId));return 0;}
      if(command==='explanation-records'){out(explanations.records(learnerId));return 0;}
      if(command==='explanation-costs'){out(explanations.addCostReceipts(learnerId,JSON.parse(readFileSync(o.path,'utf8'))));return 0;}
      if(command==='explanation-otel'){
        const {explanationOtlp}=await import('./mastery/explanation-observability.mjs');
        const payload=explanationOtlp(explanations.records(learnerId));
        if(o.out){writeFileSync(resolve(o.out),JSON.stringify(payload,null,2),{flag:'wx',mode:0o600});out({written:resolve(o.out)});}else out(payload);
        return 0;
      }
    }
    if(command==='journal'||command==='explain'){
      const explanations=createExplanationService({cacheRoot:join(dirname(dbPath),'explanations'),env,fetcher});
      const journal=withExplanations(createJournalService({db,catalogRoot}),explanations,db);
      const view=await journal.view(learnerId,now);
      out(command==='explain'?await explanations.decorate(learnerId,view,{generate:true,retryFailed:Boolean(o['retry-failed']),snapshotPlan:snapshotPlanReader(db,learnerId)}):view);return 0;
    }
    if (command==='plan') {
      const date=o.date ?? seoulDate(now);
      const frozen=await getSnapshot(db,learnerId,'plan',date);
      if (frozen) { out(o.json ? frozen : formatPlan(frozen,true)); return 0; }
      const ledger=await readLedger(db,learnerId,{cutoff:nowIso});
      const catalog=await activeCatalog(ledger);
      const preferences=await getPreferences(db,learnerId);
      const mastery=normalizeMastery(preferences.mastery ?? {});
      const priorPlans=await plansOf();
      let contestProposals=null;
      const {weekday}=weekInfo(date);
      if (preferences.weekly_rhythm[weekday]==='virtual') {
        const abilities=fitAll(ledger.attempts,catalog,ledger.rating_observations);
        const cache=(await cachedProviders(db,learnerId,now)).items;
        const report=buildWeeklyReport({ledger,catalog,plans:priorPlans,weekStart:weekInfo(date).weekStart,preferences});
        contestProposals={policy:'contests-v1',live:liveProposals({calendar:cache,abilities,mastery,from:date,to:new Date(Date.parse(date)+86400000).toISOString().slice(0,10)}),
          calendar_source:cache.length?'provider_cache':'none (no cached calendar; open the dashboard with providers enabled to fetch one)',
          virtual:virtualProposal({catalog,attempts:ledger.attempts,abilities,mastery,date,weakConcepts:report.weaknesses.map(w=>w.concept)}).candidates[0] ?? null};
      }
      const plan=planDay({date,now:nowIso,ledger,catalog,preferences,priorPlans:[...priorPlans,...mockExposurePlans(await mockContestsOf())],contestProposals});
      const saved=await saveSnapshot(db,learnerId,'plan',date,plan);
      out(o.json ? saved : formatPlan(saved,false)); return 0;
    }
    if (command==='mock-contest' && ['create','show'].includes(sub)) {
      const preferences=await getPreferences(db,learnerId);
      // Default: the next date (today included) whose weekly rhythm day is `virtual` (R5.1).
      let date=o.date;
      if (!date) for (let i=0;i<7 && !date;i++) { const d=new Date(Date.parse(seoulDate(now)+'T00:00:00Z')+i*86400000).toISOString().slice(0,10); if (preferences.weekly_rhythm[weekInfo(d).weekday]==='virtual') date=d; }
      if (!date) throw new Error('no virtual day in the weekly rhythm: pass --date');
      const stored=await getSnapshot(db,learnerId,'mock_contest',date);
      if (sub==='show') { if (!stored) { out(o.json ? {found:false,local_date:date} : `No mock contest for ${date}.`); return 1; } out(o.json ? stored : formatMockContest(stored)); return 0; }
      if (stored) { out(o.json ? {changed:false,mock_contest:stored} : `${formatMockContest(stored)}\n  (already stored; unchanged)`); return 0; }
      const ledger=await readLedger(db,learnerId,{cutoff:nowIso});
      const catalog=await activeCatalog(ledger);
      const mastery=normalizeMastery(preferences.mastery ?? {});
      const priorPlans=await plansOf();
      const report=buildWeeklyReport({ledger,catalog,plans:priorPlans,weekStart:weekInfo(date).weekStart,preferences});
      // Every snapshot kind counts: CLI `plan`, journal recommendations and earlier mock contests.
      const exposures=snapshotExposures((await db.prepare('SELECT kind,local_date,payload FROM recommendation_snapshots WHERE learner_id=? ORDER BY local_date,kind').bind(learnerId).all()).results.map(r=>({...r,payload:JSON.parse(r.payload)})));
      const record=mockContest({catalog,attempts:ledger.attempts,exposures,weakConcepts:report.weaknesses.map(w=>w.concept),
        learnerId,date,mastery,count:o.count===undefined?undefined:Number(o.count),minutes:o.minutes===undefined?null:Number(o.minutes)});
      const saved=await saveSnapshot(db,learnerId,'mock_contest',date,record);
      out(o.json ? {changed:true,mock_contest:saved} : formatMockContest(saved)); return 0;
    }
    if (command==='related') {
      const ledger=await readLedger(db,learnerId,{cutoff:nowIso});
      const catalog=await activeCatalog(ledger);
      const preferences=await getPreferences(db,learnerId);
      const states=problemStates(ledger.attempts,catalog);
      const seen=new Set([...states].filter(([,s])=>s.ever_solved).map(([k])=>k));
      out(relatedCandidates({sourceKey:o.problem,catalog,abilities:fitAll(ledger.attempts,catalog,ledger.rating_observations),seen,channel:o.channel ?? null,
        platforms:normalizeMastery(preferences.mastery ?? {}).contest_platforms,limit:Number(o.limit ?? 5)})); return 0;
    }
    if (command==='weekly') {
      const week=weekInfo(o.week ?? seoulDate(now)).weekStart;
      let report=await getWeeklyReport(db,learnerId,week);
      if (!report) {
        const cutoff=new Date(Date.parse(week+'T00:00:00+09:00')+7*86400000-1000);
        const ledger=await readLedger(db,learnerId,{cutoff:timestamp((cutoff<now?cutoff:now).toISOString())});
        const catalog=await activeCatalog(ledger);
        report=buildWeeklyReport({ledger,catalog,plans:await plansOf(),weekStart:week,preferences:await getPreferences(db,learnerId)});
        // Freeze only complete weeks; a running week is a preview.
        if (cutoff<=now) report=await saveWeeklyReport(db,learnerId,week,report);
      }
      const markdown=renderMarkdown(report);
      if (o.out) {
        const file=join(resolve(o.out),`${week}.md`);
        if (existsSync(file)) throw new Error(`refusing to overwrite ${file}`);
        mkdirSync(resolve(o.out),{recursive:true}); writeFileSync(file,markdown,{flag:'wx'});
        out(`wrote ${file} (not committed)`);
      } else out(markdown);
      return 0;
    }
    if (command==='find') {
      const ledger=await readLedger(db,learnerId,{cutoff:nowIso});
      const catalog=await activeCatalog(ledger);
      const date=seoulDate(now);
      const preferences=await getPreferences(db,learnerId);
      const cooldown=normalizeMastery(preferences.mastery ?? {}).cooldown_days;
      const exposed=new Set((await plansOf()).filter(p=>p.date>=new Date(Date.parse(date)-cooldown*86400000).toISOString().slice(0,10)).flatMap(p=>p.slots ?? []).map(s=>s.problem_key).filter(Boolean));
      createViews(db.sqlite,{catalog,states:problemStates(ledger.attempts,catalog),abilities:fitAll(ledger.attempts,catalog,ledger.rating_observations),exposed,date});
      out(findProblems(db.sqlite,{platform:o.platform,concept:o.concept,conceptStatus:o['concept-status'],tag:o.tag,difficulty:o.difficulty,predicted:o.predicted,status:o.status,
        near:o.near,channel:o.channel,maxRank:o['max-rank'],source:o.source,archiveCategory:o['archive-category'],provenance:o.provenance,date:o.date,limit:o.limit})); return 0;
    }
    if (command==='status') {
      const ledger=await readLedger(db,learnerId,{cutoff:nowIso});
      out({learner_id:learnerId,database:dbPath,high_water:ledger.high_water,attempts:ledger.attempts.length,
        active_catalog:ledger.catalog_switches.at(-1)?.payload.to_version ?? null,
        repositories:(await db.prepare('SELECT repo_id,branch,path_hint FROM git_repositories WHERE learner_id=?').bind(learnerId).all()).results}); return 0;
    }
    print(USAGE); return 1;
  } finally { db.close(); }
}

export function formatPlan(plan,frozen) {
  const lines=[`Plan ${plan.date} (${plan.day_type}) ${frozen?'[frozen]':'[saved]'} policy ${plan.policy_bundle}, catalog ${plan.catalog_version}`];
  for (const s of plan.slots) {
    if (!s.problem_key) { lines.push(`- ${s.kind}: ${s.reason}`);
      if (s.proposals?.virtual) lines.push(`  virtual: ${s.proposals.virtual.contest_id} expected solved ${s.proposals.virtual.expected_solved} coverage ${s.proposals.virtual.weakness_coverage}`);
      for (const l of (s.proposals?.live ?? []).filter(x=>x.proposed)) lines.push(`  live: ${l.title} ${l.start_at} (${l.kind})`);
      continue; }
    const p=s.prediction?.p_mean===null || s.prediction?.p_mean===undefined ? 'P unknown' : `P ${s.prediction.p_conservative}-${s.prediction.p_mean}${s.prediction.status==='insufficient_evidence'?' (sparse)':''}`;
    lines.push(`- ${s.kind}${s.sub_kind?'/'+s.sub_kind:''} ${s.problem_key} "${s.title}" ${p} [${(s.concepts ?? []).join(', ')}] ${s.estimated_minutes}m`,`  ${s.reason}${s.evidence_label?` (${s.evidence_label})`:''}`,`  ${s.url ?? ''}`);
  }
  if (plan.notes?.length) lines.push(...plan.notes.map(n=>`  note: ${n}`));
  lines.push(`  ${plan.rotation_note}`);
  return lines.join('\n');
}

export function formatMockContest(m) {
  const lines=[`Mock contest ${m.local_date}, solving time ${m.duration_minutes}m, policy ${m.policy}, catalog ${m.catalog_version}`,
    `  mix: Easy ${m.composition.easy} / Medium ${m.composition.medium} / Hard ${m.composition.hard}, expected solved ${m.expected_solved} (target ${m.target_expected})`];
  if (m.evidence_label) lines.push(`  ${m.evidence_label}`);
  for (const p of m.problems) lines.push(`${p.slot}. [${['','Easy','Medium','Hard'][p.level]}] ${p.problem_key} "${p.title}" P ${p.p_conservative}-${p.p_mean}`,`   ${p.reason}`,`   ${p.url ?? ''}`);
  if (m.notes?.length) lines.push(...m.notes.map(n=>`  note: ${n}`));
  lines.push('  Solve on leetcode.com; record results with the daily log, Git trailers or `attempt`.');
  return lines.join('\n');
}

if (process.argv[1] && import.meta.url===pathToFileURL(resolve(process.argv[1])).href) {
  run(process.argv.slice(2)).then(code=>{ process.exitCode=code; },error=>{ console.error(`error: ${error.message}`); process.exitCode=2; });
}
