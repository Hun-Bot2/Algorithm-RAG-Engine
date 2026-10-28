// Local polling wrapper for registered Git repositories. It never registers repos
// or inspects working-tree files; git-ingest reads committed history only.
import {ingestRepository} from './git-ingest.mjs';
import {readLedger} from './ledger.mjs';

export function createGitSync({db,learnerId,intervalMs=5000,catalogRoot,catalogProvider=null}) {
  if (!db || typeof learnerId!=='string' || !learnerId) throw new TypeError('db and learnerId are required');
  if (!Number.isFinite(intervalMs) || intervalMs<1) throw new RangeError('intervalMs must be a positive number');
  let timer=null, inFlight=null, closed=false;
  const state={running:false,started:false,closed:false,lastStartedAt:null,lastFinishedAt:null,lastResults:[],lastWarnings:[],lastError:null};

  const warningsByRepo=new Map();
  async function runTick() {
    state.running=true;
    state.lastStartedAt=new Date().toISOString();
    state.lastError=null;
    const results=[];
    try {
      const repos=(await db.prepare('SELECT repo_id FROM git_repositories WHERE learner_id=? ORDER BY repo_id').bind(learnerId).all()).results;
      for (const {repo_id:repoId} of repos) {
        try { results.push(await ingestRepository(db,learnerId,repoId,{catalogRoot,catalog:catalogProvider ? await catalogProvider(await readLedger(db,learnerId)) : null})); }
        catch (error) { results.push({repo_id:repoId,status:'error',error:String(error?.message ?? error)}); }
      }
      state.lastResults=results;
      for(const result of results)if(result.status==='ok' && result.commits>0){
        const warnings=Object.values(result.flags ?? {}).flat();
        if(result.unclassified?.length)warnings.push('unclassified_commits');
        warningsByRepo.set(result.repo_id,warnings);
      }
      state.lastWarnings=[...warningsByRepo].filter(([,flags])=>flags.length).map(([repo_id,flags])=>({repo_id,flags:[...new Set(flags)]}));
      const failures=results.filter(result=>result.status==='error');
      state.lastError=failures.length ? failures.map(result=>`${result.repo_id}: ${result.error}`).join('; ') : null;
    } catch (error) {
      state.lastResults=results;
      state.lastError=String(error?.message ?? error);
    } finally {
      state.running=false;
      state.lastFinishedAt=new Date().toISOString();
    }
    return status();
  }

  function tick() {
    if (closed) return Promise.resolve(status());
    if (inFlight) return inFlight;
    inFlight=runTick().finally(()=>{inFlight=null;});
    return inFlight;
  }

  function start() {
    if (closed) throw new Error('Git sync is closed');
    if (!timer) {
      state.started=true;
      timer=setInterval(()=>{ void tick(); },intervalMs);
      timer.unref?.();
    }
    return status();
  }

  function status() { return {...state,lastWarnings:state.lastWarnings.map(w=>({...w,flags:[...w.flags]})),lastResults:state.lastResults.map(result=>({...result}))}; }

  async function close() {
    if (timer) clearInterval(timer);
    timer=null;
    state.started=false;
    closed=true;
    state.closed=true;
    if (inFlight) await inFlight;
    return status();
  }

  return {start,tick,close,status};
}
