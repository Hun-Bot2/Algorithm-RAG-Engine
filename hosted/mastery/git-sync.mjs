// Local polling wrapper for registered Git repositories. It never registers repos
// or inspects working-tree files; git-ingest reads committed history only.
import {ingestRepository} from './git-ingest.mjs';

export function createGitSync({db,learnerId,intervalMs=5000}) {
  if (!db || typeof learnerId!=='string' || !learnerId) throw new TypeError('db and learnerId are required');
  if (!Number.isFinite(intervalMs) || intervalMs<1) throw new RangeError('intervalMs must be a positive number');
  let timer=null, inFlight=null, closed=false;
  const state={running:false,started:false,closed:false,lastStartedAt:null,lastFinishedAt:null,lastResults:[],lastError:null};

  async function runTick() {
    state.running=true;
    state.lastStartedAt=new Date().toISOString();
    state.lastError=null;
    const results=[];
    try {
      const repos=(await db.prepare('SELECT repo_id FROM git_repositories WHERE learner_id=? ORDER BY repo_id').bind(learnerId).all()).results;
      for (const {repo_id:repoId} of repos) {
        try { results.push(await ingestRepository(db,learnerId,repoId)); }
        catch (error) { results.push({repo_id:repoId,status:'error',error:String(error?.message ?? error)}); }
      }
      state.lastResults=results;
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

  function status() { return {...state,lastResults:state.lastResults.map(result=>({...result}))}; }

  async function close() {
    if (timer) clearInterval(timer);
    timer=null;
    closed=true;
    state.closed=true;
    if (inFlight) await inFlight;
    return status();
  }

  return {start,tick,close,status};
}
