import {appendParticipation,listParticipation} from '../contest-history.mjs';
import {exportLearningBackup,importLearningBackup} from '../backup.mjs';
import {operationsSnapshot} from '../operations.mjs';
import {cachedProviders,syncProvider} from '../provider-service.mjs';
import {authenticateAccess} from './auth.mjs';
import {confirmedSeconds, identifier, timestamp, timerState} from './events.mjs';
import {appendEvents, exportDocument, getMission, getPreferences, importDocument, learnerSnapshot, listManualContests, putManualContest, putPreferences} from './store.mjs';
import {buildMission, deriveProgress, seoulDate} from './policy.mjs';
import {listDeliveries} from '../notifications.mjs';

const json = (body,status=200) => new Response(JSON.stringify(body),{status,headers:{'content-type':'application/json; charset=utf-8','cache-control':'no-store'}});
const readJson = async request => {
  const length = Number(request.headers.get('content-length'));
  if (length > 2_000_000) throw new Error('request too large');
  if (!request.body) throw new Error('missing request body');
  const reader=request.body.getReader(), chunks=[];
  let size=0;
  while (true) {
    const {done,value}=await reader.read();
    if (done) break;
    size+=value.byteLength;
    if (size>2_000_000) { await reader.cancel(); throw new Error('request too large'); }
    chunks.push(value);
  }
  const bytes=new Uint8Array(size);
  let offset=0;
  for (const chunk of chunks) { bytes.set(chunk,offset); offset+=chunk.byteLength; }
  return JSON.parse(new TextDecoder().decode(bytes));
};
function manualContest(input,now) {
  if (!input || typeof input !== 'object' || Array.isArray(input) || Object.keys(input).sort().join('|') !== 'duration_minutes|id|platform|start_at|suitable|title|url') throw new Error('invalid manual contest');
  identifier(input.id);
  if (typeof input.title !== 'string' || !input.title.trim() || !['atcoder','codeforces','other'].includes(input.platform) || typeof input.suitable !== 'boolean' || !Number.isInteger(input.duration_minutes) || input.duration_minutes < 1) throw new Error('invalid manual contest');
  const url = new URL(input.url);
  if (url.protocol !== 'https:' || url.username || url.password) throw new Error('HTTPS contest URL required');
  return {...input,start_at:timestamp(input.start_at),source:'manual',fetched_at:timestamp(now.toISOString())};
}
export function createApp({env={},db=env.DB,curriculum,annotations={},authenticate=request=>authenticateAccess(request,env),assets=env.ASSETS,contests=({learnerId,now})=>cachedProviders(db,learnerId,now),delivery=({learnerId})=>listDeliveries(db,learnerId),now=()=>new Date()}={}) {
  if (!db || !curriculum) throw new Error('hosted core bindings missing');
  return {async fetch(request) {
    let identity;
    try { identity = await authenticate(request); identifier(identity?.learner_id); }
    catch { return json({error:'unauthorized'},401); }
    const learnerId = identity.learner_id;
    const url = new URL(request.url), path=url.pathname, method=request.method;
    try {
      if (path.startsWith('/api/') && !['GET','HEAD'].includes(method)) {
        if (request.headers.get('origin') !== url.origin || request.headers.get('content-type')?.split(';')[0].trim().toLowerCase() !== 'application/json') return json({error:'same-origin JSON request required'},403);
      }
      if (path === '/api/state' && method === 'GET') {
        const snap = await learnerSnapshot(db,learnerId);
        const clock = now(), cutoff = timestamp(clock.toISOString());
        const intervals = timerState(snap.events,cutoff);
        const manual = await listManualContests(db,learnerId);
        const upstream = await contests({learnerId,now:clock});
        const allContests=[...manual,...(Array.isArray(upstream)?upstream:upstream?.items ?? [])];
        const frozen=await getMission(db,learnerId,seoulDate(clock));
        const mission=frozen ?? buildMission({curriculum,annotations,events:snap.events,preferences:snap.preferences,now:clock,contests:allContests});
        const contest_sources=upstream?.sources ?? (Array.isArray(upstream) ? [] : [{source:upstream?.source ?? 'provider',status:upstream?.status ?? 'unavailable',fetched_at:upstream?.fetched_at ?? null,error:upstream?.error ?? null}]);
        return json({learner_id:learnerId,capabilities:{providers_enabled:env.PROVIDERS_ENABLED==='true',delivery_enabled:env.DELIVERY_ENABLED==='true'},preferences:snap.preferences,curriculum,annotations,events:snap.events,mission:{...mission,snapshot_status:frozen?'frozen':'preview'},snapshot_status:frozen?'frozen':'preview',progress:deriveProgress(curriculum,snap.events,cutoff),timers:{intervals,confirmed_seconds:confirmedSeconds(intervals)},contests:allContests,contest_sources,delivery:await delivery({learnerId,now:clock}),participation:await listParticipation(db,learnerId),delivery_reconciliation_required:Boolean((await db.prepare('SELECT delivery_reconciliation_required FROM learners WHERE learner_id=?').bind(learnerId).first())?.delivery_reconciliation_required)});
      }
      if (path === '/api/participation' && method === 'POST') return json({added:await appendParticipation(db,learnerId,await readJson(request))},201);
      if (path === '/api/backup/export' && method === 'GET') return json(await exportLearningBackup(db,learnerId));
      if (path === '/api/backup/import' && method === 'POST') return json(await importLearningBackup(db,learnerId,await readJson(request)));
      if (path === '/api/operations' && method === 'GET') return json(await operationsSnapshot(db,learnerId));
      if (path === '/api/providers/sync' && method === 'POST') {
        const input=await readJson(request);
        if (!input || Object.keys(input).some(key=>!['platform','handle'].includes(key))) return json({error:'invalid provider request'},400);
        if (env.PROVIDERS_ENABLED!=='true') return json({error:'providers disabled'},409);
        return json(await syncProvider({db,learnerId,...input,env,now:now()}));
      }
      if (path === '/api/preferences' && method === 'PUT') return json(await putPreferences(db,learnerId,await readJson(request)));
      if (path === '/api/events' && method === 'POST') return json({added:await appendEvents(db,learnerId,[await readJson(request)])},201);
      if (path === '/api/export' && method === 'GET') return json(await exportDocument(db,learnerId));
      if (path === '/api/import' && method === 'POST') return json({added:await importDocument(db,learnerId,await readJson(request))});
      if (path === '/api/contests/manual' && method === 'POST') return json(await putManualContest(db,learnerId,manualContest(await readJson(request),now())),201);
      if (path.startsWith('/api/')) return json({error:'not found'},404);
      if (method !== 'GET' && method !== 'HEAD') return json({error:'method not allowed'},405);
      if (!assets?.fetch) return json({error:'assets unavailable'},404);
      const response = await assets.fetch(request);
      const result = new Response(response.body,response);
      result.headers.set('cache-control','private, no-store');
      return result;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'invalid request';
      if (/SQLITE|D1_|constraint failed|database is locked|UNIQUE constraint|CHECK constraint/i.test(message)) return json({error:'write conflict'},409);
      const known=/^(invalid |unsupported |unexpected |missing |participation |conflicting participation ID|restore target is not empty|atomic backup restore failed|provider sync in progress|request too large|learner ownership mismatch|event occurs after recording|event cannot supersede itself|imports are reported completion only|import platform mismatch|independent outcome requires explicit no hints|timer actions require manual evidence|confirmation requires one stopped interval|confirmation outside recorded actions|duplicate timer start|duplicate or misplaced timer stop|timer action without start|correction target missing|correction target already superseded|conflicting event ID|platform-qualified problem key required|UTC timestamp required|HTTPS contest URL required)/;
      if (known.test(message)) return json({error:message},/conflict|superseded/i.test(message)?409:400);
      if (error instanceof SyntaxError) return json({error:'invalid JSON'},400);
      return json({error:'internal error'},500);
    }
  }};
}
