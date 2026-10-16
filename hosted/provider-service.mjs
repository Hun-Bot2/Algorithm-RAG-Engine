import {fetchContests, fetchSubmissions} from './providers.mjs';
import {appendEvents} from './core/store.mjs';
const clockText = date => date.toISOString();

export async function cachedProviders(db,learnerId,now=new Date()) {
  const rows=(await db.prepare('SELECT provider_key,payload FROM provider_cache WHERE learner_id=?').bind(learnerId).all()).results;
  const calendars=new Map(rows.filter(r=>r.provider_key.endsWith(':calendar')).map(r=>[r.provider_key.split(':')[0],JSON.parse(r.payload)]));
  const sources=['codeforces','atcoder'].map(platform=>{
    const envelope=calendars.get(platform);
    const age=now.getTime()-Date.parse(envelope?.fetched_at);
    const status=envelope?.status==='fresh' && Number.isFinite(age) && age>=0 && age<6*3600000 ? 'fresh' : envelope?.fetched_at ? 'stale' : 'unavailable';
    return {source:platform,status,fetched_at:envelope?.fetched_at ?? null,error:envelope?.error ?? (envelope?'stale':'not_synced'),
      items:(envelope?.items ?? []).map(item=>({...item,stale:status!=='fresh'}))};
  });
  return {sources,items:sources.flatMap(s=>s.items)};
}

export async function syncProvider({db,learnerId,platform,handle,env={},now=new Date(),fetcher}) {
  if (env.PROVIDERS_ENABLED!=='true') throw new Error('providers disabled');
  if (!['codeforces','atcoder'].includes(platform)) throw new Error('invalid provider');
  const key=platform+(handle?':submissions':':calendar');
  const token=crypto.randomUUID(), at=clockText(now), until=clockText(new Date(now.getTime()+60000));
  await db.prepare('INSERT OR IGNORE INTO provider_cache(learner_id,provider_key) VALUES (?,?)').bind(learnerId,key).run();
  const lock=await db.prepare('UPDATE provider_cache SET lease_until=?,lease_token=? WHERE learner_id=? AND provider_key=? AND (lease_until IS NULL OR lease_until<=?) RETURNING payload').bind(until,token,learnerId,key,at).first();
  if (!lock) throw new Error('provider sync in progress');
  try {
    const previous=JSON.parse(lock.payload);
    // A handle change must not reuse another account's watermark.
    const prior=handle && previous.handle!==handle ? {} : previous;
    const snapshot=handle ? await fetchSubmissions(platform,{handle,learnerId,previous:prior,now,fetcher,enabled:true})
      : await fetchContests(platform,{previous,now,fetcher,enabled:true});
    const added=handle && snapshot.events?.length ? await appendEvents(db,learnerId,snapshot.events) : 0;
    const saved=handle?{...snapshot,events:[],handle}:snapshot;
    await db.prepare('UPDATE provider_cache SET payload=?,lease_until=NULL,lease_token=NULL WHERE learner_id=? AND provider_key=? AND lease_token=?').bind(JSON.stringify(saved),learnerId,key,token).run();
    return {status:snapshot.status,imported:added,truncated:snapshot.truncated ?? false,historical_gap:snapshot.historical_gap ?? false,error:snapshot.error};
  } finally {
    await db.prepare('UPDATE provider_cache SET lease_until=NULL,lease_token=NULL WHERE learner_id=? AND provider_key=? AND lease_token=?').bind(learnerId,key,token).run();
  }
}
