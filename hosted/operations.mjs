/** Privacy-preserving operational counts. Wall time is explicitly not CPU billing. */
const ROUTES = new Set(['/api/state','/api/preferences','/api/events','/api/export','/api/import',
  '/api/contests/manual','/api/providers/sync','/api/participation','/api/backup/export','/api/backup/import','/api/operations']);
export async function recordObservation(db,{path,status,elapsedMs,now=new Date()}) {
  const route=ROUTES.has(path)?path:path.startsWith('/api/')?'unknown_api':'static';
  const date=now.toISOString().slice(0,10);
  const elapsed=Math.max(0,Number.isFinite(elapsedMs)?elapsedMs:0);
  await db.prepare(`INSERT INTO runtime_observations(utc_date,route,status,requests,elapsed_sum_ms,elapsed_max_ms)
    VALUES (?,?,?,1,?,?) ON CONFLICT(utc_date,route,status) DO UPDATE SET
    requests=requests+1,elapsed_sum_ms=elapsed_sum_ms+excluded.elapsed_sum_ms,
    elapsed_max_ms=MAX(elapsed_max_ms,excluded.elapsed_max_ms)`).bind(date,route,status,elapsed,elapsed).run();
}
export async function operationsSnapshot(db,learnerId) {
  const counts=await db.prepare(`SELECT COUNT(*) AS event_count,COALESCE(SUM(length(canonical)),0) AS event_json_characters
    FROM events WHERE learner_id=?`).bind(learnerId).first();
  const observations=(await db.prepare('SELECT * FROM runtime_observations ORDER BY utc_date DESC,route,status LIMIT 200').all()).results;
  return {counts,observations,limitations:{wall_time_is_cpu:false,d1_billing_usage:null,cloudflare_free_tier_passed:null,
    phone_alerts_measured:false,retention:'Bounded query; retain or export operational counters per private pilot policy.'}};
}
