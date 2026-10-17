// Opt-in Codeforces rating history (MS-605). One documented API call per sync; the fetcher is
// injected so tests and offline runs never touch the network. Output is rating_observations.
const ENDPOINT='https://codeforces.com/api/user.rating?handle=';
const iso=seconds=>new Date(seconds*1000).toISOString().replace(/\.\d{3}Z$/,'.000000Z');
export function ratingEntries(learnerId,handle,changes,recordedAt) {
  if (!/^[A-Za-z0-9_.-]{1,40}$/.test(handle)) throw new Error('invalid Codeforces handle');
  return changes.map((c,i)=>({schema_version:1,entry_id:`cf-rating-${handle.toLowerCase()}-${c.contestId}`,learner_id:learnerId,
    occurred_at:iso(c.ratingUpdateTimeSeconds),recorded_at:recordedAt,source:'codeforces_api',supersedes:null,
    payload:{platform:'codeforces',handle,rating:c.newRating,rated_participations:i+1,contest_id:String(c.contestId),old_rating:c.oldRating,rank:c.rank}}));
}
export async function fetchCodeforcesRatings(handle,{fetcher=fetch,timeoutMs=10000}={}) {
  if (!/^[A-Za-z0-9_.-]{1,40}$/.test(handle)) throw new Error('invalid Codeforces handle');
  const response=await fetcher(ENDPOINT+encodeURIComponent(handle),{headers:{accept:'application/json','user-agent':'AlgorithmRAGEngine-mastery/1.0'},signal:AbortSignal.timeout(timeoutMs)});
  if (response.status===403 || response.status===429) throw new Error(`Codeforces API refused (${response.status}); stopping`);
  const body=await response.json();
  if (body.status!=='OK' || !Array.isArray(body.result)) throw new Error('Codeforces API returned no rating history');
  return body.result.filter(c=>Number.isInteger(c.contestId) && Number.isFinite(c.newRating) && Number.isFinite(c.oldRating) && Number.isInteger(c.ratingUpdateTimeSeconds));
}
