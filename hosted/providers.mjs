/** Optional public-data adapters. Core missions also work with all adapters disabled. */
const ENDPOINTS = {
  codeforces: 'https://codeforces.com/api/contest.list',
  atcoder: 'https://kenkoooo.com/atcoder/resources/contests.json',
};
const MAX_ITEMS = 5000;
const epoch = value => new Date(value).getTime();
const iso = value => new Date(value).toISOString();
const validDate = value => typeof value === 'string' && Number.isFinite(epoch(value));

export function createRequestGate({ fetcher = fetch, sleep = ms => new Promise(r => setTimeout(r, ms)),
  now = Date.now, gapMs = 2000 } = {}) {
  let tail = Promise.resolve(), last = -Infinity;
  return (url, options = {}) => {
    const run = tail.then(async () => {
      const wait = Math.max(0, last + gapMs - now());
      if (wait) await sleep(wait);
      last = now();
      return fetcher(url, { ...options, signal: AbortSignal.timeout(10000) });
    });
    // Include completion in the serialized chain, even after a rejected request.
    tail = run.catch(() => {});
    return run;
  };
}

const defaultFetch = createRequestGate();

async function json(fetcher, url) {
  const response = await fetcher(url, { signal: AbortSignal.timeout(10000) });
  if (!response.ok) throw new Error(`http_${response.status}`);
  const text = await response.text();
  if (text.length > 4_000_000) throw new Error('response_too_large');
  return JSON.parse(text);
}

function fallback(source, previous, error) {
  return { source, fetched_at: previous?.fetched_at ?? null,
    status: previous?.fetched_at ? 'stale' : 'unavailable',
    items: Array.isArray(previous?.items) ? previous.items : [], error };
}

export async function fetchContests(platform, { fetcher = defaultFetch, now = new Date(),
  previous = null, enabled = false } = {}) {
  const source = ENDPOINTS[platform];
  if (!source) throw new Error('unsupported_platform');
  if (!enabled) return fallback(source, previous, 'disabled');
  if (validDate(previous?.fetched_at) && epoch(now) - epoch(previous.fetched_at) >= 0 &&
      epoch(now) - epoch(previous.fetched_at) < 6 * 3600000) return { ...previous, status: 'fresh', error: null };
  try {
    const raw = await json(fetcher, source);
    const records = platform === 'codeforces' ? (raw.status === 'OK' ? raw.result : null) : raw;
    if (!Array.isArray(records) || records.length > MAX_ITEMS) throw new Error('invalid_calendar');
    const items = records.filter(r => platform !== 'codeforces' || r.phase === 'BEFORE').map(r => {
      const id = platform === 'codeforces' ? String(r.id) : r.id;
      const start = platform === 'codeforces' ? r.startTimeSeconds : r.start_epoch_second;
      const duration = platform === 'codeforces' ? r.durationSeconds : r.duration_second;
      const title = platform === 'codeforces' ? r.name : r.title;
      if (!/^[A-Za-z0-9_-]+$/.test(id) || !Number.isFinite(start) || !Number.isFinite(duration) ||
          duration <= 0 || typeof title !== 'string' || !title.trim()) throw new Error('invalid_contest');
      return { id: `${platform}:${id}`, platform, title: title.slice(0, 200),
        url: platform === 'codeforces' ? `https://codeforces.com/contest/${id}` : `https://atcoder.jp/contests/${id}`,
        start_at: iso(start * 1000), duration_minutes: duration / 60, suitable: false,
        source, fetched_at: iso(now) };
    }).filter(r => epoch(r.start_at) >= epoch(now)).sort((a,b) => a.start_at.localeCompare(b.start_at)).slice(0,100);
    return { source, fetched_at: iso(now), status: 'fresh', items, error: null };
  } catch {
    return fallback(source, previous, 'provider_failed');
  }
}

export function submissionEvent(platform, item, learnerId, recordedAt) {
  let accepted, id, key, seconds;
  if (platform === 'codeforces') {
    accepted = item.verdict === 'OK'; id = item.id; seconds = item.creationTimeSeconds;
    if (!item.problem || !Number.isInteger(item.problem.contestId) ||
        !/^[A-Za-z0-9]+$/.test(item.problem.index)) throw new Error('invalid_submission');
    key = `codeforces:${item.problem.contestId}${item.problem.index}`;
  } else if (platform === 'atcoder') {
    accepted = item.result === 'AC'; id = item.id; seconds = item.epoch_second;
    if (!/^[A-Za-z0-9_]+$/.test(item.problem_id)) throw new Error('invalid_submission');
    key = `atcoder:${item.problem_id}`;
  } else throw new Error('unsupported_platform');
  if (!Number.isInteger(id) || !Number.isFinite(seconds) || seconds < 0) throw new Error('invalid_submission');
  if (!accepted) return null;
  return { schema_version: 1, event_id: `import:${platform}:${learnerId}:${id}`, learner_id: learnerId,
    occurred_at: iso(seconds * 1000), recorded_at: iso(recordedAt), kind: 'attempt', source: 'platform_import',
    payload: { problem_key: key, outcome: 'reported_completion', hints: null, platform, external_id: String(id) }, supersedes: null };
}

export async function fetchSubmissions(platform, { handle, learnerId, previous = {}, now = new Date(),
  fetcher = defaultFetch, enabled = false, pageLimit = 4 } = {}) {
  if (!enabled) return { ...previous, events: [], status: 'unavailable', error: 'disabled' };
  if (typeof handle !== 'string' || !/^[A-Za-z0-9_.-]{1,64}$/.test(handle)) throw new Error('invalid_handle');
  if (!['codeforces','atcoder'].includes(platform)) throw new Error('unsupported_platform');
  const seen = new Set(previous.seen_ids ?? []), events = [], nextSeen = new Set(seen);
  let cursor = Number(previous.cursor ?? 0), truncated = true;
  if (!Number.isInteger(pageLimit) || pageLimit < 1 || pageLimit > 20) throw new Error('invalid_page_limit');
  try {
    for (let page = 0; page < pageLimit; page++) {
      const url = platform === 'codeforces'
        ? `https://codeforces.com/api/user.status?handle=${encodeURIComponent(handle)}&from=${page * 100 + 1}&count=100`
        : `https://kenkoooo.com/atcoder/atcoder-api/v3/user/submissions?user=${encodeURIComponent(handle)}&from_second=${cursor}`;
      const raw = await json(fetcher, url);
      const list = platform === 'codeforces' ? (raw.status === 'OK' ? raw.result : null) : raw;
      if (!Array.isArray(list) || list.length > 500) throw new Error('invalid_submissions');
      let newItems = 0, reached = false;
      for (const item of list) {
        if (seen.has(String(item.id))) reached = true;
        const event = submissionEvent(platform, item, learnerId, now);
        if (!nextSeen.has(String(item.id))) {
          newItems++;
          nextSeen.add(String(item.id));
          if (event) events.push(event);
        }
      }
      if (platform === 'codeforces') {
        if (list.length < 100 || reached) { truncated = false; break; }
      } else {
        const last = Math.max(cursor, ...list.map(r => r.epoch_second));
        // Re-read the boundary second and deduplicate, never skip equal timestamps.
        if (list.length < 500) { cursor = last; truncated = false; break; }
        if (last === cursor && newItems === 0) break; // saturated boundary: expose incomplete coverage
        cursor = last;
      }
    }
    return { status: 'fresh', fetched_at: iso(now), events, seen_ids: [...nextSeen].slice(-10000),
      cursor, truncated, historical_gap: Boolean(previous.historical_gap || truncated), error: null, source: platform };
  } catch {
    // Never advance the watermark after a failed partial synchronization.
    return { ...previous, events: [], status: previous.fetched_at ? 'stale' : 'unavailable',
      truncated: true, error: 'provider_failed' };
  }
}
