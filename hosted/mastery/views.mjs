// Read-only SQL views for filtering (R13, MS-303). Node-only: TEMP objects over the attached
// catalog plus a per-run temp table of derived learner status. Nothing is written to the
// catalog or to the learner ledgers.
import {predict} from './calibration.mjs';
import {canonicalKey} from './related.mjs';
import {seoulDate} from '../core/policy.mjs';

export const STATUSES = ['unseen','exposed','attempted','solved_reported','solved_independent','review_due'];

export function createViews(sqlite,{catalog,states,abilities,exposed=new Set(),date}) {
  sqlite.exec(`DROP TABLE IF EXISTS temp.learner_problem;
    CREATE TEMP TABLE learner_problem (problem_key TEXT PRIMARY KEY, status TEXT NOT NULL, p_mean REAL, p_conservative REAL, last_attempt_date TEXT);
    DROP VIEW IF EXISTS temp.v_problem_facets; DROP VIEW IF EXISTS temp.v_problem_status; DROP VIEW IF EXISTS temp.v_neighbors; DROP VIEW IF EXISTS temp.v_archive;
    CREATE TEMP VIEW v_problem_facets AS
      SELECT p.problem_key, p.platform, p.title, p.url, p.contest_id, p.contest_start, p.difficulty_kind, p.difficulty_value, p.access, p.eligible, p.status AS catalog_status,
             p.first_seen_version, p.provenance, COALESCE(r.tier,'unknown') AS reliability FROM catalog.problems p LEFT JOIN catalog.reliability r USING (problem_key);
    CREATE TEMP VIEW v_problem_status AS
      SELECT f.*, COALESCE(l.status,'unseen') AS learner_status, l.p_mean, l.p_conservative, l.last_attempt_date FROM v_problem_facets f LEFT JOIN learner_problem l USING (problem_key);
    CREATE TEMP VIEW v_neighbors AS SELECT src, dst, channel, rank, score FROM catalog.neighbors;
    CREATE TEMP VIEW v_archive AS SELECT r.record_id, r.version, r.archive_version, r.category, r.organization, r.exam, r.year, r.rights_status, r.confidence, r.recorded_by,
             r.source_kind, r.acquisition, r.superseded, l.problem_key, l.relation, l.status AS link_status FROM catalog.archive_records r LEFT JOIN catalog.archive_links l USING (record_id, version);`);
  const insert=sqlite.prepare('INSERT INTO learner_problem VALUES (?,?,?,?,?)');
  sqlite.exec('BEGIN');
  try {
    for (const p of catalog.problems.values()) {
      const s=states.get(p.key);
      let status=exposed.has(p.key) ? 'exposed' : 'unseen';
      if (s) status = s.due_date && s.due_date<=date ? 'review_due' : s.state==='demonstrated' || s.state==='transfer-evidenced' ? 'solved_independent' : s.ever_solved ? 'solved_reported' : 'attempted';
      const prediction=predict(p,abilities);
      if (status!=='unseen' || prediction.p_mean!==null) insert.run(p.key,status,prediction.p_mean,prediction.p_conservative,s ? seoulDate(new Date(s.last_at)) : null);
    }
    sqlite.exec('COMMIT');
  } catch (error) { sqlite.exec('ROLLBACK'); throw error; }
}

// Builds a parameterized query over the views. Every date filter names its basis (R13.2).
export function findQuery(filters={}) {
  const where=[], args=[];
  const list=value=>String(value).split(',').map(s=>s.trim()).filter(Boolean);
  if (filters.platform) { const v=list(filters.platform); where.push(`s.platform IN (${v.map(()=>'?').join(',')})`); args.push(...v); }
  if (filters.status) { const v=list(filters.status); if (v.some(x=>!STATUSES.includes(x))) throw new Error('invalid status filter'); where.push(`s.learner_status IN (${v.map(()=>'?').join(',')})`); args.push(...v); }
  if (filters.concept) {
    const statuses=filters.conceptStatus ? list(filters.conceptStatus) : ['approved','ai_reviewed'];
    where.push(`EXISTS (SELECT 1 FROM catalog.problem_concepts c WHERE c.problem_key=s.problem_key AND c.concept_id IN (${list(filters.concept).map(()=>'?').join(',')}) AND c.evidence_status IN (${statuses.map(()=>'?').join(',')}))`);
    args.push(...list(filters.concept),...statuses);
  }
  if (filters.tag) { where.push(`EXISTS (SELECT 1 FROM catalog.problem_tags t WHERE t.problem_key=s.problem_key AND t.native_tag IN (${list(filters.tag).map(()=>'?').join(',')}))`); args.push(...list(filters.tag)); }
  const range=(column,value)=>{ const m=/^(-?\d+(?:\.\d+)?)?\.\.(-?\d+(?:\.\d+)?)?$/.exec(String(value)); if (!m) throw new Error(`invalid range for ${column}`);
    if (m[1]!==undefined) { where.push(`${column} >= ?`); args.push(Number(m[1])); } if (m[2]!==undefined) { where.push(`${column} <= ?`); args.push(Number(m[2])); } };
  if (filters.difficulty) range('s.difficulty_value',filters.difficulty);
  if (filters.predicted) range('s.p_mean',filters.predicted);
  if (filters.eligibleOnly!==false) where.push("s.eligible = 1 AND s.catalog_status = 'active'");
  if (filters.source) { const v=list(filters.source); where.push(`(${v.map(()=>"s.provenance LIKE ?").join(' OR ')})`); args.push(...v.map(x=>x==='catalog'?'catalog:%':x==='user_entered'?'user_entered%':`${x}%`)); }
  if (filters.archiveCategory || filters.provenance) {
    const parts=['a.problem_key = s.problem_key','a.superseded = 0'];
    if (filters.archiveCategory) { parts.push(`a.category IN (${list(filters.archiveCategory).map(()=>'?').join(',')})`); args.push(...list(filters.archiveCategory)); }
    if (filters.provenance) { parts.push(`a.source_kind IN (${list(filters.provenance).map(()=>'?').join(',')})`); args.push(...list(filters.provenance)); }
    where.push(`EXISTS (SELECT 1 FROM v_archive a WHERE ${parts.join(' AND ')})`);
  }
  if (filters.near) {
    const channel=filters.channel ?? 'title_v2', maxRank=Number(filters.maxRank ?? 30);
    where.push('EXISTS (SELECT 1 FROM v_neighbors n WHERE n.src = ? AND n.dst = s.problem_key AND n.channel = ? AND n.rank <= ?)'); args.push(filters.near,channel,maxRank);
  }
  if (filters.date) {
    const m=/^(attempt|contest|first_seen):(.*)$/.exec(filters.date);
    if (!m) throw new Error('date filter must name its basis: attempt:, contest: or first_seen:');
    if (m[1]==='first_seen') { where.push('s.first_seen_version = ?'); args.push(m[2]); }
    else {
      const [from,to]=m[2].split('..');
      const column=m[1]==='attempt' ? 's.last_attempt_date' : 'substr(s.contest_start,1,10)';
      if (from) { where.push(`${column} >= ?`); args.push(from); } if (to) { where.push(`${column} <= ?`); args.push(to); }
    }
  }
  const limit=Math.min(Number(filters.limit ?? 50),1000);
  return {sql:`SELECT s.problem_key, s.platform, s.title, s.difficulty_kind, s.difficulty_value, s.learner_status, s.p_mean, s.p_conservative, s.last_attempt_date, s.url
    FROM v_problem_status s ${where.length?'WHERE '+where.join(' AND '):''} ORDER BY s.platform, s.difficulty_value IS NULL, s.difficulty_value, s.problem_key LIMIT ${limit}`,args};
}
export const findProblems=(sqlite,filters)=>{ const {sql,args}=findQuery(filters); return sqlite.prepare(sql).all(...args).map(r=>({...r})); };
