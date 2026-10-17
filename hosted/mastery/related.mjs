// related-v1 (design.md section 6, R5). Pure. Proposes Codeforces/AtCoder practice for a
// source problem. Every candidate needs concept overlap or a reviewed edge (R5.2) and carries
// its score components and evidence label (R5.3). Never claims algorithmic equivalence.
import {predict} from './calibration.mjs';

export const RELATED_VERSION = 'related-v1';
export const SEMANTIC_ONLY_LABEL = 'Semantic title neighbor: concept not classified (title similarity only, not an algorithm match)';
export const SEMANTIC_ONLY_PENALTY = -1.5;
export const CHANNELS = ['concept','title_v2','title_v2_cross'];
const TRUSTED = new Set(['approved','ai_reviewed']);
export const trustedConcepts = problem => new Set((problem?.concepts ?? []).filter(c=>TRUSTED.has(c.status)).map(c=>c.id));
export const canonicalKey = (catalog,key) => catalog.aliases.get(key) ?? key;
export const recommendable = p => p && p.eligible && p.status==='active' && ['free','free_at_collection'].includes(p.access);

export function relatedCandidates({sourceKey,catalog,abilities,seen=new Set(),channel=null,platforms=['codeforces','atcoder'],reviewedEdges=new Set(),limit=3}) {
  const source=catalog.problems.get(canonicalKey(catalog,sourceKey));
  if (!source) return {source_key:sourceKey,channel,candidates:[],status:'source_not_in_catalog'};
  const sourceConcepts=trustedConcepts(source);
  const ranks=new Map();
  const channels=channel ? [channel] : CHANNELS;
  for (const ch of channels.filter(c=>c!=='concept')) for (const n of catalog.neighbors(source.key,ch)) if (!ranks.has(n.dst) || ranks.get(n.dst).rank>n.rank) ranks.set(n.dst,{rank:n.rank,channel:ch});
  const pool=new Set();
  if (channels.includes('concept') && sourceConcepts.size) for (const p of catalog.problems.values()) if (platforms.includes(p.platform) && [...trustedConcepts(p)].some(c=>sourceConcepts.has(c))) pool.add(p.key);
  for (const key of ranks.keys()) pool.add(key);
  for (const edge of reviewedEdges) { const [a,b]=edge.split('\0'); if (a===source.key) pool.add(b); }
  const out=[];
  for (const key of pool) {
    const p=catalog.problems.get(key);
    if (!p || !platforms.includes(p.platform) || !recommendable(p) || key===source.key || seen.has(key)) continue;
    const concepts=trustedConcepts(p);
    const shared=[...concepts].filter(c=>sourceConcepts.has(c)).sort();
    const union=new Set([...concepts,...sourceConcepts]).size;
    const edge=reviewedEdges.has(source.key+'\0'+key) ? 1 : 0;
    // Semantic fallback (atcoder-concepts.md R8): an unclassified candidate may still be proposed from a title-neighbor
    // channel when both titles are reliable. It is labelled as title similarity only and never credits a concept.
    const rank=ranks.get(key);
    const semanticOnly=!shared.length && !edge && !concepts.size && Boolean(rank) && p.reliability==='reliable' && source.reliability==='reliable';
    if (!shared.length && !edge && !semanticOnly) continue;
    const prediction=predict(p,abilities);
    if (prediction.p_conservative===null) continue;
    const components={
      reviewed_edge:4*edge,
      concept_jaccard:Math.round(3*(union ? shared.length/union : 0)*1000)/1000,
      channel_rank:rank ? Math.round(2*(1/(60+rank.rank))*60*1000)/1000 : 0,
      difficulty_fit:Math.round(2*Math.max(0,1-Math.abs(prediction.p_conservative-0.70)/0.70)*1000)/1000,
      degraded_title:(p.reliability!=='reliable' || source.reliability!=='reliable') ? -1 : 0,
      semantic_only:semanticOnly ? SEMANTIC_ONLY_PENALTY : 0,
    };
    const score=Math.round(Object.values(components).reduce((a,b)=>a+b,0)*1000)/1000;
    const conceptStatuses=new Set(p.concepts.filter(c=>shared.includes(c.id)).map(c=>c.status));
    out.push({problem_key:key,platform:p.platform,title:p.title,url:p.url,score,components,shared_concepts:shared,
      channel:rank?.channel ?? (shared.length?'concept':'reviewed_edge'),channel_rank:rank?.rank ?? null,prediction,semantic_only:semanticOnly,
      evidence_label:semanticOnly ? SEMANTIC_ONLY_LABEL : edge ? 'reviewed relationship' : conceptStatuses.has('approved') ? 'approved concept match' : 'AI-reviewed concept match (unverified relatedness)'});
  }
  out.sort((a,b)=>b.score-a.score || a.problem_key.localeCompare(b.problem_key));
  return {source_key:source.key,channel,policy:RELATED_VERSION,candidates:out.slice(0,limit),pool_size:out.length,
    note:'Related practice candidates share reviewed or AI-reviewed concepts; they are not claimed to be algorithmically equivalent. Candidates marked semantic_only rest on title similarity alone.'};
}
