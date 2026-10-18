// Recommendation impact of AtCoder concept rows and the semantic fallback (atcoder-concepts.md). Aggregate output only.
//   node research/mastery_catalog/impact.mjs <catalog-before.sqlite3> <catalog-after.sqlite3> [stride]
import {DatabaseSync} from 'node:sqlite';
import {loadCatalog,attachCatalog} from '../../hosted/mastery/catalog.mjs';
import {relatedCandidates,trustedConcepts} from '../../hosted/mastery/related.mjs';
import {fitAll} from '../../hosted/mastery/calibration.mjs';

function open(path) { const db=new DatabaseSync(':memory:'); attachCatalog(db,path); return loadCatalog(db); }
const [,, beforePath, afterPath, strideArg] = process.argv;
const stride=Number(strideArg ?? 20);
const before=open(beforePath), after=open(afterPath);
const atcoderEligible=c=>[...c.problems.values()].filter(p=>p.platform==='atcoder' && p.eligible);
const pools=c=>({atcoder_eligible:atcoderEligible(c).length,atcoder_with_trusted_concept:atcoderEligible(c).filter(p=>trustedConcepts(p).size>0).length});

function run(catalog,{semantic}) {
  const abilities=fitAll([],catalog,[]);
  const sources=[...catalog.problems.values()].filter(p=>['codeforces','leetcode'].includes(p.platform) && p.eligible && trustedConcepts(p).size>0 && p.reliability==='reliable')
    .sort((a,b)=>a.key.localeCompare(b.key)).filter((_,i)=>i%stride===0);
  let withAny=0,total=0,semanticTotal=0,withSemantic=0,labeledHits=0;
  const reached=new Set();
  for (const s of sources) {
    const result=relatedCandidates({sourceKey:s.key,catalog,abilities,platforms:['atcoder'],limit:5});
    const cands=result.candidates.filter(c=>semantic || !c.semantic_only);
    withAny+=cands.length>0; total+=cands.length;
    const sem=cands.filter(c=>c.semantic_only).length; semanticTotal+=sem; withSemantic+=sem>0; labeledHits+=cands.length-sem;
    for (const c of cands) reached.add(c.problem_key);
  }
  return {sources:sources.length,sources_with_atcoder_candidate:withAny,share_sources_with_candidate:+(withAny/sources.length).toFixed(3),
    atcoder_candidates:total,labeled_concept_candidates:labeledHits,semantic_only_candidates:semanticTotal,
    share_semantic_only:total?+(semanticTotal/total).toFixed(3):0,distinct_atcoder_problems_reached:reached.size};
}
const out={stride,pools:{before:pools(before),after:pools(after)},
  related_top5_atcoder:{
    before_catalog_concept_only:run(before,{semantic:false}),
    after_catalog_concept_only:run(after,{semantic:false}),
    after_catalog_with_semantic_fallback:run(after,{semantic:true})}};
console.log(JSON.stringify(out,null,1));
