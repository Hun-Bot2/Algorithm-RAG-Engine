/** Bounded candidate evidence. These signals never establish learner mastery. */
const tokens = text => new Set(String(text).toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []);
const jaccard = (a,b) => { const union=new Set([...a,...b]);return union.size?[...a].filter(v=>b.has(v)).length/union.size:0; };
export function relationshipProviders(curriculum,annotations={}) {
  const problems=new Map(curriculum.problems.map(p=>[p.problem_key,p]));
  const version=curriculum.curriculum_version;
  const build=(id,score,evidence)=>({id,asset_version:version,candidates(source){
    if(!problems.has(source))return [];
    return [...problems.keys()].filter(key=>key!==source).map(key=>({problem_key:key,score:score(source,key),evidence:evidence(source,key),asset_version:version}))
      .filter(row=>row.score>0).sort((a,b)=>b.score-a.score||a.problem_key.localeCompare(b.problem_key)).slice(0,20);
  }});
  const text=key=>tokens([annotations[key]?.title,annotations[key]?.concept_rationale,...(problems.get(key)?.concepts ?? [])].join(' '));
  return [
    build('reviewed-graph',(source,target)=>curriculum.relationships.some(r=>r.review.status==='approved' && r.source===source && r.target===target)?1:0,
      (source,target)=>curriculum.relationships.filter(r=>r.review.status==='approved'&&r.source===source&&r.target===target).map(r=>({type:r.type,review:r.review}))),
    build('metadata',(source,target)=>jaccard(new Set(problems.get(source).concepts),new Set(problems.get(target).concepts)),
      (source,target)=>({shared_concepts:problems.get(source).concepts.filter(c=>problems.get(target).concepts.includes(c))})),
    build('lexical',(source,target)=>jaccard(text(source),text(target)),()=>({basis:'independently_authored_annotation_token_overlap'})),
  ];
}
export function fuseCandidates(providers,sourceKeys) {
  const candidates=new Map(),statuses=[];
  for(const provider of providers){
    if(!sourceKeys.length){statuses.push({provider:provider.id,status:"not_requested"});continue;}
    try{
      const rows=sourceKeys.flatMap(source=>provider.candidates(source).map((row,index)=>({...row,source,rank:index+1})));
      if(rows.some(row=>typeof row.problem_key!=='string'||!Number.isFinite(row.score)||!Number.isInteger(row.rank)))throw Error('invalid candidates');
      for(const row of rows){
        const item=candidates.get(row.problem_key) ?? {rank_fusion:0,evidence:[]};
        // Reciprocal rank fusion, k=60: no averaging incompatible dense scores.
        item.rank_fusion+=1/(60+row.rank);item.evidence.push({provider:provider.id,...row});candidates.set(row.problem_key,item);
      }
      statuses.push({provider:provider.id,status:'available',asset_version:provider.asset_version});
    }catch{statuses.push({provider:provider.id,status:'unavailable'});}
  }
  return {candidates,statuses};
}
