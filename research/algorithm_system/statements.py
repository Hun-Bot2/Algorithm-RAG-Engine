"""Statement-first corpus and CPU-only matched-cohort retrieval experiments."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import argparse,hashlib,json,re,resource,sys,time
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from research.full_catalog.retrieve import Rows,topk,dense_cosine,write_results
from taxonomy import normalize_tags,load_taxonomy

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def unit(x):
    x=np.asarray(x,dtype=np.float32);n=np.linalg.norm(x,axis=1,keepdims=True)
    if not np.isfinite(x).all() or np.any(n==0):raise ValueError('invalid embedding')
    return x/n

def build_corpus(source,catalog):
    """Keep alternative publisher rows; select longest statement per native ID explicitly."""
    bycat={r['problem_key']:r for r in catalog};groups=defaultdict(list)
    for r in source:
        if r.get('statement_text','').strip():groups[r.get('native_key') or r['dataset_key']].append(r)
    corpus=[];alternatives=[]
    for key,versions in sorted(groups.items()):
        ordered=sorted(versions,key=lambda r:(-len(r['statement_text']),r['dataset_key']))
        r=dict(ordered[0]);r['problem_key']=key;r['publisher_versions']=[v['dataset_key'] for v in ordered]
        c=bycat.get(key,{})
        r['source_tags']=c.get('tags',r.get('source_tags')) if r['platform']=='codeforces' else None
        r['native_difficulty']=c.get('native_difficulty',r.get('source_difficulty'))
        r['taxonomy']=normalize_tags(r['platform'],r['source_tags'])
        corpus.append(r)
        if len(ordered)>1:alternatives.append({'key':key,'selected':r['dataset_key'],'versions':r['publisher_versions'],'rule':'longest statement then immutable dataset key; all source rows retained'})
    return corpus,alternatives

def duplicate_audit(corpus):
    hashes=defaultdict(list);titles=defaultdict(list)
    for r in corpus:
        hashes[r['content_sha256']].append(r['problem_key'])
        title=re.sub(r'\W+','',r['title'].casefold())
        if title:titles[title].append(r)
    exact=[v for v in hashes.values() if len(v)>1]
    ambiguous=[{'keys':[r['problem_key'] for r in rows],'basis':'normalized title equality; not verified semantic duplicates','status':'pending_human_review'} for rows in titles.values() if len(rows)>1 and len({r['platform'] for r in rows})>1]
    return exact,ambiguous

def feature_tags(corpus,hierarchy=False):
    features=[]
    for r in corpus:
        if hierarchy:
            norm=r['taxonomy'];direct={m['concept_id'] for m in norm['mapped_concepts'] if m['concept_id'] and m['confidence']>=.85 and m['relation'] in {'equivalent','narrower'}}
            features.append(direct|set(norm['ancestors']))
        else:features.append(set(r.get('source_tags') or []))
    return features

def jaccard_block(features):
    def score(start,stop):
        return np.array([[len(a&b)/len(a|b) if a|b else 0 for b in features] for a in features[start:stop]],dtype=np.float32)
    return score

def token_windows(token_ids,window=254):
    if window<1 or window>254 or any(not ids for ids in token_ids):
        raise ValueError('nonempty tokens and encoder-compatible window required')
    return [(i,ids[a:a+window]) for i,ids in enumerate(token_ids) for a in range(0,len(ids),window)]

def run(source,output):
    if output.exists():raise ValueError('fresh immutable run required')
    started=time.perf_counter();output.mkdir(parents=True)
    (output/'implementation.py').write_bytes(Path(__file__).read_bytes())
    catalog=read(ROOT/'private-learning/full-catalog/catalog-v1/catalog.v1.jsonl')
    corpus,alternatives=build_corpus(read(source),catalog)
    exact,ambiguous=duplicate_audit(corpus)
    with (output/'corpus.jsonl').open('x') as f:
        for row in corpus:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    (output/'corpus.jsonl').chmod(0o600)
    write(output/'alternatives.json',alternatives);write(output/'duplicate-review.json',{'exact_statement_groups':exact,'ambiguous_title_groups':ambiguous,'no_records_deleted':True})
    keys=[r['problem_key'] for r in corpus];write(output/'keys.json',keys)
    rows=Rows(keys,[r['platform'] for r in corpus],[r['content_sha256'] for r in corpus],[None]*len(corpus))
    metrics={};vectors={};texts=[r['statement_text'] for r in corpus]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import save_npz
    for name,config in [('tfidf_word',{'analyzer':'word','ngram_range':(1,2)}),('tfidf_char',{'analyzer':'char','ngram_range':(3,5),'max_features':100000})]:
        tick=time.perf_counter();v=TfidfVectorizer(dtype=np.float32,min_df=2,**config);x=v.fit_transform(texts);save_npz(output/(name+'.npz'),x)
        def block(a,b):return (x[a:b]@x.T).toarray()
        hashes,counts=write_results(rows,topk(rows,block,50,'cross',block_size=128),output/name)
        metrics[name]={'seconds':time.perf_counter()-tick,'shape':list(x.shape),'nonzero':x.nnz,'outputs':hashes,**counts}
        print(name,metrics[name]['seconds'],flush=True)
    # Validate pinned model before opening offline-only local weights.
    from src.learning.embedding_backend import LocalEmbeddingBackend
    manifest=json.loads((ROOT/'research/minilm_pilot/model.manifest.json').read_text())
    backend=LocalEmbeddingBackend(ROOT/'private-learning/models/minilm-1110a243fdf4706b3f48f1d95db1a4f5529b4d41',manifest)
    encoder=backend.encoder;tokenizer=encoder.tokenizer
    token_ids=[tokenizer.encode(t,add_special_tokens=False,truncation=False) for t in texts]
    lengths=np.array([len(t) for t in token_ids]);
    if np.any(lengths==0):raise ValueError('empty tokenized description')
    write(output/'token-lengths.json',lengths.tolist())
    def encode(ts):return unit(encoder.encode(ts,batch_size=64,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True))
    for name,ts in [('minilm_title',[r['title'] for r in corpus]),('minilm_title_statement',[r['title']+'\n'+r['statement_text'] for r in corpus])]:
        tick=time.perf_counter();vectors[name]=encode(ts);np.save(output/(name+'.npy'),vectors[name])
        metrics[name]={'encode_seconds':time.perf_counter()-tick,'max_tokens':256,'truncation':'prefix beyond tokenizer limit'}
    tick=time.perf_counter()
    # Disjoint original token windows avoid decode/re-tokenize boundary changes.
    import torch
    pooled=np.zeros((len(corpus),384),dtype=np.float64);totals=np.zeros(len(corpus))
    positions=token_windows(token_ids)
    with torch.inference_mode():
        for start in range(0,len(positions),64):
            batch=positions[start:start+64]
            prepared=[tokenizer.prepare_for_model(ids,add_special_tokens=True,truncation=False,return_attention_mask=True) for _,ids in batch]
            if any(len(p['input_ids'])>256 for p in prepared):raise ValueError('chunk exceeds encoder limit')
            features=tokenizer.pad(prepared,padding=True,return_tensors='pt')
            embedded=encoder(features)['sentence_embedding'];embedded=torch.nn.functional.normalize(embedded,p=2,dim=1).cpu().numpy()
            for (owner,ids),v in zip(batch,embedded):pooled[owner]+=v*len(ids);totals[owner]+=len(ids)
    if not np.array_equal(totals,lengths):raise ValueError('chunk coverage invariant failed')
    vectors['minilm_chunk']=unit(pooled/totals[:,None]);np.save(output/'minilm_chunk.npy',vectors['minilm_chunk'])
    metrics['minilm_chunk']={'encode_seconds':time.perf_counter()-tick,'chunks':len(positions),'window_content_tokens':254,'overlap':0,'aggregation':'token-count weighted mean of normalized chunks, then L2','all_content_tokens_covered':True,'statement_completeness':'publisher available description; section completeness unverified'}
    for name,v in vectors.items():
        tick=time.perf_counter();hashes,counts=write_results(rows,topk(rows,dense_cosine(v),50,'cross',block_size=128),output/name)
        # Numerical sanity independent of BLAS on selected entries.
        sample=np.einsum('ij,ij->i',v[:20].astype(float),v[-20:].astype(float))
        with np.errstate(all='ignore'):actual=np.sum(v[:20]*v[-20:],axis=1)
        if not np.allclose(sample,actual,atol=1e-5):raise ValueError('dense sanity failed')
        metrics[name].update(retrieval_seconds=time.perf_counter()-tick,outputs=hashes,**counts,vector_sha256=digest(output/(name+'.npy')),vector_bytes=v.nbytes)
        print(name,'retrieved',flush=True)
    tagged=[r for r in corpus if r.get('source_tags')]
    tag_platforms=Counter(r['platform'] for r in tagged)
    if len(tag_platforms)<2:
        metrics['tag_jaccard']=metrics['hierarchical_taxonomy']=metrics['hybrid']={'status':'blocked_no_cross_platform_matched_tag_coverage','available_tagged_platforms':dict(tag_platforms),'missing_tags_not_imputed':True}
    else:
        # Matched tag-covered cohort only: every method gets the same candidates.
        indices=[i for i,r in enumerate(corpus) if r.get('source_tags')]
        tr=Rows([r['problem_key'] for r in tagged],[r['platform'] for r in tagged],[r['content_sha256'] for r in tagged],[None]*len(tagged))
        semantic=dense_cosine(vectors['minilm_chunk'][indices])
        raw=jaccard_block(feature_tags(tagged));hier=jaccard_block(feature_tags(tagged,True))
        def hybrid(a,b):return .5*semantic(a,b)+.5*hier(a,b)
        for name,score in [('tag_jaccard',raw),('hierarchical_taxonomy',hier),('hybrid',hybrid)]:
            hashes,counts=write_results(tr,topk(tr,score,50,'cross',block_size=128),output/('tag_cohort_'+name))
            metrics[name]={'status':'complete','matched_tag_cohort_rows':len(tagged),'outputs':hashes,**counts,'hybrid_weights':{'semantic':.5,'taxonomy':.5} if name=='hybrid' else None}
    receipt={'schema_version':1,'source_sha256':digest(source),'corpus_sha256':digest(output/'corpus.jsonl'),'count':len(corpus),'platforms':dict(Counter(r['platform'] for r in corpus)),'native_identity_available':sum(bool(r['native_key']) for r in corpus),'catalog_matches':sum(r['catalog_match'] for r in corpus),'model_manifest_sha256':digest(ROOT/'research/minilm_pilot/model.manifest.json'),'token_lengths':{'min':int(lengths.min()),'median':float(np.median(lengths)),'p95':float(np.quantile(lengths,.95)),'max':int(lengths.max()),'exceed_254':int(sum(lengths>254))},'methods':metrics,'seconds':time.perf_counter()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'leetcodes_permitted_statements':0,'no_model_promotion':True,'exact_cross_platform_all_rows':True,'top_k':[10,20,50]}
    receipt['baseline']={'status':'complete','method':'minilm_chunk','dimensions':384,'corpus_sha256':receipt['corpus_sha256'],'vector_sha256':digest(output/'minilm_chunk.npy'),'top50_sha256':digest(output/'minilm_chunk/top50.jsonl'),'top50_path':'minilm_chunk/top50.jsonl'}
    write(output/'manifest.json',receipt);return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source,a.output)
