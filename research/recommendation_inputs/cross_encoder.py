"""Explicit pinned Cross-encoder preparation; never downloads on import or inference."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import urllib.parse

from .compare import ROOT, digest

MODEL_ID='cross-encoder/ms-marco-MiniLM-L6-v2'
REVISION='233902d25c440f23af6f7d6e94d2946bac0bee0a'
WEIGHTS_SHA256='821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae'
FILES=('model.safetensors','config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.txt','README.md')
PLAN={'model_id':MODEL_ID,'revision':REVISION,'weights_sha256':WEIGHTS_SHA256,'max_download_bytes':120*1024*1024,
      'files':list(FILES),'license':'Apache-2.0','purpose':'generic passage-reranking baseline, not validated algorithm equivalence',
      'scope':'offline CPU inference on authored curriculum candidate union; no training or paid API',
      'approval_required_by':'AGENTS.md: only user-approved models may be fetched'}


def download_approved(output,manifest_path,fetcher=None):
    # Caller must obtain approval for this exact pinned model before invoking this function.
    from scripts.learning.download_minilm import fetch,write_new,json_bytes
    fetcher=fetcher or fetch
    output,manifest_path=Path(output),Path(manifest_path)
    if output.exists() or manifest_path.exists() or manifest_path.resolve().is_relative_to(output.resolve()):
        raise ValueError('fresh model directory and external manifest required')
    base=f'https://huggingface.co/{MODEL_ID}'
    metadata=json.loads(fetcher(f'https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true',8*1024*1024))
    if metadata.get('id')!=MODEL_ID or metadata.get('sha')!=REVISION:
        raise ValueError('model identity/revision mismatch')
    siblings={r['rfilename']:r for r in metadata.get('siblings',[])}
    if any(name not in siblings for name in FILES):
        raise ValueError('required allowlisted file missing')
    if siblings['model.safetensors'].get('lfs',{}).get('sha256')!=WEIGHTS_SHA256:
        raise ValueError('upstream weights differ from reviewed hash')
    output.mkdir(parents=True,mode=0o700)
    total=0;hashes={}
    for name in FILES:
        entry=siblings[name];size=entry.get('lfs',{}).get('size',entry.get('size'))
        if not isinstance(size,int) or size<0 or total+size>PLAN['max_download_bytes']:
            raise ValueError('unknown/oversized file')
        data=fetcher(f'{base}/resolve/{REVISION}/{urllib.parse.quote(name)}',size)
        h=hashlib.sha256(data).hexdigest()
        if len(data)!=size or name=='model.safetensors' and h!=WEIGHTS_SHA256:
            raise ValueError('downloaded size/hash mismatch')
        write_new(output/name,data);hashes[name]=h;total+=len(data)
        print(f'downloaded {name}: {len(data)} bytes',flush=True)
    manifest={**PLAN,'files_sha256':hashes,'total_bytes':total,'metadata_sha256':hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()}
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    write_new(manifest_path,json_bytes(manifest))
    return manifest


def load_local(directory,manifest_path):
    directory=Path(directory);manifest=json.loads(Path(manifest_path).read_text())
    if manifest.get('model_id')!=MODEL_ID or manifest.get('revision')!=REVISION or manifest.get('weights_sha256')!=WEIGHTS_SHA256:
        raise ValueError('unreviewed Cross-encoder manifest')
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    if actual!=set(FILES) or set(manifest['files_sha256'])!=set(FILES):
        raise ValueError('model directory membership mismatch')
    for name,h in manifest['files_sha256'].items():
        if digest(directory/name)!=h:
            raise ValueError('model file hash mismatch')
    if manifest['files_sha256']['model.safetensors']!=WEIGHTS_SHA256:
        raise ValueError('weights hash mismatch')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch
    from sentence_transformers import CrossEncoder
    torch.set_num_threads(1)
    model=CrossEncoder(str(directory),device='cpu',max_length=512,activation_fn=torch.nn.Identity(),
                       local_files_only=True,trust_remote_code=False,model_kwargs={'use_safetensors':True})
    return model,manifest


def source_query(problem,query_mode='instruction'):
    rationale=problem['input_views']['idea_rationale']
    if query_mode=='plain':return rationale
    if query_mode=='instruction':return 'Which problem uses the same core solving idea as this? '+rationale
    raise ValueError('unsupported query mode')


def rerank_bundle(bundle,score_pairs,query_mode='instruction'):
    keys=list(bundle['problems']);pairs=[];owners=[];pool_sizes={}
    for source in keys:
        p=bundle['problems'][source]
        query=source_query(p,query_mode)
        pool=sorted({target for name in ('idea_rationale','idea_structured') for target,_ in bundle['rankings'][name][source]})
        if source in pool or any(target not in bundle['problems'] for target in pool):
            raise ValueError('invalid reranking candidate pool')
        pool_sizes[source]=len(pool)
        for target in pool:
            pairs.append((query,bundle['problems'][target]['input_views']['idea_structured']))
            owners.append((source,target))
    scores=score_pairs(pairs)
    if len(scores)!=len(pairs) or any(not math.isfinite(float(s)) for s in scores):
        raise ValueError('invalid reranker output')
    result={source:[] for source in keys}
    for (source,target),score in zip(owners,scores):result[source].append([target,float(score)])
    for source,rows in result.items():
        rows.sort(key=lambda p:(-p[1],p[0]));result[source]=rows[:10]
    return result,{'pair_count':len(pairs),'query_count':len(keys),'pool_sizes':pool_sizes,'score_kind':'raw logit; not probability or cosine','query_mode':query_mode}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--download-approved-model',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--manifest',type=Path)
    args=parser.parse_args()
    if not args.download_approved_model:
        print(json.dumps(PLAN,indent=2));return
    if not args.output or not args.manifest:parser.error('--output and --manifest are required for an approved download')
    download_approved(args.output,args.manifest)


if __name__=='__main__':main()
