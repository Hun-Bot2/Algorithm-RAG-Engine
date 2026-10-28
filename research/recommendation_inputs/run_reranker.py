"""Run an explicitly prepared offline Cross-encoder on matched curriculum candidates."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from .compare import ROOT,digest,verify,compare_rankings
from .cross_encoder import load_local,rerank_bundle,source_query,MODEL_ID,REVISION


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--model-manifest',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=ROOT/'private-learning/recommendation-inputs/curriculum-ideas-v1')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--query-mode',choices=['instruction','plain'],default='instruction')
    args=parser.parse_args()
    if args.output.exists() or args.receipt.exists():raise ValueError('fresh output/receipt required')
    source_receipt=json.loads((ROOT/'research/recommendation_inputs/curriculum-ideas-v1.receipt.json').read_text())
    if json.loads((args.source/'manifest.json').read_text())!=source_receipt:raise ValueError('source manifest mismatch')
    started=time.perf_counter();model,model_manifest=load_local(args.model,args.model_manifest)
    args.output.mkdir(parents=True,mode=0o700)
    scopes={};runs={};all_seconds=0
    for name,info in source_receipt['scopes'].items():
        path=args.source/info['file'];verify(path,info['sha256'])
        bundle=json.loads(path.read_text());truncation={'over_limit':0,'max_pair_tokens':0};tick=time.perf_counter()
        def score(pairs):
            lengths=[len(model.tokenizer.encode(a,b,add_special_tokens=True,truncation=False)) for a,b in pairs]
            truncation['over_limit']=sum(n>512 for n in lengths);truncation['max_pair_tokens']=max(lengths,default=0)
            if truncation['over_limit']:raise ValueError('pair inputs exceed context; no silent truncation')
            import torch
            return model.predict(pairs,batch_size=32,show_progress_bar=False,activation_fn=torch.nn.Identity(),convert_to_numpy=True).reshape(-1).tolist()
        rankings,r=rerank_bundle(bundle,score,query_mode=args.query_mode);elapsed=time.perf_counter()-tick;all_seconds+=elapsed
        result=copy.deepcopy(bundle)
        result['rankings']['cross_encoder']=rankings
        result['methods']['cross_encoder']={'label':'Cross-encoder 재정렬 (실험)','description':'MS MARCO 범용 모델 · 두 풀이 입력을 함께 처리 · raw logit, 확률/코사인 아님','score_label':'Cross-encoder raw logit (정답 확률 아님)'}
        result['default_a']='idea_structured';result['default_b']='cross_encoder'
        result['label']=('LeetCode' if name=='strategy_leetcode' else '3플랫폼')+' 재정렬 ('+('설명 직접' if args.query_mode=='plain' else '질문형')+')'
        result['coverage']=f"{result['label']} · {len(bundle['problems'])}개 기준 문제 · 두 설명 검색의 Top10 합집합 후보"
        result['notice']='범용 Cross-encoder로 같은 후보를 재정렬한 실험입니다. 알고리즘 관계나 학습 효과를 검증한 모델이 아니며 raw logit은 확률이 아닙니다.'
        for p in result['problems'].values():p['input_views']['cross_encoder_query']=source_query(p,args.query_mode)
        scope=('rerank_plain_' if args.query_mode=='plain' else 'rerank_')+('leetcode' if name=='strategy_leetcode' else 'all')
        destination=args.output/f'{scope}.json'
        with destination.open('x') as handle:json.dump(result,handle,ensure_ascii=False,separators=(',',':'))
        destination.chmod(0o600)
        scopes[scope]={'file':destination.name,'sha256':digest(destination),'label':result['label']}
        diagnostics=compare_rankings({method:{q:[key for key,_ in values] for q,values in rows.items()} for method,rows in result['rankings'].items()})
        runs[scope]={**r,**truncation,'inference_and_pool_seconds':elapsed,'diagnostics':diagnostics}
        print(json.dumps({'scope':scope,'pairs':r['pair_count'],'seconds':elapsed}),flush=True)
    receipt={'schema':'cross-encoder-experiment-v1','model_id':MODEL_ID,'revision':REVISION,'query_mode':args.query_mode,'model_manifest_sha256':digest(args.model_manifest),
             'source_receipt_sha256':digest(ROOT/'research/recommendation_inputs/curriculum-ideas-v1.receipt.json'),
             'scopes':scopes,'runs':runs,'run_seconds':time.perf_counter()-started,'inference_seconds':all_seconds,
             'human_quality_evaluation':False,'new_embeddings_generated':False,'training_performed':False,'paid_api_calls':0}
    text=json.dumps(receipt,ensure_ascii=False,indent=2)+'\n';(args.output/'manifest.json').write_text(text)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    with args.receipt.open('x') as handle:handle.write(text)


if __name__=='__main__':main()
