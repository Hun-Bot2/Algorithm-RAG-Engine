"""Offline input ablations over admitted publisher text or authored curriculum."""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np

from .compare import ROOT, digest, verify, compare_rankings
from .inputs import section_inputs, curriculum_inputs, token_chunks
from research.full_catalog.retrieve import dense_cosine, rank_row

MODEL = ROOT / 'private-learning/models/minilm-1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MODEL_MANIFEST = ROOT / 'research/minilm_pilot/model.manifest.json'


def unit(vectors):
    v = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    if not np.isfinite(v).all() or np.any(norms <= 0):
        raise ValueError('invalid vectors')
    return v / norms


def backend():
    import torch
    from src.learning.embedding_backend import LocalEmbeddingBackend
    torch.set_num_threads(1)
    return LocalEmbeddingBackend(MODEL, json.loads(MODEL_MANIFEST.read_text()))


def encode_chunks(texts, model):
    import torch
    encoder = model.encoder
    tokenizer = encoder.tokenizer
    ids = [tokenizer.encode(text, add_special_tokens=False, truncation=False) for text in texts]
    chunks = token_chunks(ids)
    pooled = np.zeros((len(texts), 384), dtype=np.float64)
    covered = np.zeros(len(texts), dtype=np.int64)
    start_time = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(chunks), 64):
            batch = chunks[start:start+64]
            prepared = [tokenizer.prepare_for_model(tokens, add_special_tokens=True, truncation=False, return_attention_mask=True) for _, tokens in batch]
            if any(len(row['input_ids']) > 256 for row in prepared):
                raise ValueError('chunk exceeds encoder context')
            features = tokenizer.pad(prepared, padding=True, return_tensors='pt')
            v = torch.nn.functional.normalize(encoder(features)['sentence_embedding'], p=2, dim=1).cpu().numpy()
            for (owner, tokens), vector in zip(batch, v):
                pooled[owner] += vector * len(tokens)
                covered[owner] += len(tokens)
            if start % 1024 == 0:
                print(f'encoded {min(start+64,len(chunks))}/{len(chunks)} chunks', flush=True)
    if not np.array_equal(covered, np.array([len(tokens) for tokens in ids])):
        raise ValueError('original token coverage lost')
    return unit(pooled / covered[:, None]), {'encode_seconds': time.perf_counter()-start_time,
            'chunks': len(chunks), 'selected_tokens': int(sum(covered)), 'all_selected_tokens_covered': True}


def retrieve(problems, vectors, mode='all', k=10, hashes=None):
    keys = list(problems)
    vectors = unit(vectors)
    if len(vectors) != len(keys):
        raise ValueError('vector/key alignment mismatch')
    platforms = np.array([problems[key]['platform'] for key in keys])
    hashes = np.array(hashes if hashes is not None else keys)
    result = {}
    score = dense_cosine(vectors)
    for start in range(0, len(keys), 128):
        stop = min(start+128, len(keys))
        with np.errstate(all='ignore'):
            scores = score(start, stop)
        if not np.isfinite(scores).all():
            raise ValueError('invalid cosine block')
        for offset in range(stop-start):
            i = start+offset
            mask = hashes == hashes[i]
            if mode == 'cross':
                mask = mask | (platforms == platforms[i])
            elif mode == 'same':
                mask = mask | (platforms != platforms[i])
            elif mode != 'all':
                raise ValueError('invalid candidate scope')
            picked = rank_row(scores[offset], mask, k)
            result[keys[i]] = [[keys[j], float(scores[offset, j])] for j in picked]
    return result


def compare_results(results):
    # Only matched lists with enough candidates contribute to overlap; no quality claim.
    runs = {method: {q: [key for key, score in values] for q, values in rows.items()} for method, rows in results.items()}
    return compare_rankings(runs)


def metadata_problem(row):
    fields = ('problem_key','title','platform','native_key','statement_text','source_dataset','source_revision','license','collection_status','constraints_status','source_tags','native_difficulty','original_url')
    return {key: row.get(key) for key in fields}


def write_scope(output, name, bundle):
    path = output / f'{name}.json'
    with path.open('x') as handle:
        json.dump(bundle, handle, ensure_ascii=False, separators=(',', ':'))
    path.chmod(0o600)
    return {'file': path.name, 'sha256': digest(path), 'label': bundle['label']}


def curriculum(output, model):
    release = json.loads((ROOT/'assets/curriculum_release.v1.json').read_text())
    for name, expected in release['files'].items():
        verify(ROOT/name, expected)
    seed = json.loads((ROOT/'curriculum/seed.v1.json').read_text())
    annotations = json.loads((ROOT/'curriculum/annotations.v1.json').read_text())['annotations']
    problems, texts = {}, {name: [] for name in ('title','idea_rationale','idea_structured')}
    for p in sorted(seed['problems'], key=lambda p: p['problem_key']):
        key = p['problem_key']; a = annotations[key]; view = curriculum_inputs(p,a)
        problems[key] = {'problem_key': key, 'title': a['title'], 'platform': a['platform'], 'native_key': key,
                         'original_url': a['source_url'], 'native_difficulty': p['native_difficulty'], 'source_tags': None,
                         'statement_text': None, 'source_dataset': 'authored curriculum annotations.v1',
                         'source_revision': seed['curriculum_version'], 'license': None,
                         'collection_status': 'authored AI-drafted/reviewed rationale; not original problem text',
                         'constraints_status': 'not provided', 'input_views': view,
                         'input_provenance': release['provenance']}
        for name in texts:
            texts[name].append(view[name])
    vectors, timings = {}, {}
    for name, values in texts.items():
        print('encoding curriculum '+name, flush=True)
        vectors[name], timings[name] = encode_chunks(values, model)
        np.save(output/f'{name}.npy', vectors[name])
    methods = {
        'idea_rationale': {'label':'핵심 풀이 설명', 'description':'새 임베딩 · 기존 작성 핵심 설명만 사용 · AI 검토 자료, 정답 검증 아님','score_label':'작성 설명의 코사인'},
        'idea_structured': {'label':'구조화한 풀이 설명', 'description':'새 임베딩 · CORE IDEA + CONCEPTS · 미기록 증명/복잡도는 생성하지 않음','score_label':'구조화 설명의 코사인'},
        'title': {'label':'제목 (실험 기준선)', 'description':'동일 모델·동일 문제 집합에서 새로 계산한 제목 기준선','score_label':'제목 코사인'},
    }
    scopes, diagnostics = {}, {}
    all_keys = list(problems)
    for scope, platform in [('strategy_leetcode','leetcode'), ('strategy_all',None)]:
        selected = [i for i,key in enumerate(all_keys) if platform is None or problems[key]['platform']==platform]
        ps = {all_keys[i]:problems[all_keys[i]] for i in selected}
        results = {name: retrieve(ps,vectors[name][selected]) for name in methods}
        diagnostics[scope] = compare_results(results)
        label = 'LeetCode 풀이 설명 실험' if platform else '3플랫폼 풀이 설명 실험'
        bundle = {'problems':ps,'rankings':results,'methods':methods,'default_a':'idea_rationale','default_b':'idea_structured',
                  'default_key':'leetcode:two-sum' if 'leetcode:two-sum' in ps else next(iter(ps)), 'label':label,
                  'coverage':f'{label} · 동일 {len(ps)}개 문제 · 새 384D 임베딩',
                  'notice':'커리큘럼의 작성·AI 검토 설명을 임베딩한 새 실험입니다. 문제 본문이나 사람 정답 라벨이 아닙니다. 입력 필드와 후보를 직접 확인해 주세요.',
                  'empty_message':'이 작성 설명 코호트에 없는 문제입니다. 전체 LeetCode 제목/태그 모드에서 확인할 수 있습니다.',
                  'footer':'원래 문제 본문은 포함하지 않습니다. AI 초안·검토된 작성 설명이며 인간 gold나 학습 성과 검증이 아닙니다.'}
        scopes[scope] = write_scope(output,scope,bundle)
    return {'kind':'curriculum','counts':dict(Counter(p['platform'] for p in problems.values())),
            'input_sha256':release['files'],'authorship':release['provenance'],'timings':timings,'diagnostics':diagnostics,
            'scopes':scopes,'new_embeddings_generated':True,'quality_status':'unavailable; no applicable human judgments'}


def publisher(output, model):
    source = ROOT/'private-learning/algorithm-system/statement-run-v1'
    old = json.loads((ROOT/'research/algorithm_system/statement-baseline.v1.json').read_text())
    verify(source/'corpus.jsonl',old['corpus_sha256'])
    rows = [json.loads(line) for line in (source/'corpus.jsonl').read_text().splitlines()]
    if old['model_manifest_sha256']!=digest(MODEL_MANIFEST) or json.loads((source/'keys.json').read_text())!=[r['problem_key'] for r in rows]:
        raise ValueError('baseline model/key alignment mismatch')
    parsed = [section_inputs(row['statement_text']) for row in rows]
    selected = [i for i, view in enumerate(parsed) if not view['issues']]
    if not selected:
        raise ValueError('no matched parsed cohort')
    problems = {}
    for i in selected:
        p = metadata_problem(rows[i]); v = parsed[i]
        p['input_views'] = {'task_sections':v['task_text'], 'io_constraints':v['io_text'], 'section_presence':v['present'],
                            'extraction':'exact standalone headings; example/note blocks excluded from new inputs'}
        p['input_provenance'] = 'Source publisher description; deterministic section extraction, no algorithm inference'
        problems[p['problem_key']] = p
    vectors, timings = {}, {}
    for name in ('title','full_statement'):
        old_name = 'minilm_title' if name=='title' else 'minilm_chunk'
        path = source/f'{old_name}.npy'
        verify(path,old['methods'][old_name]['vector_sha256'])
        raw = np.load(path,allow_pickle=False)
        if len(raw)!=len(rows):
            raise ValueError('baseline vector alignment mismatch')
        vectors[name] = raw[selected]
        timings[name] = {'reused_existing_vectors':True}
    for name, field in [('task_sections','task_text'),('io_constraints','io_text')]:
        print('encoding publisher '+name,flush=True)
        vectors[name],timings[name] = encode_chunks([parsed[i][field] for i in selected],model)
        np.save(output/f'{name}.npy',vectors[name])
    np.save(output/'source-indices.npy',np.array(selected,dtype=np.int32))
    methods = {
        'task_sections':{'label':'과제·입출력·제약','description':'새 임베딩 · description + Input + Output + 별도 Constraints · 예제/노트 제외','score_label':'섹션 구조화 코사인'},
        'full_statement':{'label':'전체 본문 (기준선)','description':'기존 full-chunk 벡터 · 같은 선택 코호트에서 검색 재계산','score_label':'전체 본문 코사인'},
        'io_constraints':{'label':'입출력·제약만','description':'새 임베딩 · Input + Output + 별도 Constraints · 이야기/예제 제외','score_label':'입출력/제약 코사인'},
        'title':{'label':'제목 (실험 기준선)','description':'기존 제목 벡터 · 같은 선택 코호트에서 검색 재계산','score_label':'제목 코사인'},
    }
    started = time.perf_counter()
    result = {name:retrieve(problems,vectors[name],mode='cross',hashes=[rows[i]['content_sha256'] for i in selected]) for name in methods}
    retrieval_seconds = time.perf_counter()-started
    bundle = {'problems':problems,'rankings':result,'methods':methods,'default_a':'task_sections','default_b':'full_statement',
              'default_key':next((k for k,p in problems.items() if p['platform']=='codeforces' and p.get('native_key')),next(iter(problems))),
              'label':'본문 구조 실험 (CF↔AtCoder)','coverage':f'본문 구조 실험 · 동일 {len(problems):,}개 문제 · 반대 플랫폼 후보',
              'notice':'실제 본문에서 섹션을 분리해 새로 임베딩했습니다. 예제/노트를 제거한 효과와 원본 전체 본문을 비교합니다. 해법을 추론한 입력은 아닙니다.',
              'empty_message':'공통 파싱 조건을 통과한 코호트에서 찾지 못했습니다.',
              'footer':'기존 허용 publisher description 연구 코호트. LeetCode 본문 제외. AtCoder 네이티브 ID 미확인 유지. 정답/교육 효과 검증 전입니다.'}
    scopes={'structured_content':write_scope(output,'structured_content',bundle)}
    issues = Counter(issue for view in parsed for issue in view['issues'])
    return {'kind':'publisher','source_count':len(rows),'matched_count':len(selected),
            'counts':dict(Counter(rows[i]['platform'] for i in selected)), 'excluded_issues':dict(issues),
            'missing_standalone_constraints':sum(not parsed[i]['present']['constraints'] for i in selected),
            'input_corpus_sha256':old['corpus_sha256'],'timings':timings,'retrieval_seconds':retrieval_seconds,
            'diagnostics':compare_results(result),'scopes':scopes,'new_embeddings_generated':True,
            'quality_status':'unavailable; no applicable human judgments'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--kind',choices=['curriculum','publisher'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise ValueError('fresh immutable output and receipt required')
    args.output.mkdir(parents=True,mode=0o700)
    started=time.perf_counter()
    result=(curriculum if args.kind=='curriculum' else publisher)(args.output,backend())
    result.update(schema='input-ablation-v1',model_manifest_sha256=digest(MODEL_MANIFEST),dimensions=384,
                  run_seconds=time.perf_counter()-started,training_performed=False,paid_api_calls=0)
    for p in args.output.iterdir():
        if p.is_file():p.chmod(0o600)
    result['files_sha256']={p.name:digest(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    (args.output/'manifest.json').write_text(text)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    with args.receipt.open('x') as handle:handle.write(text)
    print(json.dumps({'kind':args.kind,'scopes':list(result['scopes']),'seconds':result['run_seconds'],'output':str(args.output)}),flush=True)


if __name__=='__main__':main()
