"""Build same-LeetCode title cosine and tag baselines from existing metadata/vectors."""
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np

from .compare import ROOT, digest, verify

METHODS = {
    'lc_title_cosine': {'label': '제목 코사인', 'description': '기존 MiniLM 제목 벡터 · LeetCode 내부에서 새로 계산 · 본문/해법 비교 아님', 'score_label': '제목 벡터 코사인 유사도'},
    'lc_tag_jaccard': {'label': '태그 겹침', 'description': '플랫폼 태그 Jaccard · 공유 태그가 있는 후보만 · 알고리즘 동등성 미검증', 'score_label': '태그 교집합 / 합집합'},
}


def rankings(problems, vectors, k=10):
    keys = list(problems)
    if len(keys) != len(vectors) or vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError('invalid aligned vector matrix')
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0):
        raise ValueError('zero vector')
    unit = vectors / norms[:, None]
    targets = [i for i, key in enumerate(keys) if problems[key]['platform'] == 'leetcode' and problems[key]['eligible'] and problems[key]['status'] == 'active']
    results = {name: {} for name in METHODS}
    tags = [set(problems[key]['source_tags']) for key in keys]
    for i, key in enumerate(keys):
        if problems[key]['platform'] != 'leetcode':
            raise ValueError('source platform must be LeetCode')
        with np.errstate(all='ignore'):
            scores = unit[targets] @ unit[i]
        if not np.isfinite(scores).all():
            raise ValueError('non-finite cosine')
        pairs = [(keys[j], float(score)) for j, score in zip(targets, scores) if j != i]
        pairs.sort(key=lambda pair: (-pair[1], pair[0]))
        results['lc_title_cosine'][key] = pairs[:k]
        overlaps = []
        if tags[i]:
            for j in targets:
                shared = tags[i] & tags[j]
                if i != j and shared:
                    overlaps.append((keys[j], len(shared) / len(tags[i] | tags[j])))
        overlaps.sort(key=lambda pair: (-pair[1], pair[0]))
        results['lc_tag_jaccard'][key] = overlaps[:k]
    return results, len(targets)


def build(catalog_dir):
    receipt = json.loads((catalog_dir / 'receipt.json').read_text())
    catalog_file = catalog_dir / receipt['output']
    verify(catalog_file, receipt['output_sha256'])
    run = catalog_dir / 'run'
    verify(run / 'keys.json', receipt['input_sha256']['keys'])
    keys = json.loads((run / 'keys.json').read_text())
    vectors = np.load(run / 'vectors.npy', allow_pickle=False)
    if vectors.shape != (len(keys), 384) or vectors.dtype != np.float32 or len(keys) != len(set(keys)):
        raise ValueError('unexpected cached vector identity/shape')
    conn = sqlite3.connect(f'file:{catalog_file}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in conn.execute("SELECT problem_key,title,platform,url,difficulty_value,access,eligible,status FROM problems WHERE platform='leetcode' ORDER BY problem_key")]
        tag_rows = conn.execute("SELECT problem_key,native_tag FROM problem_tags WHERE problem_key LIKE 'leetcode:%' ORDER BY problem_key,native_tag").fetchall()
    finally:
        conn.close()
    tags = {row['problem_key']: [] for row in rows}
    for key, tag in tag_rows:
        tags[key].append(tag)
    refresh = json.loads((catalog_dir / 'refresh-receipt.json').read_text())
    problems = {}
    for row in rows:
        key = row['problem_key']
        problems[key] = {**row, 'native_key': key, 'original_url': row.pop('url'), 'statement_text': None,
                         'source_tags': tags[key], 'native_difficulty': {1: 'Easy', 2: 'Medium', 3: 'Hard'}.get(row['difficulty_value']),
                         'source_dataset': receipt['version']+' metadata + cached title vectors', 'source_revision': refresh['model_revision'],
                         'license': None, 'collection_status': 'metadata_only; statement not loaded', 'constraints_status': 'not_loaded'}
    positions = {key: i for i, key in enumerate(keys)}
    selected = vectors[[positions[key] for key in problems]]
    result, target_count = rankings(problems, selected)
    bundle = {'problems': problems, 'rankings': result, 'methods': METHODS, 'default_key': 'leetcode:buddy-strings',
              'coverage': f'LeetCode 내부 · 등록 {len(problems):,}개 · 후보 {target_count:,}개 · 기존 제목 벡터로 재계산',
              'notice': 'LeetCode 문제끼리 비교합니다. 기존 제목 벡터의 코사인과 플랫폼 태그 겹침이며, 본문·풀이 전략 유사도는 아닙니다. 다른 방식의 점수는 직접 비교하지 마세요.',
              'empty_message': '현재 등록된 LeetCode 제목/ID에서 찾지 못했습니다.'}
    provenance = {'schema': 'leetcode-within-platform-v1', 'source_count': len(problems), 'candidate_count': target_count,
                  'dimensions': 384, 'catalog_sha256': digest(catalog_file), 'keys_sha256': digest(run / 'keys.json'),
                  'cached_vectors_sha256': digest(run / 'vectors.npy'), 'model_revision': refresh['model_revision'],
                  'vectors_verification': 'current file hash recorded; catalog/key hashes verified against prior receipt',
                  'new_embeddings_generated': False, 'statement_fields_read': False, 'human_quality_evaluation': False,
                  'methods': list(METHODS), 'top_k': 10}
    return bundle, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', type=Path, default=ROOT / 'private-learning/mastery/catalog-v4')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise ValueError('use fresh output and receipt paths')
    bundle, receipt = build(args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(bundle, handle, ensure_ascii=False, separators=(',', ':'))
    args.output.chmod(0o600)
    receipt['output_sha256'] = digest(args.output)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
