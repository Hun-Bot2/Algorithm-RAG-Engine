"""Compare immutable, matched retrieval outputs without reading statement text.

python3 -m research.recommendation_inputs.compare --run PATH --output NEW_JSON
"""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('minilm_title', 'minilm_title_statement', 'minilm_chunk', 'tfidf_word', 'tfidf_char')


def digest(path):
    h = sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'hash mismatch: {path}')
    return actual


def load_rankings(path, keys, k=10):
    rankings = {}
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            query = row.get('query')
            if query not in keys or query in rankings:
                raise ValueError('unknown or duplicate query')
            pairs = row.get('results')
            if not isinstance(pairs, list) or len(pairs) != k:
                raise ValueError('incomplete ranking')
            targets, scores = [], []
            for pair in pairs:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError('invalid result pair')
                target, score = pair
                if target not in keys or target == query or target in targets:
                    raise ValueError('unknown, self or duplicate target')
                if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
                    raise ValueError('invalid score')
                targets.append(target)
                scores.append(score)
            if scores != sorted(scores, reverse=True):
                raise ValueError('ranking scores are not descending')
            rankings[query] = targets
    if set(rankings) != keys:
        raise ValueError('query cohort mismatch')
    return rankings


def compare_rankings(runs, k=10):
    if not runs:
        raise ValueError('no runs')
    queries = set(next(iter(runs.values())))
    if not queries or any(set(run) != queries for run in runs.values()):
        raise ValueError('query cohort mismatch')
    if any(len(targets) != k or len(set(targets)) != k for run in runs.values() for targets in run.values()):
        raise ValueError('invalid ranking length or duplicates')
    query_order = sorted(queries)
    overlaps = []
    for a, b in combinations(runs, 2):
        shared = [len(set(runs[a][q]) & set(runs[b][q])) for q in query_order]
        overlaps.append({'a': a, 'b': b, 'queries': len(queries),
                         'mean_shared_top10': sum(shared) / len(shared),
                         'top10_overlap_fraction': sum(shared) / (len(shared) * k),
                         'top1_agreement_fraction': sum(runs[a][q][0] == runs[b][q][0] for q in query_order) / len(queries),
                         'zero_overlap_queries': shared.count(0)})
    concentration = {}
    for method, run in runs.items():
        counts = Counter(target for targets in run.values() for target in targets)
        concentration[method] = {'distinct_targets': len(counts), 'max_indegree': max(counts.values()),
                                 'largest_candidate_query_share': max(counts.values()) / len(queries)}
    return {'queries': len(queries), 'k': k, 'pairwise': overlaps, 'concentration': concentration}


def analyze(run, baseline_path, evaluation_path):
    run = Path(run)
    baseline = json.loads(Path(baseline_path).read_text())
    if json.loads((run / 'manifest.json').read_text()) != baseline:
        raise ValueError('private manifest differs from committed baseline')
    evaluation = json.loads(Path(evaluation_path).read_text())
    corpus_hash = verify(run / 'corpus.jsonl', baseline['corpus_sha256'])
    if evaluation['inputs']['corpus']['sha256'] != corpus_hash:
        raise ValueError('evaluation corpus differs')
    label_source = evaluation['inputs']['labels']
    label_path = Path(label_source['path'])
    verify(label_path if label_path.is_absolute() else ROOT / label_path, label_source['sha256'])
    key_list = json.loads((run / 'keys.json').read_text())
    keys = set(key_list)
    if len(keys) != len(key_list) or len(keys) != baseline['count']:
        raise ValueError('invalid catalog identity count')
    runs, hashes, methods = {}, {}, {}
    for method in METHODS:
        receipt = baseline['methods'][method]
        top10_hash = verify(run / method / 'top10.jsonl', receipt['outputs']['top10.jsonl'])
        top50_hash = verify(run / method / 'top50.jsonl', receipt['outputs']['top50.jsonl'])
        if evaluation['inputs']['runs'][method]['sha256'] != top50_hash:
            raise ValueError('evaluation rankings differ')
        runs[method] = load_rankings(run / method / 'top10.jsonl', keys)
        metric = evaluation['metrics']['10'][method]
        if metric['query_count'] != len(keys):
            raise ValueError('evaluation query count differs')
        hashes[method] = {'top10': top10_hash, 'top50': top50_hash}
        methods[method] = {
            'historical_encode_seconds': receipt.get('encode_seconds'),
            'historical_retrieval_seconds': receipt.get('retrieval_seconds'),
            'historical_fit_and_retrieval_seconds': receipt.get('seconds'),
            'judged_slots': metric['judged_slots'], 'judged_coverage': metric['judged_coverage'],
            'observed_precision_judged': metric['observed_precision_judged'],
            'quality_status': 'unavailable_without_applicable_judgments' if not metric['judged_slots'] else 'conditional_on_judged_slots',
        }
    diagnostics = compare_rankings(runs)
    # Independent recomputation must agree with the frozen evaluator's concentration fields.
    for method, values in diagnostics['concentration'].items():
        old = evaluation['metrics']['10'][method]['hubness']
        if any(values[k] != old[k] for k in ('distinct_targets', 'max_indegree')):
            raise ValueError('candidate concentration differs from frozen evaluation')
    return {'schema': 'recommendation-input-comparison-v1', 'count': len(keys),
            'platforms': baseline['platforms'], 'leetcodes_permitted_statements': baseline['leetcodes_permitted_statements'],
            'provenance': {'baseline_sha256': digest(baseline_path), 'evaluation_sha256': digest(evaluation_path),
                           'corpus_sha256': corpus_hash, 'keys_sha256': digest(run / 'keys.json'),
                           'model_manifest_sha256': baseline['model_manifest_sha256'], 'rankings': hashes},
            'methods': methods, 'diagnostics': diagnostics,
            'interpretation': 'Ranking differences and concentration describe input sensitivity, not accuracy or learning usefulness.',
            'new_embeddings_generated': False, 'ranking_policy_changed': False,
            'missing_variants': ['source-grounded solving-idea text', 'learner solution code', 'matched concept/semantic fusion', 'learner-conditioned remediation']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output exists; use a new immutable version')
    result = analyze(args.run, ROOT / 'research/algorithm_system/statement-baseline.v1.json',
                     ROOT / 'research/algorithm_system/evaluation.v1.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'queries': result['count'], 'methods': len(result['methods']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
