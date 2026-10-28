"""2D maps of stored experiment vectors for the read-only viewer.

The map is a layout aid only. Neighbors and scores always come from the stored 384D rankings,
and a vector set is mapped only when it reproduces those rankings exactly.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from threading import Lock
import numpy as np

from .experiments import retrieve, unit

K = 10
_cache, _lock = {}, Lock()


def attach_vectors(scopes, keys, vectors):
    """Give every scope whose problems are a subset of `keys` the matching vector rows (by method name)."""
    row = {key: i for i, key in enumerate(keys)}
    for scope in scopes.values():
        problems = list(scope['problems'])
        if not all(key in row for key in problems):
            continue
        index = [row[key] for key in problems]
        scope['vectors'] = {name: v[index] for name, v in vectors.items() if name in scope.get('rankings', {})}


def verified(scope, method):
    """True only if recomputing top-K from the vectors gives the stored ranking for every query."""
    vectors = scope.get('vectors', {}).get(method)
    stored = scope.get('rankings', {}).get(method)
    if vectors is None or stored is None:
        return False
    k = max((len(rows) for rows in stored.values()), default=0)
    recomputed = retrieve(scope['problems'], vectors, k=k)
    return all([key for key, _ in recomputed[q]] == [key for key, _ in rows] for q, rows in stored.items())


def layout(vectors, seed=0):
    from sklearn.manifold import TSNE
    v = unit(vectors)
    perplexity = max(2.0, min(15.0, (len(v) - 1) / 3))
    with np.errstate(all='ignore'):
        xy = TSNE(n_components=2, metric='cosine', init='pca', perplexity=perplexity, random_state=seed, max_iter=1000).fit_transform(v)
    xy = xy - xy.min(axis=0)
    return xy / max(float(xy.max()), 1e-9)


def preservation(xy, keys, stored, k=K):
    """Mean share of each problem's stored top-k that are also among its k nearest points on the map."""
    d = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    shares = []
    for i, key in enumerate(keys):
        truth = [t for t, _ in stored[key][:k]]
        if not truth:
            continue
        near = {keys[j] for j in np.argsort(d[i], kind='stable')[:len(truth)]}
        shares.append(len(near & set(truth)) / len(truth))
    return float(np.mean(shares)) if shares else None


def scope_map(name, scope, method, platform=None):
    """Layout plus the top-K neighbors drawn on it. With `platform`, both are recomputed within that platform."""
    with _lock:
        if (name, method, platform) in _cache:
            return _cache[(name, method, platform)]
        problems = scope['problems']
        keys = [k for k in problems if platform is None or problems[k]['platform'] == platform]
        if not verified(scope, method):
            result = {'available': False, 'method': method, 'platform': platform,
                      'reason': '이 방식은 저장된 벡터가 없거나 저장된 순위와 일치하지 않아 지도를 그리지 않아요.'}
        elif len(keys) < 5:
            result = {'available': False, 'method': method, 'platform': platform, 'reason': '이 플랫폼의 문제가 너무 적어 지도를 그리지 않아요.'}
        else:
            row = {k: i for i, k in enumerate(problems)}
            vectors = scope['vectors'][method][[row[k] for k in keys]]
            if platform is None:
                neighbors = scope['rankings'][method]
            else:
                # Same 384D cosine as the stored ranking, restricted to one platform (the vectors were verified above).
                with np.errstate(all='ignore'):
                    neighbors = retrieve({k: problems[k] for k in keys}, vectors, k=K)
            xy = layout(vectors)
            result = {'available': True, 'method': method, 'platform': platform, 'algorithm': 't-SNE (코사인, PCA 초기화, 고정 시드)', 'k': K,
                      'neighbor_preservation': preservation(xy, keys, neighbors),
                      'neighbors': {k: [t for t, _ in neighbors[k][:K]] for k in keys},
                      'points': [{'problem_key': key, 'title': problems[key]['title'], 'platform': problems[key]['platform'],
                                  'x': round(float(x), 5), 'y': round(float(y), 5)} for key, (x, y) in zip(keys, xy)]}
        _cache[(name, method, platform)] = result
        return result
