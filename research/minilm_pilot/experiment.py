"""Pinned local-only MiniLM pilot on equal English annotations, not statements."""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import time
import warnings
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from src.learning.embedding_backend import LocalEmbeddingBackend
from src.learning.research import build_profiles, TfidfProvider, reviewed_judgments, score_judgments

REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DenseProvider:
    def __init__(self, keys, matrix, name='minilm'):
        self.name, self.matrix = name, matrix
        self.positions = {key: index for index, key in enumerate(keys)}

    def score(self, source, target, index=None):
        return float(self.matrix[self.positions[source], self.positions[target]])


def ranks(matrix, keys, own, *, cross_platform=False, limit=5):
    eligible = [i for i, key in enumerate(keys) if i != own and
                (not cross_platform or key.split(':')[0] != keys[own].split(':')[0])]
    return sorted(eligible, key=lambda i: (-float(matrix[own, i]), keys[i]))[:limit]


def _run(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = ['curriculum/seed.v1.json', 'curriculum/annotations.v1.json',
              'research/relationship-judgments.v1.json', 'assets/problem_identity_aliases.v1.json',
              'assets/cross_platform_sources.v1.json']
    before = {name: digest(root/name) for name in inputs}
    curriculum = json.loads((root/inputs[0]).read_text())
    annotations = json.loads((root/inputs[1]).read_text())
    profiles = {key: replace(value, concepts=()) for key, value in build_profiles(curriculum, annotations).items()}
    keys = sorted(profiles)
    assert len(keys) == 90 and all(profiles[k].title and profiles[k].concept_rationale for k in keys)
    model_root = root/'private-learning/models'/('minilm-'+REVISION)
    manifest_path = root/'research/minilm_pilot/model.manifest.json'
    manifest = json.loads(manifest_path.read_text())
    started = time.perf_counter()
    backend = LocalEmbeddingBackend(model_root, manifest)
    import torch
    import umap
    torch.set_num_threads(1)
    load_seconds = time.perf_counter()-started
    results = {'executed_at': datetime.now(timezone.utc).isoformat(), 'diagnostic_only': True,
               'model': manifest, 'inputs_sha256': before, 'cohort_counts': dict(Counter(k.split(':')[0] for k in keys)),
               'scope': '90 reviewed English annotations, not 21,244 downloaded metadata records or full statements',
               'model_load_seconds': load_seconds,
               'load_timing_scope': 'LocalEmbeddingBackend construction plus first Torch/SentenceTransformers/UMAP imports; not model-file I/O in isolation',
               'variants': {}, 'ordered_keys': keys,
               'environment': {name: importlib.metadata.version(name) for name in
                               ['torch','sentence-transformers','transformers','numpy','scikit-learn','umap-learn']},
               'platform': platform.platform(), 'threads': 1, 'training': False, 'promotion': False}
    matrices = {}
    vectors_by_variant = {}
    judgments, admission = reviewed_judgments(json.loads((root/inputs[2]).read_text()), purpose='evaluation')
    judgments = [row for row in judgments if row['type'] == 'similarity']
    for variant in ['title_rationale', 'title_only']:
        current = profiles if variant == 'title_rationale' else {k: replace(p, concept_rationale='') for k,p in profiles.items()}
        texts = [current[k].text for k in keys]
        lengths = [len(backend.encoder.tokenizer(text, truncation=False)['input_ids']) for text in texts]
        started = time.perf_counter()
        vectors = np.asarray(backend.encode(texts, role='passage'), dtype=np.float32)
        encode_seconds = time.perf_counter()-started
        assert vectors.shape == (90,384) and np.isfinite(vectors).all()
        matrix = vectors @ vectors.T
        assert np.isfinite(matrix).all()
        reference=np.einsum('ik,jk->ij',vectors.astype(np.float64),vectors.astype(np.float64),optimize=False)
        dot_error=float(np.max(np.abs(matrix-reference)))
        assert dot_error < 1e-5
        matrices[variant], vectors_by_variant[variant] = matrix, vectors
        providers = [TfidfProvider(current), DenseProvider(keys, matrix)]
        scoring = score_judgments(judgments, current, providers,
                                  curriculum_relationships=curriculum['relationships'])
        results['variants'][variant] = {
            'fields': ['title','concept_rationale'] if variant == 'title_rationale' else ['title'],
            'text_sha256': hashlib.sha256(json.dumps(dict(zip(keys,texts)),sort_keys=True).encode()).hexdigest(),
            'encode_seconds': encode_seconds, 'vector_bytes': vectors.nbytes,
            'token_lengths': {'min': min(lengths), 'max': max(lengths), 'median': float(np.median(lengths)),
                              'over_256': sum(length > 256 for length in lengths)},
            'non_blas_float64_cosine_max_error': dot_error,
            'norm_range': [float(np.linalg.norm(vectors,axis=1).min()),float(np.linalg.norm(vectors,axis=1).max())],
            'scoring': scoring,
        }
    matrix = matrices['title_rationale']
    examples = []
    # Predetermined queries include each platform; no selection by favorable score.
    query_keys = ['leetcode:two-sum','leetcode:valid-parentheses','leetcode:coin-change',
                  'atcoder:abc086_a','atcoder:abc086_c','atcoder:abc085_c',
                  'codeforces:4A','codeforces:112A','codeforces:339A']
    for key in query_keys:
        own=keys.index(key)
        examples.append({'source':key,'title':profiles[key].title,
                         'neighbors':[{'problem_key':keys[i],'title':profiles[keys[i]].title,'cosine':float(matrix[own,i])}
                                      for i in ranks(matrix,keys,own,cross_platform=True,limit=3)]})
    results['cross_platform_examples'] = examples
    results['example_boundary'] = 'Fixed queries, model proposals only; cosine is not relevance probability or approved relationship.'
    results['top5_cross_platform_fraction'] = float(np.mean([
        np.mean([keys[j].split(':')[0] != key.split(':')[0] for j in ranks(matrix,keys,i)]) for i,key in enumerate(keys)]))
    results['title_ablation_top5_overlap'] = float(np.mean([
        len(set(ranks(matrix,keys,i)) & set(ranks(matrices['title_only'],keys,i)))/5 for i in range(len(keys))]))
    # Full catalog geometry, small cohort only; never compare with legacy 1536-D space.
    fig, ax = plt.subplots(figsize=(8,7),constrained_layout=True)
    image=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm',interpolation='nearest')
    counts=Counter(k.split(':')[0] for k in keys); positions=[]; offset=0
    for name,count in sorted(counts.items()):
        positions.append((offset+(count-1)/2,f'{name} (n={count})'));offset+=count
        if offset < len(keys): ax.axhline(offset-.5,color='black',linewidth=.8);ax.axvline(offset-.5,color='black',linewidth=.8)
    ax.set_xticks([x for x,_ in positions],[s for _,s in positions],rotation=20)
    ax.set_yticks([x for x,_ in positions],[s for _,s in positions])
    ax.set_title('MiniLM cosine similarities: 90 English title + rationale inputs')
    fig.colorbar(image,ax=ax,label='Cosine similarity (not a relevance probability)')
    fig.savefig(out/'similarity.png',dpi=160);plt.close(fig)
    x=vectors_by_variant['title_rationale']
    projections={'PCA':PCA(n_components=2,random_state=17).fit_transform(x),
                 't-SNE':TSNE(n_components=2,perplexity=15,max_iter=750,init='random',random_state=17,learning_rate='auto').fit_transform(x),
                 'UMAP':umap.UMAP(n_neighbors=10,min_dist=.1,random_state=17,n_jobs=1,metric='cosine').fit_transform(x)}
    results['projection_parameters']={'seed':17,'tsne_perplexity':15,'tsne_iterations':750,'tsne_init':'random','umap_neighbors':10,'umap_min_dist':.1,'umap_metric':'cosine'}
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),constrained_layout=True)
    styles={'leetcode':('#2166ac','o'),'atcoder':('#b35806','^'),'codeforces':('#6a51a3','s')}
    for ax,(name,coords) in zip(axes,projections.items()):
        assert np.isfinite(coords).all()
        for platform_name,(color,marker) in styles.items():
            idx=[i for i,k in enumerate(keys) if k.startswith(platform_name+':')]
            ax.scatter(coords[idx,0],coords[idx,1],label=f'{platform_name} ({len(idx)})',color=color,marker=marker,alpha=.8,s=32)
        ax.set_title(name);ax.set_xlabel('Projection coordinate 1');ax.set_ylabel('Projection coordinate 2');ax.set_xticks([]);ax.set_yticks([])
    axes[0].legend(fontsize=9)
    fig.suptitle('Exploratory layouts only: separation does not prove algorithmic similarity')
    fig.savefig(out/'projections.png',dpi=160);plt.close(fig)
    private_output=root/'private-learning/minilm-pilot-v1';private_output.mkdir(parents=True,exist_ok=True,mode=0o700)
    np.save(private_output/'vectors.npy',x)
    (private_output/'ordered_keys.json').write_text(json.dumps(keys,indent=2)+'\n')
    os.chmod(private_output,0o700)
    os.chmod(private_output/'vectors.npy',0o600)
    os.chmod(private_output/'ordered_keys.json',0o600)
    results['vectors_sha256']=digest(private_output/'vectors.npy')
    after={name:digest(root/name) for name in inputs};assert before==after
    registry=json.loads((root/'assets/source_registry.v1.json').read_text())
    for entry in registry['assets']:
        if entry['status']=='present': assert digest(root/entry['path'])==entry['sha256']
    results['input_hashes_unchanged']=True
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    results['process_peak_rss_bytes']=peak if platform.system()=='Darwin' else peak*1024
    results['memory_boundary']='Whole notebook kernel peak including imports, model and plotting; not incremental model memory or OCI proof.'
    results['figures']=['similarity.png','projections.png']
    return results


def run(root,out):
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1), \
            patch('socket.socket.connect', side_effect=RuntimeError('Python network disabled during offline experiment')), \
            patch('socket.create_connection', side_effect=RuntimeError('Python network disabled during offline experiment')):
        warnings.simplefilter('always')
        result=_run(root,out)
    result['offline_boundary'] = 'HF/Transformers offline mode and Python socket connect guards during load/inference; not an OS-level egress firewall test'
    result['warnings']=[{'category':category,'message':message,'count':count} for (category,message),count in
                        Counter((type(w.message).__name__,str(w.message)) for w in caught).items()]
    (Path(out)/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
