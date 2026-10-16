"""Explicit offline-only compact encoders. Never imported by the hosted core."""
import math
from pathlib import Path
import re

from .research_contracts import ResearchError, json_bytes, sha256_bytes, sha256_file
from .research_profiles import catalog_hash

MODELS = {'sentence-transformers/all-MiniLM-L6-v2': (384, 256),
          'intfloat/multilingual-e5-small': (384, 512)}


class LocalEmbeddingBackend:
    def __init__(self, directory, manifest, *, loader=None):
        root = Path(directory).resolve()
        if not root.is_dir():
            raise ResearchError('local model directory unavailable; use metadata fallback')
        required={'schema_version','namespace','model_id','revision','license_reference','dimensions','max_tokens','files_sha256'}
        if not isinstance(manifest,dict) or set(manifest)!=required or manifest['schema_version']!=1:
            raise ResearchError('invalid local encoder manifest')
        model=manifest['model_id']
        if model not in MODELS or not re.fullmatch(r'[0-9a-f]{40}',manifest['revision']):
            raise ResearchError('supported model and immutable revision required')
        dimensions,max_tokens=MODELS[model]
        if manifest['dimensions']!=dimensions or type(manifest['max_tokens']) is not int or not 1<=manifest['max_tokens']<=max_tokens:
            raise ResearchError('incompatible dimensions or token limit')
        for field in ['namespace','license_reference']:
            if not isinstance(manifest[field],str) or not manifest[field].strip():
                raise ResearchError('encoder provenance missing')
        hashes=manifest['files_sha256']
        if not isinstance(hashes,dict) or not hashes or not any(name.endswith('.safetensors') for name in hashes):
            raise ResearchError('reviewed safetensors file manifest required')
        actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
        if actual!=set(hashes):
            raise ResearchError('model directory differs from reviewed manifest')
        for name,digest in hashes.items():
            target=(root/name).resolve()
            if not target.is_relative_to(root) or not target.is_file() or sha256_file(target)!=digest:
                raise ResearchError('model file hash/path mismatch')
        if loader is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ResearchError('optional Sentence Transformers dependency unavailable; use metadata fallback') from exc
            loader=SentenceTransformer
        self.encoder=loader(str(root),device='cpu',local_files_only=True,trust_remote_code=False,
                            revision=manifest['revision'],model_kwargs={'use_safetensors':True})
        self.encoder.max_seq_length=manifest['max_tokens']
        self.manifest=dict(manifest)

    def encode(self,texts,*,role):
        if role not in {'query','passage'} or not isinstance(texts,list) or any(not isinstance(t,str) for t in texts):
            raise ResearchError('explicit query/passage text role required')
        prefix=(role+': ') if self.manifest['model_id'].startswith('intfloat/') else ''
        vectors=self.encoder.encode([prefix+t for t in texts],normalize_embeddings=True,
                                    batch_size=16,show_progress_bar=False,convert_to_numpy=True,prompt='')
        result=[list(map(float,row)) for row in vectors]
        if len(result)!=len(texts) or any(len(row)!=self.manifest['dimensions'] or any(not math.isfinite(x) for x in row)
                                        or abs(math.fsum(x*x for x in row)-1)>0.001 for row in result):
            raise ResearchError('encoder returned invalid/non-normalized vectors')
        return result

    def neighbors(self,profiles,*,limit=10):
        if type(limit) is not int or not 1<=limit<=100:
            raise ResearchError('bounded neighbor count required')
        keys=sorted(profiles)
        if len(keys)>10000:
            raise ResearchError('offline pilot corpus exceeds bounded scan size')
        texts=[profiles[key].text for key in keys]
        queries=self.encode(texts,role='query')
        passages=self.encode(texts,role='passage')
        neighbors={}
        for i,source in enumerate(keys):
            scored=[(math.fsum(a*b for a,b in zip(queries[i],row,strict=True)),target)
                    for target,row in zip(keys,passages,strict=True) if source!=target]
            scored.sort(key=lambda item:(-item[0],item[1]))
            neighbors[source]=[{'problem_key':target,'score':score} for score,target in scored[:limit]]
        manifest=self.manifest
        return {'manifest':{'schema_version':1,'namespace':manifest['namespace'],'revision':manifest['revision'],
            'model_id':manifest['model_id'],'license_reference':manifest['license_reference'],'dimensions':manifest['dimensions'],
            'normalization':'l2','metric':'cosine','preprocessing':'authored-profile-v1; explicit query/passage prefixes for E5',
            'preprocessing_sha256':sha256_bytes(json_bytes({'model':manifest['model_id'],'max_tokens':manifest['max_tokens'],'fields':['title','concept_rationale','concepts']})),
            'model_files_sha256':manifest['files_sha256'],'input_sha256':catalog_hash(profiles),
            'ordered_keys':keys,'ordered_keys_sha256':sha256_bytes(json_bytes(keys))},'neighbors':neighbors}
