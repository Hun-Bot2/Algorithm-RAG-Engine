"""Pinned, statement-column-only CodeContests import through bounded HTTP ranges."""
import io
import json
import hashlib
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

REVISION='802411c3010cb00d1b05bad57ca77365a3c699d6'
DATASET='deepmind/code_contests'
SIDECAR_SHA256='6289cbfb168f2bde925e27bdada4cdaed65d7a4b9c060c75946504584f3cbdc0'
SOURCE_NAMES=['UNKNOWN_SOURCE','CODECHEF','CODEFORCES','HACKEREARTH','CODEJAM','ATCODER','AIZU']
COLUMNS=['name','description','source','public_tests','cf_contest_id','cf_index',
         'cf_rating','cf_tags','is_description_translated']
MAX_RANGE=32*1024*1024
MAX_TRANSFER=512*1024*1024


def sha(data):return hashlib.sha256(data).hexdigest()


class RangeReader(io.RawIOBase):
    """Reject non-range responses; never silently read a full Parquet shard."""
    def __init__(self,url,size,output,budget):
        super().__init__();self.url=url;self.size=size;self.pos=0
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.ranges=[];self.cache=[];self.budget=budget
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if pos<0:raise ValueError('negative seek')
        self.pos=pos;return pos
    def read(self,n=-1):
        if n<0:raise ValueError('whole-shard read forbidden')
        n=min(n,max(0,self.size-self.pos))
        if not n:return b''
        start=self.pos;end=start+n-1
        if n>MAX_RANGE:raise ValueError('selected range exceeds cap')
        for lo,body in self.cache:
            if lo<=start and end<lo+len(body):self.pos+=n;return body[start-lo:start-lo+n]
        wait=.15-(time.monotonic()-self.budget.get('last',0))
        if wait>0:time.sleep(wait)
        self.budget['last']=time.monotonic()
        req=Request(self.url,headers={'Range':f'bytes={start}-{end}','User-Agent':'Algorithm-RAG-Engine-statement-research/2.0','Accept-Encoding':'identity'})
        with urlopen(req,timeout=45) as response:
            if response.status!=206:raise ValueError('server ignored Range; refusing full archive')
            cr=response.headers.get('Content-Range','')
            if cr!=f'bytes {start}-{end}/{self.size}':raise ValueError('unexpected Content-Range '+cr)
            body=response.read(n+1)
        if len(body)!=n:raise ValueError('range length mismatch')
        self.budget['bytes']=self.budget.get('bytes',0)+len(body)
        if self.budget['bytes']>MAX_TRANSFER:raise ValueError('run transfer cap exceeded')
        path=self.output/f'{start}-{end}.bin'
        if path.exists():assert path.read_bytes()==body
        else:path.write_bytes(body);path.chmod(0o600)
        self.ranges.append({'start':start,'end':end,'sha256':sha(body),'bytes':len(body)})
        self.cache.append((start,body));self.pos+=n;return body


def source_names(schema, sidecar=None):
    metadata=json.loads((schema.metadata or {}).get(b'huggingface',b'{}'))
    feature=metadata.get('info',{}).get('features',{}).get('source',{})
    names=feature.get('names')
    if sidecar is not None:
        sf=sidecar['deepmind--code_contests']['features']['source']
        if sf.get('num_classes')!=7 or sf.get('names')!=SOURCE_NAMES:raise ValueError('pinned source feature differs')
        if names is not None and names!=sf['names']:raise ValueError('source encodings disagree')
        names=sf['names']
    if names != SOURCE_NAMES:
        raise ValueError('source ClassLabel metadata missing; do not guess enum')
    return names


def normalize(record,dataset_id,split,position,names,catalog,shard="fixture"):
    source=record.get('source')
    if not isinstance(source,int) or not 0<=source<len(names):raise ValueError('invalid source enum')
    platform={'CODEFORCES':'codeforces','ATCODER':'atcoder'}.get(names[source])
    if platform is None:return None
    description=record.get('description')
    if not isinstance(description,str) or not description.strip():return None
    native=None;url=None;basis='unresolved_dataset_identity'
    if platform=='codeforces':
        contest,index=record.get('cf_contest_id'),record.get('cf_index')
        if isinstance(contest,int) and contest>0 and isinstance(index,str) and re.fullmatch(r'[A-Za-z0-9]+',index):
            native=f'codeforces:{contest}{index}';url=f'https://codeforces.com/problemset/problem/{contest}/{index}';basis='publisher_native_contest_id_and_index'
    # Some descriptions explicitly contain the original task URL; names alone never join.
    if platform=='atcoder':
        found=set(re.findall(r'https?://atcoder\.jp/contests/[A-Za-z0-9_-]+/tasks/([A-Za-z0-9_-]+)',description))
        if len(found)==1:
            task=next(iter(found));native='atcoder:'+task
            match=re.search(r'https?://atcoder\.jp/contests/[A-Za-z0-9_-]+/tasks/'+re.escape(task),description)
            url=match.group();basis='explicit_native_url_in_description'
    dataset_key=f'codecontests:{REVISION}:{split}:{shard}:{position}'
    return {'problem_key':native or dataset_key,'native_key':native,'catalog_match':native in catalog if native else False,
            'dataset_key':dataset_key,'platform':platform,'title':record.get('name'),'statement_text':description.strip(),
            'raw_content_sha256':sha(description.encode()),'content_sha256':sha(description.strip().encode()),
            'language':'en' if not re.search('[\u3040-\u30ff\u4e00-\u9fff\u0400-\u04ff]',description) else 'mixed_or_non_latin',
            'language_basis':'publisher card English plus script heuristic, not human verification',
            'public_samples':record.get('public_tests'),'constraints_status':'embedded_not_separately_parsed',
            'source_tags':record.get('cf_tags') if platform=='codeforces' else None,
            'source_difficulty':record.get('cf_rating') if platform=='codeforces' else None,
            'is_description_translated':record.get('is_description_translated'),'original_url':url,'identity_basis':basis,
            'source_dataset':DATASET,'source_revision':REVISION,'source_split':split,'source_row':position,'source_shard':shard,
            'license':'CC-BY-4.0 non-code publisher grant; preserve separate third-party conditions',
            'collection_status':'publisher_statement_available'}


def run(root,output,receipt,*,limit_files=None):
    import pyarrow.parquet as pq
    root,output,receipt=Path(root),Path(output),Path(receipt)
    metadata=json.loads((output/'metadata.json').read_text())
    assert metadata['sha']==REVISION and metadata['id']==DATASET
    target=output/'statements.jsonl'
    if target.exists() or receipt.exists():raise ValueError('immutable run exists')
    cat={json.loads(line)['problem_key'] for line in (root/'private-learning/full-catalog/catalog-v1/catalog.v1.jsonl').read_text().splitlines()}
    files=sorted([x for x in metadata['siblings'] if x['rfilename'].endswith('.parquet')],key=lambda x:x['rfilename'])
    sidecar_bytes=(output/'dataset_infos.json').read_bytes()
    if sha(sidecar_bytes)!=SIDECAR_SHA256:raise ValueError('pinned schema sidecar hash mismatch')
    sidecar=json.loads(sidecar_bytes)
    if limit_files is not None:files=files[:limit_files]
    budget={'bytes':0};captures=[];count=0;started=time.perf_counter()
    with target.open('x') as stream:
        for item in files:
            name=item['rfilename'];split=Path(name).name.split('-')[0]
            rr=RangeReader(f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}',item['size'],output/'ranges'/Path(name).stem,budget)
            pf=pq.ParquetFile(rr,pre_buffer=False)
            names=source_names(pf.schema_arrow,sidecar)
            cols=[c for c in COLUMNS if c in pf.schema_arrow.names]
            if not {'description','name','source','cf_contest_id','cf_index'}.issubset(cols):raise ValueError('required schema missing')
            position=0;observed={};written=0
            for group in range(pf.num_row_groups):
                for record in pf.read_row_group(group,columns=cols,use_threads=False).to_pylist():
                    value=record['source']
                    if type(value) is not int or not 0<=value<len(names):raise ValueError('invalid source enum')
                    source=names[value];observed[source]=observed.get(source,0)+1
                    normalized=normalize(record,DATASET,split,position,names,cat,Path(name).name)
                    if normalized is not None:
                        stream.write(json.dumps(normalized,ensure_ascii=False,sort_keys=True)+'\n');count+=1;written+=1
                    position+=1
            captures.append({'path':name,'upstream_size':item['size'],'upstream_lfs_sha256_not_fully_verified':item.get('lfs',{}).get('sha256'),
                             'row_count':position,'source_names':names,'observed_sources':observed,'selected_columns':cols,'statements_written':written,'ranges':rr.ranges})
            print(f'{name}: rows={position} statements={written} range_bytes={sum(r["bytes"] for r in rr.ranges)}',flush=True)
    target.chmod(0o600)
    result={'schema_version':1,'dataset':DATASET,'revision':REVISION,'statement_rows':count,'shards':captures,
            'transferred_bytes':budget['bytes'],'whole_shards_downloaded':False,'solution_private_generated_test_columns_read':False,
            'schema_sidecar_sha256':sha((output/'dataset_infos.json').read_bytes()),'metadata_sha256':sha((output/'metadata.json').read_bytes()),'columns':COLUMNS,'output_sha256':sha(target.read_bytes()),'seconds':time.perf_counter()-started,
            'license_source':'https://github.com/google-deepmind/code_contests/blob/main/README.md#license',
            'license_scope':'Publisher non-code CC-BY-4.0 grant with attribution and separate third-party conditions; no direct platform permission inferred'}
    receipt.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--limit-files',type=int)
    a=p.parse_args();run(Path(__file__).resolve().parents[2],a.output,a.receipt,limit_files=a.limit_files)
