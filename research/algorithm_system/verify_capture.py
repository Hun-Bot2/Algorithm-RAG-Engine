"""Reconstruct selected Parquet columns offline from hashed captured byte ranges."""
import io,json,hashlib
from pathlib import Path
from codecontests import source_names,normalize,sha,SIDECAR_SHA256

class CapturedReader(io.RawIOBase):
    def __init__(self,size,ranges):self.size=size;self.ranges=ranges;self.pos=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,n,w=0):self.pos=n if w==0 else self.pos+n if w==1 else self.size+n;return self.pos
    def read(self,n=-1):
        if n<0:raise ValueError('whole-file read forbidden')
        n=min(n,max(0,self.size-self.pos));start=self.pos
        if not n:return b''
        for lo,b in self.ranges:
            if lo<=start and start+n<=lo+len(b):self.pos+=n;return b[start-lo:start-lo+n]
        raise ValueError('uncaptured range requested')

def verify(root):
    import pyarrow.parquet as pq
    root=Path(root);capture=root/'private-learning/algorithm-system/codecontests-v1'
    receipt=json.loads((root/'research/algorithm_system/codecontests-source.v1.json').read_text())
    side=capture/'dataset_infos.json';assert sha(side.read_bytes())==SIDECAR_SHA256
    sidecar=json.loads(side.read_text());catalog={json.loads(x)['problem_key'] for x in (root/'private-learning/full-catalog/catalog-v1/catalog.v1.jsonl').read_text().splitlines()}
    digest=hashlib.sha256();count=0
    for item in receipt['shards']:
        ranges=[]
        for r in item['ranges']:
            b=(capture/'ranges'/Path(item['path']).stem/f"{r['start']}-{r['end']}.bin").read_bytes()
            assert sha(b)==r['sha256'] and len(b)==r['bytes'];ranges.append((r['start'],b))
        pf=pq.ParquetFile(CapturedReader(item['upstream_size'],ranges),pre_buffer=False)
        names=source_names(pf.schema_arrow,sidecar);position=0
        for group in range(pf.num_row_groups):
            for row in pf.read_row_group(group,columns=item['selected_columns'],use_threads=False).to_pylist():
                r=normalize(row,'deepmind/code_contests',Path(item['path']).name.split('-')[0],position,names,catalog,Path(item['path']).name)
                if r is not None:digest.update((json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n').encode());count+=1
                position+=1
    assert digest.hexdigest()==receipt['output_sha256'] and count==receipt['statement_rows']
    return {'offline_replay_verified':True,'records':count,'output_sha256':digest.hexdigest(),'network_calls':0}
if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parents[2]),indent=2))
