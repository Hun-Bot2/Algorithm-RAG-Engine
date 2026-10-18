"""Build the source-backed companion; execute with nbconvert after all run receipts exist."""
from pathlib import Path
import nbformat
HERE=Path(__file__).resolve().parent
cells=[nbformat.v4.new_markdown_cell('''# Statement-first algorithm similarity study

This notebook reads immutable local receipts. The CF+AtCoder publisher-description
cohort is separate from the 24,297-row native metadata catalog. LeetCode text is
quarantined; AtCoder dataset rows without native IDs are not catalog joins.
Existing 3,647 judgments are AI-reviewed, not human gold. Unknown relevance is
never a negative. Compression neighbor preservation is not educational utility.
'''),nbformat.v4.new_code_cell('''from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path.cwd()
if ROOT.name == 'algorithm_system': ROOT=ROOT.parents[1]
assert (ROOT/'research/algorithm_system').is_dir()
R=ROOT/'research/algorithm_system'
RUN=ROOT/'private-learning/algorithm-system/statement-run-v1'
EVAL=ROOT/'private-learning/algorithm-system/statement-evaluation-v1'
COMP=ROOT/'private-learning/algorithm-system/compression-v1'
FIG=R/'figures'; FIG.mkdir(exist_ok=True)
manifest=json.loads((RUN/'manifest.json').read_text())
assert hashlib.sha256((RUN/'corpus.jsonl').read_bytes()).hexdigest()==manifest['corpus_sha256']
print('Native metadata catalog: 24,297 problems')
print('Publisher statement cohort:',manifest['count'],manifest['platforms'])
print('Native identity available:',manifest['native_identity_available'])
print('Native catalog matches:',manifest['catalog_matches'])
print('No model promotion:',manifest['no_model_promotion'])
'''),nbformat.v4.new_code_cell('''corpus=[json.loads(x) for x in (RUN/'corpus.jsonl').read_text().splitlines()]
lengths=json.loads((RUN/'token-lengths.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4))
for platform in sorted({r['platform'] for r in corpus}):
    values=sorted(lengths[i] for i,r in enumerate(corpus) if r['platform']==platform)
    axes[0].step(values,np.arange(1,len(values)+1)/len(values),where='post',label=f'{platform} (n={len(values):,})')
axes[0].axvline(254,color='black',linestyle='--',label='single-window content budget')
axes[0].set(xscale='log',xlabel='Publisher description tokens (log scale)',ylabel='Empirical CDF'); axes[0].legend(fontsize=8)
names=[k for k,v in manifest['methods'].items() if 'status' not in v]
secs=[manifest['methods'][k].get('encode_seconds',0)+manifest['methods'][k].get('seconds',0)+manifest['methods'][k].get('retrieval_seconds',0) for k in names]
axes[1].barh(names,secs);axes[1].set(xlabel='Measured elapsed seconds',title='Local CPU run; different method stages included')
fig.tight_layout();fig.savefig(FIG/'statement-baseline.png',dpi=160)
plt.show()
pd.DataFrame([{'method':k,**v} for k,v in manifest['methods'].items()]).fillna('unavailable')
'''),nbformat.v4.new_code_cell('''print('Evaluation files:',sorted(p.name for p in EVAL.glob('*')))
for p in sorted(EVAL.glob('*.json')):
    obj=json.loads(p.read_text())
    print(p.name, json.dumps(obj,ensure_ascii=False)[:5000])
print('Interpretation: cross-platform native-ID overlap may have zero usable judgments.')
print('Do not interpret observed coverage, tag agreement, or neighbor overlap as learning gains.')
'''),nbformat.v4.new_code_cell('''print('Compression receipts:',sorted(p.name for p in COMP.glob('*.json')))
for p in sorted(COMP.glob('*.json')):
    obj=json.loads(p.read_text())
    if isinstance(obj,dict): print(p.name,json.dumps(obj,ensure_ascii=False)[:12000])
print('Original 384D embeddings remain the reference; no automatic promotion.')
'''),nbformat.v4.new_markdown_cell('''## Reproducibility and limits

See `CHECKPOINT1.md`, `CHECKPOINT2.md`, `REPORT.md`, source receipts, model manifest,
and private run manifests for hashes and source conditions. Measured runtime and
peak RSS describe this local machine, not OCI feasibility. Native difficulty
percentiles summarize different populations and are provisional across platforms.
No supervised difficulty calibration or relevance model was fitted. Human-review
queues are unlabelled deliverables requiring future adjudication. New statements
are kept private; this notebook exports aggregate figures, not statement mirrors.
''')]
cells.insert(-1,nbformat.v4.new_code_cell("""compression=json.loads((COMP/'receipt.json').read_text())
records=[]
for name,m in compression['methods'].items():
    if m.get('status')=='complete':
        records.append({'method':name,'dimensions':m['dimensions'],'Top10 overlap':m['neighbor_overlap']['top10']['mean_fraction'],'vector MiB':m['array_bytes']/2**20,'retrieval seconds':m['retrieval_seconds']})
table=pd.DataFrame(records)
display(table)
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].barh(table['method'],table['Top10 overlap']);axes[0].set(xlim=(0,1),xlabel='Mean Top10 overlap with original 384D')
axes[1].scatter(table['vector MiB'],table['Top10 overlap'])
for _,r in table.iterrows():axes[1].annotate(r['method'],(r['vector MiB'],r['Top10 overlap']),fontsize=8)
axes[1].set(xlabel='Vector array MiB (transform storage additional)',ylabel='Top10 neighbor overlap')
fig.suptitle('Compression preserves neighbors, not proven relevance')
fig.tight_layout();fig.savefig(FIG/'compression-ablation.png',dpi=160);plt.show()
"""))
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
nbformat.write(nb,HERE/'comparison.ipynb')
