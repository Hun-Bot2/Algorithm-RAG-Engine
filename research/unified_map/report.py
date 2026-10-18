"""Create an executed, aggregate-only notebook from versioned local receipts."""
import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def build():
    nb=nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('''# Unified cross-platform similarity map

Semantic title proximity, statement semantic proximity and native-tag agreement
are separate evidence. This notebook does not establish algorithm equivalence,
prerequisite direction or learning improvement. Missing labels remain unknown.
Private statements and problem-level exports are not embedded here.'''),
        nbformat.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path.cwd()
MAP=ROOT/'private-learning/unified-map/map-v2'
RED=ROOT/'private-learning/unified-map/reduction-v2'
print('Map receipt files:',[p.name for p in MAP.glob('*.json')])
for p in MAP.glob('*manifest*.json'):
    obj=json.loads(p.read_text());print(p.name,json.dumps(obj,indent=2)[:18000])
'''),
        nbformat.v4.new_code_cell('''evaluation=json.loads((MAP/'evaluation.json').read_text())
rows=[]
for cohort,report in evaluation.items():
    for k,methods in report['metrics'].items():
        for method,v in methods.items():
            rows.append(dict(cohort=cohort,k=int(k),method=method,queries=v['query_count'],judged_coverage=v['judged_coverage'],observed_precision_judged=v['observed_precision_judged'],conditional_ndcg_queries=v['ndcg_common_judged_query_count'],tag_jaccard=v['tag_consistency']['mean_primary_concept_jaccard']))
df=pd.DataFrame(rows)
display(df)
print('Tag agreement is circular for tag rankers. Precision is conditional on sparse historical AI judgments, not human gold.')
fig,ax=plt.subplots(figsize=(10,4))
sub=df[df.k==10];ax.barh(sub.cohort+': '+sub.method,sub.judged_coverage)
ax.set(xlabel='Judged fraction of retrieved slots',title='Top10 historical judgment coverage')
fig.tight_layout();plt.show()
'''),
        nbformat.v4.new_code_cell('''quality=json.loads((RED/'quality-diagnostics.json').read_text())
display(pd.DataFrame(quality['methods']).T[['queries','judged_slots','judged_coverage','observed_precision_judged','mean_tag_jaccard']])
print('Reduction receipt files:',[p.name for p in RED.glob('*.json')])
for p in RED.glob('*receipt*.json'):
    obj=json.loads(p.read_text());print(p.name,json.dumps(obj,indent=2)[:24000])
from IPython.display import Image, display
for name in ('pca2','svd2','umap2','tsne2'):
    display(Image(filename=str(RED/f'sample-{name}.png')))
print('PCA/SVD preserve neighborhoods; UMAP/t-SNE are sampled geometric diagnostics, not algorithm relevance validation.')
'''),
        nbformat.v4.new_markdown_cell('''## Unverified areas

AtCoder native tags are unavailable. LeetCode statement admission is unavailable.
Some native catalog rows are excluded or incompatible with existing English title
embeddings; their node metadata remains present. Approved identity overlays do not
validate statement fidelity. Human relevance, difficulty-ordering and learner
outcome evidence remains insufficient. No deployment or model promotion occurs.''')])
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    client=NotebookClient(nb,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
    client.execute()
    nbformat.write(nb,HERE/'comparison.ipynb')
    from nbconvert import HTMLExporter
    body,_=HTMLExporter().from_notebook_node(nb)
    (HERE/'comparison.html').write_text(body)

if __name__=='__main__':build()
