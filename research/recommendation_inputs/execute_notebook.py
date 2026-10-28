"""Execute the saved notebook with the project venv and export self-contained HTML.

Run from the repository: .venv/bin/python -m research.recommendation_inputs.execute_notebook
This overwrites only derived notebook outputs/HTML/figures, not frozen experiment runs.
"""
import json,os,re,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[2]
task=Path(tempfile.mkdtemp(prefix='recommendation-notebook-'))
kernel=task/'kernels'/'algorithm-rag-research';kernel.mkdir(parents=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[str(root/'.venv/bin/python'),'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Algorithm RAG research venv','language':'python'}))
os.environ.update(JUPYTER_PATH=str(task),JUPYTER_RUNTIME_DIR=str(task/'runtime'),IPYTHONDIR=str(task/'ipython'),MPLCONFIGDIR=str(task/'matplotlib'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VECLIB_MAXIMUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for name in ['runtime','ipython','matplotlib']:(task/name).mkdir(exist_ok=True)
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
p=root/'research/recommendation_inputs/recommendation_experiments.ipynb'
nb=nbformat.read(p,as_version=4)
client=NotebookClient(nb,timeout=300,kernel_name='algorithm-rag-research',resources={'metadata':{'path':str(root)}},record_timing=True)
try:client.execute()
except Exception:
 nbformat.write(nb,task/'failed.ipynb');print('Failed notebook saved in',task,flush=True);raise
nbformat.validate(nb);nbformat.write(nb,p)
exporter=HTMLExporter();exporter.require_js_url='';exporter.mathjax_url=''
html,resources=exporter.from_notebook_node(nb)
html=re.sub(r'<script\b[^>]*>.*?</script>','',html,flags=re.I|re.S)
html=re.sub(r'<link\b[^>]*href=["\']https?://[^>]*>','',html,flags=re.I)
p.with_suffix('.html').write_text(html)
print(json.dumps({'notebook':str(p),'code_cells':sum(c.cell_type=='code' for c in nb.cells),'executed':sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells),'errors':sum(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[])),'html':str(p.with_suffix('.html'))}),flush=True)
