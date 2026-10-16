# MiniLM pilot: three platforms, 90 English annotations

Start with [the measured report](REPORT.md), [executed notebook](minilm_pilot.ipynb),
or [HTML](minilm_pilot.html). The model was actually downloaded and run locally,
not trained. The full 21,244-item downloaded API metadata catalog was not encoded.

Inputs are the same title + authored rationale for TF-IDF and MiniLM, with a
separate title-only ablation. Explicit concept lists, platform and difficulty are
excluded. Results retain query-level scores, graph-overlap exclusions and sparse
judgment coverage. Only three groups can test ranking; no model promotion follows.
Nine predetermined cross-platform queries illustrate both proposals and failures.
PCA/t-SNE/UMAP images visualize the cohort, not educational effectiveness.

## Provenance

- [Pinned model manifest](model.manifest.json): exact revision, local file SHA-256,
  384 dimensions, normalization/input contract described in experiment.py.
- [Download receipt](download-receipt.json): 11 files, 91,578,415 bytes; safetensors
  SHA-256 verified against upstream LFS metadata. Weights stay in private-learning/.
- [Results](results.json): text/input hashes, environment, token lengths, CPU times,
  whole-kernel peak RSS, all warnings and independent float64 cosine cross-checks.
- Codeforces catalog source is official; AtCoder Problems is a community dataset.
  These sources supply metadata, not statements. The curated English annotations
  are separate AI-reviewed content, not human gold or API-downloaded statements.
- Old curriculum keys and labels remain frozen; AtCoder legacy/native ID mappings
  are recorded separately. No hosted URL/import migration is performed here.

## Reproduce

Use Python 3.13 and the research environment; do not add these dependencies to the
hosted service. Direct dependency versions are recorded rather than a full
cross-platform transitive lock.

```bash
.venv/bin/python -m pip install -r research/minilm_pilot/requirements-research.txt
MPLCONFIGDIR=/tmp/algorithm-vector-mpl JUPYTER_RUNTIME_DIR=/tmp/algorithm-vector-jupyter .venv/bin/python research/minilm_pilot/build_notebook.py --execute
.venv/bin/python -m unittest research.minilm_pilot.test_experiment tests.learning.test_embedding_backend -v
```

The builder uses a temporary .venv kernelspec. Loading and inference use offline
Hugging Face/Transformers settings and Python socket guards. This is not an OS
firewall test. No fallback remote service exists. Rerunning replaces research
presentation outputs and the pilot's own private vectors, never original assets.

On a new checkout **without the model**, fetch the already-approved pinned
revision into a fresh directory. The downloader refuses overwrite and can run
with standard-library Python alone:

```bash
python3 scripts/learning/download_minilm.py --output private-learning/models/minilm-1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --metadata private-learning/models/minilm-revision-metadata.json --manifest private-learning/models/minilm-local.manifest.json --receipt private-learning/models/minilm-download-receipt.json
python3 - <<'PY'
import json
from pathlib import Path
assert json.loads(Path('private-learning/models/minilm-local.manifest.json').read_text()) == json.loads(Path('research/minilm_pilot/model.manifest.json').read_text())
PY
```

Do not copy this inference environment to the reported 1 GB OCI VM. Its resource
suitability is unverified; the observed development-kernel peak leaves little room
for a small VM's OS and agents. The core application consumes precomputed bounded
assets and remains independent of this experiment.
