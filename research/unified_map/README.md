# Unified map research

Start with [REPORT.md](REPORT.md), [comparison.ipynb](comparison.ipynb) or
[comparison.html](comparison.html). The private package entry point is
`private-learning/unified-map/package-v1/manifest.json`.

Inspect one native problem without conflating evidence channels:

```sh
python3 -m research.unified_map.query atcoder:abc084_c
.venv/bin/python -m research.unified_map.verify private-learning/unified-map/map-v2
.venv/bin/python -m unittest discover -s research/unified_map -v
```

Reproduction uses local pinned assets and fresh output directories:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m research.unified_map.build --output private-learning/unified-map/map-NEW
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m research.unified_map.reduction --output private-learning/unified-map/reduction-NEW
.venv/bin/python -m research.unified_map.evaluate private-learning/unified-map/map-NEW
```

The statement-evaluation/package/report scripts bind the documented final version
paths. For a new release, update those paths explicitly in a reviewed change;
never overwrite the earlier outputs. Notebook execution needs a local Jupyter
kernel and loopback socket permission. Private statement text is never exported.
`map-v1` and `reduction-v1` are preliminary; `v2` contains reviewed corrections.

No model inference/download or API key is needed. Existing title, statement and
legacy OpenAI vector namespaces remain separate. Retain original384 until new
independent algorithm-level judgments support another choice.

## Full-catalog interactive map (all 24,297 nodes)

See [FULL-MAP-REPORT.md](FULL-MAP-REPORT.md) and [full_map.ipynb](full_map.ipynb). The interactive page is private
(`private-learning/unified-map/full-v1/interactive.html`); rebuild it with fresh output paths:

```sh
.venv/bin/python -m research.unified_map.full_embed --output private-learning/unified-map/full-NEW --receipt research/unified_map/full-receipts/embeddings.NEW.json
# then full_project (pca, umap, tsne, tsne --tsne-init random), full_quality neighbors/metrics, full_html, full_verify
```

Title similarity, tag evidence and judged pairs stay separate channels; see the report for coverage and limits.

Language-bias correction (English-translated titles, `full-v2`): see [../language_bias/REPORT.md](../language_bias/REPORT.md).
