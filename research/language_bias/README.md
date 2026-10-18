# Language-bias study

Start with [REPORT.md](REPORT.md) and [language_bias.ipynb](language_bias.ipynb). Spec:
`specs/unified-similarity-map/language-bias.md` (includes the selection rule fixed before any result).

Reproduce with fresh output paths (models are local, pinned and hash-verified; `full-v1` is never written):

```sh
.venv/bin/python -m research.language_bias.evaluate --candidates C0,C1,shuffled,random,C2,C3,C4,C5 --output research/language_bias/receipts/evaluation.NEW.json
.venv/bin/python -m research.language_bias.select_representation --evaluation research/language_bias/receipts/evaluation.NEW.json --output research/language_bias/receipts/selection.NEW.json
.venv/bin/python -m research.language_bias.rebuild prepare --output private-learning/unified-map/full-NEW
# then research.unified_map.full_quality neighbors, full_project (pca, umap x3, tsne, tsne --tsne-init random x2),
# full_quality metrics, rebuild language-maps, html_v2, verify
.venv/bin/python -m unittest discover -s research/language_bias -t .
```

Model files: `models.py` downloads only the three approved models (e5-small, paraphrase-multilingual, opus-mt-ja-en) and refuses
anything else; LaBSE was fetched by the user and registered with `register_existing`. Outputs under `private-learning/` stay private.
