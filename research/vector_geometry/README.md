# Private vector geometry study

## Research and requirements

The decision is whether the existing 3,053-row index needs FAISS or dimensionality reduction. The study MUST use existing local assets without API calls, regeneration, hosted release, or changes to production. User-requested private analysis is a narrow additional use; historical rights remain unreviewed and the immutable source registry is unchanged.

## Design and tasks

- [x] Measure exact NumPy and FAISS Flat search with matching metrics and bounded, seeded queries.
- [x] Compare PCA and uncentered truncated SVD at several dimensions against original-space neighbors.
- [x] Inspect PCA, SVD, linear discriminant LDA, t-SNE and UMAP projections.
- [x] Execute and retain a notebook, figures, measured results and image-based interpretation.
- [x] Verify input hashes, numerical agreement, notebook execution and rendered figures.

Neighbor overlap measures preservation of an existing embedding geometry, not algorithmic correctness, prerequisite direction, recommendation quality or learning outcomes. All unsupervised reductions fit the complete catalog (transductive diagnostics). Difficulty-supervised LDA is an in-sample illustration and cannot establish generalization. Topic-model LDA requires nonnegative token counts and is not applied to signed dense embeddings. t-SNE/UMAP maps do not establish meaningful global distances or clusters.

The experiment runs on the development machine, not the 1 GB OCI VM or Cloudflare Worker. Model inference and mission scheduling remain independent. No production promotion follows automatically.

## Method references

- [FAISS index selection](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Truncated SVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
- [Linear discriminant analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
- [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [UMAP reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html)

## Artifacts and reproduction

Start with [the image-based report](REPORT.md), [executed notebook](vector_geometry.ipynb),
or [standalone HTML](vector_geometry.html). The notebook executes the editable
`experiment.py` helper, retains its figures/tables and writes `results.json`.
`report.py` renders interpretations from those measurements; `build_notebook.py`
adds the computed summary after successful execution. See REPORT.md for commands.

Validation: three numerical checks cover exact-search agreement, partial selection,
self exclusion, projection translation invariance and cosine scaling. Historical
input hashes are checked against the immutable registry and before/after execution.
No retrieval relevance labels or learner outcomes are used in this study.
