# MiniLM cross-platform pilot

## Research, requirements and design

The user authorized downloading and running MiniLM on 2026-10-16. Codeforces metadata came from its official API; AtCoder metadata came from the unofficial AtCoder Problems community dataset. Neither catalog includes statements. This pilot SHALL use the existing 90 reviewed English title/rationale annotations (60 LeetCode, 20 AtCoder, 10 Codeforces), not imply full-catalog statement encoding.

Pin `sentence-transformers/all-MiniLM-L6-v2` to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, licensed Apache-2.0 according to its model card. Download only safetensors and required tokenizer/config/pooling files; retain upstream revision and file hashes. Inference MUST run CPU-local with offline mode, no remote code, no OpenAI and no server. Do not train or promote the model. Existing 1,536-dimensional vectors remain untouched and incompatible with this 384-dimensional namespace.

All ranking methods SHALL receive the same title + independently authored English concept rationale. Exclude explicit concept lists, native difficulty, platform and relationship edges from encoder/TF-IDF input. Include a title-only ablation. Native catalog identity checks use the reviewed alias registry but original curriculum/labels remain frozen. Labels and annotations are AI-reviewed and may share conceptual language, so even good scores would remain conditional diagnostics.

## Task breakdown and validation plan

- [x] Download pinned allowlisted assets and validate upstream safetensors SHA and local file manifest.
- [x] Run the existing local-only embedding backend on 90 common-input texts; record CPU latency, token lengths/truncation, norms, dimensions, memory and package versions.
- [x] Compare same-input TF-IDF and MiniLM; show cross-platform nearest-neighbor examples and PCA/t-SNE/UMAP figures.
- [x] Evaluate similarity-only independently reviewed candidate groups, exclude seed-graph overlap, report sparse conditional denominators and unjudged missingness. Do not treat unjudged candidates as negatives.
- [x] Execute notebook, inspect PNG/HTML artifacts, verify historical hashes and commit only intended source/reports.

## Risks and operating boundary

This is an offline pilot on the development computer, not OCI/Cloudflare inference. The model truncates at 256 wordpieces; measure actual lengths. Projection layouts do not prove prerequisites or transfer benefit. Human-independent labels, full statement retrieval and sufficient query coverage remain absent. No automatic curriculum changes, deployment, Slack sends or paid compute follow from this experiment.

## Executed receipt

The pinned 91,578,415-byte model snapshot was downloaded and its safetensors
SHA-256 verified. Two local-only 90-input variants executed successfully. The
conditional similarity diagnostic retained 35 query groups, only three rankable;
TF-IDF nDCG@5 was 1.000 and MiniLM 0.877 for both variants. This is insufficient
coverage to estimate full-catalog quality, and does not favor promotion. Fixed
examples expose narrative/technique confusion. See research/minilm_pilot/REPORT.md
for measured resources, independent float64 checks, provenance and exact limits.
