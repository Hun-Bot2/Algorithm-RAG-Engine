# Algorithm RAG Engine - STAR+Alpha Portfolio

## One-Line Summary

Algorithm RAG Engine is a Python-based RAG recommendation pipeline that reads my Baekjoon algorithm study notes, finds similar LeetCode problems with embeddings and vector search, generates a short Korean recommendation reason with GPT-4o-mini, and sends the result to Slack through a Docker/GitHub Actions workflow.

## Why I Made This

I made this because my algorithm practice history was split across Korean Baekjoon problems and English LeetCode problems. After solving a Baekjoon problem, I wanted a repeatable way to find a useful LeetCode follow-up problem with a similar algorithmic pattern instead of manually searching by title, tag, or memory.

The core learning goal was cross-lingual algorithm retrieval: match problems by algorithm logic, not by surface language. The project also let me practice a production-style workflow around AI features: data collection, preprocessing, embedding generation, vector retrieval, LLM explanation, evaluation, Docker packaging, artifact storage, CI workflow design, and Slack delivery.

## STAR

### Situation

Baekjoon problem notes are written in Korean and stored as Markdown/MDX study records. LeetCode problems are English, structured differently, and have different tag conventions. Simple keyword search is weak for this use case because similar algorithm problems often use different wording across platforms.

The repository contains:

- 3,053 collected LeetCode raw records in `data/raw/leetcode_raw_data.jsonl`.
- 3,053 processed LeetCode records in `data/processed/leetcode_processed.jsonl`.
- A NumPy LeetCode index with 3,053 metadata records in `indexes/numpy_leetcode/metadata.json`.
- A 100-pair evaluation dataset in `Evaluation_100/data/ground_truth_v2.json`.
- FAISS indexes from an earlier implementation and NumPy indexes used by the current main pipeline.

### Task

Build a system that can:

- Detect recently changed Baekjoon Markdown/MDX study files.
- Parse each problem's frontmatter metadata.
- Embed the problem title and tags.
- Retrieve similar LeetCode problems from a prebuilt index.
- Avoid recommending the exact same title or duplicate titles.
- Generate a short Korean explanation for why each LeetCode problem is a good follow-up.
- Save a recommendation map as JSON.
- Deliver the recommendations to Slack.
- Package heavy AI work and light Slack delivery separately.
- Evaluate retrieval quality with a controlled benchmark.

### Action

I implemented the production path as two jobs:

- Heavy job: `src/main/generate_index_map.py`
  - Uses GitPython to inspect the latest commit in a sparse-checked-out study repository.
  - Falls back to scanning all Markdown/MDX files if Git processing fails.
  - Parses frontmatter fields such as `id`, `title`, and `tags`.
  - Uses `OpenAIEmbeddings(model="text-embedding-3-small")` through LangChain.
  - Loads `vectors.npy` and `metadata.json` from `indexes/numpy_leetcode`.
  - Calculates squared L2 distance with NumPy.
  - Uses `np.argpartition` to build a top-N candidate pool efficiently.
  - Randomly selects from the top candidate pool to add recommendation diversity.
  - Calls GPT-4o-mini through `ChatOpenAI` to generate one-sentence Korean recommendation comments.
  - Saves `artifacts/recommendation_map.json`.

- Light job: `src/main/slack_bot_daily_review.py`
  - Loads the generated recommendation map from Docker or local paths.
  - Builds Slack Block Kit message sections.
  - Converts LeetCode titles into URL slugs.
  - Sends the message through Slack Incoming Webhook with `requests`.
  - Limits displayed problems and recommendations to avoid oversized Slack messages.

I also built supporting pipelines:

- LeetCode collection through the LeetCode GraphQL API.
- Incremental duplicate avoidance using a slug manifest.
- LeetCode preprocessing with HTML removal, whitespace normalization, difficulty validation, tag extraction, and embedding-text creation.
- FAISS index building for the earlier implementation.
- FAISS-to-NumPy migration for lighter runtime dependencies.
- Vectorization verification scripts.
- Retrieval evaluation scripts with Precision@K, Recall@K, MRR, and nDCG.
- A separate `Evaluation_100` experiment folder for cross-lingual benchmark work, model comparison, visual analysis, and a 100-pair ground truth dataset.

### Result

The current repository has a working production design that separates expensive AI/index generation from lightweight Slack delivery.

Measured and documented assets in the repository:

- 3,053 LeetCode records collected and preprocessed in the main data pipeline.
- 3,053 LeetCode metadata records in the NumPy index.
- 100 Baekjoon-LeetCode ground truth pairs in `Evaluation_100/data/ground_truth_v2.json`.
- Evaluation reports comparing OpenAI, Jina, and BGE-M3 embeddings.
- Generated benchmark images in `Evaluation_100/images`.
- Docker multi-stage build for heavy and light runtime environments.
- GitHub Actions workflows for manual execution and GHCR image publishing.

Benchmark results documented in `Evaluation_100/docs/REPORT.md` and `Evaluation_100/src/final.py`:

| Model | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| OpenAI text-embedding-3-small | 0.22 | 0.50 | 0.65 | 0.354 |
| Jina-v3 | 0.17 | 0.47 | 0.63 | 0.319 |
| BGE-M3 | 0.15 | 0.40 | 0.53 | 0.272 |

Important accuracy note: `README.md` describes a daily 07:00 KST Slack pipeline, but the current `.github/workflows/daily_algorithm_pipeline.yml` has the schedule commented out. In the current checked-in workflow, execution is manual through `workflow_dispatch`. The Docker Compose and workflow structure still support the heavy-job then light-job pipeline.

## Problem -> Solution -> Result -> Functions

### Problem 1: Cross-Lingual Algorithm Matching

Problem:

Baekjoon problems and LeetCode problems are written in different languages and formats. Direct title or keyword matching does not reliably capture algorithmic similarity.

Solution:

Use embedding-based retrieval. The main production pipeline embeds Baekjoon problem metadata with OpenAI `text-embedding-3-small` and searches a LeetCode vector index. The evaluation pipeline also tests Jina-v3 and BGE-M3 to compare multilingual retrieval quality.

Result:

The 100-pair benchmark shows OpenAI `text-embedding-3-small` had the best documented score among the tested models: Recall@10 of 0.65 and MRR of 0.354.

Functions/classes:

- `generate_index_map.py`
  - `load_user_problems`
  - `load_numpy_index`
  - `search_random_numpy`
  - `generate_reasoning`
  - `main`
- `Evaluation_100/scripts/evaluation/evalute_model_ver2.py`
  - `AdvancedEvaluator`
  - `calculate_metrics`
- `Evaluation_100/scripts/evaluation/evaluate_model.py`
  - `RAGEvaluator`
  - `calculate_metrics`
  - `diagnostic_check`

### Problem 2: Heavy Dependencies Made CI Runtime and Image Size Worse

Problem:

The project originally used FAISS and heavy ML dependencies. That increased runtime environment size and made repeated pipeline execution less efficient.

Solution:

Keep FAISS-related scripts and indexes for previous/experimental work, but move the current main pipeline to NumPy vector search. Store vectors as `vectors.npy` and metadata as `metadata.json`. Use a multi-stage Dockerfile so heavy recommendation work and light Slack delivery have different dependency sets.

Result:

The checked-in production job no longer imports FAISS in `src/main/generate_index_map.py`; it uses NumPy for squared L2 distance search. `requirements-heavy.txt` explicitly comments out FAISS, sentence-transformers, and torch as removed for optimization. `requirements-light.txt` excludes heavy ML libraries for Slack delivery.

Functions/classes:

- `scripts/faiss-to-numpy.py`
  - `migrate`
- `src/main/generate_index_map.py`
  - `load_numpy_index`
  - `search_random_numpy`
- `scripts/preprocessing/build_faiss_index.py`
  - `build_index`
- `scripts/preprocessing/build_production_faiss_index.py`
  - `FAISSIndexBuilder`

### Problem 3: Re-Embedding Every Study File Would Waste API Calls

Problem:

Embedding every Baekjoon note on every run would increase cost and runtime.

Solution:

Use GitPython to detect files changed in the latest commit under the target study directory. If Git detection fails, fall back to scanning all Markdown/MDX files.

Result:

The production script processes only recently changed files when Git history is available.

Functions/classes:

- `src/main/generate_index_map.py`
  - `get_latest_changed_files`
  - `get_all_files`
  - `load_user_problems`

### Problem 4: Recommendations Needed Explanation, Not Only Similarity Scores

Problem:

A similarity score alone does not explain why a LeetCode problem is useful after solving a Baekjoon problem.

Solution:

After vector retrieval, call GPT-4o-mini with a concise tutor prompt. The prompt asks for one Korean sentence focused on the algorithmic concept.

Result:

Each saved recommendation includes `ai_comment`. The sample `artifacts/recommendation_map.json` contains Korean explanations for each recommendation.

Functions/classes:

- `src/main/generate_index_map.py`
  - `generate_reasoning`

### Problem 5: Recommendations Needed Daily Delivery

Problem:

Even if recommendations are generated, they are not useful if I have to open files manually every day.

Solution:

Create a lightweight Slack bot that reads `recommendation_map.json`, formats Slack Block Kit messages, creates LeetCode links, and sends them through Slack Incoming Webhook.

Result:

The light bot can send up to 10 problem blocks and up to 3 recommendations per problem. It is packaged separately from the heavy AI job.

Functions/classes:

- `src/main/slack_bot_daily_review.py`
  - `DailyReviewBot`
  - `load_recommendations`
  - `generate_leetcode_url`
  - `build_slack_block`
  - `send_notification`
  - `run`

### Problem 6: LeetCode Data Needed to Be Collected and Kept Updated

Problem:

The recommendation engine needs a LeetCode corpus with problem metadata, content, difficulty, tags, and stable slugs.

Solution:

Collect LeetCode free problems through GraphQL. Save raw data as JSONL. Maintain a slug manifest to avoid duplicate collection. Add scripts to detect new LeetCode problems through either GraphQL or a faster public JSON source.

Result:

The repository contains 3,053 raw and processed LeetCode records in the main data folder.

Functions/classes:

- `scripts/collection/leetcode_data_collection.py`
  - `LeetCodeDataCollector`
  - `fetch_problem_list`
  - `fetch_problem_detail`
  - `run`
- `scripts/collection/leetcode_check_new.py`
  - `LeetCodeNewChecker`
  - `run`
- `scripts/collection/leetcode_check_new_fast.py`
  - `load_local_slugs`
  - `fetch_remote_slugs`
  - `main`

### Problem 7: Raw Problem Text Is Noisy

Problem:

LeetCode content contains HTML and long problem statements. Raw text can include noise that weakens embeddings.

Solution:

Preprocess records by validating required fields, cleaning HTML, normalizing whitespace, filtering by content length, extracting tags, and creating a compact embedding text from difficulty, title, tags, and content.

Result:

The main processed LeetCode dataset contains 3,053 processed records. The evaluation folder also includes raw, preprocessed, refined, and normalized stages for 100 Baekjoon and 100 LeetCode examples.

Functions/classes:

- `scripts/preprocessing/preprocess_leetcode.py`
  - `LeetCodePreprocessor`
  - `validate_record`
  - `clean_content`
  - `extract_metadata`
  - `filter_record`
  - `process_record`
  - `process_file`
- `src/utils/text_processing.py`
  - `clean_html`
  - `normalize_whitespace`
  - `extract_keywords`
  - `create_embedding_text`

### Problem 8: Retrieval Quality Needed Measurement

Problem:

Without evaluation, I could not know whether a model or preprocessing strategy actually improved retrieval.

Solution:

Build a 100-pair Baekjoon-LeetCode ground truth dataset and evaluate models with Recall@1, Recall@5, Recall@10, and MRR. Add separate retrieval metrics such as Precision@K, Recall@K, MRR, and nDCG for test-query evaluation.

Result:

The documented benchmark showed OpenAI text-embedding-3-small performed best among the tested models. The evaluation assets also include visual reports for embedding quality, cross-lingual alignment, t-SNE, UMAP, and model comparison.

Functions/classes:

- `scripts/evaluation/evaluate_retrieval.py`
  - `precision_at_k`
  - `recall_at_k`
  - `mrr`
  - `ndcg_at_k`
  - `evaluate`
- `Evaluation_100/scripts/evaluation/evaluate_model.py`
  - `RAGEvaluator`
- `Evaluation_100/scripts/evaluation/evalute_model_ver2.py`
  - `AdvancedEvaluator`
- `Evaluation_100/scripts/evaluation/evalute_model_ver3.py`
  - `AdvancedRecommendationService`
- `Evaluation_100/src/final.py`
  - `visualize_results`

### Problem 9: I Needed Production Observability and Feedback Ideas

Problem:

Recommendation quality should eventually be tracked with user feedback and LLM trace data.

Solution:

Add LangSmith integration helpers and Slack reaction feedback collection helpers. These files initialize clients from environment variables, log recommendation/feedback data when configured, and provide status/setup commands.

Result:

The code for observability and feedback exists, but the current main production script does not directly call the LangSmith tracker. Feedback collection depends on Slack bot token, channel ID, and LangSmith configuration.

Functions/classes:

- `src/evaluation/langsmith_integration.py`
  - `LangSmithTracker`
  - `LangSmithEvaluator`
  - `setup_langsmith`
- `src/evaluation/recommendation_evaluator.py`
  - `RecommendationEvaluator`
- `src/evaluation/slack_feedback_collector.py`
  - `SlackFeedbackCollector`
  - `SlackInteractiveMessageBuilder`

## Why I Chose This Tech

| Technology | Why I Chose It | How It Is Used |
|---|---|---|
| Python 3.10 | Good ecosystem for data processing, embeddings, APIs, and automation. | Main application, scripts, evaluation, data processing. |
| OpenAI `text-embedding-3-small` | Strong documented result in my 100-pair cross-lingual benchmark and easy API integration. | Production query embedding and evaluation benchmark. |
| GPT-4o-mini | Lower-cost LLM suitable for short recommendation explanations and reranking experiments. | Korean recommendation comments and experimental reranking. |
| LangChain | Provides wrappers for OpenAI embeddings/chat and prompt chains. | `OpenAIEmbeddings`, `ChatOpenAI`, `ChatPromptTemplate`, output parser. |
| NumPy | Lightweight local vector computation without FAISS runtime dependency. | Current production vector search over `vectors.npy`. |
| FAISS | Useful earlier vector index implementation and migration source. | Legacy/experimental index scripts and stored indexes. |
| scikit-learn | Standard metrics and similarity utilities. | Evaluation, cosine similarity, NearestNeighbors comparison. |
| GitPython | Lets the heavy job detect latest changed study files from Git history. | Incremental processing in `generate_index_map.py`. |
| Requests | Simple HTTP client for LeetCode GraphQL, Slack webhook, and Jina API. | Data collection, Slack delivery, external API calls. |
| BeautifulSoup | Reliable HTML-to-text cleaning for LeetCode content. | `clean_html` in preprocessing utilities. |
| Slack Incoming Webhook | Simple way to push daily review messages without building a full Slack app. | `DailyReviewBot.send_notification`. |
| Slack SDK | Needed for reaction-based feedback collection experiments. | `SlackFeedbackCollector`. |
| LangSmith | Intended tracing/evaluation/feedback storage for LLM recommendation quality. | Evaluation helper modules. |
| Docker multi-stage build | Separates heavy recommendation dependencies from light Slack delivery dependencies. | `heavy` and `light` stages in `Dockerfile`. |
| Docker Compose | Runs heavy and light jobs with volumes and environment variables. | `docker-compose.yml`. |
| GitHub Actions | CI workflow orchestration and GHCR image publishing. | `.github/workflows/*.yml`. |
| GitHub Container Registry | Stores prebuilt heavy Docker image. | `publish_docker.yml`, `docker-compose.yml` image reference. |
| AWS S3 | Stores indexes/artifacts outside Git and passes generated JSON between workflow jobs. | Workflow downloads NumPy index and uploads/downloads `recommendation_map.json`. |
| JSON / JSONL | Simple durable format for records, metadata, ground truth, and recommendation maps. | Data, index metadata, evaluation files, artifacts. |
| Pandas | Convenient tabular analysis in the evaluation experiments. | `Evaluation_100` analysis and production experiment engine. |
| Matplotlib / Seaborn | Visualization for benchmark and embedding-quality reports. | `Evaluation_100/src/final.py`, report scripts, generated images. |
| t-SNE / UMAP | Visual analysis of embedding clusters and cross-lingual alignment. | Evaluation notebook/docs/images. |
| Jina embeddings v3 | Candidate multilingual embedding model for comparison. | `Evaluation_100` embedding generation and benchmark. |
| BGE-M3 | Local multilingual embedding model for comparison. | `Evaluation_100` benchmark scripts. |
| Hugging Face sentence-transformers | Local embedding fallback/experiments. | FAISS builder and verification scripts. |
| python-dotenv | Loads local environment variables without hardcoding secrets. | Main scripts, evaluation scripts, integrations. |
| tqdm | Progress display for batch processing and recommendation generation. | Heavy job and embedding generation. |

## Complete Technology and Dependency Inventory

Production dependencies from `requirements-heavy.txt`:

- `python-dotenv`
- `tqdm`
- `requests`
- `gitpython`
- `langchain`
- `langchain-core`
- `langchain-community`
- `langchain-openai`
- `langsmith`
- `openai`
- `tiktoken`
- `numpy`
- `scikit-learn`

Light Slack dependencies from `requirements-light.txt`:

- `python-dotenv`
- `python-dateutil`
- `requests`
- `slack-sdk`
- `langchain`
- `langchain-openai`
- `langsmith`
- `PyGithub`
- `python-json-logger`

Additional libraries used in code or evaluation files:

- `faiss`
- `sentence-transformers`
- `torch` is mentioned as removed from production heavy requirements but implied by sentence-transformers/BGE-M3 experiments.
- `pandas`
- `matplotlib`
- `seaborn`
- `bs4` / BeautifulSoup
- `openai` official SDK
- `sklearn.metrics.pairwise.cosine_similarity`
- `sklearn.neighbors.NearestNeighbors`
- `sklearn.manifold.TSNE`
- `umap` is referenced in evaluation documentation/notebook outputs.

External services and platforms:

- OpenAI API
- Jina AI Embeddings API
- LeetCode GraphQL API
- Slack Incoming Webhook
- Slack Web API for reaction feedback experiments
- LangSmith
- AWS S3
- GitHub Actions
- GitHub Container Registry
- GitHub repository sparse checkout

Environment variables/secrets used:

- `OPENAI_API_KEY`
- `JINA_API_KEY`
- `SLACK_WEBHOOK_URL`
- `SLACK_BOT_TOKEN`
- `SLACK_CHANNEL_ID`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `LANGCHAIN_PROJECT`
- `REPO_URL`
- `BRANCH_NAME`
- `TARGET_FOLDER`
- `DATA_DIR`
- `OUTPUT_DIR`
- `ARTIFACT_PATH`
- `REPO_ROOT`
- `TARGET_SUBDIR`
- `LOG_LEVEL`

## Complete Python Function/Class Inventory

This inventory is generated from the checked-in Python files under `src`, `scripts`, `Evaluation_100/src`, and `Evaluation_100/scripts`.

| File | Classes / Functions |
|---|---|
| `src/main/generate_index_map.py` | `get_latest_changed_files`, `get_all_files`, `load_user_problems`, `load_numpy_index`, `search_random_numpy`, `generate_reasoning`, `main` |
| `src/main/slack_bot_daily_review.py` | `DailyReviewBot.__init__`, `DailyReviewBot.load_recommendations`, `DailyReviewBot.generate_leetcode_url`, `DailyReviewBot.build_slack_block`, `DailyReviewBot.send_notification`, `DailyReviewBot.run`, `main` |
| `src/rag/faiss_recommendation_engine.py` | `FAISSRecommendationEngine.__init__`, `_load_faiss_index`, `_load_metadata`, `_load_index_info`, `_load_embedding_model`, `_init_llm_client`, `get_embedding`, `search_similar`, `rerank_with_llm`, `_detect_leetcode_input`, `get_recommendations` |
| `src/utils/file_io.py` | `read_jsonl`, `write_jsonl`, `read_json`, `write_json` |
| `src/utils/logger.py` | `get_logger` |
| `src/utils/text_processing.py` | `clean_html`, `normalize_whitespace`, `extract_keywords`, `create_embedding_text` |
| `src/evaluation/check_faiss_score.py` | `load_faiss_data`, `compare_performance`, `search_random_numpy`, `test_randomization`, `main` |
| `src/evaluation/langsmith_integration.py` | `LangSmithTracker.__init__`, `log_recommendation`, `log_feedback`, `_feedback_to_score`, `get_dashboard_url`, `log_evaluation_metrics`, `LangSmithEvaluator.__init__`, `create_evaluation_dataset`, `run_evaluation`, `get_performance_summary`, `setup_langsmith` |
| `src/evaluation/recommendation_evaluator.py` | `RecommendationEvaluator.__init__`, `save_feedback`, `_feedback_to_score`, `calculate_daily_metrics`, `print_evaluation_report`, `main` |
| `src/evaluation/slack_feedback_collector.py` | `SlackFeedbackCollector.__init__`, `get_messages_with_reactions`, `extract_recommendation_id_from_message`, `process_message_reactions`, `collect_recent_feedbacks`, `SlackInteractiveMessageBuilder.build_recommendation_message`, `main` |
| `scripts/collection/leetcode_data_collection.py` | `LeetCodeDataCollector.__init__`, `_load_existing_data`, `fetch_problem_list`, `fetch_problem_detail`, `run`, `_print_summary` |
| `scripts/collection/leetcode_check_new.py` | `LeetCodeNewChecker.__init__`, `_load_existing_slugs`, `fetch_problem_list`, `run` |
| `scripts/collection/leetcode_check_new_fast.py` | `load_local_slugs`, `fetch_remote_slugs`, `main` |
| `scripts/preprocessing/preprocess_leetcode.py` | `LeetCodePreprocessor.__init__`, `validate_record`, `clean_content`, `extract_metadata`, `filter_record`, `process_record`, `process_file`, `print_stats`, `main` |
| `scripts/preprocessing/build_faiss_index.py` | `get_embedding_client`, `load_documents`, `build_index`, `main` |
| `scripts/preprocessing/build_production_faiss_index.py` | `FAISSIndexBuilder.__init__`, `_load_embedding_model`, `_get_dimension`, `load_records`, `generate_embeddings`, `build_faiss_index`, `save_index_and_metadata`, `run_build`, `_verify_index`, `main` |
| `scripts/preprocessing/verify_vectorization.py` | `VectorizationVerifier.__init__`, `_load_embedding_model`, `verify_embedding_text_quality`, `verify_vector_generation`, `verify_similarity_calculation`, `verify_cross_difficulty_similarity`, `run_verification`, `print_summary`, `main` |
| `scripts/evaluation/evaluate_retrieval.py` | `get_embedding_client`, `precision_at_k`, `recall_at_k`, `mrr`, `ndcg_at_k`, `evaluate`, `main` |
| `scripts/faiss-to-numpy.py` | `migrate` |
| `Evaluation_100/src/production_engine.py` | `ProductionAlgorithmEngine.__init__`, `_load_jsonl`, `_normalize_tags`, `get_embeddings`, `index_data`, `_calculate_metadata_score`, `rerank_candidates`, `recommend` |
| `Evaluation_100/src/final.py` | `visualize_results` |
| `Evaluation_100/scripts/analysis/data_report.py` | `DatasetVisualizer.__init__`, `_load_jsonl`, `plot_tsne_alignment`, `plot_tag_comparison`, `plot_difficulty_dist`, `plot_text_length`, `plot_gt_coverage`, `plot_potential_candidates`, `run_analysis` |
| `Evaluation_100/scripts/analysis/report.py` | `visualize_comprehensive_results` |
| `Evaluation_100/scripts/collection/baekjoon_data_collection.py` | `BaekjoonDatasetGenerator.__init__`, `fetch_problem`, `generate` |
| `Evaluation_100/scripts/collection/leetcode_data_collection.py` | `LeetCodeDataCollector.__init__`, `fetch_problem_list`, `fetch_problem_detail`, `run` |
| `Evaluation_100/scripts/evaluation/evaluate_model.py` | `RAGEvaluator.__init__`, `get_openai_embeddings`, `get_jina_embeddings`, `get_bge_embeddings`, `diagnostic_check`, `plot_tsne`, `calculate_metrics`, `run_evaluation` |
| `Evaluation_100/scripts/evaluation/evalute_model_ver2.py` | `AdvancedEvaluator.__init__`, `get_openai_embeddings`, `get_jina_embeddings`, `get_bge_embeddings`, `calculate_metrics`, `run` |
| `Evaluation_100/scripts/evaluation/evalute_model_ver3.py` | `AdvancedRecommendationService.__init__`, `_load_jsonl`, `get_embeddings`, `prepare_service`, `_calculate_tag_similarity`, `rerank_with_llm`, `recommend` |
| `Evaluation_100/scripts/preprocessing/data_normal.py` | `normalize_baekjoon` |
| `Evaluation_100/scripts/preprocessing/generate_embeddings.py` | `JinaEmbedding.__init__`, `JinaEmbedding.encode`, `OpenAIEmbedding.__init__`, `OpenAIEmbedding.encode`, `EmbeddingGenerator.__init__`, `extract_text`, `generate_embeddings`, `process_jsonl_file`, `main` |
| `Evaluation_100/scripts/preprocessing/ground_truth_finder.py` | `GTCandidateFinder.__init__`, `find_candidates`, `run` |
| `Evaluation_100/scripts/preprocessing/leetcode_match_pgt.py` | `build_gt` |
| `Evaluation_100/scripts/preprocessing/preprocess.py` | `clean_content`, `preprocess_file` |
| `Evaluation_100/scripts/preprocessing/refine_baekjoon.py` | `BaekjoonRefiner.__init__`, `extract_skeleton`, `process_all` |
| `Evaluation_100/scripts/preprocessing/refine_leetcode.py` | `LeetCodeRefiner.__init__`, `extract_skeleton`, `process_all` |

## Repository Structure and File Roles

### Production application

| Path | Role |
|---|---|
| `src/main/generate_index_map.py` | Main heavy job: changed-file detection, metadata parsing, OpenAI embedding, NumPy search, GPT explanation, JSON artifact output. |
| `src/main/slack_bot_daily_review.py` | Main light job: reads recommendation JSON and sends Slack Block Kit message. |
| `src/rag/faiss_recommendation_engine.py` | Older/alternate FAISS recommendation engine with optional LLM reranking and CLI. |
| `src/utils/file_io.py` | JSON/JSONL read-write helpers. |
| `src/utils/logger.py` | Shared stdout logger configuration. |
| `src/utils/text_processing.py` | HTML cleaning, whitespace normalization, keyword truncation, embedding-text construction. |

### Collection, preprocessing, indexing, migration

| Path | Role |
|---|---|
| `scripts/collection/leetcode_data_collection.py` | Collects free LeetCode problems through GraphQL and writes JSONL. |
| `scripts/collection/leetcode_check_new.py` | Checks LeetCode GraphQL for new free problems not in local raw data. |
| `scripts/collection/leetcode_check_new_fast.py` | Checks new problems from a public JSON source and local slug manifest. |
| `scripts/preprocessing/preprocess_leetcode.py` | Cleans and validates LeetCode raw JSONL into processed JSONL. |
| `scripts/preprocessing/build_faiss_index.py` | Builds LangChain FAISS index from raw LeetCode data. |
| `scripts/preprocessing/build_production_faiss_index.py` | Builds production-style FAISS index and metadata files. |
| `scripts/preprocessing/verify_vectorization.py` | Verifies embedding text quality, vector shape, finite values, and similarity behavior. |
| `scripts/faiss-to-numpy.py` | Converts LangChain FAISS index vectors and metadata into NumPy files. |

### Evaluation and feedback

| Path | Role |
|---|---|
| `scripts/evaluation/evaluate_retrieval.py` | General retrieval evaluation with Precision@K, Recall@K, MRR, nDCG. |
| `src/evaluation/check_faiss_score.py` | Compares FAISS, scikit-learn brute force, and NumPy search outputs and timing. |
| `src/evaluation/langsmith_integration.py` | LangSmith tracking/evaluation helper classes. |
| `src/evaluation/recommendation_evaluator.py` | Saves feedback to LangSmith and prints evaluation report guidance. |
| `src/evaluation/slack_feedback_collector.py` | Reads Slack reactions and maps emojis to feedback labels. |

### Evaluation_100 experiment folder

| Path | Role |
|---|---|
| `Evaluation_100/src/production_engine.py` | Experimental 3-stage RAG engine: dense retrieval, hybrid tag/vector scoring, GPT-4o-mini reranking. |
| `Evaluation_100/src/final.py` | Visualizes documented benchmark results. |
| `Evaluation_100/scripts/collection/baekjoon_data_collection.py` | Baekjoon collection script for evaluation dataset. |
| `Evaluation_100/scripts/collection/leetcode_data_collection.py` | LeetCode collection script for evaluation dataset. |
| `Evaluation_100/scripts/preprocessing/preprocess.py` | Evaluation preprocessing script. |
| `Evaluation_100/scripts/preprocessing/generate_embeddings.py` | Generates Jina/OpenAI embeddings for raw/preprocessed/refined JSONL files. |
| `Evaluation_100/scripts/preprocessing/refine_baekjoon.py` | Refines Baekjoon data for logical skeleton style evaluation. |
| `Evaluation_100/scripts/preprocessing/refine_leetcode.py` | Refines LeetCode data for logical skeleton style evaluation. |
| `Evaluation_100/scripts/preprocessing/data_normal.py` | Normalizes evaluation data formatting. |
| `Evaluation_100/scripts/preprocessing/ground_truth_finder.py` | Creates candidate ground truth mappings. |
| `Evaluation_100/scripts/preprocessing/leetcode_match_pgt.py` | Refines potential ground truth into final mapping. |
| `Evaluation_100/scripts/evaluation/evaluate_model.py` | Evaluates OpenAI, Jina, and BGE-M3 with diagnostics and t-SNE output. |
| `Evaluation_100/scripts/evaluation/evalute_model_ver2.py` | Advanced 100-set benchmark evaluation. |
| `Evaluation_100/scripts/evaluation/evalute_model_ver3.py` | Experimental recommendation service with hybrid scoring and GPT reranking. |
| `Evaluation_100/scripts/analysis/data_report.py` | Data distribution analysis. |
| `Evaluation_100/scripts/analysis/report.py` | Performance report visualization. |
| `Evaluation_100/notebooks/visualize.ipynb` | Notebook for staged validation and visualization. |
| `Evaluation_100/docs/*.md` | Written evaluation reports and validation notes. |
| `Evaluation_100/images/*` | Generated evaluation charts and architecture images. |

### Infrastructure and artifacts

| Path | Role |
|---|---|
| `Dockerfile` | Multi-stage Docker image: `heavy` for recommendation generation, `light` for Slack bot. |
| `docker-compose.yml` | Runs heavy job with sparse checkout and mounted indexes/artifacts; runs light bot with artifact volume. |
| `.github/workflows/daily_algorithm_pipeline.yml` | Manual GitHub Actions pipeline: download NumPy index from S3, run heavy job, upload artifact, download artifact, run light bot. Schedule is currently commented out. |
| `.github/workflows/publish_docker.yml` | Manual workflow that builds and pushes the heavy Docker image to GHCR. |
| `artifacts/recommendation_map.json` | Sample/current generated recommendation output. |
| `indexes/faiss_leetcode` | Earlier FAISS index files. |
| `indexes/faiss_production` | Production-style FAISS index files from the older pipeline. |
| `indexes/numpy_leetcode` | Current production NumPy vector and metadata index. |
| `images/*` | README architecture and Slack/GitHub Actions images. |
| `docs/*` | Additional update and embedding visualization notes. |

## Current Production Workflow

1. GitHub Actions starts the pipeline manually through `workflow_dispatch`.
2. AWS credentials are configured.
3. The NumPy LeetCode index is downloaded from S3 into `indexes/numpy_leetcode`.
4. Docker Compose pulls/runs the heavy job image.
5. The heavy job sparse-checks out `Hun-Bot2.github.io`, targeting `study/docs/Algorithm/`.
6. `src/main/generate_index_map.py` analyzes changed Baekjoon files.
7. The heavy job writes `artifacts/recommendation_map.json`.
8. The workflow syncs `artifacts` to S3.
9. The light job downloads `recommendation_map.json`.
10. `src/main/slack_bot_daily_review.py` sends the Slack message.

## Current Limitations and Honest Notes

- The current GitHub Actions schedule is disabled in the workflow file; only manual execution is active.
- The current main production search uses squared L2 distance over NumPy vectors, not FAISS.
- The current main production retrieval uses title and tags from Baekjoon frontmatter as query text; it does not parse full problem body content.
- The current `generate_index_map.py` has a hardcoded index directory of `/app/indexes/numpy_leetcode`, so local execution needs matching paths or code/config adjustment.
- `src/evaluation/langsmith_integration.py` and feedback collection code exist, but the current main production job does not directly call the tracker.
- The sample `artifacts/recommendation_map.json` includes some weak recommendations, which means retrieval quality is not perfect.
- `src/rag/faiss_recommendation_engine.py` is still present, but it belongs to the FAISS-based earlier/alternate path.
- Some README claims, such as runtime reduction, are documented in the README but are not directly measured by a checked-in benchmark log.
- `Evaluation_100` contains experimental scripts and reports; it should not be described as the exact same code path as the current Docker production pipeline.

## Portfolio Keywords

RAG, cross-lingual retrieval, algorithm recommendation, OpenAI embeddings, GPT-4o-mini, NumPy vector search, FAISS migration, LangChain, Slack bot, GitHub Actions, Docker, AWS S3, GHCR, LeetCode GraphQL, Baekjoon, retrieval evaluation, Recall@K, MRR, nDCG, LangSmith, data preprocessing, JSONL pipeline, embedding visualization.
