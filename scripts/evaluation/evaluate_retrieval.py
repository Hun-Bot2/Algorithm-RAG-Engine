import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

INDEX_DIR = ROOT / "indexes" / "faiss_leetcode"


def get_embedding_client(model: str):
    model = model.lower().strip()
    if model == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    elif model == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError("Unknown embedding model. Use 'openai' or 'local'.")


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Precision@k: fraction of top-k that are relevant."""
    if k == 0:
        return 0.0
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Recall@k: fraction of relevant items found in top-k."""
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / len(relevant_set)


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    """Mean Reciprocal Rank: 1 / rank of first relevant item."""
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 1)
    
    # Ideal DCG: all relevant items at top
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(test_set_path: Path, k: int, model: str) -> Dict:
    """Run evaluation on a test set and return aggregated metrics."""
    if not test_set_path.exists():
        raise FileNotFoundError(f"Test set not found: {test_set_path}")
    
    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_DIR}")
    
    # Load index
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # workaround for OpenMP conflict
    embeddings = get_embedding_client(model)
    db = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    
    # Load test queries
    test_queries = []
    with open(test_set_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_queries.append(json.loads(line))
    
    # Compute metrics per query
    results = []
    for query_data in test_queries:
        query = query_data["query"]
        relevant_slugs = query_data.get("relevant_slugs", [])
        
        # Retrieve top-k
        docs = db.similarity_search(query, k=k)
        retrieved_slugs = [doc.metadata.get("slug", "") for doc in docs]
        
        # Compute metrics
        p_at_k = precision_at_k(retrieved_slugs, relevant_slugs, k)
        r_at_k = recall_at_k(retrieved_slugs, relevant_slugs, k)
        mrr_score = mrr(retrieved_slugs, relevant_slugs)
        ndcg_score = ndcg_at_k(retrieved_slugs, relevant_slugs, k)
        
        results.append({
            "query": query,
            "precision@k": p_at_k,
            "recall@k": r_at_k,
            "mrr": mrr_score,
            "ndcg@k": ndcg_score,
            "retrieved": retrieved_slugs,
            "relevant": relevant_slugs,
        })
    
    # Aggregate
    avg_precision = np.mean([r["precision@k"] for r in results])
    avg_recall = np.mean([r["recall@k"] for r in results])
    avg_mrr = np.mean([r["mrr"] for r in results])
    avg_ndcg = np.mean([r["ndcg@k"] for r in results])
    
    return {
        "k": k,
        "model": model,
        "num_queries": len(results),
        "avg_precision@k": avg_precision,
        "avg_recall@k": avg_recall,
        "avg_mrr": avg_mrr,
        "avg_ndcg@k": avg_ndcg,
        "per_query": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate FAISS retrieval quality")
    parser.add_argument("--test-set", required=True, help="Path to test queries JSONL")
    parser.add_argument("--k", type=int, default=5, help="Top-k for metrics")
    parser.add_argument("--model", choices=["openai", "local"], default="openai", help="Embedding model")
    parser.add_argument("--output", help="Optional output JSON path for results")
    args = parser.parse_args()
    
    if args.model == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Falling back to local embeddings.")
        args.model = "local"
    
    metrics = evaluate(Path(args.test_set), k=args.k, model=args.model)
    
    # Print summary
    print(f"\n=== Evaluation Results (k={metrics['k']}, model={metrics['model']}) ===")
    print(f"Queries: {metrics['num_queries']}")
    print(f"Avg Precision@{metrics['k']}: {metrics['avg_precision@k']:.4f}")
    print(f"Avg Recall@{metrics['k']}: {metrics['avg_recall@k']:.4f}")
    print(f"Avg MRR: {metrics['avg_mrr']:.4f}")
    print(f"Avg nDCG@{metrics['k']}: {metrics['avg_ndcg@k']:.4f}")
    
    # Save if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
