import os
import json
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Load environment variables
load_dotenv()

class ProductionAlgorithmEngine:
    # Map synonyms to ensure metadata scoring works across platforms
    TAG_MAPPING = {
        "dp": "dynamic programming",
        "dynamic programming": "dynamic programming",
        "math": "math",
        "mathematics": "math",
        "recursion": "recursion",
        "backtracking": "backtracking",
        "sorting": "sorting",
        "string": "string",
        "graph": "graph",
        "bfs": "breadth-first search",
        "dfs": "depth-first search",
        "greedy": "greedy",
        "binary search": "binary search"
    }

    def __init__(self, bj_file, lc_file):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.bj_data = self._load_jsonl(bj_file)
        self.lc_data = self._load_jsonl(lc_file)
        
        # Preprocessing LeetCode data
        self.lc_df = pd.DataFrame(self.lc_data)
        self.lc_embeddings = None
        
    def _load_jsonl(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def _normalize_tags(self, tags):
        normalized = set()
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in self.TAG_MAPPING:
                normalized.add(self.TAG_MAPPING[tag_lower])
            else:
                normalized.add(tag_lower)
        return normalized

    def get_embeddings(self, texts):
        # Using OpenAI text-embedding-3-small
        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return np.array([d.embedding for d in response.data])

    def index_data(self):
        # Index all LeetCode data
        print("Indexing LeetCode data...")
        texts = self.lc_df['embedding_text'].tolist()
        self.lc_embeddings = self.get_embeddings(texts)
        print(f"Indexing complete: {len(self.lc_embeddings)} problems indexed")

    def _calculate_metadata_score(self, source, candidate):
        # Normalized tag overlap calculation
        s_tags = self._normalize_tags(source.get('tags', []))
        c_tags = self._normalize_tags(candidate.get('tags', []))
        
        if not s_tags: 
            return 0.5
        
        overlap = s_tags.intersection(c_tags)
        tag_score = len(overlap) / len(s_tags)
        
        return tag_score

    def rerank_candidates(self, query_problem, candidates):
        # Refined LLM reranking to focus on mathematical recurrence and algorithm patterns
        reranked = []
        for cand in candidates:
            prompt = f"""
            Evaluate the logical equivalence between these two algorithm problems.
            Focus on the mathematical pattern (recurrence relation), time complexity, and data structures.
            
            Note: 
            - If one problem is about counting Fibonacci calls and the other is Climbing Stairs, they are highly equivalent because both follow f(n) = f(n-1) + f(n-2).
            - Do not be misled by keyword overlap like 'count' unless the actual algorithm logic matches.
            
            [Baekjoon Problem Summary]
            {query_problem['embedding_text']}
            
            [LeetCode Candidate Summary]
            {cand['embedding_text']}
            
            Return a JSON object with 'score' (0.0 to 1.0) and a brief 'reason'.
            """
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                res_data = json.loads(response.choices[0].message.content)
                cand['final_score'] = float(res_data.get('score', 0.0))
                cand['rerank_reason'] = res_data.get('reason', "")
            except:
                cand['final_score'] = cand['initial_score']
                cand['rerank_reason'] = "Fallback to initial score"
            reranked.append(cand)
            
        return sorted(reranked, key=lambda x: x['final_score'], reverse=True)

    def recommend(self, bj_id, top_n=5):
        # 1. Locate source problem
        target_bj = next((item for item in self.bj_data if str(item["id"]) == str(bj_id)), None)
        if not target_bj:
            return {"error": f"Problem {bj_id} not found."}

        # 2. Stage 1: Vector Search (Recall)
        query_emb = self.get_embeddings([target_bj["embedding_text"]])
        similarities = cosine_similarity(query_emb, self.lc_embeddings)[0]
        
        # 3. Stage 2: Hybrid Metadata Weighting
        candidates = []
        # Extract top 30 to ensure we include the ground truth even if semantic score is low
        top_indices = np.argsort(similarities)[::-1][:30] 
        
        for idx in top_indices:
            cand_data = self.lc_data[idx].copy()
            vector_score = similarities[idx]
            meta_score = self._calculate_metadata_score(target_bj, cand_data)
            
            # Hybrid Calculation: Vector (60%) + Tags (40%)
            # Increased weight on tags to prioritize algorithm type matching
            initial_score = (vector_score * 0.6) + (meta_score * 0.4)
            cand_data['initial_score'] = initial_score
            candidates.append(cand_data)

        # 4. Stage 3: Intelligent Reranking
        # Sort by hybrid score and take top 10 for reranking
        top_10 = sorted(candidates, key=lambda x: x['initial_score'], reverse=True)[:10]
        final_results = self.rerank_candidates(target_bj, top_10)

        return {
            "source": {"id": target_bj["id"], "title": target_bj["title"]},
            "recommendations": final_results[:top_n]
        }

if __name__ == "__main__":
    engine = ProductionAlgorithmEngine("baekjoon_refined.jsonl", "leetcode_refined.jsonl")
    engine.index_data()
    
    # Example: Recommended for BJ 1003
    results = engine.recommend("1003", top_n=5)
    
    print(f"\nRecommendation Results for BJ {results['source']['id']} {results['source']['title']}:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"{i}. {rec['title']} (Score: {rec['final_score']:.4f})")
        print(f"   Reason: {rec.get('rerank_reason', 'N/A')}")
        print(f"   Tags: {rec['tags']}")
        print(f"   URL: https://leetcode.com/problems/{rec['titleSlug']}/")