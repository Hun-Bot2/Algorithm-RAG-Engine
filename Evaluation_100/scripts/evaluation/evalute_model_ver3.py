import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# 환경 변수 로드
load_dotenv()

class AdvancedRecommendationService:
    def __init__(self, bj_file, lc_file):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.bj_data = self._load_jsonl(bj_file)
        self.lc_data = self._load_jsonl(lc_file)
        
        # 리트코드 임베딩 및 텍스트 데이터 준비
        self.lc_texts = [item["embedding_text"] for item in self.lc_data]
        self.lc_embeddings = None

    def _load_jsonl(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def get_embeddings(self, texts):
        # OpenAI v3-small 모델 사용
        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return np.array([d.embedding for d in response.data])

    def prepare_service(self):
        # 서비스 시작 전 리트코드 전체 데이터 임베딩 로드
        # 실제 운영 환경에서는 Vector DB(Pinecone 등)에서 로드하는 것을 권장
        print("Initializing LeetCode embeddings...")
        self.lc_embeddings = self.get_embeddings(self.lc_texts)
        print("Service ready.")

    def _calculate_tag_similarity(self, source_tags, target_tags):
        # 태그 일치도를 점수화 (0.0 ~ 1.0)
        if not source_tags or not target_tags:
            return 0.0
        s_set = set(t.lower() for t in source_tags)
        t_set = set(t.lower() for t in target_tags)
        intersection = s_set.intersection(t_set)
        return len(intersection) / len(s_set) if len(s_set) > 0 else 0.0

    def rerank_with_llm(self, query_problem, candidates):
        # 상위 후보들을 LLM을 통해 정밀 재정렬 (Cross-Attention 효과)
        # 비용 효율성을 위해 상위 10개 정도로 제한 권장
        reranked = []
        
        for cand in candidates:
            prompt = f"""
            Compare two algorithm problems and evaluate their logical equivalence.
            Focus on the core algorithm pattern, constraints, and solving strategy.
            
            [Problem A (Baekjoon)]
            {query_problem['embedding_text']}
            
            [Problem B (LeetCode)]
            {cand['embedding_text']}
            
            Return a similarity score between 0.0 and 1.0.
            Output format: {{"score": float}}
            """
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                score_data = json.loads(response.choices[0].message.content)
                cand['rerank_score'] = score_data.get('score', 0.0)
            except:
                cand['rerank_score'] = cand['vector_score'] # 실패 시 벡터 점수 유지
            
            reranked.append(cand)
            
        # 리랭킹 점수 기준으로 재정렬
        reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
        return reranked

    def recommend(self, bj_id, top_n=5, use_reranker=True):
        # 특정 백준 문제에 대한 추천 수행
        target_bj = next((item for item in self.bj_data if str(item["id"]) == str(bj_id)), None)
        
        if not target_bj:
            return {"error": "Problem not found."}

        # 1. 벡터 검색 (Stage 1: Retrieval)
        query_emb = self.get_embeddings([target_bj["embedding_text"]])
        vector_sims = cosine_similarity(query_emb, self.lc_embeddings)[0]
        
        # 리랭킹을 위해 넉넉하게 후보 추출 (예: 15개)
        candidate_indices = np.argsort(vector_sims)[::-1][:15]
        
        initial_candidates = []
        for idx in candidate_indices:
            tag_sim = self._calculate_tag_similarity(target_bj.get("tags", []), self.lc_data[idx].get("tags", []))
            
            # 하이브리드 스코어링 (벡터 점수 70% + 태그 점수 30%)
            hybrid_score = (vector_sims[idx] * 0.7) + (tag_sim * 0.3)
            
            initial_candidates.append({
                "title": self.lc_data[idx]["title"],
                "titleSlug": self.lc_data[idx]["titleSlug"],
                "difficulty": self.lc_data[idx]["difficulty"],
                "tags": self.lc_data[idx]["tags"],
                "embedding_text": self.lc_data[idx]["embedding_text"],
                "vector_score": float(vector_sims[idx]),
                "hybrid_score": float(hybrid_score)
            })

        # 2. 리랭킹 (Stage 2: Re-ranking)
        if use_reranker:
            # 하이브리드 점수 상위 10개만 리랭킹 수행 (비용 절감)
            top_for_rerank = sorted(initial_candidates, key=lambda x: x['hybrid_score'], reverse=True)[:10]
            final_recommendations = self.rerank_with_llm(target_bj, top_for_rerank)
        else:
            final_recommendations = sorted(initial_candidates, key=lambda x: x['hybrid_score'], reverse=True)

        return {
            "source": {"id": target_bj["id"], "title": target_bj["title"]},
            "results": final_recommendations[:top_n]
        }

if __name__ == "__main__":
    service = AdvancedRecommendationService("baekjoon_refined.jsonl", "leetcode_refined.jsonl")
    service.prepare_service()
    
    # 테스트: 백준 1003번 (피보나치 함수)
    res = service.recommend("1003", top_n=5, use_reranker=True)
    
    print(f"\nRecommendations for BJ {res['source']['id']} {res['source']['title']}:")
    for i, item in enumerate(res['results'], 1):
        score = item.get('rerank_score', item['hybrid_score'])
        print(f"{i}. {item['title']} (Score: {score:.4f})")
        print(f"   Tags: {item['tags']}")