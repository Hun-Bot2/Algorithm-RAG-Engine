import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSRecommendationEngine:
    """FAISS 기반 LeetCode 추천 엔진"""
    
    def __init__(
        self,
        index_dir: str = "indexes/faiss_production",
        model_type: str = "openai",
        llm_reranking: bool = True
    ):
        """
        Args:
            index_dir: FAISS 인덱스 디렉토리
            model_type: 'openai' or 'local'
            llm_reranking: GPT-4o-mini 리랭킹 사용 여부
        """
        self.index_dir = Path(index_dir)
        self.model_type = model_type
        self.use_llm_reranking = llm_reranking
        
        # FAISS 인덱스 로드
        self.index = self._load_faiss_index()
        self.metadata = self._load_metadata()
        self.index_info = self._load_index_info()
        
        # 임베딩 모델 로드
        self.embedding_model = self._load_embedding_model()
        
        # LLM 클라이언트 초기화
        if self.use_llm_reranking:
            self._init_llm_client()
    
    def _load_faiss_index(self):
        """FAISS 인덱스 로드"""
        index_path = self.index_dir / "faiss_index"
        
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        import faiss
        index = faiss.read_index(str(index_path))
        logger.info(f"✓ Loaded FAISS index with {index.ntotal} vectors")
        return index
    
    def _load_metadata(self) -> List[Dict]:
        """메타데이터 로드"""
        metadata_path = self.index_dir / "metadata.jsonl"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        metadata = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                metadata.append(json.loads(line))
        
        logger.info(f"✓ Loaded metadata for {len(metadata)} problems")
        return metadata
    
    def _load_index_info(self) -> Dict:
        """인덱스 정보 로드"""
        info_path = self.index_dir / "index_info.json"
        
        if not info_path.exists():
            return {}
        
        with open(info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_embedding_model(self):
        """임베딩 모델 로드"""
        logger.info(f"Loading embedding model: {self.model_type}")
        
        if self.model_type == "openai":
            from langchain_openai import OpenAIEmbeddings
            model = OpenAIEmbeddings(model="text-embedding-3-small")
        elif self.model_type == "local":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"✓ Embedding model loaded: {self.model_type}")
        return model
    
    def _init_llm_client(self):
        """LLM 클라이언트 초기화"""
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. LLM reranking disabled.")
            self.use_llm_reranking = False
            return
        
        self.llm_client = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=api_key,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        logger.info("✓ LLM client initialized for reranking")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """텍스트 임베딩 생성"""
        # 길이 제한 (OpenAI API 최대 길이)
        text_truncated = text.replace("\n", " ")[:8000]
        
        embeddings = self.embedding_model.embed_documents([text_truncated])
        return np.array(embeddings[0], dtype=np.float32).reshape(1, -1)
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 10) -> tuple:
        """FAISS에서 유사한 문제 검색"""
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0:  # 유효한 인덱스 확인
                meta = self.metadata[idx].copy()
                meta['l2_distance'] = float(dist)
                meta['rank'] = i + 1
                results.append(meta)
        
        return results
    
    def rerank_with_llm(self, candidates: List[Dict], query_text: str) -> List[Dict]:
        """LLM을 이용한 리랭킹"""
        if not self.use_llm_reranking or not hasattr(self, 'llm_client'):
            return candidates
        
        from langchain_core.messages import HumanMessage
        from langchain_core.output_parsers import JsonOutputParser
        
        reranked = []
        
        for cand in candidates[:5]:  # 비용 절감: 상위 5개만 리랭킹
            try:
                prompt = f"""Evaluate the algorithmic similarity between the following Baekjoon and LeetCode problems.

Baekjoon Problem:
{query_text[:1000]}

LeetCode Problem:
Title: {cand['title']}
Difficulty: {cand['difficulty']}
Tags: {', '.join(cand['tags'])}

Evaluation Criteria:
- Same algorithm pattern usage
- Problem-solving approach similarity
- Difficulty level match

Respond ONLY in the following JSON format (no other text):
{{"score": 0.0~1.0, "reason": "Brief explanation in English"}}
"""
                
                response = self.llm_client.invoke([HumanMessage(content=prompt)])
                result = json.loads(response.content)
                cand['llm_score'] = float(result.get('score', 0.0))
                cand['llm_reason'] = result.get('reason', '')
            except Exception as e:
                logger.warning(f"LLM reranking failed for {cand['title']}: {e}")
                cand['llm_score'] = 1.0 - (cand['l2_distance'] / 2.0)  # Fallback
                cand['llm_reason'] = "LLM 평가 실패 - 벡터 유사도 기반"
            
            reranked.append(cand)
        
        # LLM 점수로 정렬
        reranked.sort(key=lambda x: x['llm_score'], reverse=True)
        
        return reranked + candidates[5:]  # 상위 5개는 리랭킹, 나머지는 원래 순서
    
    def _detect_leetcode_input(self, text: str) -> bool:
        """
        입력이 LeetCode 문제인지 감지
        
        LeetCode 입력이면 리랭킹 스킵 (비용 절감)
        """
        text_lower = text.lower()
        
        # LeetCode 키워드 확인
        leetcode_keywords = [
            'leetcode',
            'lc-',
            'problem ',
            'solution',
            'accepted',
            'easy',
            'medium',
            'hard'
        ]
        
        for keyword in leetcode_keywords:
            if keyword in text_lower:
                return True
        
        # 백준 키워드 확인 (명시적으로 백준이면 LeetCode 아님)
        if any(x in text_lower for x in ['baekjoon', 'boj', '백준']):
            return False
        
        # 영어 비율이 높으면 LeetCode 가능성 높음
        english_count = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        total_alpha = sum(1 for c in text if c.isalpha())
        
        if total_alpha > 0:
            english_ratio = english_count / total_alpha
            if english_ratio > 0.8:
                return True
        
        return False
    
    def get_recommendations(
        self,
        query_text: str,
        top_n: int = 3,
        rerank: bool = None
    ) -> List[Dict]:
        """
        추천 문제 조회
        
        Args:
            query_text: 문제 내용 (백준, LeetCode 모두 가능)
            top_n: 반환 개수
            rerank: LLM 리랭킹 여부 (None=자동 감지)
        
        Returns:
            추천 문제 리스트 (LeetCode 문제 정보)
        """
        # 1. 쿼리 임베딩
        query_embedding = self.get_embedding(query_text)
        
        # 2. FAISS 검색 (상위 10개)
        candidates = self.search_similar(query_embedding, k=10)
        
        # 3. 리랭킹 결정 (자동 감지 또는 명시적 설정)
        if rerank is None:
            # LeetCode 입력이면 리랭킹 스킵 (비용 절감)
            is_leetcode = self._detect_leetcode_input(query_text)
            rerank = not is_leetcode
            
            if is_leetcode:
                logger.info("✓ LeetCode 입력 감지: 리랭킹 스킵 (비용 절감)")
            else:
                logger.info("✓ 백준/일반 입력: 리랭킹 적용")
        
        # 4. LLM 리랭킹
        if rerank and self.use_llm_reranking:
            candidates = self.rerank_with_llm(candidates, query_text)
        
        # 5. 상위 N개 반환
        return candidates[:top_n]


# CLI 인터페이스
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Get LeetCode recommendations")
    parser.add_argument("--query", type=str, help="Query problem description")
    parser.add_argument("--top-n", type=int, default=3, help="Number of recommendations")
    parser.add_argument("--no-rerank", action="store_true", help="Disable LLM reranking")
    args = parser.parse_args()
    
    # 엔진 초기화
    engine = FAISSRecommendationEngine(
        llm_reranking=not args.no_rerank
    )
    
    # 샘플 쿼리 또는 입력
    if args.query:
        query = args.query
    else:
        print("Enter your problem description (or quit to exit):")
        query = input().strip()
    
    if query.lower() == 'quit':
        exit(0)
    
    # 추천 조회
    print(f"\nSearching recommendations for: {query[:100]}...\n")
    
    recommendations = engine.get_recommendations(
        query,
        top_n=args.top_n,
        rerank=not args.no_rerank
    )
    
    # 결과 출력
    print("="*80)
    print(f"Top {len(recommendations)} Recommendations")
    print("="*80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']} [{rec['difficulty']}]")
        print(f"   Tags: {', '.join(rec['tags'])}")
        print(f"   URL: https://leetcode.com/problems/{rec['slug']}/")
        print(f"   L2 Distance: {rec['l2_distance']:.4f}")
        
        if 'llm_score' in rec:
            print(f"   LLM Score: {rec['llm_score']:.2f}")
            print(f"   Reason: {rec['llm_reason']}")
    
    print("\n" + "="*80)
