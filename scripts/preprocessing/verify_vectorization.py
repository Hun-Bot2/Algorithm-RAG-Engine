import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import Counter

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorizationVerifier:
    """벡터화 검증 클래스"""
    
    def __init__(self, model_type: str = "openai"):
        """
        Args:
            model_type: 'openai' or 'local'
        """
        self.model_type = model_type
        self.results = {
            'embedding_text_quality': {},
            'vector_properties': {},
            'similarity_tests': {},
            'model_info': {}
        }
        self.embedding_model = self._load_embedding_model()
    
    def _load_embedding_model(self):
        """임베딩 모델 로드"""
        logger.info(f"Loading embedding model: {self.model_type}")
        
        if self.model_type == "openai":
            from langchain_openai import OpenAIEmbeddings
            model = OpenAIEmbeddings(model="text-embedding-3-small")
            self.results['model_info'] = {
                'type': 'openai',
                'model': 'text-embedding-3-small',
                'expected_dimension': 1536
            }
        elif self.model_type == "local":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.results['model_info'] = {
                'type': 'local',
                'model': 'sentence-transformers/all-MiniLM-L6-v2',
                'expected_dimension': 384
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"✓ Model loaded: {self.results['model_info']}")
        return model
    
    def verify_embedding_text_quality(self, records: List[Dict]):
        """Embedding text 품질 검증"""
        logger.info("Verifying embedding text quality...")
        
        lengths = []
        has_difficulty = 0
        has_tags = 0
        has_content = 0
        empty_count = 0
        
        for rec in records:
            emb_text = rec.get('embedding_text', '')
            
            if not emb_text:
                empty_count += 1
                continue
            
            lengths.append(len(emb_text))
            
            # 구조 검증
            if rec['difficulty'] in emb_text:
                has_difficulty += 1
            if rec.get('tags') and any(tag in emb_text for tag in rec['tags']):
                has_tags += 1
            if rec.get('content_clean', '')[:50] in emb_text:
                has_content += 1
        
        total = len(records)
        self.results['embedding_text_quality'] = {
            'total_records': total,
            'empty_embeddings': empty_count,
            'avg_length': np.mean(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'has_difficulty_ratio': has_difficulty / total,
            'has_tags_ratio': has_tags / total,
            'has_content_ratio': has_content / total
        }
        
        print("\n" + "="*80)
        print("EMBEDDING TEXT QUALITY")
        print("="*80)
        print(f"Total records: {total}")
        print(f"Empty embeddings: {empty_count}")
        print(f"Average length: {self.results['embedding_text_quality']['avg_length']:.0f} chars")
        print(f"Length range: {self.results['embedding_text_quality']['min_length']:.0f} - {self.results['embedding_text_quality']['max_length']:.0f}")
        print(f"Contains difficulty: {has_difficulty/total:.1%}")
        print(f"Contains tags: {has_tags/total:.1%}")
        print(f"Contains content: {has_content/total:.1%}")
        print("="*80)
    
    def verify_vector_generation(self, records: List[Dict]):
        """벡터 생성 검증"""
        logger.info("Verifying vector generation...")
        
        # 샘플 텍스트로 벡터 생성
        sample_texts = [rec['embedding_text'] for rec in records[:5]]
        
        print("\nGenerating sample embeddings...")
        vectors = self.embedding_model.embed_documents(sample_texts)
        
        # 벡터 속성 검증
        vectors_array = np.array(vectors)
        
        self.results['vector_properties'] = {
            'dimension': vectors_array.shape[1],
            'expected_dimension': self.results['model_info']['expected_dimension'],
            'dimension_match': vectors_array.shape[1] == self.results['model_info']['expected_dimension'],
            'sample_count': len(vectors),
            'vector_norm_mean': float(np.mean(np.linalg.norm(vectors_array, axis=1))),
            'vector_norm_std': float(np.std(np.linalg.norm(vectors_array, axis=1))),
            'all_finite': bool(np.all(np.isfinite(vectors_array))),
            'has_zeros': bool(np.any(vectors_array == 0))
        }
        
        print("\n" + "="*80)
        print("VECTOR PROPERTIES")
        print("="*80)
        print(f"Vector dimension: {self.results['vector_properties']['dimension']}")
        print(f"Expected dimension: {self.results['vector_properties']['expected_dimension']}")
        print(f"Dimension match: {'✓' if self.results['vector_properties']['dimension_match'] else '✗'}")
        print(f"Sample vectors: {self.results['vector_properties']['sample_count']}")
        print(f"Vector norm (mean ± std): {self.results['vector_properties']['vector_norm_mean']:.4f} ± {self.results['vector_properties']['vector_norm_std']:.4f}")
        print(f"All values finite: {'✓' if self.results['vector_properties']['all_finite'] else '✗'}")
        print(f"Contains zeros: {'Yes' if self.results['vector_properties']['has_zeros'] else 'No'}")
        print("="*80)
        
        return vectors_array
    
    def verify_similarity_calculation(self, vectors: np.ndarray, records: List[Dict]):
        """유사도 계산 검증"""
        logger.info("Verifying similarity calculations...")
        
        # L2 distance 계산
        query_vec = vectors[0]  # 첫 번째 벡터를 쿼리로 사용
        distances = np.linalg.norm(vectors - query_vec, axis=1)
        
        # Cosine similarity 계산
        query_norm = query_vec / np.linalg.norm(query_vec)
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        cosine_sims = np.dot(vectors_norm, query_norm)
        
        self.results['similarity_tests'] = {
            'query_problem': records[0]['title'],
            'l2_distances': {
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
                'mean': float(np.mean(distances)),
                'self_distance': float(distances[0])
            },
            'cosine_similarities': {
                'min': float(np.min(cosine_sims)),
                'max': float(np.max(cosine_sims)),
                'mean': float(np.mean(cosine_sims)),
                'self_similarity': float(cosine_sims[0])
            }
        }
        
        print("\n" + "="*80)
        print("SIMILARITY CALCULATION TEST")
        print("="*80)
        print(f"Query problem: {records[0]['title']} [{records[0]['difficulty']}]")
        print(f"\nL2 Distances:")
        print(f"  Self-distance: {distances[0]:.6f} (should be ~0)")
        print(f"  Range: {np.min(distances):.4f} - {np.max(distances):.4f}")
        print(f"  Mean: {np.mean(distances):.4f}")
        print(f"\nCosine Similarities:")
        print(f"  Self-similarity: {cosine_sims[0]:.6f} (should be ~1)")
        print(f"  Range: {np.min(cosine_sims):.4f} - {np.max(cosine_sims):.4f}")
        print(f"  Mean: {np.mean(cosine_sims):.4f}")
        
        # Top 5 most similar (excluding self)
        print(f"\nTop 5 most similar to '{records[0]['title']}':")
        sorted_indices = np.argsort(distances)[1:6]  # Exclude index 0 (self)
        for i, idx in enumerate(sorted_indices, 1):
            print(f"  {i}. {records[idx]['title']} [{records[idx]['difficulty']}]")
            print(f"     L2 distance: {distances[idx]:.4f}, Cosine: {cosine_sims[idx]:.4f}")
        
        print("="*80)
    
    def verify_cross_difficulty_similarity(self, records: List[Dict]):
        """난이도 간 유사도 패턴 검증"""
        logger.info("Verifying cross-difficulty similarity patterns...")
        
        # 난이도별 샘플 수집
        by_difficulty = {'Easy': [], 'Medium': [], 'Hard': []}
        for rec in records:
            diff = rec['difficulty']
            if len(by_difficulty[diff]) < 3:
                by_difficulty[diff].append(rec)
        
        # 각 난이도에서 1개씩 임베딩 생성
        test_samples = []
        for diff in ['Easy', 'Medium', 'Hard']:
            if by_difficulty[diff]:
                test_samples.append(by_difficulty[diff][0])
        
        if len(test_samples) == 3:
            texts = [s['embedding_text'] for s in test_samples]
            vectors = np.array(self.embedding_model.embed_documents(texts))
            
            print("\n" + "="*80)
            print("CROSS-DIFFICULTY SIMILARITY")
            print("="*80)
            
            for i in range(3):
                for j in range(i+1, 3):
                    dist = np.linalg.norm(vectors[i] - vectors[j])
                    print(f"{test_samples[i]['difficulty']} '{test_samples[i]['title']}'")
                    print(f"  vs {test_samples[j]['difficulty']} '{test_samples[j]['title']}'")
                    print(f"  → L2 distance: {dist:.4f}\n")
            
            print("="*80)
    
    def run_verification(self, input_path: Path, sample_size: int = 10):
        """전체 검증 실행"""
        logger.info(f"Starting vectorization verification: {input_path}")
        
        # 데이터 로드
        records = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                records.append(json.loads(line))
        
        logger.info(f"Loaded {len(records)} sample records")
        
        # 검증 단계 실행
        print("\n" + "🔍 VECTORIZATION VERIFICATION REPORT" + "\n")
        
        # 1. Embedding text 품질
        self.verify_embedding_text_quality(records)
        
        # 2. 벡터 생성
        vectors = self.verify_vector_generation(records)
        
        # 3. 유사도 계산
        self.verify_similarity_calculation(vectors, records)
        
        # 4. 난이도 간 유사도
        self.verify_cross_difficulty_similarity(records)
        
        # 최종 요약
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """검증 결과 요약"""
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)
        
        checks = []
        
        # Embedding text quality
        if self.results['embedding_text_quality']['empty_embeddings'] == 0:
            checks.append("✓ All records have embedding text")
        else:
            checks.append(f"✗ {self.results['embedding_text_quality']['empty_embeddings']} records missing embedding text")
        
        if self.results['embedding_text_quality']['has_difficulty_ratio'] > 0.95:
            checks.append("✓ Most embeddings contain difficulty info")
        else:
            checks.append("⚠ Some embeddings missing difficulty info")
        
        # Vector properties
        if self.results['vector_properties']['dimension_match']:
            checks.append("✓ Vector dimensions match expected")
        else:
            checks.append("✗ Vector dimension mismatch")
        
        if self.results['vector_properties']['all_finite']:
            checks.append("✓ All vector values are valid (no NaN/Inf)")
        else:
            checks.append("✗ Vector contains invalid values")
        
        # Similarity
        if abs(self.results['similarity_tests']['l2_distances']['self_distance']) < 1e-6:
            checks.append("✓ Self-distance is near zero")
        else:
            checks.append("⚠ Self-distance is not zero (numerical precision issue)")
        
        if abs(self.results['similarity_tests']['cosine_similarities']['self_similarity'] - 1.0) < 1e-6:
            checks.append("✓ Self-cosine similarity is near 1.0")
        else:
            checks.append("⚠ Self-similarity is not 1.0")
        
        for check in checks:
            print(f"  {check}")
        
        # Overall status
        failed = sum(1 for c in checks if c.startswith('✗'))
        warnings = sum(1 for c in checks if c.startswith('⚠'))
        
        print()
        if failed == 0 and warnings == 0:
            print("🎉 OVERALL: PASSED - Ready for FAISS indexing!")
        elif failed == 0:
            print(f"⚠  OVERALL: PASSED WITH WARNINGS ({warnings} warnings)")
        else:
            print(f"❌ OVERALL: FAILED ({failed} failures, {warnings} warnings)")
        
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Verify vectorization logic")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/leetcode_processed.jsonl",
        help="Input processed JSONL file"
    )
    parser.add_argument(
        "--model",
        choices=["openai", "local"],
        default="openai",
        help="Embedding model to test"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Number of sample records to test"
    )
    args = parser.parse_args()
    
    # Path 변환
    input_path = Path(args.input)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # API key 확인
    if args.model == "openai" and not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set. Switching to local model.")
        args.model = "local"
    
    # 검증 실행
    verifier = VectorizationVerifier(model_type=args.model)
    results = verifier.run_verification(input_path, sample_size=args.sample)


if __name__ == "__main__":
    main()
