import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import time

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSIndexBuilder:
    """FAISS 인덱스 빌드 클래스"""
    
    def __init__(self, model_type: str = "openai"):
        """
        Args:
            model_type: 'openai' or 'local'
        """
        self.model_type = model_type
        self.embedding_model = self._load_embedding_model()
        self.dimension = self._get_dimension()
    
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
        
        logger.info(f"✓ Model loaded: {self.model_type}")
        return model
    
    def _get_dimension(self) -> int:
        """임베딩 차원 반환"""
        if self.model_type == "openai":
            return 1536
        elif self.model_type == "local":
            return 384
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def load_records(self, input_path: Path) -> Tuple[List[Dict], List[str]]:
        """처리된 JSONL 로드"""
        logger.info(f"Loading records from {input_path}")
        
        records = []
        embedding_texts = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    record = json.loads(line)
                    records.append(record)
                    embedding_texts.append(record['embedding_text'])
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {i+1}: Invalid JSON - {e}")
                except KeyError as e:
                    logger.warning(f"Line {i+1}: Missing field {e}")
        
        logger.info(f"✓ Loaded {len(records)} records")
        return records, embedding_texts
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """텍스트 배치 임베딩 생성"""
        logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})")
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx:batch_idx+batch_size]
            batch_num = batch_idx // batch_size + 1
            
            start_time = time.time()
            try:
                batch_embeddings = self.embedding_model.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                elapsed = time.time() - start_time
                rate = len(batch_texts) / elapsed if elapsed > 0 else 0
                logger.info(f"Batch {batch_num}/{total_batches}: {len(batch_texts)} texts in {elapsed:.2f}s ({rate:.1f} texts/s)")
            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                raise
        
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"✓ Generated embeddings shape: {embeddings_array.shape}")
        
        return embeddings_array
    
    def build_faiss_index(self, embeddings: np.ndarray) -> 'faiss.Index':
        """FAISS 인덱스 생성 (L2 distance)"""
        logger.info("Building FAISS index...")
        
        import faiss
        
        # L2 거리 기반 인덱스
        index = faiss.IndexFlatL2(self.dimension)
        index.add(embeddings)
        
        logger.info(f"✓ FAISS index created with {index.ntotal} vectors")
        
        return index
    
    def save_index_and_metadata(self, index, records: List[Dict], output_dir: Path):
        """인덱스와 메타데이터 저장"""
        logger.info(f"Saving index and metadata to {output_dir}")
        
        import faiss
        
        # 디렉토리 생성
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # FAISS 인덱스 저장
        index_path = output_dir / "faiss_index"
        faiss.write_index(index, str(index_path))
        logger.info(f"✓ Index saved: {index_path}")
        
        # 메타데이터 저장 (병렬 검색용)
        metadata_path = output_dir / "metadata.jsonl"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for record in records:
                meta = {
                    'id': record['id'],
                    'title': record['title'],
                    'slug': record['slug'],
                    'difficulty': record['difficulty'],
                    'tags': record['tags'],
                    'url': record['url']
                }
                f.write(json.dumps(meta, ensure_ascii=False) + '\n')
        logger.info(f"✓ Metadata saved: {metadata_path}")
        
        # 인덱스 정보 저장
        info = {
            'model': self.model_type,
            'dimension': self.dimension,
            'total_vectors': index.ntotal,
            'metric': 'L2_DISTANCE',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        info_path = output_dir / "index_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Index info saved: {info_path}")
    
    def run_build(self, input_path: Path, output_dir: Path, batch_size: int = 32):
        """전체 빌드 프로세스 실행"""
        logger.info("="*80)
        logger.info("STARTING FAISS INDEX BUILD")
        logger.info("="*80)
        
        start_time = time.time()
        
        # 1. 데이터 로드
        records, embedding_texts = self.load_records(input_path)
        
        if not records:
            logger.error("No records loaded!")
            return
        
        # 2. 임베딩 생성
        embeddings = self.generate_embeddings(embedding_texts, batch_size=batch_size)
        
        # 3. FAISS 인덱스 생성
        index = self.build_faiss_index(embeddings)
        
        # 4. 저장
        self.save_index_and_metadata(index, records, output_dir)
        
        # 5. 검증
        self._verify_index(index, records)
        
        elapsed = time.time() - start_time
        logger.info("="*80)
        logger.info(f"✓ BUILD COMPLETE in {elapsed:.2f}s")
        logger.info("="*80)
    
    def _verify_index(self, index, records: List[Dict]):
        """인덱스 검증"""
        logger.info("Verifying index...")
        
        # 1. 샘플 쿼리
        sample_embedding = index.reconstruct(0)
        distances, indices = index.search(
            np.array([sample_embedding], dtype=np.float32), k=5
        )
        
        print("\n" + "="*80)
        print("INDEX VERIFICATION")
        print("="*80)
        print(f"Query: {records[0]['title']} [{records[0]['difficulty']}]")
        print(f"\nTop 5 most similar:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            print(f"  {i}. {records[idx]['title']} [{records[idx]['difficulty']}]")
            print(f"     Distance: {dist:.4f}")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build production FAISS index")
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
        help="Embedding model to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="indexes/faiss_production",
        help="Output directory for FAISS index"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation"
    )
    args = parser.parse_args()
    
    # Path 변환
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # API key 확인
    if args.model == "openai" and not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set. Switching to local model.")
        args.model = "local"
    
    # 빌드 실행
    builder = FAISSIndexBuilder(model_type=args.model)
    builder.run_build(input_path, output_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
