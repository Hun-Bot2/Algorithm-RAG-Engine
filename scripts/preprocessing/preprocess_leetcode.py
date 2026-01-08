import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.text_processing import clean_html, normalize_whitespace, create_embedding_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LeetCodePreprocessor:
    """전처리 파이프라인 클래스"""
    
    def __init__(self, min_content_length: int = 50, max_content_length: int = 5000):
        """
        Args:
            min_content_length: 최소 content 길이 (필터링용)
            max_content_length: 최대 content 길이 (너무 긴 문제 제외)
        """
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        self.stats = {
            'total': 0,
            'valid': 0,
            'filtered': 0,
            'errors': 0,
            'difficulties': Counter(),
            'tags': Counter()
        }
    
    def validate_record(self, record: Dict) -> bool:
        """레코드 유효성 검증"""
        required_fields = ['id', 'title', 'titleSlug', 'difficulty', 'content']
        
        # 필수 필드 체크
        for field in required_fields:
            if field not in record or not record[field]:
                logger.warning(f"Missing or empty field '{field}' in record id={record.get('id', 'unknown')}")
                return False
        
        # Difficulty 값 검증
        if record['difficulty'] not in ['Easy', 'Medium', 'Hard']:
            logger.warning(f"Invalid difficulty '{record['difficulty']}' for {record['titleSlug']}")
            return False
        
        return True
    
    def clean_content(self, html_content: str) -> str:
        """HTML content 정제"""
        # HTML 태그 제거
        text = clean_html(html_content)
        
        # 공백 정규화
        text = normalize_whitespace(text)
        
        return text
    
    def extract_metadata(self, record: Dict) -> Dict:
        """메타데이터 추출 및 보강"""
        # 기본 메타데이터
        metadata = {
            'id': str(record['id']),
            'title': record['title'].strip(),
            'slug': record['titleSlug'].strip(),
            'difficulty': record['difficulty'],
            'url': f"https://leetcode.com/problems/{record['titleSlug']}/"
        }
        
        # Tags 처리
        tags = record.get('tags', []) or []
        metadata['tags'] = [tag.strip() for tag in tags if tag and tag.strip()]
        
        # Content 정제
        raw_content = record.get('content', '')
        clean_content = self.clean_content(raw_content)
        metadata['content_clean'] = clean_content
        metadata['content_length'] = len(clean_content)
        
        # Embedding text 생성
        metadata['embedding_text'] = create_embedding_text(
            title=metadata['title'],
            content=raw_content,
            tags=metadata['tags'],
            difficulty=metadata['difficulty']
        )
        
        return metadata
    
    def filter_record(self, metadata: Dict) -> bool:
        """품질 필터링"""
        content_len = metadata['content_length']
        
        # Content 길이 체크
        if content_len < self.min_content_length:
            logger.debug(f"Filtered {metadata['slug']}: content too short ({content_len} chars)")
            return False
        
        if content_len > self.max_content_length:
            logger.debug(f"Filtered {metadata['slug']}: content too long ({content_len} chars)")
            return False
        
        # Embedding text 체크
        if not metadata['embedding_text'] or len(metadata['embedding_text']) < 20:
            logger.warning(f"Filtered {metadata['slug']}: embedding text too short")
            return False
        
        return True
    
    def process_record(self, record: Dict) -> Optional[Dict]:
        """단일 레코드 처리"""
        try:
            # 1. 유효성 검증
            if not self.validate_record(record):
                self.stats['errors'] += 1
                return None
            
            # 2. 메타데이터 추출
            metadata = self.extract_metadata(record)
            
            # 3. 품질 필터링
            if not self.filter_record(metadata):
                self.stats['filtered'] += 1
                return None
            
            # 4. 통계 업데이트
            self.stats['difficulties'][metadata['difficulty']] += 1
            for tag in metadata['tags']:
                self.stats['tags'][tag] += 1
            
            self.stats['valid'] += 1
            
            # 5. 출력용 데이터 구성
            output = {
                'id': metadata['id'],
                'title': metadata['title'],
                'slug': metadata['slug'],
                'difficulty': metadata['difficulty'],
                'tags': metadata['tags'],
                'url': metadata['url'],
                'content_clean': metadata['content_clean'],
                'embedding_text': metadata['embedding_text']
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Error processing record {record.get('id', 'unknown')}: {e}")
            self.stats['errors'] += 1
            return None
    
    def process_file(self, input_path: Path, output_path: Path):
        """전체 파일 처리"""
        logger.info(f"Starting preprocessing: {input_path} -> {output_path}")
        
        # 출력 디렉토리 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                self.stats['total'] += 1
                
                try:
                    record = json.loads(line)
                    processed = self.process_record(record)
                    
                    if processed:
                        outfile.write(json.dumps(processed, ensure_ascii=False) + '\n')
                    
                    # Progress
                    if line_num % 500 == 0:
                        logger.info(f"Processed {line_num} records ({self.stats['valid']} valid)")
                
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error at line {line_num}: {e}")
                    self.stats['errors'] += 1
        
        # 최종 통계
        self.print_stats()
        logger.info(f"Preprocessing complete: {output_path}")
    
    def print_stats(self):
        """처리 통계 출력"""
        print("\n" + "="*80)
        print("PREPROCESSING STATISTICS")
        print("="*80)
        print(f"Total records processed: {self.stats['total']}")
        print(f"Valid records: {self.stats['valid']}")
        print(f"Filtered out: {self.stats['filtered']}")
        print(f"Errors: {self.stats['errors']}")
        print(f"\nDifficulty distribution:")
        for diff, count in sorted(self.stats['difficulties'].items()):
            print(f"  {diff}: {count}")
        print(f"\nTop 10 tags:")
        for tag, count in self.stats['tags'].most_common(10):
            print(f"  {tag}: {count}")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess LeetCode raw data")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/leetcode_raw_data.jsonl",
        help="Input raw JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/leetcode_processed.jsonl",
        help="Output processed JSONL file"
    )
    parser.add_argument(
        "--min-content",
        type=int,
        default=50,
        help="Minimum content length (characters)"
    )
    parser.add_argument(
        "--max-content",
        type=int,
        default=5000,
        help="Maximum content length (characters)"
    )
    args = parser.parse_args()
    
    # Path 변환
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # 전처리 실행
    preprocessor = LeetCodePreprocessor(
        min_content_length=args.min_content,
        max_content_length=args.max_content
    )
    preprocessor.process_file(input_path, output_path)


if __name__ == "__main__":
    main()
