import os
import sys
import json
import datetime
import requests
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.logger import get_logger
from src.rag.faiss_recommendation_engine import FAISSRecommendationEngine
from src.evaluation.langsmith_integration import LangSmithTracker

logger = get_logger(__name__)


class GitHubAPIClient:
    """GitHub API를 통한 원격 저장소 조회"""
    
    REPO_OWNER = "Hun-Bot2"
    REPO_NAME = "Hun-Bot2.github.io"
    BASE_PATH = "study/docs/Algorithm"
    
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        self.headers = {}
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
    
    def get_file_list(self, path: str = "") -> List[Dict]:
        """디렉토리 내 파일 목록 조회"""
        full_path = f"{self.BASE_PATH}/{path}".rstrip("/")
        url = f"https://api.github.com/repos/{self.REPO_OWNER}/{self.REPO_NAME}/contents/{full_path}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get file list from {full_path}: {e}")
            return []
    
    def get_file_content(self, filepath: str) -> Optional[str]:
        """파일 내용 조회 (raw content)"""
        url = f"https://raw.githubusercontent.com/{self.REPO_OWNER}/{self.REPO_NAME}/main/{filepath}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.debug(f"Failed to get content for {filepath}: {e}")
            return None
    
    @staticmethod
    def parse_frontmatter_date(content: str) -> Optional[str]:
        """
        마크다운 frontmatter에서 date 필드 추출
        
        예시:
        ---
        id: bj-11660
        date: 2025-12-23
        ---
        
        Returns:
            YYYY-MM-DD 형식의 날짜 문자열 또는 None
        """
        if not content:
            return None
        
        # frontmatter 추출 (--- 사이의 내용)
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
        if not frontmatter_match:
            return None
        
        frontmatter = frontmatter_match.group(1)
        
        # date 필드 추출
        date_match = re.search(r'^date:\s*(\d{4}-\d{2}-\d{2})', frontmatter, re.MULTILINE)
        if date_match:
            return date_match.group(1)
        
        return None
    
    def walk_directory(self, path: str = "") -> List[Dict]:
        """재귀적으로 모든 파일 탐색"""
        all_files = []
        items = self.get_file_list(path)
        
        for item in items:
            if item['type'] == 'file':
                all_files.append(item)
            elif item['type'] == 'dir':
                # 재귀 탐색
                subpath = item['path'].replace(f"{self.BASE_PATH}/", "")
                all_files.extend(self.walk_directory(subpath))
        
        return all_files


class ReviewCycleCalculator:
    """복습 주기 계산"""
    
    EXCLUDE_FILES = {"intro.md", "intro.mdx", "_category_.json"}
    
    @staticmethod
    def get_days_since(date_str: str, today: datetime.date) -> int:
        """특정 날짜로부터의 일수 계산"""
        if not date_str:
            return -1
        
        try:
            target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            return (today - target_date).days
        except ValueError:
            return -1
    
    @staticmethod
    def should_review(days_since: int, review_cycles: List[int]) -> bool:
        """복습 여부 판단"""
        return days_since in review_cycles
    
    @classmethod
    def find_review_problems(
        cls,
        review_cycles: List[int],
        today: Optional[datetime.date] = None
    ) -> List[Dict]:
        if today is None:
            today = datetime.date.today()

        review_problems = []
        github = GitHubAPIClient()

        # 모든 파일 조회
        all_files = github.walk_directory()
        logger.info(f"Found {len(all_files)} files in repository")
        
        for file_info in all_files:
            filename = file_info['name']
            
            # 제외 파일 체크
            if filename in cls.EXCLUDE_FILES:
                continue
            
            # 마크다운 파일만 처리
            if not (filename.endswith('.md') or filename.endswith('.mdx')):
                continue
            
            filepath = file_info['path']
            
            # 파일 내용 조회
            content = github.get_file_content(filepath)
            if not content:
                logger.debug(f"Failed to get content for {filepath}")
                continue
            
            # frontmatter에서 date 추출
            first_date_str = github.parse_frontmatter_date(content)
            
            if not first_date_str:
                logger.debug(f"No date found in frontmatter for {filepath}")
                continue
            
            days = cls.get_days_since(first_date_str, today)
            
            # 복습 주기 확인
            if cls.should_review(days, review_cycles):
                # 플랫폼 추출 (study/docs/Algorithm/Baekjoon/xxx.md -> Baekjoon)
                path_parts = filepath.split('/')
                platform = path_parts[3] if len(path_parts) > 3 else "General"
                
                problem_name = filename.replace('.md', '').replace('.mdx', '')
                
                logger.info(f"Review needed: {problem_name} (D+{days})")
                
                review_problems.append({
                    'filepath': filepath,
                    'filename': filename,
                    'problem_name': problem_name,
                    'platform': platform,
                    'first_date': first_date_str,
                    'days_since': days,
                    'content': content,
                    'url': cls._get_doc_url(filepath)
                })
        
        return review_problems
    
    @staticmethod
    def _get_doc_url(filepath: str) -> str:
        """문서 URL 생성 (GitHub Pages 기준)"""
        base_url = "https://hun-bot2.github.io/docs"
        rel_path = filepath.replace('study/docs/', '').replace('.mdx', '').replace('.md', '')
        return f"{base_url}/{rel_path}"


class RecommendationService:
    """추천 서비스 (Git log + FAISS + LLM)"""
    
    def __init__(self, index_dir: str = "indexes/faiss_production"):
        """
        Args:
            index_dir: FAISS 인덱스 디렉토리
        """
        self.engine = FAISSRecommendationEngine(
            index_dir=index_dir,
            llm_reranking=True
        )
    
    def get_recommendations_for_problem(
        self,
        problem_content: str,
        problem_name: str,
        top_n: int = 3
    ) -> List[Dict]:
        """문제에 대한 추천 조회"""
        try:
            recommendations = self.engine.get_recommendations(
                problem_content,
                top_n=top_n,
                rerank=True
            )
            return recommendations
        except Exception as e:
            logger.error(f"Failed to get recommendations for {problem_name}: {e}")
            return []


class SlackNotifier:
    """Slack 알림"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Args:
            webhook_url: Slack webhook URL (env: SLACK_WEBHOOK_URL)
        """
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    
    def send_message(self, text: str, blocks: Optional[List[Dict]] = None) -> bool:
        """Slack 메시지 전송"""
        if not self.webhook_url:
            logger.warning("SLACK_WEBHOOK_URL not set. Skipping Slack notification.")
            return False
        
        try:
            import requests
            
            payload = {"text": text}
            if blocks:
                payload["blocks"] = blocks
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False
    
    def format_recommendation_block(
        self,
        problem_name: str,
        platform: str,
        doc_url: str,
        days_since: int,
        recommendations: List[Dict]
    ) -> str:
        """추천 결과를 포맷팅"""
        block = f"*[{platform}] {problem_name}* (D+{days_since} 복습)\n"
        block += f"<{doc_url}|문제 링크>\n\n"
        
        if recommendations:
            block += "추천 LeetCode 문제:\n"
            for i, rec in enumerate(recommendations, 1):
                block += f"  {i}. *{rec['title']}* ({rec['difficulty']})\n"
                block += f"     태그: {', '.join(rec['tags'][:3])}\n"
                
                if 'llm_reason' in rec:
                    block += f"     이유: {rec['llm_reason']}\n"
                else:
                    block += f"     유사도: {1.0 - (rec['l2_distance'] / 2.0):.0%}\n"
                
                block += f"     <https://leetcode.com/problems/{rec['slug']}|LeetCode 링크>\n"
        else:
            block += "추천 문제를 찾을 수 없습니다.\n"
        
        block += "\n"
        return block
    
    def create_daily_summary(
        self,
        review_data: List[Dict],
        today: datetime.date
    ) -> str:
        """일일 복습 요약 생성"""
        summary = f"📚 오늘의 알고리즘 복습 (기준일: {today})\n\n"
        
        if not review_data:
            summary += "오늘은 복습할 문제가 없습니다.\n"
            return summary
        
        summary += f"총 {len(review_data)}개 문제 복습 필요\n\n"
        
        blocks = []
        for item in review_data:
            block = self.format_recommendation_block(
                item['problem_name'],
                item['platform'],
                item['url'],
                item['days_since'],
                item['recommendations']
            )
            blocks.append(block)
        
        summary += "".join(blocks)
        return summary


class RecommendationPipeline:
    """전체 파이프라인 (GitHub API 기반)"""
    
    def __init__(self, review_cycles: List[int]):
        """
        Args:
            review_cycles: 복습 주기 리스트
        """
        self.review_cycles = review_cycles
        self.service = RecommendationService()
        self.notifier = SlackNotifier()
        self.langsmith_tracker = LangSmithTracker()
    
    def run(
        self,
        today: Optional[datetime.date] = None,
        dry_run: bool = False,
        send_slack: bool = False
    ) -> Dict:
        """파이프라인 실행"""
        if today is None:
            today = datetime.date.today()
        
        logger.info("="*80)
        logger.info("RUNNING RECOMMENDATION PIPELINE (GitHub API)")
        logger.info("="*80)
        logger.info(f"Date: {today}")
        logger.info(f"Repository: Hun-Bot2/Hun-Bot2.github.io")
        logger.info(f"Review cycles: {self.review_cycles}")
        
        # 1. 복습할 문제 찾기 (GitHub API)
        review_problems = ReviewCycleCalculator.find_review_problems(
            review_cycles=self.review_cycles,
            today=today
        )
        
        logger.info(f"Found {len(review_problems)} problems to review")
        
        # 2. 각 문제에 대한 추천 조회
        review_data = []
        for problem in review_problems:
            logger.info(f"Processing: {problem['problem_name']}")
            
            recommendations = self.service.get_recommendations_for_problem(
                problem['content'],
                problem['problem_name'],
                top_n=3
            )
            
            problem['recommendations'] = recommendations
            
            # LangSmith에 로깅 (평가는 LangSmith에서 자동)
            if recommendations:
                self.langsmith_tracker.log_recommendation(
                    baekjoon_problem=problem['problem_name'],
                    recommendations=recommendations,
                    metadata={
                        'date': today.isoformat(),
                        'platform': problem.get('platform'),
                        'difficulty': problem.get('difficulty', 'unknown')
                    }
                )
            
            review_data.append(problem)
        
        # 3. Slack 메시지 생성 및 전송
        if review_data:
            summary = self.notifier.create_daily_summary(review_data, today)
            
            logger.info("Summary:")
            print(summary)
            
            if send_slack and not dry_run:
                success = self.notifier.send_message(summary)
                if success:
                    logger.info("✓ Slack message sent")
                else:
                    logger.warning("✗ Failed to send Slack message")
        else:
            logger.info("No problems to review today")
        
        logger.info("="*80)
        logger.info("✓ PIPELINE COMPLETE")
        logger.info("="*80)
        
        return {
            'date': today.isoformat(),
            'problems_to_review': len(review_problems),
            'review_data': review_data,
            'langsmith_url': self.langsmith_tracker.get_dashboard_url()
        }


def main():
    parser = argparse.ArgumentParser(
        description="Algorithm Review Recommendation Service (GitHub API)"
    )
    parser.add_argument(
        "--review-cycles",
        type=int,
        nargs="+",
        default=[1, 3, 7, 14, 30],
        help="Review cycle days"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Target date (YYYY-MM-DD, default: today)"
    )
    parser.add_argument(
        "--send-slack",
        action="store_true",
        help="Send message to Slack"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (no Slack message)"
    )
    args = parser.parse_args()
    
    # 날짜 파싱
    today = None
    if args.date:
        try:
            today = datetime.datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            sys.exit(1)
    
    # GitHub API 토큰 확인
    if not os.getenv("GITHUB_TOKEN"):
        logger.warning("GITHUB_TOKEN not set - API rate limit will be lower (60/hour)")
    
    # 파이프라인 실행
    pipeline = RecommendationPipeline(args.review_cycles)
    result = pipeline.run(
        today=today,
        dry_run=args.dry_run,
        send_slack=args.send_slack
    )
    
    return result


if __name__ == "__main__":
    main()
