import os
import sys
import json
import datetime
import argparse
from pathlib import Path
from typing import Optional

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.logger import get_logger
from src.main.recommendation_service import (
    RecommendationPipeline,
    ReviewCycleCalculator
)

logger = get_logger(__name__)


class DailyReviewBot:
    """일일 복습 추천 봇 (GitHub API 기반)"""
    
    DEFAULT_REVIEW_CYCLES = [1, 3, 7, 14, 30]
    
    def __init__(self, review_cycles: Optional[list] = None):
        """
        Args:
            review_cycles: 복습 주기 (env: REVIEW_CYCLES)
        """
        self.review_cycles = (
            review_cycles or 
            self._parse_env_review_cycles() or
            self.DEFAULT_REVIEW_CYCLES
        )
        
        self.pipeline = RecommendationPipeline(self.review_cycles)
    
    @staticmethod
    def _parse_env_review_cycles() -> Optional[list]:
        """환경 변수에서 복습 주기 파싱"""
        cycles_str = os.getenv("REVIEW_CYCLES")
        if cycles_str:
            try:
                return [int(x.strip()) for x in cycles_str.split(',')]
            except ValueError:
                logger.warning(f"Invalid REVIEW_CYCLES format: {cycles_str}")
        return None
    
    def run(self, send_slack: bool = True, dry_run: bool = False):
        """봇 실행"""
        logger.info("="*80)
        logger.info("DAILY REVIEW BOT STARTED (GitHub API)")
        logger.info("="*80)
        
        # 파이프라인 실행
        result = self.pipeline.run(
            today=None,  # 오늘 날짜 자동 사용
            dry_run=dry_run,
            send_slack=send_slack
        )
        
        logger.info("="*80)
        logger.info("DAILY REVIEW BOT FINISHED")
        logger.info("="*80)
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Daily Algorithm Review Bot (GitHub API)"
    )
    parser.add_argument(
        "--review-cycles",
        type=int,
        nargs="+",
        help="Review cycle days (env: REVIEW_CYCLES or default: 1,3,7,14,30)"
    )
    parser.add_argument(
        "--no-slack",
        action="store_true",
        help="Don't send Slack message"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()
    
    # 봇 초기화 (GitHub API 사용)
    bot = DailyReviewBot(review_cycles=args.review_cycles)
    
    # 봇 실행
    bot.run(
        send_slack=not args.no_slack,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
