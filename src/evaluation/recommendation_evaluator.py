import os
import sys
import json
import datetime
from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RecommendationEvaluator:
    """추천 시스템 평가 (LangSmith 기반)"""
    
    def __init__(self):
        """LangSmith 클라이언트 초기화"""
        self.api_key = os.getenv("LANGSMITH_API_KEY")
        self.project_name = os.getenv("LANGCHAIN_PROJECT", "Algorithm-RAG-Engine")
        
        if not self.api_key:
            logger.warning("LANGSMITH_API_KEY not set. Feedback will not be saved.")
            self.client = None
            return
        
        try:
            from langsmith import Client
            self.client = Client(api_key=self.api_key)
            logger.info(f"✓ LangSmith evaluator initialized (project: {self.project_name})")
        except ImportError:
            logger.error("langsmith package not installed")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith client: {e}")
            self.client = None
    
    def save_feedback(
        self,
        recommendation_id: str,
        feedback_type: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        피드백을 LangSmith에 저장
        
        Args:
            recommendation_id: LangSmith run ID
            feedback_type: 'completed', 'helpful', 'not_helpful'
            metadata: 추가 메타데이터
        
        Returns:
            성공 여부
        """
        if not self.client:
            logger.debug("LangSmith client not available; skipping feedback")
            return False
        
        try:
            # 피드백 타입을 점수로 변환
            score = self._feedback_to_score(feedback_type)
            
            self.client.create_feedback(
                run_id=recommendation_id,
                key=f"user_feedback_{feedback_type}",
                score=score,
                comment=json.dumps({
                    'feedback_type': feedback_type,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'metadata': metadata or {}
                }, ensure_ascii=False)
            )
            
            logger.debug(f"Feedback saved to LangSmith: {recommendation_id} -> {feedback_type}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save feedback to LangSmith: {e}")
            return False
    
    @staticmethod
    def _feedback_to_score(feedback_type: str) -> float:
        """피드백 타입을 점수로 변환"""
        mapping = {
            'completed': 1.0,
            'helpful': 1.0,
            'not_helpful': 0.0,
            'partial': 0.5
        }
        return mapping.get(feedback_type, 0.0)
    
    def calculate_daily_metrics(
        self,
        date: Optional[datetime.date] = None
    ) -> Optional[Dict]:
        """
        일별 메트릭 계산 (LangSmith API 사용)
        
        Args:
            date: 날짜 (기본값: 어제)
        
        Returns:
            메트릭 딕셔너리
        """
        if not self.client:
            logger.info("LangSmith client not available; cannot calculate metrics")
            return None
        
        if date is None:
            date = datetime.date.today() - datetime.timedelta(days=1)
        
        try:
            # LangSmith에서 프로젝트의 runs 조회
            # 날짜 범위 설정
            start_time = datetime.datetime.combine(date, datetime.time.min)
            end_time = datetime.datetime.combine(date, datetime.time.max)
            
            # Note: LangSmith API로 runs와 feedbacks 조회
            # 실제 구현은 LangSmith SDK 업데이트에 따라 조정 필요
            logger.info(f"Fetching metrics from LangSmith for {date}")
            
            # 기본 메트릭 구조 반환 (LangSmith 웹 대시보드 사용 권장)
            metrics = {
                'date': date.isoformat(),
                'source': 'langsmith',
                'dashboard_url': f"https://smith.langchain.com/o/your-org/projects/p/{self.project_name}"
            }
            
            logger.info(f"Metrics available at LangSmith dashboard: {metrics['dashboard_url']}")
            return metrics
        
        except Exception as e:
            logger.error(f"Failed to calculate metrics from LangSmith: {e}")
            return None
    
    def print_evaluation_report(
        self,
        date: Optional[datetime.date] = None
    ):
        """평가 리포트 출력 (LangSmith 대시보드 안내)"""
        if date is None:
            date = datetime.date.today() - datetime.timedelta(days=1)
        
        print(f"\n{'='*80}")
        print(f"EVALUATION REPORT - {date}")
        print(f"{'='*80}")
        
        if not self.client:
            print("LangSmith not configured. Please set LANGSMITH_API_KEY.")
        else:
            print(f"Project: {self.project_name}")
            print(f"\nView detailed metrics and analytics at:")
            print(f"https://smith.langchain.com/")
            print(f"\nFeedback data is automatically tracked in LangSmith.")
            print(f"Use the dashboard for:")
            print(f"  - Recommendation success rate")
            print(f"  - User feedback trends")
            print(f"  - Performance analytics")
            print(f"  - A/B testing results")
        
        print(f"{'='*80}\n")


def main():
    """테스트용 메인 함수"""
    evaluator = RecommendationEvaluator()
    
    # 어제 메트릭 계산
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    metrics = evaluator.calculate_daily_metrics(yesterday)
    
    if metrics:
        evaluator.print_evaluation_report(yesterday)
    else:
        print(f"No feedbacks for {yesterday}")


if __name__ == "__main__":
    main()
