import os
import sys
import json
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import wraps

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LangSmithTracker:
    """LangSmith 통합 추적"""
    
    def __init__(self):
        """LangSmith 클라이언트 초기화"""
        self.api_key = os.getenv("LANGSMITH_API_KEY")
        self.project_name = os.getenv(
            "LANGSMITH_PROJECT",
            "algorithm-recommendations"
        )
        self.enabled = bool(self.api_key)
        
        if self.enabled:
            try:
                from langsmith import Client
                self.client = Client(api_key=self.api_key)
                logger.info(f"✓ LangSmith initialized (project: {self.project_name})")
            except ImportError:
                logger.warning("langsmith package not installed. Disabling LangSmith integration.")
                self.enabled = False
        else:
            logger.warning("LANGSMITH_API_KEY not set. Disabling LangSmith integration.")
    
    def log_recommendation(
        self,
        baekjoon_problem: str,
        recommendations: List[Dict],
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        추천 결과를 LangSmith에 로깅
        
        Args:
            baekjoon_problem: 백준 문제명
            recommendations: LeetCode 추천 리스트
            metadata: 추가 메타데이터
        
        Returns:
            Run ID (LangSmith에서 추적 가능)
        """
        if not self.enabled:
            return None
        
        try:
            from langsmith import traceable
            
            @traceable(
                name="problem_recommendation",
                project_name=self.project_name
            )
            def recommend():
                return {
                    "baekjoon_problem": baekjoon_problem,
                    "recommendations": recommendations,
                    "metadata": metadata or {},
                    "timestamp": datetime.datetime.now().isoformat()
                }
            
            result = recommend()
            logger.info(f"Logged to LangSmith: {baekjoon_problem}")
            return result
        
        except Exception as e:
            logger.error(f"Failed to log to LangSmith: {e}")
            return None
    
    def log_feedback(
        self,
        recommendation_id: str,
        baekjoon_problem: str,
        leetcode_problem: str,
        feedback_type: str,
        user_note: Optional[str] = None,
        metrics: Optional[Dict] = None
    ) -> bool:
        """
        피드백을 LangSmith에 로깅
        
        Args:
            recommendation_id: 추천 ID
            baekjoon_problem: 백준 문제명
            leetcode_problem: LeetCode 문제명
            feedback_type: helpful/not_helpful/partially/skipped
            user_note: 사용자 노트
            metrics: 평가 메트릭
        
        Returns:
            성공 여부
        """
        if not self.enabled:
            return False
        
        try:
            from langsmith import Client
            
            client = Client(api_key=self.api_key)
            
            # 평가 데이터 생성
            feedback_data = {
                "baekjoon": baekjoon_problem,
                "leetcode": leetcode_problem,
                "feedback": feedback_type,
                "note": user_note,
                "metrics": metrics or {},
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # LangSmith에 피드백 저장
            client.create_feedback(
                run_id=recommendation_id,
                key=f"feedback_{feedback_type}",
                score=self._feedback_to_score(feedback_type),
                comment=json.dumps(feedback_data)
            )
            
            logger.info(f"Feedback logged to LangSmith: {recommendation_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to log feedback to LangSmith: {e}")
            return False
    
    @staticmethod
    def _feedback_to_score(feedback_type: str) -> float:
        """피드백 타입을 점수로 변환"""
        mapping = {
            "helpful": 1.0,
            "partially": 0.5,
            "not_helpful": 0.0,
            "skipped": None
        }
        return mapping.get(feedback_type, 0.0)
    
    def get_dashboard_url(self) -> Optional[str]:
        """LangSmith 대시보드 URL"""
        if not self.enabled:
            return None
        
        return f"https://smith.langchain.com/projects/{self.project_name}"
    
    def log_evaluation_metrics(
        self,
        date: datetime.date,
        metrics: Dict
    ) -> bool:
        """
        일일 평가 지표를 LangSmith에 로깅
        
        Args:
            date: 날짜
            metrics: 평가 지표
        
        Returns:
            성공 여부
        """
        if not self.enabled:
            return False
        
        try:
            # 메트릭을 별도 이벤트로 로깅
            log_data = {
                "date": date.isoformat(),
                "accuracy": metrics.get('accuracy', 0.0),
                "accuracy_percent": metrics.get('accuracy_percent', '0.0%'),
                "total_recommendations": metrics.get('total_recommendations', 0),
                "helpful_count": metrics.get('helpful_count', 0),
                "not_helpful_count": metrics.get('not_helpful_count', 0),
                "partially_count": metrics.get('partially_count', 0),
                "skipped_count": metrics.get('skipped_count', 0),
                "feedback_count": metrics.get('feedback_count', 0),
            }
            
            logger.info(f"Daily metrics logged: {log_data['date']} (accuracy: {log_data['accuracy_percent']})")
            return True
        
        except Exception as e:
            logger.error(f"Failed to log metrics to LangSmith: {e}")
            return False


class LangSmithEvaluator:
    """LangSmith 기반 평가"""
    
    def __init__(self):
        """평가기 초기화"""
        self.tracker = LangSmithTracker()
        self.api_key = os.getenv("LANGSMITH_API_KEY")
        self.project_name = os.getenv("LANGSMITH_PROJECT", "algorithm-recommendations")
        self.enabled = self.tracker.enabled
    
    def create_evaluation_dataset(
        self,
        name: str,
        description: str,
        examples: List[Dict]
    ) -> Optional[str]:
        """
        LangSmith 평가 데이터셋 생성
        
        Args:
            name: 데이터셋 이름
            description: 설명
            examples: 예시 리스트
        
        Returns:
            데이터셋 ID
        """
        if not self.enabled:
            return None
        
        try:
            from langsmith import Client
            
            client = Client(api_key=self.api_key)
            
            # 데이터셋 생성
            dataset = client.create_dataset(
                dataset_name=name,
                description=description
            )
            
            # 예시 추가
            for example in examples:
                client.create_example(
                    dataset_id=dataset.id,
                    inputs=example.get("inputs", {}),
                    outputs=example.get("outputs", {})
                )
            
            logger.info(f"✓ Created evaluation dataset: {name}")
            return dataset.id
        
        except Exception as e:
            logger.error(f"Failed to create evaluation dataset: {e}")
            return None
    
    def run_evaluation(
        self,
        dataset_id: str,
        evaluator_func,
        config: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        LangSmith 평가 실행
        
        Args:
            dataset_id: 데이터셋 ID
            evaluator_func: 평가 함수
            config: 평가 설정
        
        Returns:
            평가 결과
        """
        if not self.enabled:
            return None
        
        try:
            from langsmith import Client, evaluate
            
            client = Client(api_key=self.api_key)
            
            # 평가 실행
            results = evaluate(
                evaluator_func,
                data=dataset_id,
                experiment_prefix="algorithm_recommendations"
            )
            
            logger.info(f"✓ Evaluation completed")
            return results
        
        except Exception as e:
            logger.error(f"Failed to run evaluation: {e}")
            return None
    
    def get_performance_summary(self) -> Optional[Dict]:
        """
        성능 요약 조회
        
        Returns:
            성능 요약
        """
        if not self.enabled:
            return None
        
        try:
            from langsmith import Client
            
            client = Client(api_key=self.api_key)
            
            # 최근 실행 조회
            runs = list(client.list_runs(
                project_name=self.project_name,
                limit=100
            ))
            
            if not runs:
                return None
            
            # 성능 지표 계산
            total_runs = len(runs)
            feedback_count = sum(1 for run in runs if run.feedback_stats)
            avg_latency = sum(
                (run.end_time - run.start_time).total_seconds()
                for run in runs if run.end_time and run.start_time
            ) / total_runs if total_runs > 0 else 0
            
            return {
                "total_runs": total_runs,
                "feedback_count": feedback_count,
                "average_latency_seconds": round(avg_latency, 3),
                "project_url": self.tracker.get_dashboard_url()
            }
        
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return None


# CLI 명령어
def setup_langsmith():
    """LangSmith 설정 안내"""
    print("\n" + "="*80)
    print("LangSmith SETUP GUIDE")
    print("="*80)
    
    print("""
1. LangSmith 계정 생성
   → https://smith.langchain.com
   
2. API Key 발급
   → Settings → API Keys → Create Key
   
3. .env 파일에 추가
   LANGSMITH_API_KEY=lsv2_pt_...
   LANGSMITH_PROJECT=algorithm-recommendations
   
4. 패키지 설치
   pip install langsmith
   
5. 대시보드 확인
   → https://smith.langchain.com/projects/algorithm-recommendations
""")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LangSmith Integration")
    parser.add_argument("--setup", action="store_true", help="Show setup guide")
    parser.add_argument("--status", action="store_true", help="Check LangSmith status")
    args = parser.parse_args()
    
    if args.setup:
        setup_langsmith()
    elif args.status:
        tracker = LangSmithTracker()
        evaluator = LangSmithEvaluator()
        
        print(f"\nLangSmith Status:")
        print(f"  Enabled: {tracker.enabled}")
        if tracker.enabled:
            print(f"  Project: {tracker.project_name}")
            print(f"  Dashboard: {tracker.get_dashboard_url()}")
            
            summary = evaluator.get_performance_summary()
            if summary:
                print(f"\nPerformance Summary:")
                print(f"  Total runs: {summary['total_runs']}")
                print(f"  Feedback count: {summary['feedback_count']}")
                print(f"  Avg latency: {summary['average_latency_seconds']}s")
        print()
