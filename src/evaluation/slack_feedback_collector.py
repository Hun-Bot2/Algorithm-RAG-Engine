import os
import sys
import json
import datetime
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.utils.logger import get_logger
from src.evaluation.recommendation_evaluator import RecommendationEvaluator

logger = get_logger(__name__)


class SlackFeedbackCollector:
    """Slack 반응 기반 피드백 수집"""
    
    # 반응 → 피드백 매핑
    EMOJI_TO_FEEDBACK = {
        'white_check_mark': 'completed',   # ✅ 완료
        'heavy_check_mark': 'completed',
        'ballot_box_with_check': 'completed',
        'thumbsup': 'helpful',             # 👍 추천
        '+1': 'helpful',
        'thumbsdown': 'not_helpful',       # 👎 비추천
        '-1': 'not_helpful'
    }
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        channel_id: Optional[str] = None
    ):
        """
        Args:
            bot_token: Slack Bot Token (env: SLACK_BOT_TOKEN)
            channel_id: 모니터링할 채널 ID (env: SLACK_CHANNEL_ID)
        """
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN")
        self.channel_id = channel_id or os.getenv("SLACK_CHANNEL_ID")
        
        if not self.bot_token:
            logger.error("SLACK_BOT_TOKEN not set")
            self.client = None
            return
        
        self.client = WebClient(token=self.bot_token)
        self.evaluator = RecommendationEvaluator()
        
        logger.info(f"✓ Slack client initialized")
    
    def get_messages_with_reactions(
        self,
        hours: int = 1,
        limit: int = 100
    ) -> List[Dict]:
        """
        최근 메시지 중 반응이 있는 메시지 조회
        
        Args:
            hours: 지난 시간 (기본값: 1시간)
            limit: 조회 제한
        
        Returns:
            메시지 리스트
        """
        if not self.client or not self.channel_id:
            logger.error("Slack client not initialized")
            return []
        
        try:
            # 시간 기반 쿼리
            oldest = (datetime.datetime.now() - datetime.timedelta(hours=hours)).timestamp()
            
            result = self.client.conversations_history(
                channel=self.channel_id,
                oldest=oldest,
                limit=limit
            )
            
            messages = result.get('messages', [])
            
            # 반응이 있는 메시지만 필터링
            messages_with_reactions = [
                m for m in messages if m.get('reactions')
            ]
            
            logger.info(f"Found {len(messages_with_reactions)} messages with reactions")
            
            return messages_with_reactions
        
        except SlackApiError as e:
            logger.error(f"Failed to get messages: {e}")
            return []
    
    def extract_recommendation_id_from_message(
        self,
        message_text: str
    ) -> Optional[str]:
        """
        메시지에서 추천 ID 추출
        
        형식: *[platform] problem_name* (D+X 복습) 에서 추천 ID 파싱
        실제로는 메시지의 timestamp를 기반으로 처리
        """
        # 메시지 텍스트에서 날짜와 문제명 추출
        # 예: "[test] fibonacci (D+0 복습)"
        
        if "*[" not in message_text or "]*" not in message_text:
            return None
        
        try:
            start = message_text.index("*[") + 2
            end = message_text.index("]*")
            platform_and_problem = message_text[start:end]
            
            # platform과 problem 분리
            if "] " in platform_and_problem:
                parts = platform_and_problem.split("] ", 1)
                problem_name = parts[1]
                
                # 날짜는 context에서 가져와야 함
                # 여기서는 간단히 문제명만 반환
                return problem_name
        except Exception as e:
            logger.debug(f"Failed to extract recommendation ID: {e}")
        
        return None
    
    def process_message_reactions(
        self,
        message: Dict,
        date: Optional[datetime.date] = None
    ) -> int:
        """
        메시지의 반응 처리
        
        Args:
            message: Slack 메시지 객체
            date: 추천 날짜 (기본값: 오늘)
        
        Returns:
            처리된 반응 개수
        """
        if date is None:
            date = datetime.date.today()
        
        message_text = message.get('text', '')
        reactions = message.get('reactions', [])
        ts = message.get('ts')
        
        processed = 0
        
        for reaction in reactions:
            reaction_name = reaction['name']
            reaction_count = reaction['count']
            
            # 반응 매핑
            feedback_type = self.EMOJI_TO_FEEDBACK.get(reaction_name)
            if not feedback_type:
                continue
            
            # 추천 ID 추출 (임시: ts 기반)
            recommendation_id = f"{date.isoformat()}_{ts}_{reaction_name}"
            
            logger.info(f"Processing reaction: {reaction_name} ({reaction_count}x) on message {ts}")
            
            # 피드백 저장 (반응 개수만큼)
            for _ in range(reaction_count):
                success = self.evaluator.save_feedback(
                    recommendation_id=recommendation_id,
                    feedback_type=feedback_type
                )
                if success:
                    processed += 1
        
        return processed
    
    def collect_recent_feedbacks(self, hours: int = 1) -> Dict:
        """최근 피드백 수집"""
        logger.info(f"Collecting feedbacks from last {hours} hour(s)...")
        
        messages = self.get_messages_with_reactions(hours=hours)
        
        total_processed = 0
        for message in messages:
            processed = self.process_message_reactions(message)
            total_processed += processed
        
        result = {
            'timestamp': datetime.datetime.now().isoformat(),
            'messages_checked': len(messages),
            'reactions_processed': total_processed
        }
        
        logger.info(f"✓ Collected {total_processed} feedbacks from {len(messages)} messages")
        
        return result


class SlackInteractiveMessageBuilder:
    """Slack 인터랙티브 메시지 빌더"""
    
    @staticmethod
    def build_recommendation_message(
        review_data: List[Dict],
        today: datetime.date,
        base_domain: str = "hun-bot2.github.io"
    ) -> str:
        """
        반응 버튼이 있는 메시지 생성
        
        Args:
            review_data: 복습 데이터
            today: 오늘 날짜
            base_domain: 문서 도메인
        
        Returns:
            포맷팅된 메시지
        """
        message = f"📚 오늘의 알고리즘 복습 (기준일: {today})\n"
        message += f"✅ 완료 | 👍 추천 도움됨 | 👎 추천 별로\n\n"
        
        if not review_data:
            message += "오늘은 복습할 문제가 없습니다.\n"
            return message
        
        message += f"총 {len(review_data)}개 문제 복습 필요\n\n"
        
        for item in review_data:
            problem_name = item['problem_name']
            platform = item['platform']
            doc_url = item['url']
            recommendations = item.get('recommendations', [])
            
            message += f"*[{platform}] {problem_name}*\n"
            message += f"<{doc_url}|문제 링크>\n"
            
            if recommendations:
                message += f"추천 LeetCode 문제:\n"
                for i, rec in enumerate(recommendations, 1):
                    title = rec['title']
                    difficulty = rec['difficulty']
                    tags = ', '.join(rec['tags'][:3]) if rec.get('tags') else ''
                    slug = rec['slug']
                    
                    message += f"  {i}. *{title}* ({difficulty})\n"
                    if tags:
                        message += f"     태그: {tags}\n"
                    message += f"     <https://leetcode.com/problems/{slug}/|LeetCode>\n"
            
            message += "\n"
        
        return message


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect Slack feedbacks")
    parser.add_argument(
        "--check-last-hour",
        action="store_true",
        help="Check last 1 hour for feedbacks"
    )
    parser.add_argument(
        "--check-last-hours",
        type=int,
        default=1,
        help="Check last N hours"
    )
    parser.add_argument(
        "--channel-id",
        type=str,
        help="Slack channel ID (env: SLACK_CHANNEL_ID)"
    )
    args = parser.parse_args()
    
    collector = SlackFeedbackCollector(channel_id=args.channel_id)
    
    hours = args.check_last_hours if args.check_last_hour else args.check_last_hours
    
    result = collector.collect_recent_feedbacks(hours=hours)
    
    print(f"\n{'='*80}")
    print("FEEDBACK COLLECTION RESULT")
    print(f"{'='*80}")
    print(f"Messages checked: {result['messages_checked']}")
    print(f"Reactions processed: {result['reactions_processed']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
