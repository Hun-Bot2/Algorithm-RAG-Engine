import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DailyReviewBot:
    """
    Light Job: Daily Slack notification bot
    
    This runs every day and:
    1. Loads the pre-computed recommendation map from S3/artifacts
    2. Selects problems based on review cycles (1, 3, 7, 14, 30 days)
    3. Sends Slack notifications with recommendations
    """
    
    def __init__(self):
        """Initialize Slack client and load configuration"""
        
        # Slack setup
        slack_token = os.getenv("SLACK_BOT_TOKEN")
        if not slack_token:
            raise ValueError("SLACK_BOT_TOKEN not set in environment")
        
        self.slack_client = WebClient(token=slack_token)
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        
        # Review configuration
        self.review_cycles = self._parse_review_cycles()
        
        logger.info(f"✓ DailyReviewBot initialized")
        logger.info(f"  Review cycles: {self.review_cycles}")
    
    def _parse_review_cycles(self) -> List[int]:
        """Parse review cycles from environment variable"""
        cycles_str = os.getenv("REVIEW_CYCLES", "1,3,7,14,30")
        try:
            return [int(c.strip()) for c in cycles_str.split(",")]
        except ValueError:
            logger.warning(f"Invalid REVIEW_CYCLES format: {cycles_str}. Using defaults.")
            return [1, 3, 7, 14, 30]
    
    def load_recommendations(self) -> Dict:
        """
        Load pre-computed recommendation map
        
        Priority:
        1. Docker mount path: /app/recommendation_map.json
        2. S3 download path: ./recommendation_map.json
        3. Local test path: ./artifacts/recommendation_map.json
        """
        
        paths = [
            "/app/recommendation_map.json",
            "recommendation_map.json",
            "artifacts/recommendation_map.json"
        ]
        
        for path in paths:
            if os.path.exists(path):
                logger.info(f"Loading recommendation map from: {path}")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    logger.info(f"✓ Loaded {len(data)} problems from recommendation map")
                    return data
                except Exception as e:
                    logger.error(f"Error loading from {path}: {e}")
                    continue
        
        logger.error("Recommendation map not found in any expected location")
        return {}
    
    def select_problems_for_today(self, 
                                   recommendation_map: Dict,
                                   user_problem_history: Dict = None) -> List[Dict]:
        """
        Select problems for today based on review cycles
        
        Args:
            recommendation_map: Pre-computed recommendation map
            user_problem_history: User's problem solving history {problem_id: last_solved_date}
        
        Returns:
            List of problems to review today
        """
        
        if not recommendation_map:
            return []
        
        problems_for_today = []
        today = datetime.utcnow().date()
        
        for problem_id, entry in recommendation_map.items():
            
            # If no history, include all new problems
            if not user_problem_history or problem_id not in user_problem_history:
                entry['review_reason'] = "New Problem"
                problems_for_today.append(entry)
                continue
            
            # Check if problem is due for review
            last_solved = user_problem_history[problem_id]
            if isinstance(last_solved, str):
                last_solved = datetime.fromisoformat(last_solved).date()
            
            days_since_solved = (today - last_solved).days
            
            # Check against review cycles
            for cycle in self.review_cycles:
                if days_since_solved == cycle:
                    entry['review_reason'] = f"Review Cycle: {cycle} days"
                    problems_for_today.append(entry)
                    break
        
        logger.info(f"Selected {len(problems_for_today)} problems for today's review")
        return problems_for_today
    
    def build_slack_message(self, problem: Dict) -> Dict:
        """
        Build Slack message block for a single problem
        
        Args:
            problem: Problem entry from recommendation map
        
        Returns:
            Slack message block in JSON format
        """
        
        problem_id = problem.get('baekjoon_id', 'UNKNOWN')
        title = problem.get('title', 'Unknown Title')
        difficulty = problem.get('difficulty', 'UNKNOWN')
        tags = problem.get('tags', [])
        recommendations = problem.get('recommendations', [])
        
        # Color coding by difficulty
        difficulty_colors = {
            'Bronze': '#CD7F32',      # Bronze
            'Silver': '#C0C0C0',      # Silver
            'Gold': '#FFD700',        # Gold
            'Platinum': '#E5E4E2',    # Platinum
            'Diamond': '#B9F2FF',     # Diamond
        }
        color = difficulty_colors.get(difficulty, '#808080')
        
        # Build recommendation text
        rec_text = ""
        if recommendations:
            rec_text = "*Recommended LeetCode Problems:*\n"
            for rec in recommendations[:3]:  # Top 3 recommendations
                rec_title = rec.get('title', 'Unknown')
                rec_difficulty = rec.get('difficulty', '?')
                rec_text += f"  • {rec_title} ({rec_difficulty})\n"
        else:
            rec_text = "No recommendations available"
        
        # Build message block
        message = {
            "type": "section",
            "block_id": f"problem_{problem_id}",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Problem:*\n{problem_id} - {title}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Difficulty:*\n{difficulty}"
                }
            ]
        }
        
        if tags:
            message['fields'].append({
                "type": "mrkdwn",
                "text": f"*Tags:*\n{', '.join(tags[:3])}"  # Show top 3 tags
            })
        
        return message
    
    def send_daily_notification(self, problems: List[Dict]) -> bool:
        """
        Send daily review notification to Slack
        
        Args:
            problems: List of problems to review today
        
        Returns:
            True if successful, False otherwise
        """
        
        if not problems:
            logger.info("No problems to review today. Skipping notification.")
            return True
        
        try:
            # Build message blocks
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"📚 Daily Algorithm Review - {datetime.now().strftime('%Y-%m-%d')}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Today you have *{len(problems)}* problems to review!"
                    }
                },
                {
                    "type": "divider"
                }
            ]
            
            # Add problem blocks (limit to 10 to avoid message size issues)
            for problem in problems[:10]:
                blocks.append(self.build_slack_message(problem))
                blocks.append({"type": "divider"})
            
            # Add footer
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"🤖 Generated by Algorithm RAG Engine | {datetime.utcnow().isoformat()}"
                    }
                ]
            })
            
            # Send to Slack
            if self.slack_webhook_url:
                # Use webhook (no Slack API limit issues)
                import requests
                response = requests.post(
                    self.slack_webhook_url,
                    json={"blocks": blocks},
                    timeout=10
                )
                response.raise_for_status()
                logger.info(f"✓ Sent notification via webhook for {len(problems)} problems")
            else:
                logger.warning("SLACK_WEBHOOK_URL not set. Skipping notification.")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False
    
    def run(self):
        """Main execution flow"""
        logger.info("[Light Job] Starting Daily Review Bot...")
        
        try:
            # Step 1: Load pre-computed recommendations
            recommendation_map = self.load_recommendations()
            if not recommendation_map:
                logger.error("Failed to load recommendation map. Aborting.")
                return False
            
            # Step 2: Select problems for today
            problems_for_today = self.select_problems_for_today(recommendation_map)
            
            # Step 3: Send notification
            success = self.send_daily_notification(problems_for_today)
            
            if success:
                logger.info("[Light Job] Daily Review Bot completed successfully!")
            else:
                logger.error("[Light Job] Daily Review Bot encountered errors")
            
            return success
            
        except Exception as e:
            logger.error(f"[Light Job] Unexpected error: {e}", exc_info=True)
            return False


def main():
    """Main entry point for Docker container"""
    
    try:
        bot = DailyReviewBot()
        success = bot.run()
        
        # Exit with status code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()