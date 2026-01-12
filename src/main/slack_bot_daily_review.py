import json
import os
import sys
import re
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DailyReviewBot:
    """
    Light Job: Daily Slack notification bot
    """
    
    def __init__(self):
        """Initialize Slack configuration"""
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self.slack_token = os.getenv("SLACK_BOT_TOKEN") 
        
        logger.info(f"[INFO] DailyReviewBot initialized")
    
    def load_recommendations(self) -> Dict:
        """Load pre-computed recommendation map"""
        paths = [
            "/app/artifacts/recommendation_map.json", # Docker volume path
            "artifacts/recommendation_map.json",      # Local dev path
            "recommendation_map.json"                 # Fallback
        ]
        
        for path in paths:
            if os.path.exists(path):
                logger.info(f"Loading recommendation map from: {path}")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    logger.info(f"[INFO] Loaded {len(data)} problems")
                    return data
                except Exception as e:
                    logger.error(f"Error loading {path}: {e}")
        
        logger.error("Recommendation map not found.")
        return {}
    
    def generate_leetcode_url(self, title: str) -> str:
        """
        Converts a problem title into a LeetCode URL slug.
        Ex: "Best Time to Buy and Sell Stock" -> "best-time-to-buy-and-sell-stock"
        """
        if not title or title == "Unknown":
            return "https://leetcode.com/problemset/all/"
            
        # 1. Lowercase and remove leading/trailing spaces
        slug = title.lower().strip()
        # 2. Replace spaces with hyphens
        slug = slug.replace(" ", "-")
        # 3. Remove non-alphanumeric characters (except hyphens)
        slug = re.sub(r'[^a-z0-9\-]', '', slug)
        
        return f"https://leetcode.com/problems/{slug}/"

    def build_slack_block(self, problem: Dict) -> dict:
        """
        Build a simplified Slack message block with Links.
        """
        
        # 1. Data Mapping
        p_id = problem.get('id', 'Unknown ID')
        title = problem.get('title', 'Unknown Title')
        tags = problem.get('tags', [])
        recs = problem.get('recommendations', [])
        
        # 2. Format Tags
        tag_str = ", ".join([f"`{t}`" for t in tags[:3]]) if tags else "No tags"
        
        # 3. Format Recommendations with Links & AI Comment
        rec_str = ""
        if recs:
            rec_lines = []
            for r in recs[:3]: # Show max 3 recommendations
                r_title = r.get('title', 'Unknown')
                r_sim = r.get('similarity', 0)
                r_comment = r.get('ai_comment', '')
                
                # Generate Link
                r_url = self.generate_leetcode_url(r_title)
                
                # Format: <URL|Title> (Similarity)
                line = f"• <{r_url}|{r_title}> (Sim: {r_sim})"
                
                # Add AI Comment if exists
                if r_comment:
                     line += f"\n  > {r_comment}"
                
                rec_lines.append(line)
            rec_str = "\n".join(rec_lines)
        else:
            rec_str = "• No recommendations available"

        # 4. Assemble Text Content
        text_content = (
            f"*[{p_id}] {title}*\n"
            f"Tags: {tag_str}\n"
            f"*Similar LeetCode Problems:*\n{rec_str}"
        )

        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": text_content
            }
        }
    
    def send_notification(self, problems: List[Dict]) -> bool:
        """Send notification via Incoming Webhook"""
        
        if not problems:
            logger.info("No problems to review.")
            return True
        
        if not self.slack_webhook_url:
            logger.error("SLACK_WEBHOOK_URL is missing!")
            return False

        try:
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"Daily Algorithm Review ({datetime.now().strftime('%Y-%m-%d')})"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"You have *{len(problems)}* problems to review today."
                    }
                },
                {"type": "divider"}
            ]
            
            for problem in problems[:10]:
                blocks.append(self.build_slack_block(problem))
                blocks.append({"type": "divider"})
            
            blocks.append({
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": "Generated by Algorithm RAG Engine"
                }]
            })
            
            import requests
            response = requests.post(
                self.slack_webhook_url,
                json={"blocks": blocks},
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"Slack API Error: {response.status_code} - {response.text}")
                return False
                
            logger.info(f"[INFO] Notification sent successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False

    def run(self):
        logger.info("[Light Job] Starting...")
        data = self.load_recommendations()
        if not data: return False
        problem_list = list(data.values())
        return self.send_notification(problem_list)

def main():
    bot = DailyReviewBot()
    sys.exit(0 if bot.run() else 1)

if __name__ == "__main__":
    main()