import requests
import json
import time
import os
from pathlib import Path
from typing import Dict, List

# Paths
ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data" / "raw" / "leetcode_raw_data.jsonl"
REPORT_PATH = ROOT / "data" / "raw" / "new_leetcode_problems.json"


class LeetCodeNewChecker:
    def __init__(self):
        self.url = "https://leetcode.com/graphql"
        self.headers = {
            "Content-Type": "application/json",
            "Referer": "https://leetcode.com/problemset/all/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.existing_slugs = self._load_existing_slugs()

    def _load_existing_slugs(self) -> set:
        slugs = set()
        if not RAW_DATA.exists():
            return slugs
        with open(RAW_DATA, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    slugs.add(item.get("titleSlug"))
                except Exception:
                    continue
        return slugs

    def fetch_problem_list(self, limit=100, skip=0):
        query = """
        query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
          problemsetQuestionList: questionList(
            categorySlug: $categorySlug
            limit: $limit
            skip: $skip
            filters: $filters
          ) {
            total: totalNum
            questions: data {
              questionId
              title
              titleSlug
              difficulty
              isPaidOnly
            }
          }
        }
        """
        variables = {"categorySlug": "", "skip": skip, "limit": limit, "filters": {}}
        try:
            resp = requests.post(self.url, json={"query": query, "variables": variables}, headers=self.headers)
            if resp.status_code == 200:
                return resp.json()["data"]["problemsetQuestionList"]
        except Exception as e:
            print(f"Error fetching list at skip={skip}: {e}")
        return None

    def run(self, max_pages: int = None) -> Dict[str, List[Dict]]:
        batch_size = 100
        skip = 0
        new_items: List[Dict] = []
        paid_count = 0

        while True:
            data = self.fetch_problem_list(limit=batch_size, skip=skip)
            if not data or not data.get("questions"):
                break

            for q in data["questions"]:
                if q.get("isPaidOnly"):
                    paid_count += 1
                    continue
                slug = q.get("titleSlug")
                if slug not in self.existing_slugs:
                    new_items.append({
                        "id": q.get("questionId"),
                        "title": q.get("title"),
                        "titleSlug": slug,
                        "difficulty": q.get("difficulty"),
                        "url": f"https://leetcode.com/problems/{slug}/"
                    })

            skip += batch_size
            if max_pages is not None and (skip // batch_size) >= max_pages:
                break

            time.sleep(0.3)  # be polite

            total = data.get("total", 0)
            if skip >= total:
                break

        report = {
            "existing_count": len(self.existing_slugs),
            "new_count": len(new_items),
            "paid_only_skipped": paid_count,
            "new_problems": new_items,
        }

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"Existing: {report['existing_count']} | New: {report['new_count']} | Paid skipped: {report['paid_only_skipped']}")
        print(f"Saved report: {REPORT_PATH}")
        return report


if __name__ == "__main__":
    LeetCodeNewChecker().run()
