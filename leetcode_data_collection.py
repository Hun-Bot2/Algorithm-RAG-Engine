import requests
import json
import time

class LeetCodeDataCollector:
    def __init__(self):
        self.url = "https://leetcode.com/graphql"
        self.headers = {
            "Content-Type": "application/json",
            "Referer": "https://leetcode.com/problemset/all/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }

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
        response = requests.post(self.url, json={"query": query, "variables": variables}, headers=self.headers)
        return response.json()['data']['problemsetQuestionList'] if response.status_code == 200 else None

    def fetch_problem_detail(self, title_slug):
        query = """
        query questionContent($titleSlug: String!) {
          question(titleSlug: $titleSlug) {
            content
            topicTags { name slug }
          }
        }
        """
        variables = {"titleSlug": title_slug}
        response = requests.post(self.url, json={"query": query, "variables": variables}, headers=self.headers)
        return response.json()['data']['question'] if response.status_code == 200 else None

    def run(self, total_to_collect=100):
        all_data = []
        batch_size = 50
        collected_count = 0
        while collected_count < total_to_collect:
            list_data = self.fetch_problem_list(limit=batch_size, skip=collected_count)
            if not list_data or not list_data['questions']: break
            for prob in list_data['questions']:
                if prob['isPaidOnly']: continue
                detail = self.fetch_problem_detail(prob['titleSlug'])
                if detail:
                    all_data.append({
                        "id": prob['questionId'], "title": prob['title'], "titleSlug": prob['titleSlug'],
                        "difficulty": prob['difficulty'], "content": detail['content'],
                        "tags": [tag['name'] for tag in detail['topicTags']]
                    })
                    print(f"Collected LC: {prob['titleSlug']}")
                time.sleep(1.5)
                if len(all_data) >= total_to_collect: break
            collected_count += batch_size
        with open("leetcode_raw_data.jsonl", "w", encoding="utf-8") as f:
            for item in all_data: f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    LeetCodeDataCollector().run(total_to_collect=100)