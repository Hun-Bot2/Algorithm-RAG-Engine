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

    def fetch_problem_list(self, limit=50, skip=0):
        """Get a list of problems with basic info"""
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
        variables = {
            "categorySlug": "",
            "skip": skip,
            "limit": limit,
            "filters": {}
        }
        
        response = requests.post(self.url, json={"query": query, "variables": variables}, headers=self.headers)
        if response.status_code == 200:
            return response.json()['data']['problemsetQuestionList']
        else:
            print(f"Error fetching list: {response.status_code}")
            return None

    def fetch_problem_detail(self, title_slug):
        """Get detailed content and tags of a specific problem."""
        query = """
        query questionContent($titleSlug: String!) {
          question(titleSlug: $titleSlug) {
            content
            topicTags {
              name
              slug
            }
          }
        }
        """
        variables = {"titleSlug": title_slug}
        
        response = requests.post(self.url, json={"query": query, "variables": variables}, headers=self.headers)
        if response.status_code == 200:
            return response.json()['data']['question']
        else:
            print(f"Error fetching detail for {title_slug}: {response.status_code}")
            return None

    def run(self, total_to_collect=100):
        """Main: fetch list then map detailed data"""
        all_data = []
        batch_size = 50
        collected_count = 0

        while collected_count < total_to_collect:
            print(f"Fetching batch starting at {collected_count}...")
            list_data = self.fetch_problem_list(limit=batch_size, skip=collected_count)
            
            if not list_data or not list_data['questions']:
                break

            for prob in list_data['questions']:
                # Exclude paid problems as their content cannot be collected
                if prob['isPaidOnly']:
                    continue
                
                print(f"  > Processing: {prob['titleSlug']}")
                detail = self.fetch_problem_detail(prob['titleSlug'])
                
                if detail:
                    combined = {
                        "id": prob['questionId'],
                        "title": prob['title'],
                        "titleSlug": prob['titleSlug'],
                        "difficulty": prob['difficulty'],
                        "content": detail['content'],
                        "tags": [tag['name'] for tag in detail['topicTags']]
                    }
                    all_data.append(combined)
                
                # Delay to prevent rate limiting
                time.sleep(1.5)
            
            collected_count += batch_size
            
        # Save results
        with open("leetcode_raw_data.jsonl", "w", encoding="utf-8") as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"Successfully collected {len(all_data)} problems.")

if __name__ == "__main__":
    collector = LeetCodeDataCollector()
    # For testing, collect only 50 problems initially
    collector.run(total_to_collect=50)