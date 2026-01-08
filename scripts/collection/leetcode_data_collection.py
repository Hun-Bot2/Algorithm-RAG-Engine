import requests
import json
import time
import os
from pathlib import Path
from typing import Optional
from pathlib import Path

class LeetCodeDataCollector:
    def __init__(self, output_file: Optional[str] = None):
        self.url = "https://leetcode.com/graphql"
        self.headers = {
            "Content-Type": "application/json",
            "Referer": "https://leetcode.com/problemset/all/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        # Default output under repo root: data/raw/leetcode_raw_data.jsonl
        root = Path(__file__).resolve().parents[2]
        default_output = root / "data" / "raw" / "leetcode_raw_data.jsonl"
        self.output_file = Path(output_file) if output_file else default_output
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        # Slugs manifest to avoid parsing entire JSONL every time
        self.slugs_manifest = self.output_file.parent / "leetcode_slugs.txt"
        self.collected_slugs = self._load_existing_data()

    def _load_existing_data(self):
        """Load already collected problem slugs to avoid duplicates with minimal cost"""
        slugs = set()
        # Prefer a lightweight manifest if it exists
        if self.slugs_manifest.exists():
            with open(self.slugs_manifest, 'r', encoding='utf-8') as f:
                for line in f:
                    slug = line.strip()
                    if slug:
                        slugs.add(slug)
        elif self.output_file.exists():
            # Fallback: parse JSONL once to bootstrap manifest
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        slugs.add(data.get('titleSlug'))
                    except Exception:
                        continue
            # Write manifest for next runs
            if slugs:
                with open(self.slugs_manifest, 'w', encoding='utf-8') as mf:
                    for s in sorted(slugs):
                        mf.write(s + "\n")
        if slugs:
            print(f"Loaded {len(slugs)} existing problems")
        return slugs

    def fetch_problem_list(self, limit=100, skip=0):
        """Fetch list of problems from LeetCode"""
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
            response = requests.post(self.url, json={"query": query, "variables": variables}, headers=self.headers)
            if response.status_code == 200:
                return response.json()['data']['problemsetQuestionList']
        except Exception as e:
            print(f"Error fetching problem list: {e}")
        
        return None

    def fetch_problem_detail(self, title_slug):
        """Fetch detailed content for a specific problem"""
        query = """
        query questionContent($titleSlug: String!) {
          question(titleSlug: $titleSlug) {
            content
            topicTags { name slug }
          }
        }
        """
        variables = {"titleSlug": title_slug}
        
        backoff = 1.0
        for attempt in range(5):
            try:
                response = requests.post(self.url, json={"query": query, "variables": variables}, headers=self.headers, timeout=20)
                if response.status_code == 200:
                    return response.json()['data']['question']
                # Handle rate limiting or transient errors
                if response.status_code in (429, 502, 503):
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                    continue
            except Exception as e:
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
                continue
        return None

    def run(self, total_to_collect=4000, buffer_flush=50, base_delay=1.0):
        """
        Collect LeetCode problems (free only)
        
        Args:
            total_to_collect: Maximum number of problems to collect (default: 4000)
            buffer_flush: Flush to disk every N writes to reduce I/O overhead
            base_delay: Base delay between detail requests (auto backoff applied on errors)
        """
        print(f"Starting LeetCode data collection")
        print(f"Target: {total_to_collect} free problems")
        print(f"Output: {self.output_file}\n")
        
        batch_size = 50
        skip = 0
        new_collected = 0
        skipped_paid = 0
        skipped_existing = 0
        write_buffer = []
        
        with open(self.output_file, 'a', encoding='utf-8') as f, open(self.slugs_manifest, 'a', encoding='utf-8') as mf:
            while True:
                list_data = self.fetch_problem_list(limit=batch_size, skip=skip)
                
                if not list_data or not list_data['questions']:
                    print("\nNo more problems available")
                    break
                
                total_available = list_data['total']
                
                for prob in list_data['questions']:
                    if prob['isPaidOnly']:
                        skipped_paid += 1
                        continue
                    
                    if prob['titleSlug'] in self.collected_slugs:
                        skipped_existing += 1
                        continue
                    
                    detail = self.fetch_problem_detail(prob['titleSlug'])
                    
                    if detail and detail.get('content'):
                        problem_data = {
                            "id": prob['questionId'],
                            "title": prob['title'],
                            "titleSlug": prob['titleSlug'],
                            "difficulty": prob['difficulty'],
                            "content": detail['content'],
                            "tags": [tag['name'] for tag in detail['topicTags']]
                        }
                        
                        write_buffer.append(json.dumps(problem_data, ensure_ascii=False) + "\n")
                        if len(write_buffer) >= buffer_flush:
                            f.writelines(write_buffer)
                            f.flush()
                            write_buffer.clear()
                        
                        self.collected_slugs.add(prob['titleSlug'])
                        mf.write(prob['titleSlug'] + "\n")
                        mf.flush()
                        new_collected += 1
                        
                        total_collected = len(self.collected_slugs)
                        print(f"[{total_collected:4d}] {prob['titleSlug']:<50} | Paid: {skipped_paid} | Exists: {skipped_existing}")
                    
                    time.sleep(base_delay)
                    
                    if len(self.collected_slugs) >= total_to_collect:
                        # Flush remaining buffer
                        if write_buffer:
                            f.writelines(write_buffer)
                            f.flush()
                            write_buffer.clear()
                        print(f"\nTarget reached: {len(self.collected_slugs)} problems collected")
                        self._print_summary(new_collected, skipped_paid, skipped_existing)
                        return
                
                skip += batch_size
                
                if skip >= total_available:
                    # Flush remaining buffer before exit
                    if write_buffer:
                        f.writelines(write_buffer)
                        f.flush()
                        write_buffer.clear()
                    print(f"\nProcessed all {total_available} problems")
                    break
        
        self._print_summary(new_collected, skipped_paid, skipped_existing)

    def _print_summary(self, new_collected, skipped_paid, skipped_existing):
        """Print collection summary"""
        print(f"\nCollection Summary:")
        print(f"  Total collected: {len(self.collected_slugs)}")
        print(f"  New in this run: {new_collected}")
        print(f"  Skipped (paid): {skipped_paid}")
        print(f"  Skipped (existing): {skipped_existing}")
        print(f"  Saved to: {self.output_file}")


if __name__ == "__main__":
    collector = LeetCodeDataCollector(output_file="../../data/raw/leetcode_raw_data.jsonl")
    collector.run(total_to_collect=4000)
