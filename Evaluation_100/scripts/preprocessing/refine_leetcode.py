import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LeetCodeRefiner:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini" # 비용 효율성을 고려하되, 품질 저하 시 4o로 상향 검토

    def extract_skeleton(self, title, tags, content):
        # 리트코드 특유의 예시와 제약사항을 제거하고 논리만 추출하도록 지시
        prompt = f"""
        Extract the core algorithm logic from the LeetCode problem below.
        Discard all examples, specific input numbers, and boilerplate text.
        
        Title: {title}
        Tags: {tags}
        Content: {content}
        
        Output format:
        [Algorithm Type] Concise category
        [Summary] Pure logical steps without examples
        [Complexity] Time and Space complexity (e.g., O(N log N))
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error refining {title}: {e}")
            return None

    def process_all(self, input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            all_problems = [json.loads(line) for line in f]

        processed_count = 0
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for item in all_problems:
                print(f"Processing ({processed_count + 1}/{len(all_problems)}): {item['title']}")
                
                # 기존 embedding_text를 Skeleton으로 교체
                skeleton = self.extract_skeleton(
                    item['title'], 
                    item['tags'], 
                    item['content_cleaned']
                )
                
                if skeleton:
                    item["embedding_text"] = skeleton
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    processed_count += 1
                
                # Rate Limit 준수 및 안전한 처리를 위한 미세 지연
                time.sleep(0.05)

if __name__ == "__main__":
    refiner = LeetCodeRefiner()
    refiner.process_all("leetcode_preprocessed.jsonl", "leetcode_refined.jsonl")