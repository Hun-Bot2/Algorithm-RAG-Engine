import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class BaekjoonRefiner:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"

    def extract_skeleton(self, title, tags, content):
        # Extract core algorithm logic with unified format
        prompt = f"""
        Extract the core algorithm logic from the following Korean problem description.
        Translate them into a concise English summary for technical retrieval.
        
        Title: {title}
        Tags: {tags}
        Content: {content}
        
        Output format (strictly follow this structure):
        [Algorithm Type] Concise algorithm category name
        [Problem Summary] Core logic description without examples
        [Complexity] Time: O(...), Space: O(...)
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
                
                skeleton = self.extract_skeleton(
                    item['title'], 
                    item.get('tags', []), 
                    item.get('content_cleaned', item.get('content', ''))
                )
                
                if skeleton:
                    item["embedding_text"] = skeleton
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    processed_count += 1
                
                time.sleep(0.05)

if __name__ == "__main__":
    refiner = BaekjoonRefiner()
    refiner.process_all(
        "Evaluation_100/data/baekjoon_preprocessed.jsonl", 
        "Evaluation_100/data/baekjoon_refined.jsonl"
    )