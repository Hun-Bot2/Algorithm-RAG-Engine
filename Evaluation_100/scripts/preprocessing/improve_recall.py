import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class QueryOptimizer:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def refine_query(self, bj_content): 
        prompt = f"""
        Extract the core algorithm logic and constraints from the following Korean problem description.
        Translate them into a concise English summary for technical retrieval.
        Focus on: Algorithm type, Data structures, and Complexity constraints.
        
        Problem Description:
        {bj_content}
        
        Output format: [Algorithm Type] Summary of core logic.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content

    def process_dataset(self, input_file, output_file):
        processed_data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                print(f"Refining query for problem: {item.get('id')}")
                # 원문 대신 정제된 영문 쿼리를 embedding_text로 사용
                refined_logic = self.refine_query(item.get("content", ""))
                item["embedding_text"] = f"Logical Skeleton: {refined_logic}"
                processed_data.append(item)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    optimizer = QueryOptimizer()
    optimizer.process_dataset("baekjoon_preprocessed.jsonl", "baekjoon_refined.jsonl")