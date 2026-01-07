import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class GTCandidateFinder:
    """
    백준 문제와 유사한 리트코드 후보를 LLM으로 자동 추출하여 
    GT(Ground Truth) 제작 속도를 10배 이상 높이는 도구입니다.
    """
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def find_candidates(self, bj_problem, lc_summaries):
        prompt = f"""
        Below is a Korean algorithm problem. Find 3 most similar LeetCode problems from the list provided.
        Only provide the 'titleSlug' of the matching problems.
        
        [Baekjoon Problem]
        Title: {bj_problem['title']}
        Logic: {bj_problem['embedding_text']}
        
        [LeetCode Candidates (Slugs only)]
        {lc_summaries}
        
        Output format (JSON): {{"candidates": ["slug1", "slug2", "slug3"]}}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error finding candidates for {bj_problem['id']}: {e}")
            return {"candidates": []}

    def run(self, bj_file, lc_file, output_file):
        with open(bj_file, 'r', encoding='utf-8') as f:
            bj_data = [json.loads(line) for line in f]
        with open(lc_file, 'r', encoding='utf-8') as f:
            lc_data = [json.loads(line) for line in f]

        # 리트코드 요약본(Slugs) 생성
        lc_slugs = ", ".join([d['titleSlug'] for d in lc_data])
        
        results = {}
        for i, bj in enumerate(bj_data):
            print(f"Finding matches for BJ {bj['id']} ({i+1}/{len(bj_data)})...")
            matches = self.find_candidates(bj, lc_slugs)
            results[bj['id']] = matches.get("candidates", [])
            time.sleep(0.1)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Potential GT candidates saved to {output_file}")

if __name__ == "__main__":
    finder = GTCandidateFinder()
    finder.run("baekjoon_refined.jsonl", "leetcode_refined.jsonl", "potential_gt.json")