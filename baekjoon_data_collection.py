import requests
from bs4 import BeautifulSoup
import json
import time

class BaekjoonDatasetGenerator:
    def __init__(self):
        self.solve_ac_api = "https://solved.ac/api/v3/problem/show"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def fetch_problem(self, p_id):
        # 1. 메타데이터 (solved.ac)
        meta_res = requests.get(self.solve_ac_api, params={"problemId": p_id})
        if meta_res.status_code != 200: return None
        meta = meta_res.json()

        # 2. 본문 크롤링 (acmicpc.net)
        content_res = requests.get(f"https://www.acmicpc.net/problem/{p_id}", headers=self.headers)
        if content_res.status_code != 200: return None
        soup = BeautifulSoup(content_res.text, "html.parser")
        desc = soup.find("div", {"id": "problem_description"})
        
        return {
            "id": str(p_id),
            "title": meta.get("titleKo"),
            "difficulty": meta.get("level"),
            "tags": [tag["key"] for tag in meta.get("tags", [])],
            "content": desc.get_text(strip=True) if desc else ""
        }

    def generate(self, problem_ids):
        with open("baekjoon_raw_data.jsonl", "w", encoding="utf-8") as f:
            for p_id in problem_ids:
                data = self.fetch_problem(p_id)
                if data:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    print(f"Collected No.{p_id}: {data['title']}")
                time.sleep(1.5) # Rate Limit 준수

if __name__ == "__main__":
    ids = [
        1003, 1149, 1463, 1912, 1932, 2156, 2579, 9095, 9251, 9461,
        1012, 11724, 1260, 1697, 2178, 2468, 2606, 2667, 7562, 7576,
        11047, 11399, 13305, 1541, 1931, 2217, 2839, 5585, 1026, 1439,
        1152, 1157, 1316, 2941, 5052, 5430, 9012, 9935, 1259, 1764,
        1654, 1920, 2110, 2805, 1011, 1929, 1978, 4948, 6064, 1002
    ]
    BaekjoonDatasetGenerator().generate(ids)