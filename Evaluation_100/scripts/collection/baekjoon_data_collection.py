import requests
from bs4 import BeautifulSoup
import json
import time

class BaekjoonDatasetGenerator:
    def __init__(self):
        self.session = requests.Session()
        self.solve_ac_api = "https://solved.ac/api/v3/problem/show"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def fetch_problem(self, p_id):
        try:
            meta_res = self.session.get(self.solve_ac_api, params={"problemId": p_id}, timeout=10)
            if meta_res.status_code != 200: return None
            meta = meta_res.json()

            content_res = self.session.get(f"https://www.acmicpc.net/problem/{p_id}", headers=self.headers, timeout=10)
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
        except Exception as e:
            print(f"Error at {p_id}: {e}")
            return None

    def generate(self, problem_ids):
        with open("baekjoon_raw_data.jsonl", "w", encoding="utf-8") as f:
            for idx, p_id in enumerate(problem_ids):
                data = self.fetch_problem(p_id)
                if data:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    print(f"[{idx+1}/{len(problem_ids)}] Collected: {data['title']}")
                time.sleep(2.0)

if __name__ == "__main__":
    # 백준 100개 문제 ID 리스트
    target_ids = [
        1003, 1149, 1463, 1912, 1932, 2156, 2579, 9095, 9251, 9461,
        1000, 1001, 1008, 1010, 1018, 11053, 11054, 11726, 11727, 12865,
        1012, 11724, 1260, 1697, 2178, 2468, 2606, 2667, 7562, 7576,
        1753, 1916, 11404, 1197, 1238, 1504, 9370, 13549, 1162, 1865,
        11047, 11399, 13305, 1541, 1931, 2217, 2839, 5585, 1026, 1439,
        1152, 1157, 1316, 2941, 5052, 5430, 9012, 9935, 1259, 1764,
        10828, 10773, 1874, 1966, 11279, 11286, 17298, 2493, 5397, 1406,
        14503, 3190, 14891, 15686, 16236, 17144, 14499, 14502, 14888, 14889,
        1654, 1920, 2110, 2805, 1011, 1929, 1978, 4948, 6064, 1002,
        1181, 11650, 11651, 10814, 10989, 2750, 2751, 1427, 2108, 18870
    ]
    BaekjoonDatasetGenerator().generate(target_ids)