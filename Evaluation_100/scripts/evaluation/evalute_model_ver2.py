import os
import json
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

class AdvancedEvaluator:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.jina_api_key = os.getenv("JINA_API_KEY")
        print("BGE-M3 로컬 모델 로딩 중...")
        self.bge_model = SentenceTransformer('BAAI/bge-m3')

    def get_openai_embeddings(self, texts):
        res = self.openai_client.embeddings.create(input=texts, model="text-embedding-3-small")
        return np.array([d.embedding for d in res.data])

    def get_jina_embeddings(self, texts):
        url = "https://api.jina.ai/v1/embeddings"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.jina_api_key}"}
        # 쿼리와 지문의 형식이 유사하므로 둘 다 passage로 처리하여 정렬 개선
        data = {
            "model": "jina-embeddings-v3",
            "task": "retrieval.passage",
            "dimensions": 1024,
            "input": texts
        }
        res = requests.post(url, headers=headers, json=data)
        return np.array([item['embedding'] for item in res.json()['data']])

    def get_bge_embeddings(self, texts):
        return self.bge_model.encode(texts, batch_size=12, show_progress_bar=False)

    def calculate_metrics(self, q_embs, d_embs, queries, docs, gt_mapping):
        sim_matrix = cosine_similarity(q_embs, d_embs)
        results = {"Recall@1": 0, "Recall@5": 0, "Recall@10": 0, "MRR": 0}
        
        lc_slug_map = {doc["titleSlug"]: idx for idx, doc in enumerate(docs)}
        valid_queries = 0

        for i, q in enumerate(queries):
            bj_id = str(q["id"])
            if bj_id not in gt_mapping: continue
            
            target_slug = gt_mapping[bj_id]
            if target_slug not in lc_slug_map: continue
            
            valid_queries += 1
            sorted_idx = np.argsort(sim_matrix[i])[::-1]
            
            rank = 999
            for r, idx in enumerate(sorted_idx):
                if docs[idx]["titleSlug"] == target_slug:
                    rank = r + 1
                    break
            
            if rank <= 1: results["Recall@1"] += 1
            if rank <= 5: results["Recall@5"] += 1
            if rank <= 10: results["Recall@10"] += 1
            if rank <= 100: results["MRR"] += 1 / rank

        if valid_queries > 0:
            for k in results: results[k] /= valid_queries
            
        return results, valid_queries

    def run(self, bj_file, lc_file, gt_file):
        with open(bj_file, 'r', encoding='utf-8') as f:
            bj_data = [json.loads(line) for line in f]
        with open(lc_file, 'r', encoding='utf-8') as f:
            lc_data = [json.loads(line) for line in f]
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_mapping = json.load(f)

        bj_texts = [item["embedding_text"] for item in bj_data]
        lc_texts = [item["embedding_text"] for item in lc_data]

        report = {}
        models = [
            ("OpenAI-v3", self.get_openai_embeddings),
            ("Jina-v3", self.get_jina_embeddings),
            ("BGE-M3", self.get_bge_embeddings)
        ]

        for name, embed_func in models:
            print(f"Evaluating {name}...")
            q_embs = embed_func(bj_texts)
            d_embs = embed_func(lc_texts)
            res, count = self.calculate_metrics(q_embs, d_embs, bj_data, lc_data, gt_mapping)
            report[name] = res
            print(f"Completed {name} with {count} samples.")

        print("\n=== Final 100-Set Benchmarking Report ===")
        print(json.dumps(report, indent=4))

if __name__ == "__main__":
    evaluator = AdvancedEvaluator()
    evaluator.run("baekjoon_refined.jsonl", "leetcode_refined.jsonl", "ground_truth_v2.json")