import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# 환경 변수 로드
load_dotenv()

class RAGEvaluator:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.jina_api_key = os.getenv("JINA_API_KEY")
        
        print("BGE-M3 모델 로드 중 (로컬)...")
        self.bge_model = SentenceTransformer('BAAI/bge-m3')
        
        # 시각화 한글 폰트 설정 (Mac/Linux 대응)
        plt.rcParams['font.family'] = 'AppleGothic' if os.path.exists('/Library/Fonts/AppleGothic.ttf') else 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False

    def get_openai_embeddings(self, texts):
        # OpenAI 임베딩 생성 (v3-small)
        response = self.openai_client.embeddings.create(
            input=texts, 
            model="text-embedding-3-small"
        )
        return np.array([d.embedding for d in response.data])

    def get_jina_embeddings(self, texts, is_query=True):
        # Jina v3 임베딩 생성 (Task 설정 주의)
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.jina_api_key}"
        }
        # 백준 요약본이 길 경우 retrieval.query보다 retrieval.passage가 더 적합할 수 있음
        task = "retrieval.query" if is_query else "retrieval.passage"
        data = {
            "model": "jina-embeddings-v3",
            "task": task,
            "dimensions": 1024,
            "input": texts
        }
        res = requests.post(url, headers=headers, json=data)
        if res.status_code != 200:
            print(f"Jina API Error: {res.text}")
            return np.zeros((len(texts), 1024))
        return np.array([item['embedding'] for item in res.json()['data']])

    def get_bge_embeddings(self, texts):
        # BGE-M3 Dense 임베딩 생성
        return self.bge_model.encode(texts, batch_size=12, show_progress_bar=False)

    def diagnostic_check(self, query_embs, doc_embs, bj_data, lc_data, gt_mapping, model_name):
        # 데이터 정합성 및 유사도 분포 진단
        print(f"\n--- [{model_name}] Diagnostic Report ---")
        
        lc_slug_map = {doc["titleSlug"]: idx for idx, doc in enumerate(lc_data)}
        found_count = 0
        total_gt = len(gt_mapping)
        
        similarities_to_gt = []
        similarities_to_top1 = []

        for i, q in enumerate(bj_data):
            bj_id = str(q["id"])
            if bj_id not in gt_mapping:
                continue
            
            target_slug = gt_mapping[bj_id]
            if target_slug not in lc_slug_map:
                # DB에 정답 슬러그가 없는 경우 경고
                continue
            
            found_count += 1
            target_idx = lc_slug_map[target_slug]
            
            # 유사도 계산
            sims = cosine_similarity(query_embs[i:i+1], doc_embs)[0]
            sorted_indices = np.argsort(sims)[::-1]
            
            gt_sim = sims[target_idx]
            top1_idx = sorted_indices[0]
            top1_sim = sims[top1_idx]
            
            similarities_to_gt.append(gt_sim)
            similarities_to_top1.append(top1_sim)
            
            if i < 5: # 상위 5개 샘플만 상세 출력
                print(f"BJ {bj_id} -> GT: {target_slug} (Sim: {gt_sim:.4f}) | Top1: {lc_data[top1_idx]['titleSlug']} (Sim: {top1_sim:.4f})")

        print(f"GT Connectivity: {found_count}/{total_gt} (DB 내 정답 존재 비율)")
        if similarities_to_gt:
            print(f"Average GT Similarity: {np.mean(similarities_to_gt):.4f}")
            print(f"Average Top-1 Similarity: {np.mean(similarities_to_top1):.4f}")

    def plot_tsne(self, query_embs, doc_embs, bj_data, lc_data, model_name):
        # 임베딩 공간 시각화 (데이터 유사도 검증)
        all_embs = np.vstack([query_embs, doc_embs])
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embs)-1))
        reduced = tsne.fit_transform(all_embs)
        
        plt.figure(figsize=(10, 7))
        n_q = len(query_embs)
        
        plt.scatter(reduced[:n_q, 0], reduced[:n_q, 1], label="Baekjoon (Query)", alpha=0.6, c='blue')
        plt.scatter(reduced[n_q:, 0], reduced[n_q:, 1], label="LeetCode (Doc)", alpha=0.6, c='red', marker='x')
        
        plt.title(f"t-SNE Alignment: {model_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"tsne_{model_name.lower()}.png")
        print(f"t-SNE chart saved: tsne_{model_name.lower()}.png")

    def calculate_metrics(self, query_embs, doc_embs, queries, docs, gt_mapping, k_list=[1, 5, 10]):
        sim_matrix = cosine_similarity(query_embs, doc_embs)
        results = {f"Recall@{k}": 0 for k in k_list}
        results["MRR"] = 0
        
        valid_queries = 0
        lc_slug_map = {doc["titleSlug"]: idx for idx, doc in enumerate(docs)}

        for i, q in enumerate(queries):
            bj_id = str(q["id"])
            if bj_id not in gt_mapping:
                continue
            
            target_slug = gt_mapping[bj_id]
            if target_slug not in lc_slug_map:
                continue
            
            valid_queries += 1
            sorted_idx = np.argsort(sim_matrix[i])[::-1]
            
            rank = 999
            for r, idx in enumerate(sorted_idx):
                if docs[idx]["titleSlug"] == target_slug:
                    rank = r + 1
                    break
            
            for k in k_list:
                if rank <= k:
                    results[f"Recall@{k}"] += 1
            if rank <= 100:
                results["MRR"] += 1 / rank

        if valid_queries == 0:
            return results

        for k in k_list:
            results[f"Recall@{k}"] /= valid_queries
        results["MRR"] /= valid_queries
        return results

    def run_evaluation(self, bj_file, lc_file, gt_file):
        # 데이터 로드
        with open(bj_file, 'r', encoding='utf-8') as f:
            bj_data = [json.loads(line) for line in f]
        with open(lc_file, 'r', encoding='utf-8') as f:
            lc_data = [json.loads(line) for line in f]
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_mapping = json.load(f)

        # 텍스트 추출 (Logical Skeleton 접두사 일관성 체크 필요)
        bj_texts = [item["embedding_text"] for item in bj_data]
        lc_texts = [item["embedding_text"] for item in lc_data]

        final_report = {}

        # 1. OpenAI 평가
        print("\n평가 시작: OpenAI v3-small")
        q_oa = self.get_openai_embeddings(bj_texts)
        d_oa = self.get_openai_embeddings(lc_texts)
        self.diagnostic_check(q_oa, d_oa, bj_data, lc_data, gt_mapping, "OpenAI")
        self.plot_tsne(q_oa, d_oa, bj_data, lc_data, "OpenAI")
        final_report["OpenAI"] = self.calculate_metrics(q_oa, d_oa, bj_data, lc_data, gt_mapping)

        # 2. Jina v3 평가
        print("\n평가 시작: Jina v3")
        q_jina = self.get_jina_embeddings(bj_texts, is_query=True)
        d_jina = self.get_jina_embeddings(lc_texts, is_query=False)
        self.diagnostic_check(q_jina, d_jina, bj_data, lc_data, gt_mapping, "Jina_v3")
        final_report["Jina_v3"] = self.calculate_metrics(q_jina, d_jina, bj_data, lc_data, gt_mapping)

        # 3. BGE-M3 평가
        print("\n평가 시작: BGE-M3")
        q_bge = self.get_bge_embeddings(bj_texts)
        d_bge = self.get_bge_embeddings(lc_texts)
        self.diagnostic_check(q_bge, d_bge, bj_data, lc_data, gt_mapping, "BGE_M3")
        final_report["BGE_M3"] = self.calculate_metrics(q_bge, d_bge, bj_data, lc_data, gt_mapping)

        print("\n=== 최종 평가 리포트 ===")
        print(json.dumps(final_report, indent=4))

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    evaluator.run_evaluation(
        "baekjoon_refined.jsonl",
        "leetcode_refined.jsonl",
        "ground_truth.json"
    )