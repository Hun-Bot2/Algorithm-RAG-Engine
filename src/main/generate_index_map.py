import os
import json
import re
import sys
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple

# GitPython Import
from git import Repo

# OpenAI Imports (Only for Embedding & Chat, FAISS removed)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuration
REPO_ROOT = os.getenv("REPO_ROOT", "/app/repo")
TARGET_SUBDIR = os.getenv("TARGET_SUBDIR", "study/docs/Algorithm/Baekjoon")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./artifacts")
ARTIFACT_FILE = "recommendation_map.json"

# Changed: Point to new Numpy Index Directory
INDEX_DIR = "/app/indexes/numpy_leetcode" 

# Recommendation Config
MAX_RECOMMENDATIONS = 3
POOL_SIZE = 10  # Random selection pool

EXCLUDED_IDS = ["algorithm", "intro", "overview", "readme", "template"]

def get_latest_changed_files(repo_path: str, target_subdir: str) -> List[str]:
    """Detects files changed in the latest commit."""
    print(f"[INFO] Checking git history in: {repo_path}")
    changed_files = []
    try:
        repo = Repo(repo_path)
        if not repo.head.is_valid():
            return get_all_files(repo_path, target_subdir)

        head_commit = repo.head.commit
        if not head_commit.parents:
            for item in head_commit.tree.traverse():
                if item.path.startswith(target_subdir) and item.path.endswith((".md", ".mdx")):
                    changed_files.append(os.path.join(repo_path, item.path))
        else:
            parent = head_commit.parents[0]
            diffs = parent.diff(head_commit)
            for diff in diffs:
                if diff.b_path and not diff.deleted_file:
                    if diff.b_path.startswith(target_subdir) and diff.b_path.endswith((".md", ".mdx")):
                        changed_files.append(os.path.join(repo_path, diff.b_path))
    except Exception as e:
        print(f"[ERROR] Git processing failed: {e}")
        return get_all_files(repo_path, target_subdir)
    return changed_files

def get_all_files(repo_path: str, target_subdir: str) -> List[str]:
    """Fallback: Get all .md files."""
    all_files = []
    base_path = os.path.join(repo_path, target_subdir)
    if not os.path.exists(base_path): return []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith((".md", ".mdx")):
                all_files.append(os.path.join(root, file))
    return all_files

def load_user_problems(file_paths: List[str]) -> List[Dict]:
    """Parses markdown files to create query vectors."""
    print(f"[INFO] Parsing {len(file_paths)} files...")
    problems = []
    for path in file_paths:
        if not os.path.exists(path): continue
        try:
            with open(path, "r", encoding="utf-8") as f: content = f.read()
            match = re.search(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
            if not match: continue
            
            meta = {}
            for line in match.group(1).splitlines():
                if ':' in line:
                    k, v = line.split(':', 1)
                    meta[k.strip()] = v.strip()
            
            if "id" not in meta: continue
            if any(ex in meta["id"].lower() for ex in EXCLUDED_IDS): continue
            
            tag_str = meta.get("tags", "[]").strip("[]")
            tags = [t.strip() for t in tag_str.split(",") if t.strip()]
            
            problems.append({
                "id": meta["id"],
                "title": meta.get("title", "Unknown"),
                "tags": tags,
                "query_text": f"Title: {meta.get('title', 'Unknown')}. Tags: {', '.join(tags)}"
            })
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
    return problems

def load_numpy_index(index_dir: str) -> Tuple[np.ndarray, List[Dict]]:
    """Loads vectors and metadata from Numpy/JSON files."""
    vec_path = os.path.join(index_dir, "vectors.npy")
    meta_path = os.path.join(index_dir, "metadata.json")
    
    if not os.path.exists(vec_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Index files not found.")
        
    print(f"[INFO] Loading Numpy index from {index_dir}...")
    vectors = np.load(vec_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    print(f"[INFO] Loaded {len(vectors)} vectors.")
    return vectors, metadata

def search_random_numpy(query_vector: np.ndarray, all_vectors: np.ndarray, 
                       metadata: List[Dict], k: int = 3, pool_size: int = 10) -> List[Dict]:
    """
    Performs similarity search with random selection from a top-N pool.
    """
    # 1. Calculate Squared L2 Distance
    # (x-y)^2 -> Sum
    diff = all_vectors - query_vector
    dists = np.sum(diff**2, axis=1)
    
    # 2. Top-N Pool Selection
    # If pool_size is larger than data, use all data
    actual_pool_size = min(pool_size, len(dists))
    
    # argpartition allows partial sorting (faster than full sort)
    partitioned_indices = np.argpartition(dists, actual_pool_size)[:actual_pool_size]
    
    # Sort within the pool to ensure we have the best N candidates
    sorted_pool_indices = partitioned_indices[np.argsort(dists[partitioned_indices])]
    
    # 3. Random Selection from Pool (Diversity)
    # Pick k items randomly from the sorted pool
    if len(sorted_pool_indices) < k:
        selected_indices = sorted_pool_indices
    else:
        selected_indices = np.random.choice(sorted_pool_indices, k, replace=False)
    
    # 4. Map back to metadata and scores
    results = []
    for idx in selected_indices:
        score = dists[idx] # Squared L2 score
        meta = metadata[idx]
        results.append({
            "meta": meta,
            "score": float(score)
        })
        
    # Sort final results by score (closest first)
    results.sort(key=lambda x: x["score"])
    return results

def generate_reasoning(user_title: str, rec_title: str, tags: list) -> str:
    """Generates AI reasoning using GPT-4o-mini."""
    try:
        if not os.getenv("OPENAI_API_KEY"): return "AI explanation unavailable."
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        prompt = ChatPromptTemplate.from_template(
            """
            You are an Algorithm Tutor.
            User solved: "{user_title}" (Tags: {tags}).
            Recommendation: "{rec_title}".
            Briefly explain in Korean why this is a good follow-up problem.
            (One sentence only. Focus on the algorithmic concept.)
            """
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"user_title": user_title, "rec_title": rec_title, "tags": ", ".join(tags)})
    except Exception:
        return "유사한 알고리즘 유형입니다."

def main():
    # 1. Git Logic
    changed_files = get_latest_changed_files(REPO_ROOT, TARGET_SUBDIR)
    if not changed_files:
        print("[INFO] No files changed. Exiting.")
        return 0
        
    # 2. Parse User Problems
    user_problems = load_user_problems(changed_files)
    if not user_problems: return 0

    # 3. Load Index (Numpy)
    try:
        index_vectors, index_metadata = load_numpy_index(INDEX_DIR)
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    # 4. Generate Recommendations
    recommendation_map = {}
    print(f"[INFO] Processing {len(user_problems)} problems...")
    
    for user_prob in tqdm(user_problems):
        # Embed query
        query_vec = np.array([embeddings_model.embed_query(user_prob["query_text"])])
        
        # Search (Randomized)
        # Fetch slightly more to handle duplicates filtering
        raw_recs = search_random_numpy(
            query_vec, 
            index_vectors, 
            index_metadata, 
            k=MAX_RECOMMENDATIONS + 2, 
            pool_size=POOL_SIZE
        )
        
        recs = []
        seen_title = set()
        user_title_clean = re.sub(r'[^a-zA-Z0-9가-힣]', '', user_prob["title"].lower())

        for item in raw_recs:
            if len(recs) >= MAX_RECOMMENDATIONS: break
            
            meta = item["meta"]
            score = item["score"]
            rec_title = meta.get("title", "Unknown")
            rec_title_clean = re.sub(r'[^a-zA-Z0-9가-힣]', '', rec_title.lower())

            if user_title_clean in rec_title_clean or rec_title_clean in user_title_clean: continue
            if rec_title_clean in seen_title: continue

            seen_title.add(rec_title_clean)
            
            reason = generate_reasoning(user_prob["title"], rec_title, user_prob["tags"])
            
            recs.append({
                "id": meta.get("id", "Unknown"),
                "title": rec_title,
                "difficulty": meta.get("difficulty", "Unknown"),
                "similarity": round(score, 4),
                "ai_comment": reason
            })
            
        recommendation_map[user_prob["id"]] = {
            "id": user_prob["id"],
            "title": user_prob["title"],
            "tags": user_prob["tags"],
            "recommendations": recs
        }

    # 5. Save Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, ARTIFACT_FILE), "w", encoding="utf-8") as f:
        json.dump(recommendation_map, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved to {OUTPUT_DIR}/{ARTIFACT_FILE}")
    return 0

if __name__ == "__main__":
    sys.exit(main())