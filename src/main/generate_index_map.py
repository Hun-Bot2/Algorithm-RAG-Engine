import os
import json
import re
import sys
from tqdm import tqdm
from typing import List, Dict

# GitPython Import
from git import Repo

# LangChain & OpenAI Imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuration
# Docker volume mount point for the repository root
REPO_ROOT = os.getenv("REPO_ROOT", "/app/repo")
# Relative path from REPO_ROOT to the docs folder
TARGET_SUBDIR = os.getenv("TARGET_SUBDIR", "study/docs/Algorithm")

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./artifacts")
ARTIFACT_FILE = "recommendation_map.json"
INDEX_PATH = "/app/indexes/faiss_leetcode"
MAX_RECOMMENDATIONS = 3

# IDs to exclude
EXCLUDED_IDS = ["algorithm", "intro", "overview", "readme", "template"]

def get_latest_changed_files(repo_path: str, target_subdir: str) -> List[str]:
    """
    Detects files changed in the latest commit (HEAD) within the target subdirectory.
    """
    print(f"[INFO] Checking git history in: {repo_path}")
    changed_files = []
    
    try:
        repo = Repo(repo_path)
        
        # Ensure we have commits to compare
        if not repo.head.is_valid():
            print("[WARN] No valid HEAD found (empty repo?). Scanning all files.")
            return get_all_files(repo_path, target_subdir)

        head_commit = repo.head.commit
        
        # If no parents (first commit), scan all files in the tree
        if not head_commit.parents:
            print("[INFO] First commit detected. Scanning all files.")
            for item in head_commit.tree.traverse():
                if item.path.startswith(target_subdir) and item.path.endswith((".md", ".mdx")):
                    full_path = os.path.join(repo_path, item.path)
                    changed_files.append(full_path)
        else:
            # Compare HEAD with HEAD~1 (Previous commit)
            parent = head_commit.parents[0]
            diffs = parent.diff(head_commit)
            
            for diff in diffs:
                # We only care about added (A) or modified (M) files
                # diff.b_path is the new path. If deleted, it might be None or verify deleted_file flag
                if diff.b_path and not diff.deleted_file:
                    if diff.b_path.startswith(target_subdir) and diff.b_path.endswith((".md", ".mdx")):
                        full_path = os.path.join(repo_path, diff.b_path)
                        changed_files.append(full_path)

    except Exception as e:
        print(f"[ERROR] Git processing failed: {e}")
        print("[INFO] Falling back to scanning all files.")
        return get_all_files(repo_path, target_subdir)

    return changed_files

def get_all_files(repo_path: str, target_subdir: str) -> List[str]:
    """
    Fallback function: Get all .md/.mdx files if Git logic fails or first run.
    """
    all_files = []
    base_path = os.path.join(repo_path, target_subdir)
    
    if not os.path.exists(base_path):
        print(f"[WARN] Path does not exist: {base_path}")
        return []

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith((".md", ".mdx")):
                all_files.append(os.path.join(root, file))
    return all_files

def load_user_problems(file_paths: List[str]) -> List[Dict]:
    """
    Parses specific markdown files to create query vectors.
    Changed from scanning directory to processing a list of files.
    """
    print(f"[INFO] Parsing {len(file_paths)} files...")
    problems = []
    
    for path in file_paths:
        if not os.path.exists(path):
            continue
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Simple Frontmatter Parsing
            match = re.search(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
            if not match: continue
            
            meta = {}
            for line in match.group(1).splitlines():
                if ':' in line:
                    k, v = line.split(':', 1)
                    meta[k.strip()] = v.strip()
            
            if "id" not in meta: continue
            if any(ex in meta["id"].lower() for ex in EXCLUDED_IDS): continue
            
            # Tags processing
            tag_str = meta.get("tags", "[]").strip("[]")
            tags = [t.strip() for t in tag_str.split(",") if t.strip()]
            
            # Construct Query Text (Title + Tags)
            query_text = f"Title: {meta.get('title', 'Unknown')}. Tags: {', '.join(tags)}"
            
            problems.append({
                "id": meta["id"],
                "title": meta.get("title", "Unknown"),
                "tags": tags,
                "query_text": query_text
            })
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
                
    print(f"[INFO] Successfully parsed {len(problems)} user problems.")
    return problems

def generate_reasoning(user_title: str, rec_title: str, tags: list) -> str:
    """
    Generates a short reason why this problem is recommended using GPT-4o-mini.
    """
    try:
        if not os.getenv("OPENAI_API_KEY"):
            return "AI explanation unavailable (Missing API Key)."

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
        return chain.invoke({
            "user_title": user_title,
            "rec_title": rec_title,
            "tags": ", ".join(tags)
        })
    except Exception:
        return "유사한 알고리즘 유형입니다."

def main():
    # 1. Check Index Existence
    if not os.path.exists(INDEX_PATH):
        print(f"[ERROR] Index not found at {INDEX_PATH}")
        print("Did you mount the volume correctly in docker-compose.yml?")
        return 1
    
    # 2. Get Changed Files (Git Logic)
    # This logic reduces processing time by ignoring unchanged files.
    changed_files = get_latest_changed_files(REPO_ROOT, TARGET_SUBDIR)
    
    if not changed_files:
        print("[INFO] No .md/.mdx files changed in the latest commit.")
        print("[INFO] Exiting cleanly (Nothing to process).")
        return 0
        
    # 3. Load User Problems (Query) from changed files only
    user_problems = load_user_problems(changed_files)
    if not user_problems:
        print("[WARN] Changed files found, but no valid problem metadata extracted.")
        return 0

    # 4. Load FAISS Index
    print(f"[INFO] Loading FAISS Index from {INDEX_PATH}...")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.load_local(
            INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("[INFO] Index loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load index: {e}")
        return 1

    # 5. Retrieval & Generation
    recommendation_map = {}
    print(f"[INFO] Generating Recommendations for {len(user_problems)} problems...")
    
    for user_prob in tqdm(user_problems):
        # Search
        search_limit = MAX_RECOMMENDATIONS + 2  # Buffer to avoid self-match
        docs_and_scores = vectorstore.similarity_search_with_score(
            user_prob["query_text"],
            k=search_limit
        )
        
        recs = []

        # Avoid Duplicates & Self-Match
        seen_title= set()
        
        user_title_clean=re.sub(r'[^a-zA-Z0-9가-힣]', '', user_prob["title"].lower())

        for doc, score in docs_and_scores:
            if len(recs) >= MAX_RECOMMENDATIONS:
                break

            rec_title=doc.metadata.get("title", "Unknown")
            rec_title_clean=re.sub(r'[^a-zA-Z0-9가-힣]', '', rec_title.lower())

            if user_title_clean in rec_title_clean or rec_title_clean in user_title_clean:
                continue  # Skip self-match
            if rec_title_clean in seen_title:
                continue  # Skip duplicates

            seen_title.add(rec_title_clean)
            
            # Generate AI Comment
            reason = generate_reasoning(
                user_prob["title"], 
                doc.metadata.get("title", "Unknown"), 
                user_prob["tags"]
            )
            
            recs.append({
                "id": doc.metadata.get("id", "Unknown"),
                "title": doc.metadata.get("title", "Unknown"),
                "difficulty": doc.metadata.get("difficulty", "Unknown"),
                "similarity": round(float(score), 4),
                "ai_comment": reason
            })
            
        recommendation_map[user_prob["id"]] = {
            "id": user_prob["id"],
            "title": user_prob["title"],
            "tags": user_prob["tags"],
            "recommendations": recs
        }

    # 6. Save Artifact
    # Note: We append/update the map if you want to keep history, 
    # but here we overwrite for the Light Job to consume just the new updates.
    # The Light Job should handle sending messages based on this file.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, ARTIFACT_FILE), "w", encoding="utf-8") as f:
        json.dump(recommendation_map, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved recommendation map to {OUTPUT_DIR}/{ARTIFACT_FILE}")
    return 0

if __name__ == "__main__":
    sys.exit(main())