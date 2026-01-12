import os
import json
import re
import sys
from tqdm import tqdm
from typing import List, Dict

# LangChain & OpenAI Imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuration
DATA_DIR = os.getenv("DATA_DIR", "./study/docs/Algorithm")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./artifacts")
ARTIFACT_FILE = "recommendation_map.json"
INDEX_PATH = "/app/indexes/faiss_leetcode"  # Docker volume path
MAX_RECOMMENDATIONS = 3

# IDs to exclude
EXCLUDED_IDS = ["algorithm", "intro", "overview", "readme", "template"]

def load_user_problems(base_path: str) -> List[Dict]:
    """
    Scans local Baekjoon markdown files to create query vectors.
    """
    print(f"[INFO] Scanning user repository: {base_path}")
    problems = []
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if not file.lower().endswith((".md", ".mdx")): continue
            
            try:
                path = os.path.join(root, file)
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
                print(f"[WARN] Skipping {file}: {e}")
                
    print(f"[INFO] Found {len(problems)} user problems.")
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
        
    # 2. Load User Problems (Query)
    user_problems = load_user_problems(DATA_DIR)
    if not user_problems:
        print("[WARN] No user problems found.")
        return 0

    # 3. Load FAISS Index
    print(f"[INFO] Loading FAISS Index from {INDEX_PATH}...")
    try:
        # IMPORTANT: Must use the SAME model as indexing
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # allow_dangerous_deserialization=True is required for local pickle files
        vectorstore = FAISS.load_local(
            INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("[INFO] Index loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load index: {e}")
        return 1

    # 4. Retrieval & Generation
    recommendation_map = {}
    print("[INFO] Generating Recommendations...")
    
    for user_prob in tqdm(user_problems):
        # Search
        # k=3: Top 3 similar problems
        docs_and_scores = vectorstore.similarity_search_with_score(user_prob["query_text"], k=MAX_RECOMMENDATIONS)
        
        recs = []
        for doc, score in docs_and_scores:
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
                "similarity": round(float(score), 4), # L2 Distance
                "ai_comment": reason
            })
            
        recommendation_map[user_prob["id"]] = {
            "id": user_prob["id"],
            "title": user_prob["title"],
            "tags": user_prob["tags"],
            "recommendations": recs
        }

    # 5. Save Artifact
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, ARTIFACT_FILE), "w", encoding="utf-8") as f:
        json.dump(recommendation_map, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved recommendation map to {OUTPUT_DIR}/{ARTIFACT_FILE}")
    return 0

if __name__ == "__main__":
    sys.exit(main())