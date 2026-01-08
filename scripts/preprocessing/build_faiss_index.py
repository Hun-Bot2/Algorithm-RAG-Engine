import os
import sys
import json
import argparse
from pathlib import Path
from typing import List

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.text_processing import create_embedding_text
from src.utils.file_io import read_jsonl

# LangChain / Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

EMBEDDING_MODEL_DEFAULT = "openai"  # options: openai, local
INDEX_DIR = ROOT / "indexes" / "faiss_leetcode"
RAW_FILE = ROOT / "data" / "raw" / "leetcode_raw_data.jsonl"


def get_embedding_client(model: str):
    model = model.lower().strip()
    if model == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    elif model == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        # all-MiniLM-L6-v2 (384-dim) is fast and good enough for local
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError("Unknown embedding model. Use 'openai' or 'local'.")


def load_documents(raw_path: Path) -> List[Document]:
    rows = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    docs: List[Document] = []
    for r in rows:
        title = r.get("title", "")
        content = r.get("content", "")
        tags = r.get("tags", []) or []
        difficulty = r.get("difficulty", "")
        slug = r.get("titleSlug", "")
        url = f"https://leetcode.com/problems/{slug}/" if slug else ""

        embedding_text = create_embedding_text(title=title, content=content, tags=tags, difficulty=difficulty)

        doc = Document(
            page_content=embedding_text,
            metadata={
                "id": str(r.get("id", "")),
                "title": title,
                "slug": slug,
                "difficulty": difficulty,
                "tags": tags,
                "url": url,
            }
        )
        docs.append(doc)
    return docs


def build_index(model: str, batch: int = 256):
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_FILE}")

    embedding = get_embedding_client(model)
    docs = load_documents(RAW_FILE)

    # Build FAISS in batches to reduce peak memory
    index = None
    for i in range(0, len(docs), batch):
        chunk = docs[i:i+batch]
        if index is None:
            index = FAISS.from_documents(chunk, embedding)
        else:
            index.add_documents(chunk)
        print(f"Indexed {min(i+batch, len(docs))}/{len(docs)}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.save_local(str(INDEX_DIR))
    print(f"Saved FAISS index to {INDEX_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for LeetCode dataset")
    parser.add_argument("--model", choices=["openai", "local"], default=EMBEDDING_MODEL_DEFAULT,
                        help="Embedding backend: openai or local (sentence-transformers)")
    parser.add_argument("--batch", type=int, default=256, help="Batch size for indexing")
    args = parser.parse_args()

    # Safety: ensure OPENAI_API_KEY exists for openai mode
    if args.model == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Falling back to local embeddings.")
        args.model = "local"

    build_index(model=args.model, batch=args.batch)


if __name__ == "__main__":
    main()
