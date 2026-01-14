import os
import json
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

SOURCE_INDEX_PATH = "./indexes/faiss_leetcode"
OUTPUT_DIR = "./indexes/numpy_leetcode"

# OpenAI API Key
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "dummy"

def migrate():
    print(f"[1] Loading FAISS index from {SOURCE_INDEX_PATH}...")
    
    if not os.path.exists(SOURCE_INDEX_PATH):
        print(f"Error: Path {SOURCE_INDEX_PATH} not found.")
        return

    try:
        # LangChain FAISS load
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.load_local(
            SOURCE_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Access FAISS index and Docstore
        faiss_index = vectorstore.index
        index_to_docstore_id = vectorstore.index_to_docstore_id
        docstore = vectorstore.docstore
        
        num_vectors = faiss_index.ntotal
        dimension = faiss_index.d
        
        print(f"Stats: {num_vectors} vectors, {dimension} dimensions")
        
        # Extract vectors (Numpy)
        print("Extracting vectors...")
        all_vectors = faiss_index.reconstruct_n(0, num_vectors)
        
        # 2. Extract metadata (JSON)
        # Metadata must be ordered according to FAISS index order (0~N)
        print("Extracting metadata...")
        metadata_list = []
        
        for i in range(num_vectors):
            doc_id = index_to_docstore_id.get(i)
            if not doc_id:
                print(f"[WARN] No doc_id for index {i}")
                metadata_list.append({})
                continue
                
            doc = docstore.search(doc_id)
            # Extract metadata only from LangChain Document object
            metadata_list.append(doc.metadata if doc else {})

        # 3. Save
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save vectors (.npy)
        vec_path = os.path.join(OUTPUT_DIR, "vectors.npy")
        np.save(vec_path, all_vectors)
        
        # Save metadata (.json)
        meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
            
        print(f"\nMigration Complete!")
        print(f"Vectors: {vec_path}")
        print(f"Metadata: {meta_path}")
        
    except Exception as e:
        print(f"Migration failed: {e}")

if __name__ == "__main__":
    migrate()