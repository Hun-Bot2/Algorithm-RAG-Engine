import os
import time
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from sklearn.neighbors import NearestNeighbors

# [Configuration]
INDEX_PATH = "./indexes/faiss_leetcode"
POOL_SIZE = 10  # Random pool size
FINAL_K = 3     # Final recommendation count

# OpenAI API Key Dummy Setup
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-loading-class"

def load_faiss_data(index_path):
    """
    Loads the FAISS index and extracts raw vectors.
    """
    print(f"[1] Loading FAISS index from {index_path}...")
    
    if not os.path.exists(index_path):
        print(f"Error: Path {index_path} not found.")
        return None

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.load_local(
            index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        faiss_index = vectorstore.index
        num_vectors = faiss_index.ntotal
        dimension = faiss_index.d
        
        print(f"Index loaded. Total vectors: {num_vectors}, Dimension: {dimension}")
        
        # Reconstruct all vectors from the index
        all_vectors = faiss_index.reconstruct_n(0, num_vectors)
        return vectorstore, all_vectors
        
    except Exception as e:
        print(f"Failed to load index: {e}")
        return None

def compare_performance(faiss_index, all_vectors, query_vector, k=5):
    """
    Compares FAISS, Scikit-learn, and Numpy performance.
    """
    print(f"\n[2] Performance Comparison (Top-{k} Search)")
    print("-" * 60)

    # --- Case A: FAISS ---
    start_time = time.time()
    f_dists, f_inds = faiss_index.search(query_vector, k)
    faiss_duration = time.time() - start_time
    
    print(f"[A] FAISS")
    print(f"Time: {faiss_duration:.6f} sec")
    print(f"Indices: {f_inds[0]}")
    print(f"Scores (L2): {f_dists[0]}")
    print("-" * 60)

    # --- Case B: Scikit-learn ---
    start_time = time.time()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='l2')
    nbrs.fit(all_vectors)
    s_dists, s_inds = nbrs.kneighbors(query_vector)
    sklearn_duration = time.time() - start_time
    
    print(f"[B] Scikit-learn (Brute Force)")
    print(f"Time: {sklearn_duration:.6f} sec")
    print(f"Indices: {s_inds[0]}")
    print(f"Scores (L2): {s_dists[0]}")
    print("-" * 60)

    # --- Case C: Numpy (Pure Python) ---
    start_time = time.time()
    
    # Calculate Squared L2 Distance manually: sum((x-y)^2)
    # Note: FAISS typically returns Squared L2 by default.
    diff = all_vectors - query_vector
    n_dists_squared = np.sum(diff**2, axis=1)
    
    # Sort and pick top K
    sorted_indices = np.argsort(n_dists_squared)[:k]
    sorted_scores = n_dists_squared[sorted_indices]
    
    numpy_duration = time.time() - start_time
    
    print(f"[C] Numpy (Pure Python)")
    print(f"Time: {numpy_duration:.6f} sec")
    print(f"Indices: {sorted_indices}")
    print(f"Scores (Squared L2): {sorted_scores}")
    print("-" * 60)

    # Validation
    match_numpy = np.array_equal(f_inds[0], sorted_indices)
    print(f"FAISS vs Numpy Match: {'✅ YES' if match_numpy else '❌ NO'}")

def search_random_numpy(query_vector, all_vectors, pool_size, final_k):
    """
    Selects top-N candidates and randomly picks k items.
    """
    # 1. Calculate Distances
    diff = all_vectors - query_vector
    dists = np.sum(diff**2, axis=1)
    
    # 2. Get Top-N Pool
    # Use argpartition for efficiency (partial sort)
    if pool_size >= len(dists):
        pool_indices = np.arange(len(dists))
    else:
        partitioned = np.argpartition(dists, pool_size)[:pool_size]
        # Sort within the pool to ensure we have the true top-N
        sorted_indices = partitioned[np.argsort(dists[partitioned])]
        pool_indices = sorted_indices

    # 3. Random Selection from Pool
    selected_indices = np.random.choice(pool_indices, final_k, replace=False)
    
    # Return sorted by score for better readability
    selected_scores = dists[selected_indices]
    
    # Sort the final random selection by score
    final_sort_order = np.argsort(selected_scores)
    return selected_indices[final_sort_order], selected_scores[final_sort_order]

def test_randomization(all_vectors, query_vector):
    """
    Runs the randomization logic multiple times to verify diversity.
    """
    print(f"\n[3] Randomization Test (Top-{POOL_SIZE} Pool -> Pick {FINAL_K})")
    print("-" * 60)
    
    for i in range(1, 4):
        inds, scores = search_random_numpy(query_vector, all_vectors, POOL_SIZE, FINAL_K)
        print(f"[Run {i}] Indices: {inds} / Scores: {scores}")

def main():
    # 1. Load Data
    result = load_faiss_data(INDEX_PATH)
    if not result: return
    
    vectorstore, all_vectors = result
    
    # Create a query vector (using the first vector in the dataset)
    query_vector = all_vectors[0].reshape(1, -1)
    
    # 2. Run Comparison
    compare_performance(vectorstore.index, all_vectors, query_vector, k=5)
    
    # 3. Run Randomization Test
    test_randomization(all_vectors, query_vector)

if __name__ == "__main__":
    main()