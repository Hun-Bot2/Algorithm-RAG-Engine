"""
Embedding Generator for Data Quality Validation
=================================================
Generates embedding vectors for all JSONL files (Raw, Preprocessed, Refined)
to enable Stage 2-4 analysis in visualize.ipynb.

Supports multiple embedding providers:
- Jina AI (jina-embeddings-v3)
- OpenAI (text-embedding-3-small, text-embedding-3-large)

Usage:
    python generate_embeddings.py --provider jina --model jina-embeddings-v3
    python generate_embeddings.py --provider openai --model text-embedding-3-small
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import time

from dotenv import load_dotenv
load_dotenv()

class JinaEmbedding:
    """Jina AI Embedding Generator"""
    
    def __init__(self, model_name: str = "jina-embeddings-v3", api_key: str = None):
        """Initialize Jina embedding model"""
        try:
            import requests
        except ImportError:
            raise ImportError("requests required. Install: pip install requests")
        
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY not found. Set via --api-key or environment variable")
        
        self.api_url = "https://api.jina.ai/v1/embeddings"
        self.embedding_dim = 1024
        print(f"Jina model initialized: {model_name} (dimension: {self.embedding_dim})")
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings using Jina API"""
        import requests
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Jina API"):
            batch = texts[i:i + batch_size]
            
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "input": batch,
                    "encoding_type": "float"
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Jina API error: {response.status_code} - {response.text}")
            
            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            all_embeddings.extend(embeddings)
            
            time.sleep(0.1)
        
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        return embeddings_array / norms


class OpenAIEmbedding:
    """OpenAI Embedding Generator"""
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = None):
        """Initialize OpenAI embedding model"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai required. Install: pip install openai")
        
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found. Set via --api-key or environment variable")
        
        self.client = OpenAI(api_key=self.api_key)
        
        self.embedding_dim = 1536 if "small" in model_name else 3072
        print(f"OpenAI model initialized: {model_name} (dimension: {self.embedding_dim})")
    
    def encode(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="OpenAI API"):
            batch = texts[i:i + batch_size]
            
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                encoding_format="float"
            )
            
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
            
            time.sleep(0.05)
        
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        return embeddings_array / norms


class EmbeddingGenerator:
    """Unified embedding generator supporting multiple providers"""
    
    def __init__(self, provider: str = "jina", model_name: str = None, api_key: str = None):
        """
        Initialize embedding model.
        
        Args:
            provider: 'jina' or 'openai'
            model_name: Model name for the provider
            api_key: API key (optional, can use environment variable)
        """
        self.provider = provider.lower()
        
        if self.provider == "jina":
            model_name = model_name or "jina-embeddings-v3"
            self.model = JinaEmbedding(model_name=model_name, api_key=api_key)
        elif self.provider == "openai":
            model_name = model_name or "text-embedding-3-small"
            self.model = OpenAIEmbedding(model_name=model_name, api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'jina' or 'openai'")
        
        self.model_name = self.model.model_name
        self.embedding_dim = self.model.embedding_dim
        print(f"Embedding generator ready: {self.provider} - {self.model_name}")
    
    
    def extract_text(self, obj: Dict) -> str:
        """Extract best available text field from problem object"""
        for field in ["embedding_text", "content_cleaned", "content", "title"]:
            if field in obj and isinstance(obj[field], str) and obj[field].strip():
                return obj[field].strip()
        return ""
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for text list"""
        return self.model.encode(texts, batch_size=batch_size)
    
    def process_jsonl_file(self, input_path: str, output_path: str = None, batch_size: int = 32, field_suffix: str = ""):
        """
        Load JSONL, generate embeddings, and save with embedding field.
        
        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL (overwrites input if None)
            batch_size: Encoding batch size
            field_suffix: Suffix for embedding field name (e.g., '_jina', '_openai')
        """
        print(f"\nProcessing: {Path(input_path).name}")
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        
        print(f"   Loaded {len(data)} items")
        
        texts = [self.extract_text(obj) for obj in data]
        
        embedding_field = f"embedding{field_suffix}"
        has_embeddings = any(embedding_field in obj for obj in data)
        if has_embeddings:
            print(f"   Warning: {embedding_field} already exists, regenerating...")
        
        print(f"   Generating embeddings (batch_size={batch_size})...")
        embeddings = self.generate_embeddings(texts, batch_size=batch_size)
        
        for i, obj in enumerate(data):
            obj[embedding_field] = embeddings[i].tolist()
            obj[f'{embedding_field}_model'] = self.model_name
            obj[f'{embedding_field}_dim'] = len(embeddings[i])
        
        output_path = output_path or input_path
        with open(output_path, 'w', encoding='utf-8') as f:
            for obj in data:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        
        print(f"   Saved {len(data)} items to {Path(output_path).name}")
        print(f"   Embedding shape: ({len(data)}, {embeddings.shape[1]})")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for JSONL files")
    parser.add_argument(
        '--provider',
        type=str,
        required=True,
        choices=['jina', 'openai'],
        help='Embedding provider: jina or openai'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name (default: jina-embeddings-v3 for jina, text-embedding-3-small for openai)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key (or set JINA_API_KEY / OPENAI_API_KEY environment variable)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for encoding (default: 32)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/Users/jeonghun/Algorithm-RAG-Engine/Evaluation_100/data',
        help='Directory containing JSONL files'
    )
    
    args = parser.parse_args()
    
    generator = EmbeddingGenerator(
        provider=args.provider,
        model_name=args.model,
        api_key=args.api_key
    )
    
    data_dir = Path(args.data_dir)
    files = [
        'baekjoon_raw_data.jsonl',
        'baekjoon_preprocessed.jsonl',
        'baekjoon_refined.jsonl',
        'leetcode_raw_data.jsonl',
        'leetcode_preprocessed.jsonl',
        'leetcode_refined.jsonl',
    ]
    
    field_suffix = f"_{args.provider}"
    
    print("\n" + "=" * 80)
    print("EMBEDDING GENERATION PIPELINE")
    print("=" * 80)
    print(f"Provider: {args.provider}")
    print(f"Model: {generator.model_name}")
    print(f"Dimension: {generator.embedding_dim}")
    print(f"Data Directory: {data_dir}")
    print(f"Files to process: {len(files)}")
    print(f"Embedding field: embedding{field_suffix}")
    print("=" * 80)
    
    success_count = 0
    for filename in files:
        filepath = data_dir / filename
        
        if not filepath.exists():
            print(f"\nSKIPPED: {filename} (not found)")
            continue
        
        try:
            generator.process_jsonl_file(
                str(filepath),
                batch_size=args.batch_size,
                field_suffix=field_suffix
            )
            success_count += 1
        except Exception as e:
            print(f"\nERROR processing {filename}: {e}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully processed: {success_count}/{len(files)} files")
    print(f"Ready for model comparison in visualize.ipynb")
    print("=" * 80)


if __name__ == "__main__":
    main()
