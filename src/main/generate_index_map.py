import json
import os
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.rag.faiss_recommendation_engine import FAISSRecommendationEngine
from src.utils.file_io import load_jsonl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecommendationMapGenerator:
    """
    Heavy Job: Pre-compute recommendation map using FAISS index
    
    This runs once per day/week and generates a static JSON file
    containing recommendations for all Baekjoon problems.
    """
    
    def __init__(self, 
                 baekjoon_data_path: str = "Evaluation_100/data/baekjoon_refined.jsonl",
                 faiss_index_dir: str = "indexes/faiss_production",
                 output_dir: str = "/app/artifacts"):
        """
        Args:
            baekjoon_data_path: Path to refined Baekjoon problems
            faiss_index_dir: Path to FAISS index directory
            output_dir: Output directory for artifacts (must be /app/artifacts for Docker)
        """
        self.baekjoon_data_path = baekjoon_data_path
        self.faiss_index_dir = faiss_index_dir
        self.output_dir = output_dir
        
        # Initialize FAISS engine
        logger.info("Initializing FAISS Recommendation Engine...")
        self.engine = FAISSRecommendationEngine(
            index_dir=self.faiss_index_dir,
            model_type="openai",
            llm_reranking=True  # Use LLM for quality QA
        )
        
        logger.info("✓ FAISS Engine initialized")
    
    def load_baekjoon_problems(self) -> List[Dict]:
        """Load refined Baekjoon problems"""
        logger.info(f"Loading Baekjoon problems from {self.baekjoon_data_path}...")
        
        try:
            problems = load_jsonl(self.baekjoon_data_path)
            logger.info(f"✓ Loaded {len(problems)} Baekjoon problems")
            return problems
        except FileNotFoundError:
            logger.error(f"Baekjoon data file not found: {self.baekjoon_data_path}")
            return []
    
    def generate_recommendations(self, baekjoon_problem: Dict, top_k: int = 5) -> List[Dict]:
        """
        Generate LeetCode recommendations for a single Baekjoon problem
        
        Args:
            baekjoon_problem: Baekjoon problem dict
            top_k: Number of recommendations to generate
        
        Returns:
            List of recommended LeetCode problems with scores
        """
        try:
            # Create query text from problem description
            query_text = f"{baekjoon_problem.get('title', '')} {baekjoon_problem.get('description', '')}"
            
            # Get embedding from FAISS engine
            query_embedding = self.engine.get_embedding(query_text)
            
            # Search similar problems
            candidates = self.engine.search_similar(query_embedding, k=top_k * 2)
            
            # Rerank with LLM (for quality assurance)
            reranked = self.engine.rerank_with_llm(candidates, query_text)
            
            # Format recommendations
            recommendations = []
            for rank, problem in enumerate(reranked[:top_k], 1):
                rec = {
                    "rank": rank,
                    "leetcode_id": problem.get('problem_id'),
                    "title": problem.get('title'),
                    "difficulty": problem.get('difficulty'),
                    "tags": problem.get('tags', []),
                    "l2_distance": problem.get('l2_distance'),
                    "llm_score": problem.get('llm_score', 0.0),
                    "llm_reason": problem.get('llm_reason', '')
                }
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for {baekjoon_problem.get('problem_id')}: {e}")
            return []
    
    def generate_map(self) -> Dict:
        """
        Main method: Generate complete recommendation map
        
        Returns:
            recommendation_map: {baekjoon_id -> {metadata + recommendations}}
        """
        logger.info("[Heavy Job] Starting Pre-computation of Recommendation Map...")
        
        # Load Baekjoon problems
        baekjoon_problems = self.load_baekjoon_problems()
        if not baekjoon_problems:
            logger.error("No Baekjoon problems loaded. Aborting.")
            return {}
        
        # Generate recommendation map
        recommendation_map = {}
        total = len(baekjoon_problems)
        
        for idx, problem in enumerate(baekjoon_problems, 1):
            problem_id = problem.get('problem_id')
            
            if idx % 10 == 0 or idx == 1:
                logger.info(f"Processing [{idx}/{total}] {problem_id}...")
            
            # Generate recommendations for this problem
            recommendations = self.generate_recommendations(problem, top_k=5)
            
            # Build map entry
            map_entry = {
                "baekjoon_id": problem_id,
                "title": problem.get('title'),
                "difficulty": problem.get('difficulty'),
                "description": problem.get('description', '')[:500],  # Truncate for size
                "tags": problem.get('tags', []),
                "recommendations": recommendations,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            recommendation_map[problem_id] = map_entry
        
        logger.info(f"✓ Generated recommendations for {len(recommendation_map)} problems")
        return recommendation_map
    
    def save_artifacts(self, recommendation_map: Dict):
        """
        Save artifacts to /app/artifacts directory
        
        Important: Docker volume mount maps /app/artifacts to host ./artifacts
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save recommendation map
        map_path = os.path.join(self.output_dir, "recommendation_map.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(recommendation_map, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved recommendation_map.json ({len(recommendation_map)} entries)")
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        metadata = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_problems": len(recommendation_map),
            "index_info": self.engine.index_info
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved metadata.json")
        
        logger.info(f"✓ All artifacts saved to {self.output_dir}")


def main():
    """Main entry point for Docker container"""
    
    # Initialize generator
    generator = RecommendationMapGenerator()
    
    # Generate recommendation map
    recommendation_map = generator.generate_map()
    
    # Save artifacts
    if recommendation_map:
        generator.save_artifacts(recommendation_map)
        logger.info("[Heavy Job] Pre-computation completed successfully!")
    else:
        logger.error("[Heavy Job] Failed to generate recommendation map")
        sys.exit(1)


if __name__ == "__main__":
    main()