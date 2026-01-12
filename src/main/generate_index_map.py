import os
import glob
import json
import re
import sys
from typing import List, Dict, Optional, Tuple

# Constants (aligned with Docker volume mount: /app/artifacts)
DATA_DIR = "./study"
OUTPUT_DIR = "./artifacts"
ARTIFACT_FILE = "recommendation_map.json"
MAX_RECOMMENDATIONS = 3


def parse_frontmatter(content: str) -> Optional[Dict]:
    """
    Parse YAML-like frontmatter from MDX content without extra deps.
    Expects content starting with '---' and ending with the next '---'.
    """
    try:
        match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
        if not match:
            return None

        frontmatter_raw = match.group(1)
        metadata: Dict[str, object] = {}

        for line in frontmatter_raw.split('\n'):
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            if value.startswith('[') and value.endswith(']'):
                value = [v.strip() for v in value[1:-1].split(',') if v.strip()]

            metadata[key] = value

        return metadata
    except Exception as e:
        print(f"[WARN] Failed to parse frontmatter. Error: {e}")
        return None


def normalize_tags(raw_tags) -> List[str]:
    """Ensure tags are a lowercase list."""
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        return [raw_tags.strip().lower()] if raw_tags.strip() else []
    if isinstance(raw_tags, list):
        return [str(t).strip().lower() for t in raw_tags if str(t).strip()]
    return []


def jaccard_score(a: List[str], b: List[str]) -> float:
    """Simple Jaccard similarity for tag overlap."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def token_overlap(title_a: str, title_b: str) -> float:
    """Lightweight overlap score using token intersection in titles."""
    tokens_a = set(re.findall(r"\w+", title_a.lower()))
    tokens_b = set(re.findall(r"\w+", title_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def load_mdx_files(base_path: str) -> List[Dict]:
    """Recursively load .mdx files and extract frontmatter."""
    print(f"[INFO] Scanning directory: {base_path}...")

    search_pattern = os.path.join(base_path, "**", "*.mdx")
    files = glob.glob(search_pattern, recursive=True)
    print(f"[INFO] Found {len(files)} .mdx files.")

    problems: List[Dict] = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            metadata = parse_frontmatter(content)

            if metadata and "id" in metadata:
                metadata["file_path"] = file_path
                metadata["tags"] = normalize_tags(metadata.get("tags"))
                metadata["title"] = metadata.get("title", "Unknown Title")
                problems.append(metadata)
            else:
                print(f"[WARN] Skipping {file_path}: No valid 'id' in frontmatter.")

        except Exception as e:
            print(f"[ERROR] Error reading {file_path}: {e}")

    print(f"[INFO] Loaded {len(problems)} valid problem records.")
    return problems


def build_recommendations(problems: List[Dict], idx: int, top_k: int = MAX_RECOMMENDATIONS) -> List[Dict]:
    """
    Build lightweight recommendations using tag + title overlap.
    This is a placeholder until FAISS/embedding logic is wired.
    """
    target = problems[idx]
    target_tags = target.get("tags", [])
    target_title = target.get("title", "")

    scored: List[Tuple[float, Dict]] = []
    for j, cand in enumerate(problems):
        if j == idx:
            continue

        score_tags = jaccard_score(target_tags, cand.get("tags", []))
        score_title = token_overlap(target_title, cand.get("title", ""))
        score = 0.7 * score_tags + 0.3 * score_title

        scored.append((score, cand))

    scored.sort(key=lambda x: x[0], reverse=True)

    recommendations: List[Dict] = []
    for rank, (score, cand) in enumerate(scored[:top_k], 1):
        recommendations.append(
            {
                "rank": rank,
                "id": cand.get("id"),
                "title": cand.get("title"),
                "tags": cand.get("tags", []),
                "similarity": round(score, 4),
                "source": "tag+title overlap (placeholder)"
            }
        )

    return recommendations


def generate_index_and_map() -> int:
    print("[INFO] [Heavy Job] Starting Indexing & Map Generation Process...")

    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Data directory '{DATA_DIR}' not found. Ensure it is checked out or mounted.")
        return 1

    problems = load_mdx_files(DATA_DIR)
    if not problems:
        print("[ERROR] No problems found to index. Exiting.")
        return 1

    print("[INFO] Building recommendation map (placeholder heuristic)...")

    recommendation_map: Dict[str, Dict] = {}
    total = len(problems)

    for idx, prob in enumerate(problems, 1):
        prob_id = prob.get("id")
        if not prob_id:
            continue

        if idx == 1 or idx % 10 == 0:
            print(f"[INFO] Processing {idx}/{total}: {prob_id}")

        recs = build_recommendations(problems, idx - 1, top_k=MAX_RECOMMENDATIONS)

        recommendation_map[prob_id] = {
            "id": prob_id,
            "title": prob.get("title", "Unknown Title"),
            "date": prob.get("date"),
            "tags": prob.get("tags", []),
            "recommendations": recs,
        }

    print(f"[INFO] Saving artifacts to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, ARTIFACT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(recommendation_map, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Artifact generated successfully: {output_path}")
    return 0


def main() -> None:
    exit_code = generate_index_and_map()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()