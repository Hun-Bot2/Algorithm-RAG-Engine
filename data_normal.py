import json

def normalize_baekjoon(input_path, output_path):
    processed = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # 'Logical Skeleton: ' 접두사 제거 (데이터 대칭성 확보)
            if "embedding_text" in item:
                item["embedding_text"] = item["embedding_text"].replace("Logical Skeleton: ", "").strip()
            processed.append(item)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Normalized data saved to {output_path}")

if __name__ == "__main__":
    normalize_baekjoon("baekjoon_refined.jsonl", "baekjoon_normalized.jsonl")