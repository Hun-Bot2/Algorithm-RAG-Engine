import json
import os

def build_gt(potential_file, lc_file, output_file):
    with open(potential_file, 'r', encoding='utf-8') as f:
        potential_data = json.load(f)
    
    # 실제 LC 파일에 존재하는 슬러그만 필터링하기 위해 로드
    existing_lc_slugs = set()
    with open(lc_file, 'r', encoding='utf-8') as f:
        for line in f:
            existing_lc_slugs.add(json.loads(line).get("titleSlug"))
    
    final_gt = {}
    missing_count = 0
    
    for bj_id, candidates in potential_data.items():
        # 후보 중 실제 DB(LC 파일)에 존재하는 첫 번째 후보를 정답으로 채택
        found = False
        for cand in candidates:
            if cand in existing_lc_slugs:
                final_gt[bj_id] = cand
                found = True
                break
        
        if not found:
            missing_count += 1
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_gt, f, indent=2, ensure_ascii=False)
        
    print(f"Final GT created with {len(final_gt)} pairs.")
    print(f"Problems with no matching LC slug in DB: {missing_count}")

if __name__ == "__main__":
    build_gt("potential_gt.json", "leetcode_refined.jsonl", "ground_truth_v2.json")