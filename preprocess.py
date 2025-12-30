import json
import re
from bs4 import BeautifulSoup

def clean_content(html_content):
    if not html_content: return ""
    soup = BeautifulSoup(html_content, "html.parser")
    # 줄바꿈 유지 처리
    for br in soup.find_all("br"): br.replace_with("\n")
    for p in soup.find_all("p"): p.append("\n")
    text = soup.get_text()
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def preprocess_file(input_path, output_path, platform):
    results = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # 리트코드는 HTML 포함, 백준은 이미 텍스트인 경우 대응
            raw_content = item.get("content", "")
            cleaned = clean_content(raw_content) if platform == "leetcode" else raw_content.strip()
            
            tags_str = ", ".join(item.get("tags", []))
            # 검색 성능 극대화를 위한 임베딩용 텍스트 구성
            search_text = f"Title: {item['title']}\nTags: {tags_str}\nContent: {cleaned}"
            
            item["content_cleaned"] = cleaned
            item["embedding_text"] = search_text
            results.append(item)
            
    with open(output_path, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"{platform} preprocessing complete: {len(results)} items.")

if __name__ == "__main__":
    preprocess_file("leetcode_raw_data.jsonl", "leetcode_preprocessed.jsonl", "leetcode")
    preprocess_file("baekjoon_raw_data.jsonl", "baekjoon_preprocessed.jsonl", "baekjoon")