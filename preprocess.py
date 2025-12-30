import json
import re
from bs4 import BeautifulSoup

def clean_problem_content(html_content):
    """
    Remove HTML tags and extract the core text of the problem.
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")
    
    # Insert line breaks to preserve the structure of Example and Constraints sections
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for p in soup.find_all("p"):
        p.append("\n")

    text = soup.get_text()

    # Normalize special characters and unnecessary whitespace
    # Merge consecutive newlines or spaces into one
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

def process_jsonl(input_path, output_path):
    """
    Read raw_data, preprocess it, and save to a new file.
    """
    processed_data = []
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                
                # Clean the problem content
                cleaned_content = clean_problem_content(item.get("content", ""))
                
                # Create 'searchable text' for vector DB embedding (title + tags + content summary)
                tags_str = ", ".join(item.get("tags", []))
                search_text = f"Title: {item['title']}\nTags: {tags_str}\nContent: {cleaned_content}"
                
                processed_item = {
                    "id": item["id"],
                    "title": item["title"],
                    "titleSlug": item["titleSlug"],
                    "difficulty": item["difficulty"],
                    "tags": item["tags"],
                    "content_cleaned": cleaned_content,
                    "embedding_text": search_text  # This field will be vectorized later.
                }
                processed_data.append(processed_item)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        print(f"Preprocessing complete: {len(processed_data)} problems saved to {output_path}.")
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    process_jsonl("leetcode_raw_data.jsonl", "leetcode_preprocessed.jsonl")