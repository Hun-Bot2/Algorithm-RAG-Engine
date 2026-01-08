"""
Fast checker: diff against daily-updated public JSON (noworneverev/leetcode-api)
This avoids paging GraphQL and reduces time/cost.
"""
import json
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = RAW_DIR / "leetcode_slugs.txt"
REPORT = RAW_DIR / "new_leetcode_problems_fast.json"
REMOTE_JSON = "https://raw.githubusercontent.com/noworneverev/leetcode-api/main/data/leetcode_questions.json"


def load_local_slugs():
    slugs = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                slugs.add(s)
    return slugs


def fetch_remote_slugs():
    resp = requests.get(REMOTE_JSON, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # The structure contains a list of questions; fields include titleSlug and isPaidOnly
    result = []
    for q in data:
        slug = q.get("titleSlug") or q.get("title_slug")
        paid = q.get("isPaidOnly") or q.get("paid_only") or False
        if slug and not paid:
            result.append({
                "id": str(q.get("questionId") or q.get("frontendQuestionId") or ""),
                "title": q.get("title") or "",
                "titleSlug": slug,
                "difficulty": q.get("difficulty") or "",
                "url": f"https://leetcode.com/problems/{slug}/",
                "isPaidOnly": bool(paid)
            })
    return result


def main():
    local_slugs = load_local_slugs()
    remote = fetch_remote_slugs()
    new_items = [q for q in remote if q["titleSlug"] not in local_slugs]

    report = {
        "existing_count": len(local_slugs),
        "remote_count": len(remote),
        "new_count": len(new_items),
        "new_problems": new_items[:200]  # trim for readability
    }

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Existing: {report['existing_count']} | Remote free: {report['remote_count']} | New: {report['new_count']}")
    print(f"Saved: {REPORT}")


if __name__ == "__main__":
    main()
