"""Read-only loopback inspection of hash-verified historical retrieval results."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .compare import ROOT, METHODS, load_rankings, verify

METHOD_INFO = {
    'minilm_title': {'label': '제목', 'description': '제목만 임베딩 · 기존 MiniLM 384D', 'score_label': '이 방식의 코사인 유사도'},
    'minilm_title_statement': {'label': '제목 + 본문 앞부분', 'description': '제목+본문 입력을 최대 256 token으로 자른 결과', 'score_label': '이 방식의 코사인 유사도'},
    'minilm_chunk': {'label': '전체 본문 분할', 'description': '본문 전체를 254 token 단위로 나누고 가중 평균한 벡터', 'score_label': '이 방식의 코사인 유사도'},
    'tfidf_word': {'label': '단어 TF-IDF', 'description': '단어 1–2 gram · 비임베딩 비교 기준', 'score_label': '정규화된 단어 TF-IDF 유사도'},
    'tfidf_char': {'label': '문자 TF-IDF', 'description': '문자 3–5 gram · 비임베딩 비교 기준', 'score_label': '정규화된 문자 TF-IDF 유사도'},
}
FIELDS = ('problem_key', 'title', 'platform', 'native_key', 'statement_text', 'source_dataset',
          'source_revision', 'license', 'collection_status', 'constraints_status', 'source_tags', 'native_difficulty', 'input_views', 'input_provenance')


def safe_source_url(value, platform):
    if not isinstance(value, str):
        return None
    try:
        url = urlsplit(value)
        hosts = {'codeforces': {'codeforces.com', 'www.codeforces.com'}, 'atcoder': {'atcoder.jp'}, 'leetcode': {'leetcode.com', 'www.leetcode.com'}}
        return value if url.scheme == 'https' and url.hostname in hosts.get(platform, set()) and not url.username and not url.password else None
    except ValueError:
        return None


def load_dataset(run):
    receipt = json.loads((ROOT / 'research/algorithm_system/statement-baseline.v1.json').read_text())
    if json.loads((run / 'manifest.json').read_text()) != receipt:
        raise ValueError('run manifest differs from admitted baseline')
    verify(run / 'corpus.jsonl', receipt['corpus_sha256'])
    problems = {}
    with (run / 'corpus.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['platform'] not in {'codeforces', 'atcoder'} or row['problem_key'] in problems:
                raise ValueError('unexpected platform or duplicate identity')
            p = {key: row.get(key) for key in FIELDS}
            p['original_url'] = safe_source_url(row.get('original_url'), row['platform'])
            problems[p['problem_key']] = p
    if len(problems) != receipt['count']:
        raise ValueError('corpus count mismatch')
    rankings = {}
    for method in METHODS:
        path = run / method / 'top10.jsonl'
        verify(path, receipt['methods'][method]['outputs']['top10.jsonl'])
        load_rankings(path, set(problems))
        with path.open() as handle:
            rankings[method] = {row['query']: row['results'] for row in map(json.loads, handle)}
    return {'problems': problems, 'rankings': rankings}


def public_problem(problem):
    result={key: problem.get(key) for key in (*FIELDS, 'original_url')}
    result['original_url']=safe_source_url(result['original_url'], result['platform'])
    return result


def response_data(dataset, path, params):
    scopes = dataset.get('scopes')
    default_scope=next((name for name in ('strategy_leetcode','leetcode','publisher') if scopes and name in scopes),'publisher')
    scope = params.get('scope', [default_scope])[0]
    if scopes:
        if scope not in scopes:
            raise KeyError('unknown scope')
        selected = scopes[scope]
    else:
        selected = dataset
    problems = selected['problems']
    if path == '/api/meta':
        counts = {platform: sum(p['platform'] == platform for p in problems.values()) for platform in ['codeforces', 'atcoder']}
        default = next((key for key, p in problems.items() if p.get('native_key') and p['platform'] == 'codeforces'), next(iter(problems)))
        return {'coverage': selected.get('coverage', f"Codeforces {counts['codeforces']:,} · AtCoder {counts['atcoder']:,} · LeetCode 제외 · 과거 실행 결과"),
                'methods': selected.get('methods', METHOD_INFO), 'default_key': selected.get('default_key', default),
                'default_a':selected.get('default_a'), 'default_b':selected.get('default_b'), 'footer':selected.get('footer'), 'scope': scope, 'scopes': {key: data.get('label', 'LeetCode 내부 (제목·태그)' if key == 'leetcode' else 'CF ↔ AtCoder 본문 (과거)') for key,data in (scopes or {'publisher': dataset}).items()},
                'notice': selected.get('notice', '과거에 생성한 벡터와 저장된 검색 결과를 직접 확인합니다. 새 임베딩이나 추천 정확도 평가가 아닙니다.'),
                'empty_message': selected.get('empty_message', '이 코호트에서 찾지 못했습니다. LeetCode는 별도 모드에서 검색해 주세요.')}
    if path == '/api/search':
        query = params.get('q', [''])[0][:120].casefold().strip()
        found = [p for p in problems.values() if query in ((p['title'] or '')+' '+(p['native_key'] or '')+' '+p['problem_key']).casefold()]
        return [{k: p[k] for k in ('problem_key', 'title', 'platform')} for p in sorted(found, key=lambda p: (p['title'], p['problem_key']))[:30]]
    if path in {'/api/problem', '/api/compare'}:
        key = params.get('key', [''])[0]
        if key not in problems:
            raise KeyError('unknown problem')
        if path == '/api/problem':
            return public_problem(problems[key])
        methods = {}
        for method, rows in selected['rankings'].items():
            methods[method] = [{**{k: problems[target][k] for k in ('problem_key', 'title', 'platform')}, 'score': score}
                               for target, score in rows[key]]
        return {'source': public_problem(problems[key]), 'methods': methods}
    raise KeyError('unknown route')


DOCUMENTS={'/notebook.html':('recommendation_experiments.html','text/html; charset=utf-8'),'/notebook.ipynb':('recommendation_experiments.ipynb','application/x-ipynb+json'),'/blog.html':('blog.html','text/html; charset=utf-8'),'/report.html':('experiment_report.html','text/html; charset=utf-8')}

def create_server(dataset, port=0):
    html = Path(__file__).with_name('viewer.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, payload, content_type='application/json; charset=utf-8', download=None):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            if download:self.send_header('Content-Disposition',f'attachment; filename="{download}"')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
            self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            host = f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') != host or self.headers.get('Origin') not in (None, 'http://'+host):
                return self.reply(403, {'error': 'same-origin loopback only'})
            parsed = urlsplit(self.path)
            if parsed.path in DOCUMENTS:
                filename,content_type=DOCUMENTS[parsed.path]
                path=Path(__file__).with_name(filename)
                if not path.is_file():return self.reply(404,{'error':'artifact unavailable'})
                return self.reply(200,path.read_bytes(),content_type,filename if parsed.path=='/notebook.ipynb' else None)
            if parsed.path == '/':
                return self.reply(200, html, 'text/html; charset=utf-8')
            try:
                return self.reply(200, response_data(dataset, parsed.path, parse_qs(parsed.query, max_num_fields=4)))
            except KeyError:
                return self.reply(404, {'error': 'not found'})
            except ValueError:
                return self.reply(400, {'error': 'invalid query'})

        def do_POST(self):
            self.reply(405, {'error': 'read-only viewer'})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'private-learning/algorithm-system/statement-run-v1')
    parser.add_argument('--port', type=int, default=8789)
    args = parser.parse_args()
    publisher = load_dataset(args.run)
    dataset = {'scopes': {'publisher': publisher}}
    lc_file = ROOT / 'private-learning/recommendation-inputs/leetcode-title-v1.json'
    lc_receipt = ROOT / 'research/recommendation_inputs/leetcode-title-v1.receipt.json'
    if lc_file.exists() and lc_receipt.exists():
        verify(lc_file, json.loads(lc_receipt.read_text())['output_sha256'])
        dataset['scopes']['leetcode'] = json.loads(lc_file.read_text())
    for folder, receipt_name in [('curriculum-ideas-v1','curriculum-ideas-v1.receipt.json'),('publisher-structure-v1','publisher-structure-v1.receipt.json'),('curriculum-rerank-v1','curriculum-rerank-v1.receipt.json'),('curriculum-rerank-plain-v1','curriculum-rerank-plain-v1.receipt.json')]:
        root=ROOT/'private-learning/recommendation-inputs'/folder
        receipt_path=ROOT/'research/recommendation_inputs'/receipt_name
        if root.exists() and receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            if json.loads((root/'manifest.json').read_text())!=receipt:
                raise ValueError('experiment manifest differs from public receipt')
            for name,info in receipt['scopes'].items():
                if Path(info['file']).name!=info['file']:
                    raise ValueError('invalid scope file')
                path=root/info['file'];verify(path,info['sha256'])
                dataset['scopes'][name]=json.loads(path.read_text())
    server = create_server(dataset, args.port)
    print(f'Historical input comparison: http://127.0.0.1:{server.server_port}', flush=True)
    print(f'{len(publisher["problems"])} admitted descriptions; scopes: {list(dataset["scopes"])}; read-only.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
