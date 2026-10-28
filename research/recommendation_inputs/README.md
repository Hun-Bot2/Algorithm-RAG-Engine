# 추천 입력·재정렬 실험

- [실행된 Jupyter 노트북](recommendation_experiments.ipynb)
- [노트북 HTML](recommendation_experiments.html)
- [블로그 초안](BLOG.md) / [블로그 HTML](blog.html)
- [실험 범위·영수증](EXPERIMENTS.md)

학습 앱의 **문제 유사도 비교** 메뉴에서도 노트북 읽기·ipynb 다운로드·실험 글 읽기 링크를 제공한다: http://127.0.0.1:8787/#similarity

## 재실행

프로젝트 연구용 `.venv`에 설치된 패키지와 해시 고정된 로컬 입력/가중치를 사용한다. 실행 중 네트워크·유료 API·훈련은 사용하지 않는다. 노트북은 작은 90개 작성 입력의 임베딩 및 두 Cross-encoder 조건의 실제 추론을 재실행한다. 큰 본문 실험은 저장된 영수증을 검증해 사용한다.

```sh
.venv/bin/python -m research.recommendation_inputs.execute_notebook
.venv/bin/python -m research.recommendation_inputs.export_articles
```

노트북/HTML/그림의 파생 출력을 갱신한다. 사용자가 노트북 셀을 편집한 경우 `execute_notebook`은 그 셀을 보존해서 실행한다. `build_notebook.py`는 템플릿에서 노트북을 다시 만드는 별도 명령이므로 의도적으로 재생성할 때만 실행한다.

## 자료 경계

- 작성 설명 90개는 AI 초안·AI 검토 자료이며 사람 정답이나 원래 문제 본문이 아니다.
- LeetCode 내부 실험은 그중 60개를 사용한다.
- 본문 구조 실험은 별도 CF/AtCoder 9,264개 코호트이며 LeetCode 본문은 없다.
- Cross-encoder는 두 설명 검색의 고정 Top10 합집합만 재정렬한다. raw logit은 확률/코사인이 아니다.
- 독립적인 아이디어/내용/학습 적합성 평가 라벨이 없어 정확도 개선이나 운영 정책 승격을 주장하지 않는다.
- private-learning/의 가중치·원문·실험 cache는 Git/정적 배포 자산에 포함하지 않는다.

## 검사

```sh
.venv/bin/python -m unittest discover -s research/recommendation_inputs -v
node --test hosted/test/*.test.mjs
```

Cross-encoder 파일 SHA-256/정확한 구성, 입력 코호트/후보 제외, query 조건, 원래-token 보존을 검사한다. 실제 추론과 저장된 결과의 재현 검사는 노트북에서 수행한다. 그림과 HTML은 실제 브라우저에서도 확인했다. 블로그는 저장소 내 초안이며 외부 게시하지 않았다.

## 2D maps in the viewer (2026-10-28)

`maps.py` adds `/api/map?scope=&method=` to the read-only viewer: a t-SNE layout (cosine, PCA init, fixed seed) of the stored vectors
for each method. Before drawing, it recomputes top-k from the vectors and requires an exact match with the stored ranking; methods
without vectors (cross-encoder) or scopes without vectors (LeetCode title/tag, publisher) report that no map is available. The page
shows the share of each problem's stored top-10 that is also among its 10 nearest map points (about 48-58% here): positions are
approximate, and the numbered neighbors and lines always come from the 384D ranking. Tests: `.venv/bin/python -m unittest
research.recommendation_inputs.test_maps`.

The map toolbar switches between all platforms and the selected problem's platform. In same-platform mode the server lays out only
that platform's problems and recomputes the top-10 within it from the same verified vectors (`/api/map?platform=`).
