# 입력 표현 실험 실행 결과 — 2026-10-23

이번에는 과거 결과 재분석을 넘어 **새 입력을 실제로 임베딩**했다. 1·2절의 입력 실험은 이미 허용된 로컬 MiniLM을 사용했다. 이후 사용자가 승인한 Cross-encoder 다운로드·실행은 4절에 기록했다. 학습·유료 API 호출은 하지 않았다. 원래 임베딩과 기존 추천 정책은 보존했다.

## 1. 풀이 아이디어 표현: 작성 커리큘럼 90개

입력은 해시 고정된 curriculum/annotations.v1.json의 concept_rationale와 seed.v1.json의 concepts다. 자료 출처는 **Luna 초안 + Sol AI 검토**이며 원본 문제 본문이나 사람 평가 정답이 아니다. 새 알고리즘 설명을 제목에서 생성하지 않았다.

- LeetCode 60개, AtCoder 20개, Codeforces 10개.
- 같은 384D 모델에서 (a) 제목, (b) 핵심 설명, (c) CORE IDEA + CONCEPTS를 각각 새로 임베딩했다.
- 명시적 불변식/정당성/복잡도 필드는 원자료에 없어서 미기록으로 표시했다.
- LeetCode 내부 60개와 전체 플랫폼 90개를 별도 후보 집합으로 비교했다.
- 실행 6.69초. 숫자는 이 짧은 작성 설명 코호트의 로컬 실행이며 전체 카탈로그로 일반화하지 않는다.

| 동일 LeetCode 60개에서 비교 | 평균 Top10 공통 후보 | 1위 일치율 |
| --- | ---: | ---: |
| 핵심 설명 ↔ 구조화 설명 | 5.43개 | 36.67% |
| 핵심 설명 ↔ 제목 | 2.92개 | 5.00% |
| 구조화 설명 ↔ 제목 | 3.00개 | 5.00% |

예를 들어 Two Sum의 제목 기준선은 Combination Sum, 3Sum 등을 상위에 두고, 구조화 설명은 Contains Duplicate, Longest Consecutive Sequence, Group Anagrams 등을 상위에 둔다. 이것은 다른 후보를 만든다는 관찰일 뿐, 후자가 같은 풀이 원리를 더 정확히 찾았다는 정답 판정이 아니다. 공통 hash 관련 표현만으로 다른 기법이 섞일 가능성이 남는다.

실행 영수증: [curriculum-ideas-v1.receipt.json](curriculum-ideas-v1.receipt.json). 사적 출력: private-learning/recommendation-inputs/curriculum-ideas-v1/.

## 2. 본문 구조 표현: 공통 파싱 코호트 9,264개

기존 허용 publisher description 9,424개 중 description/Input/Output이 확인되고 중복 primary heading이 없는 **9,264개(AtCoder 1,275, Codeforces 7,989)**를 선택했다. 여러 제외 사유가 한 행에 중복될 수 있어 사유별 수를 합해 제외 행 수로 사용하면 안 된다. 후보는 반대 플랫폼이며 자기 자신/동일 content hash를 제외했다.

- 원래 전체 본문과 제목 벡터는 재사용하되 **같은 선택 코호트에서 검색을 재계산**했다.
- 새 입력 1: description + Input + Output + 별도 Constraints. 예제/노트를 제외했다.
- 새 입력 2: Input + Output + 별도 Constraints. 설명/이야기/예제/노트를 제외했다.
- Sample/Example 안의 반복 Input/Output을 실제 입출력 섹션으로 사용하지 않는 보수적 파서다.
- 독립 Constraints 섹션이 없는 7,991개를 별도로 기록했다. Input 안의 수치 제약까지 없다는 뜻은 아니다.
- 254-token 원래 토큰 구간을 사용하고 선택 입력 전체 토큰의 가중 pooling 및 L2 정규화를 적용했다.

| 입력 | 새 임베딩 시간 | 구간 수 | 선택 입력 token 수 |
| --- | ---: | ---: | ---: |
| 과제·입출력·제약 | 269.41초 | 17,748 | 3,351,271 |
| 입출력·제약만 | 146.15초 | 9,935 | 1,304,763 |

네 방식의 검색은 합계 2.84초, 준비·모델 로드·저장을 포함한 전체 실행은 **426.39초(약 7.1분)**였다. 두 입력 모두 선택한 token을 전부 처리했으며, tokenizer의 원문 길이 경고와 달리 실제 encoder 입력 구간은 모두 256 이하였다.

| 같은 9,264개에서 비교 | 평균 Top10 공통 후보 | 1위 일치율 |
| --- | ---: | ---: |
| 과제·입출력·제약 ↔ 전체 본문 | 6.73개 | 52.49% |
| 과제·입출력·제약 ↔ 입출력·제약만 | 2.86개 | 15.37% |
| 과제·입출력·제약 ↔ 제목 | 0.68개 | 2.62% |

구조 분리는 후보를 바꾸지만, 같은 입출력 형식/수치 조건을 가진 전혀 다른 알고리즘 문제가 가까워질 수도 있다. **정확도/교육 효과 라벨은 아직 없으며 어느 방식도 승자로 선택하지 않았다.** 이 실험에는 LeetCode 본문이 없다.

실행 영수증: [publisher-structure-v1.receipt.json](publisher-structure-v1.receipt.json). 구현/출력 해시: [input-experiment-implementation.v1.json](input-experiment-implementation.v1.json).

## 3. 학습 앱에서 확인

http://127.0.0.1:8787/#similarity — 알고리즘 학습 → 문제 유사도 비교.

새 기본 모드는 **LeetCode 풀이 설명 실험**이다. 비교 범위에서 **3플랫폼 풀이 설명 실험**, **본문 구조 실험 (CF↔AtCoder)**, 기존 전체 LeetCode 제목/태그 및 과거 본문 모드를 선택할 수 있다. 기준 문제/후보에서 실제 입력 설명을 펼치고 각 방식의 점수/후보를 확인한다. 원래 본문은 별도 펼침 영역으로 보존했다.

범위별 후보 집합을 섞지 않으며 점수는 모델/표현 내부에서만 해석한다. 조회는 학습 결과나 평가 라벨을 생성하지 않는다. 새로운 약점 추천 정책, 주간 스케줄, 품질 승격은 아직 적용하지 않았다.

## 4. 승인 후 실제 Cross-encoder 실행

사용자가 모델 다운로드와 즉시 실행을 명시적으로 승인했다. 공식 cross-encoder/ms-marco-MiniLM-L6-v2, revision 233902d25c440f23af6f7d6e94d2946bac0bee0a를 다운로드했다. 7개 파일 91,819,431 bytes이며 safetensors SHA-256은 821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae다. 로더는 전체 파일 membership/해시를 검사하고 CPU/offline/safetensors-only로 실행한다.

같은 후보군(핵심 설명/구조화 설명 각각 Top10의 합집합)에 질문형과 설명 직접 조건을 적용했다. LeetCode 60개는 874 pair, 전체 90개는 1,340 pair다. 최초 CLI 추론 합계는 질문형 7.52초, 설명 직접 6.18초였다. 최대 pair 입력은 58 token으로 잘림은 없었다. 결과값은 raw logit이며 확률/코사인이 아니다.

Two Sum의 1위는 두 조건 모두 House Robber였다. 이 관찰은 현재 표현·후보·범용 MS MARCO 모델에서의 실패 사례이며 모든 Cross-encoder의 성능 결론이나 정확도 측정이 아니다. 별도 사람 정답/학습 효과 평가 없이 운영 정책에 승격하지 않았다.

실행 영수증: curriculum-rerank-v1.receipt.json, curriculum-rerank-plain-v1.receipt.json. 모델/원본 데이터는 private-learning/ 아래에 둔다. [실행된 노트북](recommendation_experiments.ipynb)은 90개 입력을 재임베딩하고 두 조건을 실제로 재추론했으며 저장된 후보 순서와 raw logit을 검증한다. [블로그 초안](BLOG.md)에 과정과 실패 사례를 남겼다.

## 검증

- 입력/순위/Cross-encoder 준비 단위 테스트 22개 통과. Cross-encoder unit tests는 fake scorer/실패 경로를 검증하며 실제 추론의 재현 검사는 실행 노트북에서 별도로 수행했다.
- 기존 모델은 로드 전 전체 파일 membership/SHA-256 확인. 새 출력 파일 해시·벡터 유한성·단위 norm 및 표본 저장 cosine을 독립 재계산해 확인했다.
- 전체 Node 회귀 테스트 214개 및 앱 빌드 통과.
- 실제 Chrome 1440px/390px: 새 기본 모드, 실제 입력 필드, LeetCode 내부 후보, 3플랫폼/본문 코호트 전환, 점수 일치, 원래 본문 보존, 기존 모드/캘린더 복귀 확인. 스크린샷은 private-learning/recommendation-input-review/experiments-v1/에 저장했다.
- 새 본문 실험은 실제로 실행했다. 사람 평가나 새 운영 추천 정책의 품질 검증을 완료했다고 주장하지 않는다.

재현 명령은 덮어쓰지 않는 새 출력 경로를 사용한다.

```sh
.venv/bin/python -m unittest discover -s research/recommendation_inputs -v
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m research.recommendation_inputs.experiments --kind curriculum \
  --output private-learning/recommendation-inputs/curriculum-ideas-v2 \
  --receipt research/recommendation_inputs/curriculum-ideas-v2.receipt.json
```

Suggested commit: `research: run structured-content and solving-idea input ablations`.


### 노트북·블로그 최종 확인 (2026-10-23)

- 노트북 코드 셀 10개를 처음부터 끝까지 실제 실행했고 오류는 0개다. 저장된 후보 순서는 모두 재현됐으며 raw logit 최대 차이는 0이었다. 본문 대규모 실행은 저장된 해시 영수증으로 검증하며 노트북에서 재실행하지 않는다.
- PNG 2개가 노트북 출력에 포함되고 HTML에서도 로드됨을 확인했다. 최초 HTML 점검에서 그림 누락을 발견해 명시적 이미지 출력으로 수정하고 전체 노트북을 재실행했다.
- 실제 Chrome에서 앱의 Cross-encoder 점수, 문서 링크, ipynb 다운로드 헤더, 그림 로딩, 블로그 1440/390px 레이아웃을 확인했다. 모델 파일 요청은 404다. 화면 캡처와 영수증: private-learning/recommendation-input-review/notebook-blog-v1/.
- 연구 테스트 22개, Node 테스트 214개, 앱 빌드 통과. 추천 엔진·학습 기록·운영 순위에는 변경이 없다. 블로그는 외부 게시하지 않은 초안이다.
- 재실행 명령과 자료 경계는 [README](README.md), 새 추론 시간과 입력 해시는 [notebook-last-run.json](notebook-last-run.json)에 남겼다.

Suggested commit: `research: evaluate Cross-encoder reranking with notebook and blog`.
