# 무엇을 임베딩할 것인가 — 기존 결과 재비교

2026-10-23. 새 임베딩을 생성하지 않고, 기존 실행의 입력·결과 해시를 검증한 뒤 상위 후보 차이를 재계산했다. [실행 결과](comparison.v1.json), [재현 코드](compare.py).

## 이번에 확인한 범위

동일한 Codeforces 8,101개 + AtCoder 1,323개 = **9,424개** publisher-description 코호트이며, 각 문제에서 반대 플랫폼 후보를 검색한 결과다. 현재 서비스의 24,303개 네이티브 카탈로그와는 별개다. LeetCode 본문은 이 코호트에 없으며, AtCoder 1,323개는 실제 task ID와 연결되지 않은 데이터셋 ID다.

검증: 기존 공개 baseline receipt와 private manifest 일치, corpus/labels 및 방법별 top10/top50 SHA-256 일치, 동일 query 집합·유효 candidate ID·중복/자기 자신 제외·점수 정렬 확인. 본문 텍스트는 로드하지 않고 corpus 파일은 바이트 해시만 확인했다. 원본 결과와 카탈로그는 수정하지 않았다.

## 입력에 따른 후보 변화 — 이번에 재계산

| 비교 | 상위 10개 중 평균 공통 후보 | 1위 후보 일치율 |
| --- | ---: | ---: |
| 제목 ↔ 제목+본문 앞부분 | 0.87개 | 3.68% |
| 제목 ↔ 전체 본문 분할 | **0.66개** | **2.77%** |
| 제목+본문 앞부분 ↔ 전체 본문 분할 | 4.74개 | 29.69% |
| 전체 본문 분할 ↔ 단어 TF-IDF | 1.97개 | 10.99% |
| 단어 TF-IDF ↔ 문자 TF-IDF | 4.95개 | 36.99% |

제목과 전체 본문은 후보 집합이 크게 다르다. 제목과 전체 본문 방식 사이에서는 5,796개 query가 상위 10개 후보를 하나도 공유하지 않았다. 본문 앞부분과 전체 본문도 완전히 같은 검색을 하지 않는다. **차이가 크다는 사실은 품질 개선의 증거가 아니다.**

제목 방식에서는 한 후보가 1,414개 query의 Top10에 등장했고, 전체 본문 분할에서는 최대 484개였다. 이는 특정 후보로의 집중도이며 관련성/정확도가 아니다. 두 플랫폼의 크기가 다르고 반대 플랫폼 검색만 수행했으므로 전체 카탈로그의 추천 다양성으로 일반화하지 않는다.

## 비용 — 기존 실행 기록, 이번에 재측정하지 않음

동일한 고정 MiniLM 모델의 세 입력 비교다. 본문 앞부분 방식은 최대 256 token으로 잘리고, 전체 본문 방식은 254-content-token 단위 22,034개 구간을 임베딩한 뒤 token 수로 가중 평균·정규화했다. 공개 description의 모든 token을 사용했다는 의미이며, 원래 플랫폼 문제 페이지의 모든 정보가 포함됐다는 검증은 아니다.

| 입력 | 임베딩 시간 | 검색 시간 |
| --- | ---: | ---: |
| 제목 | 3.46초 | 1.39초 |
| 제목+본문 앞부분 | 64.80초 | 1.35초 |
| 전체 본문 분할 | 146.14초 | 1.37초 |

단어 TF-IDF는 fit+검색 합계 10.05초, 문자 TF-IDF는 96.41초였다. TF-IDF는 비임베딩 비교 기준이다. 다섯 방식 전체 기존 실행은 336.60초, 프로세스 peak RSS 1.50 GiB였다. 이 값들을 24,303개 문제나 다른 모델의 확정 비용으로 사용하지 않는다.

## 품질 판단의 현재 한계

모든 방법의 Top10 결과에서 기존 평가 라벨과 연결되는 slot이 **0개**다. 평가된 precision/nDCG가 없으므로 승자를 정할 수 없다. 이전 제목 기반 AI 라벨을 본문·풀이 전략 정답으로 바꿔 쓰지 않는다. 현재 비교는 입력 민감도와 비용에 대한 검증이다.

## 다음 입력 비교안

| 입력 표현 | 주된 평가 목적 | 현황/필요한 작업 |
| --- | --- | --- |
| 제목 | 제목 기반 기준선 | 기존 결과 사용 가능 |
| 문제 본문 전체 | 내용 유사 | 위 2플랫폼 코호트 결과 사용 가능; 3플랫폼 커버리지/ID 연결 필요 |
| 목표·입출력·제약을 구분한 문제 표현 | 내용 유사, 이야기 표현의 영향 확인 | 실제 원문에서 근거 있게 추출; 누락 정보는 미기록 |
| 풀이 아이디어: 핵심 기법·상태/불변식·정당성·복잡도 | 같은 아이디어 연습 | 사용 가능한 풀이/노트/검토된 관계 근거 필요; 제목만 보고 생성하지 않음 |
| 풀이 코드 | 구현·전략 유사성의 보조 비교 | 출처·언어·실패/성공 상태 보존; 단순 변수명/템플릿 유사성과 분리 평가 |
| 본문·전략·개념을 각각 검색 후 결합 | 복합 후보 선정 | 개별 입력 효과 확인 후 비교; 서로 다른 cosine을 그대로 평균하지 않음 |

약점 보완은 별도 learner snapshot을 사용한다. 문제 벡터에는 문제 정보를, 학습 정책에는 결과·재시도·복습 지정·기록 시점을 유지한다. 결과 1이나 복습 1만으로 개념 숙련/약점을 확정하지 않는다. 주간 전체 후보 갱신과 매일 최신 기록을 반영하는 방식은 이 분리를 따른다.

## 평가 절차 제안

1. 동일한 query/candidate 문제와 입력 근거가 확보된 코호트를 고정한다. 데이터가 없는 방법은 별도 코호트 결과로 표시한다.
2. 모델과 언어 전처리를 고정해 입력 표현만 먼저 비교한다. 모델·차원 변경은 다음 실험으로 분리한다.
3. 방법별 후보를 합쳐 새 버전의 blind review pool을 만든다. 기존 1,000쌍 대기 pool은 보존한다.
4. 각 pair에 아이디어 관계와 내용 관계를 별도 평가한다. 유사 이야기/다른 해법, 다른 이야기/같은 해법을 포함한다. 평가 불가는 unknown으로 둔다.
5. 약점 보완은 고정된 learner snapshot에서 왜 지금 유용한지 별도 평가한다. 정적 pair 관련성을 학습 효과로 바꾸지 않는다.
6. 문제 family/복제 관계별로 development/test를 나누고, 실제 평가 coverage와 함께 Precision@K/nDCG를 보고한다. 새 문제를 포함한 방법도 평가 pool에 포함해 기존 방법에 유리한 평가를 피한다.
7. 표본 판단이 확보된 뒤 유형별 가중치와 주간/일간 재선정 정책을 결정한다. 현재 결과만으로 임의 threshold나 새 순위를 적용하지 않는다.

## 재현 및 검증

```sh
python3 -m unittest research.recommendation_inputs.test_compare -v
python3 -m research.recommendation_inputs.compare \
  --run private-learning/algorithm-system/statement-run-v1 \
  --output /tmp/recommendation-input-comparison-new.json
```

출력은 덮어쓰지 않는다. 기존 평가 테스트 6개와 새 테스트 5개 통과. 기존 학습 UI/추천 엔진은 이 변경의 영향을 받지 않는다. 자동 일정, 새 모델 실행, 데이터 수집, 새 라벨, 품질 개선은 구현·주장하지 않았다.

Suggested commit: `research: compare matched recommendation input representations`.

같은 검증 입력으로 /tmp에 재실행한 결과가 comparison.v1.json과 바이트 단위로 일치했다. 이번 변경은 기존 research/algorithm_system/ 결과와 평가 라벨을 수정하지 않는다.

## 직접 열람

```sh
python3 -m research.recommendation_inputs.viewer --port 8789
```

`http://127.0.0.1:8789/`에서 문제 제목/네이티브 ID를 검색하고 두 방법의 저장된 Top10을 나란히 볼 수 있다. 후보를 누르면 기준 문제와 후보의 publisher 본문·출처가 표시된다. 원문 URL은 데이터에 있는 공식 HTTPS URL만 연결하며, 미확인 AtCoder ID의 링크를 추정하지 않는다. 새 임베딩 생성/사람 평가 수집/학습 상태 변경 없이 과거 결과를 열람한다. 기존 노트북 HTML은 집계 분석이고, 이 화면은 개별 사례 확인용이다.

검증: 비교/뷰어 단위 테스트 9개 통과. 실제 Chrome 1440px/390px에서 검색, 두 방법 전환, 저장된 점수 일치, 후보 본문 표시, 가로 넘침 없음 확인. 쓰기 요청 405, 외부 Origin 403, 비허용 파일 경로 404 확인. 실제 본문이 포함된 화면 및 브라우저 영수증은 private-learning/recommendation-input-review/에만 저장했다. 서비스는 루프백 8789에서 실행 중이며 종료 후에는 위 명령으로 다시 열 수 있다.

Suggested follow-up commit: `feat: add read-only historical embedding comparison viewer`.

## LeetCode 내부 비교와 학습 앱 연결 (2026-10-23)

알고리즘 학습 메뉴에 **문제 유사도 비교**를 추가했다. 실행 중인 로컬 앱에서는 http://127.0.0.1:8787/#similarity 로 바로 연다. /comparison/은 읽기 전용 허용 경로만 로컬 비교 서버에 연결하며, 비교 서비스 미가동 시 안내와 재시도 동작을 제공한다. private corpus/vector 파일은 hosted/public/에 복사하지 않는다.

LeetCode 범위는 현재 catalog-v4의 3,053개 문제를 조회하고 활성·eligible 2,949개를 후보로 사용한다. 기존 384D 제목 벡터로 동일 LeetCode 내부 코사인을 새로 계산했으며, 태그 Jaccard는 별도 비임베딩 기준이다. 태그가 없거나 교집합이 없는 후보는 채워 넣지 않는다. 본문·풀이 전략 벡터는 없고, 새 임베딩을 생성하지 않았다. 생성 기록: leetcode-title-v1.receipt.json. 사적인 결과 데이터: private-learning/recommendation-inputs/leetcode-title-v1.json.

변경 파일: research/recommendation_inputs/leetcode.py, viewer.py, viewer.html 및 단위 테스트; hosted/local-comparison.mjs, local.mjs, public/index.html, app.mjs, workspace.mjs, styles.css, test/local-comparison.test.mjs. 서버는 이 세션 소유 명령을 확인한 후 기존 설정으로 재시작했다.

검증: 연구 테스트 13개, 전체 Node 테스트 213개, 빌드 통과. 실제 Chrome 1440px/390px에서 앱 메뉴·직접 링크·Buddy Strings/Check Distances 검색·LeetCode 내부 후보·저장 점수 일치·공식 원문 링크·CF↔AtCoder 전환·캘린더 복귀·서비스 실패 후 재시도 확인. 화면은 private-learning/recommendation-input-review/integrated/에 저장하고 직접 확인했다. 기본 캘린더 진입 계약 테스트가 처음에는 리터럴 검사에서 실패하여 기본 캘린더 초기화를 보존한 뒤 전부 통과했다. 비교 데이터는 읽기 전용이며 새 학습 시도나 평가 라벨을 생성하지 않는다.

제목 코사인에 대한 사용자 관찰: 학습 추천의 핵심 근거로 제외할 방향에 동의한다. 제목 검색/문제 식별은 계속 필요하고, 과거 제목 임베딩은 입력 비교 기준선으로 남길 수 있다. 몇 사례의 육안 검토나 후보 불일치만으로 전반적인 정확도를 단정하지 않는다. 현재 주력 progression 추천은 제목 코사인을 사용하지 않는다. 제목 기반 related/fallback 경로의 제거는 별도 정책 변경으로 명시해야 하며 이번 통합 작업에서는 변경하지 않았다.

Suggested commit: `feat: integrate within-LeetCode comparison into learning workspace`.
