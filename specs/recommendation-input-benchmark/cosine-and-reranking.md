# 코사인과 재정렬 — 논의 메모 (2026-10-23)

사용자가 제공한 글에 대한 원자료 확인과 후속 실험 제안이다. 모델 설치/훈련, 새 임베딩, 추천 정책 변경은 실행하지 않았다.

## 현재 모델에 대한 확인

로컬에 고정된 all-MiniLM-L6-v2 README의 Training/Fine-tuning과 동일 revision의 공식 모델 카드는 batch 내 sentence pair cosine을 계산하고 참 pair와 cross entropy로 비교하는 contrastive objective를 명시한다. 따라서 "코사인으로 학습되지 않은 임의의 모델에 코사인을 붙였다"는 설명은 현재 모델에 해당하지 않는다. 일반 문장 관계를 학습한 모델이 제목만 보고 알고리즘 핵심 관계를 정확히 나타낸다는 보장도 없다. 현재 제목 코사인에 대한 사용자 관찰은 입력/목표 적합성 문제의 근거이며, 모든 코사인 사용의 무효를 입증하지는 않는다.

공식 근거: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md

## 글에서 구분해야 할 주장

- cosine과 cross entropy는 배타적인 훈련 목표가 아니다. cosine logits를 cross entropy에 넣을 수 있다.
- 단위 정규화한 벡터의 cosine은 dot product와 같다. Pearson은 좌표별 평균을 뺀 벡터끼리의 cosine이다. 모델 공간에 임의 중심화를 적용한다고 추천 관련성이 좋아지지는 않는다.
- cosine은 정답 확률이 아니다. 재정렬 모델의 sigmoid 출력 또한 사용자 과제에서 calibration을 확인하지 않았다면 정답 확률로 표시하지 않는다.
- Cross-encoder는 두 입력을 함께 처리하므로 캐시 벡터의 dot product보다 계산 구조가 무겁다. 기존 후보 중 재정렬할 수 있지만 retrieval이 누락한 후보를 복구할 수는 없다. 일반 검색용 reranker가 알고리즘적 동등성/학습 효과를 판단하는지는 별도 평가가 필요하다.
- "Text Embeddings Reveal (Almost) As Much As Text"는 embedding inversion/정보 노출에 관한 연구다. 알고리즘 추천 정확도나 특정 임베딩 입력의 우수성을 검증한 논문은 아니다.

Cross-encoder 근거: https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
논문: https://arxiv.org/abs/2310.06816

## 제안하는 세 목적별 입력과 평가

| 목적 | 표현할 정보 | 별도 판단 |
| --- | --- | --- |
| 문제 내용 유사 | 목표, 입출력, 제약, 문제 상황 | 과제/조건의 유사성 |
| 풀이 아이디어 연습 | 핵심 기법, 상태/불변식, 전이, 정당성, 복잡도 | 같은 기법을 같은 조건에서 적용하는지 |
| 약점 보완 | 실패/재시도/힌트 등 기록, 필요한 개념, 후보의 선수 조건 | 지금 학습자에게 유용한 방향인지 |

문제 내용/아이디어 관계는 대칭적인 pair 평가를 출발점으로 삼을 수 있지만, 학습 적합성은 learner→candidate 방향과 시점이 필요하다. source 근거 없는 전략을 제목만 보고 채우거나, 복습 표시를 곧 약점으로 간주하지 않는다. LLM으로 정리한 내용은 출처에 묶인 파생 설명이며 독립 검증과 구분한다.

## 비교할 후보 구조 (미구현)

1. 사용 가능한 원문/노트/검토 관계로 내용 표현과 아이디어 표현을 각각 구성한다. 제목은 식별/검색에 사용하고 현재 제목 벡터는 연구 기준선으로 보존한다.
2. 내용·아이디어·태그/어휘 채널에서 후보를 합친다. 각 채널의 점수/모델/출처를 유지한다.
3. 동일 후보군에 대해서 (a) 기존 검색 순서, (b) 구조화 입력의 bi-encoder, (c) Cross-encoder, 필요하면 (d) 제한된 LLM pair review를 비교한다. retrieval recall과 reranking 품질을 분리한다.
4. 내용 관계/아이디어 관계/학습 적합성 라벨을 따로 수집해 평가한다. 범용 reranker의 성능, 학습 효과, LLM 판단의 우월성은 가정하지 않는다.
5. 검증된 정적 문제 관계는 주간 cache로, 최신 학습 기록에 대한 적합성은 일간 재선정으로 연결한다.

현재 LeetCode 열람 모드는 제목과 native tags뿐이므로 본문/아이디어 입력 확보가 선행되어야 한다. 격리된 LeetCode 본문은 이 메모로 새 사용 허용이 되지 않는다.

## 비용 해석

앞서 24,303개 384D 저장 벡터의 전체 검색+Top30은 로컬 6.4초였다. 모델 추론을 하는 reranker에는 이 수치를 적용하지 않는다. 문제마다 상위 50개를 재정렬한다면 전체 카탈로그에서는 1,215,150 pair, LeetCode 3,053개만 대상으로 하면 최대 152,650 pair inference다. 이 수치는 작업 수이며 실행 시간은 모델·입력 길이·배치·기기에서 따로 측정해야 한다. 모든 카탈로그 문제에 대해 재정렬할지, 실제 학습 기준 문제만 처리할지도 별도 실험 조건이다.


## 2026-10-23 실행 후속

위 제안 중 작성된 핵심 설명/구조화 설명의 bi-encoder와 승인된 MS MARCO Cross-encoder 비교를 별도 연구로 실행했다. 질문형·설명 직접 조건 모두 실패 사례가 남았으며 사람 평가나 운영 승격은 하지 않았다. 실행 범위와 한계는 [실험 기록](../../research/recommendation_inputs/EXPERIMENTS.md), 재현 가능한 코드는 [실행 노트북](../../research/recommendation_inputs/recommendation_experiments.ipynb), 설명은 [블로그 초안](../../research/recommendation_inputs/BLOG.md)에 있다.
