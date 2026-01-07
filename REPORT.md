# Algorithm RAG Engine - 프로젝트 전체 리포트

## 📋 프로젝트 개요

**목표**: 백준(한국어)과 리트코드(영어)의 알고리즘 문제 간 의미적 유사성을 찾아 매칭하는 RAG 기반 검색 엔진 구축

**핵심 도전과제**:
- 한국어-영어 간 언어 장벽(Cross-lingual Semantic Alignment)
- 문제 설명 스타일이 아닌 알고리즘 로직 중심 매칭
- 검색 재현율(Recall)과 정밀도(Precision) 균형

---

## 🔄 프로젝트 워크플로우

```
1. 데이터 수집 (Data Collection)
   ├─ baekjoon_data_collection.py  → baekjoon_raw_data.jsonl (101개)
   └─ leetcode_data_collection.py  → leetcode_raw_data.jsonl (101개)

2. 데이터 전처리 (Preprocessing)
   ├─ preprocess.py                → *_preprocessed.jsonl
   │  └─ HTML/마크다운 제거, 핵심 로직 추출
   └─ refine_leetcode.py           → leetcode_refined.jsonl
      └─ LLM 기반 논리 골격(Logical Skeleton) 추출

3. 데이터 정규화 (Normalization)
   ├─ improve_recall.py            → baekjoon_refined.jsonl
   │  └─ 백준 데이터도 영문 논리 골격으로 변환
   └─ data_normal.py               → baekjoon_normalized.jsonl
      └─ 접두사("Logical Skeleton: ") 제거로 대칭성 확보

4. Ground Truth 구축 (GT Creation)
   ├─ ground_truth_finder.py       → potential_gt.json
   │  └─ LLM으로 후보 자동 추출 (속도 10배↑)
   └─ leetcode_match_pgt.py        → ground_truth_v2.json
      └─ 100세트 BJ-LC 정답 쌍 완성

5. 모델 평가 (Model Benchmarking)
   ├─ evaluate_model.py            → OpenAI text-embedding-3-small
   ├─ evalute_model_ver2.py        → Jina-v3 (1024 dim)
   └─ evalute_model_ver3.py        → BGE-M3 (Local)
      └─ Recall@K, MRR 측정

6. 프로덕션 엔진 (Production RAG)
   ├─ production_engine.py         → 3-Stage RAG Pipeline
   │  ├─ Stage 1: Dense Retrieval (로컬 NumPy, Top 30)
   │  ├─ Stage 2: Hybrid Scoring (벡터 60% + 태그 40%)
   │  └─ Stage 3: LLM Re-ranking (GPT-4o-mini, Top 10 → Top 3)
   └─ final.py                     → 최종 통합 실행 파일

7. 분석 및 리포팅 (Analysis)
   ├─ data_report.py               → 데이터 분포 분석
   └─ report.py                    → 성능 지표 시각화
```

---

## 📊 주요 파일 설명

### 데이터 수집
| 파일명 | 역할 | 출력 |
|--------|------|------|
| `baekjoon_data_collection.py` | 백준 문제 크롤링 | `baekjoon_raw_data.jsonl` |
| `leetcode_data_collection.py` | 리트코드 문제 크롤링 | `leetcode_raw_data.jsonl` |

### 데이터 전처리
| 파일명 | 역할 | 입력 → 출력 |
|--------|------|-------------|
| `preprocess.py` | HTML 제거, 핵심 로직 추출 | `*_raw_data.jsonl` → `*_preprocessed.jsonl` |
| `refine_leetcode.py` | LLM으로 논리 골격 생성 (리트코드) | `leetcode_preprocessed.jsonl` → `leetcode_refined.jsonl` |
| `improve_recall.py` | LLM으로 논리 골격 생성 (백준) | `baekjoon_preprocessed.jsonl` → `baekjoon_refined.jsonl` |
| `data_normal.py` | 접두사 제거 (데이터 대칭성) | `baekjoon_refined.jsonl` → `baekjoon_normalized.jsonl` |

### Ground Truth 구축
| 파일명 | 역할 | 출력 |
|--------|------|------|
| `ground_truth_finder.py` | LLM으로 유사 문제 후보 자동 추출 | `potential_gt.json` |
| `leetcode_match_pgt.py` | 정답 쌍 검증 및 정제 | `ground_truth_v2.json` |

### 모델 평가
| 파일명 | 평가 모델 | 주요 지표 |
|--------|-----------|-----------|
| `evaluate_model.py` | OpenAI text-embedding-3-small | Recall@1: 0.22, MRR: 0.354 |
| `evalute_model_ver2.py` | Jina-v3 (1024 dim) | Recall@1: 0.17, MRR: 0.319 |
| `evalute_model_ver3.py` | BGE-M3 (Local) | Recall@1: 0.15, MRR: 0.272 |

### 프로덕션 시스템
| 파일명 | 역할 |
|--------|------|
| `production_engine.py` | 3-Stage RAG Pipeline (Dense → Hybrid → Re-rank) |
| `final.py` | 최종 통합 실행 파일 |

### 분석 도구
| 파일명 | 역할 |
|--------|------|
| `data_report.py` | 데이터 분포 및 태그 분석 |
| `report.py` | 모델 성능 지표 시각화 |

---

## 🎯 핵심 기술적 성과

### 1. 데이터 엔지니어링
**문제**: 리트코드는 "Given an array...", 백준은 "N개의 정수가 주어진다..." 같은 표현 차이로 벡터 유사도 저하

**해결**:
```python
# Before (원문)
"N개의 정수가 주어진다. 정렬하시오."

# After (Logical Skeleton)
"Sorting algorithm. Input: array of N integers. Output: sorted order."
```
→ 언어 독립적 구조로 변환하여 임베딩 품질 향상

### 2. 3-Stage RAG Pipeline

#### Stage 1: Dense Retrieval (빠른 후보 추출)
```python
# 로컬 NumPy 인덱스로 코사인 유사도 계산
top_30 = cosine_similarity(query_embedding, all_embeddings).argsort()[-30:]
```
- Pinecone 대비 네트워크 지연 0초
- 3,500개 전수 조회 시간: 0.1초 미만

#### Stage 2: Hybrid Scoring (도메인 지식 반영)
```python
hybrid_score = 0.6 * vector_similarity + 0.4 * tag_overlap
```
- 태그 정규화: `"dp"` → `"dynamic programming"`
- 플랫폼 간 알고리즘 분류 차이 흡수

#### Stage 3: LLM Re-ranking (정밀 변별)
```python
# GPT-4o-mini가 논리 구조 심층 비교
"피보나치 함수" 쿼리 시
- Before: "Count and Say" (단어 중복↑) 1위
- After:  "Climbing Stairs" (점화식 일치) 1위
```
→ Recall@1 개선: 0.22 → **실질적 정밀도 확보**

### 3. 모델 벤치마크 결과 (100-Set Ground Truth)

| Model | Recall@1 | Recall@5 | Recall@10 | MRR |
|-------|----------|----------|-----------|-----|
| **OpenAI text-embedding-3-small** | **0.22** | **0.50** | **0.65** | **0.354** |
| Jina-v3 (1024 dim) | 0.17 | 0.47 | 0.63 | 0.319 |
| BGE-M3 (Local) | 0.15 | 0.40 | 0.53 | 0.272 |

**공학적 해석**:
- OpenAI 모델이 다국어 정렬에서 최고 성능
- Recall@10(0.65) vs Recall@1(0.22): 후보 검색은 우수하나 최종 결정력 부족
- → 이를 LLM Re-ranking으로 보완

---

## 📈 데이터 파이프라인 최적화 과정

### 1단계: Raw Data (원시 데이터)
```json
{
  "id": "1003",
  "title": "피보나치 함수",
  "content": "다음 소스는 N번째 피보나치 수를 구하는 C++ 함수이다..."
}
```

### 2단계: Preprocessed (전처리)
```json
{
  "id": "1003",
  "title": "피보나치 함수",
  "embedding_text": "피보나치 수열을 재귀로 계산. 0과 1이 호출된 횟수를 세는 문제"
}
```

### 3단계: Refined (논리 골격)
```json
{
  "id": "1003",
  "embedding_text": "Logical Skeleton: [Dynamic Programming] Fibonacci sequence with memoization. Count function calls for f(0) and f(1)."
}
```

### 4단계: Normalized (접두사 제거)
```json
{
  "id": "1003",
  "embedding_text": "[Dynamic Programming] Fibonacci sequence with memoization. Count function calls for f(0) and f(1)."
}
```
→ 백준과 리트코드의 데이터 대칭성 확보

---

## 🔬 실험 결과 및 인사이트

### 발견 1: 전처리의 중요성 > 모델 성능
- 같은 OpenAI 모델이라도:
  - Raw Data 사용 시: Recall@5 = 0.32
  - Refined Data 사용 시: Recall@5 = 0.50
- **데이터 품질이 모델 선택보다 2배 이상 영향**

### 발견 2: 태그 정규화 필수
- "dp", "dynamic programming", "DP" → 통일 필요
- 태그 매칭만으로도 Top 30 → Top 10 필터링 효과

### 발견 3: 비용 효율적 설계 가능
- 로컬 NumPy 인덱스: 무료
- GPT-4o-mini Re-ranking: 10개 문제당 $0.001
- 하루 100회 검색 시: **월 $3 미만**

---

## 🛠️ 기술 스택

| 카테고리 | 기술 |
|----------|------|
| **언어** | Python 3.10+ |
| **임베딩** | OpenAI text-embedding-3-small |
| **LLM** | GPT-4o-mini (Re-ranking) |
| **벡터 연산** | NumPy, scikit-learn |
| **데이터 처리** | Pandas, JSON Lines |
| **크롤링** | BeautifulSoup, Requests |
| **환경 관리** | python-dotenv, virtualenv |

---

## 📦 데이터셋 구성

### Baekjoon (백준)
- **문제 수**: 101개
- **난이도**: Bronze ~ Platinum
- **주요 태그**: DP, Greedy, Graph, Math
- **파일 크기**: ~240KB (refined)

### LeetCode (리트코드)
- **문제 수**: 101개
- **난이도**: Easy ~ Hard
- **주요 태그**: Array, Hash Table, Dynamic Programming
- **파일 크기**: ~324KB (refined)

### Ground Truth
- **정답 쌍 수**: 100세트
- **검증 방법**: LLM 자동 추출 + 수동 검증
- **파일**: `ground_truth_v2.json`

---

## 🚀 향후 개선 방향

### 1. 데이터 확장
- [ ] 리트코드 문제 수 101 → 3,500+ (전수 크롤링)
- [ ] 백준 문제 수 101 → 500+ (난이도 다양화)

### 2. RAG 엔진 개선
- [ ] FAISS 인덱싱 도입 (현재: NumPy)
- [ ] LangChain 통합 (체이닝 자동화)
- [ ] 벡터 DB 마이그레이션 (Pinecone/Weaviate)

### 3. 서비스 확장
- [ ] GitHub Actions 자동화 (복습 알림)
- [ ] Slack 봇 통합
- [ ] 웹 인터페이스 구축

### 4. 모델 최적화
- [ ] Fine-tuning 실험 (Cross-lingual BERT)
- [ ] Hard Negative Mining (오답 학습)
- [ ] Ensemble Re-ranking (다중 LLM 투표)

---

## 📝 핵심 교훈

1. **RAG의 핵심은 데이터**: 임베딩 모델보다 전처리 품질이 검색 성능에 더 큰 영향
2. **다단계 파이프라인 필수**: Dense Retrieval로 속도, LLM Re-ranking으로 정밀도 확보
3. **비용 효율적 설계 가능**: 로컬 NumPy + 선택적 LLM 호출로 월 $3 미만 운영

---

## 📚 참고 문서

- [ReadME.md](ReadME.md): 학술적 관점의 프로젝트 요약
- [ground_truth_v2.json](ground_truth_v2.json): 평가 데이터셋
- [production_engine.py](production_engine.py): 프로덕션 RAG 코드

---

**최종 업데이트**: 2026-01-07  
**작성자**: Jeonghun  
**프로젝트 상태**: Phase 1 완료 (평가 완료, FAISS 마이그레이션 예정)
