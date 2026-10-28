# Algorithm RAG Engine — Adaptive Algorithm Mastery System

## 현재 상태 (2026-10-18)

백준이 서비스를 종료해서 아래 "초기 파이프라인"은 입력(백준 풀이 노트)을 잃었고, 그 GitHub Actions 워크플로우 2개
(`daily_algorithm_pipeline.yml`, `publish_docker.yml`)는 삭제했습니다(git 이력에서 복구 가능). 현재 방향은 다음과 같습니다.

- **로컬 우선 학습 시스템** (`hosted/`, `specs/mastery-system/`): LeetCode, Codeforces, AtCoder 풀이 기록을 읽고 개인 맞춤 연습을 계획합니다.
  서버 없이 로컬에서 동작하며 유료 API가 필요하지 않습니다(추천 설명용 Gemini/NVIDIA 호출은 `explain` 명령에서만 선택적으로 실행). 날짜별 학습 일지, O/X 기록, 우선 복습과 Easy → Medium → 전이 → 지연 복습을 지원합니다.
  사용법은 [학습 일지 시작 안내](specs/mastery-journal/runbook.md)를 보세요.
- **풀이 기록 저장소**: 별도 private repo `algorithm-solution`. 이 저장소의 커밋을 `node hosted/cli.mjs repo ingest`가 읽습니다.
  push 없이 로컬 Git을 읽으며, 현재 커리큘럼 추천은 `node hosted/cli.mjs journal` 또는 로컬 학습 일지에서 확인합니다. 기존 `plan`의 일별 동결 계약도 보존합니다.
- **연구** (`research/`): 세 플랫폼 24,297문제의 제목 기반 유사도 지도, 언어 편향 보정, AtCoder 개념 분류. 결과는 `research/*/REPORT.md`와 `specs/`에 있습니다.
- 초기 파이프라인 코드(`src/main/`, `Dockerfile`, `docker-compose.yml` 등)는 2026-10-18에 삭제했고 git 태그 `legacy-pipeline-final`에서 복구할 수 있습니다.


### 학습 일지 실행

```sh
# 아직 카탈로그를 선택하지 않았다면 최초 1회 (기존 선택은 유지 가능)
node hosted/cli.mjs catalog use --version catalog-v4
node hosted/local.mjs
# http://127.0.0.1:8787
```

기존 SQLite를 사용하며 실행 위치와 관계없이 CLI와 같은 기본 DB 경로를 씁니다.
미해결 풀이도 다음 날 우선 복습에 들어갑니다. 도움 사용 여부는 기록하지 않으며 완료한 풀이는 모두 해결로 셉니다([ADR 005](specs/adaptive-learning/adrs/005-solved-only-outcomes.md)). 성공한 복습과 새로운 전이 문제의
독립 해결은 구분합니다. 테스트는 합성 학습 이력으로 수행했으며 실제 교육 효과를
입증한 것은 아닙니다. [설계·검증 기록](specs/mastery-journal/validation.md).

### 캘린더 워크스페이스 (2026-10-20)

첫 화면은 실제 풀이 시각을 기준으로 한 월간 캘린더입니다. 주간 보기, 날짜별 상세,
검색 가능한 풀이 이력, 원본 기록으로 내려가는 학습 분석과 시간순 대회 목록을 제공합니다.
조회된 대회는 메인 캘린더의 실제 날짜에도 표시됩니다. 날짜 선택은 오른쪽 패널을 갱신하고,
더보기는 하루 상세 모달을 엽니다. 월간 활동과 해결·미해결 분포도 기록에 맞춰 확인할 수 있습니다.
추천은 점선 예정 항목이며 완료로 집계하지 않습니다. 수동 기록 수정은 기존 시도를
보존하는 후속 기록입니다. 가져온 Git 기록은 원본 저장소에서 수정합니다.

Gemini/NVIDIA의 저장된 설명·오류·기본 설명·추천 근거는 추천 상세의 비교 모달에서
확인합니다. 화면 조회는 모델 호출을 실행하지 않습니다. 목표·타이머·백업은 학습 도구에 있습니다.
시간대는 현재 정책과 같은 Asia/Seoul 고정입니다. CF/AtCoder는 기존 캐시·동기화 설정을
사용합니다. 로컬 대회 조회 기능이 제공되는 경우 세 플랫폼 일정을 명시적으로 동기화할 수 있습니다.
LeetCode 커뮤니티 미러와 공식 일정은 출처를 구분하며, 수동 일정도 보존합니다.

[감사와 디자인 근거](specs/calendar-workspace/design.md) · [최신 개선·검증 및 제한](specs/calendar-workspace/refinement-2.md)

아래 내용은 2026-01 시점의 초기 파이프라인 기록입니다.

---

백준 풀이 기록을 분석해 유사한 LeetCode 문제를 자동 추천하고, 매일 07:00 KST에 Slack으로 알림을 보냅니다.

## 아키텍처 개요
- Heavy Job (`src/main/generate_index_map.py`):
	- Baekjoon MDX → OpenAI 임베딩(text-embedding-3-small) → Numpy(LeetCode 인덱스) 유사도 검색 → GPT-4o-mini로 추천 이유 생성 → `artifacts/recommendation_map.json` 저장
- Light Job (`src/main/slack_bot_daily_review.py`):
	- JSON 로드 → Slack Webhook으로 메시지 전송
- GitHub Actions: 워크플로우 삭제됨(2026-10-18). 이전에는 수동 실행(workflow_dispatch)만 지원했습니다.
- S3 사용(워크플로우에서만):
	- Job1: `aws s3 sync ./artifacts s3://hunbot-algo-rag-data/ --delete`
	- Job2: `aws s3 cp s3://hunbot-algo-rag-data/recommendation_map.json .`

## 프로젝트 개요
본 프로젝트는 개인 학습용 백준 알고리즘 풀이 기록(MDX)을 기반으로, OpenAI의 Embedding 모델과 Numpy 벡터 DB를 활용하여 가장 유사한 LeetCode 문제를 추천해주는 RAG(Retrieval-Augmented Generation) 시스템을 구축하는 것을 목표로 했습니다. 
단순한 추천에 그치지 않고 매일 자동으로 실행되어 슬랙으로 알림을 보내는 자동화된 파이프라인을 설계했습니다.

## 시스템 아키텍처 및 주요 기술 스택

* CI/CD 및 트리거: GitHub Actions를 사용하여 매일 정해진 시간에 파이프라인이 구동되도록 설정했습니다.
* 컨테이너화: 환경 격리와 의존성 관리를 위해 Docker를 사용했으며, 연산량이 많은 Heavy Job과 알림 전송용 Light Job으로 스테이지를 분리하여 효율성을 높였습니다.
* 데이터 적재 및 검색: LangChain 프레임워크를 기반으로 Numpy를 사용해 유사도 검색을 수행합니다.
* AI 모델: text-embedding-3-small 모델로 벡터화하고, gpt-4o-mini 모델을 통해 추천 사유를 생성합니다.

* 아티팩트 저장소: AWS S3를 중앙 저장소로 활용하여 대용량 인덱스 파일과 생성된 결과물(JSON)을 관리합니다.
* 모니터링: LangSmith를 연동하여 LLM의 추론 과정을 추적하고 품질을 관리합니다.

## 주요 기술적 과제와 해결 방안

* **Docker 이미지 빌드 시간 단축 및 최적화**: 초기에는 매 실행 시마다 PyTorch와 FAISS 같은 무거운 라이브러리를 설치하여 빌드 시간이 10분 이상 소요되는 문제가 있었습니다. → NumPy로 변경하고 더 속도가 빨라졌습니다.

이를 해결하기 위해 **GitHub Container Registry(GHCR)**를 도입하여 이미지를 사전에 빌드해두고, 실행 시에는 `docker pull`만 수행하도록 변경했습니다. 또한, GPU가 없는 CI 환경에 맞춰 **FAISS → NumPy 기반 벡터 검색**으로 전환하여 이미지 크기를 획기적으로 줄였습니다.

* **대용량 바이너리 파일 관리**: FAISS 인덱스 파일은 Git으로 관리하기에 부적절하여 **AWS S3**를 원본 저장소로 지정했습니다. -> Numpy로 바꾼 .npy, json파일도 **AWS S3**를 사용합니다.

GitHub Actions 실행 시점에만 S3에서 인덱스를 다운로드하고 Docker 볼륨으로 마운트하는 **하이브리드 스토리지 전략**을 통해 보안과 접근성을 모두 확보했습니다.

* **불필요한 중복 연산 제거**: 매번 모든 알고리즘 문제 파일을 다시 임베딩하는 비효율을 개선하기 위해, **GitPython**을 활용하여 가장 최근 커밋(`HEAD`)에서 변경된 파일만 감지하여 처리하는 로직을 구현했습니다. 이를 통해 API 호출 비용을 절감하고 처리 속도를 극대화했습니다.

## 최종 워크플로우 흐름

* **GitHub Actions 스케줄러**가 작동하여 워크플로우를 시작합니다.
* **GHCR**에서 사전에 빌드된 최적화된 Heavy Job 이미지를 다운로드(Pull)하고, **S3**에서 NumPy 인덱스를 가져옵니다.
* **Heavy Job 컨테이너**가 실행되며 Git 히스토리를 분석해 **최신 커밋된 알고리즘 문제 파일만 선별**합니다.
* 선별된 문제에 대해 임베딩 및 유사도 검색을 수행하고, 추천 결과(JSON)를 **S3**에 업로드(Sync)합니다.
* **Light Job 컨테이너**가 S3에서 JSON을 받아 Slack API를 통해 지정한 형식의 시각화된 메시지를 전송합니다.

### 최적화 성과

* **실행 시간 단축**: 사전 빌드 이미지 도입과 의존성 최적화를 통해 전체 파이프라인 실행 시간을 초기 약 12분에서 **2분 미만**으로 80% 이상 단축했습니다.
* **비용 절감**: 변경된 파일만 처리하는 로직을 통해 OpenAI API 호출 비용을 최소화했습니다.

### 향후 발전 방향

- 현재는 고정된 LeetCode 인덱스를 사용하고 있으나, 향후에는 새로운 문제가 추가될 때마다 인덱스를 자동으로 갱신하는 파이프라인을 추가할 계획입니다. 또한 슬랙 메시지에 인터랙티브 버튼을 도입하여 사용자 피드백을 수집하고, 이를 기반으로 추천 정확도를 높이는 루프를 구상 중입니다.
- Python 코드 내부의 병렬 처리 (Parallelism) 기법을 도입하여 대량의 알고리즘 문제를 동시에 처리하는 기능도 검토하고 있습니다. -> 공부한 뒤 적용할 계획입니다.
- 다른 사용자들이 사용할 수도 있어서(?) 최대한 범용적으로 바꿔보려고 합니다.

## 주요 파일 (초기 파이프라인, 삭제됨: 태그 `legacy-pipeline-final`)
- `src/main/generate_index_map.py` — Baekjoon MDX → OpenAI 임베딩 → NumPy 검색 → GPT-4o-mini 이유 → JSON 저장
- `src/main/slack_bot_daily_review.py` — JSON 로드 → Slack Webhook 알림
- `Dockerfile` — heavy/light 멀티스테이지 빌드 (heavy:  NumPy+LangChain, light: Slack 최소 의존성)
- `.github/workflows/daily_algorithm_pipeline.yml` — 삭제됨(2026-10-18, git 이력 참고)
- `indexes/faiss_leetcode/ → indexes/numpy_leetcode/` — 사전 구축된 LeetCode → NumPy 인덱스 (필수)
- `artifacts/recommendation_map.json` — 생성된 추천 맵 (S3 업로드 대상)

---

아래 내용은 해당 알고리즘 앱을 범용적으로 만들게 되면, 바꿀 부분입니다. 지금은 개인으로 사용하기에 불편함이 없습니다..

### GitHub Repository Secrets 설정
Repository Settings → Secrets and variables → Actions → New repository secret에서 다음 키들을 추가:

| Secret 이름 | 설명 | 획득 방법 |
|------------|------|----------|
| `OPENAI_API_KEY` | OpenAI API 키 (임베딩 + GPT-4o-mini) | [OpenAI Platform](https://platform.openai.com/api-keys) |
| `SLACK_WEBHOOK_URL` | Slack Incoming Webhook URL | [Slack App 설정](https://api.slack.com/messaging/webhooks) |
| `AWS_ACCESS_KEY_ID` | AWS S3 접근용 액세스 키 ID | AWS IAM 사용자 생성 → Access Key 발급 |
| `AWS_SECRET_ACCESS_KEY` | AWS S3 시크릿 액세스 키 | 위와 동일 |
| `LANGSMITH_API_KEY` | LangSmith 추적용 API 키 (선택) | [LangSmith 설정](https://smith.langchain.com/) |


**Last Updated**: 2026-01-15

## Adaptive learning foundation (M1-M3)

The new private learning product has a mandatory no-AI core, separate from the
legacy batch pipeline above. Its [specifications](specs/adaptive-learning/requirements.md),
[architecture](specs/adaptive-learning/design.md) and
[milestones](specs/adaptive-learning/tasks.md) define the migration boundary.
Local append-only learner events, explicit confirmed timers and JSON restore now
use standard-library SQLite; see [local usage](specs/adaptive-learning/local-history.md).
[Asset manifests](assets/README.md) preserve originals and exclude unreviewed data
from hosted bundles. Run `python3 -m unittest discover -s tests -v` and
`python3 scripts/learning/assets.py verify` offline.

This foundation does not yet generate adaptive missions or supply a hosted mobile
dashboard, reviewed curriculum or new Slack transport. Deployment, schedules,
public registration and optional model training remain gated. Existing production
jobs retain their current OpenAI dependency and manual workflow trigger.


## Private adaptive pilot implementation

The no-AI pilot now includes a reviewed 90-problem curriculum, deterministic
practice/review/transfer policy, configurable weekly rhythm, mobile dashboard,
explicit study intervals and contest records, private backups and a durable
Slack outbox. Worker/D1/Access deployment code and local verification tools are
under [hosted/](hosted/); start with [deployment instructions](specs/adaptive-learning/deployment.md).
The existing batch production application is unchanged.

Local checks cover Python/Node, actual Chrome mobile flows and local workerd/D1.
The service is not deployed and Slack/schedules remain disabled pending account,
identity and destination selection. The optional [research tools](specs/adaptive-learning/research-contract.md)
retain sparse-label limitations and prevent current evaluation data from training
a model. No trained model, 30-day delivery result or learning benefit is claimed.
