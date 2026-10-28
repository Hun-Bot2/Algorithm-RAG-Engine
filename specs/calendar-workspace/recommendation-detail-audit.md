# Recommendation detail and evidence audit

## Research summary
The primary journal plan uses progression candidate selection, not vector-score ranking. Related exploration in the base planner is separate and discarded by journalPlan except remediation/reviews; relatedCandidates uses neighbor rank in a weighted score but drops stored cosine values. explanationEvidence sends policy_reason and a small slot subset, omitting components/rank/raw cosine. A non-null policy reason is not evidence of vector-based selection.

## Requirements and design
- Recommendation details SHALL omit the solve-result recording action; actual activity workflows remain available.
- System-recommended problems SHALL carry a compact `추천받음` badge in recommendation rows/calendar/details and corresponding actual records when the recommendation predates them. Manual review labels MUST NOT create the badge.
- Evidence wording SHALL distinguish policy selection, rank-based related exploration, semantic fallback and unavailable evidence. No similarity values SHALL be fabricated.
- Recommendation ranking, policies, snapshots, cached Gemini/NVIDIA outputs and evidence contracts MUST remain unchanged in this UI/audit task.

## Tasks and validation plan
- [x] Audit current source and latest local snapshot read-only.
- [x] Implement badge provenance projection and modal action removal.
- [x] Add focused regression tests; run existing/full suites.
- [x] Verify actual Chrome modal/badges and responsive display using synthetic records.
- [x] Document findings and limitations; suggest commit.

## Risks
A snapshot records system selection, not proof of notification delivery or actual learning. Vectors in this catalog are title-based; cosine does not establish shared algorithms. Changing selection weights requires a separate evaluated policy version rather than adding plausible-sounding similarity to AI text.

## 확인 결과 (2026-10-23)

1. **현재 캘린더 추천은 학습 정책 기반이다.** `journalPlan`은 기본 플래너의 review 슬롯만 남기고, progression의 다음 문제를 추가한다. 고난도 실패시 remediation 예외가 있다. 따라서 기본 플래너의 related-channel 탐색이 존재해도 캘린더의 주된 추천 경로가 되지는 않는다.
2. **Easy/Medium 단계는 벡터 유사도순이 아니다.** `progression.candidateFor`는 확인된/AI 검토 개념, 난이도, 미시도·미노출, variation 조건으로 거른 뒤 문제 키 정렬의 첫 후보를 고른다. 일반 cross-platform 단계는 보수적 풀이확률로 정렬한다.
3. **related 경로는 코사인 값 대신 이웃 순위만 사용한다.** catalog reader는 `{dst,rank,score}`를 읽지만 relatedCandidates는 `{rank,channel}`만 보존한다. 점수는 reviewed_edge + concept_jaccard + channel_rank + difficulty_fit + penalties의 합이다. channel_rank는 `120/(60+rank)`를 반올림한 값이며 코사인 유사도가 아니다. 실제 코사인 값만 뒤집는 재현 테스트에서도 결과가 동일했다.
4. **제목 의미 유사 fallback도 수치 유사도순이 아니다.** progression의 semantic fallback은 title_v2 이웃 목록으로 후보를 좁힌 뒤 문제 키로 정렬한다. 이후 `semantic_only`로 표시하고 개념 숙련 증거로 인정하지 않는다.
5. **AI 설명 입력에서도 수치가 누락된다.** `explanationEvidence`는 policy_reason, stage, difficulty, concepts, semantic_only 등의 제한된 필드만 넘긴다. score components, source problem, channel rank, raw cosine은 넘기지 않는다. 두 모델은 동일한 제한된 근거를 받으므로, 현재 설명에 구체적인 유사도 수치가 없는 것은 UI의 단순 누락만은 아니다.
6. **원래 데이터 자체는 있다.** catalog.neighbors는 score를 저장한다. 카탈로그 빌더는 full-v2의 any/cross neighbor score 배열을 읽고, 해당 neighbor 생성기는 dense_cosine을 사용한다. 이 채널은 제목 기반 이웃이며 문제 본문의 알고리즘적 동등성으로 해석하면 안 된다.

확인한 최신 로컬 journal snapshot은 2026-10-23/catalog-v4/journal-v1이었다. 일반 추천 `leetcode:check-if-any-element-has-prime-frequency`는 leetcode_easy 단계이며, reason에 Easy 독립풀이 2/3·Medium 0/2와 미시도/미노출 조건이 기록돼 있고 cosine/channel/components는 없었다. 사용자 DB는 SQLite 읽기 전용 연결로 조사했다.

## 구현과 검증

- 추천 상세: 결과 기록 버튼 제거, 원문과 Gemini/NVIDIA 비교 유지, 선정 경로 안내와 추천받음 배지 표시.
- 캘린더·선택일 목록·풀이 이력·문제 상세: 시스템 추천 provenance 기반 배지. 수동 복습·미래 추천·확인 가능한 추천시각보다 앞선 시도는 배지에서 제외. 배지는 추천 선택 이력을 뜻하며 알림 전송 영수증이 아니다.
- 변경 파일: hosted/public/workspace.mjs, workspace-data.mjs, styles.css; hosted/test/workspace-data.test.mjs와 recommendation-audit.test.mjs; 이 문서.
- 기존 workspace 관련 테스트 15개를 먼저 실행했다. 배지/근거 구분 테스트 후 17개 통과. 현재 로직을 재현하는 조사 테스트 3개 통과.
- 실제 Chrome 합성 시나리오 3개: 모달 배지, 결과 버튼 부재, 원문 유지, AI 비교 접근, 데스크톱/390px 모바일 확인. 모바일 스크린샷 직접 확인. /tmp/recommendation-modal-browser/receipt.json 및 PNG. 실제 모델 호출/학습 DB 쓰기 없음.
- 이전 광범위 브라우저 테스트의 만료된 대회 픽스처는 이번 조사 범위 밖이다. 이번 브라우저 검증은 대회 구간을 제외한 별도 합성 시나리오다.

## 후속 설계 제안 (미구현)

정책 적합성을 후보 자격 조건으로 유지한 뒤, 사용자가 풀었던 특정 문제와의 관계를 재순위화하는 별도 버전이 필요하다. source_key, catalog/model/channel, cosine, rank, shared concepts와 점수 기여도를 슬롯 및 동결 AI 근거에 함께 저장해야 실제 선정 이유로 설명할 수 있다. 가중치·회귀 평가 없이 현재 추천에 사후 유사도를 붙이면 선정 이유를 잘못 설명하게 된다. 이번 변경은 추천 순위·정책·캐시를 바꾸지 않았다.

Suggested commit: `fix: clarify recommendation provenance and detail actions`.

Final validation: full Node regression suite 209/209 passed; `npm run build` passed. No ranking/backend changes, deployment or commit were performed. Static UI updates apply on page reload.

## Follow-up: icon and inline comparison
Requirements: replace textual recommendation badge with an accessible icon, put a verified HTTPS source link directly beside the modal problem title, and render both cached provider outputs immediately in the recommendation modal. Keep evidence disclosures, provider failures, fallback, focus behavior and read-only cache viewing. No vector-ranking implementation is authorized by the discussion request. Affected files: workspace.mjs, styles.css, browser checks. Validation: baseline/full Node tests plus Chrome desktop/mobile, source link, Escape/focus and no new provider calls.

Inline comparison follow-up implementation: recommendation icon replaces visible badge text and carries role=img/accessible label/tooltip. Problem modals put the original HTTPS source action beside the title. Recommendation modal embeds both cached provider outputs directly and remains wide; comparison button/second modal removed, original evidence disclosure retained. User clarified all three recommendation purposes must be considered separately. No vector ranking changes were made.

Discussion state (proposal, not approved implementation):
- Same solving idea: compare technique/solution structure grounded in learner notes/code and reviewed relationships; title cosine alone is insufficient.
- Weakness remediation: derive a learning need from actual unsuccessful attempts and explicit learner evidence, then identify relevant practice; this is not itself a cosine-similarity metric. A review flag alone does not establish weakness.
- Similar problem content: compare permitted problem statements (input/output/constraints/task), distinctly from solving-idea evidence. Current catalog title neighbors cannot be relabeled statement similarity; quarantined statement sources remain unavailable.
- Show recommendation type and its own evidence, source problem, representation/model, cosine (if genuinely computed), and policy contribution separately. Do not average incompatible spaces or claim one universal similarity threshold. Representation/source selection and calibration remain open discussion topics.

Inline follow-up validation: baseline workspace tests 17/17; final full root-run Node suite 209/209 and build passed. Running the full suite from hosted/ first failed on root-relative private catalog fixture paths; corrected to the documented root invocation. Chrome synthetic scenarios 3/3 passed with immediate dual explanations, accessible icon, title-adjacent original link and no comparison button; final mobile screenshot visually inspected at /tmp/inline-recommendation-browser-final/. Existing cache views made no new model requests. Suggested commit: `refactor: show recommendation explanations inline with source links`.
