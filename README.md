# QE Agents

LangGraph 기반 QE(Quality Engineering) 에이전트 파이프라인.
실행 중인 SUT의 **라이브 OpenAPI 스펙만으로 grounding**하여(소스코드는 읽지 않음 — 스펙 = 의도, 런타임 = 실제 동작), **분석 → 테스트 플랜 → 테스트 생성 → HITL 리뷰 → 샌드박스 실행 → 결함 트리아지**의 end-to-end 슬라이스를 수행합니다.

```
GET /openapi.json ─▶ ground ─▶ plan ─▶ [HITL: ambiguity] ─▶ generate ─▶ static check
                                                                ▲            │
                                              (revise: feedback)┤            ▼
report ◀─ triage(Auditor) ◀─ execute(Executor) ◀─ approve ─[HITL: test review]─ edited ┐
   ▲                                                            │                      │
   └──────────────────────────────────────────── abort ─────────┘   (re-validate) ─────┘
```

- **Grounding & Analysis** — 라이브 스펙에서 데이터 구조/비즈니스 규칙을 파악하고, 문서화된 규칙과 추론된 불변식(INFERRED)을 구분해 시스템 모델을 구축.
- **Test Planning** — 리스크 기반 시나리오 12~16개: boundary, negative, 5xx-hunting, state transition, idempotency, **concurrency**. 스펙이 침묵하는 부분은 추측하지 않고 질문으로 표면화(HITL).
- **Test Generation** — 시나리오당 pytest 모듈. 구문 검사(fast-fail)를 통과한 코드만 리뷰 게이트에 도달.
- **HITL Review Gate** — QE가 생성 코드를 열람/제외/직접 수정/피드백과 함께 재생성 요청/중단. 수정·재생성된 코드는 구문 검사를 재통과.
- **Execution (Executor)** — **Docker 샌드박스**에서 실행: 러너 컨테이너는 외부 이그레스가 차단된 internal 네트워크에 격리되어 SUT 컨테이너에만 도달 가능(read-only rootfs, cap-drop ALL, non-root, pids/mem/cpu 제한). 실패 시 재시도로 flaky 증거 수집.
- **Defect Triaging (Auditor)** — 실패 시그니처 클러스터링 → real/flaky/test_bug 분류(재시도 통과 시 flaky 하드 룰), 심각도/우선순위, 블랙박스 root cause 가설, owner 추정.

SUT는 `sut/`의 광고 캠페인 API로, `sut/bugs.yaml`에 라벨링된 planted bugs(7개) + 재현 가능한 flakiness(2개)를 포함합니다. 이 매니페스트는 평가(eval) 전용 ground truth이며 에이전트는 스펙 외에 아무것도 읽지 않습니다.

## Setup

요구사항: Python 3.12+, **Docker** (샌드박스 실행에 필수).

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
cp .env.example .env   # GEMINI_API_KEY 입력
```

## Run

```bash
# 데모: HITL 게이트 대화형 (ambiguity 응답, 테스트 리뷰/수정/재생성)
.venv/bin/qe run

# 비대화형 (모호성은 가정으로 진행, 리뷰 자동 승인)
.venv/bin/qe run --auto
```

SUT 컨테이너와 샌드박스 네트워크는 `qe run`이 자동으로 관리합니다
(첫 실행 시 샌드박스 이미지 빌드로 ~1분 소요).

산출물은 `.qe_runs/<timestamp>/`에 저장됩니다: `openapi.json`(grounding 입력 스냅샷), 생성된 테스트, `report.md`, `defects.json`.

**Exit codes (SDLC/CI 통합):** `0` 결함 없음/flaky만, `2` real 결함 발견, `3` 리뷰 게이트에서 중단.

## Test / Lint

```bash
.venv/bin/pytest          # 유닛 테스트 (LLM/네트워크 불필요)
.venv/bin/ruff check .
.venv/bin/black --check .
```

## Layout

| path | 내용 |
|---|---|
| `src/qe_agent/graph.py` | LangGraph 오케스트레이션 (HITL interrupt 2곳 + 리뷰 피드백 루프) |
| `src/qe_agent/stages/` | grounding·planning / generation / review·execution / triage 노드 |
| `src/qe_agent/security.py` | 프롬프트 인젝션 방어 (스펙 spotlighting·스캐너) + 구문 검사 |
| `src/qe_agent/sandbox.py` | Docker 샌드박스 (internal 네트워크 격리, JUnit 파싱) |
| `docker/Dockerfile.sandbox` | SUT/러너 공용 샌드박스 이미지 |
| `sut/` | 테스트 대상 API + `bugs.yaml` ground truth |
| `eval/` | 평가 하네스 (Phase 3) |
