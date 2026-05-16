# project-evaluator 설계 문서

- 날짜: 2026-05-16
- 상태: 설계 승인 대기 → 구현 계획(writing-plans)으로 전환 예정

## 1. 배경 / 목적

git URL을 입력하면 해당 레포를 클론/갱신하고 **코드 자체를 정확하게 평가**하는 대화형 에이전트.
평가의 중심축은 "기술 의사결정의 질":

- (a) 사용된 기술이 **의도된 목적대로** 쓰였는가
- (b) **관용적/올바른** 사용인가
- (c) **오버엔지니어링**(불필요한 추상화·MSA·메시지큐 등) 또는 언더엔지니어링인가

부가 가치: 평가 이력을 로컬에 남겨, 재실행 시 **변경/추가된 부분만 증분 재평가**하고 델타를 보여준다.

레포 코드를 실행하지 않는다(빌드/테스트 안 함, "안 돌아가도 됨"). 정적 읽기 + LLM 추론 + 베스트프랙티스 웹검색만 사용.

### 스코프에서 제외 (명시적으로 안 함)

- 코드 품질 린팅, 테스트 실행/커버리지, CI 분석, 보안/CVE 스캔 서브에이전트
- 레포 코드 실행, 샌드박스/Docker
- 웹 UI(코어는 UI 무관으로 설계하여 추후 추가는 가능하나 이번 범위 아님)

## 2. 기술 스택 / 핵심 결정

| 항목 | 결정 |
|---|---|
| 에이전트 프레임워크 | deepagents (LangChain, LangGraph 기반) |
| 모델 | OpenAI 고정. 기본 `openai:gpt-5.4`, `EVALUATOR_MODEL` env로 교체 가능. `OPENAI_API_KEY` 사용 |
| 인터페이스 | CLI TUI — `rich` 기반 스트리밍 REPL 채팅 |
| 영속화 | 로컬 SQLite 단일 파일. 평가 이력은 자체 스키마(§5.2), 대화 메모리는 LangGraph `SqliteSaver`(별도 테이블, 동일 DB 파일) |
| 증분 diff 단위 | **파일 단위** (git diff가 파일 기반이라 자연스러움) |
| 언어 | Python 3.11+ |
| 코어 분리 | 에이전트·도구·영속화는 UI 무관 라이브러리, `cli.py`는 얇은 클라이언트 |

## 3. 아키텍처

대화형 CLI TUI. TUI는 얇은 터미널 클라이언트(스트리밍 채팅 + 마크다운 리포트 렌더). 두뇌는 deepagents 에이전트 1개(메인 오케스트레이터).

- LangGraph 체크포인터(SQLite)로 대화 메모리 유지 → "그 부분 더 자세히" 같은 후속 질문 가능
- 메인 에이전트 진행 흐름: git URL 접수 → `write_todos`로 계획 → `task`로 서브에이전트 위임 → 종합 평가 → SQLite 기록 → 후속 Q&A
- deepagents 빌트인 사용: `write_todos`(계획), 가상 FS(`ls/read_file/write_file/edit_file`), `task`(서브에이전트 호출). `execute`/샌드박스는 **사용 안 함**

## 4. 서브에이전트 (3개, deepagents `subagents` dict)

각 서브에이전트는 전용 `system_prompt` + 전용 `tools`를 가짐. 메인은 최종 결과만 수신(컨텍스트 격리).

### 4.1 `recon` (정찰)

- 역할: 레포를 캐시 워크스페이스에 클론/갱신 → 언어, 디렉터리 구조, 빌드 시스템, 엔트리포인트, 규모 지도화. 결과를 가상 FS에 기록해 다른 서브에이전트가 공유
- 도구: `clone_or_update_repo(url, ref=None)`(레포별 캐시 워크스페이스 유지, 현재 commit SHA + 직전 평가 대비 변경 파일 목록 반환), 가상 FS 읽기
- 거대 레포 처리: `node_modules`, `.git`, vendored, 빌드 산출물 스킵; 파일 수·크기 상한; 큰 파일 샘플링

### 4.2 `dependency-stack` (의존성·기술스택)

- 역할: 매니페스트 파싱(`package.json`, `pyproject.toml`, `requirements.txt`, `go.mod`, `pom.xml`, `Cargo.toml` 등) → 프레임워크/라이브러리/버전 인벤토리
- 도구: 매니페스트 파서(읽기 전용 파싱, 네트워크/CVE 조회 없음)

### 4.3 `tech-fit` (기술적합성·오버엔지니어링) ★핵심

- 역할: 주요 기술/패턴별로 (a) 목적부합 (b) 관용적 정확성 (c) 과/소설계를 `file:line` 근거 + 근거 설명과 함께 판정
- 도구: 타깃 코드 리드(`read_file`, grep), 베스트프랙티스 웹검색 — deepagents 런타임용 **커스텀 도구**로 구현(실제 검색 백엔드: Tavily 또는 DuckDuckGo, env로 설정, API 키 없으면 우아하게 비활성화). Claude Code 빌트인 `WebSearch`/`WebFetch`는 배포 런타임에 없으므로 사용하지 않음

## 5. 영속화 (SQLite)

### 5.1 메인 에이전트용 커스텀 도구

- `get_repo_history(url)` → 직전 평가: commit SHA, 요약, 파일별 판정, 기술 findings
- `diff_since_last(url)` → 캐시 워크스페이스의 git diff로 변경/추가/삭제 파일 목록
- `record_evaluation(url, commit_sha, summary, file_evals[], tech_findings[])` → 새 평가 행 기록

### 5.2 스키마

```sql
repos(
  id INTEGER PK, url TEXT UNIQUE, name TEXT,
  workspace_path TEXT, created_at, updated_at
)
evaluations(
  id INTEGER PK, repo_id FK, commit_sha TEXT,
  created_at, overall_summary TEXT, model TEXT
)
file_evaluations(
  id INTEGER PK, evaluation_id FK, repo_id FK,
  file_path TEXT, content_hash TEXT,
  verdict TEXT, notes TEXT, tech_tags TEXT
)
tech_findings(
  id INTEGER PK, evaluation_id FK,
  technology TEXT, purpose_fit TEXT, correctness TEXT,
  overengineering TEXT, rationale TEXT, evidence TEXT
)
```

시작 시 버전드 스키마 초기화(간단 마이그레이션).

## 6. 데이터 흐름

### 6.1 첫 실행

URL → 메인이 `write_todos` 계획 → `recon` 클론(commit SHA 기록)·지도 작성 → `dependency-stack` 매니페스트 파싱 → `tech-fit` 파일/기술별 평가 → 메인이 전체 종합 → `record_evaluation` 저장 → TUI에 리포트 렌더 → 후속 Q&A.

### 6.2 재실행 (같은 URL)

`get_repo_history` + `diff_since_last` → `recon`이 워크스페이스 갱신(`git pull`, 기본 최신 HEAD, ref 지정 가능) → **변경/추가 파일만** recon 지도 갱신 + `tech-fit` 재평가, **삭제** 파일은 제거, **안 바뀐** 파일은 저장된 판정 재사용 → 메인이 **델타 리포트** 생성(무엇이 바뀜 / 재평가 결과 / 전체 평가가 어떻게 이동) → commit 참조하는 새 평가 행 기록.

## 7. 에러 처리

| 상황 | 처리 |
|---|---|
| 잘못/접근불가 URL | 친절 메시지, DB 미기록 |
| 비공개 레포(인증 필요) | 클론 실패 감지 → 자격증명 안내(human-in-the-loop) |
| 거대 레포 | 스킵 규칙 + 파일 수/크기 상한 + 큰 파일 샘플링 |
| 모델/API 오류 | 백오프 재시도, 명확히 표면화 |
| 서브에이전트 부분 실패 | 메인이 최선 리포트 + 누락 영역 표시 |
| DB | 시작 시 버전드 스키마 초기화 |

## 8. 테스트 전략 (superpowers TDD)

- 단위(네트워크 없음, 로컬 임시 git 픽스처):
  - SQLite 도구: `record_evaluation` / `get_repo_history` / `diff_since_last`
  - git 워크스페이스 캐시 + diff 로직(첫 실행 vs 재실행, 변경/추가/삭제)
  - 매니페스트 파서(샘플 매니페스트 → 인벤토리)
  - 프로젝트 지도 빌더(스킵 규칙·상한)
- 서브에이전트 판정: 특성 알려진 소형 픽스처 레포(예: 의도적으로 과설계한 토이 레포)로 verdict가 오버엔지니어링을 지적하는지 검증. LLM 의존 테스트는 최소화·플래그 처리
- 오케스트레이션 로직: LLM 목으로 분기 검증

## 9. 프로젝트 레이아웃

```
src/project_evaluator/
  __init__.py
  cli.py              # TUI 진입점 (rich 스트리밍 REPL)
  agent.py            # create_deep_agent, 메인 시스템 프롬프트, 와이어링
  config.py           # 모델·경로·상한 (env 기반)
  subagents/
    recon.py
    dependency_stack.py
    tech_fit.py
  tools/
    repo.py           # clone_or_update_repo, diff_since_last (git)
    persistence.py    # SQLite store + 에이전트용 도구
    manifests.py      # 매니페스트 파서
    websearch.py      # 베스트프랙티스 검색 래퍼
  db/
    schema.sql
    store.py
tests/
docs/superpowers/specs/2026-05-16-project-evaluator-agent-design.md
pyproject.toml        # console script: pe / project-evaluator
```

의존성: `deepagents`, `langchain-openai`, `gitpython`, `rich`, `pydantic`, `pytest`.
패키징: `pyproject.toml`, 콘솔 스크립트 `pe` / `project-evaluator`, Python 3.11+.

## 10. 미해결/추후 결정 (이번 범위 밖)

- 웹 UI 추가(코어 UI 무관 → 추가는 비파괴적)
- 다중 레포 비교/대시보드
- 평가 기준 커스터마이징(사용자 정의 루브릭)
