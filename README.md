# SymphonAI

A coding-agent runtime for agent-native development. A leader model delegates to
subagents, and a dashboard keeps the person directing them on the loop rather
than in it.

## Why it exists

Agent-native development means the agents write the code and the person works
one level up, on intent and judgement. Three things that a single-agent chat
tool treats as incidental become the design here.

**Many models, by API, not one subscription.** A run is assembled from
providers: Anthropic, OpenAI, Gemini, and any OpenAI-compatible endpoint. The
leader and its subagents need not share a model or a vendor.

**Spec-driven development.** The leader hands a subagent a written spec with
file paths, symbols, the contract to satisfy, and the acceptance criteria. The
subagent hands back a report of what it changed, what it ran, and which criteria
it met.

```
the spec                     the report
  ## Goal                      ### Summary
  ## Scope                     ### Changed Files
  ## Context                   ### Validation
  ## Contract                  ### Acceptance Criteria
  ## Acceptance criteria       ### Risks & Notes
  ## Tests
  ## Validation
  ## Report
```

Both templates are in the repository:
[`specs/TEMPLATE.md`](specs/TEMPLATE.md) and
[`specs/REPORT-TEMPLATE.md`](specs/REPORT-TEMPLATE.md).

**Human on the loop, not in it.** Being in the loop means approving each tool
call and following every turn. Being on the loop means setting direction and
still being able to say what the agents did and what state the project is in,
without replaying anything. The app's dashboard shows the roadmap, and opening
an item puts the spec next to the report that came back.

## Running it

Python 3.11 or later and Node 18 or later. Node's built-in test runner checks
the JavaScript app, so there is no npm install. Install from a checkout with
`pip install .` (or `pip install -e .` for development). API keys come from the
environment or Settings › Models, never from a file in the repository.

```bash
python3 scripts/check.py                 # the check suite (--only selects a subset)
pip install .                            # install from this checkout
pip install -e .                         # editable install for development
pip install -e ".[tui]"                  # optional Textual UI
python3 scripts/tui.py                   # run it
symphonai app                            # open the app for the current project
symphonai app --repo-root /work/project --no-open  # print its address without opening a browser
python3 -m symphonai_host                # development entry point for the host
```

For development and desktop-shell integration, `python3 -m symphonai_host`
keeps its stdout JSON handshake as the contract used by the shell and scripts.

### Run without the app

Run one task from a project directory, or pass `--repo-root` to choose one.
Use `--json` for a machine-readable result. `ask` mode is the default; use
`--mode allow` when an unattended run should apply tool actions.

```bash
symphonai run --json "Run the test suite and summarize failures"
symphonai run --repo-root /work/project --mode allow "Fix the failing tests"
```

In a CI shell step, the command's exit code determines whether the step passes:

```bash
symphonai run --json "Run the test suite"
```

Exit codes are `0` for a final answer, `1` for a failed run, `2` for usage or
configuration errors, `3` when a turn limit or budget stops the run, and `130`
when interrupted.

### Desktop shell

The macOS Tauri shell uses the same files as the browser app and starts the
frozen host as a sidecar. Build the existing onedir host first, then build or
run the shell:

```bash
python3 scripts/build_host.py
cd packaging/tauri
cargo tauri build
# or, during shell development
cargo tauri dev
```

The Tauri bundle copies the complete target-suffixed host directory from
`dist/`; the executable and `_internal/` must remain together. The resulting
application launches without a system Python installation. Its shell receives
the bearer token through the piped handshake and owns the sidecar until exit.

## How it was built

The same way it describes. Claude Code writes the specs and reviews the diffs,
and Codex implements them. Every phase in the history went through that
exchange.

---

# SymphonAI (한국어)

Agent-native한 개발을 위한 코딩 에이전트 런타임입니다. 리더 모델이 서브에이전트에게 작업을 위임하며, 사용자는 에이전트들의 작업 현황과 계획을 보여주는 대시보드를 통해 Humman in the loop가 아닌 Humman on the loop로 개발 할 수 있습니다.

## 개발 배경

에이전트 네이티브 개발은 에이전트가 코드를 직접 작성하고, 사람은 상위 수준에서 의도 정의와 의사결정에 집중하는 개발 방식입니다. 기존 단일 에이전트 채팅 도구에서 부차적으로 다루던 세 가지 요소가 SymphonAI 설계의 핵심이 됩니다.

**단일 구독 대신 API 기반 멀티 모델 활용.** Anthropic, OpenAI, Gemini뿐만 아니라 모든 OpenAI 호환 엔드포인트를 조합하여 실행할 수 있습니다. 리더 모델과 서브에이전트가 반드시 동일한 모델이나 공급업체를 사용할 필요는 없습니다.

**스펙 주도 개발.** 리더 에이전트는 파일 경로, 심볼, 인터페이스 계약, 인수 기준을 명시한 스펙 문서를 서브에이전트에 전달합니다. 서브에이전트는 변경 사항, 실행한 검증 내역, 충족한 기준을 정리한 리포트로 결과를 회신합니다.

```
the spec                     the report
  ## Goal                      ### Summary
  ## Scope                     ### Changed Files
  ## Context                   ### Validation
  ## Contract                  ### Acceptance Criteria
  ## Acceptance criteria       ### Risks & Notes
  ## Tests
  ## Validation
  ## Report
```

두 템플릿 문서는 저장소의 [`specs/TEMPLATE.md`](specs/TEMPLATE.md)와
[`specs/REPORT-TEMPLATE.md`](specs/REPORT-TEMPLATE.md)에서 확인할 수 있습니다.

**Humman in the loop가 아니라 Humman on the loop.** 루프 안에 머무른다는 것은 도구 호출마다 일일이 승인하고 모든 턴을 직접 따라가야 함을 의미합니다. 반면 루프 위에서 조율한다는 것은 작업의 방향을 설정한 뒤, 대화 기록을 일일이 되짚지 않고도 에이전트가 수행한 작업과 프로젝트의 현재 상태를 즉각 파악할 수 있음을 뜻합니다. 앱 대시보드에서 전체 로드맵을 확인할 수 있으며, 특정 항목을 열면 작성된 스펙과 반환된 리포트를 나란히 비교하여 검토할 수 있습니다.

## 실행 방법

Python 3.11 이상 및 Node 18 이상 환경이 필요합니다. JavaScript 앱은 Node의 내장 테스트 러너로 검증하므로 별도의 `npm install` 과정이 필요하지 않습니다. 체크아웃에서 `pip install .`로 설치하고, 개발 중에는 `pip install -e .`를 사용합니다. API 키는 환경 변수 또는 설정 › 모델에서 관리하며 저장소 파일에서는 읽지 않습니다.

```bash
python3 scripts/check.py                 # 전체 검사 스위트 실행 (--only 옵션으로 일부 선택 가능)
pip install .                            # 현재 체크아웃에서 설치
pip install -e .                         # 개발용 editable 설치
pip install -e ".[tui]"                  # 선택 사항: Textual UI 설치
python3 scripts/tui.py                   # TUI 실행
symphonai app                            # 현재 프로젝트의 앱을 브라우저에서 열기
symphonai app --repo-root /work/project --no-open  # 브라우저를 열지 않고 주소 출력
python3 -m symphonai_host                # 개발용 호스트 진입점
```

개발 및 데스크톱 셸 연동에서는 `python3 -m symphonai_host`를 사용하며 stdout의 JSON 핸드셰이크는 셸과 스크립트가 사용하는 계약입니다.

### 데스크톱 셸

macOS용 Tauri 셸은 브라우저 앱과 동일한 파일을 사용하며, 사전 빌드된 호스트를 사이드카 프로세스로 구동합니다. 먼저 onedir 형태의 호스트를 빌드한 뒤 셸을 빌드하거나 실행합니다.

```bash
python3 scripts/build_host.py
cd packaging/tauri
cargo tauri build
```

Tauri 번들은 `dist/` 디렉터리에서 타깃 접미사가 붙은 호스트 디렉터리 전체를 복사합니다. 실행 파일과 `_internal/` 디렉터리는 반드시 함께 유지되어야 합니다. 이렇게 빌드된 애플리케이션은 시스템에 별도의 Python이 설치되어 있지 않아도 단독 실행됩니다. 셸은 파이프로 연결된 핸드셰이크를 통해 베어러 토큰을 전달받으며, 앱이 종료될 때까지 사이드카 프로세스를 관리합니다.

## 개발 방식

이 문서에 기술된 방식 그대로 개발되었습니다. Claude Code가 스펙을 작성하고 변경 사항을 검토하면, Codex가 이를 구현합니다. 커밋 히스토리에 기록된 모든 개발 단계가 이 협업 과정을 거쳐 완성되었습니다.
