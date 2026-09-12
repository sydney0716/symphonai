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
leader and its subagents need not share a model or a vendor. Tool-schema shaping
and model discovery are decided by wire format rather than by vendor name, so
adding a new OpenAI-compatible vendor is a matter of configuration rather than
code.

**Spec-driven development.** The leader hands a subagent a written spec with
file paths, symbols, the contract to satisfy, and the acceptance criteria. The
subagent hands back a report of what it changed, what it ran, and which criteria
it met. That exchange is the unit of work, and it is where the person's effort
goes. Writing the spec is the design work, and reading the report against it is
the review. A report can be checked against a spec, while a transcript can only
be read.

**Human on the loop, not in it.** Being in the loop means approving each tool
call and watching turns scroll past, and that attention does not scale past one
agent. Being on the loop means setting direction and still being able to say
what the agents did and what state the project is in, without replaying
anything. So progress is kept as data, and the views over it are generated from
that data. The app's left pane is one such view. It renders the roadmap from
JSON, and opening an item shows the spec that implemented it next to the report
that came back.

## What exists today

The runtime uses only the standard library, and 625 checks cover it without
making a single network call.

- **Providers**: Anthropic, OpenAI, Gemini, and OpenAI-compatible endpoints,
  with model discovery, streaming, typed retry that separates foreground from
  background work, and round-trip fidelity for whatever a vendor needs handed
  back.
- **Agent loop**: tools, permissions with typed decisions and named modes,
  parallel calls that treat mutations as barriers, and cancellation that reaches
  into HTTP reads and running shell commands.
- **Leader and subagents**: dispatch, a run graph with per-agent control, agents
  defined as files, failure breakers, and budgets for turns, wall time, tokens,
  and cost.
- **Context**: compaction, tool-result offload to an addressable store, and an
  instruction hierarchy that records where each instruction came from and what
  it costs.
- **Sessions**: an append-only JSONL record per run, resume, and fork from an
  earlier message.
- **Tools**: read, write, edit, multi-edit, list, glob, grep, shell, fetch, and
  search, each one gated by permission.
- **Extensibility**: scoped configuration with a capability ceiling, hooks that
  cover subagents too, skills, plugins, a trust list, and MCP servers merged into
  the tool registry for the life of a run. All of it resolves once at run start
  and is carried as a value.
- **Host process**: a separate package that owns a run and serves its typed
  events over SSE, taking prompts, approvals, and stop as POSTs. Nothing imports
  the runtime across that boundary.

## The app

A web UI over the host boundary, written in HTML, CSS, and JavaScript with no
dependencies, which leaves the choice of shell until last. This is the current
phase and it is unfinished. Working today: the roadmap pane and its
spec-and-report view, a three-state turn lifecycle that queues input instead of
rejecting it, and approvals answered in place.

The Textual UI in `symphonai_tui/` is frozen. It still works because it is the
only way to run a turn from a terminal, and it is not being extended.

## Running it

Python 3.11 or later and Node 18 or later. Node's built-in test runner checks
the JavaScript app, so there is no npm install. API keys are read from the
environment, never from a file in the repository.

```bash
python3 scripts/check.py                 # the check suite (--only selects a subset)
pip install -e ".[tui]"                  # optional Textual UI
python3 scripts/tui.py                   # run it
python3 -m symphonai_host --provider openai   # the host, for the browser app
```

The host prints a JSON handshake with `port` and `token` on its first line. Open
`http://127.0.0.1:<port>/app?token=<token>` with those values. The query token
works only for that first navigation. Later page requests use a session cookie
scoped to `/app/`, and data and control requests authenticate by header. The
host serves the app from the sibling `symphonai_app/` directory. Installed
without it, it returns `app is not installed` rather than files from your
project.

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

에이전트 네이티브 개발을 위한 코딩 에이전트 런타임입니다. 리더 모델이
서브에이전트에게 작업을 맡기고, 대시보드는 이들을 지휘하는 사람이 루프 안으로
끌려 들어가지 않고 루프 위에 남게 합니다.

## 왜 만들었나

에이전트 네이티브 개발은 코드를 에이전트가 쓰고 사람은 한 단계 위에서 의도와
판단을 맡는 방식입니다. 에이전트 하나짜리 채팅 도구에서는 부수적이던 세 가지가
여기서는 설계의 출발점이 됩니다.

**구독이 아니라 API로 여러 모델을.** 한 번의 실행을 여러 제공자로 조합합니다.
Anthropic, OpenAI, Gemini는 물론 OpenAI 호환 엔드포인트면 무엇이든 쓸 수 있고,
리더와 서브에이전트가 같은 모델이거나 같은 벤더일 필요도 없습니다. 도구 스키마
변환과 모델 탐색을 벤더 이름이 아니라 wire format으로 결정하기 때문에, 새 OpenAI
호환 벤더를 붙이는 일은 코드가 아니라 설정 문제입니다.

**스펙 주도 개발.** 리더는 서브에이전트에게 글로 쓴 스펙을 넘깁니다. 파일 경로,
심볼, 지켜야 할 계약, 수용 기준이 거기 담깁니다. 서브에이전트는 무엇을 고쳤고
무엇을 실행했으며 어떤 기준을 충족했는지 적은 리포트로 답합니다. 이 주고받음이
작업의 단위이자 사람이 시간을 쓰는 지점입니다. 스펙을 쓰는 일이 설계이고,
리포트를 스펙에 비추어 읽는 일이 리뷰입니다. 리포트는 스펙과 대조해 검증할 수
있지만 대화 기록은 읽어 보는 것 말고는 할 수 있는 일이 없습니다. 이 프로젝트도
모든 단계를 그렇게 만들었습니다.

**Humman in the loop가 아니라 Humman on the loop.** 루프 안에 있다는 것은 도구 호출마다 승인
버튼을 누르고 지나가는 턴을 눈으로 좇는 일입니다. 그런 주의력은 에이전트가 둘만
되어도 감당하기 어렵습니다. 루프 위에 있다는 것은 방향만 정해 두고도, 기록을
되짚지 않고 에이전트들이 무엇을 했고 프로젝트가 지금 어떤 상태인지 바로 답할 수
있다는 뜻입니다. 그래서 진행 상황을 데이터로 남기고, 화면은 손으로 쓰는 대신 그
데이터에서 만들어 냅니다. 앱 왼쪽 패널이 그렇게 만든 화면입니다. 로드맵을
JSON에서 읽어 그리고, 항목을 열면 그 일을 구현한 스펙과 돌아온 리포트를 나란히
보여 줍니다.

## 지금 있는 것

런타임은 표준 라이브러리만 쓰고, 네트워크를 전혀 타지 않는 검사 625개가 이를
검증합니다.

- **제공자**: Anthropic, OpenAI, Gemini, OpenAI 호환 엔드포인트. 모델 탐색,
  스트리밍, 전경 작업과 배경 작업을 구분하는 재시도, 벤더가 돌려받아야 하는 값을
  그대로 되돌려 주는 왕복 처리.
- **에이전트 루프**: 도구, 타입이 있는 결정과 이름 붙은 모드로 이뤄진 권한, 변경
  작업을 경계로 삼는 병렬 호출, HTTP 읽기와 실행 중인 셸 명령까지 끊어 내는 취소.
- **리더와 서브에이전트**: 작업 분배, 에이전트별로 제어하는 실행 그래프, 파일로
  정의하는 에이전트, 실패 차단기, 턴·시간·토큰·비용 예산.
- **컨텍스트**: 압축, 주소로 꺼내 쓰는 저장소에 도구 결과 내보내기, 출처와 사용량을
  함께 남기는 지시 계층.
- **세션**: 실행마다 덧붙이기만 하는 JSONL 기록, 재개, 이전 메시지에서 갈라내기.
- **도구**: 읽기, 쓰기, 편집, 다중 편집, 목록, glob, grep, 셸, fetch, 검색. 모두
  권한을 거칩니다.
- **확장성**: 능력 상한이 있는 범위별 설정, 훅(서브에이전트에도 걸립니다), 스킬,
  플러그인, 신뢰 목록, 실행이 끝날 때까지 도구 목록에 합쳐지는 MCP 서버. 이 모두를
  실행 시작 시 한 번에 정리해 값으로 넘깁니다.
- **호스트 프로세스**: 실행 하나를 맡아 이벤트를 SSE로 내보내는 별도 패키지.
  프롬프트와 승인, 중지는 POST로 받습니다. 이 경계 너머로 런타임을 import 하지
  않습니다.

## 앱

호스트 경계 위에 올린 웹 UI입니다. 의존성 없는 HTML, CSS, JavaScript로 되어 있어
어떤 껍데기에 담을지는 마지막에 정하면 됩니다. 지금 진행 중인 단계라 아직
완성되지 않았습니다. 현재 동작하는 부분은 로드맵 패널과 스펙·리포트 보기, 입력을
거부하지 않고 큐에 쌓아 두는 세 상태 턴 처리, 그 자리에서 답하는 승인입니다.

`symphonai_tui/`의 Textual UI는 동결했습니다. 아직은 터미널에서 턴을 돌릴 유일한
방법이라 동작만 유지하고 더 손대지 않습니다.

## 실행 방법

Python 3.11 이상과 Node 18 이상이 필요합니다. JavaScript 앱은 Node에 내장된 테스트
러너로 검사하므로 npm install은 하지 않아도 됩니다. API 키는 저장소 안의 파일이
아니라 환경 변수에서 읽습니다.

```bash
python3 scripts/check.py                 # 검사 모음 (--only 로 일부만 고름)
pip install -e ".[tui]"                  # 선택 사항인 Textual UI
python3 scripts/tui.py                   # 실행
python3 -m symphonai_host --provider openai   # 브라우저 앱을 띄울 호스트
```

호스트를 띄우면 첫 줄에 `port`와 `token`이 담긴 JSON 핸드셰이크가 나옵니다. 그
값을 넣어 `http://127.0.0.1:<port>/app?token=<token>` 을 여세요. 쿼리 토큰은 첫
화면을 여는 데만 쓰이고, 그다음부터 페이지 요청은 `/app/` 범위로 제한된 세션
쿠키를, 데이터와 제어 요청은 헤더를 씁니다. 호스트는 옆 디렉터리인
`symphonai_app/`에서 앱을 찾아 내보냅니다. 그 디렉터리 없이 설치했다면 사용자의
프로젝트 파일을 대신 내보내는 대신 `app is not installed` 를 돌려줍니다.

## 어떻게 만들었나

이 문서에 적은 방식 그대로 만들었습니다. Claude Code가 스펙을 쓰고 diff를
리뷰하며, Codex가 구현합니다. 히스토리에 남은 모든 단계가 그 과정을 거쳤습니다.
