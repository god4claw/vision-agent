# vision-agent (MVP / early beta)

Autonomous vision agent — MVP / early beta. The Stage 2–4 features (gRPC
transport, native ONNX OCR, the data flywheel, and LLM reasoning/planning) are
already implemented; see the sections below. Closes the loop:

```
screen -> perception -> episodic memory -> decision -> action (dry-run) -> screen
```

Built per the approved DEPLOY MANIFEST. **DRY-RUN by default**: no real mouse/keyboard
input is performed until you explicitly enable live mode.

## Project status & how to resume

Current starting point for the next session:

- **Action set** — `move`, `click`, `drag`, `key`, `none`. The model never invents
  coordinates; it references a detected element by index and the agent resolves the
  center (see `internal/reason/reason.go` `ToAction`).
- **Guardrails** (`internal/executor`) — allowlist (`-allow`), no-go zones
  (`-no-go`), ESC kill-switch, and a min-interval rate limit; drag checks both
  endpoints.
- **Telemetry** (`internal/agent/telemetry.go`) — every scored action is bucketed
  into progress / regress / stall, with derived `progress_rate`, `regress_rate`,
  `stall_rate`, `efficiency`, and `net` logged in `run segment done`.
- **Screen targeting** — `-display N` and `-roi x,y,w,h` select what to capture;
  `CaptureOrigin` offsets action coordinates back to absolute screen space so a
  click lands correctly on a region / second / virtual monitor.

**Known risk (learned the hard way):** in `-live` mode on the **primary** display
the agent sees and can click its **own windows** (terminal/IDE), which can kill the
process and leave a zombie holding `bin/agent.exe`. Always isolate live runs with
`-display`/`-roi` onto a separate (or virtual) screen, keep ESC ready, and if a run
ends abnormally, `taskkill /IM agent.exe /F` before rebuilding.

**Next steps:** isolate live runs on a virtual/second display; export telemetry to
CSV/JSON for time-series graphs; try a larger reasoner model to improve targeting
(reduce `stall_rate`).

## Stack

| Layer | Implementation |
|-------|----------------|
| Orchestration | Go (single binary) |
| Screen capture | Go, `kbinani/screenshot` (Windows win32, no CGo) |
| Perception (OCR) | Python sidecar (RapidOCR) **or** native ONNX in Go (`internal/ocr`, PP-OCRv6 via `onnxruntime_go`) |
| Memory | `chromem-go` (embedded vector store) |
| Embeddings | Local offline embedder by default; Ollama optional |
| Actions | Dry-run executor by default; real input + guardrails via `-live` |
| IPC | HTTP/JSON, gRPC, or in-process native (`-transport`) |

## Prerequisites

- Go 1.25+
- Python 3.10+ (only for the HTTP/gRPC perception sidecar)
- A C compiler (Mingw-w64 gcc) + `CGO_ENABLED=1` — only for `-transport native`
- Ollama (optional, only if you use `-embed-model`)

## First build

```powershell
go mod tidy
go build ./...
go test ./...
```

## Run

1. Start the perception sidecar (separate terminal):

```powershell
python perception/server.py
```

2. Run the agent (dry-run, 200 iterations):

```powershell
go run ./cmd/agent -iters 200
```

Infinite loop (Ctrl+C to stop):

```powershell
go run ./cmd/agent
```

### Useful flags

```
-iters N           max iterations (0 = forever)
-tick MS           delay between iterations (default 200ms)
-transport T       perception transport: http | grpc (default http)
-perception URL    sidecar URL (http transport, default http://127.0.0.1:8089)
-grpc-addr ADDR    perception gRPC address (grpc transport, default 127.0.0.1:8090)
-embed-model NAME  use Ollama embeddings (e.g. nomic-embed-text); empty = local
-max-episodes N    bounded memory cap (default 5000)
-roi "x,y,w,h"     OCR only this screen region (default: full display)
-diff F            frame-diff threshold; skip OCR when frame change <= F (0 = always OCR; default 1.5)
-live              ENABLE REAL ACTIONS (off by default)
-memory-dir DIR    persist episodic memory across runs (empty = in-memory, cleared on exit)
-telemetry-out F   write the final telemetry snapshot to F on exit (.csv or .json)
-metrics-addr ADDR serve Prometheus /metrics at ADDR (e.g. 127.0.0.1:9090; empty = off)
-version           print version and exit
```

## Run over gRPC

1. Start the gRPC sidecar (reuses the same OCR engine):

```powershell
python perception/grpc_server.py
```

2. Point the agent at it:

```powershell
go run ./cmd/agent -transport grpc -iters 200
```

Protocol is defined in `proto/perception.proto`. To regenerate stubs after
editing the proto (Go plugins must be on PATH via `go install`):

```powershell
$env:Path = "$(go env GOPATH)\bin;" + $env:Path
python -m grpc_tools.protoc -I proto --go_out=. --go_opt=module=visionagent --go-grpc_out=. --go-grpc_opt=module=visionagent --python_out=perception --grpc_python_out=perception proto/perception.proto
```

## Run fully native (Stage 4, no Python)

The `native` transport runs the PP-OCRv6 detection + recognition ONNX models
in-process via `onnxruntime_go` (CGo). No Python sidecar is needed.

Vendored assets live in `assets/`:

```
assets/onnxruntime.dll              onnxruntime shared library
assets/onnxruntime_providers_shared.dll
assets/models/{det,rec,cls}.onnx    PP-OCRv6 detection / recognition / angle
assets/ppocr_keys.txt               recognition char dictionary (CTC classes)
```

Build + run (CGo requires gcc on PATH):

```powershell
$env:CGO_ENABLED = 1
$env:Path = "C:\ProgramData\mingw64\mingw64\bin;" + $env:Path
go run ./cmd/agent -transport native -iters 200
```

Native flags: `-native-recognize=false` (detection boxes only), `-det-model`,
`-rec-model`, `-rec-keys`, `-ort-dll`, and the flywheel flags `-collect`,
`-collect-thresh`, `-collect-dir`, `-watch-models`.

Detection produces rotated min-area boxes (convex hull + rotating calipers) and
recognition runs in padded batches for throughput.

Inspect what the engine sees on your screen:

```powershell
go run ./cmd/ocrprobe -top 25
```

Regenerate the char dictionary from the rec model metadata:

```powershell
python scripts/extract_dict.py
```

## Data flywheel (Stage 3)

The native engine can log its own low-confidence recognitions for later
retraining, and hot-swap a retrained model with no restart:

```powershell
# collect hard samples while running
go run ./cmd/agent -transport native -collect -collect-thresh 0.85 -iters 500

# pick up a retrained model live
go run ./cmd/agent -transport native -watch-models -diff 0
```

Full loop (collect -> correct -> fine-tune -> export -> hot-swap) is documented
in `docs/FLYWHEEL.md`.

## Performance: frame-diff cache

The agent skips the expensive OCR call when the frame is essentially unchanged
(a 16x16 grayscale signature compared against `-diff`). On a static screen this
cuts average perceive latency dramatically (measured ~3200ms -> ~370ms over 10
iterations: 1 real OCR + 9 cache hits). Combine with `-roi` to shrink the OCR
area further. The run log reports `avg_perceive`, `cache_hits`, `cache_misses`.

## Performance: attention (diff-ROI)

`-attn` is a finer-grained, transport-agnostic skip strategy that overrides the
whole-frame cache. It tiles the frame (`-attn-tile`, default 32px), detects
which tiles changed since the last frame (`-attn-thresh`), and runs OCR only on
the bounding region of the changed tiles, merging fresh boxes with the cached
boxes from the unchanged screen area:

```powershell
go run ./cmd/agent -transport native -attn -iters 200
```

The run log reports `attn_full` / `attn_partial` / `attn_skipped`. Note: the
dirty region is a single bounding box of all changed tiles, so scattered
changes (e.g. a blinking cursor plus a clock in opposite corners) can widen it;
clustered changes get the biggest win.

## Decision policy: LLM reasoner

By default the agent uses a tiny built-in heuristic (replay nearest memory,
else click the first element). With `-reasoner` it instead asks a local Ollama
model to choose each action toward a `-goal`:

```powershell
go run ./cmd/agent -transport native -ep directml -ep-device 1 \
    -ort-dll assets/dml/onnxruntime.dll -attn \
    -reasoner -reason-model llama3.2:1b -goal "open the File menu"
```

The model never invents pixel coordinates: the perceiver's detected text boxes
are presented as a numbered list, and the model replies with strict JSON
(`format:"json"`) naming an action and an element index. Supported actions are
`click`, `move` (position the cursor over an element without clicking; `hover`
is an alias), `drag` (press on the `target` element, drag to the `to` element,
release), `key`, and `none`:

```json
{"action":"click","target":0,"key":"","reason":"open the file menu"}
{"action":"move","target":2,"key":"","reason":"hover over the File menu"}
{"action":"drag","target":3,"to":7,"key":"","reason":"drag file into folder"}
```

The agent resolves the index back to the element's centre (`reason.ToAction`).
Key actions are validated against a whitelist (`executor.ValidKey`): only real
pressable keys/combos (`a`, `Enter`, `Ctrl+Shift+P`, `F5`, ...) are accepted, so
an LLM that "presses" a sentence like `"Editor"` is rejected before any input.
Any reasoner failure (unreachable model, malformed JSON, out-of-range index,
invalid key) **degrades gracefully** to the heuristic and is not counted as a
loop error, so a flaky model never stalls the loop. Decision quality scales with the model:
tiny models (e.g. `llama3.2:1b`) often emit valid JSON but weak choices; pull a
larger model (`ollama pull llama3.2:3b` / `qwen2.5:7b`) for grounded actions.

## Multi-step planning

`-plan` (with a `-goal`) decomposes a high-level goal into an ordered list of
concrete sub-steps via the Ollama planner (`reason.OllamaPlanner.Plan`), then
pursues them one at a time:

```powershell
go run ./cmd/agent -transport native -ep directml -ep-device 1 \
    -ort-dll assets/dml/onnxruntime.dll -attn \
    -reasoner -reason-model llama3.2:3b -plan \
    -goal "open the File menu and click New File"
```

Each tick the agent hands the **current sub-step** (not the whole goal) to the
reasoner. Before choosing, it asks the planner `StepComplete(step, screen)` and
advances the cursor past every leading sub-step the screen already satisfies
(so already-done prefixes are skipped). Reward is scored against the active
sub-step too. Planning failure leaves the plan empty and the agent pursues the
goal directly. The plan is held in `agent.Plan` (steps + cursor); once
exhausted, `Current()` falls back to the overall goal.

Cost note: planning adds one upfront call, and a `StepComplete` call per tick on
top of the decision call — heaviest of the LLM modes, so it is opt-in.

## Learning: reward & episodic memory

The agent learns without any model training. Each action is scored by a
**grounded reward signal**: an action "succeeded" if the screen changed after
it (it did something); a no-op never counts (`scoreSuccess`). Because an
action's effect is only visible on the *next* frame, scoring is **deferred one
tick** — the agent holds the pending action and resolves its episode against the
following observation, then persists it with the real success flag (no more
hard-coded `Success: true`).

Replay is gated by similarity: the heuristic only repeats a remembered action
when the past episode succeeded **and** the current state's cosine similarity to
it is `>= -replay-threshold` (default `0.7`), so an unrelated screen never
triggers a stale replay. The LLM reasoner is also fed only *successful* nearby
episodes as reference. Memory stays bounded by `-max-episodes` (oldest evicted).

### Goal-aware reward

The screen-change baseline only knows an action *did something*, not whether it
*helped*. With `-goal-reward` (and a `-goal`), reward instead comes from an
Ollama **judge** (`reason.OllamaScorer`): it is shown the goal plus the screen
text before and after the action and returns `{"progress": true|false}`. So an
action is credited only when it moved the screen *closer to the goal*:

```powershell
go run ./cmd/agent -transport native -ep directml -ep-device 1 \
    -ort-dll assets/dml/onnxruntime.dll -attn \
    -reasoner -reason-model llama3.2:3b \
    -goal "open the File menu" -goal-reward
```

This adds a second LLM call per scored tick. A judge failure degrades gracefully
to the screen-change baseline and is never counted as a loop error. No-op
actions are never scored.

Verdicts are memoized by a bounded `CachingScorer` (`-score-cache`, default
4096): an identical `(goal, before, after)` transition is judged only once, so
static or repeating screens cost no further LLM calls. The run log reports
`score_hits` / `score_misses` (e.g. a static screen yields ~80% hits). Errors
are never cached.

## GPU acceleration (DirectML)

`-ep directml` runs inference on any Windows GPU via the DirectML execution
provider; `-ep cuda` targets NVIDIA + CUDA/cuDNN. This needs a DirectML-enabled
`onnxruntime.dll` (the CPU build does not support it) whose ORT C-API version
matches the `onnxruntime_go` binding (currently API 26 -> ORT 1.27). The
DirectML `onnxruntime.dll` (1.27, ORT-Nightly feed) plus `DirectML.dll` (1.15.4)
live in `assets/dml/`:

```powershell
go run ./cmd/agent -transport native -ep directml -ep-device 1 \
    -ort-dll assets/dml/onnxruntime.dll
```

**Pick the right adapter.** DirectML device 0 is often the weak integrated GPU.
On this laptop, `-ep-device 0` selected the AMD iGPU (no speedup) while
`-ep-device 1` selected the NVIDIA RTX 3050 Ti (confirmed via `nvidia-smi`:
~40% util, ~840 MB). Sweep the index for your machine.

**Profiling findings** (`-profile` logs per-stage timings). Controlled A/B on
the same full 1080p screen (~205 text boxes):

| Stage | CPU | DirectML (RTX) |
|-------|-----|----------------|
| detect | ~100 ms | ~30 ms (3x) |
| recognize | ~3900 ms (~18.5 ms/box) | ~2480 ms (~12 ms/box, 1.5x) |
| frame | ~4000 ms (~0.24 fps) | ~2480 ms (~0.40 fps) |

Detection (constant input shape) accelerates well on GPU. Recognition was
originally *no faster* on GPU because it is batched with per-batch dynamic
widths, so DirectML re-optimized the graph on every call. The fix
(`engine_rec.go`): on GPU providers the rec input is snapped to fixed **width
buckets** (`recBuckets`) and a fixed batch size, so the shape only ever takes a
handful of values and the graph is compiled once per shape. That unlocks ~1.5x
on recognition (CPU keeps tight per-batch widths to avoid over-padding waste).

**The biggest speed lever is still cutting how many boxes are recognized per
frame:** combining `-ep directml -ep-device 1 -attn` drops a steady-screen frame
to ~150 ms (boxes=2-3) vs ~2700 ms full-CPU — about 17x. A fully dynamic
text-heavy screen is bounded by recognition at ~0.4 fps (~24 frames/min).

## Telemetry: progress / regress / stall

Every scored (non-no-op) action is bucketed by `agent.Telemetry` so you can
track how effectively the agent is advancing toward its goal:

- **progress** — the reward judged the action moved closer to the goal.
- **regress** — no progress, but the screen changed (it did something that did
  not help, i.e. moved the wrong way).
- **stall** — no progress and the screen did not change (no visible effect).

The run-segment log reports counts plus derived rates, `efficiency`, and `net`
(= progress - regress), e.g. `scored=5 progress=2 regress=1 stall=2
progress_rate=0.40 regress_rate=0.20 efficiency=0.67 net=1`. During long runs a
compact `telemetry` line is also logged every 50 iterations. On a static screen
in dry-run every action is a `stall` (no real input changes nothing), which is
the expected baseline.

Reading the metrics (which direction is healthy):

- **`efficiency` -> high is good** (the headline quality metric): of the actions
  that actually changed the screen, the share that helped. `efficiency=1.0` means
  every effective action moved toward the goal. It ignores stalls, so it is not
  diluted by harmless no-effect ticks.
- **`progress_rate` -> high is good**, **`net` -> positive & rising is good**.
- **`regress_rate` -> low is good** (few wrong-way actions).
- **`stall_rate` -> low is good** (few no-effect actions: better targeting /
  more interactive elements hit).

So a healthy agent looks like *high `efficiency` + positive `net` + low
`regress_rate` + low `stall_rate`*.

## Live mode & guardrails

By default the agent is **dry-run**: it logs intended actions but performs no
real input. `-live` enables real mouse/keyboard input (Windows `user32.dll`:
`SetCursorPos` + `mouse_event` + `keybd_event`). Real input is always wrapped in
a `SafeExecutor` that enforces guardrails before anything reaches the OS:

```powershell
go run ./cmd/agent -transport native -ep directml -ep-device 1 \
    -ort-dll assets/dml/onnxruntime.dll -attn \
    -reasoner -reason-model llama3.2:3b -goal "open the File menu" \
    -live -safe-min-ms 300 -no-go "0,0,1920,40;0,1040,1920,40"
```

Guardrails:

- **Kill-switch** — hold **ESC** and every action is suppressed (`AbortRequested`
  polls `GetAsyncKeyState`). Release to resume.
- **No-go zones** (`-no-go "x,y,w,h;..."`) — clicks/moves inside any zone are
  blocked (e.g. fence off the taskbar and window title bar).
- **Action allowlist** (`-allow "move"` or `-allow "move,click"`) — when set,
  only those action types are forwarded; everything else is suppressed. Use
  `-allow move` for a cursor-only run with no clicks or keystrokes.
- **Rate limit** (`-safe-min-ms`, default 250) — minimum delay between real
  actions.
- **Key whitelist** — only `executor.ValidKey` combinations are pressed.

Suppressed actions are counted (`actions_done` / `actions_blocked` in the run
log), never forwarded, and never treated as loop errors. Off Windows, real
input returns an error (caught by the watchdog).

## Endurance

Validated over 1500 iterations (`errors=0`) and a sustained real-OCR stress run
(`-diff 0`) with stable memory (GC sawtooth around a flat baseline, no leak) —
ORT input/output tensors are destroyed every inference.

## Optional: real embeddings via Ollama

```powershell
ollama pull nomic-embed-text
go run ./cmd/agent -iters 200 -embed-model nomic-embed-text
```

## Definition of Done (Stage 1 — met)

- [x] `go build ./...` clean
- [x] sidecar `/health` responds
- [x] agent captures screen and calls perception
- [x] perception returns text + boxes
- [x] episode written to memory; recall@1 works
- [x] >=100 iterations, no crash, bounded memory (`go test`)
- [x] actions dry-run only

Stage 1 is complete; Stage 2 (gRPC), Stage 3 (flywheel) and Stage 4 (native ONNX)
are also implemented. See `CHANGELOG.md` for the released increments.
- [ ] watchdog continues on component failure
- [ ] this README

## Roadmap

- **Stage 2**: PaddleOCR -> ONNX, perception native in Go (`onnxruntime_go`); gRPC IPC.
- **Stage 3**: data flywheel (log low-confidence -> offline LoRA -> hot-swap), drift detection.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
