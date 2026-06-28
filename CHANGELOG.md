# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Real ONNX-OCR tests covering both perception paths. A Python test
  (`perception/test_ocr_real.py`) renders text in-memory with Pillow and runs
  it through the sidecar's real OCR path (`server._run_ocr`, RapidOCR / PP-OCR
  ONNX models), asserting the recognized text and boxes; it skips cleanly when
  RapidOCR / Pillow are unavailable. A skip-guarded native Go test
  (`internal/ocr/engine_real_test.go`) drives the in-process `Engine`
  (detection + recognition) on a committed sample image and asserts the known
  text is read; it skips when the gitignored `assets/` models + onnxruntime
  shared library are absent, so CI stays green. See `docs/ASSETS.md`.
- Real-Ollama integration test for the LLM action reasoner
  (`internal/reason/ollama_integration_test.go`): it drives the actual
  `OllamaReasoner` against a locally running Ollama with a small realistic
  perception result and asserts the returned `Decision` is structurally valid
  (parseable, action verb in the allowed set, chosen element index in range so
  the model never invents coordinates). The test probes the Ollama endpoint
  first and skips cleanly when the server is unreachable or the model is not
  pulled, so CI stays green without Ollama. The model is configurable via the
  `VA_OLLAMA_URL` and `VA_OLLAMA_MODEL` environment variables.

## [0.2.0] - 2026-06-28

### Added
- End-to-end integration test for the agent loop that wires real, non-network
  components together (synthetic capturer, an `httptest` server standing in for
  the perception `/ocr` endpoint driving the real `HTTPPerceiver` transport, the
  offline embedder, bounded episodic memory, the dry-run executor and the
  built-in heuristic reasoner) and asserts a clean, self-consistent run.
- Prometheus telemetry export: `Snapshot.Prometheus()` renders the telemetry
  snapshot in the Prometheus text exposition format (counters for
  scored/progress/regress/stall, gauges for the rates, efficiency and net), plus
  a `-metrics-addr` flag that serves it at `/metrics` over HTTP and is shut down
  on the graceful-shutdown path.

[0.2.0]: https://github.com/god4claw/vision-agent/releases/tag/v0.2.0

## [0.1.0] - 2026-06-28

### Added
- Continuous integration workflow (`ci.yml`) running gofmt, `go vet`, the test
  suite (CGO enabled) and `golangci-lint`.
- Test coverage for the memory subsystem (including persistent episodic
  memory), the embedding pipeline and OCR geometry, plus benchmarks for the
  pure-Go packages.
- Dockerfiles for the Go agent and the Python perception sidecar, with a
  `docker-compose.yml` wiring them together.
- Telemetry export in CSV and JSON formats via the `-telemetry-out` flag.
- Persistent episodic memory via the `-memory-dir` flag, plus a fix to the
  memory eviction logic.
- A `Makefile` with build, test, vet, fmt, lint, bench and docker targets.
- Graceful shutdown handling for the agent.
- A `-version` flag that prints the build version, stamped at link time via
  `-ldflags "-X main.version=..."`.
- Release workflow (`release.yml`) that builds cross-platform binaries
  (linux/amd64, windows/amd64, darwin/amd64, darwin/arm64) and publishes them
  to a GitHub Release on pushed `v*` tags.

[0.1.0]: https://github.com/god4claw/vision-agent/releases/tag/v0.1.0
