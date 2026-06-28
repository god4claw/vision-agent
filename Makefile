# Development tasks for vision-agent.
#
# Tests, vet and the native OCR transport require CGO (MinGW gcc on Windows,
# build-essential on Linux) + the onnxruntime shared library, so CGO is enabled
# here to mirror CI (.github/workflows/ci.yml).

export CGO_ENABLED := 1

GO ?= go
# Image tags for the container builds.
AGENT_IMAGE ?= vision-agent
PERCEPTION_IMAGE ?= vision-agent-perception

# Version string injected into the agent binary via -ldflags. Falls back to
# "dev" when git is unavailable or the tree is not a repository.
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)

.PHONY: build test vet fmt lint bench docker docker-agent docker-perception clean

# Build all binaries (stamps main.version into the agent).
build:
	$(GO) build -ldflags "-X main.version=$(VERSION)" ./...

# Run the full test suite (CGO enabled, like CI).
test:
	$(GO) test ./...

# Static analysis.
vet:
	$(GO) vet ./...

# Format the tree in place.
fmt:
	gofmt -w .

# Fail if anything is not gofmt-clean (mirrors the CI gofmt step).
lint:
	@unformatted="$$(gofmt -l .)"; \
	if [ -n "$$unformatted" ]; then \
		echo "gofmt needs to be run on:"; echo "$$unformatted"; exit 1; \
	fi

# Run benchmarks for the pure-Go packages (no model/DLL required).
bench:
	$(GO) test -bench=. -benchtime=10x -run '^$$' ./internal/ocr/ ./internal/embed/ ./internal/memory/

# Build both container images (Go agent + Python perception sidecar).
docker: docker-agent docker-perception

docker-agent:
	docker build -f Dockerfile -t $(AGENT_IMAGE) .

docker-perception:
	docker build -f perception/Dockerfile -t $(PERCEPTION_IMAGE) perception

# Remove build artifacts and test caches.
clean:
	$(GO) clean ./...
	$(GO) clean -testcache
