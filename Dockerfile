# Multi-stage build for the Go vision-agent binary (Linux container).
#
# This image builds the agent with CGO disabled, so it does NOT include the
# native ONNX OCR transport (`-transport native`), which requires CGO + the
# onnxruntime shared library and the gitignored assets/ (see docs/ASSETS.md).
# The containerized agent therefore uses the HTTP/gRPC perception sidecar
# (see perception/Dockerfile and docker-compose.yml). To build with the native
# transport you need a CGO toolchain, onnxruntime.so, and the vendored models.
FROM golang:1.25 AS build
WORKDIR /src

# Cache module downloads.
COPY go.mod go.sum ./
RUN go mod download

# Build the static agent binary (no CGO, no native OCR).
COPY . .
ENV CGO_ENABLED=0
RUN go build -trimpath -ldflags="-s -w" -o /out/agent ./cmd/agent

# Minimal runtime image.
FROM gcr.io/distroless/static-debian12
COPY --from=build /out/agent /agent
ENTRYPOINT ["/agent"]
# Default to the HTTP perception sidecar reachable on the compose network.
CMD ["-transport", "http", "-perception", "http://perception:8089"]
