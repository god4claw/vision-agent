# Python perception sidecar (RapidOCR / PP-OCR on ONNX Runtime).
#
# Build from the project root so the build context includes this directory:
#   docker build -f perception/Dockerfile -t vision-agent-perception perception
FROM python:3.12-slim

WORKDIR /app

# OpenCV (pulled in by rapidocr) needs a few shared libraries at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so the layer is cached across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY . .

# HTTP perception endpoint (server.py listens on 127.0.0.1:8089 by default;
# 0.0.0.0 binding for container networking is handled by the orchestrator).
EXPOSE 8089

CMD ["python", "server.py"]
