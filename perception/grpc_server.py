#!/usr/bin/env python3
"""Stage 2 perception sidecar over gRPC.

Reuses the OCR logic from server.py (`_run_ocr`, OCR_BACKEND) and exposes it
through the Perception gRPC service defined in proto/perception.proto.

Run from the project root:
    python perception/grpc_server.py
"""
from concurrent import futures

import grpc

import perception_pb2
import perception_pb2_grpc
from server import OCR_BACKEND, _OCR_ERR, _run_ocr

ADDR = "127.0.0.1:8090"


class PerceptionServicer(perception_pb2_grpc.PerceptionServicer):
    def Health(self, request, context):  # noqa: N802
        return perception_pb2.HealthReply(status="ok", ocr=OCR_BACKEND, err=_OCR_ERR)

    def Ocr(self, request, context):  # noqa: N802
        out = _run_ocr(request.image)
        boxes = [
            perception_pb2.Box(
                x=b["x"], y=b["y"], w=b["w"], h=b["h"],
                text=b["text"], conf=b["conf"],
            )
            for b in out["boxes"]
        ]
        return perception_pb2.OcrReply(text=out["text"], boxes=boxes)


def serve(addr: str = ADDR) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    perception_pb2_grpc.add_PerceptionServicer_to_server(PerceptionServicer(), server)
    server.add_insecure_port(addr)
    server.start()
    print(f"perception gRPC server listening on {addr} (ocr={OCR_BACKEND})")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nperception gRPC server stopped")


if __name__ == "__main__":
    serve()
