package perception

import (
	"context"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "visionagent/internal/perceptionpb"
)

// GrpcPerceiver talks to the Python perception sidecar over gRPC.
type GrpcPerceiver struct {
	conn   *grpc.ClientConn
	client pb.PerceptionClient
}

func NewGrpcPerceiver(addr string) (*GrpcPerceiver, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	return &GrpcPerceiver{conn: conn, client: pb.NewPerceptionClient(conn)}, nil
}

func (g *GrpcPerceiver) Health(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	_, err := g.client.Health(ctx, &pb.HealthRequest{})
	return err
}

func (g *GrpcPerceiver) Perceive(ctx context.Context, framePNG []byte) (Result, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	reply, err := g.client.Ocr(ctx, &pb.OcrRequest{Image: framePNG})
	if err != nil {
		return Result{}, err
	}
	boxes := make([]Box, 0, len(reply.GetBoxes()))
	for _, b := range reply.GetBoxes() {
		boxes = append(boxes, Box{
			X:    int(b.GetX()),
			Y:    int(b.GetY()),
			W:    int(b.GetW()),
			H:    int(b.GetH()),
			Text: b.GetText(),
			Conf: b.GetConf(),
		})
	}
	return Result{Text: reply.GetText(), Boxes: boxes}, nil
}

func (g *GrpcPerceiver) Close() error {
	return g.conn.Close()
}
