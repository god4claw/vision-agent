package main

import (
	"context"
	"flag"
	"fmt"
	"image"
	"log/slog"
	"os"
	"os/signal"
	"sort"
	"strings"
	"time"

	"visionagent/internal/agent"
	"visionagent/internal/capture"
	"visionagent/internal/embed"
	"visionagent/internal/executor"
	"visionagent/internal/memory"
	"visionagent/internal/ocr"
	"visionagent/internal/perception"
	"visionagent/internal/reason"
)

func main() {
	var (
		live          = flag.Bool("live", false, "enable REAL actions (default: dry-run only)")
		iters         = flag.Int("iters", 0, "max iterations (0 = run forever)")
		transport     = flag.String("transport", "http", "perception transport: http | grpc | native")
		perceptionURL = flag.String("perception", "http://127.0.0.1:8089", "perception sidecar URL (http transport)")
		grpcAddr      = flag.String("grpc-addr", "127.0.0.1:8090", "perception gRPC address (grpc transport)")
		ortDLL        = flag.String("ort-dll", "assets/onnxruntime.dll", "onnxruntime shared library (native transport)")
		detModel      = flag.String("det-model", "assets/models/det.onnx", "detection ONNX model (native transport)")
		recModel      = flag.String("rec-model", "assets/models/rec.onnx", "recognition ONNX model (native transport)")
		recKeys       = flag.String("rec-keys", "assets/ppocr_keys.txt", "recognition char dictionary (native transport)")
		nativeRec     = flag.Bool("native-recognize", true, "run text recognition in native transport (false = boxes only)")
		ep            = flag.String("ep", "cpu", "execution provider: cpu | directml | cuda")
		epDevice      = flag.Int("ep-device", 0, "GPU device index for directml/cuda")
		profile       = flag.Bool("profile", false, "log per-stage perceive timings (decode/detect/recognize)")
		collect       = flag.Bool("collect", false, "flywheel: save low-confidence rec crops for retraining")
		collectThresh = flag.Float64("collect-thresh", 0.6, "flywheel: collect crops with conf below this")
		collectDir    = flag.String("collect-dir", "data/flywheel", "flywheel: output directory")
		watchModels   = flag.Bool("watch-models", false, "flywheel: hot-reload rec model when its file changes")
		ollamaURL     = flag.String("ollama", "http://127.0.0.1:11434", "ollama base URL")
		embModel      = flag.String("embed-model", "", "ollama embedding model; empty = local offline embedder")
		reasonerOn    = flag.Bool("reasoner", false, "use the Ollama LLM policy to decide actions (else built-in heuristic)")
		reasonModel   = flag.String("reason-model", "llama3.2", "ollama model for the action reasoner")
		goal          = flag.String("goal", "", "objective handed to the LLM reasoner")
		replayThresh  = flag.Float64("replay-threshold", 0.7, "min memory similarity (0..1) to replay a past successful action")
		goalReward    = flag.Bool("goal-reward", false, "score reward by goal progress via the Ollama judge (needs -goal; else screen-change)")
		planOn        = flag.Bool("plan", false, "decompose -goal into ordered sub-steps and pursue them one at a time")
		scoreCacheCap = flag.Int("score-cache", 4096, "max cached goal-reward verdicts (0 = unbounded)")
		safeMinMS     = flag.Int("safe-min-ms", 250, "live: minimum milliseconds between real actions")
		noGo          = flag.String("no-go", "", "live: forbidden click zones 'x,y,w,h;...'")
		allow         = flag.String("allow", "", "live: comma-list of allowed action types (e.g. 'move' or 'move,click'); empty = all")
		maxEpisodes   = flag.Int("max-episodes", 5000, "bounded episodic memory cap (0 = unbounded)")
		tickMS        = flag.Int("tick", 200, "delay between iterations in milliseconds")
		display       = flag.Int("display", 0, "display index to capture and act on (0 = primary); use a second/virtual monitor to isolate the agent")
		roi           = flag.String("roi", "", "region of interest 'x,y,w,h' in absolute screen coords (empty = full display)")
		diff          = flag.Float64("diff", 1.5, "frame-diff threshold; skip OCR when frame change <= this (0 = always OCR)")
		attnFlag      = flag.Bool("attn", false, "diff-ROI attention: OCR only changed screen regions (overrides -diff)")
		attnTile      = flag.Int("attn-tile", 32, "attention tile size in pixels")
		attnThresh    = flag.Float64("attn-thresh", 2.0, "attention per-tile change threshold (0..255)")
	)
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stdout, nil))

	var emb embed.Embedder = embed.LocalEmbedder{}
	if *embModel != "" {
		emb = embed.NewOllamaEmbedder(*ollamaURL, *embModel)
		log.Info("embedder: ollama", "model", *embModel)
	} else {
		log.Info("embedder: local (offline, no model required)")
	}

	store, err := memory.NewStore(emb, *maxEpisodes)
	if err != nil {
		log.Error("memory init failed", "err", err)
		os.Exit(1)
	}

	// Select perception transport.
	type healthChecker interface {
		Health(context.Context) error
	}
	var base perception.Perceiver
	switch *transport {
	case "grpc":
		gp, err := perception.NewGrpcPerceiver(*grpcAddr)
		if err != nil {
			log.Error("grpc perceiver init failed", "err", err)
			os.Exit(1)
		}
		base = gp
		log.Info("perception transport: grpc", "addr", *grpcAddr)
	case "http":
		base = perception.NewHTTPPerceiver(*perceptionURL)
		log.Info("perception transport: http", "url", *perceptionURL)
	case "native":
		eng, err := ocr.NewEngine(*detModel, *recModel, *recKeys, *ortDLL, *ep, *epDevice)
		if err != nil {
			log.Error("native ocr engine init failed", "err", err)
			os.Exit(1)
		}
		log.Info("native ocr engine ready", "ep", *ep, "device", *epDevice, "dll", *ortDLL)
		opts := ocr.NativeOptions{Recognize: *nativeRec, Watch: *watchModels, Profile: *profile, Log: log}
		if *collect {
			col, err := ocr.NewCollector(*collectDir, *collectThresh)
			if err != nil {
				log.Error("flywheel collector init failed", "err", err)
				os.Exit(1)
			}
			opts.Collect = col
			log.Info("flywheel collection enabled", "dir", *collectDir, "thresh", *collectThresh)
		}
		base = ocr.NewNativePerceiver(eng, opts)
		log.Info("perception transport: native (onnxruntime)", "recognize", *nativeRec, "watch", *watchModels)
	default:
		log.Error("unknown -transport (want http|grpc|native)", "value", *transport)
		os.Exit(1)
	}
	if hc, ok := base.(healthChecker); ok {
		if err := hc.Health(context.Background()); err != nil {
			log.Warn("perception health check failed (loop will retry)", "err", err)
		} else {
			log.Info("perception sidecar healthy")
		}
	}

	// Skip strategy: attention diff-ROI (fine-grained) or whole-frame cache.
	perceiver := base
	var cache *perception.CachingPerceiver
	var attn *perception.AttentionPerceiver
	switch {
	case *attnFlag:
		attn = perception.NewAttentionPerceiver(base, *attnTile, *attnThresh)
		perceiver = attn
		log.Info("attention diff-ROI enabled", "tile", *attnTile, "thresh", *attnThresh)
	case *diff > 0:
		cache = perception.NewCachingPerceiver(base, *diff)
		perceiver = cache
		log.Info("frame-diff cache enabled", "threshold", *diff)
	}

	capturer := capture.ScreenCapturer{Display: *display}
	if *roi != "" {
		var x, y, w, h int
		if _, err := fmt.Sscanf(*roi, "%d,%d,%d,%d", &x, &y, &w, &h); err != nil {
			log.Error("invalid -roi (want 'x,y,w,h')", "err", err)
			os.Exit(1)
		}
		capturer = capture.ScreenCapturer{Display: *display, X: x, Y: y, W: w, H: h}
		log.Info("ROI capture enabled", "x", x, "y", y, "w", w, "h", h)
	}
	origin := capturer.Origin()
	if *display != 0 || origin.X != 0 || origin.Y != 0 {
		log.Info("capture target", "display", *display, "origin_x", origin.X, "origin_y", origin.Y)
	}

	var exec executor.Executor = executor.DryRunExecutor{Log: log}
	var safeExec *executor.SafeExecutor
	if *live {
		zones, err := parseZones(*noGo)
		if err != nil {
			log.Error("invalid -no-go (want 'x,y,w,h;...')", "err", err)
			os.Exit(1)
		}
		allowed := parseAllow(*allow)
		safeExec = executor.NewSafeExecutor(executor.RealExecutor{}, executor.Guardrails{
			NoGo:        zones,
			Allow:       allowed,
			MinInterval: time.Duration(*safeMinMS) * time.Millisecond,
			Abort:       executor.AbortRequested,
			Log:         log,
		})
		exec = safeExec
		log.Warn("LIVE mode: REAL input ENABLED - hold ESC to suppress actions (kill-switch)",
			"min_ms", *safeMinMS, "no_go_zones", len(zones), "allow", allowKeys(allowed))
	} else {
		log.Info("DRY-RUN mode: no real input will be performed")
	}

	telemetry := &agent.Telemetry{}
	a := &agent.Agent{
		Capturer:        capturer,
		Perceiver:       perceiver,
		Embedder:        emb,
		Store:           store,
		Executor:        exec,
		Log:             log,
		Goal:            *goal,
		ReplayThreshold: *replayThresh,
		Telemetry:       telemetry,
		CaptureOrigin:   origin,
		TickDelay:       time.Duration(*tickMS) * time.Millisecond,
	}
	if *reasonerOn {
		a.Reasoner = reason.NewOllamaReasoner(*ollamaURL, *reasonModel)
		log.Info("action policy: ollama reasoner", "model", *reasonModel, "goal", *goal)
	} else {
		log.Info("action policy: built-in heuristic")
	}
	var scoreCache *agent.CachingScorer
	if *goalReward {
		if *goal == "" {
			log.Warn("-goal-reward set without -goal; reward falls back to screen-change baseline")
		} else {
			scoreCache = agent.NewCachingScorer(reason.NewOllamaScorer(*ollamaURL, *reasonModel), *scoreCacheCap)
			a.Scorer = scoreCache
			log.Info("reward signal: goal-aware (ollama judge)", "model", *reasonModel, "cache", *scoreCacheCap)
		}
	} else {
		log.Info("reward signal: screen-change baseline")
	}
	var planCache *agent.CachingPlanner
	if *planOn {
		if *goal == "" {
			log.Warn("-plan set without -goal; no plan will be built")
		} else {
			planCache = agent.NewCachingPlanner(reason.NewOllamaPlanner(*ollamaURL, *reasonModel), *scoreCacheCap)
			a.Planner = planCache
			log.Info("multi-step planning enabled", "model", *reasonModel, "cache", *scoreCacheCap)
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// Watchdog supervisor: keep the loop alive in non-terminating mode,
	// restarting a run segment if it returns unexpectedly.
	for {
		select {
		case <-ctx.Done():
			log.Info("shutdown requested")
			return
		default:
		}

		segment := *iters
		if segment <= 0 {
			segment = 1000 // segment size for infinite mode
		}

		st, err := a.Run(ctx, segment)
		attrs := []any{
			"iterations", st.Iterations,
			"errors", st.Errors,
			"actions", st.Actions,
			"memory", store.Len(),
			"avg_perceive", st.AvgPerceive().Round(time.Millisecond).String(),
		}
		if cache != nil {
			hits, misses := cache.Stats()
			attrs = append(attrs, "cache_hits", hits, "cache_misses", misses)
		}
		if attn != nil {
			full, partial, skipped := attn.Stats()
			attrs = append(attrs, "attn_full", full, "attn_partial", partial, "attn_skipped", skipped)
		}
		if scoreCache != nil {
			hits, misses := scoreCache.Stats()
			attrs = append(attrs, "score_hits", hits, "score_misses", misses)
		}
		if safeExec != nil {
			done, blocked := safeExec.Stats()
			attrs = append(attrs, "actions_done", done, "actions_blocked", blocked)
		}
		if planCache != nil {
			hits, misses := planCache.Stats()
			attrs = append(attrs, "step_hits", hits, "step_misses", misses)
		}
		if tm := telemetry.Snapshot(); tm.Scored > 0 {
			attrs = append(attrs,
				"scored", tm.Scored, "progress", tm.Progress, "regress", tm.Regress, "stall", tm.Stall,
				"progress_rate", fmt.Sprintf("%.2f", tm.ProgressRate),
				"regress_rate", fmt.Sprintf("%.2f", tm.RegressRate),
				"stall_rate", fmt.Sprintf("%.2f", tm.StallRate),
				"efficiency", fmt.Sprintf("%.2f", tm.Efficiency), "net", tm.Net)
		}
		log.Info("run segment done", attrs...)
		if err != nil {
			log.Warn("run returned error; watchdog restarting", "err", err)
			time.Sleep(500 * time.Millisecond)
			continue
		}
		if *iters > 0 {
			return // finite run finished
		}
	}
}

// parseAllow parses a comma-list of action types (e.g. "move,click") into an
// allowlist for the SafeExecutor. An empty string returns nil (all allowed).
func parseAllow(s string) map[executor.ActionType]bool {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	allowed := make(map[executor.ActionType]bool)
	for _, part := range strings.Split(s, ",") {
		if p := strings.ToLower(strings.TrimSpace(part)); p != "" {
			allowed[executor.ActionType(p)] = true
		}
	}
	return allowed
}

// allowKeys renders an allowlist as a sorted slice for logging.
func allowKeys(m map[executor.ActionType]bool) []string {
	if len(m) == 0 {
		return []string{"all"}
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, string(k))
	}
	sort.Strings(keys)
	return keys
}

// parseZones parses a 'x,y,w,h;x,y,w,h;...' string into forbidden click
// rectangles for the SafeExecutor guardrails.
func parseZones(s string) ([]image.Rectangle, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	var zones []image.Rectangle
	for _, part := range strings.Split(s, ";") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		var x, y, w, h int
		if _, err := fmt.Sscanf(part, "%d,%d,%d,%d", &x, &y, &w, &h); err != nil {
			return nil, fmt.Errorf("zone %q: %w", part, err)
		}
		zones = append(zones, image.Rect(x, y, x+w, y+h))
	}
	return zones, nil
}
