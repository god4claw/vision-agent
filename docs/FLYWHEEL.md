# Data Flywheel (Stage 3)

The flywheel turns the agent's own uncertainty into better models over time:

```
run agent ──► low-confidence crops logged ──► human corrects labels
   ▲                                                   │
   │                                                   ▼
hot-reload rec.onnx ◄── export to ONNX ◄── offline fine-tune (PaddleOCR)
```

## 1. Collect hard samples (online, in the agent)

Run the native transport with collection enabled. Every recognition whose
confidence is below `-collect-thresh` is saved as a deskewed crop PNG plus a
line in `data/flywheel/samples.jsonl`.

```powershell
$env:CGO_ENABLED = 1
$env:Path = "C:\ProgramData\mingw64\mingw64\bin;" + $env:Path
go run ./cmd/agent -transport native -collect -collect-thresh 0.85 -iters 500
```

`samples.jsonl` records: `{time, file, text, conf, x, y, w, h}`.

## 2. Build a label draft

```powershell
python scripts/prepare_dataset.py
```

This writes `data/flywheel/label_draft.txt` (`crop_file<TAB>predicted_text`).
The predictions are the model's own guesses — **correct them by hand**. The
corrected file is your training label set.

## 3. Fine-tune offline (PaddleOCR)

On a machine with PaddlePaddle + PaddleOCR (GPU recommended), train a rec model
starting from the PP-OCRv6 weights, using the corrected labels:

```bash
# pseudo-workflow — see PaddleOCR docs for exact config
python tools/train.py -c configs/rec/PP-OCRv6/rec_ppocrv6.yml \
    -o Global.pretrained_model=pretrain/PP-OCRv6_rec \
       Train.dataset.label_file_list=[data/flywheel/label_draft.txt] \
       Train.dataset.data_dir=data/flywheel
```

Keep the character dictionary identical to `assets/ppocr_keys.txt` so the CTC
class indices still match the Go decoder.

## 4. Export to ONNX

```bash
paddle2onnx --model_dir inference/rec --save_file rec_finetuned.onnx \
    --opset_version 14
```

## 5. Hot-swap (online, no restart)

Run the agent with `-watch-models`. Drop the new model over
`assets/models/rec.onnx`; the engine detects the changed mtime and reloads the
session in place:

```powershell
go run ./cmd/agent -transport native -watch-models -diff 0 -iters 0
# in another shell, after retraining:
Copy-Item rec_finetuned.onnx assets\models\rec.onnx -Force
# agent logs: "rec model hot-reloaded"
```

Note: hot-reload is checked on each real perceive call. With the frame-diff
cache active (`-diff > 0`) on a fully static screen, perceives are largely
served from cache; use `-diff 0` (or a changing screen) to pick up a swap
promptly.

## Drift signal

A rising rate of low-confidence samples over time is a drift signal: the screen
content has moved away from what the current model handles well. Re-run the loop
when collection volume climbs.
