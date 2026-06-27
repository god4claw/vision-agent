"""Turn collected flywheel samples into a PaddleOCR rec training label draft.

Reads data/flywheel/samples.jsonl (written by the agent's -collect mode) and
emits a label file where each line is:

    <image_path>\t<predicted_text>

The predicted text is the model's OWN (possibly wrong) output. A human must
review and correct the labels before training -- that correction is the whole
point of the flywheel. Run after collecting samples:

    python scripts/prepare_dataset.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "flywheel")
JSONL = os.path.join(DIR, "samples.jsonl")
OUT = os.path.join(DIR, "label_draft.txt")


def main() -> None:
    if not os.path.exists(JSONL):
        print(f"no samples found at {JSONL}; run the agent with -collect first")
        return
    n = 0
    with open(JSONL, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8", newline="\n") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.write(f"{rec['file']}\t{rec.get('text', '')}\n")
            n += 1
    print(f"wrote {OUT} with {n} entries")
    print("NEXT: correct the labels by hand, then fine-tune (see docs/FLYWHEEL.md)")


if __name__ == "__main__":
    main()
