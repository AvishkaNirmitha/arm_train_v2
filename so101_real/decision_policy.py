"""Try the decision model (colab_train_decide.ipynb) on recorded decisions: for each one, what you did vs. what the
model says, and a web page with both points on the pictures (red = your click, blue = the model's).

    .venv/bin/python so101_real/decision_policy.py --lora so101_real/checkpoints/qwen35_2b_so101_decide_v1_colab/lora \
        --data so101_real/datasets/so101_decide_v1_11-11-2026-14-06-06
    ... --split train --limit 20        # rows it was trained on (shows how well it learned them, not a fair test)

Writes <lora>/../eval_<split>.html and eval_<split>.jsonl.
"""
import os
import sys
os.environ.setdefault("UNSLOTH_BLOCK_COMPILE", "0")        # must precede `import unsloth` (see real_policy.py)

import json
import math
import html
import time
import argparse

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decision_dataset import parse_answer, CAMS            # noqa: E402


class DecisionModel:
    def __init__(self, lora_dir, load_in_4bit=False):
        import unsloth  # noqa: F401  (before transformers)
        from unsloth import FastVisionModel
        print(f"loading {lora_dir} ...")
        self.model, self.tokenizer = FastVisionModel.from_pretrained(lora_dir, load_in_4bit=load_in_4bit)
        FastVisionModel.for_inference(self.model)

    def __call__(self, images, prompt, max_new_tokens=24):
        """images: [PIL top, wrist, base] (448x336); returns the answer text (greedy)."""
        tok = self.tokenizer
        messages = [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]}]
        text = tok.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tok(images=images, text=text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lora", required=True, help="the trained LoRA folder (…/lora)")
    ap.add_argument("--data", required=True, help="the dataset folder from decision_dataset.py")
    ap.add_argument("--split", default="val", choices=("val", "train"))
    ap.add_argument("--limit", type=int, default=None, help="only the first N decisions")
    ap.add_argument("--4bit", dest="four_bit", action="store_true", help="load in 4 bit (less GPU memory)")
    ap.add_argument("--out", help="output folder (default: next to the LoRA)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(os.path.join(args.data, f"{args.split}.jsonl")) if line.strip()]
    rows = [r for r in rows if not r.get("augmented")][:args.limit]
    model = DecisionModel(args.lora, args.four_bit)
    results = []
    for r in rows:
        imgs = [Image.open(os.path.join(args.data, p)).convert("RGB") for p in r["images"]]
        t0 = time.time()
        got = model(imgs, r["prompt"])
        dt = time.time() - t0
        want, pred = parse_answer(r["answer"]), parse_answer(got)
        same = pred is not None and want[0] == pred[0]
        dist = math.dist(want[1:], pred[1:]) * 1000 if same and want[1] is not None and pred[1] is not None else None
        results.append({"id": r["id"], "label": r["label"], "you": r["answer"], "model": got, "label_ok": same,
                        "dist": None if dist is None else round(dist, 1), "seconds": round(dt, 2),
                        "images": [os.path.abspath(os.path.join(args.data, p)) for p in r["images"]],
                        "want": want, "pred": pred, "prompt": r["prompt"]})
        print(f"{'OK' if same else 'XX'}  you: {r['answer']:<22} model: {got:<26}"
              + (f" dist {dist:4.0f}/1000" if dist is not None else "") + f"   ({dt:.1f} s)")

    n_ok = sum(x["label_ok"] for x in results)
    dists = [x["dist"] for x in results if x["dist"] is not None]
    summary = (f"{args.split}: label ok {n_ok}/{len(results)}"
               + (f", point distance mean {sum(dists) / len(dists):.0f} / 1000 (median "
                  f"{sorted(dists)[len(dists) // 2]:.0f})" if dists else ""))
    print("---", summary)
    out = args.out or os.path.dirname(os.path.abspath(args.lora.rstrip("/")))
    with open(os.path.join(out, f"eval_{args.split}.jsonl"), "w") as f:
        for x in results:
            f.write(json.dumps({k: v for k, v in x.items() if k not in ("images",)}) + "\n")
    page = os.path.join(out, f"eval_{args.split}.html")
    with open(page, "w") as f:
        f.write(page_html(results, summary, args))
    print("page:", page)


def page_html(results, summary, args):
    items = []
    for x in results:
        tiles = []
        for k, c in enumerate(CAMS):
            marks = ""
            for who, p in (("you", x["want"]), ("model", x["pred"])):
                if p and p[1] is not None and p[0] == f"click_{c}":
                    marks += f'<i class="pt {who}" style="left:{p[1] * 100:.1f}%;top:{p[2] * 100:.1f}%"></i>'
            tiles.append(f'<div class="pic"><img src="file://{html.escape(x["images"][k])}">{marks}<span>{c}</span></div>')
        d = f' · {x["dist"]:.0f}/1000 apart' if x["dist"] is not None else ""
        items.append(f'<section class="{"ok" if x["label_ok"] else "bad"}"><div class="hd">'
                     f'<b>{"OK" if x["label_ok"] else "WRONG"}</b> you: <code class="you">{html.escape(x["you"])}</code>'
                     f' model: <code class="model">{html.escape(x["model"])}</code>{d}'
                     f'<span class="m">{html.escape(x["id"])} · {x["seconds"]} s</span></div>'
                     f'<pre>{html.escape(x["prompt"])}</pre><div class="row">{"".join(tiles)}</div></section>')
    return (PAGE.replace("__SUM__", html.escape(summary)).replace("__LORA__", html.escape(args.lora))
            .replace("__BODY__", "".join(items)))


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Decision Model Test</title><style>
:root{--bg:#14161a;--panel:#1d2026;--text:#e6e8eb;--muted:#9aa1ab}
body{margin:0;padding:20px 16px;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}
main{max-width:1500px;margin:0 auto}h1{font-size:20px;margin:0 0 4px}p{color:var(--muted);margin:0 0 14px}
section{background:var(--panel);border-radius:8px;padding:10px;margin:0 0 12px;border-left:4px solid #5ad16a}
section.bad{border-left-color:#ff6b6b}.hd{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline;margin:0 0 6px}
.m{color:var(--muted);font-size:12px}code.you{color:#ff8a8a}code.model{color:#7fc4ff}
pre{color:var(--muted);font-size:11px;white-space:pre-wrap;margin:0 0 8px}
.row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.pic{position:relative}
.pic img{width:100%;display:block;border-radius:4px}
.pic span{position:absolute;left:6px;top:4px;background:#000a;padding:0 6px;border-radius:3px;font-size:12px}
.pt{position:absolute;width:20px;height:20px;margin:-10px 0 0 -10px;border:3px solid;border-radius:50%;box-shadow:0 0 0 2px #000}
.pt.you{border-color:#ff2d2d}.pt.model{border-color:#2d9bff;width:12px;height:12px;margin:-6px 0 0 -6px}
@media(max-width:700px){.row{grid-template-columns:1fr}}
</style></head><body><main><h1>Decision model test</h1>
<p>__SUM__ · <span style="color:#ff8a8a">red ring = your click</span> · <span style="color:#7fc4ff">blue dot = the model</span> · __LORA__</p>
__BODY__</main></body></html>"""


if __name__ == "__main__":
    sys.exit(main())
