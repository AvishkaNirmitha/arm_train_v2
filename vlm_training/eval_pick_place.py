"""Evaluate a fine-tuned VLM on the pick-and-place dataset (default: val split).

Metrics: answer parse rate, joint error (deg) per joint and per phase, gripper accuracy,
and - if mujoco + ../expert.py are importable - gripper position error (mm) obtained by
running forward kinematics on the predicted joints.

From the notebook (model already in memory):
    from eval_pick_place import evaluate
    evaluate(model, tokenizer, DATASET_DIR, n=300, out_dir="eval_results")

From the shell (loads saved LoRA adapters):
    python eval_pick_place.py --lora ../checkpoints/<run>/lora \
        --dataset ../datasets/pick_place_v1_21-10-2026-11-03-15 --n 300
"""
import os
os.environ.setdefault("UNSLOTH_BLOCK_COMPILE", "0")  # see train_pick_place.py; must precede `import unsloth`
import re
import sys
import json
import time
import argparse

import numpy as np

from pick_place_dataset import PickPlaceConversations, PROMPT_STYLES, run_prompt_style

HERE = os.path.dirname(os.path.abspath(__file__))
ACTION_RE = re.compile(r"j:\s*((?:-?\d+(?:\.\d+)?\s+){6}-?\d+(?:\.\d+)?)\s*\|\s*gripper:\s*(open|closed)")


def parse_answer(text):
    """'j: a b c d e f g | gripper: open' -> (deg[7], 'open'), or None if malformed."""
    m = ACTION_RE.search(text)
    if m is None:
        return None
    return np.array([float(v) for v in m.group(1).split()]), m.group(2)


def load_fk_env():
    try:
        sys.path.insert(0, os.path.join(HERE, ".."))
        from expert import PickPlaceEnv
        return PickPlaceEnv()
    except Exception as e:  # mujoco missing (e.g. bare Colab) -> joint metrics only
        print(f"[eval] FK disabled ({type(e).__name__}: {e}); reporting joint errors only")
        return None


def predict(model, tokenizer, images, prompt, max_new_tokens=64):
    """Greedy decode - sampling would make the joint numbers random."""
    messages = [{"role": "user",
                 "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(images=images, text=input_text, add_special_tokens=False,
                       return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def summarize(results):
    ok = [r for r in results if "joint_err_deg" in r]
    s = {"n": len(results), "parse_rate": len(ok) / max(len(results), 1)}
    if not ok:
        return s
    err = np.abs(np.array([r["joint_err_deg"] for r in ok]))
    worst = err.max(1)
    s.update({
        "joint_mae_deg": round(float(err.mean()), 3),
        "joint_mae_per_joint_deg": err.mean(0).round(3).tolist(),
        "all_joints_within_0.5deg": round(float((worst <= 0.5).mean()), 3),
        "all_joints_within_1deg": round(float((worst <= 1.0).mean()), 3),
        "gripper_acc": round(float(np.mean([r["gripper_ok"] for r in ok])), 3),
    })
    site = np.array([r["site_err_mm"] for r in ok if "site_err_mm" in r])
    if len(site):
        s.update({
            "site_err_mm_mean": round(float(site.mean()), 2),
            "site_err_mm_median": round(float(np.median(site)), 2),
            "site_err_mm_p90": round(float(np.percentile(site, 90)), 2),
            "site_within_10mm": round(float((site <= 10).mean()), 3),
        })
    s["per_phase"] = {}
    for phase in dict.fromkeys(r["phase"] for r in ok):
        rs = [r for r in ok if r["phase"] == phase]
        p = {"n": len(rs), "joint_mae_deg": round(float(np.abs([r["joint_err_deg"] for r in rs]).mean()), 3)}
        if "site_err_mm" in rs[0]:
            p["site_err_mm_mean"] = round(float(np.mean([r["site_err_mm"] for r in rs])), 2)
        s["per_phase"][phase] = p
    return s


def print_summary(s):
    print(f"\n=== eval on {s['n']} samples ===")
    print(f"parse rate             {s['parse_rate']:.1%}")
    if "joint_mae_deg" not in s:
        return
    print(f"joint MAE              {s['joint_mae_deg']:.2f} deg   (target < 0.5)")
    print("  per joint            " + "  ".join(f"j{i + 1} {v:.2f}" for i, v in enumerate(s["joint_mae_per_joint_deg"])))
    print(f"all joints <= 0.5 deg  {s['all_joints_within_0.5deg']:.1%}")
    print(f"all joints <= 1.0 deg  {s['all_joints_within_1deg']:.1%}")
    print(f"gripper accuracy       {s['gripper_acc']:.1%}")
    if "site_err_mm_mean" in s:
        print(f"gripper pos error      mean {s['site_err_mm_mean']:.1f} mm  median {s['site_err_mm_median']:.1f}"
              f"  p90 {s['site_err_mm_p90']:.1f}  (<=10mm: {s['site_within_10mm']:.1%})")
    print("per phase:")
    for phase, p in s["per_phase"].items():
        extra = f"  pos {p['site_err_mm_mean']:.1f} mm" if "site_err_mm_mean" in p else ""
        print(f"  {phase:10s} n={p['n']:4d}  joint MAE {p['joint_mae_deg']:.2f} deg{extra}")


def evaluate(model, tokenizer, dataset_dir, split="val", n=300, seed=0, out_dir=None, cams=None,
             prompt_style="full"):
    """Run the model on n random samples (n=0 -> whole split). Returns the summary dict."""
    from unsloth import FastVisionModel
    FastVisionModel.for_inference(model)

    ds = PickPlaceConversations(dataset_dir, split, cams=cams, prompt_style=prompt_style)
    if n and n < len(ds):
        idx = np.sort(np.random.default_rng(seed).choice(len(ds), n, replace=False))
    else:
        idx = np.arange(len(ds))
    env = load_fk_env()

    results, t0 = [], time.time()
    for k, i in enumerate(idx):
        row = ds.rows[i]
        text = predict(model, tokenizer, ds.images(row), ds.prompt(row))
        rec = {"id": row["id"], "phase": row["phase"], "answer": row["answer"], "prediction": text}
        parsed = parse_answer(text)
        if parsed is not None:
            q_deg, grip = parsed
            rec["joint_err_deg"] = (q_deg - np.array(row["target_joints_deg"])).round(2).tolist()
            rec["gripper_ok"] = grip == row["gripper"]
            if env is not None:
                pos = env.forward_site(np.radians(q_deg))
                rec["site_err_mm"] = round(float(1000 * np.linalg.norm(pos - row["target_site_pos"])), 2)
        results.append(rec)
        if (k + 1) % 25 == 0 or k + 1 == len(idx):
            rate = (k + 1) / (time.time() - t0)
            print(f"[eval] {k + 1}/{len(idx)}  {rate:.2f} samples/s  last: {text!r}", flush=True)

    summary = summarize(results)
    print_summary(summary)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{split}_predictions.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in results)
        with open(os.path.join(out_dir, f"{split}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[eval] saved predictions + summary to {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True, help="saved LoRA adapter dir (model.save_pretrained output)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=300, help="samples to evaluate, 0 = whole split")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--out-dir", default=None, help="default: <lora>/eval")
    ap.add_argument("--prompt-style", choices=PROMPT_STYLES, default=None,
                    help="default: the style the run was trained with (from its config.json)")
    args = ap.parse_args()

    from unsloth import FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(args.lora, load_in_4bit=args.load_in_4bit)
    prompt_style = args.prompt_style or run_prompt_style(args.lora)
    print(f"[eval] prompt style: {prompt_style}")
    evaluate(model, tokenizer, args.dataset, args.split, args.n,
             out_dir=args.out_dir or os.path.join(args.lora, "eval"), prompt_style=prompt_style)


if __name__ == "__main__":
    main()
