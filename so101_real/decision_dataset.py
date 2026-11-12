"""Training set for the decision model (Colab: colab_train_decide.ipynb) from build_decisions.py output.

One sample = the 3 clean camera pictures of a decision + a short text (gripper reading, the last decisions with their
points) -> the answer, one line:

    click_top 729 633      click_wrist 480 346      click_base 693 944      (x y = 0..1000 of that picture)
    gripper_close          gripper_open             done

    python so101_real/decision_dataset.py recordings/11_11_2026_pick_white_cube_3/decisions
    python so101_real/decision_dataset.py recordings/A/decisions recordings/B/decisions --val 3 --aug 8

Validation = whole episodes held out (--val N of them, picked with --seed), never augmented: the honest test.
Training rows are augmented --aug times (copy 0 = the original): small colour changes and a small random crop of
each picture (the clicked point moves with its picture). The history points stay as they were (they are positions
in earlier pictures).

Writes so101_real/datasets/so101_decide_v1_<time>/ (train.jsonl, val.jsonl, images/, meta.json) + the same as .zip
for Google Drive.
"""
import os
import re
import sys
import json
import random
import shutil
import argparse
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMS = ("top", "wrist", "base")
IMG_W, IMG_H = 448, 336                    # every picture, 4:3 (same size as the earlier pick / reach datasets)
INSTRUCTION = "Pick up the white cube and put it on the green pad."
LABELS = ("click_top", "click_wrist", "click_base", "gripper_close", "gripper_open", "done")


def _k(v):
    """0..1 -> 0..1000 integer (the coordinate format of the answers and the history)."""
    return int(round(min(max(float(v), 0.0), 1.0) * 1000))


def step_text(s):
    if "x" in s:
        return f"{s['label']} {_k(s['x'])} {_k(s['y'])}"
    if s["label"] == "gripper_close" and s.get("stopped_at") is not None:
        return f"gripper_close (stopped at {s['stopped_at']:.0f})"
    return s["label"]


def build_prompt(gripper_now, recent, cams=CAMS):
    """The text the model gets with the pictures (the same at training and on the robot)."""
    hist = "; ".join(step_text(s) for s in recent) or "none"
    return (f"Task: {INSTRUCTION}\n"
            f"Images: {', '.join(f'{i + 1} = {c} camera' for i, c in enumerate(cams))}.\n"
            f"Gripper now: {gripper_now:.0f} (88 = open, 40 = closed; stopped above 40 = holding something)\n"
            f"Last decisions: {hist}\n"
            "What next? Answer one of: click_top x y | click_wrist x y | click_base x y | gripper_close | "
            "gripper_open | done   (x y = 0-1000 in that camera's picture)")


def answer_text(row):
    p = row.get("point")
    if p:
        return f"{row['label']} {_k(p['x'])} {_k(p['y'])}"
    return row["label"]


def parse_answer(text):
    """'click_wrist 480 346' -> ('click_wrist', 0.480, 0.346); 'gripper_close' -> ('gripper_close', None, None);
    None if it is not an answer."""
    m = re.search(r"(click_top|click_wrist|click_base)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1), min(float(m.group(2)), 1000) / 1000, min(float(m.group(3)), 1000) / 1000
    m = re.search(r"\b(gripper_close|gripper_open|done)\b", text)
    return (m.group(1), None, None) if m else None


def _augment(img, rng, point=None):
    """Colour jitter + random crop (keeps >= 92 % of each side) resized back. Returns (img, point or None)."""
    h, w = img.shape[:2]
    for _ in range(20):
        s = rng.uniform(0.92, 1.0)
        cw, ch = int(w * s), int(h * s)
        x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        if point is None:
            break
        px, py = (point[0] * w - x0) / cw, (point[1] * h - y0) / ch
        if 0.02 <= px <= 0.98 and 0.02 <= py <= 0.98:
            break
    else:
        x0, y0, cw, ch = 0, 0, w, h
    out = img[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    out = out * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)                      # contrast, brightness
    grey = out.mean(axis=2, keepdims=True)
    out = grey + (out - grey) * rng.uniform(0.8, 1.2)                             # saturation
    out = np.clip(out, 0, 255).astype(np.uint8)
    new_point = None if point is None else ((point[0] * w - x0) / cw, (point[1] * h - y0) / ch)
    return out, new_point


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("decisions", nargs="+", help="decisions folders from build_decisions.py")
    ap.add_argument("--val", type=int, default=2, help="whole episodes held out for validation")
    ap.add_argument("--aug", type=int, default=8, help="copies of each training row (1 = no augmentation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="so101_decide_v1")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    rows = []
    for d in args.decisions:
        path = os.path.join(d, "decisions.jsonl")
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found - run build_decisions.py first")
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                if not r.get("clean"):
                    raise SystemExit(f"{r['id']}: pictures with overlays (built with --include-overlay) - not for training")
                r["_dir"] = d
                rows.append(r)
    episodes = sorted({(r["dataset"], r["episode"]) for r in rows})
    rng = random.Random(args.seed)
    val_eps = set(rng.sample(episodes, min(args.val, max(len(episodes) - 1, 0))))

    out = os.path.join(HERE, "datasets", f"{args.name}_{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}")
    os.makedirs(os.path.join(out, "images"))
    counts = {"train": 0, "val": 0}
    with open(os.path.join(out, "train.jsonl"), "w") as ftr, open(os.path.join(out, "val.jsonl"), "w") as fva:
        for r in rows:
            split = "val" if (r["dataset"], r["episode"]) in val_eps else "train"
            pics = {c: cv2.imread(os.path.join(r["_dir"], r["images"][c])) for c in CAMS if c in r["images"]}
            if len(pics) != len(CAMS):
                print(f"  skipped {r['id']}: missing pictures ({', '.join(pics)})")
                continue
            prompt = build_prompt(r["state"]["gripper"], r.get("recent", []))
            p = r.get("point")
            for k in range(1 if split == "val" else max(args.aug, 1)):
                imgs, point = [], None
                for c in CAMS:
                    img = cv2.resize(pics[c], (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
                    clicked = p is not None and p["camera"] == c
                    if k > 0:
                        img, moved = _augment(img, rng, (p["x"], p["y"]) if clicked else None)
                        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
                        if clicked:
                            point = moved
                    elif clicked:
                        point = (p["x"], p["y"])
                    rel = f"images/{r['id']}_a{k}_{c}.jpg"
                    cv2.imwrite(os.path.join(out, rel), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    imgs.append(rel)
                ans = r["label"] if point is None else f"{r['label']} {_k(point[0])} {_k(point[1])}"
                sample = {"id": f"{r['id']}_a{k}", "decision": r["id"], "dataset": r["dataset"],
                          "episode": r["episode"], "label": r["label"], "images": imgs, "prompt": prompt,
                          "answer": ans, "augmented": k > 0}
                (fva if split == "val" else ftr).write(json.dumps(sample) + "\n")
                counts[split] += 1
    meta = {"task": "decide", "instruction": INSTRUCTION, "cameras": list(CAMS), "image_size": [IMG_W, IMG_H],
            "labels": list(LABELS), "answer_format": "<label> [x y]  (x y = 0..1000 of that camera's picture)",
            "point_units": "0..1000 of the picture (x right, y down)", "sources": args.decisions,
            "val_episodes": sorted(f"{a}/{b}" for a, b in val_eps), "aug": args.aug, "seed": args.seed,
            "decisions": len(rows), "train_samples": counts["train"], "val_samples": counts["val"],
            "created": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{out}\n  {len(rows)} decisions from {len(episodes)} episodes -> {counts['train']} train samples "
          f"({args.aug} per decision), {counts['val']} val samples")
    print("  validation episodes:", ", ".join(meta["val_episodes"]))
    if not args.no_zip:
        z = shutil.make_archive(out, "zip", os.path.dirname(out), os.path.basename(out))
        print(f"  zip for Google Drive: {z} ({os.path.getsize(z) / 2**20:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())
