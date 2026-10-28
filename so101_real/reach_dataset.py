"""Generate the reach dataset: (top + wrist images, task, phase) -> 4 joint angles. See reach_task.py.

Same images, camera calibration and randomisation as real_dataset.py, but 2 phases per episode (HOVER, REACH)
and answers with the 4 working joints only:  "j: 12.4 -18.7 33.1 71.6".

    python reach_dataset.py --episodes 8000     # -> datasets/so101_reach_v1_<dd-mm-yyyy-hh-mm-ss>
"""
import os
import json
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
import mujoco

from reach_task import (ReachEnv, REACH_PHASES, REACH_JOINTS, INSTRUCTION, format_reach_action,
                        build_reach_prompt, WRIST_ROLL_FIXED)
from real_dataset import CAMERAS, make_scene_option, render, augment
from real_env import TOP_CAM_CALIB

HERE = os.path.dirname(os.path.abspath(__file__))


def run_episode(env, renderer, scene_option, cams, rng, noise_deg, augment_images):
    task = env.sample_task(rng)
    env.reset(task, rng)
    keyframes = env.plan_keyframes()
    if keyframes is None:
        return None, "ik_fail"
    samples = []
    for k, kf in enumerate(keyframes):
        frames = render(env, renderer, scene_option, cams)
        if augment_images:
            frames = {c: augment(f, rng) for c, f in frames.items()}
        samples.append({
            "phase_idx": k,
            "phase": kf.phase,
            "images": frames,
            "prompt": build_reach_prompt(k),
            "answer": format_reach_action(kf.q),
            "current_joints_deg": np.degrees(env.arm_q()[:4]).round(2).tolist(),
            "target_joints_deg": np.degrees(kf.q[:4]).round(2).tolist(),
            "target_tcp_pos": kf.tcp_pos.round(4).tolist(),
        })
        noise = np.radians(rng.normal(0, noise_deg, 4)) if noise_deg > 0 else np.zeros(4)
        env.move_to(kf.q + np.append(noise, 0.0), kf.gripper)   # wrist_roll stays fixed
    if not env.is_success():
        return None, "task_fail"
    return ({"block_xy": task["block_xy"].round(4).tolist(),
             "block_yaw_deg": round(float(np.degrees(task["block_yaw"])), 2),
             "note_xy": task["note_xy"].round(4).tolist()}, samples), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8000, help="successful episodes (2 samples each)")
    ap.add_argument("--name", default="so101_reach_v1", help="dataset name, timestamp is appended")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--noise-deg", type=float, default=0.5, help="execution noise std per joint")
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("--cams", default="top,wrist")
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    args = ap.parse_args()

    if not os.path.exists(TOP_CAM_CALIB):
        print("WARNING: no top_camera_calib.json - rendering from the default top camera, not your real one")
    cams = args.cams.split(",")
    out = args.out or os.path.join(HERE, "datasets", f"{args.name}_{datetime.now():%d-%m-%Y-%H-%M-%S}")
    os.makedirs(os.path.join(out, "images"), exist_ok=True)

    env = ReachEnv(randomize=not args.no_randomize)
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    scene_option = make_scene_option()
    rng = np.random.default_rng(args.seed)

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"robot": "SO-101 as built (wrist_roll fixed, no moving jaw, zip-tied webcam)", "task": "reach", "instruction": INSTRUCTION,
                   "phases": REACH_PHASES, "cameras": cams, "image_size": [args.width, args.height],
                   "joint_order": REACH_JOINTS, "wrist_roll_fixed_rad": WRIST_ROLL_FIXED,
                   "answer_format": "j: <j1> <j2> <j3> <j4>", "top_camera_calibrated": os.path.exists(TOP_CAM_CALIB),
                   "randomized": not args.no_randomize, "noise_deg": args.noise_deg, "seed": args.seed}, f, indent=2)

    files = {s: open(os.path.join(out, f"{s}.jsonl"), "w") for s in ("train", "val")}
    counts = {"ok": 0, "ik_fail": 0, "task_fail": 0}
    n_samples = {"train": 0, "val": 0}
    t0, ep = time.time(), 0
    while counts["ok"] < args.episodes:
        result, status = run_episode(env, renderer, scene_option, cams, rng, args.noise_deg,
                                     augment_images=not args.no_randomize)
        counts[status] += 1
        if result is None:
            continue
        task_info, samples = result
        split = "val" if rng.random() < args.val_frac else "train"
        for s in samples:
            paths = []
            for cam in cams:
                rel = f"images/ep{ep:05d}_p{s['phase_idx']}_{cam}.jpg"
                cv2.imwrite(os.path.join(out, rel), cv2.cvtColor(s["images"][cam], cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                paths.append(rel)
            row = {k: v for k, v in s.items() if k != "images"}
            row.update({"id": f"ep{ep:05d}_p{s['phase_idx']}", "episode": ep, "images": paths, **task_info})
            files[split].write(json.dumps(row) + "\n")
            n_samples[split] += 1
        ep += 1
        if ep % 100 == 0 or ep == args.episodes:
            rate = ep / (time.time() - t0)
            print(f"{ep}/{args.episodes} episodes | {rate:.1f} ep/s | ETA {(args.episodes - ep) / rate / 60:.1f} min | "
                  f"fails: ik={counts['ik_fail']} task={counts['task_fail']}", flush=True)
    for f in files.values():
        f.close()
    print(f"done: {n_samples['train']} train / {n_samples['val']} val samples in {out}")
    print(f"episode stats: {counts}")


if __name__ == "__main__":
    main()
