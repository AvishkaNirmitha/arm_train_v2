"""Generate the pick dataset: (top + wrist images, task, phase) -> 4 joint angles + gripper. See pick_task.py.

Same images, camera calibration and randomisation as reach_dataset.py. Per episode one sample per phase in
PICK_PHASES order, with --refine REFINE samples after PRE_GRASP (the arm put a bit off the PRE_GRASP pose,
answer = the PRE_GRASP pose):  "j: 12.4 -18.7 33.1 71.6 | gripper: open".

    python pick_dataset.py --episodes 4000        # -> datasets/so101_pick_v1_<dd-mm-yyyy-hh-mm-ss>
    tail -f datasets/so101_pick_v1_*/progress.log # progress (also printed)
"""
import os
import json
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
import mujoco

from pick_task import (PickEnv, PICK_PHASES, INSTRUCTION, format_pick_action, build_pick_prompt, BLOCK_YAW_DEG,
                       GRASP_AIM_LEFT)
from reach_task import REACH_JOINTS, WRIST_ROLL_FIXED, ROLL_RAND_DEG, REFINE_DEG
from real_dataset import CAMERAS, make_scene_option, render, augment
from real_env import (TOP_CAM_CALIB, WEBCAM_RAND_POS, WEBCAM_RAND_DEG, WEBCAM_RAND_FOVY, REAL_GRIPPER_OPEN,
                      REAL_GRIPPER_CLOSED)

HERE = os.path.dirname(os.path.abspath(__file__))


def run_episode(env, renderer, scene_option, cams, rng, noise_deg, augment_images, n_refine):
    task = env.sample_task(rng)
    env.reset(task, rng)
    keyframes = env.plan_keyframes()
    if keyframes is None:
        return None, "ik_fail"
    samples = []

    def add_sample(phase, kf):
        frames = render(env, renderer, scene_option, cams)
        if augment_images:
            frames = {c: augment(f, rng) for c, f in frames.items()}
        current = env.arm_q()[:4]
        k = PICK_PHASES.index(phase)
        samples.append({
            "phase_idx": k,
            "phase": phase,
            "images": frames,
            "prompt": build_pick_prompt(k, current),
            "answer": format_pick_action(kf.q, kf.gripper),
            "current_joints_deg": np.degrees(current).round(2).tolist(),
            "target_joints_deg": np.degrees(kf.q[:4]).round(2).tolist(),
            "target_tcp_pos": kf.tcp_pos.round(4).tolist(),
        })

    def move(kf):
        noise = np.radians(rng.normal(0, noise_deg, 4)) if noise_deg > 0 else np.zeros(4)
        env.move_to(kf.q + np.append(noise, 0.0), kf.gripper)   # wrist_roll stays fixed
        if kf.phase == "LIFT":
            env.note_lift()

    for kf in keyframes:
        add_sample(kf.phase, kf)
        move(kf)
        if kf.phase == "PRE_GRASP" and n_refine:
            for _ in range(n_refine):   # a bit off the PRE_GRASP pose -> the answer is the PRE_GRASP pose
                q_start = env.refine_start(kf.q, rng)
                if q_start is None:
                    continue
                env.move_to(q_start, kf.gripper)
                add_sample("REFINE", kf)
            move(kf)
    if not env.is_success():
        return None, "task_fail"
    return ({"block_xy": task["block_xy"].round(4).tolist(),
             "block_yaw_deg": round(float(np.degrees(task["block_yaw"])), 2)}, samples), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4000, help="successful episodes (7 + --refine samples each)")
    ap.add_argument("--refine", type=int, default=2, help="REFINE samples per episode")
    ap.add_argument("--name", default="so101_pick_v1", help="dataset name, timestamp is appended")
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
    log = open(os.path.join(out, "progress.log"), "w")

    def report(text):
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    env = PickEnv(randomize=not args.no_randomize)
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    scene_option = make_scene_option()
    rng = np.random.default_rng(args.seed)

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"robot": "SO-101 as built 04-11-2026 (gripper back, wrist_roll taped, webcam on top of the wrist)",
                   "task": "pick", "instruction": INSTRUCTION, "phases": PICK_PHASES, "cameras": cams,
                   "image_size": [args.width, args.height], "joint_order": REACH_JOINTS,
                   "wrist_roll_fixed_rad": WRIST_ROLL_FIXED,
                   "answer_format": "j: <j1> <j2> <j3> <j4> | gripper: <open|closed>",
                   "real_gripper": {"open": REAL_GRIPPER_OPEN, "closed": REAL_GRIPPER_CLOSED},
                   "top_camera_calibrated": os.path.exists(TOP_CAM_CALIB),
                   "randomized": not args.no_randomize, "noise_deg": args.noise_deg, "seed": args.seed,
                   "refine_per_episode": args.refine, "refine_deg": REFINE_DEG, "roll_rand_deg": ROLL_RAND_DEG,
                   "block_yaw_deg": BLOCK_YAW_DEG, "grasp_aim_left_m": GRASP_AIM_LEFT,
                   "webcam_rand": {"pos_m": WEBCAM_RAND_POS, "deg": WEBCAM_RAND_DEG, "fovy_deg": WEBCAM_RAND_FOVY}},
                  f, indent=2)

    files = {s: open(os.path.join(out, f"{s}.jsonl"), "w") for s in ("train", "val")}
    counts = {"ok": 0, "ik_fail": 0, "task_fail": 0}
    n_samples = {"train": 0, "val": 0}
    t0, ep = time.time(), 0
    report(f"generating {args.episodes} episodes into {out}")
    while counts["ok"] < args.episodes:
        result, status = run_episode(env, renderer, scene_option, cams, rng, args.noise_deg,
                                     augment_images=not args.no_randomize, n_refine=args.refine)
        counts[status] += 1
        if result is None:
            continue
        task_info, samples = result
        split = "val" if rng.random() < args.val_frac else "train"
        for i, s in enumerate(samples):
            paths = []
            for cam in cams:
                rel = f"images/ep{ep:05d}_s{i}_{cam}.jpg"
                cv2.imwrite(os.path.join(out, rel), cv2.cvtColor(s["images"][cam], cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                paths.append(rel)
            row = {k: v for k, v in s.items() if k != "images"}
            row.update({"id": f"ep{ep:05d}_s{i}", "episode": ep, "images": paths, **task_info})
            files[split].write(json.dumps(row) + "\n")
            n_samples[split] += 1
        ep += 1
        if ep % 50 == 0 or ep == args.episodes:
            rate = ep / (time.time() - t0)
            report(f"{ep}/{args.episodes} episodes ({100 * ep / args.episodes:.0f} %) | {rate:.2f} ep/s | "
                   f"ETA {(args.episodes - ep) / rate / 60:.0f} min | fails: ik={counts['ik_fail']} "
                   f"task={counts['task_fail']}")
    for f in files.values():
        f.close()
    report(f"done: {n_samples['train']} train / {n_samples['val']} val samples in {out}")
    report(f"episode stats: {counts}")
    log.close()


if __name__ == "__main__":
    main()
