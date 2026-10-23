"""Generate the SO-101 VLA dataset: (top + wrist images, task, phase) -> next joint keyframe.

Same format as ../generate_dataset.py (Panda), with two differences:
  * 5 arm joints instead of 7: answers look like "j: 12.4 -18.7 33.1 71.6 -2.0 | gripper: open"
  * the prompt has no "current joints" line (the image-only style that fixed the Panda model,
    which had learned to copy the current joints and ignore the images)

Each successful expert episode gives 8 samples, captured when a phase is about to start (the moment
the runner will ask the model). Labels are the expert IK keyframes.

    python generate_dataset.py --episodes 3000        # -> datasets/so101_v1_<dd-mm-yyyy-hh-mm-ss>
"""
import os
import json
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
import mujoco

from so101_env import SO101Env, PHASES, format_action, gripper_word

HERE = os.path.dirname(os.path.abspath(__file__))
INSTRUCTION = "Pick up the green cube and place it on the red pad."  # same scene look + task text as the Panda
CAMERAS = {"top": "top_camera", "wrist": "wrist_camera", "front": "front_camera"}


def build_prompt(phase_idx):
    """User text for one sample. Shared with run_policy.py so training and inference match."""
    return (
        f"Task: {INSTRUCTION}\n"
        f"Phase {phase_idx + 1}/{len(PHASES)}: {PHASES[phase_idx]}\n"
        "Output the target joint angles j1..j5 in degrees and the gripper command."
    )


def make_scene_option():
    """Hide sites (gripperframe / tcp markers) from the cameras."""
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    return opt


def render(env, renderer, scene_option, cams):
    frames = {}
    for cam in cams:
        renderer.update_scene(env.data, camera=CAMERAS[cam], scene_option=scene_option)
        frames[cam] = renderer.render().copy()
    return frames


def run_episode(env, renderer, scene_option, cams, rng, noise_deg):
    task = env.sample_task(rng)
    env.reset(task)
    keyframes = env.plan_keyframes()
    if keyframes is None:
        return None, "ik_fail"

    samples = []
    for k, kf in enumerate(keyframes):
        samples.append({
            "phase_idx": k,
            "phase": kf.phase,
            "images": render(env, renderer, scene_option, cams),
            "prompt": build_prompt(k),
            "answer": format_action(kf.q, kf.gripper),
            "current_joints_deg": np.degrees(env.arm_q()).round(2).tolist(),
            "target_joints_deg": np.degrees(kf.q).round(2).tolist(),
            "gripper": gripper_word(kf.gripper),
            "target_tcp_pos": kf.tcp_pos.round(4).tolist(),
        })
        # Small execution noise so the model also sees slightly-off states; labels stay clean.
        q_cmd = kf.q + np.radians(rng.normal(0, noise_deg, len(kf.q))) if noise_deg > 0 else kf.q
        env.move_to(q_cmd, kf.gripper)

    if not env.is_success():
        return None, "task_fail"
    task_info = {
        "cube_xy": task["cube_xy"].round(4).tolist(),
        "cube_yaw_deg": round(float(np.degrees(task["cube_yaw"])), 2),
        "place_xy": task["place_xy"].round(4).tolist(),
    }
    return (task_info, samples), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000, help="successful episodes to collect")
    ap.add_argument("--name", default="so101_v1", help="dataset name, timestamp is appended")
    ap.add_argument("--out", default=None, help="explicit output dir (overrides --name)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--noise-deg", type=float, default=0.5, help="execution noise std per joint")
    ap.add_argument("--cams", default="top,wrist", help=f"comma list from {list(CAMERAS)}")
    ap.add_argument("--width", type=int, default=448)   # multiples of 28 suit Qwen-VL patching
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    args = ap.parse_args()

    cams = args.cams.split(",")
    out = args.out or os.path.join(HERE, "datasets", f"{args.name}_{datetime.now():%d-%m-%Y-%H-%M-%S}")
    os.makedirs(os.path.join(out, "images"), exist_ok=True)

    env = SO101Env()
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    scene_option = make_scene_option()
    rng = np.random.default_rng(args.seed)

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({
            "robot": "SO-101",
            "instruction": INSTRUCTION,
            "phases": PHASES,
            "cameras": cams,
            "image_size": [args.width, args.height],
            "joint_order": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            "units": "degrees, 0.1 deg resolution in answers",
            "answer_format": "j: <j1> ... <j5> | gripper: <open|closed>",
            "prompt": "image only (no current joints)",
            "noise_deg": args.noise_deg,
            "seed": args.seed,
        }, f, indent=2)

    files = {s: open(os.path.join(out, f"{s}.jsonl"), "w") for s in ("train", "val")}
    counts = {"ok": 0, "ik_fail": 0, "task_fail": 0}
    n_samples = {"train": 0, "val": 0}
    t0, ep = time.time(), 0
    while counts["ok"] < args.episodes:
        result, status = run_episode(env, renderer, scene_option, cams, rng, args.noise_deg)
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

        if ep % 50 == 0 or ep == args.episodes:
            rate = ep / (time.time() - t0)
            print(f"{ep}/{args.episodes} episodes | {rate:.1f} ep/s | ETA {(args.episodes - ep) / rate / 60:.1f} min | "
                  f"fails: ik={counts['ik_fail']} task={counts['task_fail']}", flush=True)

    for f in files.values():
        f.close()
    print(f"done: {n_samples['train']} train / {n_samples['val']} val samples in {out}")
    print(f"episode stats: {counts}")


if __name__ == "__main__":
    main()
