"""Generate a VLM training dataset: (camera images + current joints + phase) -> next joint keyframe.

Each successful expert episode yields 8 samples, one per phase. A sample is captured at the
moment the phase is about to be executed (the same moment the VLM runner will query the model),
and its label is the expert's IK keyframe for that phase, in degrees.

    python generate_dataset.py --episodes 3000        # -> datasets/pick_place_v1_21-10-2026-11-20-05
    python generate_dataset.py --name pick_place_v2   # -> datasets/pick_place_v2_<dd-mm-yyyy-hh-mm-ss>

Output:
    <out>/images/ep00012_p3_top.jpg ...
    <out>/train.jsonl, <out>/val.jsonl   (one sample per line, split by episode)
    <out>/meta.json
"""
import os
import json
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
import mujoco

from expert import PickPlaceEnv, PHASES, format_action, gripper_word

INSTRUCTION = "Pick up the green cube and place it on the red pad."

CAMERAS = {"top": "top_camera", "wrist": "wrist_camera", "front": "front_camera"}


def build_prompt(phase_idx, current_q, current_gripper):
    """User text for one sample. Kept in one place so the VLM runner can reuse it."""
    joints = " ".join(f"{v:.1f}" for v in np.degrees(current_q))
    return (
        f"Task: {INSTRUCTION}\n"
        f"Phase {phase_idx + 1}/{len(PHASES)}: {PHASES[phase_idx]}\n"
        f"Current joints (deg): {joints}\n"
        f"Current gripper: {gripper_word(current_gripper)}\n"
        "Output the target joint angles j1..j7 in degrees and the gripper command."
    )


def make_scene_option():
    """Hide sites (the gripper site renders as a white sphere in the wrist view)."""
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    return opt


def run_episode(env, renderer, cams, rng, noise_deg):
    """Run one expert episode. Returns (task, list of samples with in-memory images) or None."""
    task = env.sample_task(rng)
    env.reset(task)
    keyframes = env.plan_keyframes()
    if keyframes is None:
        return None, "ik_fail"

    scene_option = make_scene_option()
    samples = []
    for k, kf in enumerate(keyframes):
        images = {}
        for cam in cams:
            renderer.update_scene(env.data, camera=CAMERAS[cam], scene_option=scene_option)
            images[cam] = renderer.render().copy()
        current_q, current_grip = env.arm_q(), env.data.ctrl[7]
        samples.append({
            "phase_idx": k,
            "phase": kf.phase,
            "images": images,
            "prompt": build_prompt(k, current_q, current_grip),
            "answer": format_action(kf.q, kf.gripper),
            "current_joints_deg": np.degrees(current_q).round(2).tolist(),
            "target_joints_deg": np.degrees(kf.q).round(2).tolist(),
            "gripper": gripper_word(kf.gripper),
            "target_site_pos": kf.site_pos.round(4).tolist(),
        })
        # Execute with a small joint perturbation so the model also sees slightly-off states.
        # Labels stay the clean IK keyframes; episodes that fail are discarded below.
        q_cmd = kf.q + np.radians(rng.normal(0, noise_deg, 7)) if noise_deg > 0 else kf.q
        env.move_to(q_cmd, kf.gripper)

    if not env.is_success():
        return None, "task_fail"
    cube_pos, cube_yaw = env.cube_pose()
    task_info = {
        "cube_xy": task["cube_xy"].round(4).tolist(),
        "cube_yaw_deg": round(float(np.degrees(task["cube_yaw"])), 2),
        "pad_xy": task["pad_xy"].round(4).tolist(),
    }
    return (task_info, samples), "ok"


def default_out_dir(name, root="datasets"):
    """datasets/<name>_<dd-mm-yyyy-hh-mm-ss>"""
    return os.path.join(root, f"{name}_{datetime.now():%d-%m-%Y-%H-%M-%S}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000, help="successful episodes to collect")
    ap.add_argument("--name", default="pick_place_v1", help="dataset name, timestamp is appended")
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
    if args.out is None:
        args.out = default_out_dir(args.name)
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    env = PickPlaceEnv()
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    rng = np.random.default_rng(args.seed)

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({
            "instruction": INSTRUCTION,
            "phases": PHASES,
            "cameras": cams,
            "image_size": [args.width, args.height],
            "joint_order": [f"joint{i}" for i in range(1, 8)],
            "units": "degrees, 0.1 deg resolution in answers",
            "answer_format": "j: <j1> ... <j7> | gripper: <open|closed>",
            "gripper_cmd": {"open": 0.04, "closed": 0.0},
            "noise_deg": args.noise_deg,
            "seed": args.seed,
            "scene": "mujoco_pick_place/franka_emika_panda/vla_scene.xml",
        }, f, indent=2)

    files = {s: open(os.path.join(args.out, f"{s}.jsonl"), "w") for s in ("train", "val")}
    counts = {"ok": 0, "ik_fail": 0, "task_fail": 0}
    n_samples = {"train": 0, "val": 0}
    t0 = time.time()
    ep = 0
    while counts["ok"] < args.episodes:
        result, status = run_episode(env, renderer, cams, rng, args.noise_deg)
        counts[status] += 1
        if result is None:
            continue
        task_info, samples = result
        split = "val" if rng.random() < args.val_frac else "train"
        for s in samples:
            paths = []
            for cam in cams:
                rel = f"images/ep{ep:05d}_p{s['phase_idx']}_{cam}.jpg"
                cv2.imwrite(os.path.join(args.out, rel), cv2.cvtColor(s["images"][cam], cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                paths.append(rel)
            row = {k: v for k, v in s.items() if k != "images"}
            row.update({"id": f"ep{ep:05d}_p{s['phase_idx']}", "episode": ep, "images": paths, **task_info})
            files[split].write(json.dumps(row) + "\n")
            n_samples[split] += 1
        ep += 1

        if ep % 50 == 0 or ep == args.episodes:
            rate = ep / (time.time() - t0)
            eta = (args.episodes - ep) / rate
            print(f"{ep}/{args.episodes} episodes | {rate:.1f} ep/s | ETA {eta / 60:.1f} min | "
                  f"fails: ik={counts['ik_fail']} task={counts['task_fail']}", flush=True)

    for f in files.values():
        f.close()
    print(f"done: {n_samples['train']} train / {n_samples['val']} val samples in {args.out}")
    print(f"episode stats: {counts}")


if __name__ == "__main__":
    main()
