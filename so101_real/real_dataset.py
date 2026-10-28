"""Generate the real-setup SO-101 VLA dataset: (top + wrist images, task, phase) -> next joint keyframe.

Same format as ../so101_vla/generate_dataset.py, in the scene of ../scene.xml (see real_env.py), plus
sim-to-real randomisation:
  * per episode (real_env.py): light strength / position, table / mat / note / block colour, top camera
    pose (+-1 cm, ~1.5 deg) and field of view
  * per image (here): sensor noise, brightness / contrast, occasional slight blur

    python real_dataset.py --episodes 3000      # -> datasets/so101_real_v1_<dd-mm-yyyy-hh-mm-ss>
"""
import os
import json
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
import mujoco

from real_env import RealSetupEnv, PHASES, format_action, gripper_word

HERE = os.path.dirname(os.path.abspath(__file__))
INSTRUCTION = "Pick up the white block and place it on the green note."
CAMERAS = {"top": "top_camera", "wrist": "wrist_camera"}


def build_prompt(phase_idx):
    """User text for one sample. Shared with real_policy.py so training and inference match."""
    return (
        f"Task: {INSTRUCTION}\n"
        f"Phase {phase_idx + 1}/{len(PHASES)}: {PHASES[phase_idx]}\n"
        "Output the target joint angles j1..j5 in degrees and the gripper command."
    )


def make_scene_option():
    """Hide sites (gripperframe / tcp markers) and the grasp pads from the cameras."""
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    return opt


def render(env, renderer, scene_option, cams):
    frames = {}
    for cam in cams:
        renderer.update_scene(env.data, camera=CAMERAS[cam], scene_option=scene_option)
        frames[cam] = renderer.render().copy()
    return frames


def augment(img, rng):
    """Camera-like degradations so the model doesn't rely on perfectly clean renders."""
    x = img.astype(np.float32)
    x = (x - 128.0) * rng.uniform(0.8, 1.25) + 128.0 + rng.uniform(-25, 25)  # contrast, brightness (exposure)
    x += rng.normal(0, rng.uniform(0, 4), x.shape)                           # sensor noise
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        x = cv2.GaussianBlur(x, (3, 3), rng.uniform(0.3, 0.8))
    return x


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
        "block_xy": task["block_xy"].round(4).tolist(),
        "block_yaw_deg": round(float(np.degrees(task["block_yaw"])), 2),
        "note_xy": task["note_xy"].round(4).tolist(),
        "note_yaw_deg": round(float(np.degrees(task["note_yaw"])), 2),
    }
    return (task_info, samples), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000, help="successful episodes to collect")
    ap.add_argument("--name", default="so101_real_v1", help="dataset name, timestamp is appended")
    ap.add_argument("--out", default=None, help="explicit output dir (overrides --name)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--noise-deg", type=float, default=0.5, help="execution noise std per joint")
    ap.add_argument("--no-randomize", action="store_true", help="nominal scene look, no image augmentation")
    ap.add_argument("--cams", default="top,wrist", help=f"comma list from {list(CAMERAS)}")
    ap.add_argument("--width", type=int, default=448)   # multiples of 28 suit Qwen-VL patching
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    args = ap.parse_args()

    cams = args.cams.split(",")
    out = args.out or os.path.join(HERE, "datasets", f"{args.name}_{datetime.now():%d-%m-%Y-%H-%M-%S}")
    os.makedirs(os.path.join(out, "images"), exist_ok=True)

    env = RealSetupEnv(randomize=not args.no_randomize)
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    scene_option = make_scene_option()
    rng = np.random.default_rng(args.seed)

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({
            "robot": "SO-101",
            "scene": "real setup (../scene.xml), robot on the mat's front edge (scene.xml robot_mount)",
            "instruction": INSTRUCTION,
            "phases": PHASES,
            "cameras": cams,
            "image_size": [args.width, args.height],
            "joint_order": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            "units": "degrees, 0.1 deg resolution in answers",
            "answer_format": "j: <j1> ... <j5> | gripper: <open|closed>",
            "prompt": "image only (no current joints)",
            "randomized": not args.no_randomize,
            "noise_deg": args.noise_deg,
            "seed": args.seed,
        }, f, indent=2)

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
