"""Let a trained reach model drive the SO-101 in the sim (closed loop) and score it. See reach_task.py.

Each phase: render top + wrist cameras -> prompt -> model answers 4 joint angles -> the arm moves there
(wrist_roll and the gripper stay fixed). Reports how far the gripper ends from the target above the block.

    python reach_policy.py --lora checkpoints/<run>/trainer/checkpoint-1000
    python reach_policy.py --lora checkpoints/<run>/lora --episodes 30 --headless
    python reach_policy.py --policy expert --headless          # check the runner without a model
"""
import os
os.environ.setdefault("UNSLOTH_BLOCK_COMPILE", "0")  # see ../so101_vla/train.py; must precede `import unsloth`

import time
import argparse

import cv2
import numpy as np
import mujoco
from PIL import Image

from reach_task import ReachEnv, REACH_PHASES, format_reach_action, parse_reach_action, build_reach_prompt
from real_dataset import make_scene_option, render
from real_policy import VLMPolicy, model_view, show


class ExpertPolicy:
    def __call__(self, images, prompt, expert_kf):
        return format_reach_action(expert_kf.q)


def run_episode(env, renderer, scene_option, policy, rng, cams, viewer, window):
    env.reset(env.sample_task(rng), rng)
    expert = env.plan_keyframes()
    errs = []
    for k, kf in enumerate(expert):
        if viewer is not None:
            viewer.sync()
        rendered = render(env, renderer, scene_option, cams)
        frames = model_view([rendered[c] for c in cams])
        t0 = time.time()
        text = policy([Image.fromarray(f) for f in frames], build_reach_prompt(k), kf)
        dt = time.time() - t0
        q4 = parse_reach_action(text)
        if q4 is None:
            print(f"  {REACH_PHASES[k]:6s} unparseable answer {text!r} -> episode aborted")
            return False, errs
        q = env.full_q(q4)
        err = 1000 * np.linalg.norm(env.forward_cam(q) - kf.tcp_pos)   # tcp_pos = planned wrist camera point
        errs.append((REACH_PHASES[k], err))
        print(f"  {REACH_PHASES[k]:6s} {dt:4.1f}s  target off by {err:5.1f} mm   model: {text}")
        if window:
            show(frames, [(f"Phase {k + 1}/{len(REACH_PHASES)}: {REACH_PHASES[k]}", (0, 255, 255)),
                          (f"model : {text}", (255, 255, 255)),
                          (f"expert: {format_reach_action(kf.q)}", (150, 150, 150)),
                          (f"target off by {err:.1f} mm", (0, 255, 0) if err < 15 else (0, 140, 255))])
        env.move_to(q, kf.gripper, viewer=viewer)
        if viewer is not None and not viewer.is_running():
            return False, errs
    return env.is_success(), errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", help="LoRA adapter dir (a run's lora/ or trainer/checkpoint-N)")
    ap.add_argument("--policy", choices=["vlm", "expert"], default="vlm")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1000, help="scene seed (the dataset used 0, so these are new scenes)")
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("--cams", default="top,wrist")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    if args.policy == "vlm" and not args.lora:
        ap.error("--lora is required for --policy vlm")
    policy = VLMPolicy(args.lora, args.load_in_4bit) if args.policy == "vlm" else ExpertPolicy()
    cams = args.cams.split(",")
    env = ReachEnv(randomize=not args.no_randomize)
    renderer = mujoco.Renderer(env.model, height=336, width=448)
    scene_option = make_scene_option()
    rng = np.random.default_rng(args.seed)
    results = []

    def run_all(viewer=None):
        for ep in range(args.episodes):
            print(f"episode {ep + 1}/{args.episodes}")
            ok, errs = run_episode(env, renderer, scene_option, policy, rng, cams, viewer, not args.headless)
            results.append((ok, env.reach_error() * 1000, env.block_moved() * 1000, errs))
            print(f"  -> {'SUCCESS' if ok else 'FAIL'}  (wrist camera {env.reach_error() * 1000:.1f} mm from the target, "
                  f"block moved {env.block_moved() * 1000:.1f} mm)")
            if viewer is not None:
                if not viewer.is_running():
                    return
                env.step(int(1.0 / env.model.opt.timestep), viewer)

    if args.headless:
        run_all()
    else:
        from mujoco import viewer
        with viewer.launch_passive(env.model, env.data) as v:
            v.cam.lookat[:] = [0.25, 0.0, 0.05]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.9, 200, -30
            run_all(v)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    n_ok = sum(r[0] for r in results)
    print(f"\n=== SO-101 reach, {args.policy} policy: success {n_ok}/{len(results)} ===")
    if results:
        final = [r[1] for r in results]
        print(f"  final wrist camera error: median {np.median(final):.1f} mm, worst {max(final):.1f} mm (success < 15 mm)")
        print(f"  block knocked (> 5 mm): {sum(r[2] > 5 for r in results)}/{len(results)}")
        for phase in REACH_PHASES:
            e = [x for r in results for p, x in r[3] if p == phase]
            if e:
                print(f"  {phase:6s} target off by {np.mean(e):6.1f} mm (mean)")


if __name__ == "__main__":
    main()
