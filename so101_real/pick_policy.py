"""Let a trained pick model drive the SO-101 in the sim (closed loop) and score it. See pick_task.py.

Each phase: render top + wrist cameras -> prompt -> model answers 4 joint angles + gripper -> the arm moves
there (wrist_roll stays fixed). After PRE_GRASP, --refine REFINE steps (prompt includes the current joints)
correct the pose. Success = the block was lifted and ends back on the mat, upright, near where it was.

    python pick_policy.py --lora checkpoints/<run>/lora --episodes 20 --headless
    python pick_policy.py --policy expert --headless          # check the runner without a model
"""
import os
os.environ.setdefault("UNSLOTH_BLOCK_COMPILE", "0")  # see ../so101_vla/train.py; must precede `import unsloth`

import time
import argparse

import cv2
import numpy as np
import mujoco
from PIL import Image

from pick_task import PickEnv, PICK_PHASES, format_pick_action, parse_pick_action, build_pick_prompt, gripper_word
from real_dataset import make_scene_option, render
from real_policy import VLMPolicy, model_view, show


class ExpertPolicy:
    def __call__(self, images, prompt, expert_kf):
        return format_pick_action(expert_kf.q, expert_kf.gripper)


def pick_steps(keyframes, n_refine):
    """(phase name, expert keyframe) in execution order: n_refine REFINE steps after PRE_GRASP."""
    steps = []
    for kf in keyframes:
        steps.append((kf.phase, kf))
        if kf.phase == "PRE_GRASP":
            steps += [("REFINE", kf)] * n_refine
    return steps


def run_episode(env, renderer, scene_option, policy, rng, cams, viewer, window, n_refine):
    env.reset(env.sample_task(rng), rng)
    expert = env.plan_keyframes()
    if expert is None:
        return None, []
    errs = []
    for phase, kf in pick_steps(expert, n_refine):
        k = PICK_PHASES.index(phase)
        if viewer is not None:
            viewer.sync()
        rendered = render(env, renderer, scene_option, cams)
        frames = model_view([rendered[c] for c in cams])
        t0 = time.time()
        text = policy([Image.fromarray(f) for f in frames], build_pick_prompt(k, env.arm_q()[:4]), kf)
        dt = time.time() - t0
        parsed = parse_pick_action(text)
        if parsed is None:
            print(f"  {phase:9s} unparseable answer {text!r} -> episode aborted")
            return False, errs
        q4, grip = parsed
        q = env.full_q(q4)
        err = 1000 * np.linalg.norm(env.forward_tcp(q) - kf.tcp_pos)
        grip_ok = gripper_word(grip) == gripper_word(kf.gripper)
        errs.append((phase, err))
        print(f"  {phase:9s} {dt:4.1f}s  grasp point off by {err:5.1f} mm{'' if grip_ok else '  WRONG GRIPPER'}"
              f"   model: {text}")
        if window:
            show(frames, [(f"Phase {k + 1}/{len(PICK_PHASES)}: {phase}", (0, 255, 255)),
                          (f"model : {text}", (255, 255, 255)),
                          (f"expert: {format_pick_action(kf.q, kf.gripper)}", (150, 150, 150)),
                          (f"grasp point off by {err:.1f} mm", (0, 255, 0) if err < 7 else (0, 140, 255))])
        env.move_to(q, grip, viewer=viewer)
        if phase == "LIFT":
            env.note_lift()
        if viewer is not None and not viewer.is_running():
            return False, errs
    return env.is_success(), errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", help="LoRA adapter dir (a run's lora/ or trainer/checkpoint-N)")
    ap.add_argument("--policy", choices=["vlm", "expert"], default="vlm")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--refine", type=int, default=2, help="REFINE steps after PRE_GRASP (0 = none)")
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
    env = PickEnv(randomize=not args.no_randomize)
    renderer = mujoco.Renderer(env.model, height=336, width=448)
    scene_option = make_scene_option()
    rng = np.random.default_rng(args.seed)
    results = []

    def run_all(viewer=None):
        for ep in range(args.episodes):
            print(f"episode {ep + 1}/{args.episodes}")
            ok, errs = run_episode(env, renderer, scene_option, policy, rng, cams, viewer, not args.headless,
                                   args.refine)
            if ok is None:
                print("  block out of reach, skipped")
                continue
            results.append((ok, bool(env.block_lifted), errs))
            print(f"  -> {'SUCCESS' if ok else 'FAIL'}  (lifted: {env.block_lifted}, block ended "
                  f"{env.block_moved() * 1000:.0f} mm from its start)")
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
    print(f"\n=== SO-101 pick, {args.policy} policy: success {n_ok}/{len(results)} ===")
    if results:
        print(f"  block lifted: {sum(r[1] for r in results)}/{len(results)}")
        for phase in PICK_PHASES:
            e = [x for r in results for p, x in r[2] if p == phase]
            if e:
                print(f"  {phase:9s} grasp point off by {np.mean(e):6.1f} mm (mean), median {np.median(e):.1f} mm")


if __name__ == "__main__":
    main()
