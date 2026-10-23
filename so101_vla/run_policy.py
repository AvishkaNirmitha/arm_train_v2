"""Let the fine-tuned VLA drive the SO-101 in MuJoCo (closed loop) and watch it pick the cube.

Each phase: render top + wrist cameras -> same prompt as in training -> model predicts 5 joint angles +
gripper -> the arm moves there in joint space (no IK at runtime). The expert keyframe is computed only to
print how far off the model is; it is never executed.

    python run_policy.py --lora checkpoints/<run>/trainer/checkpoint-1000
    python run_policy.py --lora checkpoints/<run>/lora --episodes 20 --headless   # success rate only
    python run_policy.py --policy expert                                          # check without a model
"""
import os
import sys
os.environ.setdefault("UNSLOTH_BLOCK_COMPILE", "0")  # see train.py; must precede `import unsloth`

import time
import argparse

import cv2
import numpy as np
import mujoco
from PIL import Image

from so101_env import SO101Env, PHASES, format_action, parse_action
from generate_dataset import build_prompt, make_scene_option, render

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(HERE), "vlm_training"))

WINDOW = "SO-101 VLA (cameras the model sees)"


class VLMPolicy:
    def __init__(self, lora_dir, load_in_4bit=False):
        from unsloth import FastVisionModel
        from eval_pick_place import predict  # generic greedy decode, shared with the Panda
        print(f"[run] loading {lora_dir}")
        self.model, self.tokenizer = FastVisionModel.from_pretrained(lora_dir, load_in_4bit=load_in_4bit)
        FastVisionModel.for_inference(self.model)
        self._predict = predict

    def __call__(self, images, prompt, expert_kf):
        return self._predict(self.model, self.tokenizer, images, prompt)


class ExpertPolicy:
    """Answers with the expert keyframe - checks the runner itself, no model needed."""
    def __call__(self, images, prompt, expert_kf):
        return format_action(expert_kf.q, expert_kf.gripper)


def model_view(frames, jpeg_quality=92):
    """Frames as the model saw them in training (incl. the dataset's JPEG round trip)."""
    out = []
    for f in frames:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        out.append(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return out


def show(frames, lines):
    img = np.hstack([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames])
    bar = np.full((22 * len(lines) + 10, img.shape[1], 3), 30, np.uint8)
    for i, (text, color) in enumerate(lines):
        cv2.putText(bar, text, (10, 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imshow(WINDOW, np.vstack([img, bar]))
    cv2.waitKey(1)


def run_episode(env, renderer, scene_option, policy, rng, cams, viewer, window):
    env.reset(env.sample_task(rng))
    expert_kfs = env.plan_keyframes()  # reference for error printing only
    phase_errs = []
    for k, kf in enumerate(expert_kfs):
        if viewer is not None:
            viewer.sync()
        rendered = render(env, renderer, scene_option, cams)
        frames = model_view([rendered[c] for c in cams])

        t0 = time.time()
        text = policy([Image.fromarray(f) for f in frames], build_prompt(k), kf)
        dt = time.time() - t0

        parsed = parse_action(text)
        if parsed is None:
            print(f"  {PHASES[k]:10s} unparseable answer {text!r} -> episode aborted")
            return False, phase_errs
        q, grip = parsed
        joint_err = np.abs(np.degrees(q - kf.q)).max()
        pos_err = 1000 * np.linalg.norm(env.forward_tcp(q) - kf.tcp_pos)
        grip_note = "" if grip == kf.gripper else "  (expert: other way!)"
        phase_errs.append((PHASES[k], pos_err))
        print(f"  {PHASES[k]:10s} {dt:4.1f}s  max joint err {joint_err:5.1f} deg  "
              f"TCP target off by {pos_err:5.1f} mm{grip_note}")

        if window:
            show(frames, [
                (f"Phase {k + 1}/{len(PHASES)}: {PHASES[k]}   ({dt:.1f}s to think)", (0, 255, 255)),
                (f"model : {text}", (255, 255, 255)),
                (f"expert: {format_action(kf.q, kf.gripper)}", (150, 150, 150)),
                (f"target off by {pos_err:.1f} mm, max joint err {joint_err:.1f} deg",
                 (0, 255, 0) if pos_err < 5 else (0, 140, 255)),
            ])

        env.move_to(q, grip, viewer=viewer)
        if viewer is not None and not viewer.is_running():
            return False, phase_errs
    return env.is_success(), phase_errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", help="LoRA adapter dir (a run's lora/ or trainer/checkpoint-N)")
    ap.add_argument("--policy", choices=["vlm", "expert"], default="vlm")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1000, help="scene seed (the dataset used 0, so these are new scenes)")
    ap.add_argument("--cams", default="top,wrist", help="must match the cameras the model was trained on")
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--headless", action="store_true", help="no MuJoCo viewer / camera window")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    if args.policy == "vlm" and not args.lora:
        ap.error("--lora is required for --policy vlm")
    policy = VLMPolicy(args.lora, args.load_in_4bit) if args.policy == "vlm" else ExpertPolicy()

    cams = args.cams.split(",")
    env = SO101Env()
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    scene_option = make_scene_option()
    rng = np.random.default_rng(args.seed)
    results = []

    def run_all(viewer=None):
        for ep in range(args.episodes):
            print(f"episode {ep + 1}/{args.episodes}")
            ok, errs = run_episode(env, renderer, scene_option, policy, rng, cams, viewer, window=not args.headless)
            results.append((ok, errs))
            print(f"  -> {'SUCCESS' if ok else 'FAIL'}")
            if viewer is not None:
                if not viewer.is_running():
                    return
                env.step(int(1.0 / env.model.opt.timestep), viewer)  # pause to see the result

    if args.headless:
        run_all()
    else:
        from mujoco import viewer
        with viewer.launch_passive(env.model, env.data) as v:
            v.cam.lookat[:] = [0.18, 0.0, 0.05]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.7, 150, -25
            run_all(v)
        cv2.destroyAllWindows()

    n_ok = sum(ok for ok, _ in results)
    print(f"\n=== SO-101 {args.policy} policy: success {n_ok}/{len(results)} ===")
    by_phase = {}
    for _, errs in results:
        for phase, e in errs:
            by_phase.setdefault(phase, []).append(e)
    for phase in PHASES:
        if phase in by_phase:
            print(f"  {phase:10s} TCP target off by {np.mean(by_phase[phase]):6.1f} mm (mean)")


if __name__ == "__main__":
    main()
