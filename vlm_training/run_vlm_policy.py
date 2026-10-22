"""Let the fine-tuned VLM drive the Panda in MuJoCo (closed loop) and watch it pick the cube.

Each phase: render top + wrist cameras -> same prompt as in training -> model predicts the joint
keyframe -> the arm moves there in joint space (no IK at runtime). The expert's keyframe is computed
only to print how far off the model is; it is never executed.

    python run_vlm_policy.py --lora ../checkpoints/<run>/trainer/checkpoint-1000
    python run_vlm_policy.py --lora ... --episodes 20 --headless    # success rate only, no windows
    python run_vlm_policy.py --policy expert                         # sanity check without a model
    python run_vlm_policy.py --lora ... --cube-color blue            # cube color never seen in training
    python run_vlm_policy.py --lora ... --cube-color "#ff8800" --say-color   # also name it in the prompt
    python run_vlm_policy.py --lora ... --floor-color gray           # floor color never seen in training
"""
import os
import sys
os.environ.setdefault("UNSLOTH_BLOCK_COMPILE", "0")  # see train_pick_place.py; must precede `import unsloth`

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import time
import argparse

import cv2
import numpy as np
import mujoco
from PIL import Image

from expert import PickPlaceEnv, PHASES, format_action, GRIPPER_OPEN, GRIPPER_CLOSED
from generate_dataset import build_prompt, make_scene_option, CAMERAS
from eval_pick_place import parse_answer
from pick_place_dataset import PROMPT_STYLES, apply_prompt_style, run_prompt_style

WINDOW = "VLM policy (cameras the model sees)"

COLORS = {
    "green": (0.0, 1.0, 0.0), "red": (1.0, 0.0, 0.0), "blue": (0.1, 0.3, 1.0), "yellow": (1.0, 1.0, 0.0),
    "orange": (1.0, 0.5, 0.0), "purple": (0.5, 0.0, 0.8), "cyan": (0.0, 1.0, 1.0), "magenta": (1.0, 0.0, 1.0),
    "pink": (1.0, 0.6, 0.8), "white": (1.0, 1.0, 1.0), "gray": (0.5, 0.5, 0.5), "black": (0.05, 0.05, 0.05),
    "brown": (0.55, 0.3, 0.1),
}


def parse_color(text):
    """'blue', '#3366ff' or '0.2,0.4,1.0' -> (r, g, b) in 0..1."""
    t = text.strip().lower()
    if t in COLORS:
        return COLORS[t]
    if t.startswith("#") and len(t) == 7:
        return tuple(int(t[i:i + 2], 16) / 255 for i in (1, 3, 5))
    parts = t.split(",")
    if len(parts) == 3:
        rgb = tuple(float(v) for v in parts)
        if all(0 <= v <= 1 for v in rgb):
            return rgb
    raise argparse.ArgumentTypeError(
        f"unknown color {text!r}: use a name ({', '.join(COLORS)}), '#rrggbb' or 'r,g,b' in 0..1")


# Measured in the top/wrist cameras: seen colour ~ offset + gain * material colour (clipped at 1).
# The scene's lights overexpose, so without this correction e.g. orange renders as yellow.
CUBE_LIGHTING = (0.2, 2.0)
FLOOR_LIGHTING = (0.0, 1.9)


def to_material(rgb, lighting):
    """Material colour that renders as rgb under the given (offset, gain) lighting."""
    offset, gain = lighting
    return tuple(min(max((c - offset) / gain, 0.0), 1.0) for c in rgb)


def apply_scene_colors(model, cube_color, floor_color):
    """Recolour the cube and (optionally) the floor. Green cube / no floor colour = training look."""
    cube_rgb = parse_color(cube_color)
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    if cube_color.strip().lower() == "green":
        model.geom_rgba[box] = [0.0, 1.0, 0.0, 1.0]  # exact training material
    else:
        model.geom_rgba[box] = [*to_material(cube_rgb, CUBE_LIGHTING), 1.0]
    print(f"[run] cube color : {cube_color} rgb={tuple(round(v, 2) for v in cube_rgb)}")
    if cube_rgb == COLORS["red"]:
        print("[run] note: the place pad is red too - the model may confuse cube and pad")

    if floor_color:
        floor_rgb = parse_color(floor_color)
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # The floor's colour comes from a textured material that can't be recoloured at runtime;
        # drop the material and use a plain geom colour instead.
        model.geom_matid[floor] = -1
        model.geom_rgba[floor] = [*to_material(floor_rgb, FLOOR_LIGHTING), 1.0]
        print(f"[run] floor color: {floor_color} rgb={tuple(round(v, 2) for v in floor_rgb)}")
        if np.allclose(floor_rgb, cube_rgb, atol=0.15):
            print("[run] note: floor and cube colors are almost the same - the cube will be hard to see")


class VLMPolicy:
    def __init__(self, lora_dir, load_in_4bit=False):
        from unsloth import FastVisionModel
        from eval_pick_place import predict
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


def render_cams(env, renderer, cams, scene_option, jpeg_quality=92):
    """RGB frames as the model saw them in training (incl. the JPEG round trip of the dataset)."""
    frames = []
    for cam in cams:
        renderer.update_scene(env.data, camera=CAMERAS[cam], scene_option=scene_option)
        bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        frames.append(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return frames


def show(frames, lines):
    img = np.hstack([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames])
    bar = np.full((22 * len(lines) + 10, img.shape[1], 3), 30, np.uint8)
    for i, (text, color) in enumerate(lines):
        cv2.putText(bar, text, (10, 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imshow(WINDOW, np.vstack([img, bar]))
    cv2.waitKey(1)


def run_episode(env, renderer, scene_option, policy, rng, cams, viewer, window, prompt_style, cube_word=None):
    env.reset(env.sample_task(rng))
    expert_kfs = env.plan_keyframes()  # reference for error printing only
    phase_errs = []
    for k, kf in enumerate(expert_kfs):
        if viewer is not None:
            viewer.sync()
        frames = render_cams(env, renderer, cams, scene_option)
        prompt = apply_prompt_style(build_prompt(k, env.arm_q(), env.data.ctrl[7]), prompt_style)
        if cube_word:
            prompt = prompt.replace("green cube", f"{cube_word} cube")

        t0 = time.time()
        text = policy([Image.fromarray(f) for f in frames], prompt, kf)
        dt = time.time() - t0

        parsed = parse_answer(text)
        if parsed is None:
            print(f"  {PHASES[k]:10s} unparseable answer {text!r} -> episode aborted")
            return False, phase_errs
        q_deg, grip_word = parsed
        q = np.radians(q_deg)
        grip = GRIPPER_OPEN if grip_word == "open" else GRIPPER_CLOSED
        joint_err = np.abs(q_deg - np.degrees(kf.q)).max()
        pos_err = 1000 * np.linalg.norm(env.forward_site(q) - kf.site_pos)
        grip_note = "" if grip == kf.gripper else "  (expert: other way!)"
        phase_errs.append((PHASES[k], pos_err))
        print(f"  {PHASES[k]:10s} {dt:4.1f}s  max joint err {joint_err:5.1f} deg  "
              f"gripper target off by {pos_err:5.0f} mm  gripper {grip_word}{grip_note}")

        if window:
            show(frames, [
                (f"Phase {k + 1}/{len(PHASES)}: {PHASES[k]}   ({dt:.1f}s to think)", (0, 255, 255)),
                (f"model : {text}", (255, 255, 255)),
                (f"expert: {format_action(kf.q, kf.gripper)}", (150, 150, 150)),
                (f"target off by {pos_err:.0f} mm, max joint err {joint_err:.1f} deg", (0, 255, 0) if pos_err < 15 else (0, 140, 255)),
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
    ap.add_argument("--seed", type=int, default=1000, help="scene seed (dataset used 0, so these are new scenes)")
    ap.add_argument("--cams", default="top,wrist", help="must match the cameras the model was trained on")
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--headless", action="store_true", help="no MuJoCo viewer / camera window")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--prompt-style", choices=PROMPT_STYLES, default=None,
                    help="default: the style the run was trained with (from its config.json)")
    ap.add_argument("--cube-color", default="green",
                    help=f"name ({', '.join(COLORS)}), '#rrggbb' or 'r,g,b' in 0..1. Training only used green")
    ap.add_argument("--say-color", action="store_true",
                    help="also write the color in the instruction (training prompts always said 'green cube')")
    ap.add_argument("--floor-color", default=None,
                    help="same formats as --cube-color. Default: the dark blue training floor")
    args = ap.parse_args()
    for c in filter(None, (args.cube_color, args.floor_color)):
        try:
            parse_color(c)
        except argparse.ArgumentTypeError as e:
            ap.error(str(e))

    if args.policy == "vlm" and not args.lora:
        ap.error("--lora is required for --policy vlm")
    policy = VLMPolicy(args.lora, args.load_in_4bit) if args.policy == "vlm" else ExpertPolicy()
    prompt_style = args.prompt_style or (run_prompt_style(args.lora) if args.lora else "full")
    print(f"[run] prompt style: {prompt_style}")

    cams = args.cams.split(",")
    env = PickPlaceEnv()
    apply_scene_colors(env.model, args.cube_color, args.floor_color)
    cube_word = args.cube_color if args.say_color else None
    print(f"[run] prompt says '{cube_word or 'green'} cube'")
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    scene_option = make_scene_option()
    rng = np.random.default_rng(args.seed)
    results = []

    def run_all(viewer=None):
        for ep in range(args.episodes):
            print(f"episode {ep + 1}/{args.episodes}")
            ok, errs = run_episode(env, renderer, scene_option, policy, rng, cams, viewer,
                                   window=not args.headless, prompt_style=prompt_style, cube_word=cube_word)
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
            run_all(v)
        cv2.destroyAllWindows()

    n_ok = sum(ok for ok, _ in results)
    floor = f", {args.floor_color} floor" if args.floor_color else ""
    print(f"\n=== {args.policy} policy, {args.cube_color} cube{floor}: success {n_ok}/{len(results)} ===")
    by_phase = {}
    for _, errs in results:
        for phase, e in errs:
            by_phase.setdefault(phase, []).append(e)
    for phase in PHASES:
        if phase in by_phase:
            print(f"  {phase:10s} gripper target off by {np.mean(by_phase[phase]):6.1f} mm (mean)")


if __name__ == "__main__":
    main()
