"""Run the trained VLA on the REAL SO-101: real top + wrist cameras -> model -> joint angles -> real arm.

Same loop as real_policy.py (8 phases, one model answer per phase, smooth joint moves in between), but the
images come from real cameras and the moves go to arm_server.py. wrist_roll and the gripper are not mounted
/ broken for now: the model still predicts them, they are printed and NOT sent (so the block can't actually
be grasped yet - this run checks where the arm goes).

Safety: --no-arm runs cameras + model only; by default every move waits for ENTER; slow moves (15 deg/s);
targets whose gripper point would be below the table are refused; the first move goes to a safe pose above
the mat; arm_server.py clamps to the calibrated limits and stops on Ctrl+C.

Run in .venv, with arm_server.py running in .venv-lerobot (not needed with --no-arm):

    python so101_real/real_robot_policy.py --lora <ckpt> --no-arm   # look first (cameras 2 / 4)
    python so101_real/real_robot_policy.py --lora <ckpt>            # then move
"""
import os
import sys
import argparse

os.environ.setdefault("UNSLOTH_BLOCK_COMPILE", "0")  # see ../so101_vla/train.py; must precede `import unsloth`

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from real_env import RealSetupEnv, PHASES, parse_action, gripper_word, MAT_TOP    # noqa: E402
from real_dataset import build_prompt                                             # noqa: E402
from real_policy import VLMPolicy, model_view                                     # noqa: E402
from real_arm_client import ArmClient, load_joint_map, sim_to_real, DISABLED      # noqa: E402
from reach_task import ReachEnv, REACH_PHASES, build_reach_prompt, parse_reach_action  # noqa: E402

WINDOW = "real SO-101 VLA (what the model sees)"
IMG_W, IMG_H = 448, 336
TOP_CAM, WRIST_CAM = 2, 4     # /dev/video2, /dev/video4 on this laptop
MIN_TCP_Z = MAT_TOP + 0.015   # refuse targets that would push the gripper point into the table


def open_camera(index, width=640, height=480):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"camera {index} could not be opened (try `ls /dev/video*` and ../cam_view.py)")
    return cap


def grab(cap):
    """Latest frame as RGB 448x336 (center-cropped to 4:3 like the sim cameras)."""
    for _ in range(4):            # drop buffered frames
        cap.grab()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("camera read failed")
    frame = cv2.resize(crop_to_4_3(frame), (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def crop_to_4_3(frame):
    """Center crop to the sim cameras' 4:3 aspect (shared with calibrate_top_camera.py)."""
    h, w = frame.shape[:2]
    if w * 3 > h * 4:             # too wide: crop the sides
        nw = h * 4 // 3
        return frame[:, (w - nw) // 2:(w - nw) // 2 + nw]
    if w * 3 < h * 4:             # too tall: crop top / bottom
        nh = w * 3 // 4
        return frame[(h - nh) // 2:(h - nh) // 2 + nh]
    return frame


VIEW_FILE = os.path.join(HERE, "last_view.jpg")
_has_gui = True


def show(frames, lines):
    """Camera window; without OpenCV GUI support (headless build) the view is saved to VIEW_FILE instead."""
    global _has_gui
    img = np.hstack([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames])
    bar = np.full((22 * len(lines) + 10, img.shape[1], 3), 30, np.uint8)
    for i, (text, color) in enumerate(lines):
        cv2.putText(bar, text, (10, 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    view = np.vstack([img, bar])
    if _has_gui:
        try:
            cv2.imshow(WINDOW, view)
            cv2.waitKey(1)
            return
        except cv2.error:
            _has_gui = False
            print(f"(OpenCV has no window support here: camera view saved to {VIEW_FILE} instead)")
    cv2.imwrite(VIEW_FILE, view)


def close_windows():
    if _has_gui:
        cv2.destroyAllWindows()


def checkpoint_task(lora_dir):
    """'reach' or 'pickplace', from the meta.json of the dataset the checkpoint was trained on."""
    import json
    d = os.path.abspath(lora_dir)
    for _ in range(3):
        cfg = os.path.join(d, "config.json")
        if os.path.exists(cfg):
            with open(cfg) as f:
                dataset = json.load(f).get("args", {}).get("dataset")
            meta = os.path.join(dataset or "", "meta.json")
            if os.path.exists(meta):
                with open(meta) as f:
                    return json.load(f).get("task", "pickplace")
            if dataset:   # dataset deleted since: reach datasets are named so101_reach_*
                guess = "reach" if "reach" in os.path.basename(dataset) else "pickplace"
                print(f"(dataset {os.path.basename(dataset)} is gone: task taken from its name -> {guess})")
                return guess
        d = os.path.dirname(d)
    raise SystemExit(f"can't tell which task {lora_dir} was trained for: pass --task reach / pickplace")


def make_task(name):
    """env, phases, prompt builder and answer -> (5 sim joints, note about the unused motors)."""
    if name == "reach":
        env = ReachEnv(randomize=False)

        def parse(text):
            q4 = parse_reach_action(text)
            return None if q4 is None else (env.full_q(q4), "wrist_roll / gripper fixed (reach task)")
        return env, REACH_PHASES, build_reach_prompt, parse
    env = RealSetupEnv(randomize=False)

    def parse(text):
        r = parse_action(text)
        if r is None:
            return None
        q, grip = r
        return q, f"ignored: wrist_roll {np.degrees(q[4]):.1f} deg, gripper {gripper_word(grip)}"
    return env, PHASES, build_prompt, parse


def fmt(joints):
    return "  ".join(f"{m} {v:6.1f}" for m, v in joints.items())


def ask(auto):
    if auto:
        return "y"
    a = input("  ENTER = move   s = skip   q = quit : ").strip().lower()
    return {"": "y"}.get(a, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True, help="LoRA adapter dir (checkpoints/<run>/lora or trainer/checkpoint-N)")
    ap.add_argument("--top-cam", type=int, default=TOP_CAM, help=f"real top camera index (default {TOP_CAM})")
    ap.add_argument("--wrist-cam", type=int, default=WRIST_CAM, help=f"real wrist camera index (default {WRIST_CAM})")
    ap.add_argument("--no-arm", action="store_true", help="cameras + model only, nothing moves")
    ap.add_argument("--speed", type=float, default=15.0, help="joint speed, deg/s (server caps at 60)")
    ap.add_argument("--auto", action="store_true", help="don't wait for ENTER before each move")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--task", choices=["reach", "pickplace"], default=None,
                    help="default: the task the checkpoint was trained for (from its dataset's meta.json)")
    args = ap.parse_args()

    task = args.task or checkpoint_task(args.lora)
    print(f"task: {task}")
    joint_map = load_joint_map()
    env, phases, prompt_for, parse = make_task(task)   # env: kinematics only (safety check of predicted poses)
    caps = [open_camera(args.top_cam), open_camera(args.wrist_cam)]
    policy = VLMPolicy(args.lora, args.load_in_4bit)
    arm = None if args.no_arm else ArmClient()
    print(f"not driven on the real arm: {sorted(DISABLED)} (predicted, printed, not sent)")

    try:
        if arm is not None:
            print("start pose:", fmt(arm.read()["joints"]))
            # Training episodes start with the arm parked off to the side (real_env.PARK_PAN), so the top camera
            # sees the whole mat: get the real arm there too, via a safe pose above the mat.
            for name, q in (("safe pose above the mat", env.q_seed), ("parked start pose (as in training)", env.q_home)):
                target = sim_to_real(q, joint_map)
                print(f"next: {name} -> {fmt(target)}")
                if ask(args.auto) == "y":
                    r = arm.move(target, args.speed)
                    print("  reached:", fmt(r["joints"]), f"(max err {r['max_err_deg']:.1f} deg)")

        n = len(phases)
        for k, phase in enumerate(phases):
            frames = model_view([grab(c) for c in caps])
            show(frames, [(f"Phase {k + 1}/{n}: {phase}  - thinking ...", (0, 255, 255))])
            text = policy([Image.fromarray(f) for f in frames], prompt_for(k), None)
            parsed = parse(text)
            print(f"\nPhase {k + 1}/{n} {phase}: model says {text!r}")
            if parsed is None:
                print("  unparseable answer -> stopping")
                break
            q, note_unused = parsed
            tcp = env.forward_tcp(q)
            target = sim_to_real(q, joint_map)
            print(f"  target (real deg): {fmt(target)}")
            print(f"  {note_unused}")
            print(f"  predicted gripper point (robot frame): x {tcp[0]:.3f}  y {tcp[1]:.3f}  z {tcp[2]:.3f} m")
            show(frames, [(f"Phase {k + 1}/{n}: {phase}", (0, 255, 255)), (f"model: {text}", (255, 255, 255)),
                          (f"real target: {fmt(target)}", (200, 200, 200)),
                          (f"gripper point z = {tcp[2] * 100:.1f} cm above table", (0, 255, 0))])
            if tcp[2] < MIN_TCP_Z:
                print(f"  REFUSED: gripper point would be {tcp[2] * 100:.1f} cm above the table (< "
                      f"{MIN_TCP_Z * 100:.1f} cm)")
                continue
            if arm is None:
                continue
            a = ask(args.auto)
            if a == "q":
                break
            if a != "y":
                print("  skipped")
                continue
            r = arm.move(target, args.speed)
            note = f"  clamped to limits: {r['clamped']}" if r["clamped"] else ""
            note += f"  LOCKED (recalibrate): {r['locked']}" if r.get("locked") else ""
            print(f"  reached: {fmt(r['joints'])}  (max err {r['max_err_deg']:.1f} deg){note}")
            if r.get("stopped"):
                print("  stopped with Ctrl+C on the server -> ending the run")
                break
    finally:
        if arm is not None:
            if input("\nReturn the arm to its start pose? [Y/n] ").strip().lower() != "n":
                arm.home()
            arm.close()
        for c in caps:
            c.release()
        close_windows()


if __name__ == "__main__":
    main()
