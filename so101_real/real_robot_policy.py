"""Run the trained VLA on the REAL SO-101: real top + wrist cameras -> model -> joint angles -> real arm.

Same loop as real_policy.py (one model answer per phase, smooth joint moves in between), but the images come
from real cameras and the moves go to arm_server.py. wrist_roll is taped: never sent. The gripper is sent by
the pick task (open / closed -> REAL_GRIPPER_OPEN / REAL_GRIPPER_CLOSED); reach and the old pickplace models
leave it where it is.

Safety: --no-arm runs cameras + model only; by default every move waits for ENTER; slow moves (15 deg/s);
targets whose gripper point would be below the table are refused; the first move goes to a safe pose above
the mat; arm_server.py clamps to the calibrated limits and stops on Ctrl+C.

Run in .venv, with arm_server.py running in .venv-lerobot (not needed with --no-arm):

    python so101_real/real_robot_policy.py --lora <ckpt> --no-arm   # look first (cameras 2 / 4)
    python so101_real/real_robot_policy.py --lora <ckpt>            # then move
    python so101_real/real_robot_policy.py --lora <ckpt> --manual   # you pick each phase with one key press
    python so101_real/real_robot_policy.py --lora <ckpt> --save     # + keep every step's images / answer (runs/)
"""
import os
import sys
import json
import argparse
from datetime import datetime

os.environ.setdefault("UNSLOTH_BLOCK_COMPILE", "0")  # see ../so101_vla/train.py; must precede `import unsloth`

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from real_env import RealSetupEnv, PHASES, parse_action, gripper_word, MAT_TOP    # noqa: E402
from real_dataset import build_prompt                                             # noqa: E402
from real_policy import VLMPolicy, model_view                                     # noqa: E402
from real_arm_client import ArmClient, load_joint_map, sim_to_real, real_to_sim, DISABLED  # noqa: E402
from reach_task import ReachEnv, REACH_PHASES, MOVE_PHASES, build_reach_prompt, parse_reach_action  # noqa: E402
from pick_task import PickEnv, PICK_PHASES, build_pick_prompt, parse_pick_action                     # noqa: E402
from pick_task import gripper_word as pick_gripper_word                                              # noqa: E402
from pick_task import format_pick_action, GRIPPER as PICK_GRIPPER, PLACE_DZ                          # noqa: E402
from real_env import GRASP_Z, HOVER_Z, LIFT_DZ                                                       # noqa: E402

# --top-only (pick): after PRE_GRASP every move is computed, straight down / up from the PRE_GRASP point, the same
# heights as the sim expert (pick_task.plan_keyframes). The model (and the wrist camera) is only used for PRE_GRASP.
TOP_ONLY_MOVES = {"GRASP": (GRASP_Z, "open"), "CLOSE": (GRASP_Z, "closed"), "LIFT": (GRASP_Z + LIFT_DZ, "closed"),
                  "PLACE": (GRASP_Z + PLACE_DZ, "closed"), "RELEASE": (GRASP_Z + PLACE_DZ, "open"),
                  "RETREAT": (HOVER_Z, "open")}


def top_only_answer(env, phase, anchor_xy, current):
    """The answer the sim expert would give for this phase with the block under anchor_xy, in the model's format."""
    z, grip = TOP_ONLY_MOVES[phase]
    q4, err, _ = env._solve(np.array([*anchor_xy, z]), current[:4])
    if err > 3e-3:
        return None
    return format_pick_action(q4, PICK_GRIPPER[grip])
from real_env import REAL_GRIPPER_OPEN, REAL_GRIPPER_CLOSED                                          # noqa: E402

WINDOW = "real SO-101 VLA (what the model sees)"
IMG_W, IMG_H = 448, 336
TOP_CAM, WRIST_CAM = 2, 4     # /dev/video2, /dev/video4 on this laptop
TOP_RES = (1920, 1080)        # top camera mode top_camera_calib.json was measured in (other modes may crop the sensor)
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
    """'reach', 'pick' or 'pickplace', from the meta.json of the dataset the checkpoint was trained on."""
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
            if dataset:   # dataset deleted since: named so101_reach_* / so101_pick_*
                name = os.path.basename(dataset)
                guess = "reach" if "reach" in name else "pick" if "_pick_" in name else "pickplace"
                print(f"(dataset {os.path.basename(dataset)} is gone: task taken from its name -> {guess})")
                return guess
        d = os.path.dirname(d)
    raise SystemExit(f"can't tell which task {lora_dir} was trained for: pass --task reach / pick / pickplace")


def make_task(name, n_refine=0):
    """env, phase names in run order, prompt builder (phase name, current sim joints), answer -> (5 sim joints,
    note, real gripper command or None) and extra real targets for the start poses. Reach: HOVER, REACH, then
    n_refine REFINE steps; pick: n_refine REFINE steps after PRE_GRASP."""
    if name == "pick":
        env = PickEnv(randomize=False)
        phases = ["PRE_GRASP"] + ["REFINE"] * n_refine + [p for p in PICK_PHASES if p not in ("PRE_GRASP", "REFINE")]

        def parse(text):
            r = parse_pick_action(text)
            if r is None:
                return None
            q4, grip = r
            word = pick_gripper_word(grip)
            real = REAL_GRIPPER_OPEN if word == "open" else REAL_GRIPPER_CLOSED
            return env.full_q(q4), f"gripper {word} -> {real}", real
        return (env, phases, lambda phase, current: build_pick_prompt(PICK_PHASES.index(phase), current), parse,
                {"gripper": REAL_GRIPPER_OPEN})
    if name == "reach":
        env = ReachEnv(randomize=False)
        phases = MOVE_PHASES + ["REFINE"] * n_refine

        def parse(text):
            q4 = parse_reach_action(text)
            return None if q4 is None else (env.full_q(q4), "wrist_roll / gripper fixed (reach task)", None)
        return env, phases, lambda phase, current: build_reach_prompt(REACH_PHASES.index(phase), current), parse, {}
    env = RealSetupEnv(randomize=False)

    def parse(text):
        r = parse_action(text)
        if r is None:
            return None
        q, grip = r
        return q, f"ignored: wrist_roll {np.degrees(q[4]):.1f} deg, gripper {gripper_word(grip)}", None
    return env, PHASES, lambda phase, current: build_prompt(PHASES.index(phase)), parse, {}


def fmt(joints):
    return "  ".join(f"{m} {v:6.1f}" for m, v in joints.items())


def ask(auto):
    if auto:
        return "y"
    a = input("  ENTER = move   s = skip   q = quit : ").strip().lower()
    return {"": "y"}.get(a, a)


def read_key():
    """One key press from the terminal, without ENTER (Ctrl+C still works). Falls back to a line of input."""
    if not sys.stdin.isatty():
        return (input() or " ")[0]
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def choose_phase(menu, suggested):
    """--manual: a phase name from menu, "done" or "quit", picked with one key."""
    keys = "  ".join(f"{i + 1} {p}" for i, p in enumerate(menu))
    print(f"\nnext step?  {keys}\n            SPACE = {suggested or '(nothing suggested)'}   d = episode done   "
          f"q = quit : ", end="", flush=True)
    while True:
        c = read_key().lower()
        if c.isdigit() and 1 <= int(c) <= len(menu):
            choice = menu[int(c) - 1]
        elif c in (" ", "\n") and suggested:
            choice = suggested
        elif c in ("d", "q"):
            choice = {"d": "done", "q": "quit"}[c]
        else:
            continue
        print(choice)
        return choice


def go_to_start(arm, env, joint_map, args, extra):
    """Up to the safe pose above the mat, then the parked pose training episodes start from (arm off to the
    side, so the top camera sees the whole mat). Lifting first keeps the arm from sweeping across the block."""
    for name, q in (("safe pose above the mat", env.q_seed), ("parked start pose (as in training)", env.q_home)):
        target = {**sim_to_real(q, joint_map), **extra}
        print(f"next: {name} -> {fmt(target)}")
        a = ask(args.auto)
        if a == "q":
            return False
        if a == "y":
            r = arm.move(target, args.speed)
            print("  reached:", fmt(r["joints"]), f"(max err {r['max_err_deg']:.1f} deg)")
            if r.get("stopped"):
                return False
    return True


class StepLog:
    """--save: the images the model saw and its answer at every step, and where the arm went, in
    runs/<dd-mm-yyyy-hh-mm-ss>/: ep<e>_step<s>_<PHASE>_top.jpg / _wrist.jpg and steps.jsonl (one line per step)."""

    def __init__(self, root):
        self.dir = os.path.join(root, datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        os.makedirs(self.dir, exist_ok=True)
        self.file = open(os.path.join(self.dir, "steps.jsonl"), "a")
        print(f"saving every step to {self.dir}")

    def images(self, ep, step, phase, frames):
        names = []
        for cam, f in zip(("top", "wrist"), frames):
            name = f"ep{ep:02d}_step{step:02d}_{phase}_{cam}.jpg"
            cv2.imwrite(os.path.join(self.dir, name), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            names.append(name)
        return names

    def write(self, rec):
        self.file.write(json.dumps(rec) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def run_episode(env, phases, prompt_for, parse, policy, caps, arm, joint_map, args, ep=0, log=None):
    """One episode: per phase, fresh images -> model -> (ENTER) -> move. Returns (status, last gripper point).
    The phases run in their fixed order, or with --manual in whatever order you pick (the fixed order is
    suggested). log (StepLog): every step is saved."""
    n, last_tcp, anchor_xy = len(phases), None, None
    menu = list(dict.fromkeys(phases))   # each phase once, in run order
    # Current sim joints (REFINE prompts): read from the arm; with --no-arm assume the last target was reached
    current = env.q_home.copy() if arm is None else real_to_sim(arm.read()["joints"], joint_map, env.q_home)
    k, step = 0, 0                       # k: position in the fixed order
    while True:
        if args.manual:
            phase = choose_phase(menu, phases[k] if k < n else None)
            if phase in ("done", "quit"):
                return phase, last_tcp
            # suggest what follows this phase in the fixed order (after its last repeat, if it was suggested)
            k = k + 1 if k < n and phase == phases[k] else max(i for i, p in enumerate(phases) if p == phase) + 1
            label = f"Step {step + 1}: {phase}"
        else:
            if k >= n:
                break
            phase = phases[k]
            label = f"Phase {k + 1}/{n}: {phase}"
            k += 1
        step += 1
        frames = model_view([grab(c) for c in caps])
        show(frames, [(f"{label}  - thinking ...", (0, 255, 255))])
        prompt = prompt_for(phase, current)
        rec = {"episode": ep, "step": step, "phase": phase, "prompt": prompt,
               "current_sim_deg": np.degrees(current[:4]).round(2).tolist(),
               "images": log.images(ep, step, phase, frames) if log else None}
        try:
            if args.top_only and phase in TOP_ONLY_MOVES:
                if anchor_xy is None:
                    print(f"\n{label}: --top-only needs PRE_GRASP first -> stopping this episode")
                    rec["result"] = "no PRE_GRASP yet"
                    return "error", last_tcp
                text = top_only_answer(env, phase, anchor_xy, current)
                if text is None:
                    print(f"\n{label}: --top-only: no IK solution below the PRE_GRASP point -> stopping")
                    rec["result"] = "ik fail"
                    return "error", last_tcp
                rec["computed"] = True
                label += " (computed below PRE_GRASP, no model)"
            else:
                text = policy([Image.fromarray(f) for f in frames], prompt, None)
            rec["answer"] = text
            parsed = parse(text)
            print(f"\n{label}: model says {text!r}")
            if parsed is None:
                print("  unparseable answer -> stopping this episode")
                rec["result"] = "unparseable"
                return "error", last_tcp
            q, note_unused, grip_real = parsed
            tcp = env.forward_tcp(q)
            target = sim_to_real(q, joint_map)
            if grip_real is not None:
                target["gripper"] = grip_real
            rec.update({"target_real_deg": {m: round(v, 2) for m, v in target.items()},
                        "predicted_tcp_m": tcp.round(4).tolist()})
            print(f"  target (real deg): {fmt(target)}")
            print(f"  {note_unused}")
            print(f"  predicted gripper point (robot frame): x {tcp[0]:.3f}  y {tcp[1]:.3f}  z {tcp[2]:.3f} m")
            show(frames, [(label, (0, 255, 255)), (f"model: {text}", (255, 255, 255)),
                          (f"real target: {fmt(target)}", (200, 200, 200)),
                          (f"gripper point z = {tcp[2] * 100:.1f} cm above table", (0, 255, 0))])
            if tcp[2] < MIN_TCP_Z:
                print(f"  REFUSED: gripper point would be {tcp[2] * 100:.1f} cm above the table (< "
                      f"{MIN_TCP_Z * 100:.1f} cm)")
                rec["result"] = "refused (too low)"
                continue
            last_tcp = tcp
            if phase in ("PRE_GRASP", "REFINE"):
                anchor_xy = tcp[:2].copy()       # --top-only: the moves after this go straight down from here
            if arm is None:
                current = q
                rec["result"] = "no arm"
                continue
            a = ask(args.auto)
            if a == "q":
                rec["result"] = "quit"
                return "quit", last_tcp
            if a != "y":
                print("  skipped")
                rec["result"] = "skipped"
                continue
            r = arm.move(target, args.speed)
            current = real_to_sim(r["joints"], joint_map, q)
            rec.update({"result": "stopped" if r.get("stopped") else "moved",
                        "reached_real_deg": {m: round(v, 2) for m, v in r["joints"].items()},
                        "max_err_deg": round(r["max_err_deg"], 2), "clamped": r["clamped"]})
            note = f"  clamped to limits: {r['clamped']}" if r["clamped"] else ""
            note += f"  LOCKED (recalibrate): {r['locked']}" if r.get("locked") else ""
            print(f"  reached: {fmt(r['joints'])}  (max err {r['max_err_deg']:.1f} deg){note}")
            if r.get("stopped"):
                print("  stopped with Ctrl+C on the server -> ending the run")
                return "stopped", last_tcp
        finally:
            if log:
                log.write(rec)
    return "done", last_tcp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True, help="LoRA adapter dir (checkpoints/<run>/lora or trainer/checkpoint-N)")
    ap.add_argument("--top-cam", type=int, default=TOP_CAM, help=f"real top camera index (default {TOP_CAM})")
    ap.add_argument("--wrist-cam", type=int, default=WRIST_CAM, help=f"real wrist camera index (default {WRIST_CAM})")
    ap.add_argument("--no-arm", action="store_true", help="cameras + model only, nothing moves")
    ap.add_argument("--speed", type=float, default=15.0, help="joint speed, deg/s (server caps at 60)")
    ap.add_argument("--auto", action="store_true", help="don't wait for ENTER before each move")
    ap.add_argument("--save", action="store_true",
                    help="save the model's images + answer at every step to so101_real/runs/<date-time>/")
    ap.add_argument("--manual", action="store_true",
                    help="pick each phase yourself with one key (1..8, SPACE = the usual next one, d = done)")
    ap.add_argument("--episodes", type=int, default=1,
                    help="episodes in one run (model stays loaded); you move the block in between")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--refine", type=int, default=2,
                    help="REFINE steps (reach: after REACH, pick: after PRE_GRASP; only for models trained with them)")
    ap.add_argument("--task", choices=["reach", "pick", "pickplace"], default=None,
                    help="default: the task the checkpoint was trained for (from its dataset's meta.json)")
    ap.add_argument("--top-only", action="store_true",
                    help="pick: the model only gives PRE_GRASP (top camera); GRASP..RETREAT are computed straight "
                         "down / up from that point (the wrist camera is not used)")
    args = ap.parse_args()

    task = args.task or checkpoint_task(args.lora)
    print(f"task: {task}")
    if args.top_only and task != "pick":
        raise SystemExit("--top-only is for the pick task")
    joint_map = load_joint_map()
    env, phases, prompt_for, parse, start_extra = make_task(task, args.refine)   # env: kinematics only (safety checks)
    caps = [open_camera(args.top_cam, *TOP_RES), open_camera(args.wrist_cam)]
    policy = VLMPolicy(args.lora, args.load_in_4bit)
    arm = None if args.no_arm else ArmClient()
    log = StepLog(os.path.join(HERE, "runs")) if args.save else None
    print(f"not driven on the real arm: {sorted(DISABLED)}" + ("" if task == "pick" else ", gripper (not used by this task)"))

    results = []
    try:
        if arm is not None:
            print("start pose:", fmt(arm.read()["joints"]))
        for ep in range(args.episodes):
            print(f"\n========== episode {ep + 1}/{args.episodes} ==========")
            if arm is not None and not go_to_start(arm, env, joint_map, args, start_extra):
                break
            if ep > 0 or args.episodes > 1:
                hint = ", square to the arm" if task == "pick" else ""
                a = input(f"Place the block (18-32 cm from the base{hint}), then ENTER = run   q = quit : ").strip().lower()
                if a == "q":
                    break
            status, last_tcp = run_episode(env, phases, prompt_for, parse, policy, caps, arm, joint_map, args,
                                           ep + 1, log)
            ok = None
            if status == "done" and args.episodes > 1:
                q_ok = ("Did it pick the block up and put it back?" if task == "pick" else
                        "Did it end where it should (gripper just above the block)?")
                a = input(f"{q_ok} [y/n/ENTER = skip] ")
                ok = {"y": True, "n": False}.get(a.strip().lower())
            results.append((status, ok, last_tcp))
            if status in ("quit", "stopped"):
                break
    finally:
        if len(results) > 1:
            print(f"\n=== {len(results)} episodes ===")
            for i, (status, ok, tcp) in enumerate(results):
                where = "" if tcp is None else f"  last gripper point x {tcp[0]:.3f} y {tcp[1]:.3f} z {tcp[2]:.3f} m"
                verdict = {True: "OK", False: "MISSED", None: "-"}[ok]
                print(f"  episode {i + 1}: {status:8s} {verdict:6s}{where}")
            judged = [ok for _, ok, _ in results if ok is not None]
            if judged:
                print(f"  you marked {sum(judged)}/{len(judged)} as OK")
        if arm is not None:
            if input("\nReturn the arm to its start pose? [Y/n] ").strip().lower() != "n":
                arm.home()
            arm.close()
        for c in caps:
            c.release()
        close_windows()
        if log:
            log.close()
            print(f"steps saved in {log.dir}")


if __name__ == "__main__":
    main()
