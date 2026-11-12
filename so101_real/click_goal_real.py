"""Click a goal in the real top camera (optionally also in the wrist camera) and move the real SO-101 there with IK.

Every click is a ray from the camera through the pixel, from the calibrated camera poses (top_camera_calib.json;
the wrist camera from wrist_camera_calib.json + the joint angles the arm reports at the moment of the click):
  * top click only     -> where the ray hits the surface you clicked on (P: block top / mat / any height)
  * top + wrist clicks -> the point closest to both rays (weighted least squares) and the known surface height;
                          with "any height" the rays alone. No depth camera needed.
  * base camera (BASE_CAM in .env, on the base, turns with shoulder_pan; calib_base_camera.py) -> one more ray, from
                          the side: it sees the height well. Any 1-3 cameras can be combined.
The arm goes to the hover point above the goal (gripper point at least HOVER_Z, so the jaws clear a block) and,
with --allow-down, down to it and the gripper can be closed / opened (a manual pick).

Drawn in the top view: RED = calibrated mat outline (if it doesn't sit on the mat edges, the camera moved: run
align_top_camera.py), BLUE = where the arm can hover, WHITE = where it can also go down (--allow-down),
YELLOW = the arm where it is now, MAGENTA = the planned pose, GREEN = the goal. In the base view YELLOW = where the
calibration puts the fixed jaw's tip (if it isn't on the real tip, run calib_base_camera.py again).

Needs: arm_server.py running (.venv-lerobot), unless --no-arm. Run in .venv:

    python so101_real/click_goal_real.py --no-arm       # dry run: click, see the 3-D point and the plan
    python so101_real/click_goal_real.py                # move to the hover point only
    python so101_real/click_goal_real.py --allow-down   # + D = down to the goal, X / O = close / open the gripper
    python so101_real/click_goal_real.py --no-base --allow-down   # without the base camera (on when BASE_CAM is in .env)
    python so101_real/click_goal_real.py --allow-down --record pick_red   # + record episodes (R start / end, B discard)
One window holds the three cameras, the motors sliders and the last messages (resizable; --windows = the old
separate windows, --ui-scale 0.8 = a smaller start size).
Camera indexes come from .env (TOP_CAM / WRIST_CAM / BASE_CAM, see cam_config.py); --top-cam etc. override them.

The "motors" window (motor_panel.py) has one slider per motor (wrist_roll excluded): drag + let go, wheel, -/+ or
type a value to move one motor; the sim checks the path stays above the table. A manual move clears the plan.

Keys (in a camera window):
    left-click  the goal (one point per camera window)  P      clicked surface: block top / mat / any height
    SPACE/ENTER plan: hover above the goal             D      plan: down to the goal (--allow-down)
    ENTER       do the planned move                    X / O  gripper close / open (or the motors window)
    C           clear clicks and plan                  H      plan: back to the parked pose
    R           (--record) start / end + save an episode   B  (--record) throw the running episode away
    A           (--suggest) do the model's answer          G  (--suggest) ask the model again
    N           (--suggest) forget the model's history
    arrows      move the gripper point 1 cm as seen in the top view (up = up in the picture), at the same height
    PgUp / PgDn move it 1 cm up / down                 S      arrow step 1 cm <-> 3 cm
    M           snap top clicks to the block / pad on / off (on at the start; --no-snap = off)
    Q / Esc     quit
After every move the arm's sag is corrected (shoulder_lift / elbow_flex stop 1-4 deg short under the arm's weight,
~1 cm at the gripper): a short second move to target + (target - reached), up to 2 rounds, at most 5 deg beyond the
target, never pushing the gripper point down (--no-correct = off). "corrected: 3.2 -> 0.4 deg" in the messages.
Every planned move waits for ENTER; an arrow step moves at once (small, and checked like a plan: reach, tilt, the
table, the base, no IK flip). Holding an arrow keeps stepping and stops when you let go. Ctrl+C in the arm_server
terminal stops a move on the spot. Arrow steps are recorded as "nudge" events (build_decisions.py skips them for
now: the decision model has no answer for them yet).

Snap (snap_top.py, M): a top click goes to the white block / green pad next to it (within 4 cm) - a box fitted to the
block's blob, the pad's middle - plus the top camera -> arm offset (top_arm_offset.json from fit_top_offset.py; the
arm needs to go ~2 cm from where the top camera sees things). Works for the decision model's clicks too (A). The
cyan box = the block it found; "no snap" (block partly hidden by the arm, nothing near) = the click as it is.

--suggest [LORA]: suggest mode (decision_suggest.py; default LORA: the decide_v1 checkpoint). Whenever the arm stops
the decision model (colab_train_decide.ipynb) is asked what to do next: its click shows as a BLUE dot in that camera,
its answer as text in the top view. A = accept (a click is placed like yours - then SPACE / D / ENTER as usual; a
gripper answer is sent like X / O), or ignore it and do your own step. Nothing moves without your ENTER / key. With
--record the episodes are recorded as always (your steps are the training data; the model's answers and accepts are
"suggest" / "suggest_accept" events). Needs all three cameras and ~5 GB of GPU memory (--suggest-4bit: less).

    python so101_real/click_goal_real.py --allow-down --suggest --record pick_white_cube_suggest

--record NAME: episodes go to recordings/<dd_mm_yyyy>_NAME/episode_NNN/ - the three cameras as clean pictures (MP4,
no overlays: the training data), the same with the overlays as shown in overlay/ (for the viewer; --no-rec-overlay
skips them), the joints + commands per frame, and every click / plan / move / gripper / motors-window input
(recorder.py). build_decisions.py turns the episodes into one training row per decision.
"""
import os
import sys
import json
import time
import argparse
import threading
import collections

import cv2
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from real_env import (TOP_CAM_CALIB, ARM_JOINTS, MAT_TOP, MAT_HALF, BLOCK_HALF, GRASP_Z, HOVER_Z,  # noqa: E402
                      REAL_GRIPPER_OPEN, REAL_GRIPPER_CLOSED, MOUNT_SCENE_POS, scene_to_robot_rot, _DIMS)
from pick_task import PickEnv, MAX_GRASP_TILT_DEG                                            # noqa: E402
from reach_task import REACH_RADIUS                                                           # noqa: E402
from real_arm_client import ArmClient, load_joint_map, sim_to_real, real_to_sim, MAP_FILE     # noqa: E402
from real_robot_policy import open_camera, crop_to_4_3, fmt, MIN_TCP_Z                       # noqa: E402
from cam_config import TOP_CAM, WRIST_CAM, BASE_CAM, TOP_RES, WRIST_RES, BASE_RES, REC_FPS   # noqa: E402  (.env)
from capture_wrist_calib import frame, check_top_alignment                                    # noqa: E402
import align_top_camera as align                                                              # noqa: E402
from motor_panel import MotorPanel, path_check, DRY_LIMITS, GRIP_SPEED                        # noqa: E402
from base_camera import BaseCam, fixed_tip, BASE_CLICK_CALIB                                  # noqa: E402
from recorder import Grabber, Recorder                                                        # noqa: E402
from snap_top import TopCam, snap as snap_click, load_arm_offset, OFFSET_FILE                  # noqa: E402

BLOCK_TOP = MAT_TOP + 2 * BLOCK_HALF[2]
SURFACES = [("block top", BLOCK_TOP), ("mat", MAT_TOP), ("any height", None)]   # what was clicked (P)
SURFACE_Z_SIGMA = 0.003          # how well a known surface height is known (m), next to the rays
HOVER_DZ = 0.05                  # hover: gripper point this far above the goal, and at least HOVER_Z
DOWN_DZ = BLOCK_TOP - GRASP_Z    # down: this far below the clicked point (a block top -> the grasp height)
MAX_HOVER_TILT_DEG = 2 * MAX_GRASP_TILT_DEG
LOW_Z = BLOCK_TOP + 0.02         # gripper point below this (jaws beside a block): straight up before moving away
LIFT_FIRST = 0.02                # ... if the move goes this far sideways
MAX_LIFT_TILT_DEG = 45.0         # the lift is only "up from here": any reasonable wrist angle will do
TOP_SHOW = 0.5                   # top window scale (1440x1080 -> 720x540)
TOP_WIN, WRIST_WIN = "top camera (click the goal)", "wrist camera (optional 2nd click)"
BASE_WIN = "base camera (optional click)"
BASE_SHOW = 0.5                  # base window scale (1440x1080 -> 720x540)
DASH_WIN = "SO-101 click tool"   # the one window (cameras + motors + messages); --windows = the old separate windows
TILE_W, TILE_H = 600, 450        # one camera in it (4:3); the window can be resized, clicks are mapped back
DASH_BG, DASH_FG, DASH_DIM = (26, 24, 22), (225, 228, 232), (150, 156, 165)
DASH_HELP = ["click the goal in the top camera (wrist / base = optional 2nd ray)",
             "SPACE plan hover   D plan down   ENTER move   H park   C clear   P surface",
             "X / O gripper close / open   R record start / stop   B discard episode   Q quit",
             "arrows move the gripper as seen in the top view, PgUp / PgDn up / down   S step 1 / 3 cm",
             "M snap top clicks to the block / pad on / off (cyan box = the block it found)"]
# Arrow keys (cv2.waitKeyEx codes, X11 keysyms - Qt and GTK backends): (name, (image right, image up, height))
NUDGE_KEYS = {0xFF51: ("left", (-1, 0, 0)), 0xFF53: ("right", (1, 0, 0)), 0xFF52: ("up", (0, 1, 0)),
              0xFF54: ("down", (0, -1, 0)), 0xFF55: ("raise", (0, 0, 1)), 0xFF56: ("lower", (0, 0, -1))}
NUDGE_STEPS = (0.01, 0.03)       # m per key press (S switches)
NUDGE_MAX_JOINT_DEG = 15.0       # a step needing a bigger joint change = an IK jump: refused
CORRECT_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")   # sag correction (not the gripper)
CORRECT_TOL_DEG = 0.7            # a motor further off its target than this gets a correction move
CORRECT_MAX_DEG = 5.0            # command at most this far beyond the target
CORRECT_ROUNDS = 2
SUGGEST_LORA = os.path.join(HERE, "checkpoints", "qwen35_2b_so101_decide_v1_colab", "lora")
SUGGEST_HELP = "A accept the model's answer (blue)   G ask the model again   N forget its history"
SUGGEST_COLOR = (255, 155, 45)   # BGR: the model's point / answer (blue, as on the eval page)
SUGGEST_SETTLE = 0.4             # s after the arm stopped before the pictures for the model are taken


class LogTee:
    """stdout that also keeps the last lines printed - the window shows them (goal, plan, refused, ...)."""

    def __init__(self, stream, n=300):
        self.stream, self.lines, self.part = stream, collections.deque(maxlen=n), ""

    def write(self, s):
        self.stream.write(s)
        self.part += s.replace("\r", "\n")
        *done, self.part = self.part.split("\n")
        self.lines.extend(done)
        return len(s)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, k):
        return getattr(self.stream, k)
LINKS = ["shoulder", "upper_arm", "lower_arm", "wrist", "gripper"]
REACH_MAP = os.path.join(HERE, "click_reach_map.json")   # cache of reach_map()
WRIST_CLICK_CALIB = os.path.join(HERE, "wrist_click_calib.json")   # fit_wrist_points.py: wrist camera for clicks
CLICK_PX = 1.5                   # how precisely a click lands (window pixels)
WRIST_SIGMA_SILHOUETTE = np.radians(3.0)   # wrist ray accuracy with only the silhouette fit (wrist_camera_calib.json)
MAP_AZ = np.radians(np.arange(-110, 111, 5))              # reachable zone: azimuths ...
MAP_R = np.arange(0.17, 0.425, 0.01)                      # ... and radii (m) checked


def on_mat(xy):
    """Robot-frame xy on the mat, up to its edges."""
    s = scene_to_robot_rot()[:2, :2].T @ xy + MOUNT_SCENE_POS[:2]
    return bool(np.all(np.abs(s - np.asarray(_DIMS["mat_center"])[:2]) <= MAT_HALF))


# ── Camera geometry ──────────────────────────────────────────────────────────
def cam_ray(model, data, cam, u, v, w, h):
    """World ray (origin, unit direction) through pixel (u, v) of a w x h image of MuJoCo camera cam
    (looks down its -z, x = image right, y = image up, fovy vertical)."""
    cid = model.camera(cam).id
    f = 0.5 * h / np.tan(np.radians(model.cam_fovy[cid]) / 2)
    d = data.cam_xmat[cid].reshape(3, 3) @ np.array([(u + 0.5 - w / 2) / f, -(v + 0.5 - h / 2) / f, -1.0])
    return data.cam_xpos[cid].copy(), d / np.linalg.norm(d)


def project(model, data, cam, p, w, h):
    """Pixel (u, v) of world point p, or None if it is behind the camera."""
    cid = model.camera(cam).id
    f = 0.5 * h / np.tan(np.radians(model.cam_fovy[cid]) / 2)
    pc = data.cam_xmat[cid].reshape(3, 3).T @ (np.asarray(p) - data.cam_xpos[cid])
    if -pc[2] < 1e-2:
        return None
    u, v = w / 2 + f * pc[0] / -pc[2] - 0.5, h / 2 - f * pc[1] / -pc[2] - 0.5
    return int(np.clip(u, -1e4, 1e4)), int(np.clip(v, -1e4, 1e4))


def triangulate(rays, plane_z, sigmas=None, z_sigma=SURFACE_Z_SIGMA):
    """Point closest to all rays [(o, d), ...] in the (weighted) least-squares sense:
        min_x  sum w_i |(I - d_i d_i^T)(x - o_i)|^2  [+ (x_z - plane_z)^2 / z_sigma^2]
    sigmas: angular accuracy of each ray (rad); then w_i = 1 / (sigma_i * distance_i)^2, so the more accurate
    ray counts more, and the inverse of the normal matrix is the point's covariance.
    plane_z (None = unknown): the height of the clicked surface, used as one more measurement (z_sigma), since
    near-vertical rays fix x / y well but the height poorly. A single ray is intersected with that plane.
    Returns (point, distance of the point to each ray, covariance (3x3, m^2) or None) or (None, None, None)."""
    if len(rays) == 1:
        o, d = rays[0]
        t = (plane_z - o[2]) / d[2] if plane_z is not None and abs(d[2]) > 1e-6 else -1.0
        return (o + t * d, np.zeros(1), None) if t > 0 else (None, None, None)

    def normal_eqs(w):
        A, b = np.zeros((3, 3)), np.zeros(3)
        for (o, d), wi in zip(rays, w):
            P = np.eye(3) - np.outer(d, d)
            A += wi * P
            b += wi * P @ o
        if plane_z is not None:
            A[2, 2] += 1 / z_sigma ** 2
            b[2] += plane_z / z_sigma ** 2
        return A, b

    s = np.full(len(rays), 0.01)              # 1 cm per ray until the distances are known
    A, b = normal_eqs(1 / s ** 2)
    if np.linalg.cond(A) > 1e8:               # (nearly) parallel rays and no height
        return None, None, None
    x, cov = np.linalg.solve(A, b), None
    if sigmas is not None:
        s = np.array([sig * np.linalg.norm(x - o) for (o, _), sig in zip(rays, sigmas)])   # ray accuracy at x (m)
        A, b = normal_eqs(1 / s ** 2)
        x, cov = np.linalg.solve(A, b), np.linalg.inv(A)
    res = np.array([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (x - o)) for o, d in rays])
    return x, res, cov


class Kin:
    """The as-built sim arm (PickEnv's model) on its own MjData: camera poses and link points for 5 sim joints."""

    def __init__(self, env):
        self.m = env.model
        self.d = mujoco.MjData(self.m)
        self.qadr = [self.m.jnt_qposadr[self.m.joint(j).id] for j in ARM_JOINTS]
        self.links = [self.m.body(b).id for b in LINKS]
        self.tcp = self.m.site("tcp").id

    def set(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)    # camera poses
        return self

    def tcp_pos(self):
        return self.d.site_xpos[self.tcp].copy()

    def points(self):
        """Base, the links and the gripper point: the arm as a polyline."""
        return [np.zeros(3)] + [self.d.xpos[b].copy() for b in self.links] + [self.tcp_pos()]


def draw_arm(img, kin, q, color, scale_w, scale_h):
    pts = [project(kin.m, kin.set(q).d, "top_camera", p, scale_w, scale_h) for p in kin.points()]
    pts = [p for p in pts if p is not None]
    if len(pts) >= 2:
        cv2.polylines(img, [np.int32(pts)], False, color, 4, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 7, color, -1)


def plan_nudge(env, kin, q_now, d, step):
    """Arrow key: the gripper point `step` m from where it is now, d = (image right, image up, height) in the top
    camera's view (right / up flattened onto the table, so a move is purely sideways or purely up / down).
    Returns (q, target, None) or (None, target, why it is refused) - the same checks as a planned move."""
    kin.set(q_now)
    R = kin.d.cam_xmat[kin.m.camera("top_camera").id].reshape(3, 3)
    right, up = R[:, 0].copy(), R[:, 1].copy()          # camera x = image right, y = image up
    right[2] = up[2] = 0.0
    right, up = right / np.linalg.norm(right), up / np.linalg.norm(up)
    tcp = kin.tcp_pos()
    target = tcp + step * (d[0] * right + d[1] * up) + np.array([0.0, 0.0, step * d[2]])
    if d[2] < 0:
        if tcp[2] <= MIN_TCP_Z + 0.001:
            return None, target, f"already at the lowest allowed height ({MIN_TCP_Z * 100:.1f} cm)"
        target[2] = max(target[2], MIN_TCP_Z)
    r = np.linalg.norm(target[:2])
    if r < REACH_RADIUS[0] - 0.01 and r < np.linalg.norm(tcp[:2]):
        return None, target, f"{r * 100:.1f} cm from the base (the jaws would hit it)"
    max_tilt = MAX_GRASP_TILT_DEG if target[2] < LOW_Z else MAX_HOVER_TILT_DEG
    q4, e, tilt = env._solve(target, q_now[:4])
    if e > 2e-3 or tilt > max_tilt:
        return None, target, f"not reachable (err {e * 1000:.1f} mm, tilt {tilt:.0f} deg, max {max_tilt:.0f})"
    q = env.full_q(q4)
    jump = np.degrees(np.max(np.abs(q[:4] - q_now[:4])))
    if jump > NUDGE_MAX_JOINT_DEG:
        return None, target, f"the arm would flip ({jump:.0f} deg joint change for a small step)"
    err, _ = path_check(kin, q_now, q, min_tcp_z=MIN_TCP_Z)
    if err:
        return None, target, err
    return q, target, None


def reach_map(env):
    """Where the plan checks pass, per (azimuth, radius) of MAP_AZ x MAP_R: 2 = hover and down, 1 = hover only,
    0 = neither. Takes ~30 s, so it is cached in REACH_MAP (for this wrist_roll and these joint limits)."""
    key = {"roll": round(float(env.roll), 5), "lo": np.round(env.ik_tcp.lo, 5).tolist(),
           "hi": np.round(env.ik_tcp.hi, 5).tolist(), "az": np.round(MAP_AZ, 5).tolist(),
           "r": np.round(MAP_R, 5).tolist()}
    try:
        with open(REACH_MAP) as f:
            c = json.load(f)
        if c["key"] == key:
            return np.array(c["ok"])
    except (OSError, ValueError, KeyError):
        pass
    print("computing the reachable zone (only the first time, ~30 s) ...", flush=True)
    ok = np.zeros((len(MAP_AZ), len(MAP_R)), int)
    for i, a in enumerate(MAP_AZ):
        q = env.q_seed4.copy()
        for j, r in enumerate(MAP_R):
            xy = r * np.array([np.cos(a), np.sin(a)])
            q, e, tilt = env._solve(np.array([*xy, HOVER_Z]), q)
            if e < 2e-3 and tilt <= MAX_HOVER_TILT_DEG:
                _, e2, tilt2 = env._solve(np.array([*xy, GRASP_Z]), q)
                ok[i, j] = 2 if e2 < 2e-3 and tilt2 <= MAX_GRASP_TILT_DEG else 1
    with open(REACH_MAP, "w") as f:
        json.dump({"key": key, "ok": ok.tolist()}, f)
    return ok


def reach_outlines(ok, kin, w, h, level):
    """Outlines (top image pixels, at the mat height) of the zone where reach_map() >= level."""
    mask = np.zeros((h, w), np.uint8)
    da, dr = (MAP_AZ[1] - MAP_AZ[0]) / 2, (MAP_R[1] - MAP_R[0]) / 2
    for i, a in enumerate(MAP_AZ):
        for j, r in enumerate(MAP_R):
            if ok[i, j] < level:
                continue
            cell = [project(kin.m, kin.d, "top_camera", [rr * np.cos(aa), rr * np.sin(aa), MAT_TOP], w, h)
                    for aa, rr in ((a - da, r - dr), (a + da, r - dr), (a + da, r + dr), (a - da, r + dr))]
            if all(p is not None for p in cell):
                cv2.fillConvexPoly(mask, np.int32(cell), 1)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))     # close the rounding gaps between cells
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-cam", type=int, default=TOP_CAM, help=f"top camera index (TOP_CAM in .env: {TOP_CAM})")
    ap.add_argument("--wrist-cam", type=int, default=WRIST_CAM, help=f"wrist camera index (WRIST_CAM in .env: {WRIST_CAM})")
    ap.add_argument("--no-wrist", action="store_true", help="no wrist camera")
    ap.add_argument("--base-cam", type=int, default=BASE_CAM,
                    help=f"base camera index (BASE_CAM in .env: {BASE_CAM}; used whenever it is set)")
    ap.add_argument("--no-base", action="store_true", help="don't use the base camera this time")
    ap.add_argument("--no-arm", action="store_true", help="dry run: no arm_server, nothing moves")
    ap.add_argument("--allow-down", action="store_true", help="allow D (down to the goal) and the gripper keys")
    ap.add_argument("--speed", type=float, default=15.0, help="joint speed, deg/s")
    ap.add_argument("--record", nargs="?", const="click", default=None, metavar="NAME",
                    help="record episodes (R = start / end, B = discard) to recordings/<dd_mm_yyyy>_NAME/")
    ap.add_argument("--windows", action="store_true", help="separate windows per camera + motors (the old layout)")
    ap.add_argument("--ui-scale", type=float, default=1.0, help="start size of the one window (1 = 1800 px wide)")
    ap.add_argument("--rec-fps", type=float, default=REC_FPS, help=f"recording frame rate (REC_FPS in .env: {REC_FPS:g})")
    ap.add_argument("--no-rec-overlay", action="store_true",
                    help="record only the clean camera pictures (no overlay/ copies for the viewer: half the disk)")
    ap.add_argument("--suggest", nargs="?", const=SUGGEST_LORA, default=None, metavar="LORA",
                    help="suggest mode: the decision model proposes the next step (blue), A = accept it "
                         f"(default LORA: {os.path.relpath(SUGGEST_LORA)})")
    ap.add_argument("--suggest-4bit", action="store_true", help="load the decision model in 4 bit (less GPU memory)")
    ap.add_argument("--no-snap", action="store_true", help="start with snapping off (M switches it)")
    ap.add_argument("--no-correct", action="store_true",
                    help="no sag correction after a move (the arm stops 1-4 deg short under its weight)")
    args = ap.parse_args()
    tee = None
    if not args.windows:
        tee = sys.stdout = LogTee(sys.stdout)

    with open(TOP_CAM_CALIB) as f:
        calib = json.load(f)
    top_cap = open_camera(args.top_cam, *TOP_RES)
    wrist_cap = None if args.no_wrist else open_camera(args.wrist_cam, *WRIST_RES)
    if args.no_base:
        args.base_cam = None
    base_cap = None if args.base_cam is None else open_camera(args.base_cam, *BASE_RES)
    print(f"cameras: top {args.top_cam} ({TOP_RES[0]}x{TOP_RES[1]}), wrist "
          f"{'off' if args.no_wrist else f'{args.wrist_cam} ({WRIST_RES[0]}x{WRIST_RES[1]})'}, base "
          f"{'off' if args.base_cam is None else f'{args.base_cam} ({BASE_RES[0]}x{BASE_RES[1]})'}  (defaults from .env)")
    sugg = None                          # --suggest: the decision model (decision_suggest.py)
    if args.suggest is not None:
        if wrist_cap is None or base_cap is None:
            raise SystemExit("--suggest needs all three cameras (the model was trained on top + wrist + base)")
        if not os.path.isdir(args.suggest):
            raise SystemExit(f"--suggest: {args.suggest} not found (unzip the LoRA into so101_real/checkpoints/)")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")       # the base model is in the local cache
        from decision_suggest import Suggester
        sugg = Suggester(args.suggest, args.suggest_4bit)
    env = PickEnv(randomize=False)       # wrist_roll at the taped angle, IK inside the real limits
    kin = Kin(env)
    # Ray accuracy (rad), for weighting the rays: calibration error + click error, over the focal length
    f_top = 0.5 * calib["image_size"][1] / np.tan(np.radians(calib["fovy"]) / 2)
    sigma = {"top": np.hypot(calib.get("rms_px", 2.0), CLICK_PX / TOP_SHOW) / f_top,
             "wrist": WRIST_SIGMA_SILHOUETTE}
    if os.path.exists(WRIST_CLICK_CALIB):
        with open(WRIST_CLICK_CALIB) as f:
            wc = json.load(f)
        cid = env.model.camera("wrist_camera").id
        env.model.cam_pos[cid], env.model.cam_quat[cid], env.model.cam_fovy[cid] = wc["pos"], wc["quat"], wc["fovy"]
        f_wrist = 0.5 * 480 / np.tan(np.radians(wc["fovy"]) / 2)
        sigma["wrist"] = np.hypot(wc["rms_px"], CLICK_PX) / f_wrist
        print(f"wrist camera: point calibration ({wc['rms_px']:.1f} px, new block positions {wc['cv_ray_mm']:.1f} mm)")
        if abs(wc["wrist_roll_deg"] - np.degrees(env.roll)) > 0.05:
            print(f"WARNING: wrist_click_calib.json was fitted with wrist_roll {wc['wrist_roll_deg']} deg, the model "
                  f"now has {np.degrees(env.roll):.2f} - run fit_wrist_points.py again")
    elif not args.no_wrist:
        print("wrist camera: silhouette calibration only - 2-camera clicks will be rough; run fit_wrist_points.py")
    base = None
    if base_cap is not None:
        base = BaseCam.load()
        if base is None:
            print("base camera: NOT calibrated - its clicks are ignored; run calib_base_camera.py first")
        else:
            sigma["base"] = np.hypot(base.rms_px, CLICK_PX / BASE_SHOW) / base.f
            print(f"base camera: calibrated ({base.rms_px:.1f} px), ray accuracy {np.degrees(sigma['base']):.2f} deg")
    print(f"ray accuracy: top {np.degrees(sigma['top']):.2f} deg, wrist {np.degrees(sigma['wrist']):.2f} deg")
    jm = load_joint_map()
    arm = None if args.no_arm else ArmClient()
    grip_lo, grip_hi = (DRY_LIMITS if arm is None else arm.read()["limits"])["gripper"]   # GRIPPER_MIN / MAX (.env)
    grip = float(np.clip(REAL_GRIPPER_OPEN, grip_lo, grip_hi))   # commanded gripper, sent with every move
    held = {} if arm is None else dict(arm.read().get("held", {}))   # arm_server HOLDs (motors window HOLD button)
    if "gripper" in held:
        grip = held["gripper"]
        print(f"gripper is HELD at {grip:.0f} (arm_server): it won't move - press HELD in the motors window to release")
    sim_q = env.q_home.copy()            # --no-arm: where the arm is assumed to be

    top = crop_to_4_3(frame(top_cap))
    TH, TW = top.shape[:2]
    WH, WW = frame(wrist_cap).shape[:2] if wrist_cap is not None else (480, 640)   # fitted at 640x480 (any 4:3)
    ok = reach_map(env)
    kin.set(env.q_home)
    zone_hover, zone_down = (reach_outlines(ok, kin, TW, TH, level) for level in (1, 2))
    if base is not None:
        BH, BW = crop_to_4_3(frame(base_cap)).shape[:2]
        if [BW, BH] != [base.w, base.h]:
            print(f"WARNING: base image {BW}x{BH}, calibrated at {base.w}x{base.h} - base clicks are ignored")
            base = None
    if [TW, TH] != calib["image_size"]:
        print(f"WARNING: top image {TW}x{TH}, calibrated at {calib['image_size']} - the clicks will be off")
    err = check_top_alignment(top, calib)
    if err is None or err > 12:
        print(f"WARNING: top camera alignment: {'mat edges not found' if err is None else f'{err:.0f} px off'}"
              " - run align_top_camera.py first, or the goals will be off")
    else:
        print(f"top camera aligned ({err:.0f} px)")

    def real_now():
        """{motor: real value} where the arm is now (--no-arm: the assumed pose + the commanded gripper)."""
        if arm is None:
            return {**sim_to_real(sim_q, jm), "gripper": grip}
        return arm.read()["joints"]

    def current_q():
        return real_to_sim(real_now(), jm, env.q_home)              # wrist_roll (not driven) from q_home

    cmd = real_now()                     # what each motor was last told (real units) - the "action" of a recording
    rec = None                           # Recorder (--record), made once the windows are up
    kin_rec = Kin(env)                   # its own MjData: the move-time recorder thread draws with it

    def log(kind, **data):
        """One line in the running episode's events.jsonl (nothing if not recording); --suggest: the model's
        history is rebuilt from the same events."""
        if sugg is not None:
            sugg.history.event(kind, data)
        if rec is not None:
            rec.event(kind, **data)

    def ask_model():
        """--suggest: ask the model once the arm has settled (the main loop takes the pictures)."""
        if sugg is not None:
            st["ask_at"] = time.time() + SUGGEST_SETTLE

    def record_during_move(target, speed, stop_evt):
        """Thread: while arm.move() blocks, keep recording - newest camera frames, the arm drawn where it should be
        by now (the same smoothstep as arm_server); the real trace replaces state / action afterwards."""
        start = dict(cmd)
        goal = {**start, **{m: float(v) for m, v in target.items() if m in start and m not in held}}
        dur = max(max((abs(goal[m] - start[m]) for m in goal), default=0.0) / min(max(float(speed), 1.0), 60.0), 0.3)
        t0 = time.time()
        while not stop_evt.is_set():
            a = min((time.time() - t0) / dur, 1.0)
            a = a * a * (3 - 2 * a)
            est = {m: start[m] + a * (goal[m] - start[m]) for m in goal}
            try:
                frames = grab_frames()
                rec.add(rec_views(frames, render(real_to_sim(est, jm, env.q_home), kin_rec, moving=True,
                                                 frames=frames)), state=est, action=est, moving=True)
            except Exception as ex:          # never let the recorder break a move
                print(f"  (recording during the move: {ex!r})")
                return
            stop_evt.wait(0.5 / rec.fps)

    def do_move(target, speed, label):
        """arm.move (blocking) + recording: frames go on during the move, then the joint trace and the result."""
        recording = rec is not None and rec.active
        if recording:
            log("move_start", label=label, target=target, speed=speed)
            stop_evt = threading.Event()
            th = threading.Thread(target=record_during_move, args=(target, speed, stop_evt), daemon=True)
            th.start()
        try:
            r = arm.move(target, speed, trace=True) if recording else arm.move(target, speed)
        finally:
            if recording:
                stop_evt.set()
                th.join(timeout=2.0)
        cmd.update(r.get("goal", {}))
        if recording:
            rec.fill_trace(r.get("trace"))
            log("move_done", label=label, ok=r.get("ok"), stopped=r.get("stopped"), joints=r.get("joints"),
                goal=r.get("goal"), clamped=r.get("clamped"), held_refused=r.get("held_refused"),
                max_err_deg=r.get("max_err_deg"))
        return r

    def send(name, q, pos=None):
        """One move (blocking). Returns False if it was stopped."""
        nonlocal sim_q
        target = sim_to_real(q, jm)
        if "gripper" not in held:            # HELD: leave it out (arm_server would keep it anyway)
            target["gripper"] = float(np.clip(grip, grip_lo, grip_hi))
        where = "" if pos is None else f"gripper point {np.round(pos, 3)} "
        print(f"  {name}: {where}-> {fmt(target)}")
        if arm is None:
            sim_q = np.array(q, float)
            cmd.update(target)
            log("move_done", label=name, ok=True, goal=target, dry_run=True)
            return True
        r = do_move(target, args.speed, name)
        print(f"    reached (max err {r['max_err_deg']:.1f} deg){'  CLAMPED: ' + str(r['clamped']) if r['clamped'] else ''}")
        if r.get("stopped"):
            print("    STOPPED (Ctrl+C on the server) - plan cleared")
            return False
        if not args.no_correct:
            return correct(name, target, r)
        return True

    def tcp_of(real):
        return kin.set(real_to_sim(real, jm, env.q_home)).tcp_pos()

    def correct(name, target, r):
        """The arm sags: shoulder_lift / elbow_flex stop 1-4 deg short under the arm's weight (~1 cm at the gripper).
        Command target + (target - reached) - the motor then settles on the target - up to CORRECT_ROUNDS times.
        Only the 4 arm motors (never the gripper: it stops on a block on purpose; never wrist_roll). A correction
        that would push the gripper point DOWN is not sent: sag always leaves it too low, so then something holds
        the arm up (the table, a block). Returns False if a correction move was stopped."""
        joints = [m for m in CORRECT_JOINTS if m in target and m not in held]
        aim = {m: float(target[m]) for m in joints}
        reached = r.get("joints") or {}
        if not joints or not all(m in reached for m in joints):
            return True
        first = None
        for k in range(CORRECT_ROUNDS):
            err = {m: aim[m] - float(reached[m]) for m in joints}
            worst = max(abs(e) for e in err.values())
            first = worst if first is None else first
            if worst <= CORRECT_TOL_DEG:
                break
            new = {m: float(np.clip(float(cmd.get(m, aim[m])) + err[m], aim[m] - CORRECT_MAX_DEG, aim[m] + CORRECT_MAX_DEG))
                   for m in joints}
            if tcp_of({**target, **new})[2] < tcp_of(target)[2] - 0.002:
                print(f"    correction skipped: it would push the gripper down (held up by something?) - "
                      f"off by {worst:.1f} deg")
                log("correction", label=name, skipped="push down", err=err)
                return True
            log("correction", label=name, round=k + 1, err=err, command=new)
            r = do_move({**target, **new}, args.speed, f"{name} correction")
            if r.get("stopped"):
                print("    STOPPED (Ctrl+C on the server) - plan cleared")
                return False
            reached = r.get("joints") or reached
        err = {m: aim[m] - float(reached[m]) for m in joints}
        worst = max(abs(e) for e in err.values())
        if first is not None and first > CORRECT_TOL_DEG:
            miss = np.linalg.norm(tcp_of({**target, **{m: reached[m] for m in joints}}) - tcp_of(target)) * 1000
            print(f"    corrected: {first:.1f} -> {worst:.1f} deg off the target (gripper point {miss:.0f} mm from it)")
        return True

    if arm is not None:
        input("ENTER = move to the safe pose above the mat, then the parked pose (gripper opens) ")
        if not (send("safe pose", env.q_seed) and send("parked pose", env.q_home)):
            arm.close()
            return

    st = {"clicks": {}, "goal": None, "plan": [], "surface": 0, "joints": real_now(), "t_read": time.time(),
          "ask_at": None, "big_step": False, "key_next": None, "snap": not args.no_snap, "snapped": None}
    st["q_now"] = real_to_sim(st["joints"], jm, env.q_home)
    topcam = TopCam.from_mujoco(kin.m, kin.set(env.q_home).d, TW, TH)     # snapping (snap_top.py)
    arm_offset = load_arm_offset()
    if arm_offset is None:
        print(f"WARNING: no {os.path.basename(OFFSET_FILE)} - snapped clicks go to the block itself, not where the "
              "arm needs to go (~2 cm off): run fit_top_offset.py")
        arm_offset = np.zeros(2)
    else:
        print(f"snap: on (M = off), top camera -> arm offset {np.round(arm_offset * 1000, 1)} mm")

    def update_goal():
        rays = [(o, d) for _, _, o, d in st["clicks"].values()]
        sigmas = [sigma[cam] for cam in st["clicks"]]
        if st["plan"]:
            print("  (new click: the plan was for the old goal and is cleared - SPACE / D again)")
        st["goal"], st["plan"] = None, []
        clicks = {cam: c[:2] for cam, c in st["clicks"].items()}
        if not rays:
            log("goal", goal=None, clicks=clicks)
            return
        name, z = SURFACES[st["surface"]]
        if z is None and len(rays) == 1:
            print("one click with 'any height' can't give a 3-D point: click the other camera too, or P for a "
                  "known surface")
            return
        p, res, cov = triangulate(rays, z, sigmas)
        if p is None:
            print("could not triangulate (rays parallel / ray misses the surface)")
            return
        p[2] = max(p[2], MAT_TOP)
        st["goal"] = p
        cams = list(st["clicks"])
        if len(rays) == 1:
            how = f"{cams[0]} click on the {name}"
        else:
            e_xy, e_z = np.sqrt(np.linalg.eigvalsh(cov[:2, :2]).max()), np.sqrt(cov[2, 2])
            how = (f"{' + '.join(cams)} rays{'' if z is None else ' on the ' + name}, misses {np.round(res * 1000, 1)} mm, "
                   f"expected error x/y +-{e_xy * 1000:.1f} mm, height +-{e_z * 1000:.1f} mm")
        print(f"goal {np.round(p, 3)} m ({np.linalg.norm(p[:2]) * 100:.1f} cm from the base; {how})")
        log("goal", goal=p, clicks=clicks, surface=name, how=how)

    def on_top(event, u, v, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            u, v = u / TOP_SHOW, v / TOP_SHOW                # back to calibration pixels
            extra = {}
            st["snapped"] = None
            if st["snap"]:                                   # the block / pad next to the click (snap_top.py)
                r = snap_click(grabbers["top"].get(), u, v, topcam, poly, SURFACES[st["surface"]][1], arm_offset)
                extra = {"raw_u": u, "raw_v": v, "snap": r["kind"], "snap_mm": r["dist_mm"], "snap_iou": r["iou"]}
                if r["kind"] is not None:
                    u, v = float(r["pixel"][0]), float(r["pixel"][1])
                    st["snapped"] = r
                    print(f"snapped to the {r['kind']} ({r['dist_mm']:.0f} mm from the click"
                          + (f", fit {r['iou']:.2f}" if r["iou"] is not None else "") + ")")
                else:
                    print(f"  no snap: {r['why']} - the click is used as it is")
            st["clicks"]["top"] = (int(u), int(v), *cam_ray(kin.m, kin.set(sim_q).d, "top_camera", u, v, TW, TH))
            log("click", camera="top", u=u, v=v, **extra)
            update_goal()

    def on_wrist(event, u, v, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if arm is None:
                print("wrist clicks need the arm (its joints give the wrist camera pose) - not with --no-arm")
                return
            # the camera pose from the joints the arm reports right now
            st["clicks"]["wrist"] = (u, v, *cam_ray(kin.m, kin.set(current_q()).d, "wrist_camera", u, v, WW, WH))
            log("click", camera="wrist", u=u, v=v)
            update_goal()

    def on_base(event, u, v, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if base is None:
                print("base camera not calibrated (or wrong image size): run calib_base_camera.py")
                return
            if arm is None:
                print("base clicks need the arm (shoulder_pan gives the base camera pose) - not with --no-arm")
                return
            u, v = u / BASE_SHOW, v / BASE_SHOW                   # back to calibration pixels
            st["clicks"]["base"] = (int(u), int(v), *base.ray(kin.set(current_q()), u, v))
            log("click", camera="base", u=u, v=v)
            update_goal()

    def plan(kind):
        """plan_raw + one events.jsonl line (the keyframes as real motor targets, or refused)."""
        out = plan_raw(kind)
        log("plan", plan=kind, ok=bool(out), goal=st["goal"],
            steps=[{"name": n, "gripper_point": pos, "target": sim_to_real(q, jm)} for n, q, pos in out])
        return out

    def plan_raw(kind):
        """Keyframes [(name, q, gripper point)] for kind = hover / down / park; [] (with a message) if refused."""
        q_now = current_q()
        tcp_now = kin.set(q_now).tcp_pos()
        if kind == "park":
            out = [("safe pose", env.q_seed, None), ("parked pose", env.q_home, None)]
            if tcp_now[2] < LOW_Z:
                q, e, _ = env._solve(np.array([*tcp_now[:2], HOVER_Z]), q_now[:4])
                if e > 3e-3:
                    print("can't lift straight up from here - move by hand first")
                    return []
                out.insert(0, ("lift", env.full_q(q), np.array([*tcp_now[:2], HOVER_Z])))
            return out
        p = st["goal"]
        if p is None:
            print("no goal yet: left-click it in the top camera")
            return []
        # Too far is left to the IK (error / tilt checks below): ~35 cm straight ahead still works
        r = np.linalg.norm(p[:2])
        if r < REACH_RADIUS[0] - 0.01:
            print(f"refused: {r * 100:.1f} cm from the base, closer than {(REACH_RADIUS[0] - 0.01) * 100:.0f} cm "
                  "(the jaws would hit the base)")
            return []
        if not on_mat(p[:2]):   # fine: the table is only MAT_TOP (2 mm) below the mat; reach is checked below
            print(f"  note: off the mat (red outline) - on the table, {MAT_TOP * 1000:.0f} mm lower; "
                  "make sure nothing else is in the way")
        z_hover = max(p[2] + HOVER_DZ, HOVER_Z)
        side = np.linalg.norm(tcp_now[:2] - p[:2])
        steps = []
        if tcp_now[2] < LOW_Z and side > LIFT_FIRST:
            steps.append(("lift", np.array([*tcp_now[:2], z_hover]), MAX_LIFT_TILT_DEG))
        if kind == "hover" or side > 0.01 or tcp_now[2] > z_hover + 0.01:
            steps.append(("hover", np.array([*p[:2], z_hover]), MAX_HOVER_TILT_DEG))
        if kind == "down":
            steps.append(("down", np.array([*p[:2], max(p[2] - DOWN_DZ, MIN_TCP_Z)]), MAX_GRASP_TILT_DEG))
        out, q = [], q_now[:4]
        for name, pos, max_tilt in steps:
            q, e, tilt = env._solve(pos, q)
            if e > 2e-3 or tilt > max_tilt:
                print(f"refused: {name} {np.round(pos, 3)} not reachable (err {e * 1000:.1f} mm, tilt {tilt:.0f} deg)")
                return []
            out.append((name, env.full_q(q), pos))
            print(f"  plan {name}: gripper point {np.round(pos, 3)}  tilt {tilt:.0f} deg")
        if kind == "hover" and np.linalg.norm(out[-1][2] - tcp_now) < 0.01:
            print(f"  already there ({np.linalg.norm(out[-1][2] - tcp_now) * 1000:.0f} mm): hover stays "
                  f"{HOVER_DZ * 100:.0f} cm above the goal - D = down to it" + ("" if args.allow_down else
                                                                                 " (needs --allow-down)"))
        return out

    def panel_check(target):
        """Motors window: is the straight joint move from here to target safe (sim)? + where the gripper point goes."""
        q_from = st["q_now"]
        err, tcp = path_check(kin, q_from, real_to_sim(target, jm, q_from))   # no fingertip floor
        return err, ("" if tcp is None else
                     f"gripper point there: x {tcp[0] * 100:.1f}  y {tcp[1] * 100:.1f}  z {tcp[2] * 100:.1f} cm")

    grip_src = {"v": "motors window"}   # who asked for the next gripper move (for the gripper_open / _close event)

    def panel_move(target, speed):
        """Motors window (and X / O, episode start - they go through its gripper slider): move (blocking). Clears the
        plan / wrist click (made for the old pose). A gripper change is logged as gripper_open / gripper_close."""
        nonlocal grip, sim_q
        g_before = float(cmd.get("gripper", grip))
        grip = float(target.get("gripper", grip))   # later moves keep sending this (a held block stays squeezed)
        if st["plan"]:
            print("  manual move: plan cleared")
        st["plan"] = []
        st["clicks"].pop("wrist", None)
        st["clicks"].pop("base", None)
        if arm is None:
            sim_q = real_to_sim(target, jm, sim_q)   # target = only the moved motors
            cmd.update(target)
            reached = real_now()
            log("move_done", label="motors window", ok=True, goal=target, dry_run=True)
        else:
            r = do_move(target, speed, "motors window")
            reached = r["joints"]
            if r["clamped"]:
                print(f"  clamped to the limits: {r['clamped']}")
        st["q_now"], st["t_read"], st["joints"] = real_to_sim(reached, jm, env.q_home), time.time(), dict(reached)
        if "gripper" in target and abs(float(target["gripper"]) - g_before) > 0.05:
            g_to, g_now = float(target["gripper"]), float(reached.get("gripper", target["gripper"]))
            log("gripper_open" if g_to > g_before else "gripper_close", value=g_to, before=g_before, now=g_now,
                source=grip_src["v"], blocked=abs(g_now - g_to) > 5)
        grip_src["v"] = "motors window"
        ask_model()                          # the arm stopped somewhere new
        return reached

    def set_gripper(value, source):
        """X / O keys, episode start: the gripper slider in the motors window, sent now (panel_move logs it)."""
        lo, hi = panel.limits["gripper"]
        to = float(np.clip(value, lo, hi))
        if abs(to - float(cmd.get("gripper", grip))) <= 0.05:   # already told so: the slider wouldn't send anything
            real = float((real_now() if arm is not None else {}).get("gripper", to))
            if abs(real - to) <= 3.0 or "gripper" in held:
                kind = "gripper_open" if value >= REAL_GRIPPER_OPEN or value >= hi else "gripper_close"
                log(kind, value=to, before=real, source=source, moved=False)
                print(f"gripper already at {to:.0f}")
                return
            reached = panel_move({"gripper": to}, GRIP_SPEED)   # told, but not there (pushed / was blocked): again
            panel.sync({**reached, "gripper": to})
            g_now = float(reached.get("gripper", to))
            log("gripper_open" if to > real else "gripper_close", value=to, before=real, now=g_now, source=source,
                blocked=abs(g_now - to) > 5)
            print(f"gripper -> {to:.0f} (now at {reached.get('gripper', to):.0f})")
            return
        grip_src["v"] = source
        if not panel.set_value("gripper", value):
            grip_src["v"] = "motors window"
            print(f"gripper is HELD at {held['gripper']:.0f} - not changed (press HELD in the motors window)")
            return
        now = (real_now() if arm is not None else {}).get("gripper", grip)
        print(f"gripper -> {grip:.0f} (now at {now:.0f}{', blocked: holding something?' if abs(now - grip) > 5 else ''})")

    if args.windows:
        cv2.namedWindow(TOP_WIN, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(TOP_WIN, on_top)
    r0 = arm.read() if arm is not None else None

    def panel_hold(m, on):
        """HOLD button: arm_server keeps m where it is for every script until released."""
        nonlocal grip
        if arm is None:                      # dry run: just remember it
            if on:
                held[m] = grip if m == "gripper" else real_now()[m]
            else:
                held.pop(m, None)
        else:
            new = arm.hold(m, on)["held"]
            held.clear()
            held.update(new)
        if "gripper" in held:
            grip = held["gripper"]
        print(f"  {m} {'HELD at ' + format(held[m], '.1f') if on else 'released'}")
        log("hold", motor=m, on=on, held=held)
        return dict(held)

    panel = MotorPanel(DRY_LIMITS if r0 is None else r0["limits"], real_now(), panel_move, panel_check,
                       speed=args.speed, title="motors" + (" (dry run)" if arm is None else ""), closable=False,
                       hold=panel_hold, held=held, embed=not args.windows)
    if args.windows and wrist_cap is not None:
        cv2.namedWindow(WRIST_WIN, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(WRIST_WIN, on_wrist)
    if args.windows and base_cap is not None:
        cv2.namedWindow(BASE_WIN, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(BASE_WIN, on_base)
    grabbers = {"top": Grabber(top_cap, crop_to_4_3, "top")}
    if wrist_cap is not None:
        grabbers["wrist"] = Grabber(wrist_cap, None, "wrist")
    if base_cap is not None:
        grabbers["base"] = Grabber(base_cap, crop_to_4_3, "base")
    poly = np.array([calib["corners_px"][c] for c in align.CORNERS], np.int32)

    SHOW = {"top": TOP_SHOW, "wrist": 1.0, "base": BASE_SHOW}

    def grab_frames():
        """The newest frame of every camera (one moment for the clean and the overlay pictures)."""
        return {c: g.get() for c, g in grabbers.items()}

    def rec_views(frames, views):
        """What a recorded frame holds: <cam> = the clean picture at the window size (no overlays: the training
        data; clicks map to it by view_scale like before), overlay/<cam> = the window as shown (viewer only)."""
        out = {c: f if SHOW[c] == 1.0 else cv2.resize(f, None, fx=SHOW[c], fy=SHOW[c], interpolation=cv2.INTER_AREA)
               for c, f in frames.items()}
        if not args.no_rec_overlay:
            out.update({f"overlay/{c}": v for c, v in views.items()})
        return out

    def draw_model_point(img, cam):
        """--suggest: the model's click (blue) on the full-size picture of cam."""
        s = sugg.result if sugg is not None else None
        if s and s["label"] == f"click_{cam}" and s["x"] is not None:
            h, w = img.shape[:2]
            px, r = (int(s["x"] * w), int(s["y"] * h)), max(8, w // 90)
            cv2.circle(img, px, r, SUGGEST_COLOR, max(2, r // 3), cv2.LINE_AA)
            cv2.circle(img, px, max(2, r // 4), SUGGEST_COLOR, -1, cv2.LINE_AA)

    def model_banner(view):
        """--suggest: the model's answer (or that it is thinking) in the top view."""
        if sugg is None:
            return
        s = sugg.result
        if st.get("ask_at") is not None or sugg.thinking():
            text = "model: thinking ..."
        elif s is None:
            text = "model: G = ask"
        else:
            text = f"model: {s['text'] if s['label'] else 'not understood: ' + s['text']}   A = accept"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y = 52
        cv2.rectangle(view, (4, y - th - 8), (16 + tw, y + 6), (0, 0, 0), -1)
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, SUGGEST_COLOR, 2, cv2.LINE_AA)

    def render(q_now, kr, moving=False, frames=None):
        """The camera windows as shown: overlays on copies of the newest frames (or of frames). kr: the Kin to draw
        with (the move-time recorder thread has its own). Returns {camera: image}."""
        frames = frames or grab_frames()
        views = {}
        top = frames["top"].copy()
        cv2.polylines(top, [poly], True, (0, 0, 255), 2)
        cv2.drawContours(top, zone_hover, -1, (255, 200, 0), 2, cv2.LINE_AA)
        if args.allow_down:
            cv2.drawContours(top, zone_down, -1, (255, 255, 255), 2, cv2.LINE_AA)
        draw_arm(top, kr, q_now, (0, 220, 255), TW, TH)
        if st["plan"]:
            draw_arm(top, kr, st["plan"][-1][1], (255, 0, 255), TW, TH)
        kr.set(q_now)
        if "top" in st["clicks"]:
            cv2.drawMarker(top, st["clicks"]["top"][:2], (0, 255, 255), cv2.MARKER_CROSS, 40, 3)
        if st["goal"] is not None:
            px = project(kr.m, kr.d, "top_camera", st["goal"], TW, TH)
            if px is not None:
                cv2.circle(top, px, 14, (0, 255, 0), 4)
        sn = st["snapped"]
        if sn is not None:                                   # the snapped block (cyan box) / pad (cyan ring)
            if sn["outline"] is not None:
                o = np.int32(np.round(sn["outline"]))
                for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)):
                    cv2.line(top, tuple(o[a]), tuple(o[b]), (255, 255, 0), 2, cv2.LINE_AA)
            else:
                c = topcam.project(sn["point"])[0]
                cv2.circle(top, tuple(np.int32(c)), 22, (255, 255, 0), 3, cv2.LINE_AA)
        draw_model_point(top, "top")
        view = cv2.resize(top, None, fx=TOP_SHOW, fy=TOP_SHOW, interpolation=cv2.INTER_AREA)
        model_banner(view)
        status = f"surface: {SURFACES[st['surface']][0]} (P)   " + (
            "moving ..." if moving else
            f"plan: {' -> '.join(n for n, _, _ in st['plan'])}  ENTER = go" if st["plan"] else
            "SPACE / ENTER = plan hover" + ("   D = down" if args.allow_down else ""))
        cv2.putText(view, status, (8, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        ep = rec.ep if rec is not None else None
        if ep is not None:
            cv2.circle(view, (18, 18), 8, (0, 0, 255), -1)
            cv2.putText(view, f"REC episode {ep['idx']:03d}  {rec.elapsed():.0f} s   R = end + save  B = discard",
                        (32, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        elif rec is not None:
            cv2.putText(view, "R = start recording an episode", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 255), 1, cv2.LINE_AA)
        views["top"] = view

        if "wrist" in grabbers:
            wrist = frames["wrist"].copy()
            if "wrist" in st["clicks"]:
                cv2.drawMarker(wrist, st["clicks"]["wrist"][:2], (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
            if st["goal"] is not None and arm is not None:
                px = project(kr.m, kr.d, "wrist_camera", st["goal"], WW, WH)
                if px is not None:
                    cv2.circle(wrist, px, 10, (0, 255, 0), 3)
            draw_model_point(wrist, "wrist")
            views["wrist"] = wrist

        if "base" in grabbers:
            bimg = frames["base"].copy()
            if base is not None:
                tip = base.project(kr, fixed_tip(kr))          # live check of the calibration
                if tip is not None:
                    cv2.circle(bimg, tip, 9, (0, 220, 255), 3)
                if st["goal"] is not None and arm is not None:
                    px = base.project(kr, st["goal"])
                    if px is not None:
                        cv2.circle(bimg, px, 16, (0, 255, 0), 4)
            if "base" in st["clicks"]:
                cv2.drawMarker(bimg, st["clicks"]["base"][:2], (0, 255, 255), cv2.MARKER_CROSS, 40, 3)
            draw_model_point(bimg, "base")
            views["base"] = cv2.resize(bimg, None, fx=BASE_SHOW, fy=BASE_SHOW, interpolation=cv2.INTER_AREA)
        return views

    # ── the one window: 3 cameras on top, motors + keys / messages below ──
    CAMS3 = (("top", "top camera - click the goal", on_top), ("wrist", "wrist camera - optional click", on_wrist),
             ("base", "base camera - optional click", on_base))
    dash = {"vsize": {}, "active": None, "last": None}

    def compose(views):
        ph, pw = panel.img.shape[:2]
        H_, W_ = TILE_H + ph, 3 * TILE_W
        img = np.full((H_, W_, 3), DASH_BG, np.uint8)
        for k, (cam, label, _) in enumerate(CAMS3):
            x0 = k * TILE_W
            if cam in views:
                v = views[cam]
                dash["vsize"][cam] = (v.shape[1], v.shape[0])
                img[:TILE_H, x0:x0 + TILE_W] = cv2.resize(v, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
            else:
                cv2.putText(img, f"{cam} camera off", (x0 + 200, TILE_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            DASH_DIM, 1, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x0 + TILE_W - tw - 14, 0), (x0 + TILE_W, th + 12), (0, 0, 0), -1)
            cv2.putText(img, label, (x0 + TILE_W - tw - 7, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DASH_FG, 1,
                        cv2.LINE_AA)
            if k:
                cv2.line(img, (x0, 0), (x0, TILE_H), (70, 70, 70), 2)
        img[TILE_H:TILE_H + ph, :pw] = panel.img
        cv2.line(img, (0, TILE_H), (W_, TILE_H), (70, 70, 70), 2)
        # keys + the last messages
        x0, y = pw + 16, TILE_H + 26
        for line in DASH_HELP + ([SUGGEST_HELP] if sugg is not None else []):
            cv2.putText(img, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, DASH_FG, 1, cv2.LINE_AA)
            y += 22
        cv2.line(img, (x0, y - 8), (W_ - 16, y - 8), (70, 70, 70), 1)
        width = int((W_ - x0 - 16) / 7.2)
        wrapped = []
        for line in list(tee.lines if tee is not None else [])[-40:]:
            line = line.rstrip()
            while len(line) > width:
                wrapped.append(line[:width])
                line = "    " + line[width:]
            wrapped.append(line)
        room = (H_ - y - 6) // 19
        for line in wrapped[-room:] if room > 0 else []:
            y += 19
            low = line.lower()
            col = (90, 90, 255) if ("refused" in low or "warning" in low or "error" in low or "stopped" in low) \
                else (120, 220, 120) if line.lstrip().startswith(("goal", "saved", "rec ")) else DASH_DIM
            cv2.putText(img, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        return img

    def dash_show():
        if dash["last"] is not None:
            cv2.imshow(DASH_WIN, compose(dash["last"]))

    def dash_mouse(event, x, y, flags, _param):
        """Route a click / drag / wheel to the camera or the motors part it is in."""
        def region(x, y):
            if y < TILE_H:
                return CAMS3[min(int(x // TILE_W), 2)][0]
            return "panel" if x < panel.img.shape[1] else None
        reg = region(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            dash["active"] = reg
        elif event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP) and dash["active"] == "panel":
            reg = "panel"                    # a slider drag goes on even outside the motors part
        if event == cv2.EVENT_LBUTTONUP:
            dash["active"] = None
        if reg == "panel":
            ph, pw = panel.img.shape[:2]
            panel.mouse(event, int(np.clip(x, 0, pw - 1)), int(np.clip(y - TILE_H, 0, ph - 1)), flags)
        elif reg is not None and event == cv2.EVENT_LBUTTONDOWN:
            k = [c for c, _, _ in CAMS3].index(reg)
            vw, vh = dash["vsize"].get(reg, (TILE_W, TILE_H))
            CAMS3[k][2](event, int((x - k * TILE_W) * vw / TILE_W), int(y * vh / TILE_H), flags, None)

    if not args.windows:
        panel.on_show = dash_show
        cv2.namedWindow(DASH_WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(DASH_WIN, int(3 * TILE_W * args.ui_scale), int((TILE_H + panel.h) * args.ui_scale))
        cv2.setMouseCallback(DASH_WIN, dash_mouse)

    if args.record is not None:
        rec = Recorder(args.record, fps=args.rec_fps,
                       calib_files=[TOP_CAM_CALIB, WRIST_CLICK_CALIB, BASE_CLICK_CALIB, MAP_FILE],
                       info={"tool": "click_goal_real.py",
                             "cameras": {"top": {"index": args.top_cam, "mode": TOP_RES, "view_scale": TOP_SHOW},
                                         "wrist": None if wrist_cap is None else
                                         {"index": args.wrist_cam, "mode": WRIST_RES, "view_scale": 1.0},
                                         "base": None if base_cap is None else
                                         {"index": args.base_cam, "mode": BASE_RES, "view_scale": BASE_SHOW}},
                             "clean_videos": True,          # <cam>.mp4 = no overlays (from 11-11-2026)
                             "overlay_videos": not args.no_rec_overlay,
                             "speed_deg_s": args.speed, "allow_down": args.allow_down, "dry_run": arm is None})
        print(f"recording to {rec.folder}  -  R = start an episode, R again = end + save, B = throw it away")
    print("left-click the goal in the top camera (optionally also in the wrist camera), SPACE = plan, "
          "ENTER = move, C = clear, H = park, P = surface, Q = quit")
    if sugg is not None:
        # click pixels -> 0..1 of the picture (as build_decisions: the full camera frame, before the window scale)
        sugg.history.sizes = {c: (f.shape[1], f.shape[0]) for c, f in grab_frames().items()}
        print("suggest mode: the model's answer is blue (top view: its text) - A = accept, or do your own step "
              "(that one is what gets recorded); G = ask again, N = forget its history (R does it too)")
        ask_model()

    def after_move():
        """Where the arm went: state, sliders; the wrist / base click rays were for the old pose."""
        joints = real_now()
        st["q_now"], st["t_read"], st["joints"] = real_to_sim(joints, jm, env.q_home), time.time(), joints
        panel.sync({**joints, "gripper": grip})
        st["clicks"].pop("wrist", None)
        st["clicks"].pop("base", None)
        ask_model()

    def do_nudge(name, d):
        """Arrow key: one small step of the gripper point (no ENTER: a step is 1 / 3 cm and checked like a plan)."""
        step = NUDGE_STEPS[st["big_step"]]
        q, target, why = plan_nudge(env, kin, current_q(), d, step)
        if why:
            print(f"nudge {name} refused: {why}")
            log("nudge", direction=name, step=step, ok=False, reason=why)
            return
        if st["plan"]:
            print("  nudge: plan cleared (it was for the old pose)")
        st["plan"] = []
        log("nudge", direction=name, step=step, ok=True, target=target)
        send(f"nudge {name} {step * 100:.0f} cm", q, target)
        after_move()

    def accept_model():
        """A: do what the model said - a click is placed like yours (then SPACE / D / ENTER as usual), a gripper
        answer is sent like X / O."""
        s = sugg.take()
        if s is None:
            print("no model answer yet" + (" (thinking ...)" if sugg.thinking() else ": G = ask"))
            return
        if s["label"] is None:
            print(f"model answer not understood: {s['text']!r} - G = ask again")
            return
        log("suggest_accept", label=s["label"], x=s["x"], y=s["y"], text=s["text"])
        lab = s["label"]
        if lab.startswith("click_"):
            cam = lab[len("click_"):]
            w, h = sugg.history.sizes[cam]
            handler = {"top": on_top, "wrist": on_wrist, "base": on_base}[cam]
            handler(cv2.EVENT_LBUTTONDOWN, int(round(s["x"] * w * SHOW[cam])), int(round(s["y"] * h * SHOW[cam])),
                    0, None)
            print(f"  model's {cam} click placed - SPACE / D = plan, ENTER = move (or click yourself to change it)")
        elif lab in ("gripper_close", "gripper_open"):
            set_gripper(REAL_GRIPPER_CLOSED if lab == "gripper_close" else REAL_GRIPPER_OPEN, "model")
        else:
            print("model says: done" + (" - R = end + save the episode" if rec is not None and rec.active else ""))

    try:
        while True:
            due = time.time() - st["t_read"] > 0.2 or (rec is not None and rec.due())
            if due and not panel.dragging:                    # where the arm is (each frame while recording)
                st["joints"] = real_now()
                st["q_now"], st["t_read"] = real_to_sim(st["joints"], jm, env.q_home), time.time()
                panel.set_now(st["joints"])
            frames = grab_frames()
            views = render(st["q_now"], kin, frames=frames)
            if args.windows:
                cv2.imshow(TOP_WIN, views["top"])
                if "wrist" in views:
                    cv2.imshow(WRIST_WIN, views["wrist"])
                if "base" in views:
                    cv2.imshow(BASE_WIN, views["base"])
            else:
                dash["last"] = views
                if panel.img is None:
                    panel.draw()
                cv2.imshow(DASH_WIN, compose(views))
            if rec is not None and rec.active:
                rec.add(rec_views(frames, views), state=st["joints"], action=cmd)
            if sugg is not None:
                if st["ask_at"] is not None and time.time() >= st["ask_at"]:
                    st["ask_at"] = None
                    sugg.ask(grab_frames(), float(real_now().get("gripper", grip)), SHOW)
                s = sugg.poll()
                if s is not None:
                    print(f"model: {s['text']}  ({s['seconds']:.1f} s)  A = accept")
                    log("suggest", text=s["text"], label=s["label"], x=s["x"], y=s["y"], seconds=s["seconds"],
                        gripper_now=s["gripper_now"], recent=s["recent"])

            code = st["key_next"] if st["key_next"] is not None else cv2.waitKeyEx(1)
            st["key_next"] = None
            nudge = NUDGE_KEYS.get(code & 0xFFFF) if code != -1 else None
            key = 255 if code == -1 or nudge else code & 0xFF   # as cv2.waitKey(1) & 0xFF before
            panel.poll()
            if not args.windows and cv2.getWindowProperty(DASH_WIN, cv2.WND_PROP_VISIBLE) < 1:
                break                                           # the window was closed with its X
            if nudge is not None:
                do_nudge(*nudge)
                k = cv2.waitKeyEx(1)          # an arrow pressed during the step (key repeat) is dropped: it stops
                if k != -1 and (k & 0xFFFF) not in NUDGE_KEYS:  # when you let go; another key is kept
                    st["key_next"] = k
            elif key == ord("s"):
                st["big_step"] = not st["big_step"]
                print(f"arrow step: {NUDGE_STEPS[st['big_step']] * 100:.0f} cm")
            elif key in (ord("q"), 27):
                break
            elif key == ord(" "):
                st["plan"] = plan("hover")
            elif key == ord("d"):
                st["plan"] = plan("down") if args.allow_down else []
                if not args.allow_down:
                    print("D needs --allow-down")
            elif key == ord("h"):
                st["plan"] = plan("park")
            elif key in (13, 10):                               # ENTER
                if not st["plan"]:                              # nothing shown yet: plan the hover, don't move
                    if st["goal"] is None:
                        print("no goal yet: left-click it in the top camera")
                        continue
                    st["plan"] = plan("hover")
                    if st["plan"]:
                        print("  planned hover (magenta) - ENTER again to move")
                    continue
                log("execute", steps=[n for n, _, _ in st["plan"]])
                for name, q, pos in st["plan"]:
                    if not send(name, q, pos):
                        break
                st["plan"] = []
                after_move()                                    # sliders, state; old wrist / base rays dropped
            elif key == ord("a") and sugg is not None:
                accept_model()
            elif key == ord("g") and sugg is not None:
                ask_model()
            elif key == ord("n") and sugg is not None:
                sugg.history.reset()
                print("model history forgotten (as at an episode start)")
                ask_model()
            elif key in (ord("x"), ord("o")):
                set_gripper(*((REAL_GRIPPER_CLOSED, "X key") if key == ord("x") else (REAL_GRIPPER_OPEN, "O key")))
            elif key == ord("m"):
                st["snap"] = not st["snap"]
                print(f"snap top clicks: {'on' if st['snap'] else 'off'}")
            elif key == ord("c"):
                log("clear")
                st["clicks"].clear()
                st["snapped"] = None
                update_goal()
            elif key == ord("p"):
                st["surface"] = (st["surface"] + 1) % len(SURFACES)
                print(f"clicked surface: {SURFACES[st['surface']][0]}")
                log("surface", surface=SURFACES[st["surface"]][0])
                update_goal()
            elif key == ord("r"):
                if rec is None:
                    print("not recording: start the tool with --record NAME")
                elif not rec.active:
                    idx, d = rec.start()
                    log("episode_start", joints=st["joints"], action=cmd, goal=st["goal"], surface=SURFACES[st["surface"]][0],
                        clicks={cam: c[:2] for cam, c in st["clicks"].items()}, held=held, speed=args.speed)
                    print(f"REC episode {idx:03d} -> {d}   (R = end + save, B = throw away)")
                    set_gripper(REAL_GRIPPER_OPEN, "episode start")     # every episode starts with the gripper open
                    ask_model()
                else:
                    d, n = rec.stop(keep=True)
                    print(f"saved episode: {d} ({n} frames)")
            elif key == ord("b"):
                if rec is not None and rec.active:
                    _, n = rec.stop(keep=False)
                    print(f"episode thrown away ({n} frames)")
    finally:
        if rec is not None and rec.active:
            d, n = rec.stop(keep=True)
            print(f"saved the running episode: {d} ({n} frames)")
        for g in grabbers.values():
            g.stop()
        top_cap.release()
        if wrist_cap is not None:
            wrist_cap.release()
        if base_cap is not None:
            base_cap.release()
        cv2.destroyAllWindows()
        panel.close()
        if tee is not None:
            sys.stdout = tee.stream
        if arm is not None:
            if input("\nReturn the arm to its start pose (via the safe pose)? [Y/n] ").strip().lower() != "n":
                steps = plan("park")[:-1] or [("safe pose", env.q_seed, None)]   # lift + safe, not the park
                if all(send(n, q, p) for n, q, p in steps):
                    arm.home()
            arm.close()


if __name__ == "__main__":
    main()
