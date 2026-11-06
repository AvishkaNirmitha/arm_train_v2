"""Click a goal in the real top camera (optionally also in the wrist camera) and move the real SO-101 there with IK.

Every click is a ray from the camera through the pixel, from the calibrated camera poses (top_camera_calib.json;
the wrist camera from wrist_camera_calib.json + the joint angles the arm reports at the moment of the click):
  * top click only     -> where the ray hits the surface you clicked on (P: block top / mat / any height)
  * top + wrist clicks -> the point closest to both rays (weighted least squares) and the known surface height;
                          with "any height" the rays alone. No depth camera needed.
The arm goes to the hover point above the goal (gripper point at least HOVER_Z, so the jaws clear a block) and,
with --allow-down, down to it and the gripper can be closed / opened (a manual pick).

Drawn in the top view: RED = calibrated mat outline (if it doesn't sit on the mat edges, the camera moved: run
align_top_camera.py), BLUE = where the arm can hover, WHITE = where it can also go down (--allow-down),
YELLOW = the arm where it is now, MAGENTA = the planned pose, GREEN = the goal.

Needs: arm_server.py running (.venv-lerobot), unless --no-arm. Run in .venv:

    python so101_real/click_goal_real.py --no-arm       # dry run: click, see the 3-D point and the plan
    python so101_real/click_goal_real.py                # move to the hover point only
    python so101_real/click_goal_real.py --allow-down   # + D = down to the goal, X / O = close / open the gripper

The "motors" window (motor_panel.py) has one slider per motor (wrist_roll excluded): drag + let go, wheel, -/+ or
type a value to move one motor; the sim checks the path stays above the table. A manual move clears the plan.

Keys (in a camera window):
    left-click  the goal (one point per window)        P      clicked surface: block top / mat / any height
    SPACE/ENTER plan: hover above the goal             D      plan: down to the goal (--allow-down)
    ENTER       do the planned move                    X / O  gripper close / open (or the motors window)
    C           clear clicks and plan                  H      plan: back to the parked pose
    Q / Esc     quit
Every move waits for ENTER; Ctrl+C in the arm_server terminal stops a move on the spot.
"""
import os
import sys
import json
import time
import argparse

import cv2
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from real_env import (TOP_CAM_CALIB, ARM_JOINTS, MAT_TOP, MAT_HALF, BLOCK_HALF, GRASP_Z, HOVER_Z,  # noqa: E402
                      REAL_GRIPPER_OPEN, REAL_GRIPPER_CLOSED, MOUNT_SCENE_POS, scene_to_robot_rot, _DIMS)
from pick_task import PickEnv, MAX_GRASP_TILT_DEG                                            # noqa: E402
from reach_task import REACH_RADIUS                                                           # noqa: E402
from real_arm_client import ArmClient, load_joint_map, sim_to_real, real_to_sim               # noqa: E402
from real_robot_policy import open_camera, crop_to_4_3, fmt, TOP_CAM, WRIST_CAM, TOP_RES, MIN_TCP_Z  # noqa: E402
from capture_wrist_calib import frame, check_top_alignment                                    # noqa: E402
import align_top_camera as align                                                              # noqa: E402
from motor_panel import MotorPanel, path_check, DRY_LIMITS                                    # noqa: E402

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
    ap.add_argument("--top-cam", type=int, default=TOP_CAM)
    ap.add_argument("--wrist-cam", type=int, default=WRIST_CAM)
    ap.add_argument("--no-wrist", action="store_true", help="top camera only")
    ap.add_argument("--no-arm", action="store_true", help="dry run: no arm_server, nothing moves")
    ap.add_argument("--allow-down", action="store_true", help="allow D (down to the goal) and the gripper keys")
    ap.add_argument("--speed", type=float, default=15.0, help="joint speed, deg/s")
    args = ap.parse_args()

    with open(TOP_CAM_CALIB) as f:
        calib = json.load(f)
    top_cap = open_camera(args.top_cam, *TOP_RES)
    wrist_cap = None if args.no_wrist else open_camera(args.wrist_cam)
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
    print(f"ray accuracy: top {np.degrees(sigma['top']):.2f} deg, wrist {np.degrees(sigma['wrist']):.2f} deg")
    jm = load_joint_map()
    arm = None if args.no_arm else ArmClient()
    grip = REAL_GRIPPER_OPEN             # commanded gripper, sent with every move (a held block stays squeezed)
    sim_q = env.q_home.copy()            # --no-arm: where the arm is assumed to be

    top = crop_to_4_3(frame(top_cap))
    TH, TW = top.shape[:2]
    WH, WW = frame(wrist_cap).shape[:2] if wrist_cap is not None else (480, 640)   # fitted at 640x480 (any 4:3)
    ok = reach_map(env)
    kin.set(env.q_home)
    zone_hover, zone_down = (reach_outlines(ok, kin, TW, TH, level) for level in (1, 2))
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

    def send(name, q, pos=None):
        """One move (blocking). Returns False if it was stopped."""
        nonlocal sim_q
        target = {**sim_to_real(q, jm), "gripper": grip}
        where = "" if pos is None else f"gripper point {np.round(pos, 3)} "
        print(f"  {name}: {where}-> {fmt(target)}")
        if arm is None:
            sim_q = np.array(q, float)
            return True
        r = arm.move(target, args.speed)
        print(f"    reached (max err {r['max_err_deg']:.1f} deg){'  CLAMPED: ' + str(r['clamped']) if r['clamped'] else ''}")
        if r.get("stopped"):
            print("    STOPPED (Ctrl+C on the server) - plan cleared")
            return False
        return True

    if arm is not None:
        input("ENTER = move to the safe pose above the mat, then the parked pose (gripper opens) ")
        if not (send("safe pose", env.q_seed) and send("parked pose", env.q_home)):
            arm.close()
            return

    st = {"clicks": {}, "goal": None, "plan": [], "surface": 0, "q_now": current_q(), "t_read": time.time()}

    def update_goal():
        rays = [(o, d) for _, _, o, d in st["clicks"].values()]
        sigmas = [sigma[cam] for cam in st["clicks"]]
        if st["plan"]:
            print("  (new click: the plan was for the old goal and is cleared - SPACE / D again)")
        st["goal"], st["plan"] = None, []
        if not rays:
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
        if len(rays) > 1:
            # each ray should pass within ~3 sigma of the point; more = a different point clicked or a bad calibration
            lim = np.array([3 * s * np.linalg.norm(p - o) for s, (o, _) in zip(sigmas, rays)]) + 0.002
            if np.any(res > lim):
                print(f"  ! the rays disagree more than their accuracy ({np.round(res * 1000, 1)} mm vs "
                      f"{np.round(lim * 1000, 1)} mm): same point clicked in both views?")

    def on_top(event, u, v, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            u, v = u / TOP_SHOW, v / TOP_SHOW                # back to calibration pixels
            st["clicks"]["top"] = (int(u), int(v), *cam_ray(kin.m, kin.set(sim_q).d, "top_camera", u, v, TW, TH))
            update_goal()

    def on_wrist(event, u, v, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if arm is None:
                print("wrist clicks need the arm (its joints give the wrist camera pose) - not with --no-arm")
                return
            # the camera pose from the joints the arm reports right now
            st["clicks"]["wrist"] = (u, v, *cam_ray(kin.m, kin.set(current_q()).d, "wrist_camera", u, v, WW, WH))
            update_goal()

    def plan(kind):
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
        err, tcp = path_check(kin, q_from, real_to_sim(target, jm, q_from), MIN_TCP_Z)
        return err, ("" if tcp is None else
                     f"gripper point there: x {tcp[0] * 100:.1f}  y {tcp[1] * 100:.1f}  z {tcp[2] * 100:.1f} cm")

    def panel_move(target, speed):
        """Motors window: move (blocking). Clears the plan / wrist click (made for the old pose)."""
        nonlocal grip, sim_q
        grip = float(target.get("gripper", grip))   # later moves keep sending this (a held block stays squeezed)
        if st["plan"]:
            print("  manual move: plan cleared")
        st["plan"] = []
        st["clicks"].pop("wrist", None)
        if arm is None:
            sim_q = real_to_sim(target, jm, sim_q)
            reached = dict(target)
        else:
            r = arm.move(target, speed)
            reached = r["joints"]
            if r["clamped"]:
                print(f"  clamped to the limits: {r['clamped']}")
        st["q_now"], st["t_read"] = real_to_sim(reached, jm, env.q_home), time.time()
        return reached

    def set_gripper(value):
        """X / O keys: the gripper slider in the motors window, sent now."""
        panel.set_value("gripper", value)
        now = (real_now() if arm is not None else {}).get("gripper", grip)
        print(f"gripper -> {grip:.0f} (now at {now:.0f}{', blocked: holding something?' if abs(now - grip) > 5 else ''})")

    cv2.namedWindow(TOP_WIN, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(TOP_WIN, on_top)
    r0 = arm.read() if arm is not None else None
    panel = MotorPanel(DRY_LIMITS if r0 is None else r0["limits"], real_now(), panel_move, panel_check,
                       speed=args.speed, title="motors" + (" (dry run)" if arm is None else ""), closable=False)
    if wrist_cap is not None:
        cv2.namedWindow(WRIST_WIN, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(WRIST_WIN, on_wrist)
    print("left-click the goal in the top camera (optionally also in the wrist camera), SPACE = plan, "
          "ENTER = move, C = clear, H = park, P = surface, Q = quit")

    try:
        while True:
            if time.time() - st["t_read"] > 0.2 and not panel.dragging:   # where the arm is, a few times a second
                joints = real_now()
                st["q_now"], st["t_read"] = real_to_sim(joints, jm, env.q_home), time.time()
                panel.set_now(joints)
            top = crop_to_4_3(frame(top_cap, 1))
            poly = np.array([calib["corners_px"][c] for c in align.CORNERS], np.int32)
            cv2.polylines(top, [poly], True, (0, 0, 255), 2)
            cv2.drawContours(top, zone_hover, -1, (255, 200, 0), 2, cv2.LINE_AA)
            if args.allow_down:
                cv2.drawContours(top, zone_down, -1, (255, 255, 255), 2, cv2.LINE_AA)
            draw_arm(top, kin, st["q_now"], (0, 220, 255), TW, TH)
            if st["plan"]:
                draw_arm(top, kin, st["plan"][-1][1], (255, 0, 255), TW, TH)
            kin.set(st["q_now"])
            if "top" in st["clicks"]:
                cv2.drawMarker(top, st["clicks"]["top"][:2], (0, 255, 255), cv2.MARKER_CROSS, 40, 3)
            if st["goal"] is not None:
                px = project(kin.m, kin.d, "top_camera", st["goal"], TW, TH)
                if px is not None:
                    cv2.circle(top, px, 14, (0, 255, 0), 4)
            view = cv2.resize(top, None, fx=TOP_SHOW, fy=TOP_SHOW, interpolation=cv2.INTER_AREA)
            status = f"surface: {SURFACES[st['surface']][0]} (P)   " + (
                f"plan: {' -> '.join(n for n, _, _ in st['plan'])}  ENTER = go" if st["plan"] else
                "SPACE / ENTER = plan hover" + ("   D = down" if args.allow_down else ""))
            cv2.putText(view, status, (8, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.imshow(TOP_WIN, view)

            if wrist_cap is not None:
                wrist = frame(wrist_cap, 1)
                if "wrist" in st["clicks"]:
                    cv2.drawMarker(wrist, st["clicks"]["wrist"][:2], (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
                if st["goal"] is not None and arm is not None:
                    px = project(kin.m, kin.d, "wrist_camera", st["goal"], WW, WH)
                    if px is not None:
                        cv2.circle(wrist, px, 10, (0, 255, 0), 3)
                cv2.imshow(WRIST_WIN, wrist)

            key = cv2.waitKey(1) & 0xFF
            panel.poll()
            if key in (ord("q"), 27):
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
                for name, q, pos in st["plan"]:
                    if not send(name, q, pos):
                        break
                st["plan"] = []
                joints = real_now()
                st["q_now"], st["t_read"] = real_to_sim(joints, jm, env.q_home), time.time()
                panel.sync({**joints, "gripper": grip})         # sliders := where the arm went
                st["clicks"].pop("wrist", None)                 # its ray used the old wrist camera pose
            elif key in (ord("x"), ord("o")):
                set_gripper(REAL_GRIPPER_CLOSED if key == ord("x") else REAL_GRIPPER_OPEN)
            elif key == ord("c"):
                st["clicks"].clear()
                update_goal()
            elif key == ord("p"):
                st["surface"] = (st["surface"] + 1) % len(SURFACES)
                print(f"clicked surface: {SURFACES[st['surface']][0]}")
                update_goal()
    finally:
        top_cap.release()
        if wrist_cap is not None:
            wrist_cap.release()
        cv2.destroyAllWindows()
        panel.close()
        if arm is not None:
            if input("\nReturn the arm to its start pose (via the safe pose)? [Y/n] ").strip().lower() != "n":
                steps = plan("park")[:-1] or [("safe pose", env.q_seed, None)]   # lift + safe, not the park
                if all(send(n, q, p) for n, q, p in steps):
                    arm.home()
            arm.close()


if __name__ == "__main__":
    main()
