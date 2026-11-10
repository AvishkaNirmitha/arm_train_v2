"""Calibrate the base camera (on the arm's base, turns with shoulder_pan) for click_goal_real.py.

The arm moves to ~15 LOW poses in front of the camera (take blocks off the mat first). In each one you click the TIP of the FIXED jaw in the base camera
image (the jaw that does NOT move when the gripper opens - press G to wiggle the gripper and see). The sim knows
where that tip is for the joint angles the arm reports, so the clicks give the camera's pose on the base (and its
fovy, kept near the top camera's: same camera model). No blocks to place, nothing to measure.

Needs arm_server.py running (.venv-lerobot). Run in .venv, from the project root:

    python so101_real/calib_base_camera.py            # BASE_CAM from .env (or --base-cam N)
    python so101_real/calib_base_camera.py --refit     # fit again from the saved clicks (no arm, no camera)

Clicking: left-click = the fixed jaw's tip (click again to move it; a zoom box appears - click in it to refine),
SPACE / ENTER = next pose, S = skip (tip not visible), G = wiggle the gripper, Q = stop and fit.
RED = where the fit so far puts the tip (after 5 clicks): only a hint, click what you see.
Ctrl+C in the arm_server terminal stops a move; the poses clicked so far are still fitted.
Writes base_camera_calib.json (read by click_goal_real.py) and keeps images + clicks in captures/base_calib_<time>/.
"""
import os
import sys
import glob
import json
import time
import argparse
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pick_task import PickEnv                                                      # noqa: E402
from real_arm_client import ArmClient, load_joint_map, sim_to_real, real_to_sim    # noqa: E402
from real_robot_policy import open_camera, crop_to_4_3                              # noqa: E402
from cam_config import BASE_CAM, BASE_RES                                          # noqa: E402  (.env)
from capture_wrist_calib import frame                                              # noqa: E402
from click_goal_real import Kin                                                    # noqa: E402
from motor_panel import path_check                                                 # noqa: E402
from base_camera import BASE_CLICK_CALIB, TOP_FOVY, BaseCam, fixed_tip, body_pose, focal_px   # noqa: E402

SHOW = 0.5                       # window scale (1440x1080 -> 720x540)
ZOOM, ZOOM_HALF = 4, 40          # zoom box: 80x80 image pixels shown 4x bigger (in window pixels: 4 * SHOW)
WINDOW = "base camera: click the TIP of the FIXED jaw"
SPEED = 15.0                     # deg/s
CLICK_PX = 2.0                   # assumed click accuracy (image pixels)
FOVY_SIGMA = 1.0                 # deg: prior on fovy around the top camera's (same model; the low view can't tell fovy from distance)
MIN_POINTS = 6


def pose_list(env, kin):
    """Sim joint poses with the fixed tip spread over the camera's view: 5 distances x 3 heights, in the safe pose's
    direction (over the mat), in an order that keeps every move short. The camera sits low and looks almost level:
    it only sees the tip when the gripper is low (tip ~1.5-5 cm above the mat; at 10 cm it is above the image)."""
    tcp = kin.set(env.q_seed).tcp_pos()
    az = np.arctan2(tcp[1], tcp[0])
    out, q = [], env.q_seed4.copy()
    heights = (0.065, 0.05, 0.035)                             # gripper point; the fixed tip is ~1.5 cm lower
    for i, r in enumerate((0.21, 0.24, 0.27, 0.30, 0.33)):
        for z in (heights if i % 2 == 0 else heights[::-1]):
            q, e, _ = env._solve(np.array([r * np.cos(az), r * np.sin(az), z]), q)
            if e > 3e-3:
                continue
            out.append((f"r{r * 100:.0f}_z{z * 1000:.0f}mm", env.full_q(q)))
    return out


# ── Fit ──────────────────────────────────────────────────────────────────────
def K_of(fovy, w, h):
    f = focal_px(fovy, h)
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])


def project_body(x, P, w, h):
    """x = [rvec, tvec (body -> camera, OpenCV), fovy]; P: points in the shoulder body frame -> pixels."""
    uv, _ = cv2.projectPoints(np.asarray(P, float).reshape(-1, 1, 3), x[:3], x[3:6], K_of(x[6], w, h), None)
    return uv.reshape(-1, 2)


def fit(P, uv, w, h, fovy0=TOP_FOVY):
    """PnP (SQPnP, fovy fixed) for the start, then Levenberg-Marquardt on pose + fovy (prior FOVY_SIGMA)."""
    P, uv = np.asarray(P, float), np.asarray(uv, float)
    ok, rvec, tvec = cv2.solvePnP(P, uv, K_of(fovy0, w, h), None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    x = np.array([*rvec.ravel(), *tvec.ravel(), fovy0])

    def res(x):
        return np.concatenate([((project_body(x, P, w, h) - uv) / CLICK_PX).ravel(), [(x[6] - fovy0) / FOVY_SIGMA]])

    r, lam = res(x), 1e-3
    eps = np.array([1e-6] * 3 + [1e-6] * 3 + [1e-3])
    for _ in range(50):
        J = np.column_stack([(res(x + e) - r) / e[k] for k, e in enumerate(np.diag(eps))])
        H, g = J.T @ J, J.T @ r
        while True:
            dx = -np.linalg.solve(H + lam * np.diag(np.diag(H)), g)
            r_new = res(x + dx)
            if r_new @ r_new < r @ r:
                x, r, lam = x + dx, r_new, max(lam / 3, 1e-7)
                break
            lam *= 4
            if lam > 1e6:
                return x
        if np.linalg.norm(dx) < 1e-10:
            break
    return x


def to_cam(x, w, h, rms):
    """Fit vector -> BaseCam (camera -> body rotation and the lens position in the body frame)."""
    R_cb = cv2.Rodrigues(x[:3])[0]
    return BaseCam(R_cb.T, -R_cb.T @ x[3:6], x[6], w, h, rms)


def errors(x, P, uv, w, h):
    """Pixel error per point, and how far (mm) the clicked ray passes from the true point."""
    P, uv = np.asarray(P, float), np.asarray(uv, float)
    px = np.linalg.norm(project_body(x, P, w, h) - uv, axis=1)
    R_cb = cv2.Rodrigues(x[:3])[0]
    f = focal_px(x[6], h)
    mm = []
    for p, (u, v) in zip(P, uv):
        d = np.array([(u - (w - 1) / 2) / f, (v - (h - 1) / 2) / f, 1.0])
        d /= np.linalg.norm(d)
        pc = R_cb @ p + x[3:6]
        mm.append(np.linalg.norm(pc - (pc @ d) * d) * 1000)
    return px, np.array(mm)


def fit_and_save(rows, w, h, folder, kin, env, jm):
    items = [r for r in rows if r.get("click") is not None]
    if len(items) < MIN_POINTS:
        raise SystemExit(f"only {len(items)} clicked poses - need at least {MIN_POINTS}")
    P, uv = [], []
    for r in items:
        kin.set(real_to_sim(r["reached_real_deg"], jm, env.q_home))
        pb, Rb = body_pose(kin)
        P.append(Rb.T @ (fixed_tip(kin) - pb))                  # the tip in the shoulder body frame
        uv.append(r["click"])
    P, uv = np.array(P), np.array(uv)
    x = fit(P, uv, w, h)
    if x is None:
        raise SystemExit("the fit failed (clicks inconsistent?) - run again and check you click the FIXED jaw's tip")
    px, mm = errors(x, P, uv, w, h)
    bad = px > max(3 * np.median(px), 12.0)
    if bad.any() and (~bad).sum() >= MIN_POINTS:
        print("outliers left out (a mis-click, or the other jaw?):",
              ", ".join(f"{r['name']} {p:.0f} px" for r, p, b in zip(items, px, bad) if b))
        items, P, uv = [r for r, b in zip(items, bad) if not b], P[~bad], uv[~bad]
        x = fit(P, uv, w, h)
        px, mm = errors(x, P, uv, w, h)
    rms = float(np.sqrt((px ** 2).mean()))
    print(f"\n{len(items)} points: {px.mean():.1f} px mean, {rms:.1f} px rms | ray misses the tip by {mm.mean():.1f} mm "
          f"mean, {mm.max():.1f} mm max")
    print("  per pose:", "  ".join(f"{r['name']} {p:.1f}px" for r, p in zip(items, px)))
    # honest check: fit without each pose, test on it
    loo_px, loo_mm = [], []
    for i in range(len(P)):
        keep = np.arange(len(P)) != i
        xi = fit(P[keep], uv[keep], w, h)
        if xi is not None:
            a, b = errors(xi, P[i:i + 1], uv[i:i + 1], w, h)
            loo_px.append(a[0])
            loo_mm.append(b[0])
    print(f"leave-one-out (a pose the fit hasn't seen): {np.mean(loo_px):.1f} px, ray misses by {np.mean(loo_mm):.1f} mm")
    cam = to_cam(x, w, h, rms)
    kin.set(env.q_seed)
    o, R = cam.world_pose(kin)
    look = R[:, 2]
    print(f"camera: lens {np.round(cam.t * 100, 1)} cm from the shoulder body origin, {o[2] * 100:.1f} cm above the "
          f"table, looking {np.degrees(np.arcsin(-look[2])):.0f} deg down; fovy {x[6]:.1f} (top camera {TOP_FOVY})")
    out = {"body": "shoulder", "R_body_cam": np.round(cam.R, 6).tolist(), "t_body_cam": np.round(cam.t, 5).tolist(),
           "fovy": round(float(x[6]), 2), "image_size": [w, h], "rms_px": round(rms, 2),
           "ray_mm": round(float(mm.mean()), 2), "loo_px": round(float(np.mean(loo_px)), 2),
           "loo_mm": round(float(np.mean(loo_mm)), 2), "n_points": len(items), "data": folder,
           "created": datetime.now().isoformat(timespec="seconds")}
    with open(BASE_CLICK_CALIB, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {BASE_CLICK_CALIB} (click_goal_real.py --base-cam N uses it)")


# ── Clicking ─────────────────────────────────────────────────────────────────
def click_tip(cap, title, hint, wiggle):
    """Live view until SPACE (returns [u, v]), S (None) or Q ("quit"). hint: pixel or None."""
    st = {"uv": None}
    zoom_box = {"rect": None}      # window rect of the zoom box and the image pixel at its centre

    def on_mouse(e, mu, mv, *_):
        if e != cv2.EVENT_LBUTTONDOWN:
            return
        zb = zoom_box["rect"]
        if zb is not None and zb[0] <= mu < zb[2] and zb[1] <= mv < zb[3]:      # refine inside the zoom box
            cu, cv_ = zb[4]
            s = ZOOM * SHOW
            st["uv"] = [cu - ZOOM_HALF + (mu - zb[0]) / s, cv_ - ZOOM_HALF + (mv - zb[1]) / s]
        else:
            st["uv"] = [mu / SHOW, mv / SHOW]

    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        img = crop_to_4_3(frame(cap, 1))
        view = cv2.resize(img, None, fx=SHOW, fy=SHOW, interpolation=cv2.INTER_AREA)
        if hint is not None:
            cv2.drawMarker(view, tuple(int(c * SHOW) for c in hint), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
        zoom_box["rect"] = None
        if st["uv"] is not None:
            u, v = st["uv"]
            cv2.drawMarker(view, (int(u * SHOW), int(v * SHOW)), (0, 255, 255), cv2.MARKER_CROSS, 30, 1)
            cu, cv_ = int(round(u)), int(round(v))
            pad = cv2.copyMakeBorder(img, ZOOM_HALF, ZOOM_HALF, ZOOM_HALF, ZOOM_HALF, cv2.BORDER_CONSTANT)
            crop = pad[cv_:cv_ + 2 * ZOOM_HALF, cu:cu + 2 * ZOOM_HALF]
            s = ZOOM * SHOW
            big = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            bh, bw = big.shape[:2]
            c = (int((u - cu + ZOOM_HALF) * s), int((v - cv_ + ZOOM_HALF) * s))
            cv2.drawMarker(big, c, (0, 255, 255), cv2.MARKER_CROSS, 40, 1)
            x0 = 0 if u * SHOW > view.shape[1] / 2 else view.shape[1] - bw     # the corner away from the click
            y0 = view.shape[0] - bh - 30 if v * SHOW < view.shape[0] / 2 else 30
            view[y0:y0 + bh, x0:x0 + bw] = big
            cv2.rectangle(view, (x0, y0), (x0 + bw - 1, y0 + bh - 1), (0, 255, 255), 1)
            zoom_box["rect"] = (x0, y0, x0 + bw, y0 + bh, (cu, cv_))
        cv2.rectangle(view, (0, 0), (view.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(view, f"{title}: click FIXED jaw tip | SPACE next  S skip  G wiggle  Q fit", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, view)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord(" "), 13, 10) and st["uv"] is not None:
            return [round(c, 1) for c in st["uv"]], img
        if k == ord("s"):
            return None, img
        if k == ord("g"):
            wiggle()
        if k in (ord("q"), 27):
            return "quit", img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-cam", type=int, default=BASE_CAM, help=f"base camera index (BASE_CAM in .env: {BASE_CAM})")
    ap.add_argument("--refit", action="store_true", help="fit again from the newest saved clicks (no arm)")
    args = ap.parse_args()
    if args.base_cam is None and not args.refit:
        raise SystemExit("no base camera: put BASE_CAM=N in .env or pass --base-cam N")

    env = PickEnv(randomize=False)
    kin = Kin(env)
    jm = load_joint_map()

    if args.refit:
        folder = max(glob.glob(os.path.join(HERE, "captures", "base_calib_*")), default=None)
        if folder is None:
            raise SystemExit("no captures/base_calib_* folder yet")
        rows = [json.loads(line) for line in open(os.path.join(folder, "poses.jsonl")) if line.strip()]
        w, h = rows[0]["image_size"]
        print(f"refit from {folder}")
        fit_and_save(rows, w, h, folder, kin, env, jm)
        return

    poses = pose_list(env, kin)
    prev = env.q_seed                                          # check the whole sequence in the sim first
    for name, q in poses:
        err, _ = path_check(kin, prev, q)
        if err:
            raise SystemExit(f"pose {name}: {err} - not moving (tell Claude)")
        prev = q

    cap = open_camera(args.base_cam, *BASE_RES)
    h, w = crop_to_4_3(frame(cap)).shape[:2]
    if [w, h] != [1440, 1080]:
        print(f"note: base camera gives {w}x{h} (4:3 crop), not 1440x1080 like the top camera - fine, fovy is fitted")
    arm = ArmClient()
    held = arm.read().get("held", {})
    folder = os.path.join(HERE, "captures", "base_calib_" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    os.makedirs(folder)
    rows = []

    def go(name, q):
        r = arm.move(sim_to_real(q, jm), SPEED)                # the gripper is not sent: it stays as it is
        if r.get("stopped"):
            print(f"  {name}: STOPPED (Ctrl+C on the server)")
            return None
        time.sleep(0.4)
        return arm.read()["joints"]

    def wiggle():
        if "gripper" in held:
            print("gripper is HELD - can't wiggle it (release it in the motors window)")
            return
        g = arm.read()["joints"]["gripper"]
        arm.move({"gripper": g - 25 if g > 40 else g + 25}, 60)
        arm.move({"gripper": g}, 60)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    print(__doc__.split("Clicking:")[1].split("Writes")[0].strip())
    input(f"\nENTER = move to the safe pose, then {len(poses)} poses in front of the base camera (keep clear) ")
    try:
        if go("safe pose", env.q_seed) is None:
            return
        for i, (name, q) in enumerate(poses):
            print(f"pose {i + 1}/{len(poses)} {name}")
            joints = go(name, q)
            if joints is None:
                break
            hint = None
            done = [r for r in rows if r["click"] is not None]
            if len(done) >= 5:                                 # where the fit so far puts the tip
                kin_q = real_to_sim(joints, jm, env.q_home)
                Ps, uvs = [], []
                for r in done:
                    kin.set(real_to_sim(r["reached_real_deg"], jm, env.q_home))
                    pb, Rb = body_pose(kin)
                    Ps.append(Rb.T @ (fixed_tip(kin) - pb))
                    uvs.append(r["click"])
                x = fit(Ps, uvs, w, h)
                if x is not None:
                    kin.set(kin_q)
                    pb, Rb = body_pose(kin)
                    hint = project_body(x, [Rb.T @ (fixed_tip(kin) - pb)], w, h)[0]
            click, img = click_tip(cap, f"{i + 1}/{len(poses)} {name}", hint, wiggle)
            if click == "quit":
                break
            cv2.imwrite(os.path.join(folder, f"{name}.jpg"), img)
            rows.append({"name": name, "reached_real_deg": joints, "click": click, "image_size": [w, h]})
            with open(os.path.join(folder, "poses.jsonl"), "a") as f:
                f.write(json.dumps(rows[-1]) + "\n")
    finally:
        cv2.destroyAllWindows()
        cap.release()
        print("back to the safe pose, then the parked pose ...")
        if go("safe pose", env.q_seed) is not None:
            go("parked pose", env.q_home)
        arm.close()
    fit_and_save(rows, w, h, folder, kin, env, jm)


if __name__ == "__main__":
    main()
