"""Put the real top camera back where calibrate_top_camera.py measured it (top_camera_calib.json), so the models
trained on that view keep working. No new calibration, no retraining.

Live top camera view (same 4:3 crop as real_robot_policy.py) with the calibrated mat outline in RED. The mat
edges found in the live image are drawn in GREEN, with each corner's distance from where it should be. Move /
turn the camera until green sits on red (all corners < OK_PX), then fix the camera firmly.

Move the arm off the mat edges first (parked or folded at the front edge is fine: the front edge is not used).

    python so101_real/align_top_camera.py            # TOP_CAM from .env (or --cam N)
    python so101_real/align_top_camera.py --cam 3
Keys: q / Esc = quit, s = save a snapshot (align_top_camera_snapshot.jpg)
"""
import os
import sys
import json
import argparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from real_env import TOP_CAM_CALIB                                          # noqa: E402
from real_robot_policy import open_camera, crop_to_4_3, TOP_CAM, TOP_RES   # noqa: E402

OK_PX = 6           # corner error (calibration image pixels, ~0.5 mm each on the mat) counted as aligned
SEARCH_PX = 160     # how far from the red outline an edge is looked for
WINDOW = "align top camera: move it until GREEN sits on RED (q = quit)"
CORNERS = ["front-left", "front-right", "back-right", "back-left"]
# mat edges used: (corner a, corner b) - the front edge is left out (the arm and the dark floor are there)
EDGES = {"back": ("back-left", "back-right"), "left": ("front-left", "back-left"),
         "right": ("front-right", "back-right")}


def find_edge(gray, a, b):
    """The mat edge near the reference segment a-b: strongest brightness step along the normal at many points,
    then a robust line fit. Returns (normal n, offset d) with n . p = d, or None."""
    t = (b - a) / np.linalg.norm(b - a)
    n = np.array([-t[1], t[0]])
    pts = []
    for s in np.linspace(0.12, 0.88, 40):          # stay away from the rounded corners
        p0 = a + s * (b - a)
        offs = np.arange(-SEARCH_PX, SEARCH_PX + 1)
        xy = np.round(p0[None] + offs[:, None] * n[None]).astype(int)
        ok = (xy[:, 0] >= 0) & (xy[:, 0] < gray.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < gray.shape[0])
        if ok.sum() < 20:
            continue
        prof = gray[xy[ok, 1], xy[ok, 0]].astype(float)
        step = np.abs(np.convolve(prof, [1, 1, 1, 0, -1, -1, -1], "same"))
        i = int(np.argmax(step[3:-3])) + 3
        if step[i] > 60:                            # dark mat vs light table: a clear edge
            pts.append(p0 + offs[ok][i] * n)
    if len(pts) < 12:
        return None
    pts = np.array(pts, np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    nn = np.array([-vy, vx])
    return nn, float(nn @ [x0, y0])


def intersect(l1, l2):
    A = np.array([l1[0], l2[0]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return np.linalg.solve(A, [l1[1], l2[1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=TOP_CAM, help=f"top camera index (default {TOP_CAM})")
    args = ap.parse_args()

    with open(TOP_CAM_CALIB) as f:
        calib = json.load(f)
    ref = {k: np.array(v, float) for k, v in calib["corners_px"].items()}
    cap = open_camera(args.cam, *TOP_RES)
    print(f"reference: {TOP_CAM_CALIB} (created {calib.get('created', '?')}); q = quit, s = snapshot")
    shown = None
    while True:
        for _ in range(2):
            cap.grab()
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("camera read failed")
        img = crop_to_4_3(frame)
        if (img.shape[1], img.shape[0]) != tuple(calib["image_size"]):
            raise SystemExit(f"camera gives {img.shape[1]}x{img.shape[0]} after the 4:3 crop, the calibration was "
                             f"{calib['image_size']}: wrong camera index or resolution?")
        gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        lines = {e: find_edge(gray, ref[a], ref[b]) for e, (a, b) in EDGES.items()}

        view = img.copy()
        cv2.polylines(view, [np.array([ref[c] for c in CORNERS], np.int32)], True, (0, 0, 255), 3)
        found = {}
        if lines["back"] and lines["left"]:
            found["back-left"] = intersect(lines["back"], lines["left"])
        if lines["back"] and lines["right"]:
            found["back-right"] = intersect(lines["back"], lines["right"])
        for e, (a, b) in EDGES.items():           # green: the detected edges, drawn along the red ones
            if lines[e] is None:
                continue
            nn, d = lines[e]
            pa, pb = [ref[c] - (nn @ ref[c] - d) * nn for c in (a, b)]
            cv2.line(view, tuple(pa.astype(int)), tuple(pb.astype(int)), (0, 255, 0), 2)
        errs = []
        for c, p in found.items():
            if p is None:
                continue
            e = float(np.linalg.norm(p - ref[c]))
            errs.append(e)
            cv2.circle(view, tuple(p.astype(int)), 10, (0, 255, 0), 2)
            cv2.arrowedLine(view, tuple(p.astype(int)), tuple(ref[c].astype(int)), (0, 255, 255), 2, tipLength=0.3)
            cv2.putText(view, f"{c}: {e:.0f} px", (int(ref[c][0]) - 80, int(ref[c][1]) - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        angles = []                                 # side edges: rotation of the camera about its axis
        for e in ("left", "right"):
            if lines[e] is not None:
                a, b = EDGES[e]
                t_ref = ref[b] - ref[a]
                nn = lines[e][0]
                t_now = np.array([nn[1], -nn[0]]) * np.sign(np.dot([nn[1], -nn[0]], t_ref))
                angles.append(np.degrees(np.arctan2(t_ref[0] * t_now[1] - t_ref[1] * t_now[0], t_ref @ t_now)))
        if len(errs) == 2:
            good = max(errs) < OK_PX
            msg = (f"ALIGNED (corners {errs[0]:.0f} / {errs[1]:.0f} px) - fix the camera now" if good else
                   f"move the camera: corners off by {errs[0]:.0f} / {errs[1]:.0f} px (aim < {OK_PX})")
            if angles:
                msg += f"   rotation {np.mean(angles):+.1f} deg"
            color = (0, 255, 0) if good else (0, 165, 255)
        else:
            missing = [e for e, l in lines.items() if l is None]
            msg, color = f"mat edge not found: {', '.join(missing)} (arm in the way? light?)", (0, 0, 255)
        cv2.rectangle(view, (0, 0), (view.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(view, msg, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        if msg != shown:
            print(msg)
            shown = msg
        small = cv2.resize(view, (view.shape[1] * 2 // 3, view.shape[0] * 2 // 3), interpolation=cv2.INTER_AREA)
        cv2.imshow(WINDOW, small)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            out = os.path.join(HERE, "align_top_camera_snapshot.jpg")
            cv2.imwrite(out, view)
            print("saved", out)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
