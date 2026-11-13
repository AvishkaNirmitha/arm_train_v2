"""Live check of the cube ArUco tags (make_aruco_sheet.py, any ID but 0 and 10-13) in the top camera.

    .venv/bin/python so101_real/aruco_check.py

For each tag on a block top it shows:
  * green outline + ID: the camera reads the tag (snap uses it: exact top face middle and turn)
  * red outline: a tag seen but not used (the reason is written next to it)
  * size: the tag measured on the block-top height. ~ its printed size = the calibration is right there. Much
    different = the tag is not flat on a block top, or the calibration is off at that place.
  * box fit vs tag: how far the old box fit (no tag, white blob only) is from the tag's answer, in mm
Cyan box = the block from the tag. Move the block around the mat (also near the edges) and watch the numbers.
Keys: SPACE = print the numbers, Q / ESC = quit. No arm needed.
"""
import os
import sys
import json
import argparse

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import click_goal_real as C                                           # noqa: E402  (sim scene: the top camera pose)
from real_robot_policy import open_camera, crop_to_4_3                # noqa: E402
from cam_config import TOP_CAM, TOP_RES                               # noqa: E402
from snap_top import TopCam, find_cube_tags, masks, _blobs, fit_block, _box_corners       # noqa: E402

SHOW = 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--top-cam", type=int, default=TOP_CAM)
    args = ap.parse_args()

    env = C.PickEnv(randomize=False)
    kin = C.Kin(env)
    with open(C.TOP_CAM_CALIB) as f:
        calib = json.load(f)
    TW, TH = calib["image_size"]
    cam = TopCam.from_mujoco(kin.m, kin.set(env.q_home).d, TW, TH)
    poly = np.array([calib["corners_px"][c] for c in C.align.CORNERS], float)
    cap = open_camera(args.top_cam, *TOP_RES)
    print(__doc__.split("\n\n")[0], "- SPACE = print, Q = quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        img = crop_to_4_3(frame)
        if img.shape[1] != TW:
            img = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_AREA)
        tags = find_cube_tags(img, cam, rejected=True)
        white, _ = masks(img, cam, poly)
        lines = []
        for t in tags:
            if t["why"] is not None:
                cv2.polylines(img, [np.int32(t["corners"])], True, (0, 0, 255), 3)
                cv2.putText(img, f"ID {t['id']}: {t['why']}", (int(t["corners"][:, 0].min()),
                            int(t["corners"][:, 1].min()) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                            cv2.LINE_AA)
                lines.append(f"ID {t['id']} not used: {t['why']}")
                continue
            cv2.polylines(img, [np.int32(t["corners"])], True, (0, 255, 0), 3)
            o = np.int32(np.round(cam.project(_box_corners(t["x"], t["y"], t["yaw"]))))
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)):
                cv2.line(img, tuple(o[a]), tuple(o[b]), (255, 255, 0), 2, cv2.LINE_AA)
            # the box fit on the white blob under the tag (what snap does without a tag)
            c = np.int32(np.round(t["corners"].mean(0)))
            fit = None
            for blob in _blobs(white, 50):
                if blob[0][c[1], c[0]]:
                    x, y, _yaw, q = fit_block(blob[0], cam)
                    fit = (np.hypot(x - t["x"], y - t["y"]) * 1000, q)
            txt = (f"ID {t['id']}  size {t['size_mm']:.1f} mm (printed {t['printed_mm']:g})  turn {np.degrees(t['yaw']):.0f} deg  "
                   + (f"box fit vs tag {fit[0]:.1f} mm (fit {fit[1]:.2f})" if fit else "box fit: no blob"))
            lines.append(f"{txt}   at x {t['x'] * 1000:.0f} y {t['y'] * 1000:.0f} mm")
            cv2.putText(img, f"ID {t['id']}  {t['size_mm']:.1f} mm", (int(t["corners"][:, 0].max()) + 10, c[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        view = cv2.resize(img, None, fx=SHOW, fy=SHOW, interpolation=cv2.INTER_AREA)
        msg = lines or ["no ArUco tag seen (DICT_4X4_50)"]
        for i, s in enumerate(msg):
            cv2.putText(view, s, (10, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(view, s, (10, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("aruco check (top camera)", view)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            print("\n".join(msg))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
