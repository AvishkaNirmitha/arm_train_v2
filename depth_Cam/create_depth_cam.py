"""Stereo depth camera from two identical USB webcams (UGREEN CM678, fixed focus).

Run from the project root with the venv active (source .venv/bin/activate). Steps, in order:

  0. python depth_Cam/create_depth_cam.py board
         -> depth_Cam/chessboard_A4.pdf. Print at 100 % ("actual size", NOT "fit to page"), tape it flat on
            something rigid (cardboard / clipboard), then measure one square with a ruler (mm).
  1. python depth_Cam/create_depth_cam.py preview
         -> both cams side by side. Cover one lens with a finger: the camera on the LEFT (seen from behind the
            cameras, looking where they look) must be the LEFT panel. If not: add  --left 6 --right 4  to every step.
  2. python depth_Cam/create_depth_cam.py capture
         -> SPACE saves a pair when the board is found in BOTH images (green). Collect 20-30 pairs:
            near / far, left / centre / right, tilted ~30 deg up/down/sideways. Hold the board STILL when saving
            (the two cams are not hardware-synced).
  3. python depth_Cam/create_depth_cam.py calibrate --square 25.0      (your measured square size, mm)
         -> depth_Cam/stereo_calib.npz + rectified_check.jpg. Want: stereo RMS < 0.5 px, baseline = ruler value.
  4. python depth_Cam/create_depth_cam.py depth
         -> live depth. Click on the image: prints the distance (mm) at that point. s = save, q = quit.

After calibration the two cameras must NOT move relative to each other (rigid mount). If they do: redo 2-3.
Other code can use the result:  from create_depth_cam import StereoDepth  (see the class below).
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS_DIR = os.path.join(HERE, "calib_pairs")
CALIB_FILE = os.path.join(HERE, "stereo_calib.npz")
SAVE_DIR = os.path.join(HERE, "captures")
BOARD = (9, 6)            # inner corners (cols, rows) = 10 x 7 squares; odd x even -> corner order is unambiguous
CB_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)


# ----------------------------------------------------------------------------------------------- cameras
def open_cam(index, w, h, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"cannot open /dev/video{index}  (check: v4l2-ctl --list-devices)")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # MJPG: two cams share one USB hub
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if got != (w, h):
        print(f"warning: /dev/video{index} gives {got[0]}x{got[1]}, asked {w}x{h}")
    return cap


class StereoCams:
    def __init__(self, left, right, res, fps):
        self.L = open_cam(left, *res, fps)
        self.R = open_cam(right, *res, fps)

    def read(self):
        # grab both, then decode: the two exposures are as close in time as software allows
        if not (self.L.grab() and self.R.grab()):
            return None, None
        okL, l = self.L.retrieve()
        okR, r = self.R.retrieve()
        return (l, r) if (okL and okR) else (None, None)

    def release(self):
        self.L.release()
        self.R.release()


def find_board(gray, fast=False):
    flags = CB_FLAGS | (cv2.CALIB_CB_FAST_CHECK if fast else 0)
    ok, c = cv2.findChessboardCorners(gray, BOARD, flags)
    if ok and not fast:
        c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), SUBPIX)
    return ok, c


# ----------------------------------------------------------------------------------------------- depth
class StereoDepth:
    """Rectify a left/right pair and turn it into a depth image (mm, 0 = unknown)."""

    def __init__(self, calib_file=CALIB_FILE, num_disp=128, block=7, uniq=10):
        d = np.load(calib_file)
        self.size = tuple(int(v) for v in d["size"])
        self.P1 = d["P1"]
        self.f = float(d["P1"][0, 0])                         # rectified focal length, px
        self.baseline = float(-d["P2"][0, 3] / d["P2"][0, 0])  # mm
        self.mapL = cv2.initUndistortRectifyMap(d["K1"], d["D1"], d["R1"], d["P1"], self.size, cv2.CV_16SC2)
        self.mapR = cv2.initUndistortRectifyMap(d["K2"], d["D2"], d["R2"], d["P2"], self.size, cv2.CV_16SC2)
        self.set_matcher(num_disp, block, uniq)

    def set_matcher(self, num_disp, block, uniq):
        num_disp = max(16, int(num_disp) // 16 * 16)
        block = max(3, int(block) | 1)
        self.num_disp = num_disp
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disp, blockSize=block,
            P1=8 * block * block, P2=32 * block * block, disp12MaxDiff=1, uniquenessRatio=int(uniq),
            speckleWindowSize=100, speckleRange=2, preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def rectify(self, left, right):
        l = cv2.remap(left, *self.mapL, cv2.INTER_LINEAR)
        r = cv2.remap(right, *self.mapR, cv2.INTER_LINEAR)
        return l, r

    def compute(self, left, right):
        """-> (rectified left BGR, depth mm float32 (0 = unknown), disparity px float32)."""
        l, r = self.rectify(left, right)
        gl = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        disp = self.sgbm.compute(gl, gr).astype(np.float32) / 16.0
        depth = np.zeros_like(disp)
        ok = disp > 0.5
        depth[ok] = self.f * self.baseline / disp[ok]
        return l, depth, disp

    def point_3d(self, u, v, depth, win=3):
        """(X, Y, Z) mm in the rectified LEFT camera frame at pixel (u, v), median of a small window; None if unknown."""
        patch = depth[max(0, v - win):v + win + 1, max(0, u - win):u + win + 1]
        patch = patch[patch > 0]
        if patch.size == 0:
            return None
        z = float(np.median(patch))
        fx, fy, cx, cy = self.P1[0, 0], self.P1[1, 1], self.P1[0, 2], self.P1[1, 2]
        return (u - cx) * z / fx, (v - cy) * z / fy, z


# ----------------------------------------------------------------------------------------------- steps
def cmd_board(args):
    from PIL import Image
    dpi, sq_mm = 300, args.square
    mm = lambda v: int(round(v * dpi / 25.4))
    W, H = mm(297), mm(210)                                  # A4 landscape
    nx, ny = BOARD[0] + 1, BOARD[1] + 1
    if nx * sq_mm > 277 or ny * sq_mm > 190:
        sys.exit(f"--square {sq_mm} mm does not fit on A4 ({nx}x{ny} squares)")
    img = np.full((H, W), 255, np.uint8)
    s = mm(sq_mm)
    x0, y0 = (W - nx * s) // 2, (H - ny * s) // 2
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                img[y0 + j * s:y0 + (j + 1) * s, x0 + i * s:x0 + (i + 1) * s] = 0
    cv2.putText(img, f"{BOARD[0]}x{BOARD[1]} inner corners, {sq_mm} mm squares - print at 100%, measure!",
                (x0, y0 + ny * s + mm(8)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
    out = os.path.join(HERE, "chessboard_A4.pdf")
    Image.fromarray(img).save(out, "PDF", resolution=dpi)
    print(f"wrote {out}\nprint at 100 % / actual size (not 'fit to page'), then measure a square with a ruler.")


def cmd_preview(args):
    cams = StereoCams(args.left, args.right, args.res, args.fps)
    t0, n = time.time(), 0
    while True:
        l, r = cams.read()
        if l is None:
            print("frame grab failed")
            break
        n += 1
        fps = n / (time.time() - t0)
        both = np.hstack([l, r])
        for y in range(0, both.shape[0], 40):                 # equal-height check: same object on same line
            cv2.line(both, (0, y), (both.shape[1], y), (0, 255, 0), 1)
        cv2.putText(both, f"LEFT /dev/video{args.left}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(both, f"RIGHT /dev/video{args.right}", (l.shape[1] + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        cv2.putText(both, f"{fps:.1f} fps  q = quit", (10, both.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
        cv2.imshow("stereo preview", both)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cams.release()
    cv2.destroyAllWindows()


def cmd_capture(args):
    os.makedirs(PAIRS_DIR, exist_ok=True)
    idx = len(glob.glob(os.path.join(PAIRS_DIR, "left_*.png")))
    cams = StereoCams(args.left, args.right, args.res, args.fps)
    print(f"saving into {PAIRS_DIR} (already {idx} pairs). SPACE = save (board green in both), q = quit")
    while True:
        l, r = cams.read()
        if l is None:
            print("frame grab failed")
            break
        okL, cL = find_board(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), fast=True)
        okR, cR = find_board(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY), fast=True)
        vl, vr = l.copy(), r.copy()
        cv2.drawChessboardCorners(vl, BOARD, cL, okL)
        cv2.drawChessboardCorners(vr, BOARD, cR, okR)
        both = np.hstack([vl, vr])
        col = (0, 255, 0) if (okL and okR) else (0, 0, 255)
        cv2.putText(both, f"pairs: {idx}   {'READY - SPACE' if okL and okR else 'board not in both'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cv2.imshow("stereo capture", both)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            if okL and okR:
                cv2.imwrite(os.path.join(PAIRS_DIR, f"left_{idx:03d}.png"), l)
                cv2.imwrite(os.path.join(PAIRS_DIR, f"right_{idx:03d}.png"), r)
                print(f"saved pair {idx}")
                idx += 1
            else:
                print("not saved: board must be found in BOTH images")
    cams.release()
    cv2.destroyAllWindows()


def cmd_calibrate(args):
    files = sorted(glob.glob(os.path.join(PAIRS_DIR, "left_*.png")))
    objp = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2) * args.square
    obj, ptsL, ptsR, names, size = [], [], [], [], None
    for fL in files:
        fR = fL.replace("left_", "right_")
        gL, gR = cv2.imread(fL, cv2.IMREAD_GRAYSCALE), cv2.imread(fR, cv2.IMREAD_GRAYSCALE)
        if gL is None or gR is None:
            continue
        size = gL.shape[::-1]
        okL, cL = find_board(gL)
        okR, cR = find_board(gR)
        if not (okL and okR):
            print(f"skip {os.path.basename(fL)}: board not found in both")
            continue
        obj.append(objp); ptsL.append(cL); ptsR.append(cR); names.append(os.path.basename(fL))
    if len(obj) < 10:
        sys.exit(f"only {len(obj)} usable pairs in {PAIRS_DIR}; capture at least 15-20")
    print(f"{len(obj)} pairs, image size {size[0]}x{size[1]}, square {args.square} mm")

    rmsL, K1, D1, _, _, _, _, errL = cv2.calibrateCameraExtended(obj, ptsL, size, None, None)
    rmsR, K2, D2, _, _, _, _, errR = cv2.calibrateCameraExtended(obj, ptsR, size, None, None)
    print(f"left  intrinsics RMS {rmsL:.3f} px   fx={K1[0, 0]:.1f}")
    print(f"right intrinsics RMS {rmsR:.3f} px   fx={K2[0, 0]:.1f}")
    worst = sorted(zip(np.maximum(errL.ravel(), errR.ravel()), names), reverse=True)[:5]
    print("worst pairs (delete left_/right_ files of any > 1.0 px and re-run):",
          ", ".join(f"{n} {e:.2f}" for e, n in worst))

    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj, ptsL, ptsR, K1, D1, K2, D2, size, criteria=SUBPIX, flags=cv2.CALIB_FIX_INTRINSIC)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, alpha=0)
    baseline = float(np.linalg.norm(T))
    print(f"\nstereo RMS {rms:.3f} px   (good < 0.5, usable < 1.0)")
    print(f"baseline {baseline:.1f} mm   <- compare with a ruler (lens centre to lens centre)")
    print(f"T = {T.ravel().round(1)} mm")
    if T[0, 0] > 0:
        print("WARNING: T[0] > 0 -> left and right are swapped. Re-capture with --left/--right exchanged.")
    np.savez(CALIB_FILE, K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             size=np.array(size), square=args.square, rms=rms)
    print(f"wrote {CALIB_FILE}")

    sd = StereoDepth(CALIB_FILE)
    l, r = sd.rectify(cv2.imread(files[0]), cv2.imread(files[0].replace("left_", "right_")))
    both = np.hstack([l, r])
    for y in range(0, both.shape[0], 30):
        cv2.line(both, (0, y), (both.shape[1], y), (0, 255, 0), 1)
    out = os.path.join(HERE, "rectified_check.jpg")
    cv2.imwrite(out, both)
    print(f"wrote {out}  (the same board corner must sit on the same green line in both halves)")


def cmd_depth(args):
    if not os.path.exists(CALIB_FILE):
        sys.exit(f"no {CALIB_FILE}: run capture + calibrate first")
    sd = StereoDepth(CALIB_FILE, args.num_disp, args.block)
    if tuple(args.res) != sd.size:
        print(f"note: calibration was at {sd.size[0]}x{sd.size[1]}, using that")
    cams = StereoCams(args.left, args.right, sd.size, args.fps)
    zmin = sd.f * sd.baseline / sd.num_disp
    print(f"f={sd.f:.1f}px  baseline={sd.baseline / 10:.1f}cm  closest measurable ~{zmin / 10:.0f} cm "
          f"(raise numDisp/16 to see closer)")
    print("click = distance at point (cm), s = save, q = quit")

    win = "stereo depth (left rectified | depth)"
    cv2.namedWindow(win)
    cv2.createTrackbar("numDisp/16", win, sd.num_disp // 16, 16, lambda v: None)
    cv2.createTrackbar("block", win, args.block, 21, lambda v: None)
    cv2.createTrackbar("uniq", win, 10, 30, lambda v: None)
    clicked = {"pt": None}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            clicked["pt"] = (x % sd.size[0], y)
    cv2.setMouseCallback(win, on_mouse)

    params = None
    os.makedirs(SAVE_DIR, exist_ok=True)
    while True:
        p = (max(1, cv2.getTrackbarPos("numDisp/16", win)) * 16, cv2.getTrackbarPos("block", win),
             cv2.getTrackbarPos("uniq", win))
        if p != params:
            sd.set_matcher(*p)
            params = p
        l, r = cams.read()
        if l is None:
            print("frame grab failed")
            break
        rect, depth, _ = sd.compute(l, r)
        vis = np.clip((depth - args.min_mm) / (args.max_mm - args.min_mm), 0, 1)
        vis = cv2.applyColorMap((255 - vis * 255).astype(np.uint8), cv2.COLORMAP_JET)   # red = near
        vis[depth <= 0] = 0
        if clicked["pt"]:
            u, v = clicked["pt"]
            p3 = sd.point_3d(u, v, depth)
            txt = ("no depth here" if p3 is None else
                   f"Z={p3[2] / 10:.1f}cm  X={p3[0] / 10:.1f} Y={p3[1] / 10:.1f}")
            for img in (rect, vis):
                cv2.drawMarker(img, (u, v), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(win, np.hstack([rect, vis]))
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(SAVE_DIR, f"{stamp}_left.png"), rect)
            np.save(os.path.join(SAVE_DIR, f"{stamp}_depth_mm.npy"), depth)
            print(f"saved {SAVE_DIR}/{stamp}_*")
    cams.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["board", "preview", "capture", "calibrate", "depth"])
    ap.add_argument("--left", type=int, default=4, help="/dev/videoN of the LEFT camera")
    ap.add_argument("--right", type=int, default=6, help="/dev/videoN of the RIGHT camera")
    ap.add_argument("--res", default="640x480", help="WIDTHxHEIGHT (same for capture, calibrate, depth)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--square", type=float, default=25.0, help="chessboard square size in mm (measured)")
    ap.add_argument("--num-disp", type=int, default=128, help="disparity search range, multiple of 16")
    ap.add_argument("--block", type=int, default=7, help="SGBM block size (odd)")
    ap.add_argument("--min-mm", type=float, default=200, help="depth colour scale near end")
    ap.add_argument("--max-mm", type=float, default=1500, help="depth colour scale far end")
    args = ap.parse_args()
    args.res = tuple(int(v) for v in args.res.lower().split("x"))
    {"board": cmd_board, "preview": cmd_preview, "capture": cmd_capture,
     "calibrate": cmd_calibrate, "depth": cmd_depth}[args.step](args)


if __name__ == "__main__":
    main()
