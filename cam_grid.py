"""
multi_cam_grid.py
-----------------
Finds every connected camera and shows them live in a grid.
Each tile is labelled with its camera index (the number you pass
to cv2.VideoCapture) and its live FPS.

Install:
    pip install opencv-python numpy

Run:
    python multi_cam_grid.py                 # probe indices 0-9
    python multi_cam_grid.py --max 20        # probe indices 0-19
    python multi_cam_grid.py --tile 480 360  # bigger tiles

Keys:
    q / Esc  -> quit
    s        -> save a snapshot of the whole grid
"""

import argparse
import math
import os
import platform
import threading
import time

# Silence OpenCV's "can't open camera by index" warnings while probing.
# Must be set BEFORE cv2 is imported.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def default_backend():
    """Pick the most reliable capture backend for this OS."""
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    if system == "Linux":
        return cv2.CAP_V4L2
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def quiet_opencv():
    """Hide OpenCV's noisy warnings for indices that don't exist."""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass


def find_cameras(max_index, backend, cap_w, cap_h):
    """Try indices 0..max_index-1 and return {index: opened VideoCapture}."""
    found = {}
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        # Lower resolution helps when several USB cams share one hub
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)
        ok, _ = cap.read()
        if ok:
            found[idx] = cap
            print(f"[+] Camera found at index {idx}")
        else:
            cap.release()
    return found


# ----------------------------------------------------------------------
# Threaded camera reader (so one slow camera doesn't freeze the grid)
# ----------------------------------------------------------------------
class CameraStream:
    def __init__(self, index, cap):
        self.index = index
        self.cap = cap
        self.frame = None
        self.last_ok = 0.0
        self.fps = 0.0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        count, t0 = 0, time.time()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            now = time.time()
            count += 1
            if now - t0 >= 1.0:
                self.fps = count / (now - t0)
                count, t0 = 0, now
            with self.lock:
                self.frame = frame
                self.last_ok = now

    def read(self):
        """Return (frame or None, is_live)."""
        with self.lock:
            if self.frame is None:
                return None, False
            live = (time.time() - self.last_ok) < 2.0
            return self.frame.copy(), live

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def letterbox(frame, w, h):
    """Resize keeping aspect ratio, pad with black to exactly w x h."""
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def draw_label(img, text, org, color=(255, 255, 255), bg=(0, 0, 0), scale=0.6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, 2)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + base), bg, -1)
    cv2.putText(img, text, (x, y - 2), font, scale, color, 2, cv2.LINE_AA)


def make_tile(stream, w, h):
    frame, live = stream.read()
    if frame is None:
        tile = np.zeros((h, w, 3), dtype=np.uint8)
        live = False
    else:
        tile = letterbox(frame, w, h)

    draw_label(tile, f"Index {stream.index}", (10, 28),
               color=(255, 255, 255), bg=(0, 120, 0) if live else (0, 0, 160))
    if live:
        draw_label(tile, f"{stream.fps:4.1f} FPS", (10, h - 10), scale=0.5)
    else:
        draw_label(tile, "NO SIGNAL", (w // 2 - 60, h // 2),
                   color=(0, 0, 255), bg=(0, 0, 0), scale=0.8)

    cv2.rectangle(tile, (0, 0), (w - 1, h - 1), (80, 80, 80), 1)
    return tile


def build_grid(tiles, cols, w, h):
    rows = math.ceil(len(tiles) / cols)
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    tiles = tiles + [blank] * (rows * cols - len(tiles))
    row_imgs = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
    return np.vstack(row_imgs)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Show all connected cameras in a grid.")
    ap.add_argument("--max", type=int, default=10, help="highest index to probe (exclusive)")
    ap.add_argument("--tile", type=int, nargs=2, default=[400, 300],
                    metavar=("W", "H"), help="size of each grid tile")
    ap.add_argument("--capture", type=int, nargs=2, default=[640, 480],
                    metavar=("W", "H"), help="resolution requested from cameras")
    args = ap.parse_args()

    quiet_opencv()
    backend = default_backend()
    tile_w, tile_h = args.tile

    print(f"Scanning camera indices 0..{args.max - 1} ...")
    caps = find_cameras(args.max, backend, *args.capture)
    if not caps:
        print("No cameras found.")
        return

    streams = [CameraStream(i, c) for i, c in caps.items()]
    cols = math.ceil(math.sqrt(len(streams)))
    print(f"Showing {len(streams)} camera(s): {list(caps.keys())}")
    print("Press 'q' or Esc to quit, 's' to save a snapshot.")

    win = "Connected Cameras"
    try:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        except cv2.error:
            print("\n[!] This OpenCV build has no GUI support (opencv-python-headless).\n"
                  "    Fix:  uv pip uninstall opencv-python opencv-python-headless\n"
                  "          uv pip install opencv-python\n")
            return

        while True:
            tiles = [make_tile(s, tile_w, tile_h) for s in streams]
            grid = build_grid(tiles, cols, tile_w, tile_h)
            cv2.imshow(win, grid)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                name = time.strftime("grid_%Y%m%d_%H%M%S.png")
                cv2.imwrite(name, grid)
                print(f"Saved {name}")
            # Close button on the window
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        for s in streams:
            s.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()