"""Capture tool: live wrist + top camera views; Save stores both camera images and the arm's joint angles.

Run in .venv; for joint angles arm_server.py must be running in .venv-lerobot (without it, or with --no-arm,
images are saved and the joints are left empty):

    python so101_real/capture_ui.py                                # wrist camera 5, top camera 2
    python so101_real/capture_ui.py --wrist-cam 4 --top-cam 2
    python so101_real/capture_ui.py --no-arm --out my_captures

Each Save writes three files to so101_real/captures/ (or --out):
    <name>.json          joint angles (deg), raw motor positions (encoder steps) + capture info
    <name>_wrist.jpg     wrist camera image (the selected area, or the full frame)
    <name>_top.jpg       top camera image (the selected area, or the full frame)

The window (OpenCV - Tk crashes with this Python build's display setup):
    file name box : pre-filled with the current time (hh-mm-ss-dd-mm-yyyy), keeps ticking until you edit it.
                    Click it to type; Backspace deletes, Esc stops editing, Enter saves.
    Save          : click (or press Enter / s)
    Free arm      : click twice to switch torque off (move the arm by hand), click again to hold
    camera area   : the camera clicked last is "active" (orange title). Right-click a camera = full frame again.
      box         : drag on a camera = new box; drag inside = move; drag the round handle or [ ] { } = rotate.
                    Saved image = the box content rotated upright.
      polygon     : p = start a polygon on the active camera, click its corners, Enter (or click the first
                    corner) = done, Backspace = remove the last corner, Esc = cancel. Afterwards drag a corner
                    to adjust it, drag inside to move it. Saved image: 4 corners = perspective-corrected
                    rectangle (click them as the image's top-left, top-right, bottom-right, bottom-left);
                    other corner counts (or --no-warp) = outside blacked out, cut to the bounding box.
    q / Esc       : quit (when not typing a name or drawing a polygon)
Areas are kept in <out>/crop_areas.json.
"""
import os
import re
import sys
import json
import math
import time
import argparse
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from real_arm_client import ArmClient, DISABLED  # noqa: E402

TIME_FORMAT = "%H-%M-%S-%d-%m-%Y"
WIN = "SO-101 capture"
PW, PH = 448, 336                     # preview size of each camera
PAD, TITLE_H = 16, 34
W = 2 * PW + 3 * PAD
BOX_Y = TITLE_H + PH + 2 * PAD + 54   # file name row (below the two joint lines)
H = BOX_Y + 44 + 40
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG, FG, DIM, ACCENT = (40, 40, 40), (235, 235, 235), (150, 150, 150), (0, 200, 255)
GREEN = (0, 255, 0)

NAME_BOX = (PAD, BOX_Y, PAD + 2 * PW - 120, BOX_Y + 40)          # x0, y0, x1, y1
SAVE_BTN = (NAME_BOX[2] + 12, BOX_Y, W - PAD, BOX_Y + 40)
FREE_BTN = (W - PAD - 190, TITLE_H + PH + PAD + 4, W - PAD, TITLE_H + PH + PAD + 30)
PANELS = {"wrist": (PAD, TITLE_H, PAD + PW, TITLE_H + PH),
          "top": (2 * PAD + PW, TITLE_H, 2 * PAD + 2 * PW, TITLE_H + PH)}
MIN_AREA_PX = 12      # smaller drags are treated as clicks (window pixels)
HANDLE_GAP = 22       # rotation handle distance above a box (window pixels)
HANDLE_R = 7          # handle / corner grab radius (window pixels)


def timestamp():
    return datetime.now().strftime(TIME_FORMAT)


def inside(box, x, y):
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def open_camera(index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"camera {index} could not be opened (another program using it? `ls /dev/video*`)")
    return cap


# ── areas, in full-frame pixels ─────────────────────────────────────────────
#   box     : {"type": "box", "center": [cx, cy], "size": [w, h], "angle_deg": a}  (a = clockwise on screen)
#   polygon : {"type": "polygon", "points": [[x, y], ...]}
def area_corners(a):
    if a["type"] == "polygon":
        return np.array(a["points"], float)
    (cx, cy), (w, h), t = a["center"], a["size"], math.radians(a["angle_deg"])
    c, s = math.cos(t), math.sin(t)
    return np.array([(cx + dx * c - dy * s, cy + dx * s + dy * c)
                     for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))])


def area_contains(a, x, y):
    return cv2.pointPolygonTest(area_corners(a).astype(np.float32), (float(x), float(y)), False) >= 0


def warps(a, no_warp=False):
    return a["type"] == "box" or (len(a["points"]) == 4 and not no_warp)


def crop_area(frame, a, no_warp=False):
    """Box: content rotated upright. 4-corner polygon: perspective-corrected rectangle (corner order = the
    image's top-left, top-right, bottom-right, bottom-left). Other polygons: masked bounding-box cut."""
    pts = area_corners(a).astype(np.float32)
    if warps(a, no_warp):
        w = (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])) / 2
        h = (np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])) / 2
        w, h = max(int(round(w)), 1), max(int(round(h)), 1)
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(pts, dst), (w, h), flags=cv2.INTER_LINEAR)
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(mask, [pts.round().astype(np.int32)], 255)
    x, y, w, h = cv2.boundingRect(pts.round().astype(np.int32))
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    return masked[y:y + h, x:x + w]


def load_area(v):
    """crop_areas.json entry -> area. Also reads the earlier formats ([x0, y0, x1, y1] and box without type)."""
    if not v:
        return None
    if isinstance(v, dict):
        if v.get("type") == "polygon":
            return {"type": "polygon", "points": [list(map(float, p)) for p in v["points"]]}
        return {"type": "box", "center": list(v["center"]), "size": list(v["size"]),
                "angle_deg": float(v["angle_deg"])}
    x0, y0, x1, y1 = v
    return {"type": "box", "center": [(x0 + x1) / 2, (y0 + y1) / 2], "size": [x1 - x0, y1 - y0], "angle_deg": 0.0}


def copy_area(a):
    return json.loads(json.dumps(a)) if a else None


class CaptureUI:
    def __init__(self, args):
        self.args = args
        self.out = os.path.abspath(args.out)
        os.makedirs(self.out, exist_ok=True)
        self.caps = {"wrist": open_camera(args.wrist_cam), "top": open_camera(args.top_cam)}
        self.frames = {"wrist": None, "top": None}
        self.arm = None
        if not args.no_arm:
            try:
                self.arm = ArmClient()
            except SystemExit as e:   # server not running: still capture images
                print(f"(no arm: {e})")
        self.joints = None
        self.raw = None                 # raw servo positions (encoder steps)
        self.free = False
        self.free_armed_until = 0.0     # first click on "Free arm" arms it, a second click within 3 s confirms
        self.name = timestamp()
        self.name_edited = False
        self.typing = False
        self.status = f"saving to {self.out}" + ("" if self.arm else "   (arm not connected: no joint angles)")
        self.last_joint_read = 0.0
        self.running = True
        self.areas_file = os.path.join(self.out, "crop_areas.json")
        self.areas = {"wrist": None, "top": None}       # None = full frame
        if os.path.exists(self.areas_file):
            with open(self.areas_file) as f:
                self.areas.update({k: load_area(v) for k, v in json.load(f).items() if k in self.areas})
        self.drag = None                # {"cam", "mode": new / move / rotate / vertex, "start", "area", ...}
        self.active = "top"             # camera the keys apply to
        self.poly = None                # polygon being drawn: {"cam", "points": [...]} (frame pixels)
        self.mouse = None               # last mouse position (window pixels), for the rubber-band line

    # ── coordinates ─────────────────────────────────────────────────────────
    def frame_size(self, cam):
        f = self.frames[cam]
        return (f.shape[1], f.shape[0]) if f is not None else (640, 480)

    def scale(self, cam):
        """Frame pixels per window pixel."""
        return self.frame_size(cam)[0] / PW

    def panel_at(self, x, y):
        return next((cam for cam, box in PANELS.items() if inside(box, x, y)), None)

    def to_frame(self, cam, x, y):
        """Window pixel -> pixel in that camera's full frame (clamped to the frame)."""
        x0, y0 = PANELS[cam][:2]
        fw, fh = self.frame_size(cam)
        return (float(np.clip((x - x0) * fw / PW, 0, fw)), float(np.clip((y - y0) * fh / PH, 0, fh)))

    def to_panel(self, cam, x, y):
        x0, y0 = PANELS[cam][:2]
        fw, fh = self.frame_size(cam)
        return (int(round(x0 + x * PW / fw)), int(round(y0 + y * PH / fh)))

    def handle_pos(self, cam, a):
        """Box rotation handle (frame pixels): above the middle of the top edge, turning with the box."""
        t = math.radians(a["angle_deg"])
        d = a["size"][1] / 2 + HANDLE_GAP * self.scale(cam)
        return a["center"][0] + d * math.sin(t), a["center"][1] - d * math.cos(t)

    def set_area(self, cam, area, save=True):
        self.areas[cam] = area
        if save:
            with open(self.areas_file, "w") as f:
                json.dump(self.areas, f, indent=2)
        if area is None:
            self.status = f"{cam} camera: saving the full frame"
        elif area["type"] == "box":
            (cx, cy), (w, h) = area["center"], area["size"]
            self.status = (f"{cam} camera: box {w:.0f} x {h:.0f} px at ({cx:.0f}, {cy:.0f}), "
                           f"rotated {area['angle_deg']:.1f} deg")
        else:
            n = len(area["points"])
            how = "perspective-corrected" if warps(area, self.args.no_warp) else "masked"
            self.status = f"{cam} camera: polygon with {n} corners ({how} when saved)"

    # ── input ───────────────────────────────────────────────────────────────
    def on_mouse(self, event, x, y, *_):
        self.mouse = (x, y)
        if self.drag is not None:
            self.update_drag(event, x, y)
            return
        cam = self.panel_at(x, y)
        if self.poly is not None:                       # drawing a polygon: clicks add corners
            if event == cv2.EVENT_LBUTTONDOWN and cam == self.poly["cam"]:
                pts = self.poly["points"]
                first = self.to_panel(cam, *pts[0]) if pts else None
                if len(pts) >= 3 and math.hypot(x - first[0], y - first[1]) <= HANDLE_R + 4:
                    self.finish_polygon()
                else:
                    pts.append(list(self.to_frame(cam, x, y)))
                    self.status = f"polygon: {len(pts)} corners - Enter or click the first corner to finish"
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.cancel_polygon()
            return
        if event == cv2.EVENT_RBUTTONDOWN and cam:
            self.active = cam
            self.set_area(cam, None)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if cam:
            self.typing = False
            self.active = cam
            p = self.to_frame(cam, x, y)
            a = self.areas[cam]
            self.drag = {"cam": cam, "mode": "new", "start": p, "start_win": (x, y), "area": copy_area(a)}
            if a is not None:
                corners = [self.to_panel(cam, *c) for c in area_corners(a)]
                near = [i for i, (cx, cy) in enumerate(corners) if math.hypot(x - cx, y - cy) <= HANDLE_R + 4]
                if a["type"] == "box":
                    hx, hy = self.to_panel(cam, *self.handle_pos(cam, a))
                    if math.hypot(x - hx, y - hy) <= HANDLE_R + 4:
                        self.drag["mode"] = "rotate"
                elif near:
                    self.drag.update(mode="vertex", index=near[0])
                if self.drag["mode"] == "new" and area_contains(a, *p):
                    self.drag["mode"] = "move"
        elif self.arm is not None and inside(FREE_BTN, x, y):
            self.click_free()
        elif inside(NAME_BOX, x, y):
            self.typing = True
        elif inside(SAVE_BTN, x, y):
            self.typing = False
            self.save()
        else:
            self.typing = False

    def update_drag(self, event, x, y):
        d = self.drag
        cam = d["cam"]
        bx0, by0, bx1, by1 = PANELS[cam]
        x, y = int(np.clip(x, bx0, bx1)), int(np.clip(y, by0, by1))
        p = self.to_frame(cam, x, y)
        done = event == cv2.EVENT_LBUTTONUP
        a = copy_area(d["area"])
        if d["mode"] == "new":
            (sx, sy), (swx, swy) = d["start"], d["start_win"]
            if abs(x - swx) >= MIN_AREA_PX and abs(y - swy) >= MIN_AREA_PX:
                a = {"type": "box", "center": [(sx + p[0]) / 2, (sy + p[1]) / 2],
                     "size": [abs(p[0] - sx), abs(p[1] - sy)], "angle_deg": 0.0}
        elif d["mode"] == "move":
            dx, dy = p[0] - d["start"][0], p[1] - d["start"][1]
            if a["type"] == "box":
                a["center"] = [a["center"][0] + dx, a["center"][1] + dy]
            else:
                a["points"] = [[px + dx, py + dy] for px, py in a["points"]]
        elif d["mode"] == "vertex":
            a["points"][d["index"]] = list(p)
        else:   # rotate: the box handle points from the centre towards the mouse
            vx, vy = p[0] - a["center"][0], p[1] - a["center"][1]
            if math.hypot(vx, vy) > 1:
                a["angle_deg"] = round((math.degrees(math.atan2(vx, -vy)) + 180) % 360 - 180, 1)
        self.set_area(cam, a, save=done)
        if done:
            self.drag = None

    def start_polygon(self):
        self.poly = {"cam": self.active, "points": []}
        self.status = f"polygon on the {self.active} camera: click its corners (4 = perspective-corrected), Esc cancels"

    def finish_polygon(self):
        cam, pts = self.poly["cam"], self.poly["points"]
        self.poly = None
        if len(pts) < 3:
            self.status = "a polygon needs at least 3 corners - cancelled"
            return
        self.set_area(cam, {"type": "polygon", "points": pts})

    def cancel_polygon(self):
        self.poly = None
        self.status = "polygon cancelled"

    def rotate_active(self, step):
        a = self.areas[self.active]
        if a is None or a["type"] != "box":
            self.status = f"{self.active} camera: rotation is for boxes (drag on the camera view to draw one)"
            return
        self.set_area(self.active, dict(a, angle_deg=round((a["angle_deg"] + step + 180) % 360 - 180, 1)))

    def on_key(self, key):
        if key < 0:
            return
        k = key & 0xFF
        if self.typing:
            if k in (13, 10):
                self.typing = False
                self.save()
            elif k == 27:
                self.typing = False
            elif k in (8, 127):
                self.name, self.name_edited = self.name[:-1], True
            elif 32 <= k < 127 and len(self.name) < 80:
                self.name, self.name_edited = self.name + chr(k), True
            return
        if self.poly is not None:
            if k in (13, 10):
                self.finish_polygon()
            elif k == 27:
                self.cancel_polygon()
            elif k in (8, 127) and self.poly["points"]:
                self.poly["points"].pop()
            return
        rotate = {ord("["): -1.0, ord("]"): 1.0, ord("{"): -5.0, ord("}"): 5.0}
        if k in rotate:
            self.rotate_active(rotate[k])
        elif k == ord("p"):
            self.start_polygon()
        elif k in (13, 10, ord("s")):
            self.save()
        elif k in (27, ord("q")):
            self.running = False

    def click_free(self):
        now = time.time()
        if not self.free:
            if now > self.free_armed_until:
                self.free_armed_until = now + 3.0
                self.status = "Free arm: HOLD THE ARM, then click again within 3 s to switch torque off"
                return
            self.arm.free(True)
            self.free = True
            self.status = "arm is FREE (torque off): move it by hand"
        else:
            self.arm.free(False)
            self.free = False
            self.status = "arm holds its pose (torque on)"
        self.free_armed_until = 0.0

    # ── saving ──────────────────────────────────────────────────────────────
    def save(self):
        if any(f is None for f in self.frames.values()):
            self.status = "no camera image yet"
            return
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", self.name.strip()) or timestamp()
        base, n = name, 1
        while os.path.exists(os.path.join(self.out, f"{name}.json")):   # never overwrite a capture
            name, n = f"{base}_{n}", n + 1
        joints = raw = None
        if self.arm is not None:
            try:
                r = self.arm.read()                  # read at the moment of saving
                joints, raw = r["joints"], r.get("raw")
            except Exception as e:
                print(f"joint read failed: {e}")
        images, sizes, crops = {}, {}, {}
        for cam, frame in self.frames.items():
            a = self.areas[cam]
            img = crop_area(frame, a, self.args.no_warp) if a else frame
            images[cam] = f"{name}_{cam}.jpg"
            sizes[cam] = [img.shape[1], img.shape[0]]
            cv2.imwrite(os.path.join(self.out, images[cam]), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if a is None:
                crops[cam] = None
            else:
                crops[cam] = dict(copy_area(a), corners=area_corners(a).round(1).tolist(),
                                  saved_as="perspective_corrected" if a["type"] == "polygon" and
                                  warps(a, self.args.no_warp) else
                                  "rotated_upright" if a["type"] == "box" else "masked_bounding_box")
        h, w = self.frames["top"].shape[:2]
        record = {
            "name": name,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "joints_deg": joints,
            "joint_units": "degrees, lerobot convention (0 = middle of the calibrated range)",
            "motor_positions_raw": raw,
            "motor_units": "raw servo encoder steps (0..4095 per turn), before calibration",
            "motors_not_read": sorted(DISABLED),
            "images": images,
            "cameras": {"wrist": self.args.wrist_cam, "top": self.args.top_cam},
            "frame_size": [w, h],
            "image_size": sizes,
            "crop": crops,
            "crop_units": "full-frame pixels; box angle_deg turns it clockwise on screen; corners in order "
                          "(box: top-left, top-right, bottom-right, bottom-left); null = full frame",
        }
        with open(os.path.join(self.out, f"{name}.json"), "w") as f:
            json.dump(record, f, indent=2)
        self.status = f"saved {name}.json + 2 images" + ("" if joints else "   (no joint angles: arm not connected)")
        print(f"saved {os.path.join(self.out, name)}.json")
        self.name, self.name_edited = timestamp(), False

    # ── drawing ─────────────────────────────────────────────────────────────
    def draw_area(self, view, cam, a, x0, y0):
        poly = (np.array([self.to_panel(cam, *c) for c in area_corners(a)]) - [x0, y0]).astype(np.int32)
        mask = np.zeros((PH, PW), np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        view = np.where(mask[..., None] > 0, view, (view * 0.35).astype(np.uint8))
        cv2.polylines(view, [poly], True, GREEN, 2, cv2.LINE_AA)
        if a["type"] == "box":
            hx, hy = (np.array(self.to_panel(cam, *self.handle_pos(cam, a))) - [x0, y0]).astype(int)
            tx, ty = ((poly[0] + poly[1]) / 2).astype(int)   # middle of the top edge
            cv2.line(view, (int(tx), int(ty)), (int(hx), int(hy)), GREEN, 1, cv2.LINE_AA)
            cv2.circle(view, (int(hx), int(hy)), HANDLE_R, ACCENT, -1, cv2.LINE_AA)
            cv2.putText(view, f"{a['angle_deg']:.1f} deg", (int(hx) + 10, int(hy) + 4), FONT, 0.45, ACCENT, 1,
                        cv2.LINE_AA)
        else:
            for i, (px, py) in enumerate(poly):
                cv2.circle(view, (int(px), int(py)), HANDLE_R - 2, ACCENT, -1, cv2.LINE_AA)
                if len(poly) == 4 and not self.args.no_warp:   # corner order = saved image orientation
                    cv2.putText(view, ("TL", "TR", "BR", "BL")[i], (int(px) + 7, int(py) - 7), FONT, 0.45, ACCENT,
                                1, cv2.LINE_AA)
        return view

    def draw(self):
        img = np.full((H, W, 3), BG, np.uint8)
        for cam, title in (("wrist", f"CAM 1  Wrist   (index {self.args.wrist_cam})"),
                           ("top", f"CAM 2  Top Camera   (index {self.args.top_cam})")):
            x0, y0, x1, y1 = PANELS[cam]
            cv2.putText(img, title, (x0, TITLE_H - 10), FONT, 0.6, ACCENT if cam == self.active else FG, 1,
                        cv2.LINE_AA)
            frame = self.frames[cam]
            if frame is not None:
                view = cv2.resize(frame, (PW, PH))
                if self.areas[cam] and not (self.poly and self.poly["cam"] == cam):
                    view = self.draw_area(view, cam, self.areas[cam], x0, y0)
                if self.poly and self.poly["cam"] == cam:          # polygon being drawn
                    pts = [tuple(np.array(self.to_panel(cam, *p)) - [x0, y0]) for p in self.poly["points"]]
                    for i, p in enumerate(pts):
                        cv2.circle(view, p, HANDLE_R - 2, ACCENT, -1, cv2.LINE_AA)
                        if i:
                            cv2.line(view, pts[i - 1], p, ACCENT, 2, cv2.LINE_AA)
                    if pts and self.mouse and self.panel_at(*self.mouse) == cam:
                        cv2.line(view, pts[-1], (self.mouse[0] - x0, self.mouse[1] - y0), ACCENT, 1, cv2.LINE_AA)
                img[y0:y1, x0:x1] = view
            cv2.rectangle(img, (x0 - 1, y0 - 1), (x1, y1), DIM, 1)

        jy = TITLE_H + PH + PAD + 22
        text = ("joints (deg):  " + "   ".join(f"{m} {v:.1f}" for m, v in self.joints.items())) if self.joints \
            else "joints: -" if self.arm else "joints: arm not connected"
        cv2.putText(img, text, (PAD, jy), FONT, 0.5, FG, 1, cv2.LINE_AA)
        raw = ("motors (raw steps):  " + "   ".join(f"{m} {v}" for m, v in self.raw.items())) if self.raw \
            else "motors (raw steps): -"
        cv2.putText(img, raw, (PAD, jy + 22), FONT, 0.5, DIM, 1, cv2.LINE_AA)
        if self.arm is not None:
            armed = time.time() < self.free_armed_until
            label = "Hold arm" if self.free else ("Click again!" if armed else "Free arm")
            color = (0, 0, 200) if (self.free or armed) else (90, 90, 90)
            cv2.rectangle(img, FREE_BTN[:2], FREE_BTN[2:], color, -1)
            cv2.putText(img, label, (FREE_BTN[0] + 12, FREE_BTN[3] - 8), FONT, 0.55, FG, 1, cv2.LINE_AA)

        cv2.putText(img, "file name", (NAME_BOX[0], NAME_BOX[1] - 8), FONT, 0.5, DIM, 1, cv2.LINE_AA)
        cv2.putText(img, "(camera: drag = box, handle/[ ] = rotate, p = polygon, drag inside = move, right-click = full)",
                    (NAME_BOX[0] + 95, NAME_BOX[1] - 8), FONT, 0.38, DIM, 1, cv2.LINE_AA)
        cv2.rectangle(img, NAME_BOX[:2], NAME_BOX[2:], (25, 25, 25), -1)
        cv2.rectangle(img, NAME_BOX[:2], NAME_BOX[2:], ACCENT if self.typing else DIM, 1)
        cursor = "|" if self.typing and int(time.time() * 2) % 2 == 0 else ""
        cv2.putText(img, self.name + cursor, (NAME_BOX[0] + 10, NAME_BOX[3] - 13), FONT, 0.6, FG, 1, cv2.LINE_AA)
        cv2.rectangle(img, SAVE_BTN[:2], SAVE_BTN[2:], (0, 140, 0), -1)
        cv2.putText(img, "Save", (SAVE_BTN[0] + 28, SAVE_BTN[3] - 13), FONT, 0.7, FG, 2, cv2.LINE_AA)
        cv2.putText(img, self.status, (PAD, H - 14), FONT, 0.45, ACCENT, 1, cv2.LINE_AA)
        return img

    # ── main loop ───────────────────────────────────────────────────────────
    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        try:
            while self.running:
                for cam, cap in self.caps.items():
                    ok, frame = cap.read()
                    if ok:
                        self.frames[cam] = frame
                if self.arm is not None and time.time() - self.last_joint_read > 0.2:
                    try:
                        r = self.arm.read()
                        self.joints, self.raw = r["joints"], r.get("raw")
                    except Exception as e:
                        self.joints, self.raw, self.status = None, None, f"joint read failed: {e}"
                    self.last_joint_read = time.time()
                if not self.name_edited:
                    self.name = timestamp()
                cv2.imshow(WIN, self.draw())
                self.on_key(cv2.waitKey(15))
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.close()

    def close(self):
        if self.arm is not None:
            if self.free:
                input("Torque goes back ON at the current pose: hold the arm where it should stay, press ENTER ...")
                self.arm.free(False)
            self.arm.close()
        for cap in self.caps.values():
            cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrist-cam", type=int, default=5, help="wrist camera index (default 5)")
    ap.add_argument("--top-cam", type=int, default=2, help="top camera index (default 2)")
    ap.add_argument("--out", default=os.path.join(HERE, "captures"), help="folder for the saved files")
    ap.add_argument("--no-arm", action="store_true", help="don't connect to arm_server.py (images only)")
    ap.add_argument("--no-warp", action="store_true",
                    help="4-corner polygons: save the masked area instead of a perspective-corrected rectangle")
    CaptureUI(ap.parse_args()).run()


if __name__ == "__main__":
    main()
