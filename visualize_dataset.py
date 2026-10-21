"""Browse a pick-and-place dataset made by generate_dataset.py.

    python visualize_dataset.py                                   # newest dataset in datasets/
    python visualize_dataset.py datasets/pick_place_v1_21-10-2026-11-03-15
    python visualize_dataset.py --export viz_out --count 16       # save frames + stats.png, no window

Window layout:
    top camera (+ overlays)      | wrist camera
    TARGET pose rendered in sim  | prompt / answer / task info
Overlays on the top image: green ring = cube start, red ring = place pad,
yellow cross = gripper target for this phase.

Keys:  d / Right  next sample        a / Left  previous sample
       s / Down   next episode       w / Up    previous episode
       1..8       jump to phase      space     play / pause episode
       t          train <-> val      h         statistics page
       q / Esc    quit
"""
import os
import glob
import json
import argparse

import cv2
import numpy as np
import mujoco

from expert import PickPlaceEnv, PHASES, CUBE_HALF, GRIPPER_OPEN, GRIPPER_CLOSED
from generate_dataset import make_scene_option

FONT = cv2.FONT_HERSHEY_SIMPLEX
CARRY_PHASES = {"LIFT", "PRE_PLACE", "PLACE"}
KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = 65361, 65362, 65363, 65364


def latest_dataset(root="datasets"):
    dirs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.exists(os.path.join(d, "train.jsonl"))]
    if not dirs:
        raise SystemExit(f"no datasets found in {root}/")
    return max(dirs, key=os.path.getmtime)


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


class DatasetViewer:
    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "meta.json")) as f:
            self.meta = json.load(f)
        self.splits = {s: load_rows(os.path.join(root, f"{s}.jsonl")) for s in ("train", "val")}
        self.split, self.idx = "train", 0
        self.w, self.h = self.meta["image_size"]

        self.env = PickPlaceEnv()
        self.renderer = mujoco.Renderer(self.env.model, height=self.h, width=self.w)
        self.scene_option = make_scene_option()
        self.top_cam = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_camera")
        mujoco.mj_forward(self.env.model, self.env.data)  # camera poses are static

        # Orbit camera for the target-pose render (closer than the scene's front_camera)
        self.target_cam = mujoco.MjvCamera()
        self.target_cam.lookat[:] = [0.45, 0.0, 0.15]
        self.target_cam.distance = 1.3
        self.target_cam.azimuth = 150
        self.target_cam.elevation = -25

    @property
    def rows(self):
        return self.splits[self.split]

    # ── Geometry helpers ───────────────────────────────────────────────────
    def project_top(self, p):
        """World point -> pixel in the top camera image."""
        m, d = self.env.model, self.env.data
        R = d.cam_xmat[self.top_cam].reshape(3, 3)
        pc = R.T @ (np.asarray(p, float) - d.cam_xpos[self.top_cam])
        f = (self.h / 2) / np.tan(np.radians(m.cam_fovy[self.top_cam]) / 2)
        return int(self.w / 2 + f * pc[0] / -pc[2]), int(self.h / 2 - f * pc[1] / -pc[2])

    def render_target(self, row):
        """Put the robot at the label joints and the cube where it should be, render from the side."""
        env = self.env
        d = env.data
        d.qpos[env.arm_qadr] = np.radians(row["target_joints_deg"])
        grip = GRIPPER_OPEN if row["gripper"] == "open" else GRIPPER_CLOSED
        # A closed gripper command on a cube stops at the cube width, not at 0
        d.qpos[env.finger_qadr] = max(grip, CUBE_HALF) if row["phase"] in CARRY_PHASES | {"CLOSE"} else grip

        yaw = np.radians(row["cube_yaw_deg"])
        if row["phase"] in CARRY_PHASES:
            cube = np.array(row["target_site_pos"]) - [0, 0, 0.005]  # held in the fingers
        elif row["phase"] in ("RELEASE", "RETREAT"):
            cube = [*row["pad_xy"], CUBE_HALF]
        else:
            cube = [*row["cube_xy"], CUBE_HALF]
        d.qpos[env.box_qadr:env.box_qadr + 3] = cube
        d.qpos[env.box_qadr + 3:env.box_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.mocap_pos[env.pad_mocap] = [*row["pad_xy"], 0]
        mujoco.mj_forward(env.model, d)
        self.renderer.update_scene(d, camera=self.target_cam, scene_option=self.scene_option)
        return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)

    # ── Drawing ────────────────────────────────────────────────────────────
    def load_image(self, row, cam):
        path = next((p for p in row["images"] if p.endswith(f"_{cam}.jpg")), None)
        img = cv2.imread(os.path.join(self.root, path)) if path else None
        if img is None:
            img = np.zeros((self.h, self.w, 3), np.uint8)
            label(img, f"no {cam} image", (10, self.h // 2))
        return img

    def compose(self):
        row = self.rows[self.idx]
        top = self.load_image(row, "top")
        cv2.circle(top, self.project_top([*row["cube_xy"], CUBE_HALF]), 14, (0, 255, 0), 2)
        cv2.circle(top, self.project_top([*row["pad_xy"], 0]), 18, (0, 0, 255), 2)
        cv2.drawMarker(top, self.project_top(row["target_site_pos"]), (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
        label(top, "TOP (input)  o cube start  o pad  + target", (8, 22))

        wrist = self.load_image(row, "wrist")
        label(wrist, "WRIST (input)", (8, 22))

        target = self.render_target(row)
        label(target, f"TARGET pose = label ({row['phase']})", (8, 22))

        return np.vstack([np.hstack([top, wrist]), np.hstack([target, self.text_panel(row)])])

    def text_panel(self, row):
        panel = np.full((self.h, self.w, 3), 30, np.uint8)
        cur = np.array(row["current_joints_deg"])
        tgt = np.array(row["target_joints_deg"])
        lines = [
            (f"{self.split}  {self.idx + 1}/{len(self.rows)}   id {row['id']}", (255, 255, 255)),
            (f"Phase {row['phase_idx'] + 1}/{len(PHASES)}: {row['phase']}   gripper -> {row['gripper']}", (0, 255, 255)),
            (f"answer: {row['answer']}", (255, 255, 255)),
            ("", None),
            ("joint   current   target    delta", (180, 180, 180)),
        ]
        for i in range(7):
            lines.append((f"  j{i + 1}  {cur[i]:8.1f}  {tgt[i]:8.1f}  {tgt[i] - cur[i]:+7.1f}", (220, 220, 220)))
        lines += [
            ("", None),
            (f"cube xy {row['cube_xy']}  yaw {row['cube_yaw_deg']:.1f}", (0, 255, 0)),
            (f"pad  xy {row['pad_xy']}", (80, 80, 255)),
            (f"target site {row['target_site_pos']}", (0, 255, 255)),
        ]
        y = 22
        for text, color in lines:
            if text:
                cv2.putText(panel, text, (10, y), FONT, 0.45, color, 1, cv2.LINE_AA)
            y += 19
        cv2.putText(panel, "a/d sample  w/s episode  1-8 phase  space play  t split  h stats  q quit",
                    (10, self.h - 10), FONT, 0.36, (150, 150, 150), 1, cv2.LINE_AA)
        return panel

    def stats_page(self):
        rows = self.splits["train"] + self.splits["val"]
        joints = np.array([r["target_joints_deg"] for r in rows])
        first = [r for r in rows if r["phase_idx"] == 0]  # one row per episode
        pw, ph = 320, 230
        panels = []
        for j in range(7):
            lo, hi = np.degrees(self.env.arm_range[j])
            panels.append(histogram(joints[:, j], f"j{j + 1} target (deg)  limits [{lo:.0f}, {hi:.0f}]", pw, ph))
        panels.append(histogram(np.array([r["cube_yaw_deg"] for r in first]), "cube yaw (deg)", pw, ph))
        panels.append(workspace_scatter(first, pw, ph))
        grid = np.vstack([np.hstack(panels[i:i + 3]) for i in range(0, 9, 3)])
        header = np.full((40, grid.shape[1], 3), 30, np.uint8)
        n_train, n_val = len(self.splits["train"]), len(self.splits["val"])
        label(header, f"{os.path.basename(self.root)}:  {len(first)} episodes,  {n_train} train / {n_val} val samples"
                      f",  cams {self.meta['cameras']},  noise {self.meta['noise_deg']} deg", (10, 26))
        return np.vstack([header, grid])

    # ── Navigation ─────────────────────────────────────────────────────────
    def jump_episode(self, step):
        ep = self.rows[self.idx]["episode"]
        eps = sorted({r["episode"] for r in self.rows})
        k = int(np.clip(eps.index(ep) + step, 0, len(eps) - 1))
        self.idx = next(i for i, r in enumerate(self.rows) if r["episode"] == eps[k])

    def jump_phase(self, phase_idx):
        ep = self.rows[self.idx]["episode"]
        self.idx = next((i for i, r in enumerate(self.rows)
                         if r["episode"] == ep and r["phase_idx"] == phase_idx), self.idx)

    def run(self):
        win = "dataset viewer"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        playing, show_stats = False, False
        stats_img = None
        while True:
            if show_stats:
                stats_img = stats_img if stats_img is not None else self.stats_page()
                cv2.imshow(win, stats_img)
            else:
                cv2.imshow(win, self.compose())
            key = cv2.waitKeyEx(700 if playing else 0)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            k = key & 0xFF if key < 256 else key
            if key == -1 and playing:
                last_phase = self.rows[self.idx]["phase_idx"] == len(PHASES) - 1
                playing = not last_phase and self.idx + 1 < len(self.rows)
                if playing:
                    self.idx += 1
                continue
            if k in (ord("q"), 27):
                break
            elif k in (ord("d"), KEY_RIGHT):
                self.idx = min(self.idx + 1, len(self.rows) - 1)
            elif k in (ord("a"), KEY_LEFT):
                self.idx = max(self.idx - 1, 0)
            elif k in (ord("s"), KEY_DOWN):
                self.jump_episode(+1)
            elif k in (ord("w"), KEY_UP):
                self.jump_episode(-1)
            elif ord("1") <= k <= ord("8"):
                self.jump_phase(k - ord("1"))
            elif k == ord(" "):
                playing = not playing
                if playing:
                    self.jump_phase(0)
            elif k == ord("t"):
                self.split = "val" if self.split == "train" else "train"
                self.idx = 0
            elif k == ord("h"):
                show_stats = not show_stats
        cv2.destroyAllWindows()

    def export(self, out_dir, count):
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "stats.png"), self.stats_page())
        for self.idx in range(min(count, len(self.rows))):
            cv2.imwrite(os.path.join(out_dir, f"{self.rows[self.idx]['id']}.jpg"), self.compose())
        print(f"wrote stats.png and {min(count, len(self.rows))} frames to {out_dir}")


def label(img, text, org):
    (tw, th), base = cv2.getTextSize(text, FONT, 0.5, 1)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + base + 2), (20, 20, 20), -1)
    cv2.putText(img, text, org, FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def histogram(values, title, w, h, bins=40):
    img = np.full((h, w, 3), 30, np.uint8)
    counts, edges = np.histogram(values, bins=bins)
    x0, y0, pw, ph = 35, 30, w - 50, h - 60
    bw = pw / bins
    for i, c in enumerate(counts):
        bh = int(ph * c / counts.max())
        cv2.rectangle(img, (int(x0 + i * bw), y0 + ph - bh), (int(x0 + (i + 1) * bw) - 1, y0 + ph), (230, 160, 60), -1)
    cv2.line(img, (x0, y0 + ph), (x0 + pw, y0 + ph), (150, 150, 150), 1)
    cv2.putText(img, title, (8, 18), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{edges[0]:.1f}", (x0 - 10, h - 12), FONT, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"{edges[-1]:.1f}", (x0 + pw - 30, h - 12), FONT, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"mean {np.mean(values):.1f}  std {np.std(values):.1f}", (x0 + pw // 2 - 70, h - 12),
                FONT, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def workspace_scatter(rows, w, h):
    """Top-down cube (green) and pad (red) start positions, drawn like the top camera (+x up, +y left)."""
    img = np.full((h, w, 3), 30, np.uint8)
    cv2.putText(img, "workspace: cube (green), pad (red)", (8, 18), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    x_rng, y_rng = (0.35, 0.70), (-0.30, 0.30)
    x0, y0, pw, ph = 20, 30, w - 40, h - 50

    def to_px(xy):
        u = x0 + (y_rng[1] - xy[1]) / (y_rng[1] - y_rng[0]) * pw
        v = y0 + (x_rng[1] - xy[0]) / (x_rng[1] - x_rng[0]) * ph
        return int(u), int(v)

    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (90, 90, 90), 1)
    for r in rows:
        cv2.circle(img, to_px(r["pad_xy"]), 1, (60, 60, 255), -1)
        cv2.circle(img, to_px(r["cube_xy"]), 1, (60, 255, 60), -1)
    cv2.putText(img, "robot base below  (+x up, +y left)", (x0, h - 6), FONT, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", help="dataset dir (default: newest in datasets/)")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--export", metavar="DIR", help="save frames + stats.png instead of opening a window")
    ap.add_argument("--count", type=int, default=16, help="frames to export")
    args = ap.parse_args()

    root = args.dataset or latest_dataset()
    print(f"dataset: {root}")
    viewer = DatasetViewer(root)
    viewer.split = args.split
    if args.export:
        viewer.export(args.export, args.count)
    else:
        viewer.run()


if __name__ == "__main__":
    main()
