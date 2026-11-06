"""Motor sliders for the real SO-101: one slider per driven motor (wrist_roll is taped: never commanded).

In the "motors" window, move a motor by:
  * dragging its slider (or clicking on the bar) and letting go - it moves when you let go
  * the mouse wheel over a slider, or its - / + buttons: one step (step size at the bottom)
Only that motor moves (the others hold), at the speed set at the bottom (the gripper alone always at 60).
Before anything moves the sim arm checks the whole path: the gripper point stays above the table (MIN_TCP_Z) and
no link goes below it - otherwise the move is refused and the slider jumps back. Blocks on the mat are NOT checked.
Bright knob = commanded, small orange mark = where the motor really is (they differ e.g. when the jaws hold a block).

Standalone (.venv; arm_server.py running in .venv-lerobot):
    python so101_real/motor_panel.py             # --no-arm = try the window, nothing moves;  Q / Esc = quit
click_goal_real.py opens the same window.
(Drawn with OpenCV: tkinter in this Python crashes on text.)
"""
import os
import sys
import time
import argparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from real_arm_client import ArmClient, load_joint_map, sim_to_real, real_to_sim   # noqa: E402

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "gripper"]   # never wrist_roll
WIN = "motors"
GRIPPER_OPEN, GRIPPER_CLOSED = 93.8, 2.0
SETTLE = 0.35                # s after the last wheel / button step before it is sent (one move for several steps)
STEPS = (0.5, 1.0, 2.0, 5.0, 10.0)
SPEEDS = (5, 10, 15, 20, 30, 45, 60)
GRIP_SPEED = 60.0            # units/s when only the gripper moves (arm_server caps at 60)
LINK_MIN_Z = 0.01            # m: lowest a link origin may go (robot frame, table ~0)
BASE_R, BASE_Z = 0.09, 0.14  # m: the wrist / jaws / gripper point must stay out of this cylinder round the base
DRY_LIMITS = {"shoulder_pan": (-92.6, 92.6), "shoulder_lift": (-2.0, 195.0), "elbow_flex": (-185.0, 2.0),
              "wrist_flex": (-97.6, 97.6), "gripper": (2.0, 98.0)}   # --no-arm (arm_server reports the real ones)

# layout (px) and colours (BGR)
W, ROW_H, TOP_PAD, BAR_H = 980, 74, 14, 120
NAME_X, MINUS_X, TRACK_X0, TRACK_X1, PLUS_X, VAL_X = 18, 200, 262, 772, 796, 852
BTN = 38
BG, PANEL, TRACK, FILL, KNOB = (42, 36, 32), (58, 50, 44), (75, 65, 58), (255, 163, 74), (255, 255, 255)
FG, DIM, NOW, BAD, OKC, BTN_C = (232, 232, 232), (166, 160, 154), (32, 176, 255), (85, 85, 255), (120, 220, 120), (68, 58, 51)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _near_base(pts):
    """The wrist, the jaws or the gripper point inside the cylinder round the base (they'd hit the robot itself)."""
    return any(np.hypot(p[0], p[1]) < BASE_R and p[2] < BASE_Z for p in pts[-3:])


def path_check(kin, q_from, q_to, min_tcp_z, n=12):
    """Sim check of a straight joint move q_from -> q_to: (None or the reason it's refused, tcp at q_to).
    Refused: the gripper point below min_tcp_z, a link below the table, or the jaws at the robot's base (a move
    that starts there - e.g. the folded rest pose - may pass through it, but not end in it)."""
    start_near = _near_base(kin.set(q_from).points())
    for a in np.linspace(0.0, 1.0, n):
        kin.set(q_from + a * (q_to - q_from))
        tcp, pts = kin.tcp_pos(), kin.points()
        where = "there" if a == 1 else "on the way"
        if tcp[2] < min_tcp_z:
            return f"gripper point would go to {tcp[2] * 100:.1f} cm high (min {min_tcp_z * 100:.1f}) {where}", None
        low = min(p[2] for p in pts[2:])
        if low < LINK_MIN_Z:
            return f"a link would go below the table ({low * 100:.1f} cm) {where}", None
        if _near_base(pts) and (a == 1 or not start_near):
            return f"the jaws would hit the robot's base {where}", None
    return None, kin.set(q_to).tcp_pos()


def text(img, s, org, scale=0.55, color=FG, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def button(img, x, y, w, h, label, active=False, scale=0.6):
    cv2.rectangle(img, (x, y), (x + w, y + h), FILL if active else BTN_C, -1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, FONT, scale, 1)
    text(img, label, (x + (w - tw) // 2, y + (h + th) // 2), scale, (30, 30, 30) if active else FG,
         2 if scale >= 0.8 else 1)


class MotorPanel:
    def __init__(self, limits, now, move, check=None, speed=20.0, title=WIN, closable=True):
        """limits {motor: (lo, hi)} and now {motor: value} in real units (deg, gripper 0..100).
        move(target, speed) -> {motor: reached value} (blocking). check(target) -> (None or why refused, info text)."""
        self.motors = [m for m in MOTORS if m in limits and m in now]
        self.limits = {m: tuple(map(float, limits[m])) for m in self.motors}
        self.move_fn, self.check_fn = move, check
        self.cmd = {m: float(now[m]) for m in self.motors}          # last commanded
        self.val = dict(self.cmd)                                    # what the sliders show
        self.now = {m: float(now[m]) for m in self.motors}
        self.speed = min(SPEEDS, key=lambda s: abs(s - speed))
        self.step = 1.0
        self.pending, self.pending_t, self.drag, self.closed = False, 0.0, None, False
        self.status, self.status_c = "", FG
        self.win, self.closable = title, closable
        self.h = TOP_PAD + ROW_H * len(self.motors) + BAR_H
        self.buttons = []                                            # [(x0, y0, x1, y1, action)] for clicks
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.win, self._mouse)
        self._preview()
        self.draw()

    @property
    def dragging(self):
        return self.drag is not None

    # ---- geometry --------------------------------------------------------------------------------------------
    def _row_y(self, i):
        return TOP_PAD + i * ROW_H

    def _x_of(self, m, v):
        lo, hi = self.limits[m]
        return int(round(TRACK_X0 + (v - lo) / (hi - lo) * (TRACK_X1 - TRACK_X0)))

    def _v_of(self, m, x):
        lo, hi = self.limits[m]
        v = lo + (x - TRACK_X0) / (TRACK_X1 - TRACK_X0) * (hi - lo)
        return float(np.clip(round(v * 2) / 2, lo, hi))             # 0.5 steps while dragging

    def _row_at(self, y):
        i = (y - TOP_PAD) // ROW_H
        return self.motors[i] if 0 <= i < len(self.motors) and y >= TOP_PAD else None

    # ---- drawing ---------------------------------------------------------------------------------------------
    def draw(self):
        if self.closed:
            return
        img = np.full((self.h, W, 3), BG, np.uint8)
        self.buttons = []
        for i, m in enumerate(self.motors):
            y = self._row_y(i)
            cy = y + ROW_H // 2
            lo, hi = self.limits[m]
            if i % 2 == 0:
                cv2.rectangle(img, (0, y), (W, y + ROW_H), PANEL, -1)
            text(img, m, (NAME_X, cy - 4), 0.68, FG, 2)
            hint = "0 closed .. 100 open" if m == "gripper" else f"{lo:.0f} .. {hi:.0f} deg"
            text(img, hint, (NAME_X, cy + 20), 0.45, DIM)
            self._button(img, MINUS_X, cy - BTN // 2, BTN, BTN, "-", ("step", m, -1), scale=0.9)
            self._button(img, PLUS_X, cy - BTN // 2, BTN, BTN, "+", ("step", m, +1), scale=0.9)
            # track, filled part from 0 (or from lo if 0 is outside), knob, real position
            cv2.line(img, (TRACK_X0, cy), (TRACK_X1, cy), TRACK, 10, cv2.LINE_AA)
            x0 = self._x_of(m, float(np.clip(0.0, lo, hi)))
            xv = self._x_of(m, self.val[m])
            cv2.line(img, (x0, cy), (xv, cy), FILL, 10, cv2.LINE_AA)
            for tv in (lo, 0.0, hi):
                if lo <= tv <= hi:
                    cv2.line(img, (self._x_of(m, tv), cy + 12), (self._x_of(m, tv), cy + 18), DIM, 1)
            xn = self._x_of(m, float(np.clip(self.now[m], lo, hi)))
            cv2.fillPoly(img, [np.int32([(xn, cy - 11), (xn - 7, cy - 22), (xn + 7, cy - 22)])], NOW, cv2.LINE_AA)
            moving = self.drag == m or abs(self.val[m] - self.cmd[m]) > 0.05
            cv2.circle(img, (xv, cy), 14, FILL if moving else KNOB, -1, cv2.LINE_AA)
            cv2.circle(img, (xv, cy), 14, (30, 30, 30), 1, cv2.LINE_AA)
            text(img, f"{self.val[m]:7.1f}", (VAL_X, cy - 2), 0.72, FILL if moving else FG, 2)
            off = abs(self.now[m] - self.cmd[m]) > (5.0 if m == "gripper" else 3.0)
            note = "  held?" if off and m == "gripper" else ""
            text(img, f"now {self.now[m]:6.1f}{note}", (VAL_X, cy + 22), 0.45, NOW if off else DIM)
        # bottom bar: speed, step, gripper buttons, status
        y = self._row_y(len(self.motors)) + 12
        text(img, "speed", (NAME_X, y + 24), 0.55, DIM)
        x = 80
        for s in SPEEDS:
            self._button(img, x, y + 4, 40, 30, f"{s}", ("speed", s), active=s == self.speed, scale=0.5)
            x += 44
        text(img, "deg/s", (x + 2, y + 24), 0.5, DIM)
        x += 70
        text(img, "step", (x, y + 24), 0.55, DIM)
        x += 48
        for s in STEPS:
            self._button(img, x, y + 4, 40, 30, f"{s:g}", ("stepsize", s), active=s == self.step, scale=0.5)
            x += 44
        if "gripper" in self.motors:
            self._button(img, W - 210, y + 4, 92, 30, "grip open", ("set", "gripper", GRIPPER_OPEN), scale=0.5)
            self._button(img, W - 110, y + 4, 92, 30, "grip close", ("set", "gripper", GRIPPER_CLOSED), scale=0.5)
        text(img, self.status, (NAME_X, y + 66), 0.55, self.status_c, 1)
        text(img, "drag + let go / wheel / - + : move one motor     orange mark = real position", (NAME_X, y + 94),
             0.45, DIM)
        try:
            if cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:   # closed with the window's X
                if self.closable:
                    self.closed = True
                    return
                cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)   # needed: open it again
                cv2.setMouseCallback(self.win, self._mouse)
            cv2.imshow(self.win, img)
        except cv2.error:
            self.closed = True

    def _button(self, img, x, y, w, h, label, action, active=False, scale=0.6):
        button(img, x, y, w, h, label, active, scale)
        self.buttons.append((x, y, x + w, y + h, action))

    # ---- input -----------------------------------------------------------------------------------------------
    def _mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for x0, y0, x1, y1, act in self.buttons:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self._action(act)
                    return
            m = self._row_at(y)
            if m is not None and TRACK_X0 - 16 <= x <= TRACK_X1 + 16:
                self.drag = m
                self.val[m] = self._v_of(m, x)
                self._preview()
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            self.val[self.drag] = self._v_of(self.drag, x)
            self._preview()
        elif event == cv2.EVENT_LBUTTONUP and self.drag is not None:
            self.val[self.drag] = self._v_of(self.drag, x)
            self.drag = None
            self._request(now=True)
        elif event == cv2.EVENT_MOUSEWHEEL:
            m = self._row_at(y)
            if m is not None:
                self._step(m, +1 if cv2.getMouseWheelDelta(flags) > 0 else -1)

    def _action(self, act):
        kind = act[0]
        if kind == "step":
            self._step(act[1], act[2])
        elif kind == "speed":
            self.speed = act[1]
        elif kind == "stepsize":
            self.step = act[1]
        elif kind == "set":
            self.set_value(act[1], act[2])

    def _step(self, m, sign):
        lo, hi = self.limits[m]
        self.val[m] = float(np.clip(self.val[m] + sign * self.step, lo, hi))
        self._preview()
        self._request()

    def _status_set(self, s, color=FG):
        self.status, self.status_c = s, color

    def _preview(self):
        if self.check_fn is None:
            return None
        err, info = self.check_fn(dict(self.val))
        self._status_set(f"REFUSED: {err}" if err else info, BAD if err else FG)
        return err

    def _request(self, now=False):
        self.pending, self.pending_t = True, (time.time() - SETTLE if now else time.time())

    def _send(self):
        self.pending = False
        target = dict(self.val)
        changed = [m for m in self.motors if abs(target[m] - self.cmd[m]) > 0.05]
        if not changed:
            return
        err = self._preview()
        if err:
            self.val = dict(self.cmd)        # back to what was last sent
            self._status_set(f"REFUSED: {err} - slider set back", BAD)
            return
        speed = GRIP_SPEED if changed == ["gripper"] else float(self.speed)
        self._status_set(f"moving {', '.join(changed)} ...", FILL)
        self.draw()
        cv2.pollKey()                        # show it before the (blocking) move
        try:
            reached = self.move_fn(target, speed)
        except Exception as ex:              # e.g. arm_server error: show it, keep the window alive
            self.val = dict(self.cmd)
            self._status_set(f"move failed: {ex}", BAD)
            return
        self.cmd.update(target)
        self.set_now(reached)
        self._preview()

    # ---- used by the owner -----------------------------------------------------------------------------------
    def poll(self):
        """Call after every cv2.waitKey: sends a settled slider and redraws."""
        if self.closed:
            return
        if self.pending and self.drag is None and time.time() - self.pending_t >= SETTLE:
            self._send()
        self.draw()

    def set_now(self, joints):
        """Where the motors really are (the orange marks)."""
        for m in self.motors:
            if m in joints:
                self.now[m] = float(joints[m])

    def sync(self, joints):
        """The arm was moved by someone else: sliders + command := joints."""
        self.pending, self.drag = False, None
        for m in self.motors:
            if m in joints:
                self.cmd[m] = self.val[m] = float(joints[m])
        self.set_now(joints)
        self._preview()

    def set_value(self, m, value):
        """Set one slider and send it now (e.g. gripper open / close)."""
        lo, hi = self.limits[m]
        self.pending, self.drag = False, None
        self.val = dict(self.cmd)            # only this motor changes
        self.val[m] = float(np.clip(value, lo, hi))
        self._send()

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                cv2.destroyWindow(self.win)
            except cv2.error:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-arm", action="store_true", help="try the window, nothing moves")
    ap.add_argument("--speed", type=float, default=20.0, help="joint speed, deg/s")
    args = ap.parse_args()

    from pick_task import PickEnv            # the sim arm for the path check (heavy imports: only here)
    from click_goal_real import Kin
    from real_robot_policy import MIN_TCP_Z
    env = PickEnv(randomize=False)
    kin, jm = Kin(env), load_joint_map()
    arm = None if args.no_arm else ArmClient()
    if arm is None:
        joints, limits = {**sim_to_real(env.q_home, jm), "gripper": GRIPPER_OPEN}, DRY_LIMITS
    else:
        r = arm.read()
        joints, limits = r["joints"], r["limits"]
    live = {"joints": dict(joints)}

    def check(target):
        q_from = real_to_sim(live["joints"], jm, env.q_home)        # wrist_roll (taped) from q_home
        err, tcp = path_check(kin, q_from, real_to_sim(target, jm, q_from), MIN_TCP_Z)
        return err, ("" if tcp is None else
                     f"gripper point there: x {tcp[0] * 100:.1f}  y {tcp[1] * 100:.1f}  z {tcp[2] * 100:.1f} cm")

    def move(target, speed):
        if arm is None:
            live["joints"].update(target)
            print("move (dry run):", {m: round(v, 1) for m, v in target.items()})
            return dict(live["joints"])
        r = arm.move(target, speed)
        live["joints"] = r["joints"]
        if r.get("clamped"):
            print("clamped to the limits:", r["clamped"])
        return r["joints"]

    panel = MotorPanel(limits, joints, move, check, speed=args.speed)
    print("motors window open - Q / Esc (in the window) = quit")
    t_read = 0.0
    try:
        while not panel.closed:
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if arm is not None and time.time() - t_read > 0.3 and not panel.dragging:
                live["joints"], t_read = arm.read()["joints"], time.time()
                panel.set_now(live["joints"])
            panel.poll()
    finally:
        panel.close()
        if arm is not None:
            arm.close()


if __name__ == "__main__":
    main()
