"""Serves the real SO-101 to the VLA scripts, which run in the other venv (.venv: unsloth + mujoco).

Run in .venv-lerobot, from the project root, with the arm resting:

    python so101_real/arm_server.py

Only the ATTACHED motors of so101_arm.py are driven (wrist_roll is taped, not driven; commands for it are
ignored). The gripper is in lerobot's 0..100 units (0 / 100 = the ends of its calibrated range). Safety:
  * on start every servo's goal is set to where it already is before torque comes on (no jump)
  * targets are clamped to the calibrated range minus a margin; speed is capped
  * Ctrl+C during a move stops it on the spot; Ctrl+C again (or a client "shutdown") returns slowly to
    the start pose and only then switches torque off, so the arm doesn't drop
"""
import time
import argparse
from multiprocessing.connection import Listener

import numpy as np

from so101_arm import make_arm, ATTACHED, PORT, ROBOT_ID
from arm_protocol import ADDRESS, AUTHKEY

CONTROL_HZ = 50
TRACE_EVERY = 5             # move(trace=True): joints read every 5th control step (10 Hz) for recordings
MAX_SPEED_DEG_S = 60.0      # hard cap, whatever the client asks for
LIMIT_MARGIN_DEG = 3.0
TICKS_PER_REV = 4096
# Hand-set limits (deg) for joints whose calibration recorded a full turn (range 0..4095: the encoder wrapped
# while sweeping), so the calibrated range says nothing. = sim joint range mapped with real_joint_map.json
# (lift: sim -100..100 + 99.6; elbow: sim -96.8..96.8 - 91.9), trimmed; the folded rest pose reads 0 / 0.
MANUAL_LIMITS_DEG = {"shoulder_lift": (-2.0, 195.0), "elbow_flex": (-185.0, 2.0)}
FULL_TURN_TICKS = 4000
from cam_config import GRIPPER_MIN, GRIPPER_MAX   # noqa: E402  (.env)
GRIPPER_LIMITS = (GRIPPER_MIN, GRIPPER_MAX)   # 0..100 units (GRIPPER_MIN / GRIPPER_MAX in .env); every command clamped


class ArmServer:
    def __init__(self, port, robot_id):
        self.arm = make_arm(port, robot_id)
        bus = self.arm.bus
        bus.connect()
        # Goal := present position while torque is still off, so enabling torque can't make the arm jump
        bus.disable_torque()
        bus.sync_write("Goal_Position", bus.sync_read("Present_Position"))
        self.arm.configure()                     # sets position mode + PID, then enables torque
        self.motors = list(bus.motors)
        # In degrees mode 0 = middle of the calibrated range: limits are +-half the range
        self.limits, self.locked = {}, set()
        for m, c in self.arm.calibration.items():
            if m == "gripper":
                self.limits[m] = GRIPPER_LIMITS
                continue
            if c.range_max - c.range_min >= FULL_TURN_TICKS and m in MANUAL_LIMITS_DEG:
                self.limits[m] = MANUAL_LIMITS_DEG[m]
                print(f"{m}: calibrated range is a full turn - using the hand-set limits {MANUAL_LIMITS_DEG[m]}")
                continue
            half = (c.range_max - c.range_min) / 2 * 360 / (TICKS_PER_REV - 1) - LIMIT_MARGIN_DEG
            self.limits[m] = (-max(half, 0.0), max(half, 0.0))
            if half < 10:
                # e.g. a joint that wasn't swept during calibration: its servo limits block it anyway
                self.locked.add(m)
                print(f"WARNING: {m} calibrated range is only {2 * (half + LIMIT_MARGIN_DEG):.1f} deg - it stays "
                      f"LOCKED where it is. Recalibrate it (sweep it through its whole range) to use it.")
        self.free = False
        self.held = {}                           # {motor: value} HELD: no client can change it until released
        self.start_pose = self.read()
        print("connected, holding the start pose:", self.fmt(self.start_pose))
        print("limits (deg):", {m: tuple(round(v, 1) for v in lim) for m, lim in self.limits.items()})

    @staticmethod
    def fmt(joints):
        return "  ".join(f"{m} {v:6.1f}" for m, v in joints.items())

    def read(self):
        return {m: float(v) for m, v in self.arm.bus.sync_read("Present_Position", num_retry=2).items()}

    def read_raw(self):
        """Raw servo positions (encoder steps, 0..4095 per turn), before calibration."""
        return {m: int(v) for m, v in
                self.arm.bus.sync_read("Present_Position", normalize=False, num_retry=2).items()}

    def read_goal(self):
        """What each motor was last told (Goal_Position). A loaded joint sits a little off it (sag), so moves start
        from here, not from Present_Position: then a motor that isn't moved gets exactly the same command again."""
        if self.free:
            return self.read()
        try:
            return {m: float(v) for m, v in self.arm.bus.sync_read("Goal_Position", num_retry=2).items()}
        except Exception as ex:                  # fall back to the old behaviour
            print("WARNING: can't read Goal_Position, starting from Present_Position:", ex)
            return self.read()

    def move(self, target, speed, trace=False):
        """Smooth (smoothstep) joint interpolation to target {motor: deg}. Unknown / unmounted motors are ignored.
        Motors not in target keep their current goal (they get no new command). HELD motors keep their held value
        whatever the target says (reported in "held_refused"). trace: also return [{t, cmd, joints}] at 10 Hz
        (what each motor was told and where it was; t = time.time()) - for recordings."""
        ignored = sorted(set(target) - set(self.motors))
        held_refused = sorted(m for m in target if m in self.held and abs(float(target[m]) - self.held[m]) > 0.05)
        if held_refused:
            print(f"HELD, not changed: {', '.join(held_refused)} (release it with the HOLD button / hold off)")
        start = self.read_goal()
        start.update(self.held)
        goal, clamped = {}, []
        for m in self.motors:
            if m in self.locked or m in self.held:
                goal[m] = start[m]
                continue
            v = float(target.get(m, start[m]))
            lo, hi = self.limits[m]
            if not lo <= v <= hi:
                clamped.append(m)
            goal[m] = float(np.clip(v, lo, hi))
        speed = min(max(float(speed), 1.0), MAX_SPEED_DEG_S)
        duration = max(max(abs(goal[m] - start[m]) for m in self.motors) / speed, 0.3)
        n = max(int(duration * CONTROL_HZ), 1)
        stopped, rows = False, []

        def sample(cmd):
            try:
                rows.append({"t": time.time(), "cmd": dict(cmd), "joints": self.read()})
            except Exception:            # a missed read only leaves a gap in the trace
                pass

        try:
            if trace:
                sample(start)
            for i in range(1, n + 1):
                a = i / n
                a = a * a * (3 - 2 * a)
                cmd = {m: start[m] + a * (goal[m] - start[m]) for m in self.motors}
                self.arm.send_action({f"{m}.pos": v for m, v in cmd.items()})
                if trace and i % TRACE_EVERY == 0:
                    sample(cmd)
                time.sleep(1 / CONTROL_HZ)
            time.sleep(0.3)  # settle
            if trace:
                sample(goal)
        except KeyboardInterrupt:
            here = {**self.read(), **self.held}   # a held motor keeps its value (e.g. still squeezing a block)
            self.arm.send_action({f"{m}.pos": v for m, v in here.items()})
            stopped = True
            print("\nSTOPPED by Ctrl+C, holding here:", self.fmt(here))
        final = self.read()
        out = {"ok": not stopped, "stopped": stopped, "joints": final, "goal": goal, "clamped": clamped,
               "ignored": ignored, "locked": sorted(self.locked), "held": dict(self.held), "held_refused": held_refused,
               "max_err_deg": max(abs(final[m] - goal[m]) for m in self.motors)}
        if trace:
            out["trace"] = rows
        return out

    def hold(self, motor, on, value=None):
        """on: keep motor at value (default: what it is told now) - every move (any client) leaves it there."""
        if motor not in self.motors:
            return {"ok": False, "error": f"unknown motor {motor!r}"}
        if on:
            if value is not None:
                v = float(value)
            else:
                v = self.held[motor] if motor in self.held else self.read_goal()[motor]
            lo, hi = self.limits[motor]
            self.held[motor] = float(np.clip(v, lo, hi))
            print(f"HOLD {motor} at {self.held[motor]:.1f} - no move can change it until it is released")
        elif self.held.pop(motor, None) is not None:
            print(f"released {motor}")
        return {"ok": True, "held": dict(self.held), "joints": self.read()}

    def set_free(self, free):
        """free=True: torque off so the arm can be moved by hand (hold it - it will sag!)."""
        loose = [m for m in self.motors if m not in self.held]    # a HELD motor keeps its torque and goal
        if free:
            self.arm.bus.disable_torque(loose)
        else:
            bus = self.arm.bus
            present = bus.sync_read("Present_Position")
            bus.sync_write("Goal_Position", {m: present[m] for m in loose})
            bus.enable_torque(loose)
        self.free = free
        return {"ok": True, "free": free, "joints": self.read()}

    def shutdown(self):
        print("returning to the start pose ...")
        try:
            self.set_free(False)
            self.move(self.start_pose, speed=15)
        finally:
            self.arm.disconnect()  # torque off: the arm is resting again
        print("torque off, disconnected")


def handle(server, msg):
    cmd = msg.get("cmd")
    if cmd == "read":
        return {"ok": True, "joints": server.read(), "raw": server.read_raw(), "limits": server.limits,
                "held": dict(server.held)}
    if cmd == "move":
        return server.move(msg["target"], msg.get("speed", 20.0), bool(msg.get("trace", False)))
    if cmd == "free":
        return server.set_free(bool(msg.get("free", True)))
    if cmd == "hold":
        return server.hold(msg["motor"], bool(msg.get("on", True)), msg.get("value"))
    if cmd == "home":
        return server.move(server.start_pose, msg.get("speed", 15.0))
    return {"ok": False, "error": f"unknown command {cmd!r}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--id", default=ROBOT_ID)
    args = ap.parse_args()

    print(f"driving {ATTACHED} (wrist_roll is ignored)")
    server = ArmServer(args.port, args.id)
    listener = Listener(ADDRESS, authkey=AUTHKEY)
    print(f"listening on {ADDRESS[0]}:{ADDRESS[1]}  (Ctrl+C: stop; Ctrl+C when idle: return to start and quit)")
    try:
        while True:
            with listener.accept() as conn:
                print("client connected")
                while True:
                    try:
                        msg = conn.recv()
                    except EOFError:
                        print("client disconnected (still holding the pose)")
                        break
                    if msg.get("cmd") == "shutdown":
                        conn.send({"ok": True})
                        raise KeyboardInterrupt
                    try:
                        conn.send(handle(server, msg))
                    except Exception as e:  # report, keep serving
                        conn.send({"ok": False, "error": repr(e)})
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()
        server.shutdown()


if __name__ == "__main__":
    main()
